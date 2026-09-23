# 本机实跑记录（2026-09-23）

目标：下载权重、修 GPU、用本地 ScanNet 训练。本文记录**实测事实**与仍缺的东西。

## 1. GPU 诊断与修复

| 事实 | 值 |
|---|---|
| GPU / 驱动 | RTX 3090 24 GB，NVIDIA 535.309.01（对外 CUDA 12.2） |
| 系统 python、conda `d4rt`/`stablemvs`/`trellis2` | torch **2.12.1+cu130** → `cuda_avail False`（cu130 需要 ≥580 驱动）|
| `~/Desktop/dl_env` | torch **2.6.0+cu124** → **GPU 可用**（4k 矩阵乘实测通过） |
| 结论 | 不是 CUDA/驱动坏了，是 **wheel 与驱动代次不匹配**；两条修法：装 cu12x wheel（无需 root）或升驱动到 580（需 root，会影响其它环境） |

`scripts/fix_gpu.sh` 做第一条：用阿里云镜像（pypi 官方源实测 5 KB/s，阿里云 3.7 MB/s）建 `~/Desktop/pixels_env` 并自检。
本项目实跑一律用 `~/Desktop/dl_env/bin/python`。

**已验证的 GPU 计算链路**：真实冻结 Wan VAE 编码 `21×192×256 → z_obs (1,16,6,24,32)` → 归一化 → L4AR 前向 → `points (1,21,192,256,3)`；paper 规模（`l4ar_paper.yaml`）**404.4 M 参数**能装进 24 GB 单卡。

## 2. 权重下载

| 权重 | 状态 | 来源/位置 |
|---|---|---|
| Wan2.1 视频 VAE（共享 latent 接口，126.9 M） | ✅ 已下载并在 GPU 跑通 | ModelScope `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` → `/mnt/data/pixels-weights/wan-vae/vae/`（507,591,892 B，194 tensors 校验通过） |
| DINOv2-base（346,3 MB）+ CLIP-ViT-L/14（1,710.5 MB） | ✅ 已下载，本机加载成功 | ModelScope `facebook/dinov2-base`、`AI-ModelScope/clip-vit-large-patch14`（注意：`AI-ModelScope/dinov2-base`、`openai/...` 在 ModelScope 上不存在） |
| **4RC（31 层精化 + 两头的预训练初始化）** | ❌ **拿不到**：ModelScope 无镜像（404），huggingface.co 本机不可达（0 B/s），hf-mirror 亦不通 | 需能访问 HF 的机器拉 `Luo-Yihang/4RC`，或作者另发镜像 |
| Wan2.1/2.2 DiT（采 z^gen，~17 GB/个） | ⏳ 可选，只有做 Table 1 生成评测才需要（ModelScope 有同名仓库） | 未拉，避免占用 17 GB×2 与带宽 |

`scripts/fetch_weights.sh` 走 HF/hf-mirror（本机不通）；本机可用通道另写为 `scripts/fetch_weights_modelscope.sh`，带断点续传与内容长度校验。

缺 4RC 的后果已如实处理：`tools/train.py` 现在**警告后继续随机初始化**（结构与训练动力学有效，绝对数值不可与论文对比）。

**顺带修正的论文配置**：VAE 的归一化常数原先标 `verify`，现已从 checkpoint 的 `vae/config.json` 取真值写入
`configs/model/l4ar_paper.yaml`（16 维 `latents_mean` / `latents_std`），`SharedLatentInterface.normalize()` 改为逐通道
`(z-mean)/std`。

**真实指标通路已在 GPU 上跑通**（`tools/eval_generation.py --extractor dinov2`，权重全部本地）：
`Text CLIP 10.68 / CLIP-I 66.18 / DINO global 27.31 / DINO match 27.30 / DINO F1 3.87`。
数值低是应当的（几何头随机初始化 + 随机合成 latent 用例），验证的是**五列指标的 computation 与本地权重加载**。
顺手修了 `ClipScorers`：transformers 5.0 的 `get_text_features()/get_image_features()` 返回
`BaseModelOutputWithPooling` 而非张量，现由 `_features()` 统一取 `pooler_output`。

## 3. ScanNet 数据：**本机没有**

全盘核查（`find` 到 `scene0*`、`*.sens`、`*_00.aggregation.json`、`scannetv2` 等签名）结果：

- 只有上游仓库的 **split 列表**：`Desktop/MVS/stableMVS_mnh_zl/mvsanywhere/data_splits/ScanNetv2/{standard_split,dvmvs_split}`
- 那些仓库的 config 指向 `/mnt/nas3/shared/datasets/scannet`、`/mnt/scannet-data-png2/` —— **是别的机器的路径，本机不存在**
- `~/16t/e`、`~/16t/f`（NAS CIFS 挂载点）**当前未挂载**，挂载需 sudo 密码（我不能执行）；`/mnt/data` 除权重目录外是空的

所以第 3 项目标卡在数据位置上。已经把所有**不依赖数据到位**的部分做完并测试通过：

- `l4d/data/scannet.py`：自动识别两种常见布局（官方 `frame-%06d.jpg/-depth.png/-pose.txt + intrinsic/`；png2 风格 `images|color / depth / pose + intrinsics`），uint16 毫米深度→米，4×4 c2w 位姿，**内参随分辨率缩放**，深度用 NEAREST 重采样，输出共享世界系下的 `points/ray_dirs/depth/mask/camera_rotation/camera_centers/fov`
- `tools/prepare_scannet.py`：`--report-only` 先盘点场景可读性，再导出 `manifest.jsonl + clips/ + gt/*.npz`
- `configs/data/scannet.yaml`（192×256、21 帧、1143 clips 目标）与 `configs/data/scannet_probe.yaml`
- `tests/test_scannet.py`：3 项检查（布局识别、**世界点↔相机深度一致性**、导出结果可被 `ReconstructionClips` 加载且视频落在 VAE 的 [-1,1]），用合成 ScanNet 夹具跑通
- 修正 `ReconstructionClips._load_video` 输出到 `[-1,1]`（真实视频 VAE 的取值域）

**用 ScanNet 格式的 6 个 clip 已在 GPU 上完成三阶段真训练**（`prepare_scannet.py` 导出 → 真实冻结 Wan VAE 编码 →
`l4ar_probe.yaml`（token_dim 512 / depth 12）→ 每阶段 60 step，`runs/gpu_scannet_full/`）：

| 阶段 | 激活 | loss 起→止 | 备注 |
|---|---|---|---|
| stage1_adapter | alignment | 7.52 → **5.45** | 仅 A_phi 可训练 |
| stage2_heads | +heads | 6.75 → **5.22** | depth 0.81→0.75、ray 1.42→0.92 |
| stage3_lora | +rank-16 LoRA | 5.62 → **4.66** | `point_mean_err` 0.83→0.80 m、`point_tail_err` 1.60→1.57 m |

单阶段 60 step 用时 13–15 s（24 GB 单卡）。这些是**合成 ScanNet 夹具**上的数（结构与接口验证），
不是论文数值——数据换成真实 ScanNet、初始化换成 4RC 后才可比。

## 4. 待你决定

1. **ScanNet 到底在哪**：要我在挂载点上跑，请执行
   `sudo mount -t cifs -o username=16t,uid=zbf //<ip>/<share> /home/zbf/16t/e`（凭据在你 `Desktop/mount.txt`），
   然后告诉我目录；或直接给路径。之后一条命令即可盘点：
   `~/Desktop/dl_env/bin/python tools/prepare_scannet.py --root <scannet> --report-only`
2. **4RC 权重**：能否从有 HF 出口的机器 scp `/mnt/data/pixels-weights/4rc/model.safetensors` 过来？
   到位后跑 `tools/inspect_checkpoint.py` 反推真实 `token_dim/depth/tap`，替换配置里的 `verify` 值。
3. 是否要我把 `d4rt` 等 conda 环境的 torch 也降到 cu124（会改动它们的依赖，需你同意）。
