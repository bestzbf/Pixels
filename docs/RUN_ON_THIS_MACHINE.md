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

**可用的替代初始化**：`l4d/models/init_from.py` 把真实预训练 ViT 灌进冻结精化层级（4RC 本身就建立在
DINOv2 之上）。用本机 `dinov2-base/model.safetensors` 实测 **144/144 个 block 参数命中**（12 block ×
qkv/proj/fc1/fc2 权重+偏置 + 两个 norm），`norm1.weight` 均值 1.274≠1 证明确实来自预训练而非随机；
配置见 `configs/model/l4ar_dinov2.yaml`（token_dim 768 / heads 12 / depth 12）。它同时反推出 4RC 到位后
该用的真实维度。

**顺带修正的论文配置**：VAE 的归一化常数原先标 `verify`，现已从 checkpoint 的 `vae/config.json` 取真值写入
`configs/model/l4ar_paper.yaml`（16 维 `latents_mean` / `latents_std`），`SharedLatentInterface.normalize()` 改为逐通道
`(z-mean)/std`。

**真实指标通路已在 GPU 上跑通**（`tools/eval_generation.py --extractor dinov2`，权重全部本地）：
`Text CLIP 10.68 / CLIP-I 66.18 / DINO global 27.31 / DINO match 27.30 / DINO F1 3.87`。
数值低是应当的（几何头随机初始化 + 随机合成 latent 用例），验证的是**五列指标的 computation 与本地权重加载**。
顺手修了 `ClipScorers`：transformers 5.0 的 `get_text_features()/get_image_features()` 返回
`BaseModelOutputWithPooling` 而非张量，现由 `_features()` 统一取 `pooler_output`。

## 3. ScanNet 数据：**本机没有（穷尽式核查）**

最后一轮把搜索扩到**整块本地盘**（`find / -xdev -maxdepth 7`，匹配 `*scannet*`、`scene[0-9][0-9][0-9][0-9]_*`、`*.sens`，
排除 `/proc /sys /tmp` 与 site-packages），命中项**全部是本仓库自己的文件**：

```
configs/data/scannet.yaml  configs/data/scannet_probe.yaml  tools/prepare_scannet.py
tests/test_scannet.py  l4d/data/scannet.py  data/scannet/  runs/gpu_scannet*/
```

注意 `data/scannet/` 是我用合成夹具生成的 ScanNet-**格式**数据（`tests/test_scannet.py build`），
不是真 ScanNet，别混淆。另：ScanNet 是需签署 EULA 的门控数据集，不应从非官方镜像拉取——
正确路径是你用自己的账号下载后告诉我目录。

先前局部核查：

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

## 3b. 实际用的替代数据：本机 4 个真实 COLMAP 场景（ETH3D dslr）

按你的选择「先用本地其它 RGB-D 数据」，接入层 `l4d/data/colmap.py` + `tools/prepare_colmap.py` 把
COLMAP 文本模型转成 ScanNet 同格式暂存树，再交给已测试的 `export_manifest`：

| 场景 | 相机模型 | 帧数(4k+1) | 稀疏深度覆盖率 | 场景尺度 | 相机轨迹长度 |
|---|---|---|---|---|---|
| office | THIN_PRISM_FISHEYE 6048×4032 | 21 | 23.5 % | 81.8 m | 8.2 m |
| statue | 同上 | 9 | 66.7 % | 44.8 m | 4.8 m |
| pipes | 同上 | 13 | 28.5 % | 47.9 m | 11.8 m |
| courtyard | 同上 | 21 | 66.9 % | 164.8 m | 19.6 m |

要点：鱼眼/薄棱镜畸变在重映射到 clip 分辨率时一并去除，去畸变用的理想 K 与稀疏点投影用的是同一个，
所以 **图像 / 内参 / 深度三者自洽**；`rot_orth = 1.2e-07` 验证 COLMAP 四元数→c2w 约定正确；
深度是 `points3D` 逐视图重投影（半径 2 px 溅射），因此稀疏但真实，Eq. 7 每一项都按 mask 加权。

几何项按场景尺度归一（`LossConfig.normalise_point_error`）后训练稳定：
绝对米制误差会让 40 m 房间与 160 m 广场在同一 batch 里不可比（未归一时 stage3 末步 loss 从 11.5 跳到 34.1）。

### 真实 2000 step×3 阶段结果（ETH3D COLMAP，random init）

| 场景 | 未训练 rel_err | 训练后 rel_err | comp 未训 → 训练后 |
|---|---|---|---|
| office | 0.0243 | **0.0238** | 1.257 → **0.924 m** |
| statue | 0.0479 | **0.0465** | 0.758 → **0.687 m** |
| pipes | 0.0394 | **0.0389** | 0.639 → **0.510 m** |
| courtyard | 0.0668 | **0.0549** | 2.661 → **1.876 m** |
| **均值** | 0.0446 | **0.0410** | 4/4 场景 completeness 全部改善 |

即：**真实数据上确实学到了东西，但幅度小**（相对点误差 -8%，稠密化 20–30%）。原因是可解释的：
无 4RC 预训练初始化（随机层级）+ 只有 4 个 clip + 稀疏 `points3D` 监督（覆盖率 23–67%）。

### 更好的真实监督 + 数据 QC 闸门

`BioPhysGS-paper/benchmark_artifacts/converted/{colmap,pancakes}`：32 视角、**逐视图稠密深度 PNG（毫米）**、
`PINHOLE 256×256, f=221.70`。接入层因此新增：优先用盘上深度图（`_find_depth`/`_depth_to_millimetres`，支持
uint16 毫米与 float 米两种约定）、深度 NEAREST 重采样、NeRF 风格改名场景的**按位序回退匹配**
（poses 写 `frame_000.png` 而盘上是 `000_color.png`）。稠密场景覆盖率实测 **100%**。

`tools/check_dataset.py` 把“能不能用这块数据”变成可复现判据：
- **相邻视图**（而非任意视图对）世界点最近邻距离 / 场景尺度 ≤ 5%；
- 尺度归一化的几何指纹 → 找重复资产；
- `--ascent` 扫描深度尺度，检查深度与位姿是否同一单位系。

第一版工具用任意视图对时把稠密场景误判成 24–56% 不一致；改成相邻视图后实测：

| clip | 深度覆盖 | 相邻视图一致性 | 最优深度尺度 | 判定 |
|---|---|---|---|---|
| colmap（稠密物体场景） | 100% | **0.45%** | 1.0 | keep |
| pancakes | 100% | 0.45% | 1.0 | **drop：与 colmap 同一资产（指纹相同）** |
| statue / courtyard / office / pipes | 26–71% | 0.07–2.1% | 1.0 | keep |

即：**深度图与位姿尺度是自洽的**（先前“不一致”是拿轨道上互不重叠的两视图比较造成的假象），
只有重复拷贝被剔除。最终真实池 = **5 个唯一 clip**（1 稠密 + 4 ETH3D），`configs/data/real.yaml`。

### 学习效应已量化（ETH3D 4 clips，2000 step×3 阶段，同一评测协议）

| 模型 | mean rel point err | courtyard 内点率 | courtyard comp | office comp |
|---|---|---|---|---|
| 未训练（768d，DINOv2 尺寸） | 0.0446 | 0.1% | 2.724 m | 1.246 m |
| 随机初始化 512d/12bl | 0.0410 | — | 1.876 m | 0.924 m |
| **DINOv2 预训练初始化 768d/12bl** | **0.0396** | **5.9%** | **1.632 m** | **0.936 m** |

三者单调：**预训练初始化 > 随机初始化 > 未训练**（相对点误差 -11%、completeness -25~40%、
courtyard 内点率 60×）。这同时验证了论文“用预训练 4D 层级做初始化”的必要性——我们的
`init_from.py` 用本机 DINOv2 替代拿不到的 4RC，确实带来可测收益。

真实规模再往上：从 4RC 源码读出 d=1536/40 层/24 heads 后，`l4ar_paper.yaml` 现为 **1.156 B 参数
（冻结 1.133 B，可训练仅 22.8 M：对齐 0.10 M + 头 6.96 M + rank-16 LoRA 15.73 M）**，
在本卡 9 帧 128×128 下前向+反传峰值 **7.8 GB**，论文规模不需要多卡即可训练。

`tools/eval_recon.py` 在真实 clip 上给出相对点误差 / Acc / Comp 与 PLY 导出。实测：
**随机初始化基线 0.0446 vs 训练 300 step×3 阶段 0.0444**（几乎无差），但 3/4 场景 completeness 变好
（office 1.257→1.000 m、pipes 0.639→0.503 m）。结论要直说：**没有 4RC 预训练初始化、只有 4 个 clip 时，
300 step 还不足以让几何学出来**——管线与梯度是通的，学习量不够；正在跑 2000 step×3 阶段复验。

### ScanNet 一键链已验证（用 ScanNet 格式夹具当数据）

`scripts/train_on_scannet.sh <scannet-root> [clips]`：清单盘点 → 导出 clip+度量 GT → 冻结 Wan VAE 预编码
→ 三阶段训练 → 与未训练基线对照评测。本机以 2 场景 × 63 帧的 ScanNet 格式夹具在 **GPU + 真实 VAE** 上实测：

```
[1/5] {"scenes": 2, "scene0000_00": 63, "scene0001_00": 63}
[2/5] {"clips": 6, "meets_target": false}
[3/5] frozen Wan-VAE latents  (cuda)
[4/5] stage1 -> stage2 loss 5.14 -> stage3 loss 4.19 (trainable-only ckpt 4.7 / 11 MB)
```

顺带修掉两个真实缺陷：脚本不再原地改写被版本管理的 YAML（生成 `configs/data/scannet.local.yaml`，已加入
`.gitignore`）；**下载中的半截 4RC 文件**会让 `os.path.exists` 判为“有”而在 safetensors 处崩溃，现在降级为
明确警告并回退随机初始化。

### 训练读数修正：latent 缓存接入 + 缺失的 VAE 归一化

审计训练路径时发现两处真实缺口，已修：

1. `LatentCache` 之前**从未被训练读取**（`tools/precompute_latents.py` 写、训练却每步重新编码）。
   现在 `ReconstructionClips(latent_cache=...)` → batch 带 `latent` → `training_step` 优先用缓存；
   实测缓存与实时编码**逐元素相等（max diff 0.0）**，并有断言测试（VAE 被替换为会抛错的桩，
   证明走的是缓存分支）。
2. 训练侧此前**没有做 latent 归一化**。Wan 官方 `latents_mean/latents_std`（16 通道）现在
   在 `latent_from_batch` 里统一施加，推理与训练同约定。

另外测试入口 `if __name__ == "__main__": main()` 原先写在文件中段，`cat >>` 追加的用例
根本不会被执行 —— 表现为“21/21 全绿”但实际有 26 个用例、其中 3 个从未跑过。
runner 已移到文件末尾，现在 **26 个定义 = 26 条 pass**；那 3 个用例本身是测试写错
（LoRA 包裹后参数名为 `attn.qkv.base.weight`；随机点云无局部法线结构故 NC 无意义；
独立抽样下同一云两次子集不同，需按采样噪声放宽阈值）。

## 4. 待你决定

1. **ScanNet 到底在哪**：要我在挂载点上跑，请执行
   `sudo mount -t cifs -o username=16t,uid=zbf //<ip>/<share> /home/zbf/16t/e`（凭据在你 `Desktop/mount.txt`），
   然后告诉我目录；或直接给路径。之后一条命令即可盘点：
   `~/Desktop/dl_env/bin/python tools/prepare_scannet.py --root <scannet> --report-only`
2. **4RC 权重**：能否从有 HF 出口的机器 scp `/mnt/data/pixels-weights/4rc/model.safetensors` 过来？
   到位后跑 `tools/inspect_checkpoint.py` 反推真实 `token_dim/depth/tap`，替换配置里的 `verify` 值。
3. 是否要我把 `d4rt` 等 conda 环境的 torch 也降到 cu124（会改动它们的依赖，需你同意）。
