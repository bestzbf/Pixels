# Pixels — Latent-to-4D 复现框架 (arXiv:2608.10744)

论文：**Beyond Pixels: From Video Priors to 4D Worlds** (Liu, Shen, Zhou, Quan, Yang — ReLER, CCAI, ZJU)
方法：**Latent-to-4D**（核心网络 **L4AR** = Latent-to-4D Alignment and Refinement）。
项目页：<https://hayd-zju.github.io/Beyond-Pixels/>

见 [`RESULTS.md`](RESULTS.md)：**本机实测数字台账**（权重校验、GPU 诊断、数据 QC、学习效应对照、Table 1/3 状态与 reproduce 命令）。

论文原文不随本仓库分发（版权），用 `bash scripts/fetch_paper.sh` 拉取 arXiv v1（9 页，**不含附录**）并生成本地文本抽取
`paper_main_text.txt` / `paper_html_text.txt`（均在 `.gitignore` 中）。分析结论已全部写入 `docs/PAPER_ANALYSIS.md`。
论文逐条分析见 [`docs/PAPER_ANALYSIS.md`](docs/PAPER_ANALYSIS.md)，复现计划见 [`docs/REPRODUCTION_PLAN.md`](docs/REPRODUCTION_PLAN.md)。

## 方法一句话

共享同一 VAE 的视频 DiT，其**最终去噪 latent** 可直接作为显式 4D 预测的**可复用接口**：
先用 Alignment Module（固定三线性重采样 $\mathcal{R}$ + 学习型 3D 卷积 $\mathcal{S}_\phi$）把视频 latent
对齐到预训练 4D 重建器（4RC）的 token 网格，再用 31 层 frame-wise / global 交替注意力（冻结权重 + rank-16
LoRA）精化，最后由 4D decoder 预测相机（9D pose–FOV）与动态世界坐标点图
$\widehat{\mathbf P}_t(u)=\widehat{\mathbf o}_t+\widehat d_t(u)\widehat{\mathbf r}_t(u)$，全程**不经过 RGB**。
训练只用 ~1K（最终阶段 1,143）个带 4D 标注的重建 clips；一个 checkpoint 不改权重即可跨
Wan2.1-T2V-14B / Wan2.1-T2V-1.3B / Wan2.2-I2V-A14B 三个 DiT 使用。

## 目录结构

```
Pixels/
├── paper_2608.10744.pdf            # 下载的论文原文（v1，无附录）
├── paper_main_text.txt             # 论文正文纯文本（pdftotext）
├── paper_html_text.txt             # arXiv HTML 版正文（表格数字来源）
├── docs/
│   ├── PAPER_ANALYSIS.md            # 论文分析：公式/模块/实验协议/全部表格数字/开放问题
│   └── REPRODUCTION_PLAN.md          # 分阶段复现计划、验收标准、算力预算、风险
├── configs/
│   ├── model/l4ar_paper.yaml         # 论文规模配置
│   ├── model/l4ar_tiny.yaml          # CPU 冒烟测试规模（结构同构）
│   ├── train/staged.yaml             # 三阶段渐进激活训练（1,143 clips 终态）
│   ├── data/clips_final.yaml         # 6 个重建数据集 / manifest / latent 缓存
│   ├── data/smoke.yaml               # 离线合成数据
│   └── eval/{text4d200,i4d200,gt_ablation}.yaml
├── l4d/
│   ├── models/    alignment.py refinement.py lora.py heads.py decoder.py l4ar.py video_interface.py
│   ├── losses/    objectives.py          # L = L_unc + L_cam + L_geom
│   ├── data/      dataset.py             # ClipRecord/manifest/ReconstructionClips/LatentCache
│   │              scannet.py             # ScanNet v2 两种布局 -> 世界系点图/射线/相机
│   │              colmap.py              # COLMAP(cameras/images/points3D) -> ScanNet 同格式暂存树
│   ├── eval/      projection_metrics.py gt_metrics.py residual_probe.py baselines.py protocol.py
│   └── utils/     geometry.py config.py
├── tools/         train.py infer.py eval_generation.py eval_gt.py eval_recon.py residual_sensitivity.py
│                  precompute_latents.py build_benchmark.py prepare_scannet.py prepare_colmap.py check_dataset.py
├── scripts/       install.sh smoke_test.sh train_all.sh eval_all.sh fetch_weights.sh
└── tests/         test_reproduction.py   # 21 项结构/数学/协议一致性检查
```

## 快速开始（离线，CPU 即可）

```bash
bash scripts/install.sh                      # 依赖（torch / numpy / pyyaml / pillow；diffusers 可选）
bash scripts/smoke_test.sh                   # 21 项测试 + 三阶段训练 + 推理 + 两套评测 + 残差诊断
```

手动运行：

```bash
python tests/test_reproduction.py                                       # 结构/数学检查
python tools/train.py --device cpu --steps 3 --output runs/smoke         # stage1→2→3 渐进训练
python tools/infer.py --model configs/model/l4ar_tiny.yaml --device cpu --out runs/infer
python tools/eval_gt.py --device cpu --clips 1                            # Table 3 五个变体
python tools/residual_sensitivity.py --device cpu --rho 0.2 0.6 1.0       # Fig. 6
python tools/eval_generation.py --synthetic 4 --device cpu                # Table 1 指标管线
```

真实复现（GPU + 权重 + 数据）：

```bash
bash scripts/fetch_weights.sh $HF_TOKEN          # Wan2.1 VAE/DiT、4RC、DINOv2、CLIP
python tools/precompute_latents.py --data configs/data/clips_final.yaml
python tools/build_benchmark.py --eval configs/eval/text4d200.yaml --conditions data/prompts/text4d200.jsonl
python tools/train.py --model configs/model/l4ar_paper.yaml --data configs/data/clips_final.yaml
python tools/eval_generation.py --eval configs/eval/text4d200.yaml --checkpoint runs/.../stage3_lora.pt
```

## 论文结论 → 代码位置

| 论文内容 | 位置 |
|---|---|
| Eq.(1)(5) 对齐 $\mathcal{A}_\phi=\mathcal{S}_\phi\!\circ\!\mathcal{R}$ | `l4d/models/alignment.py` |
| Eq.(4) $\mathcal{D}_\omega(\mathcal{H}_{\psi,\Delta\psi}(\mathcal{A}_\phi(\mathbf z);\mathbf S))$ | `l4d/models/l4ar.py` |
| 31-block frame-wise/global 交替 + 多深度 tap + 冻结 camera/time token | `l4d/models/refinement.py` |
| rank-16 LoRA（$\Delta\psi$） | `l4d/models/lora.py` |
| Eq.(6) 射线反投影 + 9D pose–FOV 相机 | `l4d/models/heads.py`, `decoder.py`, `l4d/utils/geometry.py` |
| Eq.(7) $\mathcal{L}_{unc}+\mathcal{L}_{cam}+\mathcal{L}_{geom}$ | `l4d/losses/objectives.py` |
| 共享 VAE 兼容条件（同 checkpoint/归一化/layout/压缩/形状） | `l4d/models/video_interface.py` |
| Table 1 数字与 matched same-latent 对比 | `l4d/eval/baselines.py` |
| Table 2 人评偏好 + bootstrap 区间 | `l4d/eval/protocol.py` |
| Table 3 消融变体（w/o Grid / 3D Conv / Frame / Global） | `configs`, `tools/eval_gt.py`, `l4d/eval/protocol.py` |
| Fig. 6 $\mathbf z_{45}-\mathbf z_{50}$ → Grid-Align null 投影敏感性 | `l4d/eval/residual_probe.py`, `tools/residual_sensitivity.py` |
| 训练期 $\mathbf z^{obs}$ / 推理期 $\mathbf z^{gen}$ 不对称 | `tools/precompute_latents.py`, `tools/build_benchmark.py`, `tools/infer.py` |

## 当前状态

* 已完成：论文下载与逐条分析；结构级复现框架（对齐→精化→4D 解码全链路）；三阶段渐进冻结训练；
  Eq.6/Eq.7/兼容性门/消融变体/Fig.6 诊断/投影指标管线；21 项测试全绿；CPU 端到端冒烟通过。
* 待补：4RC 权重与真实维度、6 个数据集清单与 1,143 clips 划分、Text4D-200/I4D-200 用例锁、
  各阶段超参 —— 均已在 `docs/PAPER_ANALYSIS.md` 第 8 节以「开放问题」列出并在 YAML 中以 `verify` 标注。
  v1 PDF 无附录，这些量需等作者补充材料或按 `REPRODUCTION_PLAN.md` 的替代方案确定。

## 本机实跑（GPU / 权重 / ScanNet）

见 [`docs/RUN_ON_THIS_MACHINE.md`](docs/RUN_ON_THIS_MACHINE.md)：驱动 535 与 cu130 wheel 不匹配的诊断与
`scripts/fix_gpu.sh`；ModelScope 通道拉到的真实冻结 Wan VAE（含官方 16 维 `latents_mean/std`，已写入配置）；
`l4d/data/scannet.py` + `tools/prepare_scannet.py` 的 ScanNet 接入层；以及三阶段 GPU 训练实测
（stage1 7.52→5.45、stage2 6.75→5.22、stage3 5.62→4.66）。

所有**实测数字与结论以 [`RESULTS.md`](RESULTS.md) 为准**（含 4RC 权重逐张量校验、训练/对照配对评测、
本地 Table 3 的负面结果）。ScanNet 尚未实跑的原因已定位：`/etc/fstab` 里 NAS 的
`//192.168.2.233/{e,f}` 挂在 `~/16t/{e,f}`，当前既无到 `192.168.2.0/24` 的路由、
`/etc/cifs-credentials` 也是 root 私有，需你重新接入该网络后 `sudo mount -a`；
届时 `bash scripts/train_on_scannet.sh <scannet_root> 1143` 一条命令即可，该链路五步已在真实数据的
ScanNet 同格式树（`data/real_staging`）上全程跑通，见 `RESULTS.md` §3.1。
