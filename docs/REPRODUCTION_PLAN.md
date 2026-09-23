# 复现计划 — Latent-to-4D (arXiv:2608.10744)

目标：**单个 checkpoint 跨多个共享 VAE 的视频 DiT，直接把最终去噪 latent 映射为相机 + 动态点图**，
并在 Text4D-200 / I4D-200 上复现相对 matched Wan+4RC 级联的 DINO-F1 增益（T2D +2.88~3.45，I2D +5.81）。

## 阶段与验收门

| 阶段 | 产出 | 验收标准（gate） | 状态 |
|---|---|---|---|
| P0 论文解析 | `paper_2608.10744.pdf`、`paper_main_text.txt`、`docs/PAPER_ANALYSIS.md` | 公式 (1)–(7)、模块表、Table 1/2/3 全部数字、Fig.6 数字落到文档 | ✅ 完成（v1 无附录，开放项已登记） |
| P1 结构复现 | `l4d/models/*` | latent→tokens→31 层交替注意力→depth/ray/camera→点图，形状与 Eq.(5)(6) 一致；冻结/可训练集合与论文一致 | ✅ 完成，21/21 测试通过 |
| P2 训练复现 | `tools/train.py`, `configs/train/staged.yaml` | 三阶段渐进激活可跑通、损失有限且下降；stage3 含 rank-16 LoRA；1,143 clips 的 manifest 校验 | ✅ 框架完成（CPU 合成数据 3 step 冒烟通过）；真实数据/GPU 训练 ⏳ |
| P3 生成评测 | `tools/eval_generation.py`, `l4d/eval/projection_metrics.py` | 双 off-axis 投影 → Text CLIP / CLIP-I / DINO global / DINO match / DINO F1(×100) 全管线；与 Table 1 参考行对比并检查增益声明 | ✅ 管线完成（surrogate encoder 冒烟通过）；真实 DINOv2+200 用例 ⏳ |
| P4 GT 诊断与消融 | `tools/eval_gt.py`, `tools/residual_sensitivity.py` | 7-Scenes/NRGBD 上 Acc/Comp(cm)/NC 五变体可出表；Fig.6 残差 null 投影诊断 ours < cascade | ✅ 工具完成（随机权重数值无意义）；权重就绪后重跑 ⏳ |
| P5 跨 DiT 复用验证 | `tools/build_benchmark.py` + 三个 generator | 同一 checkpoint 不改权重跑 Wan2.1-14B / 1.3B / Wan2.2-I2V，DINO-F1 差异 < 0.1（论文 57.01 vs 57.09） | ⏳ 需 DiT 权重与 GPU |

## 数据准备

1. `bash scripts/fetch_weights.sh $HF_TOKEN`：Wan2.1-T2V-1.3B/14B（VAE+DiT）、Wan2.2-I2V-A14B、
   4RC（refinement 层级 + heads 初始化）、DINOv2、CLIP。
2. 6 个重建数据集 → `data/manifest.jsonl`（`ClipRecord` 字段），要求
   `tools/precompute_latents.py` 打印 `manifest_report` 中 `meets_target: true`（≥1,143 clips）。
   **未确认**：v1 未列出 6 个数据集名字与划分；候选见 `configs/data/clips_final.yaml`。
3. 冻结 VAE 预编码 `z^obs = mu(E_v(V))` 到 `data/latents/`（训练期不运行 DiT，也不运行 decoder）。
4. Benchmark：`tools/build_benchmark.py` 用冻结 DiT 采样 terminal latent + 记录轨迹（含 `z_45,z_50`），
   写 `lock.json`（case_id + sha256），确保 ours 与 matched cascade **共享同一 latent**。

## 训练配方（论文给出的硬事实）

* 冻结：video generators、VAE、原 Transformer 权重、camera/time token、motion decoder、tracking head。
* 可训练：Alignment Module、prediction heads、refinement 的 rank-16 LoRA（`Δψ`）。
* 渐进激活：`set_stage(1)=alignment → 2=+heads → 3=+LoRA`；起点为 “4RC-aligned latent adapter”。
* 监督：`L = L_unc + L_cam + L_geom`（confidence-weighted depth / depth-gradient / world-ray；相机
  translation/rotation/FOV；metric depth+ray、世界点 mean 与 tail 误差、surface-normal）。
* 未确认（YAML 中 `verify` 标注）：各阶段步数、batch、LR、优化器、warmup、LoRA 注入位置全集。

## 评测协议

* **Table 1**：每序列 2 个 off-axis 相机投影，分数 ×100；对照 matched Wan+4RC（同 latent）。
  诚实性：CogVideoX-5B+4RC 的 CLIP-I 更高，论文不主张全指标领先 → 我们的报表同样保留该列。
* **Table 2**：50 人 × 50 例 × 10 次成对比较，4 个维度偏好率 + 95% bootstrap 区间。
* **Table 3**：GT 基准上的组件消融（Acc/Comp cm、NC），用于验证设计而非主张重建 OOD 优势。
* **Fig. 6**：`z_45 - z_50` 投影到 Grid-Align width-null space，同扰动施加于两条通路，
  `ρ=0.6` 处点图漂移 ours ≈ 0.005/0.005，cascade ≈ 0.38/0.32。

## 算力与时间预算（估计）

| 项 | 估计 |
|---|---|
| 1,143 clips VAE 预编码 | 1×A100 40GB，<2 h |
| 三阶段训练（token_dim 1024×31 block + LoRA） | 4×A100/H100 80GB，~3–5 天（batch 2/卡 + 梯度累积） |
| 3 个 DiT × 200 用例 latent 采样 | 1×80GB，每模型 ~6–10 h（50 steps） |
| 投影指标（DINOv2 + 双视图 × 200 × 4 方法） | <2 h |
| 本机现状 | 驱动过旧（CUDA 不可用）→ 仅 CPU 冒烟；训练需在 GPU 机执行 |

## 风险与对策

| 风险 | 对策 |
|---|---|
| 4RC 权重不可得 / 维度不符 | 已把 `depth/heads/dim` 全配置化；`load_4rc_init()` 做名称级 best-effort 映射并报告命中数；缺权重时随机初始化仍可验证结构与训练动力学 |
| “width-null space” 附录定义未知 | 实现为逐 token 感受野算子的零空间；若算子满秩则退化到最小奇异方向（代码内注明该歧义） |
| DINO set-F1 / match 精确定义未知 | `l4d/eval/projection_metrics.py` 给出阈值化集合 F1 参考实现 + 与论文数字的对照表，附录到位后替换函数体即可 |
| Benchmark 用例无法获得 | `build_benchmark.py` 自建并 sha256 锁定；报表明确标注「自锁基准，非官方」 |
| off-axis 两相机外参未知 | `OffAxisProtocol` 全可配（elevation/azimuth/distance/fovy），默认值标注 `verify` |

## 下一步（按优先级）

1. 在可用 GPU 的机器上 `fetch_weights.sh` + `precompute_latents.py`，用真实 1,143 clips 跑 stage1。
2. 用 4RC 权重初始化后重跑 `eval_gt.py` 五变体，比对 Table 3 的相对次序（去 3D Conv / 去任一注意力应显著变差）。
3. 采样三个 DiT 的 terminal latent，跑 `eval_generation.py` 与 matched cascade，检查 DINO-F1 增益是否落入 [2.88, 3.45] 与 5.81。
4. 作者补充材料发布后，用其超参/数据集清单/指标定义替换所有 `verify` 项，并更新 `docs/PAPER_ANALYSIS.md` 第 8 节。
