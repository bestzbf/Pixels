# 论文分析：Beyond Pixels: From Video Priors to 4D Worlds

- **arXiv**: 2608.10744v1 [cs.CV], 11 Aug 2026
- **作者**: Zihao Liu, Xiaolong Shen, Zhenglin Zhou, Ruijie Quan, Yi Yang (通讯) — ReLER, CCAI, Zhejiang University
- **项目页**: https://hayd-zju.github.io/Beyond-Pixels/
- **本地 PDF**: `paper_2608.10744.pdf`（17.6 MB，**9 页，v1 不含附录**；正文 `paper_main_text.txt`，
  arXiv HTML 渲染 `paper_html_text.txt`）。文中 7 处 “Appendix” 均为前向引用——架构超参、数据集清单、
  指标精确定义、人评细节在 v1 中**不可获得**，只能等作者补充材料。

---

## 1. 一句话总结

提出 **direct latent-to-4D generation**：不再生成 RGB 视频后再重建 4D，而是把「共享同一个 VAE 的视频扩散模型（DiT）的最终去噪 latent」当作**可复用接口**，直接映射到预训练 4D 重建器的 token 空间并解码出相机 + 动态世界坐标点图。实例化模型记为 **Latent-to-4D**，其核心网络记为 **L4AR**（Latent-to-4D Alignment and Refinement）。

## 2. 动机与两类现有范式的缺陷

| 范式 | 代表 | 缺陷 |
|---|---|---|
| Generate-then-reconstruct（先生成后重建） | Diffusion4D, 4Diffusion, CAT4D | RGB 接口把「只在窄重建数据上训练的独立重建器」插到视频先验与 4D 输出之间 → 分布失配、误差传播（开放域内容与时序伪影直接进入几何） |
| Integrated feed-forward（一体化前馈生成） | 4DNeX, Diff4Splat, WorldReel, WorldForge | 4D 预测绑定到**特定生成器 / 特定条件方式**，换生成器或换条件需要重新做几何监督训练 |

关键约束：**4D 监督相对海量视频数据极度稀缺** → 需要一个「绕过生成 RGB、且一条几何监督通路可服务多个兼容视频生成器」的可复用接口。

## 3. 核心洞察（可复用接口的来源）

共享同一 VAE checkpoint 的视频生成器，其 **DiT 主干与条件方式可以不同**，但只要满足：

1. 同一 VAE checkpoint；
2. 同一 latent 归一化（scaling / normalization）；
3. 同一 tensor layout；
4. 同一压缩约定（空间/时间下采样倍率）；
5. 受支持的 latent 形状；

那么它们的**最终去噪 latent 落在同一个表示空间** $\mathcal{Z}_v$（分布可以不同）。该表示位于 RGB 解码**上游**，因此可作为访问外观/运动/条件信息的公共入口。

## 4. 问题形式化（论文式 1–7）

- 视频时空 VAE：$(E_v, D_v)$，latent 空间 $\mathcal{Z}_v$。
- 预训练 4D 层级的结构化 token 空间：$\mathcal{Z}_{4D}$，元素 $\mathbf{Q}$ 支持相机与动态几何预测。
- 二者虽是同为时空表示，但**时间分辨率、空间网格、特征维度都不同**，不可直接互换。

(1) 对齐映射
$$\mathcal{A}_\phi:\mathcal{Z}_v\rightarrow\mathcal{Z}_{4D},\quad \mathbf{Q}^{(0)}=\mathcal{A}_\phi(\mathbf{z}_v)$$

(2) 离散 4D 场景输出
$$\mathcal{Y}=\{(\mathbf{C}_t,\mathbf{P}_t)\}_{t=1}^{T},\quad \mathbf{P}_t\in\mathbb{R}^{H\times W\times 3}$$
$\mathbf{C}_t$：第 $t$ 帧相机；$\mathbf{P}_t$：共享世界坐标系下的稠密点图。

(3) 传统级联（要被绕过的）
$$\widehat{\mathcal{Y}}_{rgb}=R\!\left(D_v(\mathbf{z}_v)\right)$$

(4) 本方法的因子化（对齐 → 时空精化 → 4D 解码）
$$\widehat{\mathcal{Y}}=\mathcal{D}_\omega\!\left(\mathcal{H}_{\psi,\Delta\psi}\bigl(\mathcal{A}_\phi(\mathbf{z});\mathbf{S}\bigr)\right)$$
- $\mathcal{H}_{\psi,\Delta\psi}$：**冻结**预训练权重 $\psi$ + **轻量可训练**更新 $\Delta\psi$（LoRA）；
- $\mathbf{S}$：**冻结**的 camera token 与 time token。
- $\mathbf{z}\in\{\mathbf{z}^{obs},\mathbf{z}^{gen}\}$，其中
  $\mathbf{z}^{obs}=\mu(E_v(\mathbf{V}))$（后验均值），$\mathbf{z}^{gen}=G_\theta(c,\boldsymbol\epsilon)$（兼容 DiT 在 VAE 解码前的最终去噪 latent）。

**训练/推理不对称**（这是复现的关键点）：训练时只用观测视频的 VAE 编码 $\mathbf{z}^{obs}$，**不运行也不优化 DiT**；推理时把 $\mathbf{z}^{obs}$ 换成 $\mathbf{z}^{gen}$，下游通路完全不变，也不需要任务 ID / 条件分支。

(5) Alignment 细节
$$\mathbf{Q}^{(0)}=\mathcal{A}_\phi(\mathbf{z})=\operatorname{Flatten}\!\left(\mathcal{S}_\phi\!\left(\mathcal{R}(\mathbf{z})\right)\right)\in\mathbb{R}^{T\times M\times d}$$
- $\mathcal{R}$：**固定**三线性重采样，对齐所需时空分辨率（不学习）；
- $\mathcal{S}_\phi$：**学习型 3D 卷积**，做局部时空邻域聚合 + 通道数投影到 4D 层级的 $d$；
- $T$ 帧数，$M$ 每帧空间 token 数，$d$ token 维度。

(6) 4D 解码（射线参数化）
$$\widehat{\mathbf{P}}_t(u)=\widehat{\mathbf{o}}_t+\widehat{d}_t(u)\,\widehat{\mathbf{r}}_t(u)$$
- multi-level geometry head：每帧每像素预测深度 $\widehat d_t(u)$、世界射线原点 $\widehat{\mathbf o}_t$ 与单位方向 $\widehat{\mathbf r}_t(u)$，**并各自带 confidence**；
- camera head：9D 姿态 + FOV 参数化预测 $\widehat{\mathbf C}_t$。

(7) 训练目标
$$\mathcal{L}=\mathcal{L}_{unc}+\mathcal{L}_{cam}+\mathcal{L}_{geom}$$
- $\mathcal{L}_{unc}$：confidence-weighted 的 depth loss、depth-gradient loss、world-ray loss；
- $\mathcal{L}_{cam}$：相机 translation / rotation / FOV；
- $\mathcal{L}_{geom}$：metric depth + ray 监督，世界点 mean error 与 **tail error**，加 surface-normal loss。

## 5. 模块结构（复现对应关系）

| 模块 | 论文描述 | 可训练性 | 代码 |
|---|---|---|---|
| Alignment Module $\mathcal{A}_\phi$ | 固定三线性网格重采样 + 学习型 3D 卷积 + flatten | 训练 | `l4d/models/alignment.py` |
| Spatiotemporal Refinement $\mathcal{H}$ | **31 个 block**，先用 frame-wise 建立每帧结构，然后与 global 交替；多深度 tap 拼接 | $\psi$ 冻结 + $\Delta\psi$ = rank-16 LoRA | `l4d/models/refinement.py`, `l4d/models/lora.py` |
| 4D Decoder $\mathcal{D}_\omega$ | 由预训练重建器（4RC）初始化；geometry head + camera head | head 训练；camera/time token、motion decoder、tracking head 冻结 | `l4d/models/decoder.py`, `l4d/models/heads.py` |
| 视频侧接口 | 冻结 Wan VAE（训练）/ 冻结兼容 DiT（推理） | 全冻结 | `l4d/models/video_interface.py` |
| 整体装配 | L4AR | 分阶段渐进激活 | `l4d/models/l4ar.py` |

Refinement 的两种注意力（形状约定，务必按论文）：
- frame-wise：$(B,T,M,d)\rightarrow(BT,M,d)$，在 $M$ 个空间 token 上做 attention（帧内结构）；
- global：$(B,T,M,d)\rightarrow(B,TM,d)$，在 $TM$ 个 token 上做 attention（跨帧对应/视点/运动）；
- multi-level 表示：在选定深度处把**最近的 frame-wise 特征与最近的 global 特征拼接**。

## 6. 实验设置（复现必须对齐的协议）

**实现细节**
- 冻结 **Wan VAE**；31-block refinement 层级与预测头由 **4RC** 初始化（引 Oquab 2024 = DINOv2, Luo 2026 = 4RC）。
- 训练 Alignment Module + 预测头；refinement 层级用 **rank-16 LoRA** 适配；视频模型与预训练 base weights 冻结。
- 从 *4RC-aligned latent adapter* 起步做多阶段几何监督训练；**最终阶段用 6 个重建数据集的 1,143 个 clips**（正文摘要统称 "roughly 1K clips"）。
- 所有 benchmark 均为 held-out，**不做 test-time adaptation**。
- 渐进激活可训练组件（stage-wise），以避免破坏预训练几何先验。

**Baseline / 生成器（同一 checkpoint 跨 3 个 DiT）**
- T2V：Wan2.1-T2V-14B、Wan2.1-T2V-1.3B、CogVideoX-5B（仅级联 baseline 用）
- I2V：Wan2.2-I2V-A14B
- 级联（decode 同一 latent 后重建）：+4RC、+$\pi^3$、+Any4D；原生 I4D baseline：4DNeX
- **受控同-latent 对比** = Ours vs matched Wan+4RC。

**Benchmark 与指标**
- Text4D-200 / I4D-200：锁定 200 例，每方法在全部 200 例上评测。
- 每个预测点序列从 **2 个 off-axis 相机**渲染；报告 Text CLIP、RGB-reference CLIP-I、DINO global、DINO match（valid-patch）、DINO set F1；分数 ×100。
- 生成的 RGB **仅用于评测与上色**，不作为几何输入。
- off-axis DINO 分数是「可见几何一致性/完整性」的外观相关代理，**不是度量 4D 精度**。
- GT 诊断基准：7-Scenes（18）、NRGBD（9），指标 Acc / Comp（cm，越低越好）、NC（越高越好）。

## 7. 主要结果（复现目标数字）

Table 1（×100，越高越好；Ours 与 matched Wan cascade 共享同一生成 latent）

| Method | Text CLIP | CLIP-I | DINO global | DINO match | **DINO F1** |
|---|---|---|---|---|---|
| CogVideoX-5B + 4RC | 26.887 | 75.33 | 42.71 | 53.47 | 53.27 |
| CogVideoX-5B + π³ | 26.293 | 74.84 | 38.74 | 50.29 | 49.72 |
| CogVideoX-5B + Any4D | 25.414 | 72.66 | 29.46 | 45.97 | 45.12 |
| Wan2.1-14B + 4RC | 28.116 | 71.20 | 42.45 | 54.31 | 53.56 |
| Wan2.1-14B + π³ | 26.594 | 68.70 | 34.79 | 48.69 | 47.50 |
| Wan2.1-14B + Any4D | 26.261 | 67.56 | 29.97 | 46.80 | 45.55 |
| Wan2.1-1.3B + 4RC | 27.829 | 71.34 | 43.30 | 54.97 | 54.21 |
| Wan2.1-1.3B + π³ | 27.034 | 70.23 | 38.51 | 51.23 | 50.10 |
| Wan2.1-1.3B + Any4D | 26.326 | 68.19 | 31.00 | 47.26 | 46.06 |
| **Ours (Wan2.1-14B)** | 28.544 | 72.24 | 45.43 | 57.52 | **57.01** |
| **Ours (Wan2.1-1.3B)** | 28.434 | 72.32 | 46.02 | 57.64 | **57.09** |
| 4DNeX | 22.844 | 61.11 | 11.31 | 30.53 | 28.33 |
| Wan2.2-I2V-A14B + 4RC | 26.340 | 70.55 | 47.83 | 56.25 | 55.79 |
| Wan2.2-I2V-A14B + π³ | 24.678 | 66.65 | 33.50 | 45.08 | 43.82 |
| Wan2.2-I2V-A14B + Any4D | 24.362 | 65.30 | 27.21 | 43.54 | 42.17 |
| **Ours (I2V)** | 27.340 | 72.87 | 54.85 | 61.85 | **61.60** |

- Text→4D：相对 matched 4RC 的 DINO-F1 增益 **2.88–3.45**（57.01−54.21=2.80? 论文给 2.88–3.45，即 14B: 57.01−53.56=3.45；1.3B: 57.09−54.21=2.88）。
- Image→4D：**5.81**（61.60−55.79）。
- 诚实性说明：CogVideoX-5B+4RC 的 CLIP-I 最高（75.33），作者**不声称全指标一致领先**。

Table 2（用户偏好，Ours 胜率 %，95% bootstrap 区间；50 名参与者 × 每 benchmark 50 例 × 每例 10 次评分）

| Task | condition fidelity | geometry & completeness | temporal stability | overall |
|---|---|---|---|---|
| Text-to-4D | 59.2 [54.1–64.3] | 66.8 [62.0–71.5] | 63.5 [58.4–68.5] | 65.7 [60.8–70.5] |
| Image-to-4D | 66.4 [61.8–70.9] | 72.1 [67.8–76.3] | 68.3 [63.7–72.8] | 70.6 [66.2–74.9] |

Table 3（组件消融，GT 基准；Acc/Comp 单位 cm，NC 越高越好）

| Variant | 7S Acc | 7S Comp | 7S NC | NRGBD Acc | NRGBD Comp | NRGBD NC |
|---|---|---|---|---|---|---|
| w/o Grid | 3.783 | 6.844 | 0.608 | 5.823 | 9.686 | 0.726 |
| w/o 3D Conv | 6.944 | 15.806 | 0.554 | 12.439 | 26.511 | 0.594 |
| w/o Frame | 6.688 | 20.806 | 0.513 | 12.367 | 36.982 | 0.502 |
| w/o Global | 6.754 | 19.742 | 0.559 | 13.818 | 36.207 | 0.515 |
| **Full** | **3.121** | **5.418** | **0.628** | **5.202** | **8.187** | **0.766** |

注意：**w/o Grid 意外地不差**（Acc 3.783 vs Full 3.121，Comp 6.844 vs 5.418 仍劣于 Full 但明显好于其他消融）→ 复现时 Grid 项要单独核查协议一致性；最大跌幅来自去掉 3D Conv 或任一注意力范围。

Figure 6（DiT 残差敏感度诊断）
- 把经验近终端残差 $\mathbf{z}_{45}-\mathbf{z}_{50}$ 投影到 **Grid-Align 的 width-null space**，加到观测视频 latent 上；对两条通路施加同一扰动。
- 30 项对比全部利于 Ours；$\rho=0.6$ 时点图漂移 Ours = 0.0053 / 0.0047（7-Scenes / NRGBD）vs baseline = 0.3827 / 0.3160，相机估计同趋势。

Broader applications：同一 L4AR checkpoint 直接继承上游 motion / appearance / pose / trajectory / manipulation / navigation 控制（Fig. 7–8），作者明确这只是**接口兼容性**，不主张动作成功率或物理正确性。

## 8. 复现风险与开放问题（正文未给全，需从附录/猜测确认）

| 未知项 | 影响 | 复现策略 |
|---|---|---|
| 4RC 权重是否公开、其 31-block 具体维度/注意力头数 | 决定 $\psi$、heads、token $d$ | 先按 `d=1024`（DINOv2-g 量级）参数化，权重可用性写进 `configs/model/*.yaml`；不可用时用随机初始化跑通形状与训练动力学 |
| 6 个重建数据集具体名单与 1,143 clips 划分 | 数据复现 | 候选：7-Scenes, NRGBD, RealEstate10K, DL3DF, Co3Dv2, OMNI/4D 类；以 `data/manifest.jsonl` 外置，名单待附录确认 |
| 多阶段训练各阶段预算、LR、batch、优化器 | 训练复现 | 在 `configs/train/stage*.yaml` 中给出可编辑初值并标注 TODO-verify |
| Text4D-200 / I4D-200 的 prompt/image 集合 | 评测不可比 | 需自建并 lock；本框架提供 `tools/build_benchmark.py` 与锁文件 hash 校验 |
| 「DINO set F1 / valid-patch matching」精确定义 | 指标不可比 | 附录有定义；本框架先按论文语义实现（双视图投影 + 匹配/集合 F1），并在 `l4d/eval/projection_metrics.py` 留 reference 实现 |
| off-axis 两相机的具体外参 | 指标 | 配置化（azimuth/elevation/距离），默认值标注待确认 |

**结论**：主接口（VAE latent → 4D）与 L4AR 结构、损失、评测协议在正文已足够完整以实现**结构级复现**；数值级复现受限于 4RC 权重、数据清单与附录超参，需按上表逐项补齐。

## 9. 框架落地与验证矩阵（本仓库当前状态）

论文条目 → 实现 → 离线验证方式 → 实测结果（CPU、`configs/model/l4ar_tiny.yaml`、`bash scripts/smoke_test.sh`）：

| 论文条目 | 实现 | 验证 | 实测 |
|---|---|---|---|
| Eq.(5) 网格对齐 $\mathcal{S}_\phi(\mathcal{R}(\mathbf z))$ | `models/alignment.py` | `tests::test_alignment_grid_and_token_shape` | latent (1,16,6,12,16) → tokens (1,21,48,64)，grid=(21,6,8) |
| w/o Grid、w/o 3D Conv 消融语义 | `alignment.py` 的 `grid_mode` / `use_local_conv` | `test_alignment_without_*` | 去 Grid 后 token 数落在原生 latent 网格；去 3D Conv 后退化为 1×1×1 逐点投影 |
| 31 层 frame/global 交替 + 多深度 tap | `models/refinement.py` | `test_alternating_scope_pattern_starts_with_frame`, `test_attention_token_counts_differ_by_scope` | scope 序列 F,G,F,…；tap 处拼接 (frame‖global) = 2d |
| rank-16 LoRA 且原权重冻结 | `models/lora.py` | `test_lora_rank_and_frozen_base_weights` | 所有 `lora_a` 行数为 16；base 权重 `requires_grad=False` |
| Eq.(6) 射线反投影 + 9D pose–FOV | `heads.py`/`decoder.py`/`utils/geometry.py` | `test_point_recovery_follows_equation_six`, `test_camera_9d_roundtrip` | points == o + d·r（1e-5）；相机编码 9 维可逆 |
| Eq.(7) 三项损失（unc/cam/geom，含 tail 与 normal） | `losses/objectives.py` | `test_confidence_weighted_formula`, `test_full_loss_is_finite_and_decomposes` | loss=5.47 且分解项均有限，反向传播 OK |
| 渐进激活（避免破坏预训练先验） | `l4ar.set_stage()` | `test_progressive_activation_order` | 可训练参数量 4,160 → 38,895 → 59,375 单调递增 |
| 共享 VAE 兼容条件（拒绝异族 latent） | `models/video_interface.py` | `test_compatibility_gate_rejects_foreign_vae`, `test_channel_mismatch_is_caught` | CogVideoX VAE / 通道数不符 → ValueError |
| Table 1 指标管线（双 off-axis 投影） | `eval/projection_metrics.py` | `test_off_axis_rendering_and_evaluation_pipeline`, `tools/eval_generation.py` | surrogate encoder 下 5 项指标全部产出，与论文参考行并排打印 |
| Table 3 五变体 | `eval/protocol.py` + `tools/eval_gt.py` | 工具运行 | 5/5 变体均出表（Acc/Comp/NC），与论文数字并排 |
| Fig. 6 残差 null 投影诊断 | `eval/residual_probe.py` + `tools/residual_sensitivity.py` | `test_null_space_component_is_invisible_to_grid_align` | 投影后 token 变化 ours 比 RGB 往返小 **2–10×**（随 ρ 单调，随机初始化下倍数不稳定），方向与论文“直接通路更不敏感”一致 |
| 论文数字转录一致性 | `eval/baselines.py`, `eval/protocol.py` | `test_table_one_gains_match_claims` | 3.45 / 2.88 / 5.81 增益自洽 |

合计 **21/21 通过**，五步冒烟（训练→推理→Table 3→Table 1→Fig. 6）全部可运行。

### 仍未消除的实现歧义（v1 无附录，需作者材料或自行设定）

1. **`width-null space` 精确定义**：本仓库取「逐 token 感受野算子 $\mathbb{R}^{C_z k_t k_h k_w}\!\to\!\mathbb{R}^{d}$ 的零空间」；
   该算子满秩时（如 $d=1024$、kernel $(1,2,2)$）退化为最小奇异方向（最衰减方向），代码内已注明。
2. **9D 相机编码布局**：取 `[translation(3), quaternion(w,x,y,z), (fovy, fovx)(2)]`；论文只说 “9D pose–field-of-view”。
3. **多深度 tap 位置 / patch_size / heads / token_dim / hidden 维度**：见 `configs/model/l4ar_paper.yaml` 的 `verify` 注释。
4. **训练超参（各阶段步数、LR、batch、LoRA 注入点全集）**、**6 个数据集名单与 1,143 clips 划分**：`verify` 标注。
5. **off-axis 两相机外参、DINO 匹配阈值、valid-patch 判定**：`OffAxisProtocol` / `dino_f1_threshold` 全可配，默认值待确认。
