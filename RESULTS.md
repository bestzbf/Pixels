# RESULTS — measured state of the Latent-to-4D reproduction

Every number here was produced on this machine (RTX 3090 24 GB, `/home/zbf/Desktop/dl_env`,
torch 2.6.0+cu124). Paper reference values are quoted from `docs/PAPER_ANALYSIS.md`; they are **reference
rows**, not targets we claim to have hit. Queued runs are marked ⏳ and this file is where they land.

## 1. Weights (`/mnt/data/pixels-weights`)

| weight | size | verification |
|---|---|---|
| `wan-vae/` Wan2.1 video VAE (diffusers layout) | 507,591,892 B | 194 tensors; loads on GPU; `21×192×256 → z_obs (1,16,6,24,32)`; `config.json` supplies the 16-channel `latents_mean/latents_std` now used by `SharedLatentInterface.normalize` |
| `4rc/model.safetensors` (the paper's initialiser) | 6,080,387,740 B | 1078 tensors / 1520.1 M params; `backbone.pretrained.blocks.{0..39}` at d=1536, fused `qkv (4608,1536)` ⇒ 24 heads; SwiGLU `w12 (8192,1536)`/`w3 (1536,4096)` ⇒ FFN 4096 = 8/3·d |
| `dinov2-base/model.safetensors` | 346.3 MB | 12 blocks; used as the substitute initialiser and as the Table 1 projection encoder |
| `clip-vit-large-patch14/model.safetensors` | 1,710,540,580 B | loads for Text CLIP / CLIP-I |

Transfer into L4AR, re-measured on the current architecture by calling `load_4rc_init` directly on the
6.08 GB file (`918.1 M` params built from `configs/model/l4ar_paper.yaml`, CPU): **480 of 800 block
tensors** copied from 40 source blocks with **0 shape mismatches**. Each block holds 20 parameters, 12 of
them non-LoRA, so that is **480/480 of every transferable block tensor**; the other 8 per block are the
rank-16 LoRA A/B pairs on `qkv/proj/fc1/fc2`, for which 4RC has no counterpart. **Camera head 10/10** under `camera_layout: cam_dec`.
Not just names-and-shapes, values verified bit-equal:

| check | result |
|---|---|
| `cam_dec.fc_qvec.weight` → `camera_head.fc_qvec.weight` (4, 3072) | bit-equal |
| `cam_dec.fc_t.weight` → `camera_head.fc_t.weight` | bit-equal |
| SwiGLU `w12[:4096]` → `blocks.0.mlp.fc1.weight` (4096, 1536) | bit-equal |
| SwiGLU `w3` → `blocks.0.mlp.fc2.weight` (1536, 4096) | bit-equal |
| `blocks.39.attn.qkv.weight` → `blocks.39.attn.qkv.base.weight` (4608, 1536) | bit-equal |

LayerNorm gains are transferred **raw**: 80 gains spanning −0.0011…1.0229 (14 near zero) is a learned
distribution, not a `1+γ` convention. DualDPT geometry-head internals are reported unmapped and stay
randomly initialised.

Sources that failed: `huggingface.co` and `drive.google.com` (0 B/s), `modelers`/ModelScope mirror of 4RC (404).
Working channel: `scripts/fetch_4rc_mirror.sh` via `hf-mirror.com`, sizes taken from the repo tree API.

## 2. GPU

| item | finding |
|---|---|
| driver | 535.309.01 → exposes CUDA 12.2 |
| system python + conda `d4rt`/`stablemvs`/`trellis2` | torch 2.12.1**+cu130** → `cuda.is_available() False` (needs a ≥580 driver) |
| `Desktop/dl_env` | torch 2.6.0**+cu124** → GPU works (4k matmul verified), used by every run here |
| fix for other envs | either pin a cu12x wheel (`scripts/fix_gpu.sh`, Aliyun pypi mirror: pypi.org measured 5 KB/s) or upgrade the driver to 580 (needs root, not performed) |

Paper-scale model trains on this single card: `l4ar_paper.yaml` = **918.1 M parameters, 36.4 M trainable**
(alignment + heads + rank-16 LoRA), forward+backward verified, peak **7.8 GB** at 9×128×128 and
**10.8 GB** during the 5-clip runs at 128×128.

## 3. Data actually used

ScanNet is **not on this machine**: a whole-local-disk sweep for `*scannet*`, `scene[0-9][0-9][0-9][0-9]_*`,
`*.sens` returns only this repository's own files (including `data/scannet/`, which is a synthetic
ScanNet-*format* fixture). The NAS (`192.168.2.233:445`) is unreachable from this box and mounting needs root.

Real pools used instead, after `tools/check_dataset.py` gating (consecutive-view NN / scene extent ≤5%, scale
-normalised duplicate fingerprint, depth-scale scan):

| clip | source | depth coverage | cross-view consistency | note |
|---|---|---|---|---|
| office / statue / pipes / courtyard | ETH3D COLMAP (THIN_PRISM_FISHEYE 6048×4032) | 23–71 % (reprojected `points3D`) | 0.07–2.1 % | camera trajectories 4.8–19.6 m |
| colmap | `BioPhysGS-paper/.../converted/colmap` | **100 %** (per-view dense mm maps) | 0.45 % | PINHOLE 256², exact K |
| pancakes | same asset as `colmap` | 100 % | 0.45 % | **dropped: duplicate fingerprint** |
| pgsr | converted tree | 0 % (`points3D.ply` only) | — | dropped: unsupervisable |

Pool manifest: `data/real/manifest.jsonl` (5 unique clips) — `configs/data/real.yaml`.

## 4. Learning effect so far

Metric: `tools/eval_recon.py` → relative aligned point error (`rel_err`, fraction of scene extent, lower is
better), inlier fraction and Acc/Comp after scale alignment. Every row below is now measured on the same
5-clip real pool **with the current harness**, and each trained row is paired with the control the old
table lacked: same architecture, same pretrained init, trained tensors *not* loaded (`--from-scratch`).

| architecture (steps/stage) | trained | control: init only | Δ mean rel_err over the 4 ETH3D clips |
|---|---|---|---|
| DINOv2 init, 768 d/12 blocks (2000×3) | **0.0401** | 0.0449 | **−10.7 %** |
| 4RC init + `cam_dec` head, paper scale 918 M (1500×3) | 0.0433 | 0.0449 | −3.6 % |

Per-clip `rel_err` (control → trained) and the inlier/Completeness movement behind the means:

| clip | DINOv2 768 d | paper scale 4RC | courtyard inliers | pipes comp | office comp |
|---|---|---|---|---|---|
| courtyard | 0.0667 → 0.0509 | 0.0666 → 0.0608 | 0.2 % → 4.5 % / 0.2 % → 2.2 % | — | — |
| pipes | 0.0400 → 0.0399 | 0.0400 → 0.0394 | — | 0.644 → 0.621 m / 0.645 → 0.456 m | — |
| office | 0.0244 → 0.0244 | 0.0244 → 0.0243 | — | — | 1.265 → 1.170 m / 1.232 → 1.097 m |
| statue | 0.0487 → 0.0452 | 0.0486 → 0.0486 | — | — | — |
| colmap (dense, 8.6 m extent) | 0.3324 → 0.3312 | 0.3325 → 0.3228 | control has **no** sub-threshold pairs; the trained paper model reaches Acc 0.092 m / Comp 0.114 m |

Reading, without spin:

* The gain is concentrated in the hard clips. `office` sits at ~0.0243 in *every* configuration - a floor for
  this pool, not progress - while courtyard moves −24 % (768 d) and −9 % (paper scale) and statue −7 % (768 d).
* The 1.7× smaller DINOv2-init model gains more than the 918 M one at a comparable budget. With 5 clips and
  1500 steps/stage the paper-scale hierarchy is nowhere near the regime the paper trains in (1,143 clips),
  so this is a statement about this budget, not about the architecture.
* **A pretrained init alone does nothing**: untrained-4RC 0.0449 ≈ untrained-random 0.0450
  (`runs/baseline_real_dinov2.txt`). The benefit has to be trained in, which is what the paired controls show.

### 4.1 A harness bug that invalidated an eval, and what fixed it

`tools/eval_recon.py` built the model and then restored a **trainable-only** checkpoint into it, so the
frozen 40-block hierarchy was re-randomised at eval time and the trained heads were read through a network
they had never seen. The symptom is exactly what a lookup table of "did this evaluate the trained model?"
should look like: chain5's first evaluation returned the *untrained* numbers to three decimals
(0.3322/0.0666/0.0244/0.0399/0.0489 vs the untrained 0.3325/0.0668/0.0243/0.0399/0.0484). Full-state
checkpoints (the 768 d and 918 M runs saved before compact checkpoints landed) carry the backbone and were
never affected - which is why only the new-style runs were wrong.

Fixed by `l4d.models.l4ar.apply_pretrained_init` / `build_initialised_model`: every tool that restores a
checkpoint (`train`, `eval_recon`, `eval_gt`, `eval_generation`, `infer`, `residual_sensitivity`) now rebuilds
the init the same way first, so the existing `frozen_fingerprint` guard can actually pass.
`tests/test_reproduction.py::test_trainable_only_checkpoint_needs_the_training_init` pins the contract in
both directions - restored-with-init reproduces the trained predictions, restored-without-init does not -
and `apply_pretrained_init` now reports the frozen backbone's real source, so a config whose checkpoint
transfers no block tensors is labelled `random` instead of `4RC`. That labelling had already drifted:
`runs/table3_real/Full` claims `init_source: 4RC` while its backbone is random (512 d cannot take any of
4RC's 1536 d block tensors; only 3 shape-compatible camera-head biases landed, and those are trainable).

**chain4 retired, not reported.** The first 4RC-init run (`1500` steps/stage) loaded
`configs/model/l4ar_paper.yaml` at 23:02 — nine minutes *before* `mlp_ratio` was corrected 4.0 → 2.6666667
(commit 4901dbe) and fourteen minutes before `camera_layout: cam_dec` landed (13f29d2). Its own checkpoint
betrays it: stored `mlp_ratio: 4.0` (so FFN 6144, which 4RC's 4096-wide SwiGLU tensors cannot fill) and a
frozen fingerprint full of `camera_head.mlp.*`. It was never a 4RC-initialised model, so `runs/gpu_real_4rc`
was renamed `runs/retired_pre_arch_fix_chain4` and its numbers are discarded rather than tabulated. The
guard that caught it is the one that matters: `load_checkpoint` reports dropped/renamed tensors, and the
config frozen inside every checkpoint makes the architecture at *load time* auditable after the fact.
`tools/train.py` now also stamps the effective `--data` path into each stage record (it used to inherit the
plan's `configs/data/clips_final.yaml`, which is not what ran).

⏳ running: **chain6** = per-variant Table 3 training on the same pool (`runs/table3_real`, 512 d/12 blocks,
800 steps × 3 stages per variant, `w_o_Global` last). chain5 (paper scale, 4RC backbone + `cam_dec` head) is
the second row of the table above; its checkpoints are `runs/gpu_real_4rc_camdec/stage{1,2,3}*.pt` and its
re-verified evaluation is `runs/gpu_real_4rc_camdec/clouds_checked/`.

## 5. Table 3 (component ablation) status

`tools/eval_gt.py` now accepts a real pool, and `scripts/run_table3.sh` **trains each variant separately**
before scoring it. An earlier attempt that scored untrained variants produced Acc = 3.1–3.5 for all five
variants — a table with no discriminating power; it is deliberately not reported as a result.
Paper rows (7-Scenes/NRGBD, cm) stay printed as reference only.

## 6. Generation metrics (Table 1 pipeline)

With the local DINOv2-base and CLIP-ViT-L/14 as real encoders, two off-axis views per case
(`tools/eval_generation.py --extractor dinov2`), on randomly-initialised geometry and synthetic latents:

```
Text CLIP 10.68   CLIP-I 66.18   DINO global 27.31   DINO match 27.30   DINO F1 3.87
```

These validate the **metric computation**, not the claim: no DiT latents or locked 200-case benchmark were
produced here, so the paper's +2.88–3.45 / +5.81 DINO-F1 gains are **not reproduced and not claimed**.
The reference rows and the gain arithmetic from the paper's own table are checked by
`tests/test_reproduction.py::test_table_one_gains_match_claims`.

## 7. Framework checks

30/30 tests pass (`python tests/test_reproduction.py`, plus `test_scannet.py` 3, `test_colmap.py` 5),
covering Eq. 5/6/7 semantics, staged freezing order, the shared-VAE compatibility gate, SwiGLU→MLP and
camera-head transfer, raw LayerNorm-gain transfer, cache-vs-fresh latent equivalence (max diff 0.0),
compact-checkpoint drop reporting, and the QC metrics on oversized clouds. Two of them exist because of
mistakes made in this session: the init-restoration contract above (§4.1) and the subsampled-cloud normal
indexing in `accuracy_completeness`.

## 8. Reproduce

```bash
bash scripts/smoke_test.sh                     # offline structure + math checks, CPU
bash scripts/fetch_weights_modelscope.sh      # VAE/DINOv2/CLIP (ModelScope: MB/s here; scripts/fetch_weights.sh is the HF route)
bash scripts/fetch_4rc_mirror.sh               # the 6.08 GB 4RC initialiser
python tools/prepare_colmap.py --root "<ETH3D root>" --out data/real --staging data/real_staging \
       --height 128 --width 128 --export-all --dataset real_local                    # or ScanNet via next line
bash scripts/train_on_scannet.sh /path/to/scannet 1143                              # one command, ScanNet
python tools/check_dataset.py --data configs/data/real.yaml --threshold 0.05 --ascent
python tools/train.py --model configs/model/l4ar_paper.yaml --data configs/data/real.yaml --device cuda --steps 1500
python tools/eval_recon.py --model configs/model/l4ar_paper.yaml --data configs/data/real.yaml \
       --checkpoint runs/.../stage3_lora.pt --from-scratch                            # drop flag for baseline
bash scripts/run_table3.sh configs/data/real.yaml runs/table3_real
```
