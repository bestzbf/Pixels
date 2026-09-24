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

Transfer into L4AR (`tools/train.py` logs it per run): **480/480 block tensors** (40 blocks × 12, through the
LoRA `base.weight` wrappers) and **10/10 `cam_dec` camera-head tensors** (`fc_qvec.weight` bit-equal),
0 shape mismatches. LayerNorm gains are transferred **raw**: they span −0.001…1.02 across depth, which is a
learned distribution, not a `1+γ` convention. DualDPT geometry-head internals are reported unmapped.

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

Metric: `tools/eval_recon.py` → relative aligned point error (`rel_err`), Acc/Comp after scale alignment,
inlier fraction at 2 % of scene extent.

ETH3D 4-clip pool, 2000 steps × 3 stages:

| model | mean rel_err | courtyard inliers | courtyard comp | office comp |
|---|---|---|---|---|
| untrained | 0.0446 | 0.1 % | 2.724 m | 1.246 m |
| random init, 512d/12 blocks | 0.0410 | — | 1.876 m | 0.924 m |
| **DINOv2 init, 768d/12 blocks** | **0.0396** | **5.9 %** | **1.632 m** | 0.936 m |
| random init, paper scale (1500 steps) | 0.0244 / 0.0400 / 0.0455 / 0.0576 (office/pipes/statue/courtyard) | 5.0 % | 2.164 m | 1.177 m |

Reading: a pretrained hierarchy matters more than scale — the 1.16 B random-init model barely moves
(office/pipes flat, courtyard −14 %) while the 768 d DINOv2-init run improves every clip. That is the
paper's premise, reproduced directionally on local data.

⏳ queued: 4RC-backbone-init paper scale (chain4, stage2 running), 4RC backbone + 4RC camera head (chain5),
per-variant Table 3 on the real pool (chain6). This table gets updated as they land.

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

29/29 tests pass (`python tests/test_reproduction.py`, plus `test_scannet.py` 3, `test_colmap.py` 5),
covering Eq. 5/6/7 semantics, staged freezing order, the shared-VAE compatibility gate, SwiGLU→MLP and
camera-head transfer, raw LayerNorm-gain transfer, cache-vs-fresh latent equivalence (max diff 0.0),
compact-checkpoint drop reporting, and the QC metrics on oversized clouds.

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
