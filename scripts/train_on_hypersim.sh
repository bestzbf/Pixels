#!/usr/bin/env bash
# Train on the complete Hypersim release: 446 indoor scenes, 51 buildings, 286 GB already on disk.
#
#   bash scripts/train_on_hypersim.sh                       # waits for nothing, ingests, then trains
#   FRAMES=96 STRIDE=12 STEPS=2000 bash scripts/train_on_hypersim.sh
#
# Why this release and not another: it is the closest thing on this machine to the paper's own ScanNet
# corpus - metric RGB-D indoor rooms, exact per-frame intrinsics and poses, no hand-held tracker drift -
# and it is 446 scenes against 7-Scenes' 7. ScanNet itself is EULA-gated and not on this machine, so the
# argument for Hypersim is scale: 21-frame windows at stride 12 give roughly 13 clips per scene, ~5800
# clips, fifteen times the 7-Scenes pool - the first pool here large enough to be called paper-scale.
#
# The chain is the same shared one every other route uses, so nothing is reimplemented per dataset:
#   1. tools/prepare_benchmark.py       -> COLMAP text models + uint16-millimetre depth (hypersim dialect)
#   2. tools/prepare_colmap.py          -> ScanNet-style staging tree at 256x192, FRAMES frames per scene
#   3. l4d.data.scannet.export_manifest -> overlapping 21- and 13-frame windows
#   4. tools/check_dataset.py           -> QC on depth consistency along matched rays, whole-building split
#   5. tools/precompute_latents.py      -> frozen Wan VAE latents, so training never re-encodes video
#   6. tools/train.py                   -> the paper's three stages
#   7. tools/eval_recon.py              -> held-out clips, trained arm paired against the same-init control
#
# Two measured conventions this route depends on, both in tools/prepare_benchmark.py: Hypersim's `pose`
# is camera->world (its documentation says the opposite; 0.0015 against 0.36-0.42 reprojection residual
# decides it), and `depth.npy` is float metres, which must NOT be passed through the image `divisor`.
#
# The gate is --gate coview, not the default nearest-neighbour one: a NN cloud distance floors at the
# *sampling density* of the point set (~0.2 m for 4000 points on a 192x256 view) rather than at the
# accuracy of the geometry, and Hypersim's dense hole-free depth sits above that floor at 0.05 of extent
# even when the data is perfect. Depth consistency along matched rays measures 0.0038-0.0048 here.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$(dirname "$0")/.."

HYP=${HYP:-/mnt/data/pixels-benchmarks/hypersim}
PY=${PY:-/home/zbf/Desktop/dl_env/bin/python}
MODEL=${MODEL:-configs/model/l4ar_paper.yaml}
FRAMES=${FRAMES:-96}
STRIDE=${STRIDE:-12}
STEPS=${STEPS:-2000}
GPULOCK=${GPULOCK:-/tmp/pixels_gpu.lock}
GPU_TIMEOUT=${GPU_TIMEOUT:-28800}
STAGING=data/hyp_full_staging
CONVERTED=data/hyp_full_models
POOL=data/hyppool_full
DATA=configs/data/hypersim.local.yaml
TESTDATA=configs/data/hypersim_heldout.local.yaml
OUT=${OUT:-runs/hypersim}
HOLDOUT=configs/data/hypersim_heldout_buildings.txt

log(){ date -u +%H:%M:%S" $*"; }
exec 9>"$GPULOCK"
gpu(){ log "claiming the GPU lock"; flock -w "$GPU_TIMEOUT" 9 || { log "GPU lock timed out"; exit 1; }; "$@"; local rc=$?; log "released the GPU lock (rc=$rc)"; return $rc; }

[ -d "$HYP" ] || { log "no Hypersim under $HYP - run scripts/fetch_mvs_datasets.py first"; exit 1; }
log "Hypersim root: $HYP ($(find "$HYP" -maxdepth 3 -name "*_rgb.png" | wc -l) frames)"

log "1/7 converting every scene to a COLMAP model with millimetre depth"
$PY tools/prepare_benchmark.py --root "$HYP" --out "$CONVERTED" > /tmp/hypersim_convert.json 2>&1 \
  || { log "conversion failed"; tail -20 /tmp/hypersim_convert.json; exit 1; }
log "   $(grep -c '"scene"' /tmp/hypersim_convert.json) scenes, splits $(sed -n 's/.*"train": \([0-9]*\).*/\1/p' /tmp/hypersim_convert.json) train frames"

log "2/7 staging at 256x192, ${FRAMES} frames per scene"
$PY tools/prepare_colmap.py --root "$CONVERTED" --staging "$STAGING" --out /tmp/hypersim_dummy \
  --height 192 --width 256 --frames "$FRAMES" --dataset hyp > /tmp/hypersim_staging.json 2>&1 \
  || { log "staging failed"; tail -20 /tmp/hypersim_staging.json; exit 1; }

log "3/7 exporting overlapping windows"
$PY - "$STAGING" "$STRIDE" <<'PY'
import json, sys
from l4d.data.scannet import export_manifest
staging, stride = sys.argv[1], int(sys.argv[2])
for length in (21, 13):
    print(length, json.dumps(export_manifest(staging, f"data/hyp_full{length}", clip_length=length,
          clip_stride=stride if stride < length else length, size=(192, 256),
          dataset=f"hyp{length}")), flush=True)
PY
mkdir -p "$POOL"
cat "data/hyp_full21/manifest.jsonl" "data/hyp_full13/manifest.jsonl" > "$POOL/manifest.jsonl"
log "pool before QC: $(wc -l < "$POOL/manifest.jsonl") clips"

sed -e "s#^manifest:.*#manifest: $POOL/manifest.jsonl#" -e "s#^root:.*#root: $POOL#" \
    -e "s#^latent_cache:.*#latent_cache: $POOL/latents#" -e "s#^datasets:.*#datasets: [hyp21, hyp13]#" \
    configs/data/seven.yaml > "$DATA"
sed -e "s/^split: train/split: test/" "$DATA" > "$TESTDATA"

log "4/7 QC on depth consistency, holding out whole buildings"
$PY tools/check_dataset.py --data "$DATA" --threshold 0.05 --gate coview \
  --emit-manifest "$POOL/manifest.jsonl" --holdout-file "$HOLDOUT" | tail -5
log "pool after QC: $(wc -l < "$POOL/manifest.jsonl") clips"

log "5/7 encoding frozen VAE latents"
gpu $PY tools/precompute_latents.py --data "$DATA" --model "$MODEL" --device cuda || log "encoding on the fly instead"

log "6/7 training: paper scale, 3 stages x $STEPS steps"
gpu $PY tools/train.py --model "$MODEL" --data "$DATA" --device cuda --steps "$STEPS" --output "$OUT"

log "7/7 held-out scoring: unseen buildings, paired against the same init with no trained tensors"
gpu $PY tools/eval_recon.py --model "$MODEL" --data "$TESTDATA" --device cuda \
  --checkpoint "$OUT/stage3_lora.pt" > "$OUT/heldout_trained.log" 2>&1
gpu $PY tools/eval_recon.py --model "$MODEL" --data "$TESTDATA" --device cuda > "$OUT/heldout_control.log" 2>&1
log "HYPERSIM_TRAINING_DONE"
