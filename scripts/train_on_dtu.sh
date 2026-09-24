#!/usr/bin/env bash
# Train on the complete DTU release (23 mvsnet-preprocessed scans) as soon as scripts/fetch_mvs_datasets.py has it.
#
#   bash scripts/train_on_dtu.sh                                    # waits for the fetch, then ingest+train
#   MIN_SCANS=8 SKIP_WAIT=1 STEPS=2000 bash scripts/train_on_dtu.sh
#
# Same reuse argument as the other benchmark paths: tools/prepare_benchmark.py turns
# cams/%08d_cam.txt + images/%08d.png + gt_depths/%08d.pfm into the COLMAP model and uint16-millimetre depth
# maps that the existing stager already prefers, so clip export, the QC gate, the latent cache and the staged
# trainer are shared with the local scenes rather than reimplemented per dataset.
#
# DTU is a controlled turntable set: 49 views per scan around one static object, metric in millimetres, with
# a GT mesh (scan.ply) per scan. That is a different regime from ScanNet room scans - useful geometry
# supervision, but a 21-frame clip is 21 views of one object, so clip counts are bounded by 23 scans.
set -uo pipefail
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
cd "$(dirname "$0")/.."

BENCH=${BENCH:-/mnt/data/pixels-benchmarks}
DTU=$BENCH/dtu
MIN_SCANS=${MIN_SCANS:-20}
FRAMES=${FRAMES:-49}
STRIDE=${STRIDE:-6}
STEPS=${STEPS:-2000}
MODEL=${MODEL:-configs/model/l4ar_paper.yaml}
PY=${PY:-/home/zbf/Desktop/dl_env/bin/python}
STAGING=data/dtu_staging
POOL=data/dtupool
DATA=configs/data/dtu.local.yaml
OUT=${OUT:-runs/dtu}
GPULOCK=${GPULOCK:-/tmp/pixels_gpu.lock}
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
exec 9>"$GPULOCK"
gpu() { log "claiming the GPU lock"; flock -w "${GPU_TIMEOUT:-28800}" 9 || { log "GPU lock timed out"; return 1; }
        "$@"; local rc=$?; log "released the GPU lock (rc=$rc)"; return $rc; }

scans() { find "$DTU" -mindepth 1 -maxdepth 1 -type d -name "scan*" 2>/dev/null | wc -l; }
complete() { find "$DTU" -mindepth 2 -maxdepth 2 -type f -name done 2>/dev/null | wc -l; }
if [ "${SKIP_WAIT:-0}" != 1 ]; then
  while [ "$(complete)" -lt "$MIN_SCANS" ]; do log "DTU ready: $(complete) complete scans (need $MIN_SCANS)"; sleep 300; done
fi
log "DTU: $(complete) complete scans, $(scans) directories"

mkdir -p "$POOL" "$STAGING"
$PY tools/prepare_benchmark.py --root "$DTU" --out /tmp/dtu_models --depth-divisor 1000 || exit 1
$PY tools/prepare_colmap.py --root /tmp/dtu_models --staging "$STAGING" --out /tmp/dtu_dummy \
  --height 192 --width 256 --frames "$FRAMES" --dataset dtu || exit 1
$PY - "$STAGING" "$STRIDE" <<'PY'
import json, sys
from l4d.data.scannet import export_manifest
staging, stride = sys.argv[1], int(sys.argv[2])
for length in (21, 13):
    print(length, json.dumps(export_manifest(staging, f"data/dtu{length}", clip_length=length,
          clip_stride=stride, size=(192, 256), dataset=f"dtu{length}")), flush=True)
PY
cat data/dtu21/manifest.jsonl data/dtu13/manifest.jsonl > "$POOL/manifest.jsonl"
log "pool before QC: $(wc -l < "$POOL/manifest.jsonl") clips"

sed -e "s#^manifest:.*#manifest: $POOL/manifest.jsonl#" -e "s#^root:.*#root: $POOL#" \
    -e "s#^latent_cache:.*#latent_cache: $POOL/latents#" -e "s#^datasets:.*#datasets: [dtu21, dtu13]#" \
    configs/data/seven.yaml > "$DATA"

gpu $PY tools/check_dataset.py --data "$DATA" --threshold 0.05 --ascent --emit-manifest "$POOL/manifest.jsonl" | tail -6
log "pool after QC: $(wc -l < "$POOL/manifest.jsonl") clips"
gpu $PY tools/precompute_latents.py --data "$DATA" --model "$MODEL" --device cuda || log "encoding on the fly"
gpu $PY tools/train.py --model "$MODEL" --data "$DATA" --device cuda --steps "$STEPS" --output "$OUT"

log "scoring on the same pool - DTU ships no clip-level split, so this is NOT held-out evaluation"
log "for a held-out number, hold back whole scans with MIN_SCANS and a second pool; do not read this as test-set accuracy"
gpu $PY tools/eval_recon.py --model "$MODEL" --data "$DATA" --device cuda --checkpoint "$OUT/stage3_lora.pt" --out "$OUT/clouds"
gpu $PY tools/eval_recon.py --model "$MODEL" --data "$DATA" --device cuda --from-scratch
gpu $PY tools/eval_recon.py --model "$MODEL" --data "$DATA" --device cuda --from-scratch
log "DTU_TRAINING_DONE"
