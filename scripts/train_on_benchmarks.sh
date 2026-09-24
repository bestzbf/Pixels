#!/usr/bin/env bash
# Train on the complete benchmark releases rather than a single scene.
#
#   bash scripts/train_on_benchmarks.sh                 # waits for the fetch, then ingests + trains
#   FRAMES=128 STRIDE=12 STEPS=2000 bash scripts/train_on_benchmarks.sh
#   SKIP_WAIT=1 bash scripts/train_on_benchmarks.sh     # archives already on disk
#
# Pipeline, all of it reusing the tested local-data path (no benchmark-specific trainer or metric):
#   1. tools/prepare_benchmark.py   official frames/depth/poses -> COLMAP model + uint16 mm depth maps
#   2. tools/prepare_colmap.py      -> ScanNet-style staging tree at 256x192, FRAMES frames per sequence
#   3. export_manifest              -> overlapping 21- and 13-frame windows
#   4. tools/check_dataset.py --emit-manifest -> drop cross-view outliers, duplicates and id collisions
#   5. tools/precompute_latents.py -> frozen Wan VAE cache
#   6. tools/train.py               -> three staged phases at paper scale
#   7. tools/eval_gt.py --benchmark -> Acc/Comp in cm and NC, beside the paper's reference row, on the
#      dataset's own held-out split, plus an init-only control so the delta is attributable to training.
#
# Staging is capped per sequence on purpose: 42 sequences x FRAMES frames x two window lengths is already
# several hundred clips, comparable to the paper's "roughly 1K", while the GT point maps cost ~60 MB each.
set -uo pipefail
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

BENCH=${BENCH:-/mnt/data/pixels-benchmarks}
FRAMES=${FRAMES:-128}
STRIDE=${STRIDE:-12}
STEPS=${STEPS:-2000}
STAGING=data/bench_staging
POOL=data/benchpool
MODEL=${MODEL:-configs/model/l4ar_paper.yaml}
PY=${PY:-/home/zbf/Desktop/dl_env/bin/python}
DATA=configs/data/benchmarks.local.yaml
TESTDATA=configs/data/benchmarks_test.local.yaml
OUT=${OUT:-runs/benchmarks}
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
GPU_FREE=1
wait_gpu() {  # stage 3 at paper scale peaks at 23.8 GB of this 24 GB card, so GPU stages never overlap
  while pgrep -f "tools/(train|precompute_latents|eval_recon|eval_gt)\.py" > /dev/null; do
    if [ "$GPU_FREE" = 1 ]; then GPU_FREE=0; log "GPU busy, waiting: $*"; fi
    sleep 120
  done
  GPU_FREE=1
}

if [ "${SKIP_WAIT:-0}" != 1 ]; then
  # the single stairs scene and the full release share a parent directory, so "7scenes exists" is not the
  # signal; count the official scene names instead, or this would fire after NRGBD with one scene on disk
  scenes_present() {
    find "$BENCH" -maxdepth 3 -type d 2>/dev/null \
      | grep -ciE "/(flames|heads|stairs|office|pumpkin|redkitchen|chess|gideon)$"
  }
  while :; do
    nrgb=$(find "$BENCH" -maxdepth 2 -type d -iname "*nrgb*" 2>/dev/null | head -1)
    count=$(scenes_present)
    [ -n "$nrgb" ] && [ "$count" -ge 6 ] && break
    log "waiting: NRGBD dir='$nrgb', 7-Scenes scenes present=$count/6"
    sleep 300
  done
  log "archives ready: NRGBD=$nrgb, scenes=$count"
fi
seven=${seven:-$BENCH/7-scenes}; nrgbd=${nrgb:-$BENCH/nrgbd}
log "7-Scenes root: $seven"; log "NRGBD root: $nrgbd"

mkdir -p "$POOL" "$STAGING"
for root in "$seven" "$nrgbd"; do
  log "converting $root"
  $PY tools/prepare_benchmark.py --root "$root" --out /tmp/bench_models \
    || { log "conversion failed for $root"; continue; }
done
log "staging every converted sequence at 256x192"
$PY tools/prepare_colmap.py --root /tmp/bench_models --staging "$STAGING" --out /tmp/bench_dummy \
  --height 192 --width 256 --frames "$FRAMES" --dataset bench

log "exporting overlapping windows"
$PY - "$STAGING" "$STRIDE" <<'PY'
import json, sys
from l4d.data.scannet import export_manifest
staging, stride = sys.argv[1], int(sys.argv[2])
for length in (21, 13):
    print(length, json.dumps(export_manifest(staging, f"data/bench{length}", clip_length=length,
          clip_stride=stride if stride < length else length, size=(192, 256), dataset=f"bench{length}")), flush=True)
PY
cat data/bench21/manifest.jsonl data/bench13/manifest.jsonl > "$POOL/manifest.jsonl"
log "pool before QC: $(wc -l < "$POOL/manifest.jsonl") clips"

# Hold out what the datasets themselves hold out. 7-Scenes ships TrainSplit.txt/TestSplit.txt naming
# sequences by digits; NRGBD keeps its split in the directory name. Nothing here invents a partition.
$PY - "$POOL/manifest.jsonl" "$BENCH" <<'PY'
import json, os, re, sys
manifest, bench = sys.argv[1], sys.argv[2]
official = {}
for dirpath, _, filenames in os.walk(bench):
    for name in filenames:
        if re.match(r"(Train|Test)Split\.txt$", name):
            kind = "train" if name.startswith("Train") else "test"
            for line in open(os.path.join(dirpath, name), encoding="utf-8", errors="ignore"):
                digits = re.sub(r"\D", "", line).lstrip("0")
                if digits:
                    official[digits] = kind
rows = [json.loads(line) for line in open(manifest) if line.strip()]
SEQUENCE = re.compile(r"seq[-_]?0*(\d+)", re.I)
for row in rows:
    parts = re.split(r"[\\/]", row["video"])
    if any(part.lower().startswith("test") for part in parts):
        row["split"] = "test"
        continue
    match = SEQUENCE.search(row["clip_id"])
    row["split"] = official.get(match.group(1), "train") if match else "train"
with open(manifest, "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
counts = {kind: sum(1 for row in rows if row["split"] == kind) for kind in ("train", "test")}
print("split assignment:", counts, "of", len(rows), flush=True)
PY

sed -e "s#^manifest:.*#manifest: $POOL/manifest.jsonl#" -e "s#^root:.*#root: $POOL#" \
    -e "s#^latent_cache:.*#latent_cache: $POOL/latents#" -e "s#^datasets:.*#datasets: [bench21, bench13]#" \
    configs/data/seven.yaml > "$DATA"
sed -e "s/^split: train/split: test/" "$DATA" > "$TESTDATA"

log "QC gate (this walks every clip)"
$PY tools/check_dataset.py --data "$DATA" --threshold 0.05 --emit-manifest "$POOL/manifest.jsonl" | tail -4
log "pool after QC: $(wc -l < "$POOL/manifest.jsonl") clips"

log "latent cache"
wait_gpu "the queued training runs"
$PY tools/precompute_latents.py --data "$DATA" --model "$MODEL" --device cuda || log "encoding on the fly instead"

log "training: paper scale, 3 stages x $STEPS steps"
$PY tools/train.py --model "$MODEL" --data "$DATA" --device cuda --steps "$STEPS" --output "$OUT"

log "held-out scoring, in the paper's units"
wait_gpu "the staged training"
$PY tools/eval_gt.py --model "$MODEL" --data "$TESTDATA" --benchmark 7scenes --device cuda \
  --variants Full --checkpoint "$OUT/stage3_lora.pt" --out "$OUT/table3_trained.json"
$PY tools/eval_gt.py --model "$MODEL" --data "$TESTDATA" --benchmark 7scenes --device cuda \
  --variants Full --out "$OUT/table3_control.json"
$PY tools/eval_recon.py --model "$MODEL" --data "$TESTDATA" --device cuda \
  --checkpoint "$OUT/stage3_lora.pt" --out "$OUT/clouds_test"
log "BENCHMARK_TRAINING_DONE"
