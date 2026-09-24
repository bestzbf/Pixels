#!/usr/bin/env bash
# One command from a ScanNet directory to a trained, measured L4AR checkpoint.
#
#   bash scripts/train_on_scannet.sh /path/to/scannet/dense_data [clips]
#
# Stages: inventory -> export clips with metric 4D GT -> frozen-VAE latent cache -> staged training
# -> reconstruction metrics against the untrained baseline. Every stage is idempotent and resumable.
# The same chain is exercised offline by the ScanNet-format fixture in tests/test_scannet.py.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT=${1:?usage: train_on_scannet.sh <scannet-root> [clips]}
CLIPS=${2:-1143}
PY=${PY:-/home/zbf/Desktop/dl_env/bin/python}     # torch build that actually sees the GPU here
DATA=${DATA:-configs/data/scannet.local.yaml}   # generated from the tracked template, never edited in place
MODEL=${MODEL:-configs/model/l4ar_paper.yaml}
DEVICE=${DEVICE:-cuda}
OUT=${OUT:-runs/scannet}
STEPS=${STEPS:-2000}

mkdir -p data/scannet
# Generate a local dataset config from the tracked template so the repository stays clean.
$PY - "$ROOT" "$CLIPS" "$DATA" <<'PY'
import os, re, sys
root, clips, path = sys.argv[1], sys.argv[2], sys.argv[3]
template = "configs/data/scannet.yaml"
text = open(path, encoding="utf-8").read() if os.path.exists(path) else open(template, encoding="utf-8").read()
text = re.sub(r"^scannet_root:.*$", f"scannet_root: {root}", text, flags=re.M)
text = re.sub(r"^target_clips:.*$", f"target_clips: {clips}", text, flags=re.M)
open(path, "w", encoding="utf-8").write(text)
print(f"wrote {path}: scannet_root={root} target_clips={clips}")
PY

echo
echo "[1/5] scene inventory (which layouts are readable, how many frames)"
$PY tools/prepare_scannet.py --root "$ROOT" --report-only | tail -20

echo
echo "[2/5] export clips (frames + metric GT npz + manifest.jsonl)"
$PY tools/prepare_scannet.py --root "$ROOT" --out data/scannet --clip-length 21 --clip-stride 21 \
  --height 192 --width 256 --clips "$CLIPS"

echo
echo "[3/5] precompute frozen Wan-VAE latents (z^obs = mu(E_v(V)); the VAE stays frozen)"
$PY tools/precompute_latents.py --data "$DATA" --model "$MODEL" --device "$DEVICE" || \
  echo "latents will be encoded on the fly instead (slower, same result)"

echo
echo "[4/5] stage-wise training (alignment -> +heads -> +rank-16 LoRA)"
$PY tools/train.py --model "$MODEL" --data "$DATA" --device "$DEVICE" --steps "$STEPS" --output "$OUT"

echo
echo "[5/5] reconstruction metrics vs the untrained baseline"
$PY tools/eval_recon.py --model "$MODEL" --data "$DATA" --checkpoint "$OUT/stage3_lora.pt" --out "$OUT/clouds"
echo
echo "untrained reference for the same pool:"
$PY tools/eval_recon.py --model "$MODEL" --data "$DATA" --from-scratch
