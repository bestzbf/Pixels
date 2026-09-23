#!/usr/bin/env bash
# Full three-stage training on real clips (GPU machine).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 tools/precompute_latents.py --data configs/data/clips_final.yaml   # frozen VAE -> z_obs cache
python3 tools/train.py \
  --model configs/model/l4ar_paper.yaml \
  --data  configs/data/clips_final.yaml \
  --train configs/train/staged.yaml
