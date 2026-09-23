#!/usr/bin/env bash
# Offline end-to-end verification: no weights, no downloads, CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
DEV=${1:-cpu}
echo "=== 1/5 structure + math checks ==="; python3 tests/test_reproduction.py
echo "=== 2/5 staged training (alignment -> +heads -> +LoRA) ==="
python3 tools/train.py --device "$DEV" --steps 3 --output runs/smoke
echo "=== 3/5 inference ==="
python3 tools/infer.py --device "$DEV" --checkpoint runs/smoke/stage3_lora.pt --out runs/infer
echo "=== 4/5 Table 3 ablation variants ==="
python3 tools/eval_gt.py --device "$DEV" --clips 1
echo "=== 5/5 Table 1 metric pipeline + Fig. 6 residual probe ==="
python3 tools/eval_generation.py --device "$DEV" --synthetic 2 --out runs/eval/text4d200.json
python3 tools/residual_sensitivity.py --device "$DEV" --rho 0.2 0.6 1.0 --out runs/eval/residual.json
echo "smoke test complete"
