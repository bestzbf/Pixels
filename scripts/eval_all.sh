#!/usr/bin/env bash
# Paper Tables 1/3 and Fig. 6 on trained weights.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${1:?usage: eval_all.sh <stage3_lora.pt>}
MODEL=${2:-configs/model/l4ar_paper.yaml}
for GEN in Wan2.1-T2V-14B Wan2.1-T2V-1.3B; do
  for BENCH in text4d200 i4d200; do
    python3 tools/build_benchmark.py --eval configs/eval/$BENCH.yaml --model "$MODEL" --generator "$GEN"
    python3 tools/eval_generation.py --eval configs/eval/$BENCH.yaml --model "$MODEL" \
      --checkpoint "$CKPT" --extractor dinov2 --out "runs/eval/$BENCH-$GEN.json"
  done
done
python3 tools/eval_gt.py --model "$MODEL" --checkpoint "$CKPT" --benchmark 7scenes --out runs/eval/table3-7scenes.json
python3 tools/eval_gt.py --model "$MODEL" --checkpoint "$CKPT" --benchmark nrgbd   --out runs/eval/table3-nrgbd.json
python3 tools/residual_sensitivity.py --model "$MODEL" --checkpoint "$CKPT" --out runs/eval/fig6.json
echo "gains vs matched cascades: python3 -c \"from l4d.eval.baselines import dino_f1_gains; print(dino_f1_gains(...))\""
