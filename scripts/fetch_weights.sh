#!/usr/bin/env bash
# Download the weights the framework needs for real runs (no synthetic stand-ins).
#
#   bash scripts/fetch_weights.sh [dest-dir]        # default: /mnt/data/pixels-weights
#
# huggingface.co is unreachable from some networks; the script therefore honours HF_ENDPOINT and
# defaults to the hf-mirror.com endpoint. Every source below is NOT gated, so no token is required
# (a token is only needed if you also pull the full DiT checkpoints for latent sampling).
set -euo pipefail
cd "$(dirname "$0")/.."

DEST=${1:-/mnt/data/pixels-weights}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
mkdir -p "$DEST"
command -v hf >/dev/null 2>&1 || command -v huggingface-cli >/dev/null 2>&1 \
  || { echo "需要 huggingface_hub>=1.x（提供 \`hf\` 命令）: pip install -U 'huggingface_hub[cli]'"; exit 1; }
dl() {  # dl <repo> <local-subdir> [extra hf args...]
  local repo=$1 sub=$2; shift 2
  echo ">>> $repo -> $DEST/$sub   (endpoint: $HF_ENDPOINT)"
  for attempt in 1 2 3 4 5; do
    if hf download "$repo" --local-dir "$DEST/$sub" "$@"; then return 0; fi
    echo "retry $attempt for $repo"; sleep $((attempt * 10))
  done
  echo "FAILED: $repo" >&2; return 1
}

# 1. Shared video VAE (the interface itself): frozen Wan2.1 VAE, diffusers layout -> AutoencoderKLWan
#    Same 507.6 MB weights are used by Wan2.1-T2V-1.3B/14B and Wan2.2-I2V-A14B (common-VAE family).
dl Wan-AI/Wan2.1-T2V-1.3B-Diffusers wan-vae --include "vae/*"

# 2. 4RC = "4D Reconstruction via Conditional Querying Anytime and Anywhere" (arXiv:2602.10094),
#    the pretrained 4D hierarchy that initialises the 31-block refinement network and both heads.
dl Luo-Yihang/4RC 4rc --include "model.safetensors" "README.md" "LICENSE"

# 3. Projection metrics backbones (Table 1): DINOv2 for global/patch similarity, CLIP for CLIP-I.
dl facebook/dinov2-base                        dinov2-base
dl openai/clip-vit-large-patch14               clip-vit-large-patch14

# 4. Optional, only for sampling generated latents (Text4D-200 / I4D-200) — ~17 GB per model:
#    dl Wan-AI/Wan2.1-T2V-1.3B-Diffusers wan2.1-t2v-1.3b --exclude "assets/*"
#    dl Wan-AI/Wan2.2-I2V-A14B-Diffusers  wan2.2-i2v-a14b --exclude "assets/*"
#    dl zai-org/CogVideoX-5b               cogvideox-5b        # cascade baselines only

cat <<EOF

下一步（把配置指向这些路径）：
  configs/model/l4ar_paper.yaml:
    pretrained_init.checkpoint: $DEST/4rc/model.safetensors
    vae.checkpoint_id:          $DEST/wan-vae        # subfolder: vae
  评测 encoder：
    L4D_DINOV2=$DEST/dinov2-base   L4D_CLIP=$DEST/clip-vit-large-patch14
EOF
