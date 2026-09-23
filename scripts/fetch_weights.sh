#!/usr/bin/env bash
# Real-run asset download (needs network + Hugging Face access for gated Wan weights).
set -euo pipefail
cd "$(dirname "$0")/.."
TOKEN=${1:?usage: fetch_weights.sh <hf-token>}
export HUGGING_FACE_HUB_TOKEN=$TOKEN
mkdir -p weights
dl() { huggingface-cli download "$1" ${3:+--include "$3"} --local-dir "weights/$2"; }
dl Wan-AI/Wan2.1-T2V-1.3B   wan2.1-t2v-1.3b "vae/*"
dl Wan-AI/Wan2.1-T2V-14B    wan2.1-t2v-14b   "vae/*"
dl Wan-AI/Wan2.2-I2V-A14B   wan2.2-i2v-a14b  "vae/*"
dl facebook/dinov2-base     dinov2-base
dl openai/clip-vit-large-patch14 clip-vit-large-patch14
echo
echo "4RC weights are required to initialise the 31-block refinement hierarchy and the heads."
echo "Place them at weights/4rc/ and set pretrained_init.checkpoint in configs/model/l4ar_paper.yaml."
echo "CogVideoX-5B is only needed for cascade baselines (it is outside the shared Wan VAE family)."
