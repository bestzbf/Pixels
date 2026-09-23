#!/usr/bin/env bash
# Weights from ModelScope: huggingface.co and hf-mirror.com both stall at 0 B/s on this network,
# while ModelScope sustains MB/s. Complements scripts/fetch_weights.sh (HF route).
#
#   bash scripts/fetch_weights_modelscope.sh [name...]     # names: wan-vae dinov2 clip
#
# Expected sizes come from the ModelScope tree API rather than HTTP headers, because HEAD replies
# carry no content-length and a missing file answers with a 145-byte JSON error page.
set -euo pipefail
DEST=${DEST:-/mnt/data/pixels-weights}
ENDPOINT=${ENDPOINT:-https://www.modelscope.cn}
PY=${PY:-python3}
mkdir -p "$DEST"

_size_map() { # repo-id -> "path<TAB>size" lines
  timeout 60 curl -s "$ENDPOINT/api/v1/models/$1/repo/files?Recursive=true" | "$PY" -c '
import json, sys
try:
    tree = json.load(sys.stdin)["Data"]["Files"]
except Exception:
    sys.exit(1)
for entry in tree:
    if entry.get("Type") == "blob":
        print("%s\t%s" % (entry["Path"], entry.get("Size", 0)))
' 2>/dev/null
}

get() { # get <local-subdir> <repo-id> <relative-path>...
  local sub=$1 repo=$2; shift 2
  local sizes; sizes=$(_size_map "$repo") || { echo "[fail] cannot read $repo tree" >&2; return 1; }
  for path in "$@"; do
    local want
    want=$(printf '%s\n' "$sizes" | awk -F'\t' -v p="$path" '$1==p{print $2; exit}')
    if [ -z "${want:-}" ]; then echo "[skip] $repo/$path not in the repo tree" >&2; continue; fi
    mkdir -p "$DEST/$sub/$(dirname "$path")"
    local url="$ENDPOINT/models/$repo/resolve/master/$path" have=0
    for _ in $(seq 1 200); do
      have=$(stat -c %s "$DEST/$sub/$path" 2>/dev/null || echo 0)
      if [ "$have" -ge "$want" ] && [ "$have" -gt 100 ]; then echo "[done] $sub/$path ($have B)"; break; fi
      curl -sL -C - --max-time 300 -o "$DEST/$sub/$path" "$url" || true
      echo "[get ] $sub/$path $(stat -c %s "$DEST/$sub/$path" 2>/dev/null || echo 0)/$want"
    done
  done
}

for name in ${*:-wan-vae dinov2 clip}; do
  case $name in
    wan-vae) get wan-vae Wan-AI/Wan2.1-T2V-1.3B-Diffusers vae/config.json vae/diffusion_pytorch_model.safetensors ;;
    dinov2)  get dinov2-base facebook/dinov2-base config.json preprocessor_config.json model.safetensors ;;
    clip)    get clip-vit-large-patch14 AI-ModelScope/clip-vit-large-patch14 \
               config.json preprocessor_config.json vocab.json merges.txt tokenizer.json \
               special_tokens_map.json tokenizer_config.json model.safetensors ;;
    *) echo "unknown weight set: $name (choose: wan-vae dinov2 clip)" >&2; exit 1 ;;
  esac
done
du -sh "$DEST"/* 2>/dev/null || true
