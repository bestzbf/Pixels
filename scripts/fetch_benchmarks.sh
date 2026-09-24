#!/usr/bin/env bash
# Fetch the RGB-D benchmarks the paper's Table 3 is actually measured on, so the ablation can be scored
# in the paper's own units (Acc/Comp in cm, NC in degrees) instead of a local pool with no reference row.
#
#   bash scripts/fetch_benchmarks.sh [dest-dir]        # default /mnt/data/pixels-benchmarks
#   ONLY=stairs  bash scripts/fetch_benchmarks.sh      # one stage: stairs | nrgbd | seven
#
# Everything here is a plain public download through hf-mirror.com (huggingface.co is unreachable from this
# box); neither dataset is gated, but both are released for non-commercial research use - keep that in mind
# before publishing numbers derived from them. Downloads resume where they stopped and each stage is skipped
# when its output is already present, so this is safe to re-run.
set -uo pipefail

DEST=${1:-/mnt/data/pixels-benchmarks}
M=${HF_ENDPOINT:-https://hf-mirror.com}
ONLY=${ONLY:-all}
mkdir -p "$DEST"
export PYTHONUNBUFFERED=1

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }

pull() {  # pull <url> <dest> [expected-bytes]
  local url=$1 out=$2 want=${3:-0}
  [ -s "$out" ] && { [ "$want" = 0 ] || [ "$(stat -c%s "$out")" -ge "$want" ] && { log "have   $(basename "$out")"; return 0; }; }
  local attempt
  for attempt in $(seq 1 30); do
    if curl -fL -C - --retry 5 --retry-delay 5 --speed-limit 20000 --speed-time 180 -o "$out.part" "$url"; then
      local got=$(stat -c%s "$out.part" 2>/dev/null || echo 0)
      if [ "$want" = 0 ] || [ "$got" -ge "$want" ]; then mv "$out.part" "$out"; log "done   $(basename "$out") ($got B)"; return 0; fi
      log "short  $(basename "$out") $got/$want B, resuming"
    fi
    log "retry  $attempt for $(basename "$out")"
    sleep $((attempt * 5))
  done
  log "FAILED $url"; return 1
}

stage_stairs() {  # one complete 7-Scenes scene: the fastest way to a real, paper-comparable number
  pull "$M/datasets/bluenot/7scenesstair/resolve/main/stairs.zip" "$DEST/stairs.zip" 1496412431 || return 1
  if [ ! -d "$DEST/7scenes" ]; then
    mkdir -p "$DEST/7scenes" && unzip -q -o "$DEST/stairs.zip" -d "$DEST/7scenes" && log "unzipped stairs.zip"
  fi
  find "$DEST/7scenes" -maxdepth 2 | head -8
}

stage_nrgbd() {   # NRGBD (Lin et al.), the paper's second Table 3 column
  pull "$M/datasets/Aazeus/nrgbd/resolve/main/neural_rgbd_data.zip" "$DEST/neural_rgbd_data.zip" 7785287298 || return 1
  if [ ! -d "$DEST/nrgbd" ]; then mkdir -p "$DEST/nrgbd" && unzip -q -o "$DEST/neural_rgbd_data.zip" -d "$DEST/nrgbd" && log "unzipped neural_rgbd_data.zip"; fi
  find "$DEST/nrgbd" -maxdepth 2 | head -12
}

stage_seven() {   # all seven scenes, 20.7 GB, delivered as four 5 GiB parts of one tar.zst
  local part
  for part in 00 01 02 03; do
    pull "$M/datasets/kronecker0122/7-scenes/resolve/main/7-scenes.tar.zst.part-$part" \
         "$DEST/7-scenes.tar.zst.part-$part" 5368709120 || return 1
  done
  if [ ! -d "$DEST/7-scenes/heads" ]; then
    log "extracting 7-scenes.tar.zst (~33 GB uncompressed)"
    cat "$DEST"/7-scenes.tar.zst.part-* | tar --zstd -xf - -C "$DEST" && log "extracted"
  fi
  find "$DEST" -maxdepth 2 -type d -iname "*scene*" | head -12
}

case "$ONLY" in
  stairs)  stage_stairs ;;
  nrgbd)   stage_nrgbd ;;
  seven)   stage_seven ;;
  all)     stage_stairs && stage_nrgbd && stage_seven ;;
esac
log "BENCHMARKS_READY in $DEST"
