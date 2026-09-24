#!/bin/bash
# 4RC weights via hf-mirror: huggingface.co itself is unreachable from this machine.
# The authoritative size is the x-linked-size header; the redirect's own content-length is a 1 KB stub.
D=/mnt/data/pixels-weights/4rc
U=https://hf-mirror.com/Luo-Yihang/4RC/resolve/main
mkdir -p "$D"
for f in model.safetensors README.md LICENSE; do
  headers=$(curl -sIL -m 60 "$U/$f" | tr -d '\r')
  want=$(printf '%s\n' "$headers" | awk 'BEGIN{IGNORECASE=1}/^x-linked-size:/{print $2; exit}')
  case "${want:-}" in ''|*[!0-9]*) want=$(printf '%s\n' "$headers" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{print $2}' | tail -1) ;; esac
  case "${want:-}" in ''|*[!0-9]*) want=0 ;; esac
  echo "$f size=${want}"
  for i in $(seq 1 600); do
    have=$(stat -c %s "$D/$f" 2>/dev/null || echo 0)
    if [ "$want" -gt 100 ] && [ "$have" -ge "$want" ]; then echo "DONE $f $have/$want"; break; fi
    curl -sL -C - --max-time 400 -o "$D/$f" "$U/$f"
    echo "$(date +%H:%M:%S) $f $(stat -c %s "$D/$f" 2>/dev/null || echo 0)/$want"
    sleep 2
  done
done
