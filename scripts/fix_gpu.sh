#!/usr/bin/env bash
# Make PyTorch see the GPU on this machine.
#
# Diagnosis: the NVIDIA driver is 535.x, which exposes CUDA 12.2. The system and conda PyTorch builds
# are 2.12.1+cu130 (needs a >=580 driver), so torch.cuda.is_available() is False even though an RTX 3090
# is present. Two ways out; only the first needs no root:
#
#   1) install a cu12x PyTorch build into a dedicated venv (this script)
#   2) upgrade the driver: sudo apt install -y nvidia-driver-580 && reboot   (touches every other env)
#
# pypi.org runs ~5 KB/s here while the Aliyun mirror runs MB/s, so the index is pinned.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_DIR=${ENV_DIR:-$HOME/Desktop/pixels_env}
PIP_INDEX=${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
TORCH_INDEX=${TORCH_INDEX:-https://mirrors.aliyun.com/pypi/simple/}
TORCH_SPEC=${TORCH_SPEC:-"torch>=2.4,<2.7"}   # cu124 wheels work on the 535 driver

driver_cuda=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "driver $driver_cuda; gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if [ ! -x "$ENV_DIR/bin/python" ]; then
  echo "creating venv at $ENV_DIR"
  python3 -m venv --system-site-packages=false "$ENV_DIR" 2>/dev/null || python3 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install -q -i "$PIP_INDEX" --upgrade pip
"$ENV_DIR/bin/python" -m pip install -q -i "$TORCH_INDEX" "$TORCH_SPEC"
"$ENV_DIR/bin/python" -m pip install -q -i "$PIP_INDEX" numpy pyyaml pillow tqdm diffusers accelerate safetensors

"$ENV_DIR/bin/python" - <<'PY'
import time, torch
assert torch.cuda.is_available(), "still no GPU - driver and wheel are incompatible"
a = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize(); t = time.time(); b = a @ a; torch.cuda.synchronize()
print(f"torch {torch.__version__} / cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
print(f"4k matmul ok in {(time.time()-t)*1000:.1f} ms, peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
PY

echo
echo "use it for every Pixels command:"
echo "  export PATH=$ENV_DIR/bin:\$PATH"
echo "existing installs that keep working: /home/zbf/Desktop/dl_env (torch 2.6.0+cu124, verified on GPU)"
