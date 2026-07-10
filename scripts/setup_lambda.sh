#!/usr/bin/env bash
# One-shot environment setup on a Lambda (Linux, H100) box.
#   bash scripts/setup_lambda.sh
set -euo pipefail

PY=${PY:-python3.12}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}   # H100 = sm_90; cu128 covers it

command -v "$PY" >/dev/null || PY=python3
echo "using: $($PY --version)"

$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# torch matched to the local dev box (2.11.0 + CUDA 12.8). If the driver is older,
# switch TORCH_INDEX to cu124/cu121.
pip install "torch==2.11.0" --index-url "$TORCH_INDEX"
pip install -r requirements.txt

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU",
      torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "")
PY
echo "setup done. Next: bash scripts/run_lambda.sh"
