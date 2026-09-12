#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "${H3_SKIP_APT:-0}" != "1" ]]; then
  if ! command -v python3 >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
    apt-get update
    apt-get install -y python3 python3-venv python3-pip ffmpeg gcc g++ ninja-build git curl ca-certificates
  fi
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-linux.txt

SAGE_TARGET="$ROOT/runtime/python_packages/sageattention-2.2.0-cu130-torch211-py312-linux"
SAGE_WHEEL="$ROOT/cache/sageattention-2.2.0+cu130torch2.11-cp312.whl"
SAGE_URL="https://github.com/Comfy-Org/wheels/releases/download/sageattention-latest/sageattention-2.2.0%2Bcu130torch2.11-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
SAGE_SHA256="988a5b510078dfef67fa0ca517321afb659a62a3ace12d7f52f3b831d2ddb58f"
rm -rf "$SAGE_TARGET"
mkdir -p "$SAGE_TARGET" "$ROOT/cache"
curl -fL --retry 3 --retry-delay 2 "$SAGE_URL" -o "$SAGE_WHEEL"
echo "$SAGE_SHA256  $SAGE_WHEEL" | sha256sum -c -
python -m pip install --no-deps --target "$SAGE_TARGET" "$SAGE_WHEEL"
rm -f "$SAGE_WHEEL"

python scripts/verify_environment.py
echo "OK: HailuoH3 AutoDL Linux environment is ready."
