#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: Linux runtime is not installed. Run bash install_autodl.sh first." >&2
  exit 1
fi
export H3_PACKAGE_ROOT="$ROOT"
export H3_HOST="${H3_HOST:-0.0.0.0}"
export H3_PORT="${H3_PORT:-6006}"
export H3_MODEL_QUANT="${H3_MODEL_QUANT:-w4a8}"
export H3_DEPLOYMENT_MODE="${H3_DEPLOYMENT_MODE:-market}"
export H3_SAGE_PRIVATE_ROOT="$ROOT/runtime/python_packages/sageattention-2.2.0-cu130-torch211-py312-linux"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/app:$ROOT/runtime/ComfyUI:$ROOT/runtime/python_packages:$H3_SAGE_PRIVATE_ROOT"
DATA_ROOT="${H3_DATA_ROOT:-/root/autodl-tmp/Afei-H3-runtime}"
if [[ ! -d /root/autodl-tmp ]]; then
  DATA_ROOT="$ROOT/cache/runtime-data"
fi
mkdir -p "$DATA_ROOT/cache/torch" "$DATA_ROOT/cache/huggingface" "$DATA_ROOT/cache/transformers" \
  "$DATA_ROOT/cache/triton_home" "$DATA_ROOT/cache/triton" "$DATA_ROOT/temp" "$DATA_ROOT/model-downloads"
export TMPDIR="$DATA_ROOT/temp"
export TORCH_HOME="$DATA_ROOT/cache/torch"
export HF_HOME="$DATA_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$DATA_ROOT/cache/transformers"
export TRITON_HOME="$DATA_ROOT/cache/triton_home"
export TRITON_CACHE_DIR="$DATA_ROOT/cache/triton"
export H3_MODEL_STORE="$DATA_ROOT/model-downloads"
mkdir -p input output temp cache models

if [[ "$H3_DEPLOYMENT_MODE" == "market" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  "$PY" scripts/prepare_models.py --mode all --source shared-only
elif [[ "$H3_DEPLOYMENT_MODE" == "development" ]]; then
  "$PY" scripts/prepare_models.py --mode all --source auto
else
  echo "ERROR: H3_DEPLOYMENT_MODE must be market or development, got: $H3_DEPLOYMENT_MODE" >&2
  exit 1
fi
exec "$PY" app/server.py
