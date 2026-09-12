#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source .venv/bin/activate
export H3_PACKAGE_ROOT="$ROOT"
export H3_HOST="${H3_HOST:-0.0.0.0}"
export H3_PORT="${H3_PORT:-6006}"
export H3_MODEL_QUANT="${H3_MODEL_QUANT:-w4a8}"
export H3_SAGE_PRIVATE_ROOT="$ROOT/runtime/python_packages/sageattention-2.2.0-cu130-torch211-py312-linux"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/app:$ROOT/runtime/ComfyUI:$ROOT/runtime/python_packages:$H3_SAGE_PRIVATE_ROOT"
export TORCH_HOME="$ROOT/cache/torch"
export HF_HOME="$ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/cache/transformers"
export TRITON_HOME="$ROOT/cache/triton_home"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
mkdir -p input output temp cache models
exec python app/server.py
