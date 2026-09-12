#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ "${H3_SKIP_APT:-0}" != "1" ]]; then
  apt-get update
  apt-get install -y ffmpeg gcc g++ ninja-build git curl ca-certificates
fi

CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "$CONDA_BIN" && -x /root/miniconda3/bin/conda ]]; then
  CONDA_BIN=/root/miniconda3/bin/conda
fi
if [[ -z "$CONDA_BIN" ]]; then
  CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" ]]; then
  BOOTSTRAP="$ROOT/cache/miniconda.sh"
  mkdir -p "$ROOT/cache"
  curl -fL --retry 3 --retry-delay 2 \
    https://repo.anaconda.com/miniconda/Miniconda3-py312_25.5.1-1-Linux-x86_64.sh \
    -o "$BOOTSTRAP"
  bash "$BOOTSTRAP" -b -p "$ROOT/.bootstrap-conda"
  CONDA_BIN="$ROOT/.bootstrap-conda/bin/conda"
fi

rm -rf "$ROOT/.venv"
"$CONDA_BIN" create -y -p "$ROOT/.venv" python=3.12 pip
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --no-cache-dir --upgrade pip setuptools wheel
"$PY" -m pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
"$PY" -m pip install --no-cache-dir -r requirements-linux.txt

SAGE_TARGET="$ROOT/runtime/python_packages/sageattention-2.2.0-cu130-torch211-py312-linux"
SAGE_WHEEL="$ROOT/cache/sageattention-2.2.0+cu130torch2.11-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
SAGE_URL="https://github.com/Comfy-Org/wheels/releases/download/sageattention-latest/sageattention-2.2.0%2Bcu130torch2.11-cp312-cp312-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
SAGE_SHA256="988a5b510078dfef67fa0ca517321afb659a62a3ace12d7f52f3b831d2ddb58f"
rm -rf "$SAGE_TARGET"
mkdir -p "$SAGE_TARGET" "$ROOT/cache"
GLIBC_VERSION="$(ldd --version | head -n1 | grep -oE '[0-9]+\.[0-9]+' | tail -n1 || true)"
if "$PY" - "$GLIBC_VERSION" <<'PY'
import sys
parts = tuple(int(x) for x in (sys.argv[1] or "0.0").split(".")[:2])
raise SystemExit(0 if parts >= (2, 34) else 1)
PY
then
  curl -fL --retry 3 --retry-delay 2 "$SAGE_URL" -o "$SAGE_WHEEL"
  echo "$SAGE_SHA256  $SAGE_WHEEL" | sha256sum -c -
  "$PY" -m pip install --no-deps --target "$SAGE_TARGET" "$SAGE_WHEEL"
  rm -f "$SAGE_WHEEL"
else
  echo "INFO: glibc ${GLIBC_VERSION:-unknown} is below 2.34; SageAttention wheel skipped. H3 will use official-native attention."
fi

"$PY" scripts/verify_environment.py
if [[ "${H3_INSTALL_ROOT_STARTER:-1}" == "1" && -d /root ]]; then
  install -m 0755 "$ROOT/start.sh" /root/start.sh
fi

# JupyterLab one-click fallback. Register via the official
# jupyter_serverproxy_servers entry-point mechanism so AutoDL's explicit
# /init/jupyter/jupyter_config.py cannot hide the launcher configuration.
if [[ -x /root/miniconda3/bin/python ]]; then
  /root/miniconda3/bin/python -m pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple \
    jupyter-server-proxy==4.5.0
  /root/miniconda3/bin/python -m pip install --no-cache-dir "$ROOT/jupyter_launcher"
fi
echo "OK: HailuoH3 AutoDL Linux environment is ready."
