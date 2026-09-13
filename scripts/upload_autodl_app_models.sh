#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CGTOOL="${CGTOOL:-$ROOT/tools/cgtool}"
STORE="${H3_MODEL_STORE:-/root/autodl-tmp/Afei-H3-models}"
TOKEN="${AUTODL_ART_UPLOAD_TOKEN:-}"

if [[ -z "$TOKEN" ]]; then
  echo "ERROR: set AUTODL_ART_UPLOAD_TOKEN to the 24-hour AutoDL Art model-upload token." >&2
  exit 2
fi
if [[ ! -x "$CGTOOL" ]]; then
  echo "ERROR: cgtool is missing or not executable: $CGTOOL" >&2
  exit 2
fi

upload_one() {
  local path="$1"
  local repo="$2"
  local name
  name="$(basename "$path")"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required model file is missing: $path" >&2
    exit 3
  fi
  echo "UPLOAD: $name"
  "$CGTOOL" upload model \
    --token="$TOKEN" \
    --name="$name" \
    --model_repository="$repo" \
    --description="Afei H3 8775 application model" \
    "$path"
}

upload_one "$STORE/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" "Comfy-Org/MiniMax-H3"
upload_one "$STORE/minimax_h3_video_vae_fp16.safetensors" "Comfy-Org/MiniMax-H3"
upload_one "$STORE/minimax_h3_audio_vae_fp32.safetensors" "Comfy-Org/MiniMax-H3"
upload_one "$STORE/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors" "Kijai/MiniMax-H3-experimental"
upload_one "$STORE/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors" "Kijai/MiniMax-H3-experimental"
upload_one "$STORE/latent_upscaler/minimax_h3_latent_upscaler_3d_fp16.safetensors" "LBH-123-AI/Minimax_h3_latent_Upscaler"

echo "OK: submitted all six Afei H3 application model files to AutoDL Art."
