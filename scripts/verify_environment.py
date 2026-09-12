from __future__ import annotations
import os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_runtime", torch.version.cuda)
    print("cuda_available", torch.cuda.is_available())
    if not torch.cuda.is_available(): errors.append("CUDA is unavailable")
    else: print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
except Exception as exc:
    errors.append(f"torch import failed: {type(exc).__name__}: {exc}")

commands = ("nvidia-smi",) if os.environ.get("H3_SKIP_FFMPEG_CHECK") == "1" else ("ffmpeg", "ffprobe", "nvidia-smi")
for command in commands:
    print(command, shutil.which(command))
    if shutil.which(command) is None: errors.append(f"missing command: {command}")

for rel in ("app/server.py", "runtime/ComfyUI/comfy/model_management.py", "vendor/Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py"):
    if not (ROOT / rel).is_file(): errors.append(f"missing runtime source: {rel}")

if errors:
    print("FAILED")
    for item in errors: print("-", item)
    raise SystemExit(1)
print("OK: base Linux runtime verified")
