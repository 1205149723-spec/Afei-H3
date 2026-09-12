# HailuoH3 8775 - AutoDL Linux

Linux/AutoDL publication of the HailuoH3 8775 local video generator.

## Runtime target

- Ubuntu 24.04 / x86_64
- NVIDIA GPU + compatible driver
- Python 3.12
- PyTorch 2.11.0 + CUDA 13.0
- AutoDL web port: `6006`

## Quick start on AutoDL

```bash
git clone https://github.com/1205149723-spec/HailuoH3-8775-AutoDL.git
cd HailuoH3-8775-AutoDL
bash install_autodl.sh
# Put/symlink the required model files under ./models first.
bash start_autodl.sh
```

The service listens on `0.0.0.0:6006` by default.

## NVIDIA compatibility

The requested Kijai/Sage fast path is used only when its locked runtime self-check succeeds. If the GPU/runtime is not compatible, H3 automatically uses ComfyUI official-native attention instead of failing before sampling.

## Stage-2 latent upscaler

The latent-upscaler vendor source is verified by SHA256 and does not require `git.exe`/Git at runtime.

## Linux scope

Core H3 T2V/I2V/first-last/R2V generation is the Linux target. The Windows-only NVIDIA VFX / SwiftVR post-processing workers are disabled in this edition.

## Models

Large model weights are not committed to Git. See `models/README.md`.

## AutoDL publishing

AutoDL Art uses a GitHub code repository together with a saved AutoDL image/runtime. Model weights should be attached separately through AutoDL public models rather than committed to this repository.
