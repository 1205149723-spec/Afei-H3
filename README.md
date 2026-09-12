# HailuoH3 8775 - AutoDL Linux

Linux/AutoDL publication of the HailuoH3 8775 local video generator.

## Runtime target

- Ubuntu Linux / x86_64 (the saved AutoDL image is the authoritative runtime)
- NVIDIA GPU + compatible driver
- Python 3.12
- PyTorch 2.11.0 + CUDA 13.0
- AutoDL web port: `6006`

## Quick start on AutoDL

For the published image: create an instance from the Afei-H3 image, wait for startup, then open AutoDL Custom Service port `6006`. No Git clone, environment install, or model download is required for end users.

The service listens on `0.0.0.0:6006` by default. `/root/start.sh` is the one-command fallback launcher. The saved image also includes a JupyterLab launcher named `阿飞 H3 工作台`; it opens H3 through Jupyter's own web proxy, so end users do not need AutoDL-SSH-Tools.

## NVIDIA compatibility

The requested Kijai/Sage fast path is used only when its locked runtime self-check succeeds. If the GPU/runtime is not compatible, H3 automatically uses ComfyUI official-native attention instead of failing before sampling.

## Stage-2 latent upscaler

The latent-upscaler vendor source is verified by SHA256 and does not require `git.exe`/Git at runtime.

## Linux scope

Core H3 T2V/I2V/first-last/R2V generation is the Linux target. The Windows-only NVIDIA VFX / SwiftVR post-processing workers are disabled in this edition.

## Models

Large model weights are not committed to Git. The final published AutoDL image must contain the complete validated model set under `/root/Afei-H3/models`, so end users never need to download or prepare models.

## AutoDL publishing

AutoDL Art uses this GitHub code repository together with a saved AutoDL image/runtime. The release image is self-contained: runtime dependencies and the complete model set are baked into the image so users can open port 6006 and generate immediately.
