# HailuoH3 8775 - AutoDL Linux

Linux/AutoDL publication of the HailuoH3 8775 local video generator.

## Runtime target

- Ubuntu Linux / x86_64 (the saved AutoDL image is the authoritative runtime)
- NVIDIA GPU + compatible driver
- Python 3.12
- PyTorch 2.11.0 + CUDA 13.0
- AutoDL web port: `6006`

## Quick start on AutoDL

For the published application: create an instance and wait for startup. End users can either open the exposed H3 service directly, or open JupyterLab and click **“阿飞 H3 工作台”**. No AutoDL-SSH-Tools install, terminal command, Git clone, environment install, or model download is required.

The service listens on `0.0.0.0:6006` by default. `/root/start.sh` is the one-command fallback launcher. The saved image also includes a JupyterLab launcher named `阿飞 H3 工作台`; it opens H3 through Jupyter's own web proxy, so end users do not need AutoDL-SSH-Tools.

## NVIDIA compatibility

The requested Kijai/Sage fast path is used only when its locked runtime self-check succeeds. If the GPU/runtime is not compatible, H3 automatically uses ComfyUI official-native attention instead of failing before sampling.

## Stage-2 latent upscaler

The latent-upscaler vendor source is verified by SHA256 and does not require `git.exe`/Git at runtime.

## Linux scope

Core H3 T2V/I2V/first-last/R2V generation is the Linux target. The Windows-only NVIDIA VFX / SwiftVR post-processing workers are disabled in this edition.

## Models

Large model weights are not committed to Git and are not baked into the market system image. Market startup runs `scripts/prepare_models.py --source shared-only`, resolves the six required files from AutoDL's mounted public/shared model storage (`/.autodl-model/data` or `/.autodl`), validates identity/size, and creates project-local symlinks under `/root/Afei-H3/models`.

`autodl-public-models.json` contains the known repository-style shared paths and also accepts exact `/.autodl/<hash>` `instance_path` values copied from AutoDL Art's authenticated public-model search. If a required model is not mounted, market startup fails with the exact missing filename and **does not download tens of gigabytes from Hugging Face**.

For development only, set `H3_DEPLOYMENT_MODE=development`; the existing resumable download fallback remains available and stores downloaded weights under `/root/autodl-tmp/Afei-H3-runtime/model-downloads` by default.

The runtime cache/Triton/Hugging Face temporary data is also redirected to `/root/autodl-tmp/Afei-H3-runtime` when the AutoDL data disk is mounted, reducing system-disk growth. Generated H3 output remains under the package `output/` directory.

## AutoDL publishing

AutoDL Art uses this GitHub code repository together with a saved AutoDL image/runtime. The release image contains the validated Linux runtime and application code; model payloads are supplied by AutoDL's shared-model mount and linked at each boot. Before saving/publishing an image, run `python scripts/verify_market_release.py`. A market release is valid only when all required model links resolve to `/.autodl*`, port 6006 starts successfully, and a clean-instance real generation has been verified.
