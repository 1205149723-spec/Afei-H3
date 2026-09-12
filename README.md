# HailuoH3 8775 - AutoDL Linux

Linux/AutoDL publication of the HailuoH3 8775 local video generator.

## Runtime target

- Ubuntu Linux / x86_64 (the saved AutoDL image is the authoritative runtime)
- NVIDIA GPU + compatible driver
- Python 3.12
- PyTorch 2.11.0 + CUDA 13.0
- AutoDL web port: `6006`

## Quick start on AutoDL

```bash
git clone https://github.com/1205149723-spec/Afei-H3.git
cd Afei-H3
bash install_autodl.sh
./.venv/bin/python scripts/prepare_models.py --mode all
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

Large model weights are not committed to Git or baked into the saved environment image. `scripts/prepare_models.py` first reuses AutoDL shared-model mounts (`/.autodl-model/data` and `/.autodl`) and only downloads missing files into external model storage before symlinking them under `models/`. Use `--mode t2v` for the minimal T2V validation set or `--mode all` for every supported mode.

## AutoDL publishing

AutoDL Art uses a GitHub code repository together with a saved AutoDL image/runtime. Model weights should be attached separately through AutoDL public models rather than committed to this repository.
