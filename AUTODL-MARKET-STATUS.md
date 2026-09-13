# AutoDL Art market release status

## Release architecture

- The saved system image contains the validated Linux runtime and H3 application code, not the large model payloads.
- Market startup is `bash /root/start.sh` -> `start_autodl.sh` -> `prepare_models.py --source shared-only` -> H3 service on port `6006`.
- Required models must resolve from AutoDL mounted shared storage under `/.autodl-model/data` or `/.autodl` and are linked into `/root/Afei-H3/models`.
- Market mode never falls back to Hugging Face downloads. Development mode keeps the resumable data-disk download path.
- Torch/Hugging Face/Triton caches and process temporary files prefer `/root/autodl-tmp/Afei-H3-runtime` to minimize system-disk growth.

## Required model set

| Key | Filename | Expected shared repository | Status before authenticated AutoDL search |
| --- | --- | --- | --- |
| text | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `Comfy-Org/MiniMax-H3` | repository presence confirmed by an existing published AutoDL app; exact instance path still needs account search |
| video_vae | `minimax_h3_video_vae_fp16.safetensors` | `Comfy-Org/MiniMax-H3` | repository presence confirmed by an existing published AutoDL app; exact instance path still needs account search |
| audio_vae | `minimax_h3_audio_vae_fp32.safetensors` | `Comfy-Org/MiniMax-H3` | repository presence confirmed by an existing published AutoDL app; exact instance path still needs account search |
| fl2va | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` | `Kijai/MiniMax-H3-experimental` | public/shared availability not yet proven |
| ref2va | `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` | `Kijai/MiniMax-H3-experimental` | public/shared availability not yet proven |
| upscaler | `minimax_h3_latent_upscaler_3d_fp16.safetensors` | `LBH-123-AI/Minimax_h3_latent_Upscaler` | public/shared availability not yet proven |

`autodl-public-models.json` contains repository-style candidates. Market startup also scans the application-mounted `/.autodl` tree by exact filename, so a hashed `instance_path` does not need to be known in advance as long as the correct public model file is associated with the application version.

## AutoDL public-model search evidence

The current AutoDL Art frontend calls:

`POST /api/v1/application/model/file/search`

with fields `page_index`, `page_size`, `file_name`, and `model_repository`, and displays the returned `instance_path` for the symlink command. The endpoint requires an authenticated AutoDL Art session; an anonymous request returns `AuthorizeFailed`, so exact per-file paths cannot be truthfully filled from a logged-out client.

## Publication gates

1. In the AutoDL Art application creator, associate the six exact required model files with the application version. The runtime resolves either repository-style mounts or hashed `/.autodl` mounts automatically.
2. On the AutoDL release instance, update `/root/Afei-H3` to this revision and run `python scripts/prepare_models.py --mode all --source shared-only`.
3. Run `python scripts/verify_market_release.py`; all six models must be symlinks resolving under `/.autodl*` and size checks must pass.
4. Start with `bash /root/start.sh`, verify `/api/health` on port `6006`, then perform one real H3 generation from a clean cloned instance with no model download.
5. Only after those gates pass, save the AutoDL image and submit/update the AutoDL Art application for review. Saving/publishing is intentionally not automated without explicit user approval because it changes external account state and may incur charges.
