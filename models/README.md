# Required model files

The following files are required before generation:

- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_fp16.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`
- `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
- `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors`
- `latent_upscaler/minimax_h3_latent_upscaler_3d_fp16.safetensors`

Model weights are intentionally not stored in Git and must not be baked into the AutoDL market system image. `start_autodl.sh` creates these paths as symbolic links to AutoDL's mounted public/shared model storage on every market startup. Development mode may instead use the data-disk download store.
