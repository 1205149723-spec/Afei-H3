# AutoDL Art 发布卡

## 镜像

- 建议镜像名：`Afei-H3-8775-market-v1`
- 系统盘：当前 30 GiB 足够；实测约 9.5 GiB 已用，模型不进入系统盘
- 框架：PyTorch `2.11.0`
- CUDA：`13.0`
- CPU 架构：`x86_64`
- 芯片：NVIDIA
- 启动命令：`bash /root/start.sh`
- Web 服务端口：`6006`

## 应用

- 建议名称：`阿飞 H3 工作台`
- 建议版本：`1.0`
- 作者提示：`推荐 24GB 及以上 NVIDIA 显存；RTX 4090 24GB 已验证。Kijai/Sage 快速路径会按当前 GPU 实测自检，不兼容时自动回退官方原生 attention。`

## 必须关联的 6 个公共模型文件

1. `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
   - 来源：`Comfy-Org/MiniMax-H3`
   - 大小：`15687142551`
   - SHA256：`35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6`
2. `minimax_h3_video_vae_fp16.safetensors`
   - 来源：`Comfy-Org/MiniMax-H3`
   - 大小：`5207808496`
   - SHA256：`7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`
3. `minimax_h3_audio_vae_fp32.safetensors`
   - 来源：`Comfy-Org/MiniMax-H3`
   - 大小：`605254808`
   - SHA256：`8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48`
4. `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
   - 来源：`Kijai/MiniMax-H3-experimental`
   - 大小：`12540858008`
   - SHA256：`01aa7b92c007c599890461c325f9b7e3c96fb06c36f242f95b62f7f20e538dec`
5. `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors`
   - 来源：`Kijai/MiniMax-H3-experimental`
   - 大小：`11770657048`
   - SHA256：`de2c6c29c4ee702b45e48e40daae3834aeee58ab681c732d9152589a87c89910`
6. `minimax_h3_latent_upscaler_3d_fp16.safetensors`
   - 来源：`LBH-123-AI/Minimax_h3_latent_Upscaler`
   - 大小：`690592672`
   - SHA256：`043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2`

运行时会优先按仓库路径查找，也会在应用挂载的 `/.autodl` 下按精确文件名自动寻找哈希路径，所以发布前不需要手抄 `/.autodl/<hash>`。

## 正式发布顺序

1. 关机当前制作实例，保存系统盘镜像。
2. 在 AutoDL Art 创建/编辑应用版本，选择该私有镜像。
3. 关联上面 6 个公共模型文件，启动命令填写 `bash /root/start.sh`。
4. 先私下部署一个应用实例，不提交审核。
5. 在新应用实例运行 `python /root/Afei-H3/scripts/verify_market_release.py`，必须 `ok=true`。
6. 打开 6006，真实生成一次视频；确认无模型下载、输出正常。
7. 验证通过后再提交市场审核。

不要把 `/root/autodl-tmp` 数据盘模型复制进系统盘，也不要为发布删除当前数据盘模型。
