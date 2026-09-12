# H3 运行时第三方来源记录

本文件只记录已在本机发现的来源与路径；实际加载、版本和路线必须以启动时健康接口与每个任务的 execution receipt（执行回执）核实。

## 已核实来源

- KJNodes：提交 `ab8f90f02ad6ec3a4900b1b4df9c03cded7b4690`，GPL-3.0；本地文件 `D:\HailuoH3\runtime\third_party\KJNodes_ltxv_nodes_ab8f90f0.py`，许可证 `D:\HailuoH3\runtime\third_party\KJNodes_LICENSE_GPL-3.0.txt`。
- SageAttention：审计来源提交 `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`，Apache-2.0；私有包目录 `D:\HailuoH3\runtime\python_packages\sageattention-2.2.0-cu128-cp314`。
- Spectrum：来源 `xmarre/ComfyUI-Spectrum-MiniMax-H3`，提交 `8bfc235cb3910c73964277e0316ec875f4b2c011`，目录 `D:\HailuoH3\runtime\third_party\ComfyUI-Spectrum-MiniMax-H3-8bfc235`，GPL-3.0；仅实验路线，必须真实 A/B 后才可启用，且不是官方推荐。
- EasyCache：使用 ComfyUI 提交 `14b05228cef127ce529bc0c08660770d4af3e9a8` 的 `EasyCacheNode.execute` 入口；未发现独立复制的第三方目录。它是实验路线，不是官方推荐，失败必须显式写入回执。
- TE-Speed：来源 `HELPMEEADICE/TE-Speed-MiniMaxH3-OSS`，提交 `c1dacf47bc02cb9326f7b93c69280529b93d391b`，目录 `D:\HailuoH3\runtime\third_party\TE-Speed-MiniMaxH3-OSS-c1dacf47`，LGPL-3.0；仅实验路线，必须真实 A/B 后才可启用，且不是官方推荐。
- TeaCache：保留作可追溯诊断来源，不是产品、UI、API 或正式路线；不得自动加载。

任何新增、升级或重新安装第三方来源，均需先说明用途、来源、精确版本、许可证、维护状态、安装位置、影响、风险和回滚方式，并获得用户明确同意。
