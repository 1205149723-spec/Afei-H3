"""Start the server or probe dependencies with a project-only import path."""

from __future__ import annotations

import json
import runpy
import sys
import sysconfig
from pathlib import Path


APP = Path(__file__).resolve().parent
ROOT = APP.parent
ALLOWED_IMPORT_ROOTS = (
    APP,
    ROOT / "runtime" / "python_packages" / "sageattention-2.2.0-cu130-torch211-py312-linux",
    ROOT / "runtime" / "python_packages",
    ROOT / "runtime" / "ComfyUI",
)


def main() -> None:
    for path in reversed(ALLOWED_IMPORT_ROOTS):
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT):
            raise RuntimeError(f"运行时导入路径越界：{path}")
        if resolved.is_dir():
            sys.path.insert(0, str(resolved))
    if "--portable-probe" in sys.argv:
        import torch
        import torchaudio
        import torchvision

        print(json.dumps({
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "cudaAvailable": torch.cuda.is_available(),
            "cacheTag": sys.implementation.cache_tag,
            "soabi": sysconfig.get_config_var("SOABI"),
            "torchPath": str(Path(torch.__file__).resolve().relative_to(ROOT)),
            "torchaudioPath": str(Path(torchaudio.__file__).resolve().relative_to(ROOT)),
            "torchvisionPath": str(Path(torchvision.__file__).resolve().relative_to(ROOT)),
        }, ensure_ascii=False))
        if (
            torch.__version__ != "2.11.0+cu128"
            or torchaudio.__version__ != "2.11.0+cu128"
            or torchvision.__version__ != "0.26.0+cu128"
            or torch.version.cuda != "12.8"
            or sys.implementation.cache_tag != "cpython-314"
            or sysconfig.get_config_var("SOABI") != "cp314-win_amd64"
            or not torch.cuda.is_available()
        ):
            raise SystemExit(1)
        return
    runpy.run_path(str(APP / "server.py"), run_name="__main__")


if __name__ == "__main__":
    main()
