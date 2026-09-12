from __future__ import annotations

import re
from pathlib import Path


def _jupyter_token() -> str:
    config = Path("/init/jupyter/jupyter_config.py")
    try:
        text = config.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = re.search(r"c\.ServerApp\.token\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else ""


def setup_afei_h3() -> dict:
    token = _jupyter_token()
    path_info = "afei-h3/"
    if token:
        path_info += f"?token={token}"
    return {
        "command": [
            "bash",
            "-lc",
            "export H3_PORT={port}; export H3_HOST=127.0.0.1; exec bash /root/start.sh",
        ],
        "timeout": 180,
        "launcher_entry": {
            "enabled": True,
            "title": "阿飞 H3 工作台",
            "category": "Other",
            "path_info": path_info,
        },
        "new_browser_tab": False,
    }
