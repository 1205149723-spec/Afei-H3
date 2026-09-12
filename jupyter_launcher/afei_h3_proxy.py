from __future__ import annotations


def setup_afei_h3() -> dict:
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
        },
        "new_browser_tab": False,
    }
