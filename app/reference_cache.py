"""Correctness-first reference pre-processing metadata cache.

Only JSON planning metadata is cached. Video tensors, VAE latents and Qwen
features are intentionally not persisted until device/runtime equivalence is
verified. All writes are confined to the project-local cache directory.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = (PROJECT_ROOT / "cache" / "reference_preprocess").resolve()


def _cache_path(key: str) -> Path:
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key.lower()):
        raise ValueError("reference cache key must be a SHA-256 hex digest")
    path = (CACHE_ROOT / f"{key}.json").resolve()
    path.relative_to(CACHE_ROOT)
    return path


def read_manifest(key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(key)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if value.get("cacheKey") == key and value.get("schema") == "h3-reference-cache-v1" else None


def write_manifest(key: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Atomically store a metadata-only manifest under the D: cache root."""

    path = _cache_path(key)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "h3-reference-cache-v1",
        "cacheKey": key,
        "kind": "metadata_only",
        "plan": plan,
        "tensorCache": False,
        "vaeLatentCache": False,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=str(CACHE_ROOT))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
        Path(temp_name).replace(path)
    finally:
        temp_path = Path(temp_name)
        if temp_path.exists():
            temp_path.unlink()
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "cacheKey": key, "cacheHit": False, "kind": "metadata_only"}
