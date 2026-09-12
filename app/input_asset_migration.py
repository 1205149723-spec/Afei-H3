"""Non-destructive migration of legacy user media into the input seam."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from media_staging import INPUT_ROOT, LEGACY_STAGING_ROOT, METADATA_ROOT, MANIFEST_NAME, _safe_name

MEDIA = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".gif": "image",
         ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
         ".mp4": "video", ".mov": "video", ".webm": "video", ".avi": "video"}


def migrate_legacy_assets(*, source: Path = LEGACY_STAGING_ROOT, destination: Path = INPUT_ROOT,
                          apply: bool = False) -> dict[str, Any]:
    """Plan or apply a collision-safe, repeatable migration; unknown files stay put."""
    source, destination = source.resolve(), destination.resolve()
    rows: list[dict[str, Any]] = []
    for item in sorted(source.rglob("*")) if source.exists() else []:
        if not item.is_file() or item.name == MANIFEST_NAME:
            continue
        kind = MEDIA.get(item.suffix.lower())
        if not kind:
            rows.append({"source": item.relative_to(source).as_posix(), "status": "unrecognized"})
            continue
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        asset_dir = destination / kind / digest[:16]
        target = asset_dir / _safe_name(item.name)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == digest:
            rows.append({"source": item.relative_to(source).as_posix(), "target": target.relative_to(destination.parent).as_posix(), "status": "already_present", "sha256": digest})
            continue
        if target.exists():
            rows.append({"source": item.relative_to(source).as_posix(), "status": "conflict"})
            continue
        if apply:
            asset_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
            METADATA_ROOT.mkdir(parents=True, exist_ok=True)
            (METADATA_ROOT / f"{asset_dir.name}.json").write_text(json.dumps({"assetId": asset_dir.name, "createdBy": "legacy-migration", "owners": [], "cleanupEligible": False}, ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append({"source": item.relative_to(source).as_posix(), "target": target.relative_to(destination.parent).as_posix(), "status": "migrated", "sha256": digest})
        else:
            rows.append({"source": item.relative_to(source).as_posix(), "target": target.relative_to(destination.parent).as_posix(), "status": "planned", "sha256": digest})
    return {"source": source.name, "destination": destination.relative_to(destination.parent).as_posix(), "apply": apply, "assets": rows}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate legacy user media into input/ without deleting sources")
    parser.add_argument("--apply", action="store_true", help="copy recognized media; default is dry-run")
    args = parser.parse_args()
    print(json.dumps(migrate_legacy_assets(apply=args.apply), ensure_ascii=False, indent=2))
