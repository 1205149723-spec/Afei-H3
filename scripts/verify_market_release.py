from __future__ import annotations

import json
import os
import shutil

from prepare_models import MODEL_DIR, ROOT, SHARED_ROOTS, SPECS, is_shared_path, valid


def main() -> None:
    errors: list[str] = []
    models: dict[str, dict[str, object]] = {}

    for key, spec in SPECS.items():
        logical = MODEL_DIR / spec.relative
        entry: dict[str, object] = {"logical": str(logical), "symlink": logical.is_symlink()}
        try:
            resolved = logical.resolve(strict=True)
            entry["resolved"] = str(resolved)
            entry["shared"] = is_shared_path(resolved)
            entry["sizeOk"] = valid(resolved, spec, verify_hash=False)
        except OSError as exc:
            entry.update({"shared": False, "sizeOk": False, "error": str(exc)})
            resolved = None
        if not logical.is_symlink():
            errors.append(f"{key}: market model must be a symlink: {logical}")
        if resolved is None or not is_shared_path(resolved):
            errors.append(f"{key}: model does not resolve inside AutoDL shared storage")
        elif not valid(resolved, spec, verify_hash=False):
            errors.append(f"{key}: shared model size/identity contract mismatch: {resolved}")
        models[key] = entry

    for path in (ROOT / "start.sh", ROOT / "start_autodl.sh", ROOT / "app" / "server.py"):
        if not path.is_file():
            errors.append(f"missing release runtime file: {path}")

    usage = shutil.disk_usage(ROOT)
    report = {
        "ok": not errors,
        "packageRoot": str(ROOT),
        "sharedRoots": [str(path) for path in SHARED_ROOTS],
        "modelCount": len(models),
        "models": models,
        "systemDisk": {
            "totalGiB": round(usage.total / (1024**3), 2),
            "usedGiB": round(usage.used / (1024**3), 2),
            "freeGiB": round(usage.free / (1024**3), 2),
        },
        "deploymentMode": os.environ.get("H3_DEPLOYMENT_MODE", "market"),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
