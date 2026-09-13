from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DEFAULT_STORE = Path(os.environ.get("H3_MODEL_STORE", "/root/autodl-tmp/Afei-H3-models"))
SHARED_ROOTS = (Path("/.autodl-model/data"), Path("/.autodl"))
PUBLIC_MODEL_MAP = Path(os.environ.get("H3_AUTODL_MODEL_MAP", ROOT / "autodl-public-models.json"))


@dataclass(frozen=True)
class ModelSpec:
    relative: str
    url: str
    size: int | None = None
    sha256: str | None = None
    shared_repo: str | None = None


SPECS = {
    "text": ModelSpec(
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors?download=true",
        15_687_142_551,
        "35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6",
        shared_repo="Comfy-Org/MiniMax-H3",
    ),
    "video_vae": ModelSpec(
        "minimax_h3_video_vae_fp16.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors?download=true",
        5_207_808_496,
        "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
        "Comfy-Org/MiniMax-H3",
    ),
    "audio_vae": ModelSpec(
        "minimax_h3_audio_vae_fp32.safetensors",
        "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors?download=true",
        605_254_808,
        "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "Comfy-Org/MiniMax-H3",
    ),
    "fl2va": ModelSpec(
        "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors?download=true",
        12_540_858_008,
        "01aa7b92c007c599890461c325f9b7e3c96fb06c36f242f95b62f7f20e538dec",
        "Kijai/MiniMax-H3-experimental",
    ),
    "ref2va": ModelSpec(
        "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
        "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors?download=true",
        11_770_657_048,
        "de2c6c29c4ee702b45e48e40daae3834aeee58ab681c732d9152589a87c89910",
        "Kijai/MiniMax-H3-experimental",
    ),
    "upscaler": ModelSpec(
        "latent_upscaler/minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_fp16.safetensors?download=true",
        690_592_672,
        "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2",
        "LBH-123-AI/Minimax_h3_latent_Upscaler",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, spec: ModelSpec, *, verify_hash: bool = True) -> bool:
    if not path.is_file():
        return False
    if spec.size is not None and path.stat().st_size != spec.size:
        return False
    return not verify_hash or spec.sha256 is None or file_sha256(path) == spec.sha256


def is_shared_path(path: Path) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    for root in SHARED_ROOTS:
        try:
            resolved.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def configured_candidates(key: str):
    if not PUBLIC_MODEL_MAP.is_file():
        return
    document = json.loads(PUBLIC_MODEL_MAP.read_text(encoding="utf-8"))
    if int(document.get("schemaVersion", 0)) != 1:
        raise RuntimeError(f"unsupported AutoDL public model map schema: {PUBLIC_MODEL_MAP}")
    values = (document.get("models") or {}).get(key) or []
    if not isinstance(values, list):
        raise RuntimeError(f"AutoDL public model map entry must be a list: {key}")
    for value in values:
        candidate = Path(str(value))
        if candidate.is_absolute():
            yield candidate


def shared_candidates(spec: ModelSpec):
    if not spec.shared_repo:
        return
    filename = Path(spec.relative).name
    repo = Path(spec.shared_repo)
    for base in SHARED_ROOTS:
        for prefix in (base / repo, base / "huggingface" / repo, base / "models" / repo):
            if prefix.exists():
                for relative in (
                    Path(filename),
                    Path("text_encoders") / filename,
                    Path("vae") / filename,
                    Path("latent_upscaler") / filename,
                ):
                    candidate = prefix / relative
                    if candidate.is_file():
                        yield candidate
                yield from prefix.rglob(filename)


def download(spec: ModelSpec, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl", "-fL", "--retry", "6", "--retry-delay", "3", "--continue-at", "-",
            "--connect-timeout", "20", spec.url, "-o", str(target),
        ],
        check=True,
    )


def ensure_link(key: str, spec: ModelSpec, store: Path, *, source: str) -> None:
    link = MODEL_DIR / spec.relative
    shared_hash = os.environ.get("H3_VERIFY_SHARED_SHA256", "0") == "1"
    if link.is_symlink() or link.exists():
        try:
            resolved = link.resolve(strict=True)
            if valid(resolved, spec, verify_hash=shared_hash if is_shared_path(resolved) else True):
                if source != "shared-only" or is_shared_path(resolved):
                    print(f"OK existing: {spec.relative} -> {resolved}")
                    return
        except OSError:
            pass
        raise RuntimeError(
            f"model target already exists but is not valid for source={source}: {link}; "
            "it was left untouched"
        )

    if source in {"auto", "shared-only"}:
        seen: set[Path] = set()
        candidates = [*configured_candidates(key), *shared_candidates(spec)]
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if not is_shared_path(resolved):
                continue
            if valid(resolved, spec, verify_hash=shared_hash):
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(resolved)
                print(f"OK shared: {spec.relative} -> {resolved}")
                return

    if source == "shared-only":
        raise RuntimeError(
            "AutoDL market model is unavailable in the mounted shared model library: "
            f"{spec.relative}. Search this exact filename in AutoDL Art's public-model search, "
            f"then add its instance_path to {PUBLIC_MODEL_MAP}. No network download was attempted."
        )

    local = store / spec.relative
    if not valid(local, spec):
        print(f"DOWNLOAD: {spec.relative}")
        download(spec, local)
    if not valid(local, spec):
        raise RuntimeError(f"model verification failed: {spec.relative}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(local)
    print(f"OK local: {spec.relative} -> {local}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("t2v", "all"), default="all")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument(
        "--source",
        choices=("auto", "shared-only", "download"),
        default=os.environ.get("H3_MODEL_SOURCE", "auto"),
        help="shared-only is the AutoDL market mode and never downloads model weights",
    )
    args = parser.parse_args()
    keys = ["text", "video_vae", "audio_vae", "fl2va", "upscaler"]
    if args.mode == "all":
        keys.append("ref2va")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if args.source != "shared-only":
        args.store.mkdir(parents=True, exist_ok=True)
    for key in keys:
        ensure_link(key, SPECS[key], args.store, source=args.source)
    print(f"OK: prepared {len(keys)} model files for mode={args.mode}, source={args.source}")


if __name__ == "__main__":
    main()
