from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DEFAULT_STORE = Path(os.environ.get("H3_MODEL_STORE", "/root/autodl-tmp/Afei-H3-models"))
SHARED_ROOTS = (Path("/.autodl-model/data"), Path("/.autodl"))


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
    ),
    "ref2va": ModelSpec(
        "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
        "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/minimax_h3_ref2va_pruned_w4a8_mixed.safetensors?download=true",
        11_770_657_048,
        "de2c6c29c4ee702b45e48e40daae3834aeee58ab681c732d9152589a87c89910",
    ),
    "upscaler": ModelSpec(
        "latent_upscaler/minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/main/minimax_h3_latent_upscaler_3d_fp16.safetensors?download=true",
        690_592_672,
        "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid(path: Path, spec: ModelSpec) -> bool:
    if not path.is_file():
        return False
    if spec.size is not None and path.stat().st_size != spec.size:
        return False
    return spec.sha256 is None or file_sha256(path) == spec.sha256


def shared_candidates(spec: ModelSpec):
    if not spec.shared_repo:
        return
    filename = Path(spec.relative).name
    repo = Path(spec.shared_repo)
    for base in SHARED_ROOTS:
        for prefix in (base / repo, base / "huggingface" / repo, base / "models" / repo):
            if prefix.exists():
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


def ensure_link(spec: ModelSpec, store: Path) -> None:
    link = MODEL_DIR / spec.relative
    if link.is_symlink() or link.exists():
        try:
            if valid(link.resolve(), spec):
                print(f"OK existing: {spec.relative}")
                return
        except OSError:
            pass
        link.unlink(missing_ok=True)

    for candidate in shared_candidates(spec):
        if valid(candidate, spec):
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(candidate)
            print(f"OK shared: {spec.relative} -> {candidate}")
            return

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
    args = parser.parse_args()
    keys = ["text", "video_vae", "audio_vae", "fl2va", "upscaler"]
    if args.mode == "all":
        keys.append("ref2va")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    args.store.mkdir(parents=True, exist_ok=True)
    for key in keys:
        ensure_link(SPECS[key], args.store)
    print(f"OK: prepared {len(keys)} model files for mode={args.mode}")


if __name__ == "__main__":
    main()
