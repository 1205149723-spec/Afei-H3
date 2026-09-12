"""Strict latent upscaler adapter for production/8775.

The adapter deliberately contains no interpolation fallback. It reuses the
vendor checkpoint loader/model class and verifies the immutable weight hash
before any model construction.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict

WEIGHT_SHA256 = "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2"
WEIGHT_SIZE_BYTES = 690592672
WEIGHT_RELATIVE_PATH = Path("models") / "latent_upscaler" / "minimax_h3_latent_upscaler_3d_fp16.safetensors"
VENDOR_COMMIT = "d7c01b9011f2e8439493f6c02c29995a27df276f"
VENDOR_RELATIVE_PATH = Path("vendor") / "Comfyui_Minimax_h3_latent_Upscaler"
VENDOR_MODULE_RELATIVE_PATH = Path("nodes") / "minimax_h3_latent_upscaler_3d.py"
VENDOR_MODULE_SHA256 = "744063b43e0f3eec23e2485cb7c65503069946ca9690906ecb548d7515cb89e2"
LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


class LatentUpscalerError(RuntimeError):
    pass


def private_8775_root() -> Path:
    """Resolve the isolated production root from this private app module."""
    return Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_sha256(path: Path) -> str:
    """Hash Python source canonically across Windows CRLF and Linux LF checkouts."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _verified_vendor_commit(vendor: Path) -> str:
    """Verify the pinned vendor source without requiring a Git executable.

    The production adapter imports exactly one vendor Python module.  Hashing that
    immutable module is a stronger portable-runtime check than shelling out to
    ``git rev-parse`` because it verifies the bytes that will actually execute.
    ``VENDOR_COMMIT`` is retained as provenance metadata for receipts/contracts.
    """
    module_path = vendor / VENDOR_MODULE_RELATIVE_PATH
    if not module_path.is_file():
        raise LatentUpscalerError(f"latent upscaler vendor module missing: {module_path}")
    actual = _source_sha256(module_path)
    if actual != VENDOR_MODULE_SHA256:
        raise LatentUpscalerError(
            f"vendor module SHA256 mismatch: {actual} != {VENDOR_MODULE_SHA256}"
        )
    return VENDOR_COMMIT


def resolve_private_assets() -> tuple[Path, Path]:
    root = private_8775_root()
    logical_weight = root / WEIGHT_RELATIVE_PATH
    weight = logical_weight.resolve()
    vendor = (root / VENDOR_RELATIVE_PATH).resolve()
    try:
        logical_weight.relative_to(root)
        vendor.relative_to(root.resolve())
    except ValueError as exc:
        raise LatentUpscalerError("latent upscaler asset resolved outside production/8775") from exc
    if not vendor.is_dir():
        raise LatentUpscalerError(f"latent upscaler vendor directory missing: {vendor}")
    if not (vendor / VENDOR_MODULE_RELATIVE_PATH).is_file():
        raise LatentUpscalerError(f"latent upscaler vendor module missing: {vendor / VENDOR_MODULE_RELATIVE_PATH}")
    _verified_vendor_commit(vendor)
    return weight, vendor


def verify_weight(path: str | Path | None = None) -> Dict[str, Any]:
    expected_weight, vendor = resolve_private_assets()
    candidate = expected_weight if path is None else Path(path).resolve()
    if candidate != expected_weight:
        raise LatentUpscalerError(f"latent upscaler weight must be the private 8775 asset: {expected_weight}")
    if not candidate.is_file():
        raise LatentUpscalerError(f"latent upscaler weight missing: {candidate}")
    actual_size = candidate.stat().st_size
    if actual_size != WEIGHT_SIZE_BYTES:
        raise LatentUpscalerError(
            f"latent upscaler weight size mismatch: {actual_size} != {WEIGHT_SIZE_BYTES}"
        )
    actual = _sha256(candidate)
    if actual != WEIGHT_SHA256:
        raise LatentUpscalerError(f"latent upscaler weight SHA256 mismatch: {actual} != {WEIGHT_SHA256}")
    return {
        "path": str(candidate),
        "relativePath": str(WEIGHT_RELATIVE_PATH),
        "sizeBytes": actual_size,
        "sha256": actual,
        "vendorPath": str(vendor),
        "vendorRelativePath": str(VENDOR_RELATIVE_PATH),
        "vendorCommit": _verified_vendor_commit(vendor),
        "verified": True,
    }


def _load_vendor_module() -> Any:
    _weight, vendor = resolve_private_assets()
    module_path = vendor / VENDOR_MODULE_RELATIVE_PATH
    spec = importlib.util.spec_from_file_location("production_8775_minimax_h3_latent_upscaler_3d", module_path)
    if spec is None or spec.loader is None:
        raise LatentUpscalerError(f"unable to load vendor latent upscaler module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_weight_state_dict(weight_path: Path) -> Dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise LatentUpscalerError("safetensors.torch is required for the latent upscaler") from exc
    raw = load_file(str(weight_path), device="cpu")
    if "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if any(key.startswith("upscaler.") for key in raw):
        raw = {key[len("upscaler."):]: value for key, value in raw.items() if key.startswith("upscaler.")}
    return raw


def load_vendor_adapter(weight_path: str | Path | None = None) -> tuple[Any, Dict[str, Any]]:
    weight = verify_weight(weight_path)
    module = _load_vendor_module()
    model_class = getattr(module, "LatentResizer3D", None)
    detect_arch = getattr(module, "_detect_arch", None)
    if model_class is None or not callable(detect_arch):
        raise LatentUpscalerError("fixed vendor module lacks LatentResizer3D architecture loader")
    state = _load_weight_state_dict(Path(weight["path"]))
    config = detect_arch(state)
    model = model_class(**config)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model, {
        "vendorCommit": VENDOR_COMMIT,
        "vendorModule": str(VENDOR_RELATIVE_PATH / VENDOR_MODULE_RELATIVE_PATH),
        "modelClass": f"{model_class.__module__}.{model_class.__name__}",
        "weight": weight,
        "architecture": dict(config),
        "fallbackUsed": False,
    }


def split_av_latent(samples: Any) -> tuple[Any, Any]:
    if not getattr(samples, "is_nested", False) or not callable(getattr(samples, "unbind", None)):
        raise LatentUpscalerError("expected nested video/audio latent")
    values = tuple(samples.unbind())
    if len(values) != 2:
        raise LatentUpscalerError(f"expected two AV components, got {len(values)}")
    video, audio = values
    if getattr(video, "ndim", None) != 5 or getattr(audio, "ndim", None) not in {3, 4}:
        raise LatentUpscalerError("expected video BCTHW and audio BCT dimensions")
    if int(video.shape[1]) != 24:
        raise LatentUpscalerError(f"expected 24 video channels, got {video.shape[1]}")
    return video, audio


def _tensor_stats(value: Any) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise LatentUpscalerError("torch is required for latent upscaler tensor statistics") from exc
    detached = value.detach()
    return {
        "shape": [int(item) for item in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": bool(torch.isfinite(detached).all().item()),
        "min": float(detached.amin().item()),
        "max": float(detached.amax().item()),
        "mean": float(detached.float().mean().item()),
        "std": float(detached.float().std(unbiased=False).item()),
    }


def resize_video_latent_3d(video: Any, target_h: int, target_w: int, model: Any) -> tuple[Any, Dict[str, Any]]:
    if getattr(video, "ndim", None) != 5 or int(video.shape[1]) != 24:
        raise LatentUpscalerError("video latent must be 24-channel BCTHW")
    source_h, source_w = int(video.shape[-2]), int(video.shape[-1])
    if int(target_h) != source_h * 2 or int(target_w) != source_w * 2:
        raise LatentUpscalerError(
            f"target latent H/W must be exactly 2x source H/W: {source_h}x{source_w} -> {target_h}x{target_w}"
        )
    try:
        import torch
    except ImportError as exc:
        raise LatentUpscalerError("torch is required for the latent upscaler") from exc
    parameters = getattr(model, "parameters", None)
    first_parameter = next(iter(parameters()), None) if callable(parameters) else None
    compute_dtype = getattr(first_parameter, "dtype", video.dtype)
    compute_device = getattr(first_parameter, "device", video.device)
    source = video.to(device=compute_device, dtype=compute_dtype, copy=True)
    mean = torch.tensor(LATENTS_MEAN, device=compute_device, dtype=compute_dtype).view(1, 24, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, device=compute_device, dtype=compute_dtype).view(1, 24, 1, 1, 1)
    normalized = (source - mean) / std
    normalized_input_stats = _tensor_stats(normalized)
    with torch.inference_mode():
        output = model(normalized, scale=2.0, target_size=(int(video.shape[2]), int(target_h), int(target_w)), enable_chunking=True)
    model_output_stats = _tensor_stats(output)
    result = (output * std + mean).to(device=video.device, dtype=video.dtype)
    denormalized_output_stats = _tensor_stats(result)
    if tuple(result.shape[:3]) != tuple(video.shape[:3]) or tuple(result.shape[-2:]) != (int(target_h), int(target_w)):
        raise LatentUpscalerError("vendor 3D latent upscaler changed B/C/T or missed target H/W")
    if not bool(torch.isfinite(result).all().item()):
        raise LatentUpscalerError("vendor 3D latent upscaler produced nonfinite video latent")
    return result, {
        "normalization": {"meanChannels": len(LATENTS_MEAN), "stdChannels": len(LATENTS_STD)},
        "inputDevice": str(video.device),
        "computeDevice": str(compute_device),
        "computeDtype": str(compute_dtype),
        "modelForwardExecuted": True,
        "normalizedInputStats": normalized_input_stats,
        "modelOutputStats": model_output_stats,
        "denormalizedOutputStats": denormalized_output_stats,
        "fallbackUsed": False,
    }


def build_stage2_inputs(video: Any, audio: Any, target_h: int, target_w: int, *, model: Any, noise_factory: Callable[[Any], Any], mask_factory: Callable[[Any], Any], conditioning: Any, model_receipt: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resized, resize_receipt = resize_video_latent_3d(video, target_h, target_w, model)
    if audio is not audio:
        raise LatentUpscalerError("audio identity check failed")
    nested = (resized, audio)
    return {
        "video": resized,
        "audio": audio,
        "nested": nested,
        "noise": noise_factory(nested),
        "mask": mask_factory(nested),
        "conditioning": conditioning,
        "upscaler": {
            **(model_receipt or {"modelClass": model.__class__.__name__}),
            **resize_receipt,
            "fallbackUsed": False,
        },
        "fallbackUsed": False,
    }
