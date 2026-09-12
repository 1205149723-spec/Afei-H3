"""Isolated helpers for the explicitly enabled latent dual-decode experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def _tensor_sha256(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().cpu()
    header = f"{value.dtype}|{tuple(value.shape)}|".encode("ascii")
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(header + raw).hexdigest()


def _decoded_frame_count(latent_t: int) -> int:
    if latent_t < 2 or (latent_t - 2) % 5:
        raise ValueError("video latent T does not map to the H3 17k+5 frame grid")
    return 5 + 17 * ((latent_t - 2) // 5)


def _export_indices(decoded_count: int, export_count: int, model_fps: int, export_fps: int) -> list[int]:
    indices = [int(index * model_fps / export_fps) for index in range(export_count)]
    if not indices or indices[-1] >= decoded_count:
        raise ValueError("dual-decode export plan exceeds decoded frames")
    return indices


def prepare_latent_dual_decode(
    video_latent: Any,
    audio_latent: Any,
    *,
    source_model_fps: int,
    bridged_model_fps: int,
    bridged_video_latent_t: int,
    export_fps: int,
    export_frame_count: int,
) -> Dict[str, Any]:
    """Build two VAE inputs from one sampled latent without touching space or audio."""

    import torch.nn.functional as functional

    source_shape = tuple(int(value) for value in video_latent.shape)
    audio_shape = tuple(int(value) for value in audio_latent.shape)
    if len(source_shape) != 5 or source_shape[2] < 2 or (source_shape[2] - 2) % 5:
        raise ValueError("source video latent T must map to the H3 17k+5 frame grid")
    if len(audio_shape) != 4 or audio_shape[-1] < 1:
        raise ValueError("source audio latent T must be positive")
    if source_model_fps != 16 or bridged_model_fps != 24 or export_fps != 16:
        raise ValueError("latent dual decode requires the fixed 16-to-24 diagnostic identity")
    if bridged_video_latent_t < 2 or (bridged_video_latent_t - 2) % 5:
        raise ValueError("bridged video latent T must map to the H3 17k+5 frame grid")
    if export_frame_count < 1:
        raise ValueError("latent dual decode export frame count must be positive")

    bridged = functional.interpolate(
        video_latent.detach().float(),
        size=(bridged_video_latent_t, source_shape[3], source_shape[4]),
        mode="trilinear",
        align_corners=False,
    ).to(dtype=video_latent.dtype)
    bridged_shape = tuple(int(value) for value in bridged.shape)
    if bridged_shape[:2] != source_shape[:2] or bridged_shape[3:] != source_shape[3:]:
        raise ValueError("temporal bridge changed channel or spatial dimensions")

    source_sha = _tensor_sha256(video_latent)
    bridged_sha = _tensor_sha256(bridged)
    audio_sha = _tensor_sha256(audio_latent)
    direct_decoded = _decoded_frame_count(source_shape[2])
    bridged_decoded = _decoded_frame_count(bridged_shape[2])
    return {
        "directVideoLatent": video_latent,
        "bridgedVideoLatent": bridged,
        "audioLatent": audio_latent,
        "receipt": {
            "sourceVideoShape": list(source_shape),
            "bridgedVideoShape": list(bridged_shape),
            "audioShape": list(audio_shape),
            "sourceVideoDtype": str(video_latent.dtype),
            "bridgedVideoDtype": str(bridged.dtype),
            "audioDtype": str(audio_latent.dtype),
            "sourceVideoTensorSha256": source_sha,
            "bridgedVideoTensorSha256": bridged_sha,
            "audioTensorSha256": audio_sha,
            "directSamplingSourceSha256": source_sha,
            "bridgedSamplingSourceSha256": source_sha,
            "interpolation": "trilinear_temporal_only_align_corners_false",
            "alignCorners": False,
            "directDecodedFrameCount": direct_decoded,
            "bridgedDecodedFrameCount": bridged_decoded,
            "directExportIndices": _export_indices(direct_decoded, export_frame_count, source_model_fps, export_fps),
            "bridgedExportIndices": _export_indices(bridged_decoded, export_frame_count, bridged_model_fps, export_fps),
            "exportDurationSeconds": export_frame_count / export_fps,
            "fallback": False,
        },
    }


def _file_receipt(path: Path, path_base: Path | None = None) -> Dict[str, Any]:
    display_path = path.relative_to(path_base).as_posix() if path_base is not None else str(path)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def persist_latent_dual_decode(
    bundle: Dict[str, Any],
    output_dir: Path,
    *,
    path_base: Path | None = None,
) -> Dict[str, Any]:
    """Persist the exact sampled and bridged latents for later independent audit."""

    from safetensors.torch import save_file

    output_dir = Path(output_dir).resolve()
    path_base = Path(path_base).resolve() if path_base is not None else None
    if path_base is not None:
        output_dir.relative_to(path_base)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "sourceVideo": output_dir / "source_video_latent.safetensors",
        "sourceAudio": output_dir / "source_audio_latent.safetensors",
        "bridgedVideo": output_dir / "bridged_video_latent.safetensors",
    }
    save_file({"video": bundle["directVideoLatent"].detach().contiguous().cpu()}, str(paths["sourceVideo"]))
    save_file({"audio": bundle["audioLatent"].detach().contiguous().cpu()}, str(paths["sourceAudio"]))
    save_file({"video": bundle["bridgedVideoLatent"].detach().contiguous().cpu()}, str(paths["bridgedVideo"]))
    artifacts = {name: _file_receipt(path, path_base) for name, path in paths.items()}
    manifest_path = output_dir / "manifest.json"
    manifest = {"schemaVersion": 1, "receipt": bundle["receipt"], "artifacts": artifacts}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts["manifest"] = _file_receipt(manifest_path, path_base)
    return {"receipt": bundle["receipt"], "artifacts": artifacts}
