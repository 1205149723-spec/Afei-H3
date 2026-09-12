"""Deterministic H3 reference-video planning and cache identity helpers.

This module never opens a video or model. It describes the exact legal frame
window that the streaming decoder must enter into H3, plus configurable semantic
sample positions used by the H3 text-conditioning path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


FPS = 24
QWEN_FPS = 2
MIN_H3_FRAMES = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Complete is the faithful product baseline. Performance budgets are opt-in
# policy presets; they never silently replace a user's selected interval.
DEFAULT_REFERENCE_POLICY = "complete"
REFERENCE_BUDGET_PRESETS = {
    "fast_5090": (73, 39, 39),
    "balanced_5090": (141, 73, 73),
    "complete": (None, None, None),
}
REFERENCE_RUNTIME_VERSION = "h3-reference-planner-v1"
VIDEO_VAE_ID = "minimax_h3_video_vae_fp16.safetensors"
FIXED_ASSET_MAX_PIXELS = 960 * 544
MODEL_FILES = {
    "FL2VA": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "REF2VA": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
}


def adapt_reference_canvas(width: int, height: int, max_pixels: int = 768 * 1344) -> Dict[str, int]:
    if width <= 0 or height <= 0:
        raise ReferencePlanError("参考视频缺少真实宽高")
    if width * height <= max_pixels:
        scale = 1.0
    else:
        scale = math.sqrt(max_pixels / (width * height))
    target_width = max(32, int(round(width * scale / 32)) * 32)
    target_height = max(32, int(round(height * scale / 32)) * 32)
    while target_width * target_height > max_pixels:
        if target_width >= target_height:
            target_width -= 32
        else:
            target_height -= 32
    return {"width": target_width, "height": target_height}


class ReferencePlanError(ValueError):
    """A reference segment cannot be represented safely by H3."""


def align_down_h3(frame_count: int) -> int:
    """Return the greatest legal H3 count <= frame_count (17*k+5)."""

    count = int(frame_count)
    if count < MIN_H3_FRAMES:
        return 0
    return count - ((count - MIN_H3_FRAMES) % 17)


def _semantic_fps(advanced: Dict[str, Any]) -> int:
    raw = advanced.get("reference_video_fps", QWEN_FPS)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ReferencePlanError("reference_video_fps must be an integer from 1 to 8") from exc
    if value < 1 or value > 5:
        raise ReferencePlanError("reference_video_fps must be between 1 and 5")
    return value


def semantic_samples(frame_count: int, start_frame: int = 0, semantic_fps: int = QWEN_FPS) -> Dict[str, Any]:
    """Select semantic frames at the requested FPS from the 24-fps legal window."""

    if frame_count < MIN_H3_FRAMES or frame_count % 17 != 5:
        raise ReferencePlanError("语义抽样要求合法的 H3 帧数")
    value = _semantic_fps({"reference_video_fps": semantic_fps})
    sample_count = max(1, math.ceil(frame_count * value / FPS))
    local_indices = sorted({min(frame_count - 1, int(round(index * FPS / value))) for index in range(sample_count)})
    timestamps = [round(index / FPS, 6) for index in local_indices]
    return {
        "inputFps": FPS,
        "semanticFps": value,
        "strideFrames": FPS / value,
        "localFrameIndices": local_indices,
        "sourceFrameIndices": [start_frame + index for index in local_indices],
        "timestampsSeconds": timestamps,
        "sourceTimestampsSeconds": [round((start_frame + index) / FPS, 6) for index in local_indices],
    }


def _finite(value: Any, name: str, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReferencePlanError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ReferencePlanError(f"{name} must be finite")
    return result


def _policy_selection(advanced: Dict[str, Any]) -> Dict[str, Any]:
    requested = advanced.get("referenceVideoBudgets")
    preset = str(advanced.get("referencePolicyPreset") or DEFAULT_REFERENCE_POLICY)
    if preset not in REFERENCE_BUDGET_PRESETS:
        raise ReferencePlanError(f"参考范围策略必须是：{', '.join(REFERENCE_BUDGET_PRESETS)}")
    return {
        "preset": preset,
        "overridden": requested is not None,
        "defaultBudgets": list(REFERENCE_BUDGET_PRESETS[preset]),
        "requestedBudgets": list(requested) if isinstance(requested, (list, tuple)) else None,
    }


def _budget_for(video_ordinal: int, advanced: Dict[str, Any], policy: Dict[str, Any]) -> int | None:
    requested = advanced.get("referenceVideoBudgets")
    if requested is not None:
        if not isinstance(requested, (list, tuple)):
            raise ReferencePlanError("referenceVideoBudgets must be an array")
        if video_ordinal - 1 >= len(requested):
            raise ReferencePlanError("referenceVideoBudgets must cover every reference video")
        try:
            budget = int(requested[video_ordinal - 1])
        except (TypeError, ValueError) as exc:
            raise ReferencePlanError("referenceVideoBudgets must contain integers") from exc
    else:
        defaults = policy["defaultBudgets"]
        budget = defaults[min(video_ordinal - 1, len(defaults) - 1)]
    if budget is None:
        return None
    if budget < MIN_H3_FRAMES:
        raise ReferencePlanError("each reference video budget must be at least 5 frames")
    return budget


def _content_identity(reference: Dict[str, Any]) -> Dict[str, Any]:
    digest = str(reference.get("sha256") or reference.get("contentHash") or "")
    return {
        "sha256": digest,
        "assetId": str(reference.get("assetId") or ""),
        "name": str(reference.get("name") or ""),
        "size": int(reference.get("size") or 0),
    }


def reference_cache_key(
    reference: Dict[str, Any],
    segment: Dict[str, Any],
    canvas: Dict[str, Any],
    primary_model: str,
    semantic_fps: int = QWEN_FPS,
    runtime_version: str = REFERENCE_RUNTIME_VERSION,
) -> str:
    """Hash all correctness-relevant pre-processing identity fields."""

    identity = {
        "runtimeVersion": runtime_version,
        "modelFile": MODEL_FILES.get(primary_model, primary_model),
        "videoVae": VIDEO_VAE_ID,
        "content": _content_identity(reference),
        "segment": {
            "startSeconds": segment["startSeconds"],
            "endSeconds": segment["sourceWindowEndSeconds"],
            "sourceFps": reference.get("sourceFps"),
            "sourceMediaFacts": reference.get("mediaFacts") or {},
            "frameCount": segment["h3FrameCount"],
            "semanticFps": semantic_fps,
        },
        "canvas": canvas,
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _video_plan(
    reference: Dict[str, Any],
    video_ordinal: int,
    output_frame_count: int,
    canvas: Dict[str, Any],
    primary_model: str,
    advanced: Dict[str, Any],
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    budget = _budget_for(video_ordinal, advanced, policy)
    original_duration = reference.get("originalDurationSeconds")
    source_frame_count = reference.get("sourceFrameCount")
    source_fps = _finite(reference.get("sourceFps"), "sourceFps", FPS)
    source_video = (reference.get("mediaFacts") or {}).get("video") or {}
    max_pixels = FIXED_ASSET_MAX_PIXELS if str(advanced.get("assetPrecision") or "official") == "fixed_optimized" else 768 * 1344
    target_canvas = adapt_reference_canvas(int(source_video.get("width") or 0), int(source_video.get("height") or 0), max_pixels) if source_video else dict(canvas["internal"])
    availability_known = original_duration is not None or source_frame_count is not None
    if source_frame_count is None and original_duration is not None:
            source_frame_count = int(math.floor(_finite(original_duration, "originalDurationSeconds") * source_fps + 1e-6))
    if source_frame_count is not None:
        source_frame_count = int(source_frame_count)
        if source_frame_count < MIN_H3_FRAMES:
            raise ReferencePlanError(f"reference video {video_ordinal} has fewer than 5 source frames")

    start = _finite(reference.get("startSeconds"), "startSeconds", 0.0)
    if start < 0:
        raise ReferencePlanError("reference video startSeconds cannot be negative")
    if original_duration is not None and start >= _finite(original_duration, "originalDurationSeconds"):
        raise ReferencePlanError("reference video startSeconds is beyond the source duration")

    requested_end = reference.get("endSeconds")
    if requested_end is None or requested_end == "":
        if budget is None:
            if original_duration is not None:
                requested_end = _finite(original_duration, "originalDurationSeconds")
            elif source_frame_count is not None:
                requested_end = start + max(0, source_frame_count - int(round(start * FPS))) / FPS
            else:
                requested_end = start + output_frame_count / FPS
        else:
            requested_end = start + budget / FPS
    end = _finite(requested_end, "endSeconds")
    if end <= start:
        raise ReferencePlanError("reference video endSeconds must be after startSeconds")
    reasons: List[str] = []
    source_duration = _finite(original_duration, "originalDurationSeconds") if original_duration is not None else None
    if source_duration is not None and end > source_duration:
        end = source_duration
        reasons.append("选段终点已限制到源文件实际时长")

    start_frame = int(round(start * FPS))
    source_window_frames = int(math.floor((end - start) * FPS + 1e-6))
    if source_frame_count is not None:
        source_window_frames = min(source_window_frames, max(0, source_frame_count - start_frame))
    if source_window_frames < MIN_H3_FRAMES:
        raise ReferencePlanError(f"reference video {video_ordinal} segment has fewer than 5 frames at 24fps")

    requested_h3_frames = min(source_window_frames, output_frame_count)
    if budget is not None:
        requested_h3_frames = min(requested_h3_frames, budget)
    h3_frame_count = align_down_h3(requested_h3_frames)
    if h3_frame_count < MIN_H3_FRAMES:
        raise ReferencePlanError(
            f"reference video {video_ordinal} cannot fit a legal H3 segment under the output frame limit"
        )
    if source_window_frames > h3_frame_count:
        cap_reason = "合法 17*k+5 帧网格"
        if budget is not None and budget < source_window_frames:
            cap_reason += "和已选参考范围预算"
        if output_frame_count < source_window_frames:
            cap_reason += "和输出帧数上限"
        reasons.append(f"按{cap_reason}从 {source_window_frames} 帧对齐为 {h3_frame_count} 帧，末尾未采用")
    if budget is not None and budget > output_frame_count:
        reasons.append("参考预算已限制到输出帧数")
    if not availability_known:
        reasons.append("缺少源时长或帧数，解码器必须再次确认")

    segment = {
        "startSeconds": round(start_frame / FPS, 6),
        "requestedEndSeconds": round(float(requested_end), 6),
        "sourceWindowEndSeconds": round(start_frame / FPS + h3_frame_count / FPS, 6),
        "sourceWindowFrames": source_window_frames,
        "h3FrameCount": h3_frame_count,
    }
    semantic_fps = _semantic_fps(advanced)
    semantic = semantic_samples(h3_frame_count, start_frame, semantic_fps)
    cache_key = reference_cache_key(reference, segment, target_canvas, primary_model, semantic_fps=semantic_fps)
    required_identity = {
        "sha256": bool(reference.get("sha256")), "mediaFacts": bool(reference.get("mediaFacts")),
        "sourceFps": bool(reference.get("sourceFps")), "sourceFrameCount": source_frame_count is not None,
        "targetCanvas": bool(target_canvas), "model": bool(primary_model), "videoVae": bool(VIDEO_VAE_ID),
        "runtimeVersion": bool(REFERENCE_RUNTIME_VERSION),
    }
    return {
        "videoOrdinal": video_ordinal,
        "token": str(reference.get("token") or f"<Video {video_ordinal}>"),
        "name": str(reference.get("name") or f"reference-video-{video_ordinal}"),
        "originalDurationSeconds": original_duration,
        "sourceMedia": {
            "fps": source_fps,
            "frameCount": source_frame_count,
            "durationSeconds": original_duration,
            "facts": reference.get("mediaFacts") or {},
        },
        "availabilityKnown": availability_known,
        "selectedInterval": {
            "startSeconds": segment["startSeconds"],
            "endSeconds": segment["sourceWindowEndSeconds"],
            "durationSeconds": round(h3_frame_count / FPS, 6),
        },
        "sourceWindowFrames": source_window_frames,
        "requestedBudgetFrames": source_window_frames if budget is None else budget,
        "policyPreset": policy["preset"],
        "policyOverridden": policy["overridden"],
        "h3FrameCount": h3_frame_count,
        "formula": f"17*{(h3_frame_count - 5) // 17}+5",
        "decode": {
            "fps": FPS,
            "frameCount": h3_frame_count,
            "streaming": True,
            "decodeOnlyRequiredLegalFrames": True,
        },
        "officialResize": {"mode": "固定优化" if max_pixels == FIXED_ASSET_MAX_PIXELS else "官方默认", "preserveAspect": True, "upscale": False, "crop": "不裁剪", **target_canvas},
        "semanticSampling": semantic,
        "truncated": bool(reasons),
        "truncationReasons": reasons,
        "cache": {
            "key": cache_key,
            "metadataPath": f"cache/reference_preprocess/{cache_key}.json",
            "tensorCache": False,
            "identityIncludesContentSegmentCanvasModelRuntime": True,
            "identityFields": required_identity,
            "eligible": all(required_identity.values()),
            "missReason": None if all(required_identity.values()) else "缓存身份字段不完整，禁止命中",
        },
    }


def plan_reference_videos(
    references: Iterable[Dict[str, Any]],
    output_frame_count: int,
    canvas: Dict[str, Any],
    primary_model: str,
    advanced: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    advanced = advanced or {}
    policy = _policy_selection(advanced)
    videos = [ref for ref in references if ref.get("kind") == "Video"]
    plans = [
        _video_plan(ref, ordinal, output_frame_count, canvas, primary_model, advanced, policy)
        for ordinal, ref in enumerate(videos, 1)
    ]
    return {
        "inputFps": FPS,
        "semanticFps": _semantic_fps(advanced),
        "policyPreset": policy["preset"],
        "policyOverridden": policy["overridden"],
        "defaultBudgets": policy["defaultBudgets"],
        "availablePolicies": {name: list(values) for name, values in REFERENCE_BUDGET_PRESETS.items()},
        "requestedBudgets": policy["requestedBudgets"],
        "videos": plans,
        "totalBudgetFrames": sum(plan["h3FrameCount"] for plan in plans),
        "totalSemanticFrames": sum(len(plan["semanticSampling"]["localFrameIndices"]) for plan in plans),
        "tokenBudget": {
            "referenceVideoFrames": sum(plan["h3FrameCount"] for plan in plans),
            "semanticVideoFrames": sum(len(plan["semanticSampling"]["localFrameIndices"]) for plan in plans),
            "bounded": True,
        },
        "noSilentTruncation": True,
    }
