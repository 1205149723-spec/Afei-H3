"""Server-side media facts from the actual stored bytes."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict


class MediaProbeError(ValueError):
    pass


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rate(value: Any) -> float | None:
    raw = str(value or "")
    if "/" in raw:
        numerator, denominator = raw.split("/", 1)
        den = _number(denominator)
        num = _number(numerator)
        return round(num / den, 6) if num is not None and den else None
    result = _number(raw)
    return round(result, 6) if result is not None else None


def _rotation(stream: Dict[str, Any]) -> int:
    tag_value = (stream.get("tags") or {}).get("rotate")
    if tag_value is not None:
        return int(round(float(tag_value))) % 360
    for item in stream.get("side_data_list") or []:
        if item.get("rotation") is not None:
            return int(round(float(item["rotation"]))) % 360
    return 0


def probe_media(path: Path, *, timeout_seconds: float = 30.0) -> Dict[str, Any]:
    started = time.perf_counter()
    command = [
        "ffprobe", "-v", "error", "-show_error", "-show_format", "-show_streams",
        "-count_frames", "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise MediaProbeError(f"服务端无法运行媒体探测：{exc}") from exc
    try:
        raw = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProbeError("媒体探测没有返回可读结果") from exc
    streams = raw.get("streams") or []
    if result.returncode != 0 or raw.get("error") or not streams:
        detail = (result.stderr or (raw.get("error") or {}).get("string") or "文件没有可读取的媒体流").strip()
        raise MediaProbeError(f"文件损坏或无法读取：{detail}")
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    image_like = bool(video and (video.get("nb_frames") == "1" or video.get("codec_name") in {"png", "mjpeg", "webp", "gif", "tiff"}) and not audios)
    kind = "image" if image_like else ("video" if video else "audio")
    format_info = raw.get("format") or {}
    duration = _number(format_info.get("duration"))
    if duration is None and video:
        duration = _number(video.get("duration"))
    video_facts = None
    if video:
        avg_fps = _rate(video.get("avg_frame_rate"))
        nominal_fps = _rate(video.get("r_frame_rate"))
        frame_count = video.get("nb_read_frames") or video.get("nb_frames")
        video_facts = {
            "codec": video.get("codec_name"), "profile": video.get("profile"),
            "pixelFormat": video.get("pix_fmt"), "width": video.get("width"), "height": video.get("height"),
            "sampleAspectRatio": video.get("sample_aspect_ratio"), "displayAspectRatio": video.get("display_aspect_ratio"),
            "avgFps": avg_fps, "nominalFps": nominal_fps,
            "variableFrameRate": bool(avg_fps and nominal_fps and abs(avg_fps - nominal_fps) > 0.001),
            "frameCount": int(frame_count) if str(frame_count or "").isdigit() else None,
            "durationSeconds": _number(video.get("duration")) or duration, "rotationDegrees": _rotation(video),
            "colorSpace": video.get("color_space"), "colorTransfer": video.get("color_transfer"), "colorPrimaries": video.get("color_primaries"),
        }
    audio_facts = [{
        "streamIndex": item.get("index"), "codec": item.get("codec_name"), "profile": item.get("profile"),
        "sampleFormat": item.get("sample_fmt"), "sampleRate": int(item["sample_rate"]) if str(item.get("sample_rate") or "").isdigit() else None,
        "channels": item.get("channels"), "channelLayout": item.get("channel_layout"),
        "channelLayoutKnown": bool(item.get("channel_layout")),
        "durationSeconds": _number(item.get("duration")) or duration,
    } for item in audios]
    return {
        "probeVersion": "ffprobe-media-facts-v1", "kind": kind,
        "formatName": format_info.get("format_name"), "formatLongName": format_info.get("format_long_name"),
        "durationSeconds": duration, "bitRate": int(format_info["bit_rate"]) if str(format_info.get("bit_rate") or "").isdigit() else None,
        "video": video_facts, "audioStreams": audio_facts, "hasAudio": bool(audio_facts),
        "probeElapsedMs": round((time.perf_counter() - started) * 1000, 3),
    }
