"""Isolated SwiftVR GPU worker. Run only with runtime/swiftvr_env/python.exe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


CODE_COMMIT = "5ca168cef6ca7200f135fdfea85e5e13d12c5b53"
MODEL_REVISION = "743ed2530c550764905400f38eb6cc41af5abc80"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def update(manifest, task, state, progress, message):
    task.update({"state": state, "progress": progress, "message": message, "updatedAt": now()})
    atomic_write(manifest, task)


def probe(ffprobe, path):
    command = [
        ffprobe, "-v", "error", "-count_frames", "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    payload = json.loads(result.stdout)
    video = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), None)
    if not video:
        raise RuntimeError("MP4 has no decodable video stream")
    numerator, denominator = (video.get("avg_frame_rate") or "0/1").split("/", 1)
    fps = float(numerator) / float(denominator or 1)
    frames = int(video.get("nb_read_frames") or video.get("nb_frames") or 0)
    audio = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "audio"), None)
    return {
        "decodable": True,
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "fps": fps,
        "frames": frames,
        "durationSeconds": float((payload.get("format") or {}).get("duration") or video.get("duration") or 0),
        "videoCodec": video.get("codec_name"),
        "audioCodec": audio.get("codec_name") if audio else None,
        "hasAudio": bool(audio),
    }


def run_checked(command):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command[:6])}\n{result.stderr[-4000:]}")
    return result


def make_protocol_input(ffmpeg, source, folder, source_frames):
    folder.mkdir(parents=True, exist_ok=True)
    run_checked([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0", "-vsync", "0", str(folder / "%06d.png")])
    frames = sorted(folder.glob("*.png"))
    if len(frames) != source_frames:
        raise RuntimeError(f"decoded frame count changed: expected {source_frames}, got {len(frames)}")
    target_count = 4 * ((source_frames - 1 + 3) // 4) + 1
    padding = target_count - source_frames
    for index in range(padding):
        shutil.copyfile(frames[-1], folder / f"{source_frames + index + 1:06d}.png")
    return padding, target_count


def mux_from_source(ffmpeg, silent, source, output, frames, fps, has_audio):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(source), "-map", "0:v:0"]
    audio_policy = {"present": has_audio, "mode": "none", "codec": None, "reason": None, "parameters": None}
    if has_audio:
        command += ["-map", "1:a:0?", "-c:a", "copy"]
        audio_policy.update({"mode": "stream_copy", "codec": "copy"})
    command += ["-c:v", "copy", "-frames:v", str(frames), "-t", f"{frames / fps:.9f}", "-movflags", "+faststart", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode and has_audio:
        audio_policy.update({
            "mode": "transcode",
            "codec": "aac",
            "reason": "source audio stream was not MP4-copy-compatible",
            "parameters": {"codec": "aac", "bitrate": "192k"},
        })
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-frames:v", str(frames), "-t", f"{frames / fps:.9f}", "-movflags", "+faststart", str(output)]
        run_checked(command)
    elif result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return audio_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    task = json.loads(manifest.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    receipt = task["receipt"]
    started = time.perf_counter()
    stage_started = started

    def stage(state, progress, message):
        nonlocal stage_started
        current = time.perf_counter()
        previous = task.get("state")
        if previous in {"preflight", "loading", "restoring", "encoding"}:
            receipt["stageTimingsSeconds"][previous] = round(current - stage_started, 6)
        stage_started = current
        update(manifest, task, state, progress, message)

    try:
        stage("preflight", 4, "正在复核原片、模型和独立运行环境")
        if receipt.get("swiftvrCodeCommit") != CODE_COMMIT or receipt.get("swiftvrModelRevision") != MODEL_REVISION:
            raise RuntimeError("SwiftVR identity mismatch")
        source = (root / task["sourcePath"]).resolve()
        output = (root / task["outputPath"]).resolve()
        source.relative_to((root / "output").resolve())
        output.relative_to((root / "output").resolve())
        actual_source_hash = sha256_file(source)
        if actual_source_hash != task["sourceMp4Sha256"]:
            raise RuntimeError("source MP4 SHA256 changed before worker open")

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError("ffmpeg/ffprobe unavailable; no alternate tool is permitted")
        source_probe = probe(ffprobe, source)
        if not source_probe["decodable"] or source_probe["frames"] <= 0 or source_probe["fps"] <= 0:
            raise RuntimeError("source MP4 decode evidence is incomplete")
        receipt["actualInput"] = source_probe

        models = root / "models" / "SwiftVR"
        for relative, expected in receipt["modelFilesSha256"].items():
            path = models / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"SwiftVR model identity mismatch: {relative}")

        source_tree = root / "staging" / "swiftvr" / "source"
        if not source_tree.is_dir():
            raise RuntimeError("SwiftVR source tree missing")
        sys.path.insert(0, str(source_tree))
        import torch
        from swiftvr import SwiftVRPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
        if torch.__version__ != "2.10.0+cu128":
            raise RuntimeError(f"isolated torch mismatch: {torch.__version__}")
        receipt["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
        torch.cuda.reset_peak_memory_stats()

        task_work = root / "temp" / "swiftvr_work" / task["id"]
        input_frames = task_work / "input_frames"
        silent = task_work / "restored_silent.mp4"
        task_work.mkdir(parents=True, exist_ok=True)
        padding, protocol_frames = make_protocol_input(ffmpeg, source, input_frames, source_probe["frames"])
        receipt["causalProtocolAdapter"] = {"sourceFrames": source_probe["frames"], "tailPaddingFrames": padding, "protocolInputFrames": protocol_frames, "trimmedOutputFrames": padding}

        stage("loading", 12, "正在隔离进程加载 SwiftVR 官方模型")
        pipe = SwiftVRPipeline.from_pretrained(models).to("cuda", dtype="bfloat16", attention_backend="sdpa", torch_compile=False)
        stage("restoring", 20, "正在进行 SwiftVR 高清修复")
        target = task["targetResolution"]
        clip_len = 4 if target["id"] == "2k" else 24
        receipt["clipLen"] = clip_len
        receipt["queueSize"] = 1
        stats = pipe.restore_video(
            input_frames,
            silent,
            resolution=(target["width"], target["height"]),
            clip_len=clip_len,
            dit_overlap=0,
            fps=source_probe["fps"],
            quality=85,
            png_save=False,
            queue_size=1,
            verbose=True,
        )
        receipt["swiftvrStats"] = stats
        stage("encoding", 92, "正在继承原片音频并封装 MP4")
        audio = mux_from_source(ffmpeg, silent, source, output, source_probe["frames"], source_probe["fps"], source_probe["hasAudio"])
        output_probe = probe(ffprobe, output)
        if output_probe["width"] != target["width"] or output_probe["height"] != target["height"]:
            raise RuntimeError("output resolution mismatch")
        if output_probe["frames"] != source_probe["frames"]:
            raise RuntimeError("output frame count mismatch")
        if abs(output_probe["fps"] - source_probe["fps"]) > 0.01:
            raise RuntimeError("output FPS mismatch")
        if abs(output_probe["durationSeconds"] - source_probe["durationSeconds"]) > max(0.08, 1.5 / source_probe["fps"]):
            raise RuntimeError("audio/video duration mismatch")
        if sha256_file(source) != task["sourceMp4Sha256"]:
            raise RuntimeError("source MP4 changed during restoration")

        receipt["audio"] = audio
        receipt["actualOutput"] = output_probe
        receipt["outputMp4Sha256"] = sha256_file(output)
        receipt["sourceMp4Sha256After"] = sha256_file(source)
        receipt["peakGpuMemoryBytes"] = int(torch.cuda.max_memory_allocated())
        receipt["stageTimingsSeconds"]["encoding"] = round(time.perf_counter() - stage_started, 6)
        receipt["stageTimingsSeconds"]["total"] = round(time.perf_counter() - started, 6)
        receipt["workerExit"] = {"code": 0, "reason": "completed"}
        receipt["outputAuthentic"] = True
        task.update({"state": "completed", "progress": 100, "message": "高清修复完成，原始视频已保留", "completedAt": now(), "terminalAt": now(), "outputAuthentic": True})
        atomic_write(manifest, task)
    except BaseException as exc:
        receipt["stageTimingsSeconds"][str(task.get("state") or "preflight")] = round(time.perf_counter() - stage_started, 6)
        receipt["stageTimingsSeconds"]["total"] = round(time.perf_counter() - started, 6)
        receipt["workerExit"] = {"code": 1, "reason": str(exc), "traceback": traceback.format_exc()[-8000:]}
        receipt["outputAuthentic"] = False
        task.update({"state": "failed", "message": str(exc), "failedAt": now(), "terminalAt": now(), "outputAuthentic": False})
        atomic_write(manifest, task)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
