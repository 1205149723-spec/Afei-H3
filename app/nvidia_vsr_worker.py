"""Isolated NVIDIA VideoSuperRes ULTRA worker; failures never fall back."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

def now(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def atomic(path, payload):
    path = Path(path); tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(c)
    return h.hexdigest()
def probe(ffprobe, path):
    p = subprocess.run([ffprobe, "-v", "error", "-count_frames", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames,nb_read_frames,codec_name", "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8", check=True)
    d = json.loads(p.stdout); v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v: raise RuntimeError("MP4 has no decodable video stream")
    n, den = (v.get("avg_frame_rate") or "0/1").split("/", 1); fps = float(n) / float(den or 1); frames = int(v.get("nb_read_frames") or v.get("nb_frames") or 0)
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    return {"decodable": True, "width": int(v.get("width") or 0), "height": int(v.get("height") or 0), "fps": fps, "frames": frames, "durationSeconds": float((d.get("format") or {}).get("duration") or 0), "hasAudio": bool(a), "audioCodec": a.get("codec_name") if a else None}
def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode: raise RuntimeError(p.stderr[-4000:] or f"command failed: {p.returncode}")
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); args = ap.parse_args(); manifest = Path(args.manifest).resolve(); task = json.loads(manifest.read_text(encoding="utf-8")); receipt = task["receipt"]; started = time.perf_counter()
    def update(state, progress, message): task.update(state=state, progress=progress, message=message, updatedAt=now()); atomic(manifest, task)
    try:
        root = Path.cwd().resolve(); source = (root / task["sourcePath"]).resolve(); output = (root / task["outputPath"]).resolve(); output_root = (root / "output").resolve(); source.relative_to(output_root); output.relative_to(output_root)
        if sha(source) != task["sourceMp4Sha256"]: raise RuntimeError("source MP4 SHA256 changed")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe: raise RuntimeError("ffmpeg/ffprobe unavailable; no fallback permitted")
        update("decoding", 5, "正在解码原片帧"); src = probe(ffprobe, source); receipt["actualInput"] = src
        if src["frames"] <= 0 or src["fps"] <= 0: raise RuntimeError("source media evidence incomplete")
        work = root / "temp" / "nvidia_vsr_work" / task["id"]; inp, outp = work / "input", work / "output"; inp.mkdir(parents=True, exist_ok=True); outp.mkdir(parents=True, exist_ok=True)
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source), "-map", "0:v:0", "-vsync", "0", str(inp / "%08d.png")]); frames = sorted(inp.glob("*.png"))
        if len(frames) != src["frames"]: raise RuntimeError(f"decoded frame count mismatch: {len(frames)} != {src['frames']}")
        update("loading", 10, "正在加载 NVIDIA VideoSuperRes ULTRA"); sys.path.insert(0, str(root / "_private_site")); import torch; from nvvfx import VideoSuperRes
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; CPU fallback forbidden")
        with VideoSuperRes(quality=VideoSuperRes.QualityLevel.ULTRA, device=0) as vsr:
            vsr.output_width, vsr.output_height = 1920, 1088; vsr.load(); update("processing", 15, "正在执行 NVIDIA VSR ULTRA")
            for i, frame in enumerate(frames, 1):
                image = torch.from_numpy(__import__("numpy").array(__import__("PIL.Image", fromlist=["Image"]).open(frame).convert("RGB"))).permute(2, 0, 1).float().div(255).contiguous().cuda()
                result = torch.from_dlpack(vsr.run(image).image).clone().clamp(0, 1); arr = (result.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8"); __import__("PIL.Image", fromlist=["Image"]).fromarray(arr).save(outp / frame.name); task["progress"] = 15 + int(i * 75 / len(frames)); atomic(manifest, task)
        update("encoding", 92, "正在继承原片音频并封装 MP4"); output.parent.mkdir(parents=True, exist_ok=True); silent = work / "silent.mp4"; run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(src["fps"]), "-i", str(outp / "%08d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-frames:v", str(src["frames"]), str(silent)])
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "copy", "-frames:v", str(src["frames"]), "-t", f"{src['durationSeconds']:.9f}", "-movflags", "+faststart", str(output)]
        try: run(cmd); audio_mode = "stream_copy" if src["hasAudio"] else "none"
        except RuntimeError:
            if not src["hasAudio"]: raise
            fallback = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-frames:v", str(src["frames"]), "-t", f"{src['durationSeconds']:.9f}", "-movflags", "+faststart", str(output)]
            run(fallback); audio_mode = "transcode_aac"
        out = probe(ffprobe, output)
        if (out["width"], out["height"], out["frames"]) != (1920, 1088, src["frames"]): raise RuntimeError("output dimensions/frame count mismatch")
        if abs(out["fps"] - src["fps"]) > .01 or abs(out["durationSeconds"] - src["durationSeconds"]) > max(.08, 1.5 / src["fps"]): raise RuntimeError("output FPS/duration mismatch")
        receipt.update(actualOutput=out, outputMp4Sha256=sha(output), sourceMp4Sha256After=sha(source), audio={"mode": audio_mode}, outputAuthentic=True, workerExit={"code": 0, "reason": "completed"}); task.update(state="completed", progress=100, message="NVIDIA VSR 完成，原始视频已保留", completedAt=now(), terminalAt=now(), outputAuthentic=True); atomic(manifest, task); return 0
    except BaseException as exc:
        receipt.update(outputAuthentic=False, workerExit={"code": 1, "reason": str(exc), "traceback": traceback.format_exc()[-8000:]}); task.update(state="failed", message=str(exc), failedAt=now(), terminalAt=now(), outputAuthentic=False); atomic(manifest, task); return 1
if __name__ == "__main__": raise SystemExit(main())
