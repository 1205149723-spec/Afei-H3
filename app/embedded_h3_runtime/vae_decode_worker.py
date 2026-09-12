"""Isolated direct H3 VAE decoder; no server, graph, or workflow executor."""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(1, str(ROOT / "runtime" / "python_packages"))
sys.path.insert(2, str(ROOT / "runtime" / "ComfyUI"))

from embedded_h3_runtime import EmbeddedH3Runtime
from embedded_h3_runtime.runner import DirectH3Runner, _H3AudioVAEProxy, _H3VideoVAEProxy, _shape


def _write_result(path: Path, result: Dict[str, Any]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def main(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result_path = Path(manifest["resultPath"]).resolve()
    result_path.relative_to(ROOT.resolve())
    runtime = EmbeddedH3Runtime(root=ROOT)
    runtime_stages: Dict[str, Any] = {}
    started = time.perf_counter()
    torch = None
    restore_cudnn = None
    try:
        objects = runtime.objects()
        torch = objects["modules"]["torch"]
        tensors = torch.load(manifest["latentPath"], map_location="cpu", weights_only=False)
        video_latent = tensors["video"]
        audio_latent = tensors["audio"]
        tensors = None
        runtime_stages["video_vae_load"] = runtime.load_video_vae()
        objects = runtime.objects()
        modules = objects["modules"]
        model_management = modules["model_management"]
        video_vae = _H3VideoVAEProxy(objects["videoVae"], objects.get("videoVaePatcher"), model_management)
        cudnn_sdp_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
        enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
        if torch.cuda.is_available() and callable(cudnn_sdp_enabled) and callable(enable_cudnn_sdp):
            previous_cudnn = bool(cudnn_sdp_enabled())
            if previous_cudnn:
                enable_cudnn_sdp(False)
                restore_cudnn = lambda: enable_cudnn_sdp(previous_cudnn)
        runtime_stages["attention_backend"] = {
            "scope": "isolated_video_audio_vae_decode",
            "backend": "pytorch_sdpa_without_cudnn_mha",
        }
        video_started = time.perf_counter()
        with torch.inference_mode():
            video = video_vae.decode(video_latent).detach().cpu()
        runtime_stages["video_vae_decode"] = {
            "elapsedSeconds": round(time.perf_counter() - video_started, 6),
            "outputShape": _shape(video),
            "outputDevice": str(video.device),
        }
        video_latent = None
        video_vae = None
        objects = None
        gc.collect()
        runtime_stages["video_vae_unload"] = runtime.unload_video_vae()
        runtime_stages["audio_vae_load"] = runtime.load_audio_vae()
        objects = runtime.objects()
        audio_vae = _H3AudioVAEProxy(objects["audioVae"], objects.get("audioVaePatcher"), model_management)
        audio_started = time.perf_counter()
        with torch.inference_mode():
            audio = audio_vae.decode(audio_latent).detach().cpu()
        runtime_stages["audio_vae_decode"] = {
            "elapsedSeconds": round(time.perf_counter() - audio_started, 6),
            "outputShape": _shape(audio),
            "outputDevice": str(audio.device),
        }
        audio_latent = None
        output_path = DirectH3Runner(runtime=runtime)._export_mp4(
            video,
            audio,
            str(manifest["taskId"]),
            manifest["compiled"],
        )
        runtime_stages["export"] = {
            "outputPath": str(output_path),
            "elapsedSeconds": round(time.perf_counter() - started, 6),
        }
        result = {
            "status": "completed",
            "realInference": True,
            "outputAuthentic": True,
            "outputPath": str(output_path),
            "runtimeStages": runtime_stages,
            "elapsedSeconds": round(time.perf_counter() - started, 6),
        }
        _write_result(result_path, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        result = {
            "status": "error",
            "realInference": True,
            "outputAuthentic": False,
            "error": f"{type(exc).__name__}: {exc}",
            "runtimeStages": runtime_stages,
            "elapsedSeconds": round(time.perf_counter() - started, 6),
        }
        _write_result(result_path, result)
        print(json.dumps(result, ensure_ascii=False))
        return 2
    finally:
        if restore_cudnn is not None:
            restore_cudnn()
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]).resolve()))
