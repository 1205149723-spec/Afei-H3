"""Fail-closed capture and comparison for completed full-quality GPU tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from full_quality_contract import full_quality_contract, full_quality_fingerprint


BASELINE_SCHEMA_VERSION = "h3-performance-baseline-v1"
COMPARISON_FIELDS = (
    "hardwareIdentitySha256",
    "modelIdentitySha256",
    "computeBackendIdentitySha256",
    "routeFingerprint",
    "mode",
    "width",
    "height",
    "durationSeconds",
    "fps",
    "plannedFrames",
    "modelFrames",
    "exportFrames",
    "steps",
    "sampler",
    "scheduler",
    "seed",
    "assetIdentitySha256",
    "promptIdentitySha256",
)
REGRESSION_THRESHOLDS = {
    "status": "pending_owner_approval",
    "wallClockPercent": None,
    "primarySamplingPercent": None,
    "gpuMemoryUsedMiBPercent": None,
    "torchPeakAllocatedPercent": None,
    "workingSetPeakPercent": None,
    "pageFaultsPercent": None,
    "policyUntilApproved": "report_only_pending_owner_thresholds",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unavailable(reason: str) -> Dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _measurement(value: Any, *, reason: str) -> Dict[str, Any]:
    if value is None or isinstance(value, bool):
        return _unavailable(reason)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _unavailable(reason)
    return {"status": "available", "value": number}


def _elapsed(stage_timings: Dict[str, Any], name: str, reason: str = "stage_timing_missing") -> Dict[str, Any]:
    stage = stage_timings.get(name)
    return _measurement(stage.get("elapsedSeconds") if isinstance(stage, dict) else None, reason=reason)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return None


def _wall_clock(task: Dict[str, Any]) -> Dict[str, Any]:
    timing = task.get("timing") if isinstance(task.get("timing"), dict) else {}
    explicit = timing.get("totalElapsedSeconds")
    if explicit is not None:
        return _measurement(explicit, reason="wall_clock_missing")
    started = _parse_timestamp(task.get("acceptedAt") or timing.get("acceptedAt") or task.get("createdAt"))
    ended = _parse_timestamp(task.get("completedAt") or task.get("terminalAt") or timing.get("completedAt"))
    if started is None or ended is None:
        return _unavailable("accepted_or_terminal_timestamp_missing")
    return {"status": "available", "value": round((ended - started).total_seconds(), 6)}


def probe_mp4(path: Path) -> Dict[str, Any]:
    project_ffprobe = Path(__file__).resolve().parents[1] / "runtime" / "ffmpeg" / "bin" / "ffprobe.exe"
    ffprobe = project_ffprobe if project_ffprobe.is_file() else shutil.which("ffprobe")
    if ffprobe:
        try:
            completed = subprocess.run(
                [
                    str(ffprobe), "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
                    "-of", "json", str(path),
                ],
                capture_output=True, text=True, timeout=15.0, check=False,
            )
            payload = json.loads(completed.stdout or "{}")
            stream = next(iter(payload.get("streams") or []), None)
            if completed.returncode == 0 and isinstance(stream, dict):
                numerator, denominator = str(stream.get("avg_frame_rate") or "0/1").split("/", 1)
                return {
                    "decodable": True,
                    "probe": "ffprobe",
                    "width": int(stream["width"]),
                    "height": int(stream["height"]),
                    "fps": float(numerator) / float(denominator),
                    "frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in {None, "N/A"} else None,
                    "durationSeconds": float(stream["duration"]) if stream.get("duration") not in {None, "N/A"} else None,
                }
        except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
    try:
        import av

        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return {"decodable": False, "reason": "video_stream_missing"}
            decoded = next(container.decode(stream), None)
            if decoded is None:
                return {"decodable": False, "reason": "video_frame_missing"}
            fps = float(stream.average_rate) if stream.average_rate else None
            frames = int(stream.frames or 0) or None
            duration = float(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base else None
            return {
                "decodable": True,
                "width": int(decoded.width),
                "height": int(decoded.height),
                "fps": fps,
                "frames": frames,
                "durationSeconds": duration,
            }
    except Exception as exc:
        return {"decodable": False, "reason": f"{type(exc).__name__}: {exc}"}


def _resolve_output_path(project_root: Path, value: Any) -> Optional[Path]:
    if not value:
        return None
    candidate = Path(str(value))
    candidate = candidate if candidate.is_absolute() else project_root / candidate
    try:
        resolved = candidate.resolve()
        resolved.relative_to((project_root / "output").resolve())
        return resolved
    except (OSError, ValueError):
        return None


def _asset_identity(compiled: Dict[str, Any]) -> tuple[list[str], str]:
    hashes = []
    for reference in compiled.get("references") or []:
        if not isinstance(reference, dict):
            continue
        value = reference.get("sha256") or reference.get("contentSha256") or reference.get("assetSha256")
        if value:
            hashes.append(str(value))
    hashes.sort()
    return hashes, _canonical_hash(hashes)


def _model_identity(route_contract: Dict[str, Any]) -> str:
    models = {"primaryModel": route_contract.get("primaryModel"), "sharedModels": route_contract.get("sharedModels")}
    return _canonical_hash(models)


def _compute_backend_identity(result: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    receipt = result.get("executionReceipt") if isinstance(result.get("executionReceipt"), dict) else {}
    requested_kernel = receipt.get("requestedKernel")
    if requested_kernel == "official_native":
        expected = {
            "requestedKernel": "official_native",
            "actualKernel": "official_native",
            "kernelIdentity": "comfyui_official_native_attention",
            "patchedBlocks": 0,
            "scope": "official_native",
            "ffnEnabled": False,
            "fallback": False,
        }
        observed = {field: receipt.get(field) for field in expected}
        if observed != expected:
            return None, "runtime_compute_backend_mismatch"
        return _canonical_hash(observed), None
    sage = result.get("sage") if isinstance(result.get("sage"), dict) else {}
    selected = sage.get("selected") or result.get("attentionBackend")
    h3 = sage.get("h3MemoryEfficientSage") if isinstance(sage.get("h3MemoryEfficientSage"), dict) else {}
    hook = sage.get("hookReceipt") if isinstance(sage.get("hookReceipt"), dict) else {}
    if selected not in {"sage", "torch_sdpa"}:
        return None, "runtime_execution_identity_incomplete"
    if h3.get("available") is not True or h3.get("sagePackageVersion") != "2.2.0" or h3.get("sm120Verified") is not True:
        return None, "runtime_execution_identity_incomplete"
    if selected == "sage" and (
        sage.get("hookImplementation") != "kijai_kj_minimax_h3_memory_efficient_sage"
        or hook.get("backend") != "kijai_kj_minimax_h3_memory_efficient_sage"
        or hook.get("patchedBlocks") != 50
        or hook.get("scope") != "denoiser_only"
    ):
        return None, "runtime_compute_backend_mismatch"
    identity = {
        "selected": selected,
        "implementation": sage.get("hookImplementation"),
        "backend": hook.get("backend"),
        "patchedBlocks": hook.get("patchedBlocks"),
        "scope": hook.get("scope"),
        "author": h3.get("author"),
        "sourceCommit": h3.get("sourceCommit"),
        "license": h3.get("license"),
        "sageModulePathVerified": h3.get("sageModulePathVerified"),
        "sagePackageVersion": h3.get("sagePackageVersion"),
        "sm120Verified": h3.get("sm120Verified"),
    }
    return _canonical_hash(identity), None


def _input_scale(task: Dict[str, Any], receipt: Dict[str, Any]) -> Dict[str, Any]:
    plan = task.get("plan") if isinstance(task.get("plan"), dict) else {}
    compiled = plan.get("compiled") if isinstance(plan.get("compiled"), dict) else {}
    timing = compiled.get("timing") if isinstance(compiled.get("timing"), dict) else {}
    canvas = compiled.get("canvas") if isinstance(compiled.get("canvas"), dict) else {}
    internal = canvas.get("internal") if isinstance(canvas.get("internal"), dict) else {}
    effective = receipt.get("effectiveResolution") if isinstance(receipt.get("effectiveResolution"), dict) else {}
    execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
    prompt = receipt.get("promptContract") if isinstance(receipt.get("promptContract"), dict) else {}
    route_contract = receipt.get("routeContract") if isinstance(receipt.get("routeContract"), dict) else {}
    assets, asset_identity = _asset_identity(compiled)
    duration = timing.get("durationSeconds")
    if duration is None:
        duration = timing.get("exportDurationSeconds")
    return {
        "mode": compiled.get("mode") or route_contract.get("mode"),
        "width": effective.get("width") or internal.get("width"),
        "height": effective.get("height") or internal.get("height"),
        "durationSeconds": duration,
        "fps": timing.get("fps") or receipt.get("fps"),
        "plannedFrames": timing.get("frameCount"),
        "modelFrames": receipt.get("decodedFrameCount") or timing.get("frameCount"),
        "exportFrames": receipt.get("exportedFrameCount") or timing.get("exportFrameCount"),
        "steps": receipt.get("steps") or execution.get("steps"),
        "sampler": receipt.get("sampler"),
        "scheduler": receipt.get("scheduler"),
        "seed": execution.get("seed"),
        "assetCount": len(assets),
        "assetHashes": assets,
        "assetIdentitySha256": asset_identity,
        "promptIdentitySha256": prompt.get("compiledPromptSha256") or prompt.get("rawPromptSha256"),
        "routeFingerprint": receipt.get("algorithmRouteFingerprint"),
        "modelIdentitySha256": _model_identity(route_contract),
    }


def _timings(task: Dict[str, Any], receipt: Dict[str, Any]) -> Dict[str, Any]:
    stages = receipt.get("stageTimings") if isinstance(receipt.get("stageTimings"), dict) else {}
    known = {
        "wallClock": _wall_clock(task),
        "bootstrap": _elapsed(stages, "bootstrap"),
        "primaryModelLoad": _elapsed(stages, "primaryModelLoad"),
        "textEncoderLoad": _elapsed(stages, "textEncoderLoad"),
        "videoVaeLoad": _elapsed(stages, "videoVaeLoad"),
        "audioVaeLoad": _elapsed(stages, "audioVaeLoad"),
        "conditioning": _elapsed(stages, "referenceConditioning"),
        "primarySampling": _elapsed(stages, "primarySampling"),
        "videoDecode": _elapsed(stages, "videoDecode"),
        "audioDecode": _elapsed(stages, "audioDecode"),
        "mux": _elapsed(stages, "mux"),
    }
    available_sum = sum(item["value"] for key, item in known.items() if key != "wallClock" and item.get("status") == "available")
    wall = known["wallClock"]
    known["other"] = (
        {"status": "available", "value": round(max(0.0, wall["value"] - available_sum), 6), "derivation": "wallClock-minus-observed-stages"}
        if wall.get("status") == "available"
        else _unavailable("wall_clock_missing")
    )
    return known


def _telemetry(receipt: Dict[str, Any]) -> Dict[str, Any]:
    telemetry = receipt.get("resourceTelemetry") if isinstance(receipt.get("resourceTelemetry"), dict) else {}
    sample = telemetry.get("primarySamplingComplete") if isinstance(telemetry.get("primarySamplingComplete"), dict) else {}
    cuda = sample.get("cuda") if isinstance(sample.get("cuda"), dict) else {}
    process = sample.get("process") if isinstance(sample.get("process"), dict) else {}
    external = sample.get("externalUtilization") if isinstance(sample.get("externalUtilization"), dict) else {}
    gpu = external.get("gpu") if isinstance(external.get("gpu"), dict) and external.get("gpu", {}).get("available") is True else {}
    reason = "worker_telemetry_missing"
    return {
        "gpuUtilizationPercent": _measurement(gpu.get("utilizationPercent"), reason=reason),
        "gpuMemoryUsedMiB": _measurement(gpu.get("memoryUsedMiB"), reason=reason),
        "torchPeakAllocatedBytes": _measurement(cuda.get("peakAllocatedBytes"), reason=reason),
        "torchPeakReservedBytes": _measurement(cuda.get("peakReservedBytes"), reason=reason),
        "workingSetBytes": _measurement(process.get("workingSetBytes"), reason=reason),
        "peakWorkingSetBytes": _measurement(process.get("peakWorkingSetBytes"), reason=reason),
        "pageFaultCount": _measurement(process.get("pageFaultCount"), reason=reason),
        "pageFaultsDuringScope": _measurement(sample.get("pageFaultsSinceScopeStart"), reason=reason),
    }


def _rejected(result: Dict[str, Any], reason: str, input_scale: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(result)
    receipt = updated.setdefault("executionReceipt", {})
    receipt["performanceBaseline"] = {
        "schemaVersion": BASELINE_SCHEMA_VERSION,
        "status": "rejected",
        "reason": reason,
        "comparable": False,
        "inputScale": input_scale,
    }
    return updated


def _load_store(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schemaVersion": BASELINE_SCHEMA_VERSION, "records": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("baseline store is unreadable") from exc
    if not isinstance(value, dict) or value.get("schemaVersion") != BASELINE_SCHEMA_VERSION or not isinstance(value.get("records"), list):
        raise ValueError("baseline store schema is invalid")
    return value


def _write_store(path: Path, store: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _metric_value(record: Dict[str, Any], section: str, name: str) -> Optional[float]:
    item = (record.get(section) or {}).get(name) or {}
    return float(item["value"]) if item.get("status") == "available" else None


def _comparison(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {}
    for section, name in (("timings", "wallClock"), ("timings", "primarySampling"), ("telemetry", "gpuMemoryUsedMiB"), ("telemetry", "torchPeakAllocatedBytes"), ("telemetry", "peakWorkingSetBytes"), ("telemetry", "pageFaultsDuringScope")):
        before, after = _metric_value(reference, section, name), _metric_value(candidate, section, name)
        key = f"{section}.{name}"
        metrics[key] = _unavailable("reference_or_candidate_metric_missing") if before is None or after is None else {
            "status": "available", "reference": before, "candidate": after,
            "delta": round(after - before, 6),
            "deltaPercent": round((after - before) * 100.0 / before, 6) if before else None,
        }
    return {"policy": "report_only_pending_owner_thresholds", "verdict": "difference_only_no_threshold_approved", "metrics": metrics}


def finalize_performance_baseline(
    task: Dict[str, Any],
    result: Dict[str, Any],
    *,
    worker_exit_code: Any,
    project_root: Path,
    baseline_store_path: Optional[Path] = None,
    media_probe: Callable[[Path], Dict[str, Any]] = probe_mp4,
) -> Dict[str, Any]:
    """Return a result whose baseline may advance only after terminal evidence."""

    updated = copy.deepcopy(result)
    receipt = updated.setdefault("executionReceipt", {})
    scale = _input_scale(task, receipt)
    if task.get("state") not in {"completed", "running", "loading"} or updated.get("status") != "completed":
        return _rejected(updated, "task_not_completed", scale)
    if updated.get("outputAuthentic") is not True or receipt.get("outputAuthentic") is not True:
        return _rejected(updated, "output_not_authentic", scale)
    if worker_exit_code != 0:
        return _rejected(updated, "worker_exit_nonzero", scale)
    if receipt.get("algorithmRoute") != "full_quality":
        return _rejected(updated, "route_not_full_quality", scale)
    route_contract = receipt.get("routeContract") if isinstance(receipt.get("routeContract"), dict) else {}
    mode = str(scale.get("mode") or "").upper()
    requested_kernel = str(receipt.get("requestedKernel") or "kijai_fast")
    try:
        fingerprint_matches = (
            receipt.get("algorithmRouteFingerprint") == full_quality_fingerprint(mode, requested_kernel)
            and route_contract == full_quality_contract(mode, requested_kernel)
        )
    except ValueError:
        fingerprint_matches = False
    if route_contract.get("routeId") != "full_quality" or not fingerprint_matches:
        return _rejected(updated, "full_quality_fingerprint_mismatch", scale)
    compute_identity, compute_error = _compute_backend_identity(updated)
    scale["computeBackendIdentitySha256"] = compute_identity
    if compute_error:
        return _rejected(updated, compute_error, scale)
    required_scale = ("mode", "width", "height", "durationSeconds", "fps", "plannedFrames", "modelFrames", "exportFrames", "steps", "sampler", "scheduler", "seed", "promptIdentitySha256", "routeFingerprint", "modelIdentitySha256", "computeBackendIdentitySha256")
    if any(scale.get(field) is None for field in required_scale):
        return _rejected(updated, "input_scale_incomplete", scale)
    output_path = _resolve_output_path(Path(project_root), updated.get("outputPath") or receipt.get("outputPath"))
    if output_path is None or not output_path.is_file():
        return _rejected(updated, "mp4_missing", scale)
    media = media_probe(output_path)
    if media.get("decodable") is not True:
        rejected = _rejected(updated, "mp4_not_decodable", scale)
        rejected["executionReceipt"]["performanceBaseline"]["mediaProbe"] = media
        return rejected
    if (int(media.get("width") or 0), int(media.get("height") or 0)) != (int(scale["width"]), int(scale["height"])):
        return _rejected(updated, "mp4_scale_mismatch", scale)
    hardware = receipt.get("hardwareIdentity") if isinstance(receipt.get("hardwareIdentity"), dict) else {}
    hardware_required = ("gpuUuid", "gpuName", "driverVersion", "torchVersion", "cudaVersion")
    if hardware.get("status") != "available" or any(not hardware.get(field) for field in hardware_required):
        return _rejected(updated, "hardware_identity_incomplete", scale)

    identity = {**{field: scale.get(field) for field in COMPARISON_FIELDS if field != "hardwareIdentitySha256"}, "hardwareIdentitySha256": _canonical_hash(hardware)}
    record = {
        "schemaVersion": BASELINE_SCHEMA_VERSION,
        "taskId": str(task.get("id") or ""),
        "capturedAt": str(task.get("completedAt") or task.get("terminalAt") or task.get("updatedAt") or "terminal_pending_persistence"),
        "status": "ready",
        "inputScale": scale,
        "identity": identity,
        "hardwareIdentity": hardware,
        "timings": _timings(task, receipt),
        "telemetry": _telemetry(receipt),
        "mediaProbe": media,
        "regressionThresholds": copy.deepcopy(REGRESSION_THRESHOLDS),
    }
    store_path = baseline_store_path or (Path(project_root) / "cache" / "performance_baselines" / "full_quality.json")
    try:
        store = _load_store(store_path)
    except ValueError:
        return _rejected(updated, "baseline_store_unreadable", scale)
    records = store.setdefault("records", [])
    reference = next((item for item in records if item.get("role") == "referenceBaseline"), None)
    if reference is None:
        record.update({"role": "referenceBaseline", "comparable": False, "comparisonReason": "reference_baseline_created"})
        store["referenceBaselineTaskId"] = record["taskId"]
    else:
        mismatches = [field for field in COMPARISON_FIELDS if reference.get("identity", {}).get(field) != identity.get(field)]
        record.update({
            "role": "candidateBaseline",
            "referenceBaselineTaskId": reference.get("taskId"),
            "comparable": not mismatches,
            "comparisonReason": "identity_match" if not mismatches else "identity_mismatch",
            "identityMismatchFields": mismatches,
        })
        if not mismatches:
            record["comparison"] = _comparison(reference, record)
    records.append(record)
    try:
        _write_store(store_path, store)
    except (OSError, TypeError, ValueError):
        return _rejected(updated, "baseline_store_write_failed", scale)
    receipt["performanceBaseline"] = copy.deepcopy(record)
    updated["executionReceipt"] = receipt
    return updated
