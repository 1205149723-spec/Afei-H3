"""Direct H3 task runner.

The runner calls low-level model/conditioning/sampling functions directly.
There is intentionally no prompt graph, node queue, workflow JSON, or server
dependency. Test-only smoke harnesses live outside the product runner.
"""

from __future__ import annotations

import threading
import time
import math
import copy
import gc
import importlib
import json
import hashlib
import logging
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .loader import EmbeddedH3Runtime, H3RuntimeError, ROOT, MODEL_ROOT
from media_staging import StagingError, resolve_staged_path
from h3_compiler import (
    LOW_FPS_TIME_REMAP_ROUTE,
    LOW_FPS_TIME_REMAP_VERSION,
    LATENT_UPSCALE_TWO_STAGE_ROUTE,
    LATENT_UPSCALE_TWO_STAGE_VERSION,
    initial_execution_receipt,
    low_fps_time_remap_route_receipt,
    native_acceleration_route_contract,
    official_h3_sampler_contract,
    two_stage_route_contract,
    KJ_EXPERIMENT_ID,
    KJ_EXPERIMENT_KERNEL,
    KJ_EXPERIMENT_PROFILE,
    KJ_EXPERIMENT_ROUTE,
)
from reference_cache import read_manifest, write_manifest
from sage_policy import choose_sage_policy
from h3_kj_adapter import apply_h3_memory_efficient_sage_patch
from h3_teacache_b2_adapter import apply_h3_teacache_b2_diagnostic, h3_teacache_b2_runtime_receipt
from h3_ffn_chunk_adapter import apply_h3_ffn_chunk_diagnostic
from memory_admission import NATIVE_DYNAMIC_VRAM_SOURCE, NATIVE_MANAGED_DECISION, native_managed_receipt
from full_quality_contract import full_quality_contract, full_quality_fingerprint
from kernel_backend_contract import (
    expected_kernel_backend_contract,
    expected_kernel_backend_receipt,
    verify_kernel_backend_receipt,
)


def _isolated_kj_adapter():
    """Load only the isolated KJ algorithm boundary for the experiment route."""
    isolated_root = ROOT / "_isolated_research"
    if str(isolated_root) not in sys.path:
        sys.path.insert(0, str(isolated_root))
    from native_kj_h3.adapter import apply_h3_sage_patch
    from native_kj_h3.algorithms import SageRuntime
    return apply_h3_sage_patch, SageRuntime


class H3Cancelled(H3RuntimeError):
    """Cancellation observed at a safe CPU-side H3 execution boundary."""


Progress = Callable[[int, str], None]
_QUANT_WARMUP_CACHE: set[tuple[Any, ...]] = set()


def _apply_memory_strategy(model_management: Any, strategy: str) -> Dict[str, Any]:
    """Keep Comfy/DynamicVRAM's native memory policy unchanged."""
    requested = str(strategy or "auto").strip().lower()
    if requested == "auto":
        return {"requested": requested, "applied": False, "restored": False}
    raise H3RuntimeError(f"unsupported memory strategy: {requested}; only native auto is allowed")


def _restore_memory_strategy(model_management: Any, receipt: Optional[Dict[str, Any]]) -> None:
    if not isinstance(receipt, dict) or not receipt.get("applied") or receipt.get("restored"):
        return
    previous = receipt.get("previous") or {}
    if previous.get("vram_state") is not None:
        model_management.vram_state = previous["vram_state"]
    if previous.get("set_vram_to") is not None:
        model_management.set_vram_to = previous["set_vram_to"]
    receipt["restored"] = True


def _require_kijai_h3_contract(sage_receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Fail closed when the full-quality Kijai kernel identity is not exact."""

    h3_sage = sage_receipt.get("h3MemoryEfficientSage") or {}
    required = {
        "available": True,
        "classification": "official_contract_compatible_third_party_optimization",
        "author": "Kijai",
        "sourceCommit": "ab8f90f02ad6ec3a4900b1b4df9c03cded7b4690",
        "license": "GPL-3.0",
        "sagePackageVersion": "2.2.0",
        "sageModulePathVerified": True,
        "sageDistributionPathVerified": True,
        "sageDistributionName": "sageattention",
        "currentDeviceArchitectureVerified": True,
    }
    mismatches = [
        key for key, value in required.items()
        if not (
            key == "sagePackageVersion"
            and str(h3_sage.get(key) or "").startswith("2.2.0")
        )
        and h3_sage.get(key) != value
    ]
    symbols = h3_sage.get("requiredSageSymbols") or {}
    required_symbols = {"get_cuda_arch_versions", "per_thread_int8_triton", "per_warp_int8_cuda", "per_block_int8_triton", "per_channel_fp8"}
    if set(symbols) != required_symbols or any(symbols.get(name) is not True for name in required_symbols):
        mismatches.append("requiredSageSymbols")
    if mismatches:
        reason = h3_sage.get("compatibilityFailure") or "locked Kijai H3 Sage identity mismatch"
        raise H3RuntimeError(f"full_quality Kijai H3 Sage blocked: {reason}; mismatches={mismatches}")
    return h3_sage


def _verify_kijai_h3_hook(receipt: Dict[str, Any]) -> None:
    expected = {
        "backend": "kijai_kj_minimax_h3_memory_efficient_sage",
        "patchedBlocks": 50,
        "scope": "denoiser_only",
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise H3RuntimeError(f"full_quality Kijai H3 Sage hook receipt mismatch: {mismatches}")


def _apply_requested_kijai_backend(model: Any, sage_receipt: Dict[str, Any], apply_patch: Callable[[Any], tuple[Any, Dict[str, Any]]]) -> tuple[Any, Dict[str, Any]]:
    """Apply and verify Kijai only after the real primary denoiser exists."""
    _require_kijai_h3_contract(sage_receipt)
    patched_model, hook_receipt = apply_patch(model)
    _verify_kijai_h3_hook(hook_receipt)
    sage_receipt.update({
        "selected": "sage",
        "available": True,
        "compatibilityFailure": None,
        "contractValidation": "ready_after_primary_model_load",
        "selectionTiming": "after_primary_model_load_and_sigma_shift",
        "hookImplementation": "kijai_kj_minimax_h3_memory_efficient_sage",
        "hookApplied": True,
        "hookReceipt": hook_receipt,
    })
    return patched_model, hook_receipt


def _apply_kijai_or_native_fallback(
    model: Any,
    sage_receipt: Dict[str, Any],
    apply_patch: Callable[[Any], tuple[Any, Dict[str, Any]]],
) -> tuple[Any, Optional[Dict[str, Any]], str, Optional[str]]:
    """Prefer Kijai, but keep full-quality sampling alive on unsupported NVIDIA GPUs."""

    try:
        patched_model, hook_receipt = _apply_requested_kijai_backend(model, sage_receipt, apply_patch)
        return patched_model, hook_receipt, "kijai_fast", None
    except (H3RuntimeError, RuntimeError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        sage_receipt.update({
            "selected": "official_native",
            "hookApplied": False,
            "hookImplementation": None,
            "hookReason": "kijai_runtime_incompatible_fallback_to_official_native",
            "fallback_reason": reason,
            "compatibilityFallback": True,
        })
        return model, None, "official_native", reason


def _verified_kernel_backend_receipt(
    requested_kernel: str,
    hook_receipt: Optional[Dict[str, Any]],
    ffn_receipt: Optional[Dict[str, Any]],
    steps: int = 20,
    ffn_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Build and verify the real pre-sampling kernel identity."""

    if requested_kernel == "official_native":
        if hook_receipt:
            raise H3RuntimeError("官方原生内核混入 Kijai/Sage 补丁证据；已阻断。")
        if (ffn_receipt or {}).get("applied"):
            raise H3RuntimeError("官方原生内核混入 Kijai FFN 优化；已阻断。")
        actual = expected_kernel_backend_receipt("official_native", steps)
    else:
        hook = dict(hook_receipt or {})
        ffn = dict(ffn_receipt or {})
        actual = expected_kernel_backend_receipt("kijai_fast", steps, ffn_chunks)
        actual.update({
            "classification": hook.get("classification"),
            "attentionProvider": hook.get("author"),
            "source": hook.get("requestedSource") or hook.get("source"),
            "sourceCommit": hook.get("sourceCommit"),
            "sagePackageVersion": hook.get("sagePackageVersion"),
            "cudaArchitecturePolicy": "current_device_must_be_supported",
            "cudaArchitecture": hook.get("currentDeviceArchitecture"),
            "requiredSageSymbols": hook.get("requiredSageSymbols"),
            "backend": hook.get("backend"),
            "patchedBlocks": hook.get("patchedBlocks"),
            "scope": hook.get("scope"),
            "globalPatch": hook.get("globalPatch"),
            "ffnChunkEnabled": bool(ffn.get("enabled", ffn.get("applied"))),
        })
        if ffn.get("enabled", ffn.get("applied")):
            actual.update({"ffnChunks": ffn.get("chunks"), "ffnMinTokens": ffn.get("minTokens")})
    try:
        return verify_kernel_backend_receipt(requested_kernel, actual, steps, ffn_chunks)
    except ValueError as exc:
        raise H3RuntimeError(str(exc)) from exc


def finalize_kernel_backend_receipt(
    requested_kernel: str,
    hook_receipt: Optional[Dict[str, Any]],
    ffn_receipt: Optional[Dict[str, Any]],
    steps: int = 20,
    ffn_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the stable top-level kernel fields used by API and task cards."""

    verified = _verified_kernel_backend_receipt(requested_kernel, hook_receipt, ffn_receipt, steps, ffn_chunks)
    return {
        "requestedKernel": requested_kernel,
        "actualKernel": verified["kernelBackend"],
        "kernelIdentity": verified["backend"],
        "patchedBlocks": verified.get("patchedBlocks", 0),
        "scope": verified.get("scope", "official_native"),
        "ffnEnabled": verified["ffnChunkEnabled"],
        "ffnChunks": int(verified.get("ffnChunks") or 0),
        "fallback": False,
    }


def _apply_stage_ffn_chunk(
    model: Any,
    stage: str,
    chunks: int,
    *,
    adapter: Callable[..., tuple[Any, Dict[str, Any]]] = apply_h3_ffn_chunk_diagnostic,
) -> tuple[Any, Dict[str, Any]]:
    """Keep FFN wrapping at the stage that actually samples with it."""

    chunks = int(chunks)
    if stage == "stage1":
        if chunks != 1:
            raise H3RuntimeError("stage1 FFN chunks must be 1; cannot patch stage1")
        return model, {"stage": stage, "chunks": 1, "applied": False}
    if stage != "stage2" or chunks not in {2, 4}:
        raise H3RuntimeError("invalid stage-specific FFN chunk contract")
    try:
        patched, receipt = adapter(model, chunks=chunks, min_tokens=4096)
    except Exception as exc:
        raise H3RuntimeError(f"stage2 FFN chunk patch failed: {type(exc).__name__}: {exc}") from exc
    if patched is model or not bool(receipt.get("applied")):
        raise H3RuntimeError("stage2 FFN chunk patch was not applied; cannot fall back")
    return patched, {"stage": stage, **dict(receipt)}
_QUANT_WARMUP_MANIFEST = ROOT / "cache" / "runtime" / "quant_warmup.json"


def record_native_preflight_failure(acceleration_receipt: Dict[str, Any], error: BaseException) -> Dict[str, Any]:
    """Keep an adapter's fail-closed evidence in the worker-visible receipt."""

    adapter_receipt = dict(getattr(error, "receipt", {}) or {})
    adapter_receipt.update(acceleration_receipt)
    adapter_receipt.update({
        "status": "failed_preflight",
        "fallback": False,
        "fallbackReason": f"{type(error).__name__}: {error}",
    })
    return adapter_receipt


def _validate_full_quality_compiled_contract(compiled: Dict[str, Any]) -> None:
    """Reject any persisted or mutated plan that no longer matches the product contract."""

    mode = str(compiled.get("mode") or "").upper()
    requested_kernel = str(compiled.get("requestedKernel") or "kijai_fast")
    requested_steps = int((compiled.get("advanced") or {}).get("steps") or 20)
    requested_ffn = (compiled.get("advanced") or {}).get("ffnChunks")
    advanced = dict(compiled.get("advanced") or {})
    expected = two_stage_route_contract(mode, requested_kernel, requested_steps, requested_ffn, advanced)
    expected_kernel_contract = expected_kernel_backend_contract(requested_kernel, requested_steps, requested_ffn)
    route = dict(compiled.get("algorithmRoute") or {})
    route_hash_payload = dict(route)
    route_hash_payload.pop("algorithmRouteFingerprint", None)
    route_canonical = json.dumps(route_hash_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    expected_route_fingerprint = f"sha256:{hashlib.sha256(route_canonical.encode('utf-8')).hexdigest()}"
    observed = {
        "routeId": route.get("algorithmRoute"),
        "fingerprint": route.get("algorithmRouteFingerprint"),
        "routeContract": route.get("routeContract"),
        "primaryModel": (compiled.get("routing") or {}).get("primaryModel"),
        "sampler": route.get("sampler"),
        "scheduler": route.get("scheduler"),
        "steps": route.get("steps"),
        "executionProfile": route.get("algorithmProfile"),
        "accelerationMode": advanced.get("accelerationMode"),
        "routeVersion": route.get("algorithmRouteVersion"),
        "stage1Steps": route.get("stage1Steps"),
        "stage2Steps": route.get("stage2Steps"),
        "stage2Denoise": route.get("stage2Denoise"),
    }
    required = {
        "routeId": LATENT_UPSCALE_TWO_STAGE_ROUTE,
        "fingerprint": expected_route_fingerprint,
        "routeContract": expected,
        "primaryModel": "REF2VA" if mode == "R2V" else "FL2VA",
        "sampler": expected["sampling"]["sampler"],
        "scheduler": expected["sampling"]["scheduler"],
        "steps": expected["sampling"]["steps"],
        "executionProfile": expected["executionSwitches"]["executionProfile"],
        "accelerationMode": "full_quality",
        "routeVersion": LATENT_UPSCALE_TWO_STAGE_VERSION,
        "stage1Steps": int(advanced.get("stage1Steps") or 0),
        "stage2Steps": int(advanced.get("stage2Steps") or 0),
        "stage2Denoise": float(advanced.get("stage2Denoise") or 0.30),
    }
    mismatches = {key: {"expected": value, "observed": observed.get(key)} for key, value in required.items() if observed.get(key) != value}
    if mismatches:
        raise H3RuntimeError(f"full_quality contract mismatch: {mismatches}")
    if compiled.get("kernelBackendContract") != expected_kernel_contract:
        raise H3RuntimeError("full_quality kernel backend contract mismatch")


def _validate_low_fps_compiled_contract(compiled: Dict[str, Any]) -> None:
    """Reject a mutated or incomplete experimental time-remap plan."""

    timing = dict(compiled.get("timing") or {})
    route = dict(compiled.get("algorithmRoute") or {})
    advanced = dict(compiled.get("advanced") or {})
    requested_fps = int(timing.get("requestedFps") or 0)
    if requested_fps not in {12, 16}:
        raise H3RuntimeError(f"{LOW_FPS_TIME_REMAP_ROUTE} requires requestedFps 12 or 16")
    expected_route = low_fps_time_remap_route_receipt(
        str(compiled.get("mode") or ""),
        str((compiled.get("routing") or {}).get("primaryModel") or ""),
        advanced,
        timing,
    )
    required_timing = {
        "modelFrameCount": timing.get("frameCount"),
        "actualModelFps": requested_fps,
        "temporalDensityFps": requested_fps,
        "officialModelTimebaseFps": 24,
        "timeScale": 24 / requested_fps,
        "videoLatentT": 2 if int(timing.get("frameCount") or 0) <= 5 else ((int(timing.get("frameCount") or 0) - 5) // 17) * 5 + 2,
        "fallback": False,
    }
    observed_timing = {
        **{key: timing.get(key) for key in required_timing if key != "fallback"},
        "fallback": (route.get("routeContract") or {}).get("fallback"),
    }
    mismatches = {
        "route": {"expected": expected_route, "observed": route}
    } if route != expected_route else {}
    timing_mismatches = {
        key: {"expected": value, "observed": observed_timing.get(key)}
        for key, value in required_timing.items()
        if observed_timing.get(key) != value
    }
    if timing_mismatches:
        mismatches["timing"] = timing_mismatches
    if mismatches:
        raise H3RuntimeError(f"{LOW_FPS_TIME_REMAP_ROUTE} contract mismatch: {mismatches}")


def _validate_compiled_route(compiled: Dict[str, Any]) -> None:
    route = compiled.get("algorithmRoute") or {}
    route_id = str(route.get("algorithmRoute") or "")
    route = compiled.get("algorithmRoute") or {}
    route_id = str(route.get("algorithmRoute") or "")
    advanced = compiled.get("advanced") or {}
    if route_id == LATENT_UPSCALE_TWO_STAGE_ROUTE:
        contract = route.get("routeContract") or {}
        two_stage = contract.get("twoStage") or {}
        resize_contract = two_stage.get("frameResizeVaeRoundTrip")
        latent_contract = resize_contract if isinstance(resize_contract, dict) else {}
        if latent_contract.get("defaultRouteMode") == "latent":
            required_latent = {
                "algorithm": "minimax_h3_latent_upscaler_3d",
                "nvidiaRtxVsr": False,
                "videoVaeRoundTrip": False,
                "fallbackUsed": False,
            }
            if any(latent_contract.get(key) != value for key, value in required_latent.items()):
                raise H3RuntimeError("LBH latent upscale contract missing")
            source = dict(compiled.get("canvas", {}).get("stage1SourceLatent") or {})
            target = dict(compiled.get("canvas", {}).get("stage2TargetLatent") or {})
            raw = dict(compiled.get("canvas", {}).get("lbhOutputLatent") or {})
            crop = dict(compiled.get("canvas", {}).get("lbhCrop") or {})
            if (
                not source or raw.get("height") != int(source.get("height", 0)) * 2
                or raw.get("width") != int(source.get("width", 0)) * 2
                or crop.get("strategy") != "top_left_edge"
                or crop.get("height") != target.get("height")
                or crop.get("width") != target.get("width")
            ):
                raise H3RuntimeError("LBH latent grid/crop contract missing")
        elif (
            latent_contract.get("nvidiaRtxVsr") is not True
            or latent_contract.get("algorithm") != "nvidia_rtx_vsr"
            or latent_contract.get("defaultRouteMode") != "vsr_recovery"
            or latent_contract.get("fallbackUsed") is not False
        ):
            raise H3RuntimeError("VSR recovery frame resize contract missing")
        if (
            two_stage.get("stage1Sampler") != "res_multistep"
            or two_stage.get("stage1Scheduler") != "simple"
            or two_stage.get("stage2Sampler") != "euler"
            or two_stage.get("stage2Scheduler") != "beta"
        ):
            raise H3RuntimeError("two-stage sampler contract mismatch")
        if two_stage.get("directStage2Output") is not True or (two_stage.get("finalDownsample") or {}).get("executed") is not False:
            raise H3RuntimeError("stage2 direct-output contract mismatch")
        return
    if route_id == KJ_EXPERIMENT_ROUTE:
        expected = native_acceleration_route_contract(str(route.get("algorithmProfile") or ""))
        observed = route.get("routeContract") or {}
        required = (
            "routeId", "routeVersion", "experimentId", "sourceCommit",
            "requested", "actual", "patchedBlocks", "scope", "fallback",
        )
        if expected is None or any(observed.get(key) != expected.get(key) for key in required):
            raise H3RuntimeError("isolated KJ experiment contract mismatch; fallback is forbidden")
        if (
            observed.get("requested") != observed.get("actual")
            or observed.get("requested") != compiled.get("requestedKernel")
            or observed.get("patchedBlocks") != 50
            or observed.get("scope") != "denoiser_only"
            or observed.get("fallback") is not False
        ):
            raise H3RuntimeError("isolated KJ experiment runtime identity mismatch; fallback is forbidden")
        return
    if route_id == LOW_FPS_TIME_REMAP_ROUTE:
        raise H3RuntimeError("non-24 FPS routes are retired; execution plan creation rejected")
    raise H3RuntimeError(f"unsupported compiled route: {route_id or 'missing'}")


def _validate_native_experiment_route_identity(compiled: Dict[str, Any], execution_profile: str) -> None:
    """Validate native route/profile fields without conflating their meanings."""

    route = compiled.get("algorithmRoute") if isinstance(compiled.get("algorithmRoute"), dict) else {}
    expected_route = KJ_EXPERIMENT_ROUTE
    expected_profile = KJ_EXPERIMENT_PROFILE
    observed_route = route.get("algorithmRoute")
    observed_profile = route.get("algorithmProfile")
    observed_execution_profile = str(execution_profile or "")
    observed_contract_route = (route.get("routeContract") or {}).get("routeId")
    if (
        observed_route != expected_route
        or observed_profile != expected_profile
        or observed_execution_profile != expected_profile
        or observed_contract_route != expected_route
    ):
        raise H3RuntimeError(
            "native route contract mismatch: expected route/profile semantics, "
            f"observed {{'algorithmRoute': {observed_route!r}, 'algorithmProfile': {observed_profile!r}, "
            f"'executionProfile': {observed_execution_profile!r}, 'contractRouteId': {observed_contract_route!r}}}"
        )


def _apply_low_fps_time_remap_layout(
    layout: Any,
    *,
    time_scale: float,
    frame_count: int,
    requested_fps: int,
    variant_id: str = "continuous_direct",
    position_mapping: str = "continuous_time_scale",
) -> Dict[str, Any]:
    """Stretch only the target video MM-RoPE clock onto the full audio clock."""

    if requested_fps not in {12, 16} or time_scale != 24 / requested_fps:
        raise H3RuntimeError("low-fps layout identity mismatch")
    if int(frame_count) < 5 or int(frame_count) % 17 != 5:
        raise H3RuntimeError("low-fps model frame count must satisfy 17k+5")
    existing = getattr(layout, "_h3_low_fps_time_remap_receipt", None)
    if isinstance(existing, dict):
        if (
            existing.get("requestedFps") != requested_fps
            or existing.get("timeScale") != time_scale
            or existing.get("variantId") != variant_id
            or existing.get("positionMapping") != position_mapping
        ):
            raise H3RuntimeError("low-fps layout was already remapped with a different identity")
        return existing

    audio_segments = [(a, b) for a, b, kind in layout.segments if kind == "audio"]
    video_segments = [(a, b) for a, b, kind in layout.segments if kind == "video"]
    if len(audio_segments) != 1 or len(video_segments) != 1:
        raise H3RuntimeError("low-fps layout requires exactly one target audio and video segment")
    audio_start, audio_stop = audio_segments[0]
    video_start, video_stop = video_segments[0]
    cursor = layout.position_ids[audio_start, 0].clone()
    native_video_t = layout.position_ids[video_start:video_stop, 0].clone()
    remapped_video_t = cursor + (native_video_t - cursor) * time_scale
    if position_mapping == "official_24fps_integer_lattice":
        remapped_video_t = remapped_video_t.round()
    elif position_mapping != "continuous_time_scale":
        raise H3RuntimeError(f"unsupported low-fps position mapping: {position_mapping}")
    layout.position_ids[video_start:video_stop, 0] = remapped_video_t

    cond_segments = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_segments) > 1:
        last_start, last_stop = cond_segments[-1]
        native_last_t = layout.position_ids[last_start, 0].clone()
        remapped_last_t = cursor + (native_last_t - cursor) * time_scale
        if position_mapping == "official_24fps_integer_lattice":
            remapped_last_t = remapped_last_t.round()
        layout.position_ids[last_start:last_stop, 0] = remapped_last_t

    receipt = {
        "routeId": LOW_FPS_TIME_REMAP_ROUTE,
        "routeVersion": LOW_FPS_TIME_REMAP_VERSION,
        "variantId": variant_id,
        "positionMapping": position_mapping,
        "requestedFps": requested_fps,
        "actualModelFps": requested_fps,
        "temporalDensityFps": requested_fps,
        "officialModelTimebaseFps": 24,
        "modelFrameCount": int(frame_count),
        "timeScale": float(time_scale),
        "videoLatentT": int(getattr(layout, "signature")[1]),
        "audioLatentT": int(getattr(layout, "signature")[4]),
        "videoCoordinateStart": float(layout.position_ids[video_start, 0]),
        "videoCoordinateEnd": float(layout.position_ids[video_stop - 1, 0]),
        "audioCoordinateStart": float(layout.position_ids[audio_start, 0]),
        "audioCoordinateEnd": float(layout.position_ids[audio_start + (audio_stop - audio_start) // 2 - 1, 0]),
        "referenceCoordinatesChanged": False,
        "promptTimeMapping": "target_wall_clock_seconds_verbatim",
        "referenceTimeMapping": "source_24fps_and_qwen_2fps_unchanged",
        "fallback": False,
    }
    setattr(layout, "_h3_low_fps_time_remap_receipt", receipt)
    return receipt


def _replace_low_fps_audio_latent(
    latent: Dict[str, Any],
    audio_latent_t: int,
    h3: Any,
    torch: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Keep the sparse target video latent while allocating the full remapped audio clock."""

    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not hasattr(samples, "unbind"):
        raise H3RuntimeError("low-fps route requires a nested video/audio latent")
    video, audio = samples.unbind()
    if int(audio_latent_t) < 1:
        raise H3RuntimeError("low-fps audioLatentT must be positive")
    replacement = torch.zeros(
        [audio.shape[0], audio.shape[1], audio.shape[2], int(audio_latent_t)],
        dtype=audio.dtype,
        device=audio.device,
    )
    result = dict(latent)
    result["samples"] = h3.comfy.nested_tensor.NestedTensor((video, replacement))
    return result, {
        "routeId": LOW_FPS_TIME_REMAP_ROUTE,
        "videoLatentT": int(video.shape[2]),
        "audioLatentT": int(replacement.shape[-1]),
        "fallback": False,
    }


def _uses_full_quality_conditioning_contract(compiled: Dict[str, Any]) -> bool:
    route = compiled.get("algorithmRoute") if isinstance(compiled.get("algorithmRoute"), dict) else {}
    route_id = str(route.get("algorithmRoute") or "")
    contract = route.get("routeContract") if isinstance(route.get("routeContract"), dict) else {}
    return (
        route_id == LATENT_UPSCALE_TWO_STAGE_ROUTE
        and contract.get("routeId") == LATENT_UPSCALE_TWO_STAGE_ROUTE
        and contract.get("contractVersion") == LATENT_UPSCALE_TWO_STAGE_VERSION
        and contract.get("baseRouteId") == "full_quality"
        and isinstance((contract.get("twoStage") or {}).get("frameResizeVaeRoundTrip"), dict)
    )


def _prepare_low_fps_conditioning(
    compiled: Dict[str, Any],
    latent: Dict[str, Any],
    h3: Any,
    torch: Any,
    runtime_stages: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the route-specific target audio clock at the conditioning boundary."""

    route_id = str(((compiled.get("algorithmRoute") or {}).get("algorithmRoute")) or "")
    if route_id == LATENT_UPSCALE_TWO_STAGE_ROUTE:
        _validate_compiled_route(compiled)
        if not _uses_full_quality_conditioning_contract(compiled):
            raise H3RuntimeError("latent_upscale_two_stage conditioning capability contract mismatch")
        return latent
    if route_id == KJ_EXPERIMENT_ROUTE:
        _validate_compiled_route(compiled)
        return latent
    if route_id != LOW_FPS_TIME_REMAP_ROUTE:
        raise H3RuntimeError(f"unsupported conditioning route: {route_id or 'missing'}")
    timing = dict(compiled.get("timing") or {})
    remapped, receipt = _replace_low_fps_audio_latent(
        latent,
        int(timing.get("audioLatentT") or 0),
        h3,
        torch,
    )
    if receipt["videoLatentT"] != int(timing.get("videoLatentT") or 0):
        raise H3RuntimeError("low-fps conditioning videoLatentT mismatch")
    receipt.update({
        "requestedFps": int(timing["requestedFps"]),
        "actualModelFps": int(timing["actualModelFps"]),
        "temporalDensityFps": int(timing["temporalDensityFps"]),
        "modelFrameCount": int(timing["modelFrameCount"]),
        "timeScale": float(timing["timeScale"]),
    })
    runtime_stages["low_fps_audio_latent"] = receipt
    return remapped


def _install_low_fps_time_remap_wrapper(
    model: Any,
    timing: Dict[str, Any],
    runtime_stages: Dict[str, Any],
) -> Tuple[Any, Dict[str, Any]]:
    """Attach the request-scoped MM-RoPE remap through ComfyUI's public wrapper seam."""

    patcher_extension = importlib.import_module("comfy.patcher_extension")
    state: Dict[str, Any] = {
        "status": "installed",
        "wrapperCalls": 0,
        "variantId": str(timing.get("variantId") or "continuous_direct"),
        "positionMapping": str(timing.get("positionMapping") or "continuous_time_scale"),
        "decodeMode": str(timing.get("decodeMode") or "direct"),
        "bridgedModelFps": timing.get("bridgedModelFps"),
        "bridgedModelFrameCount": timing.get("bridgedModelFrameCount"),
        "bridgedVideoLatentT": timing.get("bridgedVideoLatentT"),
        "fallback": False,
    }

    def remap_wrapper(executor: Any, x: Any, timestep: Any, context: Any,
                      transformer_options: Optional[Dict[str, Any]] = None,
                      minimax_payload: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        payload = dict(minimax_payload or {})
        layout = payload.get("layout")
        if layout is None:
            raise H3RuntimeError("low-fps wrapper did not receive the official prebuilt PackedLayout")
        receipt = _apply_low_fps_time_remap_layout(
            layout,
            time_scale=float(timing["timeScale"]),
            frame_count=int(timing["modelFrameCount"]),
            requested_fps=int(timing["requestedFps"]),
            variant_id=str(timing.get("variantId") or "continuous_direct"),
            position_mapping=str(timing.get("positionMapping") or "continuous_time_scale"),
        )
        if receipt["videoLatentT"] != int(timing["videoLatentT"]):
            raise H3RuntimeError("low-fps wrapper videoLatentT mismatch")
        if receipt["audioLatentT"] != int(timing["audioLatentT"]):
            raise H3RuntimeError("low-fps wrapper audioLatentT mismatch")
        payload["layout"] = layout
        state.update(receipt)
        state["status"] = "active"
        state["wrapperCalls"] = int(state.get("wrapperCalls") or 0) + 1
        runtime_stages["low_fps_time_remap"] = dict(state)
        return executor(
            x,
            timestep,
            context,
            transformer_options or {},
            minimax_payload=payload,
            **kwargs,
        )

    cloned = model.clone()
    cloned.add_wrapper_with_key(
        patcher_extension.WrappersMP.DIFFUSION_MODEL,
        LOW_FPS_TIME_REMAP_ROUTE,
        remap_wrapper,
    )
    runtime_stages["low_fps_time_remap"] = dict(state)
    return cloned, state


def _counter_delta_value(current: Any, previous: Any) -> Any:
    """Return a monotonic counter delta without coercing unavailable data to 0."""

    if isinstance(current, int) and isinstance(previous, int):
        return max(0, current - previous)
    if isinstance(current, int) and previous is None:
        return current
    return "unavailable"


def _export_frame_indices(
    decoded_count: int,
    export_count: int,
    model_fps: int,
    output_fps: int,
) -> list[int]:
    if decoded_count < 1 or export_count < 1:
        raise H3RuntimeError("decoded and export frame counts must be positive")
    if model_fps < 1 or output_fps < 1 or output_fps > model_fps:
        raise H3RuntimeError("output FPS must be positive and cannot exceed the H3 model FPS")
    indices = [int(index * model_fps / output_fps) for index in range(export_count)]
    if indices[-1] >= decoded_count:
        raise H3RuntimeError(
            f"official VAE decoded {decoded_count} frames, fewer than the {indices[-1] + 1} required for export"
        )
    return indices


def _memory_budget_receipt(resource_telemetry: Dict[str, Any], primary_device_receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Explain native DynamicVRAM paging without changing its policy."""

    pre_sampler_cuda = (resource_telemetry.get("primary_model_pre_sampler") or {}).get("cuda") or {}
    sampling_cuda = (resource_telemetry.get("primary_sampling_complete") or {}).get("cuda") or {}
    dynamic_receipt = primary_device_receipt.get("dynamicVramNative") or {}
    model_size_bytes = dynamic_receipt.get("modelSizeBytes")
    peak_reserved_bytes = sampling_cuda.get("peakReservedBytes")
    total_vram_bytes = pre_sampler_cuda.get("totalBytes") or sampling_cuda.get("totalBytes")
    full_residency_feasible = None
    if all(isinstance(value, (int, float)) for value in (model_size_bytes, peak_reserved_bytes, total_vram_bytes)):
        full_residency_feasible = int(model_size_bytes) + int(peak_reserved_bytes) <= int(total_vram_bytes)
    return {
        "totalVramBytes": total_vram_bytes,
        "modelSizeBytes": model_size_bytes,
        "peakTorchReservedBytes": peak_reserved_bytes,
        "loadedWeightBytes": dynamic_receipt.get("loadedWeightBytes"),
        "offloadedWeightBytes": dynamic_receipt.get("offloadedWeightBytes"),
        "fullResidencyFeasible": full_residency_feasible,
        "policy": "native_comfy_aimdo_dynamic_vram",
        "offloadReason": (
            "model weights plus peak sampling workspace exceed physical VRAM; native AIMDO paging is required"
            if full_residency_feasible is False else None
        ),
        "oomOrFallback": None,
    }


def _update_memory_admission_receipt(execution_receipt: Dict[str, Any], admission: Dict[str, Any]) -> None:
    """Expose the immutable pre-denoise memory contract at receipt top level."""

    execution_receipt.update({
        "historyLookup": "disabled",
        "resolutionBucket": admission.get("resolutionBucket"),
        "durationBucket": admission.get("durationBucket"),
        "decisionSource": admission.get("decisionSource", "official_capacity_fallback"),
        "offloadPlan": admission.get("offloadPlan"),
    })


def _wait_for_isolated_vae_worker_exit(
    process: Any,
    cancel_event: threading.Event,
    deadline: Optional[float],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Wait for an isolated VAE worker without terminating its CUDA context.

    The legacy worker protocol has no child-side cancellation reader.  A
    parent must therefore record a cancellation/timeout request and wait for
    the child to reach its own natural cleanup/exit path; calling
    ``TerminateProcess`` here can skip the child's CUDA/VAE cleanup.
    """

    state: Dict[str, Any] = {
        "status": "running",
        "cooperativeCancellationSupported": False,
        "hardTerminationUsed": False,
        "cancelObservedAtMonotonic": None,
        "deadlineExceededAtMonotonic": None,
    }
    while process.poll() is None:
        now = monotonic()
        if cancel_event.is_set() and state["cancelObservedAtMonotonic"] is None:
            state["status"] = "cancel_requested_waiting_for_worker_exit"
            state["cancelObservedAtMonotonic"] = now
        if deadline is not None and now > deadline and state["deadlineExceededAtMonotonic"] is None:
            state["status"] = "deadline_exceeded_waiting_for_worker_exit"
            state["deadlineExceededAtMonotonic"] = now
        sleep(0.25)
    state["workerExitedAtMonotonic"] = monotonic()
    return state


def _quant_warmup_identity(
    torch: Any,
    selected_backend: Optional[str],
    sequence_tokens: int,
    dtype: Any,
    shapes: list[tuple[int, ...]],
) -> Dict[str, Any]:
    """Build a non-content cache identity for Triton execution metadata."""

    capability = None
    try:
        if torch.cuda.is_available():
            capability = list(torch.cuda.get_device_capability(torch.cuda.current_device()))
    except Exception:
        capability = None
    try:
        triton = importlib.import_module("triton")
        triton_version = str(getattr(triton, "__version__", "unknown"))
    except Exception:
        triton_version = "unavailable"
    try:
        kitchen = importlib.import_module("comfy_kitchen")
        kitchen_version = str(getattr(kitchen, "__version__", "unknown"))
    except Exception:
        kitchen_version = "unavailable"
    return {
        "schema": 1,
        "backend": str(selected_backend),
        "sequenceTokens": int(sequence_tokens),
        "dtype": str(dtype),
        "shapes": [list(shape) for shape in sorted(shapes)],
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": str(getattr(getattr(torch, "version", None), "cuda", None)),
        "computeCapability": capability,
        "triton": triton_version,
        "comfyKitchen": kitchen_version,
    }


def _read_persistent_warmup(identity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if not _QUANT_WARMUP_MANIFEST.is_file():
            return None
        data = json.loads(_QUANT_WARMUP_MANIFEST.read_text(encoding="utf-8"))
        if data.get("status") == "completed" and data.get("identity") == identity:
            return data
    except Exception:
        # A stale or partially-written optimization receipt must never block
        # the normal first-use path.
        return None
    return None


def _write_persistent_warmup(identity: Dict[str, Any], results: list[Dict[str, Any]]) -> Optional[str]:
    try:
        _QUANT_WARMUP_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        temporary = _QUANT_WARMUP_MANIFEST.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"status": "completed", "identity": identity, "results": results},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary.replace(_QUANT_WARMUP_MANIFEST)
        return str(_QUANT_WARMUP_MANIFEST)
    except Exception:
        return None


def _safe_staged_path(value: str) -> Path:
    try:
        return resolve_staged_path(value)
    except StagingError as exc:
        raise H3RuntimeError(str(exc)) from exc


def _verified_staged_path(reference: Dict[str, Any]) -> Path:
    path = _safe_staged_path(str(reference.get("path") or ""))
    expected = str(reference.get("sha256") or "").lower()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if not expected or actual != expected:
        raise H3RuntimeError(f"参考素材 {reference.get('token') or reference.get('name')} 的 SHA256 在 worker 打开前校验失败")
    return path


def _load_image(path: Path, torch: Any, preprocess_plan: Optional[Dict[str, Any]] = None, return_receipt: bool = False) -> Any:
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        orientation = int(source.getexif().get(274, 1) or 1)
        alpha_present = "A" in source.getbands()
        icc_present = bool(source.info.get("icc_profile"))
        transposed = ImageOps.exif_transpose(source)
        original_size = list(source.size)
        final_size = list(transposed.size)
        image = transposed.convert("RGB")
        resize = (preprocess_plan or {}).get("resize") or {}
        target_width, target_height = int(resize.get("width") or 0), int(resize.get("height") or 0)
        resized = False
        if target_width > 0 and target_height > 0 and image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            resized = True
        final_size = list(image.size)
        image = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).unsqueeze(0)
    if return_receipt:
        return tensor, {
            "exifOrientation": orientation, "exifOrientationApplied": orientation != 1,
            "sourceSize": original_size, "orientedSize": final_size, "preprocessPlan": dict(preprocess_plan or {}), "resized": resized,
            "alphaPresent": alpha_present, "alphaConversion": "discarded_by_official_rgb_conversion" if alpha_present else "none",
            "iccProfilePresent": icc_present, "colorConversion": "Pillow convert RGB",
        }
    return tensor


def _load_video(path: Path, torch: Any, video_plan: Dict[str, Any], return_receipt: bool = False) -> Any:
    import av
    import numpy as np

    start_seconds = float(video_plan["selectedInterval"]["startSeconds"])
    target_count = int(video_plan["h3FrameCount"])
    target_times = [start_seconds + index / 24.0 for index in range(target_count)]
    container = av.open(str(path))
    try:
        stream = next(stream for stream in container.streams if stream.type == "video")
        source_rate = float(stream.average_rate or stream.base_rate or 24.0)
        seek_seconds = max(0.0, start_seconds - 1.0)
        if start_seconds > 0:
            container.seek(int(seek_seconds * 1_000_000), backward=True)
        frames = []
        selected_frames = []
        target_index = 0
        decoded_index = 0
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None:
                timestamp = decoded_index / source_rate
            decoded_index += 1
            if timestamp < start_seconds:
                continue
            rgb = frame.to_rgb().to_ndarray()
            while target_index < target_count and timestamp + (0.5 / max(source_rate, 1.0)) >= target_times[target_index]:
                frames.append(rgb)
                selected_frames.append({
                    "targetIndex": target_index,
                    "targetTimestampSeconds": round(target_times[target_index], 6),
                    "decodedOrdinalAfterSeek": decoded_index - 1,
                    "sourcePts": int(frame.pts) if frame.pts is not None else None,
                    "sourceTimestampSeconds": round(float(timestamp), 6),
                })
                target_index += 1
            if target_index >= target_count:
                break
    finally:
        container.close()
    if len(frames) != target_count:
        raise H3RuntimeError(f"video could not stream the planned {target_count} frames at 24fps: {path}")
    tensor = torch.from_numpy(np.stack(frames).astype("float32") / 255.0)
    if return_receipt:
        return tensor, {
            "path": path.relative_to(MODEL_ROOT).as_posix(),
            "sourceFps": round(source_rate, 6),
            "plannedH3Frames": target_count,
            "storedFrames": len(frames),
            "sourceFramesDecodedUntilTarget": decoded_index,
            "seekRequestedSeconds": seek_seconds,
            "selectedStartSeconds": start_seconds,
            "selectedEndSeconds": round(start_seconds + target_count / 24.0, 6),
            "targetTimestampsSeconds": [round(value, 6) for value in target_times],
            "selectedSourceFrames": selected_frames,
            "streaming": True,
            "decodedWholeSource": False,
        }
    return tensor


def _load_video_force_rate(path: Path, torch: Any, force_rate: float, start_seconds: float = 0.0, end_seconds: float | None = None, cancel_event: Any = None) -> Dict[str, Any]:
    """Load frames with VideoHelperSuite's force_rate time-sampling semantics."""

    import av
    import numpy as np

    requested_rate = float(force_rate)
    if requested_rate < 1 or requested_rate > 24:
        raise H3RuntimeError("force_rate must be between 1 and 24")
    container = av.open(str(path))
    try:
        stream = next(stream for stream in container.streams if stream.type == "video")
        source_fps = float(stream.average_rate or stream.base_rate or 24.0)
        source_frame_count = int(stream.frames or 0)
        source_duration = float(stream.duration * stream.time_base) if stream.duration and stream.time_base else 0.0
        if source_duration <= 0 and source_frame_count > 0 and source_fps > 0:
            source_duration = source_frame_count / source_fps
        loaded_fps = requested_rate
        interval = 1.0 / loaded_fps
        selected_start = max(0.0, float(start_seconds))
        selected_end = source_duration if end_seconds is None else min(source_duration, float(end_seconds))
        if selected_end <= selected_start:
            raise H3RuntimeError("reference video end must be after start")
        selected_duration = selected_end - selected_start
        next_timestamp = selected_start
        frames = []
        timestamps = []
        decoded_index = 0
        for frame in container.decode(stream):
            if cancel_event is not None and cancel_event.is_set():
                raise H3Cancelled("cancelled during reference video FPS resampling")
            timestamp = float(frame.time) if frame.time is not None else decoded_index / source_fps
            decoded_index += 1
            if timestamp < selected_start:
                continue
            if timestamp > selected_end + (0.5 / max(source_fps, 1.0)):
                break
            if timestamp + (0.5 / max(source_fps, 1.0)) < next_timestamp:
                continue
            frames.append(frame.to_rgb().to_ndarray())
            timestamps.append(round(timestamp, 6))
            next_timestamp += interval
    finally:
        container.close()
    if not frames:
        raise H3RuntimeError(f"video contains no decodable frames: {path}")
    tensor = torch.from_numpy(np.stack(frames).astype("float32") / 255.0)
    loaded_frame_count = len(frames)
    return {
        "frames": tensor,
        "source_fps": round(source_fps, 6),
        "source_duration": round(source_duration, 6),
        "selected_start": round(selected_start, 6),
        "selected_end": round(selected_end, 6),
        "loaded_fps": round(loaded_fps, 6),
        "loaded_frame_count": loaded_frame_count,
        "loaded_duration": round(selected_duration, 6),
        "timestamps": timestamps,
    }


def _audio_slice_bounds(frame_start: float, samples: int, sample_rate: int, start_seconds: float, end_seconds: float | None) -> tuple[int, int]:
    left = max(0, int(round((start_seconds - frame_start) * sample_rate)))
    right = samples if end_seconds is None else min(samples, int(round((end_seconds - frame_start) * sample_rate)))
    return min(samples, left), max(0, right)


def _load_audio(
    path: Path,
    torch: Any,
    start_seconds: float = 0.0,
    end_seconds: float | None = None,
    return_receipt: bool = False,
) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any]]:
    import av
    import numpy as np

    container = av.open(str(path))
    try:
        stream = next(stream for stream in container.streams if stream.type == "audio")
        sample_rate = int(stream.rate or 32000)
        channel_layout = str(stream.layout.name) if getattr(stream, "layout", None) else None
        channels = len(stream.layout.channels) if getattr(stream, "layout", None) else None
        seek_seconds = max(0.0, float(start_seconds) - 1.0)
        if start_seconds > 0:
            container.seek(int(seek_seconds * 1_000_000), backward=True)
        chunks = []
        decoded_frames = 0
        decoded_samples = 0
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            frame_channels = len(frame.layout.channels) if getattr(frame, "layout", None) else (channels or 1)
            if array.ndim == 1:
                array = array[None, :]
            if array.shape[0] == 1 and frame_channels > 1 and not frame.format.is_planar:
                array = array.reshape(-1, frame_channels).T
            frame_samples = int(array.shape[-1])
            frame_start = float(frame.time) if frame.time is not None else decoded_samples / sample_rate
            decoded_frames += 1
            decoded_samples += frame_samples
            if end_seconds is not None and frame_start >= end_seconds:
                break
            left, right = _audio_slice_bounds(frame_start, frame_samples, sample_rate, float(start_seconds), end_seconds)
            if right > left:
                chunks.append(array[..., left:right])
    finally:
        container.close()
    if not chunks:
        raise H3RuntimeError(f"audio contains no decodable samples: {path}")
    waveform = np.concatenate(chunks, axis=-1)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.dtype.kind in "iu":
        waveform = waveform.astype("float32") / float(np.iinfo(waveform.dtype).max)
    audio = {"waveform": torch.from_numpy(waveform.astype("float32")).unsqueeze(0), "sample_rate": sample_rate}
    if return_receipt:
        return audio, {
            "selectedStartSeconds": round(float(start_seconds), 6),
            "selectedEndSeconds": round(float(end_seconds), 6) if end_seconds is not None else None,
            "sourceSampleRate": sample_rate,
            "sourceChannels": channels,
            "sourceChannelLayout": channel_layout,
            "storedSamplesPerChannel": int(waveform.shape[-1]),
            "decodedFrames": decoded_frames,
            "decodedSamplesBeforeWindowing": decoded_samples,
            "seekRequestedSeconds": seek_seconds,
            "boundedDecode": end_seconds is not None,
        }
    return audio


def _has_audio_stream(path: Path) -> bool:
    """Return whether a video has an embedded soundtrack without decoding it."""

    import av

    container = av.open(str(path))
    try:
        return any(stream.type == "audio" for stream in container.streams)
    finally:
        container.close()


def _h3_video_input(samples: Any) -> Any:
    """Convert app video frames [T,H,W,C] to MiniMax VAE [B,C,T,H,W]."""

    if samples.ndim != 4 or samples.shape[-1] != 3:
        raise H3RuntimeError(f"expected video frames [T,H,W,3], got {tuple(samples.shape)}")
    if samples.shape[0] == 1:
        return samples.movedim(-1, 1).unsqueeze(2)
    return samples.permute(3, 0, 1, 2).unsqueeze(0)


class _H3VideoVAEProxy:
    """Keep the official H3 VAE while adapting Comfy IMAGE/video layouts."""

    def __init__(self, vae: Any, patcher: Any = None, model_management: Any = None, torch_module: Any = None, prefer_tiled: bool = False) -> None:
        self._vae = vae
        self._patcher = patcher
        self._model_management = model_management
        self._torch = torch_module
        self._prefer_tiled = bool(prefer_tiled)
        self._device_load_requested = False
        self._official_wrapper = hasattr(vae, "latent_dim") and hasattr(vae, "patcher")
        self.last_encode_path: Optional[str] = None
        self.last_encode_preflight: Optional[Dict[str, Any]] = None
        self.last_decode_path: Optional[str] = None

    def _inference_call(self, fn: Callable[[], Any]) -> Any:
        if self._torch is None:
            return fn()
        inference_mode = getattr(self._torch, "inference_mode", None)
        if inference_mode is not None:
            with inference_mode():
                return fn()
        no_grad = getattr(self._torch, "no_grad", None)
        if no_grad is not None:
            with no_grad():
                return fn()
        return fn()

    def _ensure_loaded(self) -> None:
        if self._patcher is not None and self._model_management is not None and not self._device_load_requested:
            self._model_management.load_models_gpu([self._patcher], force_full_load=False)
            self._device_load_requested = True

    def require_device_load(self) -> None:
        self._device_load_requested = False

    def encode(self, samples: Any) -> Any:
        if self._official_wrapper:
            self.last_encode_path = "regular"
            self.last_encode_preflight = None
            if self._prefer_tiled and hasattr(self._vae, "encode_tiled"):
                self.last_encode_path = "tiled_preferred"
                return self._vae.encode_tiled(samples)
            preflight = self._official_encode_memory_preflight(samples)
            self.last_encode_preflight = preflight
            if preflight and preflight.get("useTiled"):
                self.last_encode_path = "tiled_preflight"
                return self._inference_call(lambda: self._vae.encode_tiled(samples))
            return self._inference_call(lambda: self._vae.encode(samples))
        self._ensure_loaded()
        value = _h3_video_input(samples)
        try:
            parameter = next(self._vae.parameters())
            if parameter.is_floating_point() and value.dtype != parameter.dtype:
                value = value.to(dtype=parameter.dtype)
        except (AttributeError, StopIteration):
            pass
        return self._inference_call(lambda: self._vae.encode(value))

    def _official_encode_memory_preflight(self, samples: Any) -> Optional[Dict[str, Any]]:
        """Estimate official regular-encode memory without allocating GPU tensors."""

        torch = self._torch
        vae = self._vae
        if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            return None
        if not all(hasattr(vae, name) for name in ("vae_encode_crop_pixels", "memory_used_encode", "vae_dtype")):
            return None
        try:
            pixel_samples = vae.vae_encode_crop_pixels(samples).movedim(-1, 1)
            if getattr(vae, "latent_dim", None) == 3 and pixel_samples.ndim < 5:
                if not getattr(vae, "not_video", False):
                    pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)
                else:
                    pixel_samples = pixel_samples.unsqueeze(2)
            required = int(vae.memory_used_encode(pixel_samples.shape, vae.vae_dtype))
            free, total = torch.cuda.mem_get_info()
            headroom = max(512 * 1024 * 1024, int(required * 0.10))
            return {
                "requiredBytes": required,
                "freeBytes": int(free),
                "totalBytes": int(total),
                "headroomBytes": headroom,
                "useTiled": int(free) < required + headroom,
            }
        except Exception:
            return None

    def decode(self, latent: Any) -> Any:
        if self._official_wrapper:
            if self._prefer_tiled and hasattr(self._vae, "decode_tiled"):
                self.last_decode_path = "tiled_preferred"
                return self._inference_call(lambda: self._vae.decode_tiled(latent))
            self.last_decode_path = "regular"
            return self._inference_call(lambda: self._vae.decode(latent))
        self._ensure_loaded()
        self.last_decode_path = "regular"
        decoded = self._inference_call(lambda: self._vae.decode(latent))
        if decoded.ndim == 5 and decoded.shape[1] == 3:
            decoded = decoded.permute(0, 2, 3, 4, 1)
        elif decoded.ndim == 4 and decoded.shape[1] == 3:
            decoded = decoded.permute(0, 2, 3, 1)
        return decoded.add(1.0).div_(2.0).clamp_(0.0, 1.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vae, name)


def _align_latent_component_device(value: Any, reference: Any) -> Any:
    if value.device == reference.device:
        return value
    return value.to(device=reference.device)


def _nvidia_rtx_vsr_resize(frames: Any, width: int, height: int, _crop: str = "crop") -> Any:
    """Call the locked KJNodes ImageResizeKJv2 NVIDIA RTX VSR branch."""
    try:
        from ComfyUI_KJNodes.nodes.image_nodes import ImageResizeKJv2
        import nvvfx
    except Exception as exc:
        raise H3RuntimeError(f"nvidia_rtx_vsr dependency unavailable: {exc}") from exc
    module_path = Path(nvvfx.__file__).resolve()
    private_root = (ROOT / "_private_site").resolve()
    if private_root not in module_path.parents:
        raise H3RuntimeError(f"nvidia-vfx must load from the 8775 private site: {module_path}")
    try:
        resized, actual_width, actual_height, _mask = ImageResizeKJv2().resize(
            frames,
            int(width),
            int(height),
            "crop",
            "nvidia_rtx_vsr",
            2,
            "0, 0, 0",
            "center",
            None,
            device="cpu",
            mask=None,
            per_batch=64,
        )
    except Exception as exc:
        raise H3RuntimeError(f"nvidia_rtx_vsr execution failed: {exc}") from exc
    if (int(actual_width), int(actual_height)) != (int(width), int(height)):
        raise H3RuntimeError(
            f"nvidia_rtx_vsr returned unexpected canvas: {actual_width}x{actual_height} != {width}x{height}"
        )
    return resized


def _kj_center_crop_for_vsr(frames: Any, width: int, height: int) -> Any:
    """Match the locked KJNodes crop branch before VSR receives CHW frames."""
    old_height, old_width = int(frames.shape[-3]), int(frames.shape[-2])
    if old_width / old_height > int(width) / int(height):
        crop_w, crop_h = round(old_height * int(width) / int(height)), old_height
    else:
        crop_w, crop_h = old_width, round(old_width * int(height) / int(width))
    return frames.narrow(-2, (old_width - crop_w) // 2, crop_w).narrow(-3, (old_height - crop_h) // 2, crop_h)


def _require_gpu_vsr_frames(torch_module: Any, frames: Any, label: str) -> None:
    if not bool(getattr(frames, "is_cuda", False)):
        raise H3RuntimeError(f"gpu-resident RTX VSR {label} must be CUDA; fallback is forbidden")
    if frames.dtype != torch_module.float32:
        raise H3RuntimeError(f"gpu-resident RTX VSR {label} must be float32 [0,1], got {frames.dtype}")
    if not bool(torch_module.isfinite(frames).all().item()):
        raise H3RuntimeError(f"gpu-resident RTX VSR {label} contains nonfinite values")
    minimum, maximum = float(frames.amin().item()), float(frames.amax().item())
    if minimum < 0.0 or maximum > 1.0:
        raise H3RuntimeError(f"gpu-resident RTX VSR {label} is outside [0,1]: min={minimum}, max={maximum}")


def _new_nvidia_rtx_vsr_context(nvvfx: Any) -> Any:
    return nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)


def _clone_nvidia_vsr_frame(torch_module: Any, dlpack_image: Any, index: int) -> Any:
    cloned = torch_module.from_dlpack(dlpack_image).clone()
    _require_gpu_vsr_frames(torch_module, cloned, f"cloned output frame {index}")
    return cloned


def _nvidia_rtx_vsr_resize_gpu_resident(frames: Any, width: int, height: int, _crop: str = "crop") -> Any:
    """Isolated GPU-resident equivalent of KJNodes' locked RTX VSR branch."""
    try:
        import nvvfx
        import torch
    except Exception as exc:
        raise H3RuntimeError(f"nvidia_rtx_vsr dependency unavailable: {exc}") from exc
    module_path = Path(nvvfx.__file__).resolve()
    private_root = (ROOT / "_private_site").resolve()
    if private_root not in module_path.parents:
        raise H3RuntimeError(f"nvidia-vfx must load from the 8775 private site: {module_path}")
    _require_gpu_vsr_frames(torch, frames, "input")
    cropped = _kj_center_crop_for_vsr(frames, int(width), int(height))
    frames_chw = cropped.movedim(-1, 1).contiguous()
    try:
        with _new_nvidia_rtx_vsr_context(nvvfx) as nvvfx_sr:
            nvvfx_sr.output_width = max(8, round(int(width) / 8) * 8)
            nvvfx_sr.output_height = max(8, round(int(height) / 8) * 8)
            nvvfx_sr.load()
            resized = torch.empty(
                (int(frames_chw.shape[0]), int(height), int(width), 3),
                device=frames.device, dtype=torch.float32,
            )
            for index in range(int(frames_chw.shape[0])):
                # NVIDIA requires this clone before the following run() invalidates the DLPack buffer.
                cloned = _clone_nvidia_vsr_frame(torch, nvvfx_sr.run(frames_chw[index]).image, index)
                resized[index].copy_(cloned.movedim(0, -1))
    except Exception as exc:
        raise H3RuntimeError(f"gpu-resident nvidia_rtx_vsr execution failed: {exc}") from exc
    if not bool(getattr(resized, "is_cuda", False)):
        raise H3RuntimeError("gpu-resident RTX VSR returned CPU frames; fallback is forbidden")
    _require_gpu_vsr_frames(torch, resized, "output")
    return resized


def _cuda_timed_stage(
    torch_module: Any, operation: Callable[[], Any], event_authority: str = "authoritative_pytorch_current_stream",
) -> tuple[Any, Dict[str, Any]]:
    """Measure a CUDA segment only after the device is synchronized at both edges."""
    started = time.perf_counter()
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)) or not cuda.is_available():
        result = operation()
        return result, {
            "wallSeconds": round(time.perf_counter() - started, 6),
            "cudaEventSeconds": None,
            "synchronization": "not_available",
            "wallSecondsAuthority": "wall_clock",
            "cudaEventAuthority": "not_available",
        }
    cuda.synchronize()
    start_event, end_event = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
    start_event.record()
    result = operation()
    end_event.record()
    cuda.synchronize()
    return result, {
        "wallSeconds": round(time.perf_counter() - started, 6),
        "cudaEventSeconds": round(float(start_event.elapsed_time(end_event)) / 1000.0, 6),
        "synchronization": "cuda.synchronize_pre_and_post",
        "wallSecondsAuthority": "authoritative_post_sync",
        "cudaEventAuthority": event_authority,
    }


def _tensor_device_name(value: Any) -> str:
    return str(getattr(value, "device", "unknown"))


def _frame_resize_vae_mode() -> str:
    mode = os.environ.get("H3_FRAME_RESIZE_VAE_MODE", "baseline_cpu_vendor").strip().lower()
    if mode == "gpu_resident_candidate":
        mode = "gpu_resident"
    if mode not in {"baseline_cpu_vendor", "gpu_resident", "gpu_resident_equivalence_probe"}:
        raise H3RuntimeError(f"unsupported H3_FRAME_RESIZE_VAE_MODE: {mode}")
    return mode


def _latent_upscale_mode() -> str:
    """Select the default latent path or an explicit VSR recovery path."""
    mode = os.environ.get("H3_LATENT_UPSCALE_MODE", "latent").strip().lower()
    if mode not in {"latent", "vsr_recovery"}:
        raise H3RuntimeError(f"unsupported H3_LATENT_UPSCALE_MODE: {mode}")
    return mode


def _latent_upscale_stage2(
    stage1_video_latent: Any,
    audio: Any,
    target_h: int,
    target_w: int,
    progress_step: Callable[[int, int], Any] | None = None,
) -> tuple[Any, Dict[str, Any]]:
    """Run the LBH 3D latent upscaler and return an isolated stage-2 input."""
    try:
        from latent_upscaler_adapter import load_vendor_adapter, resize_video_latent_3d

        model, model_receipt = load_vendor_adapter()
        model_to = getattr(model, "to", None)
        if callable(model_to):
            model_to(device=getattr(stage1_video_latent, "device", "cpu"))
        if callable(progress_step):
            progress_step(1, 3)
        source_h, source_w = (int(stage1_video_latent.shape[-2]), int(stage1_video_latent.shape[-1]))
        # The vendor model is intentionally called at its strict 2x latent
        # output size.  Some legal final canvases have odd latent axes, so the
        # even stage1 grid can be one cell larger; crop only the excess edge.
        lbh_output_h, lbh_output_w = source_h * 2, source_w * 2
        resized, resize_receipt = resize_video_latent_3d(
            stage1_video_latent, lbh_output_h, lbh_output_w, model
        )
        if callable(progress_step):
            progress_step(3, 3)
    except Exception as exc:
        if isinstance(exc, H3RuntimeError):
            raise
        raise H3RuntimeError(f"latent upscaler execution failed: {type(exc).__name__}: {exc}") from exc
    if resized is stage1_video_latent:
        raise H3RuntimeError("latent upscaler returned the stage1 video object unchanged")
    raw_output_h, raw_output_w = (int(resized.shape[-2]), int(resized.shape[-1]))
    if raw_output_h < int(target_h) or raw_output_w < int(target_w):
        raise H3RuntimeError("latent upscaler output is smaller than compiled target latent H/W")
    if raw_output_h != int(target_h) or raw_output_w != int(target_w):
        # Deterministic top-left edge crop; no interpolation and no target
        # canvas mutation are permitted on the LBH path.
        resized = resized[..., : int(target_h), : int(target_w)]
    if tuple(int(item) for item in resized.shape[-2:]) != (int(target_h), int(target_w)):
        raise H3RuntimeError("latent upscaler crop does not match compiled target latent H/W")
    weight = dict(model_receipt.get("weight") or {})
    receipt = {
        **dict(model_receipt),
        **dict(resize_receipt),
        "routeMode": "latent",
        "implementation": "LBH-123-AI.Comfyui_Minimax_h3_latent_Upscaler",
        "algorithm": "minimax_h3_latent_upscaler_3d",
        "videoLayout": "BCTHW",
        "videoChannels": 24,
        "preserveVideoT": True,
        "audioPolicy": "identity",
        "audioIdentityPreserved": True,
        "videoVaeRoundTrip": False,
        "nvidiaRtxVsr": False,
        "reNoise": {"required": True, "construction": "stage2_sampler_noise_and_beta_sigmas"},
        "weightPath": weight.get("path"),
        "weightSha256": weight.get("sha256"),
        "fallbackUsed": False,
        "rawLbhOutputLatent": {"height": raw_output_h, "width": raw_output_w},
        "cropWindow": {
            "strategy": "top_left_edge",
            "top": 0,
            "left": 0,
            "height": int(target_h),
            "width": int(target_w),
            "cropped": raw_output_h != int(target_h) or raw_output_w != int(target_w),
        },
        "finalStage2Latent": {"height": int(target_h), "width": int(target_w)},
    }
    return resized, receipt


def _frame_resize_vae_handler(mode: str) -> Callable[[Any, int, int, str], Any]:
    handlers = {
        "baseline_cpu_vendor": _nvidia_rtx_vsr_resize,
        "gpu_resident": _nvidia_rtx_vsr_resize_gpu_resident,
        "gpu_resident_equivalence_probe": _nvidia_rtx_vsr_resize_gpu_resident,
    }
    try:
        return handlers[mode]
    except KeyError as exc:
        raise H3RuntimeError(f"unsupported H3_FRAME_RESIZE_VAE_MODE: {mode}") from exc


def _gpu_resident_vsr_memory_admission(
    torch_module: Any, video_vae: _H3VideoVAEProxy, frames: Any, width: int, height: int,
) -> Dict[str, Any]:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "mem_get_info", None)):
        raise H3RuntimeError("gpu-resident RTX VSR requires CUDA memory telemetry; fallback is forbidden")
    free_bytes, total_bytes = [int(value) for value in cuda.mem_get_info()]
    vae = getattr(video_vae, "_vae", None)
    if vae is None or not all(hasattr(vae, name) for name in ("memory_used_encode", "vae_dtype", "latent_dim")):
        raise H3RuntimeError("gpu-resident RTX VSR cannot estimate VAE encode memory; fallback is forbidden")
    target_shape = (1, 3, int(frames.shape[0]), int(height), int(width))
    try:
        vae_encode_bytes = int(vae.memory_used_encode(target_shape, vae.vae_dtype))
    except Exception as exc:
        raise H3RuntimeError(f"gpu-resident RTX VSR VAE encode memory estimate failed: {exc}") from exc
    element_bytes = int(frames.element_size())
    source_height, source_width = int(frames.shape[-3]), int(frames.shape[-2])
    if source_width / source_height > int(width) / int(height):
        crop_width, crop_height = round(source_height * int(width) / int(height)), source_height
    else:
        crop_width, crop_height = source_width, round(source_width * int(height) / int(width))
    chw_contiguous_bytes = int(frames.shape[0]) * crop_height * crop_width * 3 * element_bytes
    resident_output_bytes = int(frames.shape[0]) * int(height) * int(width) * 3 * element_bytes
    single_frame_bytes = int(height) * int(width) * 3 * element_bytes
    headroom_bytes = max(1024 * 1024 * 1024, int(vae_encode_bytes * 0.10), single_frame_bytes * 2)
    required_bytes = chw_contiguous_bytes + resident_output_bytes + (single_frame_bytes * 2) + vae_encode_bytes + headroom_bytes
    receipt = {
        "freeBytesBeforeVsr": free_bytes,
        "totalBytes": total_bytes,
        "chwContiguousBytes": chw_contiguous_bytes,
        "residentOutputBytes": resident_output_bytes,
        "singleFrameNativeAndCloneBytes": single_frame_bytes * 2,
        "vaeEncodeEstimatedBytes": vae_encode_bytes,
        "headroomBytes": headroom_bytes,
        "requiredFreeBytes": required_bytes,
        "admitted": free_bytes >= required_bytes,
    }
    if not receipt["admitted"]:
        raise H3RuntimeError(f"gpu-resident RTX VSR memory admission failed: {receipt}")
    return receipt


def _decode_video_vae_for_frame_resize(
    video_vae: _H3VideoVAEProxy, latent: Any, route_mode: str, torch_module: Any,
) -> Any:
    if route_mode == "baseline_cpu_vendor":
        return video_vae.decode(latent)
    vae = getattr(video_vae, "_vae", None)
    if vae is None or not hasattr(vae, "output_device"):
        raise H3RuntimeError("gpu-resident RTX VSR requires a VAE output_device boundary; fallback is forbidden")
    device = getattr(latent, "device", None)
    if device is None or getattr(device, "type", str(device)) != "cuda":
        raise H3RuntimeError("gpu-resident RTX VSR requires CUDA stage1 latent; fallback is forbidden")
    previous_output_device = vae.output_device
    vae.output_device = device
    try:
        return video_vae.decode(latent)
    finally:
        vae.output_device = previous_output_device


def _color_anchor_strength() -> float:
    """Read the opt-in global color-anchor strength without changing the default path."""
    raw = os.environ.get("H3_COLOR_ANCHOR_STRENGTH")
    if raw is None or raw.strip().lower() in {"off", "0", "0.0"}:
        return 0.0
    try:
        strength = float(raw)
    except (TypeError, ValueError) as exc:
        raise H3RuntimeError("H3_COLOR_ANCHOR_STRENGTH must be a number in [0,1] or off") from exc
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise H3RuntimeError("H3_COLOR_ANCHOR_STRENGTH must be a number in [0,1] or off")
    return strength


def _validate_color_anchor_frames(
    torch_module: Any, frames: Any, label: str, expected_frames: Optional[int] = None,
) -> None:
    """Keep the experimental color boundary strict; never hide malformed VSR output."""
    if getattr(frames, "ndim", None) != 4 or int(frames.shape[-1]) != 3:
        raise H3RuntimeError(f"color anchor {label} must be [T,H,W,3], got {tuple(frames.shape)}")
    torch_module = torch_module if torch_module is not None else __import__("torch")
    if frames.dtype != torch_module.float32:
        raise H3RuntimeError(f"color anchor {label} must be float32 [0,1], got {frames.dtype}")
    if expected_frames is not None and int(frames.shape[0]) != int(expected_frames):
        raise H3RuntimeError(
            f"color anchor {label} frame count mismatch: {int(frames.shape[0])} != {int(expected_frames)}"
        )
    if not bool(torch_module.isfinite(frames).all().item()):
        raise H3RuntimeError(f"color anchor {label} contains nonfinite values")
    minimum, maximum = float(frames.amin().item()), float(frames.amax().item())
    if minimum < 0.0 or maximum > 1.0:
        raise H3RuntimeError(f"color anchor {label} is outside [0,1]: min={minimum}, max={maximum}")


def _color_anchor_stats(torch_module: Any, frames: Any) -> Dict[str, float]:
    weights = frames.new_tensor([0.2126, 0.7152, 0.0722])
    luma = (frames * weights).sum(dim=-1)
    chroma = frames.amax(dim=-1) - frames.amin(dim=-1)
    return {
        "lumaMean": float(luma.mean().item()),
        "lumaStd": float(luma.std(unbiased=False).item()),
        "chromaMean": float(chroma.mean().item()),
        "chromaStd": float(chroma.std(unbiased=False).item()),
    }


def _apply_color_anchor(
    decoded: Any, resized: Any, target_width: int, target_height: int, strength: float,
    torch_module: Any,
) -> tuple[Any, Dict[str, Any]]:
    """Apply one global, cross-frame luma/contrast/chroma affine correction."""
    _validate_color_anchor_frames(torch_module, decoded, "source", expected_frames=int(resized.shape[0]))
    _validate_color_anchor_frames(torch_module, resized, "target", expected_frames=int(decoded.shape[0]))
    source = _kj_center_crop_for_vsr(decoded, int(target_width), int(target_height))
    _validate_color_anchor_frames(torch_module, source, "cropped source", expected_frames=int(decoded.shape[0]))
    source_stats = _color_anchor_stats(torch_module, source)
    target_stats = _color_anchor_stats(torch_module, resized)
    weights = resized.new_tensor([0.2126, 0.7152, 0.0722])
    target_luma = (resized * weights).sum(dim=-1)
    source_std = source_stats["lumaStd"]
    target_std = target_stats["lumaStd"]
    contrast_scale = source_std / target_std if target_std > 1e-6 else 1.0
    chroma_scale = (
        source_stats["chromaMean"] / target_stats["chromaMean"]
        if target_stats["chromaMean"] > 1e-6 else 1.0
    )
    target_luma_adjusted = (
        (target_luma - target_stats["lumaMean"]) * contrast_scale + source_stats["lumaMean"]
    )
    corrected = target_luma_adjusted.unsqueeze(-1) + (
        resized - target_luma.unsqueeze(-1)
    ) * chroma_scale
    corrected = (resized + float(strength) * (corrected - resized)).clamp_(0.0, 1.0)
    _validate_color_anchor_frames(torch_module, corrected, "corrected", expected_frames=int(resized.shape[0]))
    return corrected, {
        "enabled": True,
        "strength": float(strength),
        "method": "global_luma_contrast_chroma",
        "sourceStats": source_stats,
        "targetStats": target_stats,
        "applied": True,
    }


def _frame_equality_receipt(baseline: Any, candidate: Any) -> Dict[str, Any]:
    """Compare serial VSR branches without persisting frame pixels."""
    try:
        import torch
    except Exception as exc:
        raise H3RuntimeError(f"frame equality requires torch: {exc}") from exc
    if tuple(baseline.shape) != tuple(candidate.shape) or baseline.dtype != candidate.dtype:
        return {
            "schema": "torch_equal_per_frame_v1",
            "frameCount": int(baseline.shape[0]),
            "allFramesExactlyEqual": False,
            "firstMismatch": {
                "index": 0,
                "baselineShape": list(baseline.shape),
                "candidateShape": list(candidate.shape),
                "baselineDtype": str(baseline.dtype),
                "candidateDtype": str(candidate.dtype),
            },
        }
    baseline_hashes, candidate_hashes = [], []
    first_mismatch = None
    for index in range(int(baseline.shape[0])):
        baseline_frame = baseline[index].detach().cpu().contiguous()
        candidate_frame = candidate[index].detach().cpu().contiguous()
        baseline_hash = hashlib.sha256(baseline_frame.numpy().tobytes()).hexdigest()
        candidate_hash = hashlib.sha256(candidate_frame.numpy().tobytes()).hexdigest()
        baseline_hashes.append(baseline_hash)
        candidate_hashes.append(candidate_hash)
        if first_mismatch is None and not torch.equal(baseline_frame, candidate_frame):
            first_mismatch = {
                "index": index,
                "baselineSha256": baseline_hash,
                "candidateSha256": candidate_hash,
                "shape": list(baseline_frame.shape),
                "dtype": str(baseline_frame.dtype),
                "maxAbsDiff": float((baseline_frame - candidate_frame).abs().max().item()),
            }
    return {
        "schema": "torch_equal_per_frame_v1",
        "frameCount": int(baseline.shape[0]),
        "shape": list(baseline.shape[1:]),
        "dtype": str(baseline.dtype),
        "baselineSha256": baseline_hashes,
        "candidateSha256": candidate_hashes,
        "allFramesExactlyEqual": first_mismatch is None,
        "firstMismatch": first_mismatch,
    }


def _frame_resize_vae_reencode(
    stage1_video_latent: Any,
    video_vae: _H3VideoVAEProxy,
    resize: Callable[[Any, int, int, str], Any],
    target_width: int,
    target_height: int,
    progress_step: Callable[[int, int], Any] | None = None,
    route_mode: str = "baseline_cpu_vendor",
    torch_module: Any = None,
) -> tuple[Any, Dict[str, Any]]:
    if route_mode not in {"baseline_cpu_vendor", "gpu_resident", "gpu_resident_candidate", "gpu_resident_equivalence_probe"}:
        raise H3RuntimeError(f"unsupported frame resize VAE route mode: {route_mode}")
    color_anchor_strength = _color_anchor_strength()
    torch_module = torch_module if torch_module is not None else getattr(video_vae, "_torch", None)
    total_started = time.perf_counter()
    decoded, decode_timing = _cuda_timed_stage(
        torch_module, lambda: _decode_video_vae_for_frame_resize(video_vae, stage1_video_latent, route_mode, torch_module)
    )
    if callable(progress_step):
        progress_step(1, 3)
    if decoded.ndim == 5:
        if int(decoded.shape[0]) != 1:
            raise H3RuntimeError(f"stage1 video VAE decode returned unsupported batch: {tuple(decoded.shape)}")
        decoded = decoded[0]
    if decoded.ndim != 4 or int(decoded.shape[-1]) != 3:
        raise H3RuntimeError(f"stage1 video VAE decode must return [T,H,W,3], got {tuple(decoded.shape)}")
    source_canvas = {"width": int(decoded.shape[2]), "height": int(decoded.shape[1])}
    source_frames = int(decoded.shape[0])
    decoded_device = _tensor_device_name(decoded)
    memory_admission = None
    equivalence = None
    if route_mode != "baseline_cpu_vendor":
        if not bool(getattr(decoded, "is_cuda", False)) or getattr(decoded, "dtype", None) is None:
            raise H3RuntimeError("gpu-resident RTX VSR decode did not return CUDA frames; fallback is forbidden")
        _require_gpu_vsr_frames(torch_module, decoded, "decoded output")
        memory_admission = _gpu_resident_vsr_memory_admission(
            torch_module, video_vae, decoded, int(target_width), int(target_height)
        )
    if route_mode == "gpu_resident_equivalence_probe":
        baseline_resized, baseline_vsr_timing = _cuda_timed_stage(
            torch_module, lambda: _nvidia_rtx_vsr_resize(decoded, int(target_width), int(target_height), "disabled"),
            event_authority="non_authoritative_stream_not_proven",
        )
        resized, candidate_vsr_timing = _cuda_timed_stage(
            torch_module, lambda: _nvidia_rtx_vsr_resize_gpu_resident(decoded, int(target_width), int(target_height), "disabled"),
            event_authority="non_authoritative_stream_not_proven",
        )
        equivalence = _frame_equality_receipt(baseline_resized, resized)
        if not equivalence["allFramesExactlyEqual"]:
            raise H3RuntimeError(f"gpu-resident RTX VSR frame equality failed: {equivalence['firstMismatch']}")
        vsr_timing = {"baseline": baseline_vsr_timing, "candidate": candidate_vsr_timing}
    else:
        resized, vsr_timing = _cuda_timed_stage(
            torch_module, lambda: resize(decoded, int(target_width), int(target_height), "disabled"),
            event_authority="non_authoritative_stream_not_proven",
        )
    if callable(progress_step):
        progress_step(2, 3)
    if tuple(int(item) for item in resized.shape[1:3]) != (int(target_height), int(target_width)):
        raise H3RuntimeError("frame resize did not produce the compiled stage2 target canvas")
    color_anchor = {
        "enabled": False,
        "strength": 0.0,
        "method": "disabled",
        "sourceStats": None,
        "targetStats": None,
        "applied": False,
    }
    if color_anchor_strength > 0.0:
        resized, color_anchor = _apply_color_anchor(
            decoded, resized, int(target_width), int(target_height), color_anchor_strength, torch_module
        )
    if route_mode != "baseline_cpu_vendor":
        _require_gpu_vsr_frames(torch_module, resized, "VAE encode input")
        del decoded
    encoded, encode_timing = _cuda_timed_stage(torch_module, lambda: video_vae.encode(resized))
    if callable(progress_step):
        progress_step(3, 3)
    device_path = {
        "vaeDecodeOutput": decoded_device,
        "vsrInput": "cpu" if route_mode == "baseline_cpu_vendor" else "cuda",
        "vsrOutput": _tensor_device_name(resized),
        "vaeEncodeInput": _tensor_device_name(resized),
        "transfers": (
            ["gpu_to_cpu_before_vsr", "cpu_to_gpu_for_vsr", "gpu_to_cpu_after_vsr"]
            if route_mode == "baseline_cpu_vendor" else []
        ),
    }
    timing_parts = [decode_timing, encode_timing]
    if isinstance(vsr_timing, dict) and "cudaEventSeconds" in vsr_timing:
        timing_parts.append(vsr_timing)
    else:
        timing_parts.extend(value for value in (vsr_timing or {}).values() if isinstance(value, dict))
    event_seconds = [part.get("cudaEventSeconds") for part in timing_parts]
    total_cuda_event_seconds = None if any(value is None for value in event_seconds) else round(sum(event_seconds), 6)
    return encoded, {
        "implementation": "Kijai.ComfyUI-KJNodes.ImageResizeKJv2",
        "sourceCommit": "60cd6bc1870db94c6eeb05fbe455147a8e91c4e9",
        "algorithm": "nvidia_rtx_vsr",
        "quality": "ULTRA",
        "package": "nvidia-vfx==0.1.0.1",
        "vfxSdkVersion": "1.2.0.0",
        "keepProportion": "crop",
        "cropPosition": "center",
        "device": "cpu" if route_mode == "baseline_cpu_vendor" else "gpu",
        "routeMode": route_mode,
        "devicePath": device_path,
        "gpuResidentMemoryAdmission": memory_admission,
        "frameEquality": equivalence,
        "stageTimingsSeconds": {
            "vaeDecode": decode_timing,
            "vsr": vsr_timing,
            "vaeEncode": encode_timing,
            "total": {
                "wallSeconds": round(time.perf_counter() - total_started, 6),
                "cudaEventSeconds": total_cuda_event_seconds,
                "synchronization": "sum_of_segment_events_post_sync" if total_cuda_event_seconds is not None else "not_available",
            },
        },
        "sourceCanvas": source_canvas,
        "targetCanvas": {"width": int(target_width), "height": int(target_height)},
        "sourceFrameCount": source_frames,
        "outputFrameCount": int(resized.shape[0]),
        "videoVaeDecodePath": video_vae.last_decode_path,
        "videoVaeEncodePath": video_vae.last_encode_path,
        "videoVaeRoundTrip": True,
        "nvidiaRtxVsr": True,
        "fallbackUsed": False,
        "colorAnchor": color_anchor,
    }


class _H3AudioVAEProxy:
    """Preserve official audio VAE device management for reference/output audio."""

    def __init__(self, vae: Any, patcher: Any = None, model_management: Any = None, torch_module: Any = None) -> None:
        self._vae = vae
        self._patcher = patcher
        self._model_management = model_management
        self._torch = torch_module
        self._device_load_requested = False
        self._official_wrapper = hasattr(vae, "latent_dim") and hasattr(vae, "patcher")

    def _inference_call(self, fn: Callable[[], Any]) -> Any:
        if self._torch is None:
            return fn()
        inference_mode = getattr(self._torch, "inference_mode", None)
        if inference_mode is not None:
            with inference_mode():
                return fn()
        no_grad = getattr(self._torch, "no_grad", None)
        if no_grad is not None:
            with no_grad():
                return fn()
        return fn()

    def _ensure_loaded(self) -> None:
        if self._patcher is not None and self._model_management is not None and not self._device_load_requested:
            self._model_management.load_models_gpu([self._patcher], force_full_load=False)
            self._device_load_requested = True

    def require_device_load(self) -> None:
        self._device_load_requested = False

    def encode(self, value: Any) -> Any:
        if self._official_wrapper:
            return self._inference_call(lambda: self._vae.encode(value))
        self._ensure_loaded()
        return self._inference_call(lambda: self._vae.encode(value))

    def decode(self, value: Any) -> Any:
        if self._official_wrapper:
            return self._inference_call(lambda: self._vae.decode(value))
        self._ensure_loaded()
        return self._inference_call(lambda: self._vae.decode(value))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vae, name)


def _node_args(output: Any) -> tuple[Any, ...]:
    return tuple(output.args) if hasattr(output, "args") else tuple(output)


def _build_stage2_keyframe_conditioning(
    h3: Any, clip: Any, video_vae: _H3VideoVAEProxy, prompt: str,
    width: int, height: int, length: int, first_frame: Any, last_frame: Any,
) -> Any:
    output = h3.MiniMaxH3ImageToVideo.execute(
        clip, video_vae, prompt, width, height, length,
        first_frame=first_frame, last_frame=last_frame,
    )
    return _node_args(output)[0]


def _official_h3_sampler_contract(steps: int) -> Dict[str, Any]:
    """Public MiniMax H3 R2V sampling semantics, independent of UI options."""

    return official_h3_sampler_contract(steps)


def _sampler_prepare_receipt(sampler_contract: Dict[str, Any], latent: Dict[str, Any]) -> Dict[str, Any]:
    """Map the official sampler contract to the shared receipt shape."""

    return {
        "sampler": sampler_contract["sampler"],
        "scheduler": sampler_contract["scheduler"],
        "steps": sampler_contract["steps"],
        "denoise": sampler_contract["denoise"],
        "guider": sampler_contract["guider"],
        "conditioningInputs": sampler_contract["conditioningInputs"],
        "usesNegativeConditioning": sampler_contract["usesNegativeConditioning"],
        "cfg": sampler_contract["cfg"],
        "noise": sampler_contract["noise"],
        "executor": sampler_contract["executor"],
        "latentShape": _shape(latent["samples"]),
    }


def _receipt_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _stage_timings_receipt(runtime_stages: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize observed phase durations without using them as control input."""

    sources = {
        "bootstrap": "bootstrap",
        "textEncoderLoad": "text_encoder_load",
        "videoVaeLoad": "video_vae_load",
        "audioVaeLoad": "audio_vae_load",
        "primaryModelLoad": "primary_model_load",
        "decodeVideoVaeLoad": "decode_video_vae_load",
        "decodeAudioVaeLoad": "decode_audio_vae_load",
        "referenceConditioning": "reference_conditioning",
        "qwenConditioning": "qwen_encode_from_tokens_scheduled",
        "primarySampling": "primary_sampling",
        "videoDecode": "decode_video_vae",
        "audioDecode": "decode_audio_vae",
        "mux": "mp4_mux",
        "decodeExport": "decode_export",
        "denoiserForward": "denoiser_forward",
    }
    timings: Dict[str, Any] = {}
    for receipt_name, stage_name in sources.items():
        source = runtime_stages.get(stage_name)
        if isinstance(source, dict) and source.get("elapsedSeconds") is not None:
            timings[receipt_name] = {"elapsedSeconds": source.get("elapsedSeconds")}
    return timings


def _hardware_identity_receipt(torch_module: Any) -> Dict[str, Any]:
    """Capture stable comparison identity without affecting model execution."""

    receipt: Dict[str, Any] = {
        "status": "unavailable",
        "reason": "hardware_identity_unavailable",
        "torchVersion": str(getattr(torch_module, "__version__", "") or "") or None,
        "cudaVersion": str(getattr(getattr(torch_module, "version", None), "cuda", "") or "") or None,
    }
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.0, check=False,
        )
        values = [item.strip() for item in query.stdout.splitlines()[0].split(",", 2)]
        if query.returncode != 0 or len(values) != 3 or not all(values):
            receipt["reason"] = "nvidia_smi_identity_failed"
            return receipt
        receipt.update({
            "status": "available",
            "gpuUuid": values[0],
            "gpuName": values[1],
            "driverVersion": values[2],
        })
        receipt.pop("reason", None)
        if not receipt["torchVersion"] or not receipt["cudaVersion"]:
            receipt.update({"status": "unavailable", "reason": "torch_or_cuda_identity_missing"})
    except (OSError, IndexError, subprocess.TimeoutExpired) as exc:
        receipt["reason"] = f"nvidia_smi_identity_unavailable:{type(exc).__name__}"
    return receipt


def _split_two_stage_sigmas(sigmas: Any, split_step: int, total_steps: int) -> tuple[Any, Any]:
    """Split one sigma trajectory at a shared boundary."""
    split_step = int(split_step)
    total_steps = int(total_steps)
    if split_step < 1 or split_step >= total_steps:
        raise H3RuntimeError("splitStep must be between 1 and total steps - 1")
    first = sigmas[: split_step + 1]
    second = sigmas[split_step:]
    if first.shape[-1] + second.shape[-1] - 1 != total_steps + 1 or first[-1] != second[0]:
        raise H3RuntimeError("two-stage sigma split is not contiguous")
    return first, second


def _clone_condition_value(value: Any, key: str | None = None, memo: Dict[int, Any] | None = None, visiting: set[int] | None = None) -> Any:
    memo = {} if memo is None else memo
    visiting = set() if visiting is None else visiting
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    identity = id(value)
    if identity in visiting:
        raise H3RuntimeError(f"condition graph cycle detected at {type(value).__name__}")
    if identity in memo:
        return memo[identity]
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "device"):
        return value
    visiting.add(identity)
    try:
        if isinstance(value, dict):
            cloned: Dict[Any, Any] = {}
            memo[identity] = cloned
            cloned.update({item_key: _clone_condition_value(item, str(item_key), memo, visiting) for item_key, item in value.items()})
            return cloned
        if isinstance(value, list):
            cloned_list: list[Any] = []
            memo[identity] = cloned_list
            cloned_list.extend(_clone_condition_value(item, key, memo, visiting) for item in value)
            return cloned_list
        if isinstance(value, tuple):
            cloned_tuple = tuple(_clone_condition_value(item, key, memo, visiting) for item in value)
            memo[identity] = cloned_tuple
            return cloned_tuple
        is_control = key == "control" or (hasattr(value, "previous_controlnet") and hasattr(value, "pre_run") and hasattr(value, "cleanup"))
        if is_control:
            copy_method = getattr(value, "copy", None)
            if not callable(copy_method):
                raise H3RuntimeError(f"control {type(value).__name__} has no safe copy interface")
            cloned_control = copy_method()
            if cloned_control is value:
                raise H3RuntimeError(f"control {type(value).__name__} copy returned the original object")
            memo[identity] = cloned_control
            previous = getattr(value, "previous_controlnet", None)
            cloned_previous = _clone_condition_value(previous, "control", memo, visiting) if previous is not None else None
            setter = getattr(cloned_control, "set_previous_controlnet", None)
            if callable(setter): setter(cloned_previous)
            elif hasattr(cloned_control, "previous_controlnet"): cloned_control.previous_controlnet = cloned_previous
            elif cloned_previous is not None: raise H3RuntimeError(f"control {type(value).__name__} cannot preserve previous_controlnet")
            source_multigpu = getattr(value, "multigpu_clones", {}) or {}
            if source_multigpu:
                if not hasattr(cloned_control, "multigpu_clones"): raise H3RuntimeError(f"control {type(value).__name__} cannot preserve multigpu_clones")
                cloned_control.multigpu_clones = {device: _clone_condition_value(item, "control", memo, visiting) for device, item in source_multigpu.items()}
            return cloned_control
        clone_method = getattr(value, "clone", None)
        is_hook_group = callable(clone_method) and hasattr(value, "get_type")
        is_gpu_option = callable(clone_method) and hasattr(value, "device_index") and hasattr(value, "relative_speed")
        is_gpu_group = hasattr(value, "options") and isinstance(value.options, dict) and callable(getattr(value, "add", None))
        if is_gpu_group:
            try: cloned_group = type(value)()
            except Exception as exc: raise H3RuntimeError(f"GPUOptionsGroup {type(value).__name__} cannot be reconstructed: {exc}") from exc
            memo[identity] = cloned_group
            for option in value.options.values(): cloned_group.add(_clone_condition_value(option, "gpu_options", memo, visiting))
            if any(cloned_group.options[k] is value.options[k] for k in value.options): raise H3RuntimeError("GPUOptionsGroup clone shares mutable GPUOptions")
            return cloned_group
        if key in {"hooks", "extra_hooks", "gpu_options"} or is_hook_group or is_gpu_option:
            if not callable(clone_method): raise H3RuntimeError(f"condition object {type(value).__name__} has no clone interface")
            cloned_object = clone_method()
            if cloned_object is value: raise H3RuntimeError(f"condition object {type(value).__name__} clone returned the original object")
            memo[identity] = cloned_object
            return cloned_object
        return value
    finally:
        visiting.discard(identity)


def _clone_condition_map(conds: Dict[str, Any]) -> Dict[str, Any]:
    return _clone_condition_value(conds, memo={}, visiting=set())

def _control_roots(value: Any, key: str | None = None) -> list[Any]:
    roots: list[Any] = []
    if isinstance(value, dict):
        for item_key, item in value.items(): roots.extend(_control_roots(item, str(item_key)))
    elif isinstance(value, (list, tuple)):
        for item in value: roots.extend(_control_roots(item, key))
    elif key == "control" or (hasattr(value, "previous_controlnet") and hasattr(value, "cleanup")):
        roots.append(value)
    return roots


def _unique_control_graph(conditions: Dict[str, Any]) -> list[Any]:
    ordered: list[Any] = []
    visited: set[int] = set()
    visiting: set[int] = set()
    def visit(control: Any) -> None:
        identity = id(control)
        if identity in visiting: raise H3RuntimeError(f"control cleanup graph cycle detected at {type(control).__name__}")
        if identity in visited: return
        visiting.add(identity)
        previous = getattr(control, "previous_controlnet", None)
        if previous is not None: visit(previous)
        for child in (getattr(control, "multigpu_clones", {}) or {}).values(): visit(child)
        visiting.remove(identity); visited.add(identity); ordered.append(control)
    for root in _control_roots(conditions): visit(root)
    return ordered


def _strip_control_fields(value: Any) -> Any:
    if isinstance(value, dict): return {key: _strip_control_fields(item) for key, item in value.items() if key != "control"}
    if isinstance(value, list): return [_strip_control_fields(item) for item in value]
    if isinstance(value, tuple): return tuple(_strip_control_fields(item) for item in value)
    return value


def _cleanup_condition_resources(conditions: Dict[str, Any], models: list[Any], cleanup_models: Callable[..., Any]) -> None:
    controls = _unique_control_graph(conditions)
    saved = [(control, getattr(control, "previous_controlnet", None), getattr(control, "multigpu_clones", None)) for control in controls]
    errors: list[BaseException] = []
    try:
        for control, _previous, multigpu in saved:
            if hasattr(control, "previous_controlnet"): control.previous_controlnet = None
            if multigpu is not None: control.multigpu_clones = {}
        for control, _previous, _multigpu in saved:
            try: control.cleanup()
            except BaseException as exc: errors.append(exc)
        try: cleanup_models(_strip_control_fields(conditions), models)
        except BaseException as exc: errors.append(exc)
    finally:
        for control, previous, multigpu in saved:
            if hasattr(control, "previous_controlnet"): control.previous_controlnet = previous
            if multigpu is not None: control.multigpu_clones = multigpu
    if errors: raise ExceptionGroup("condition resource cleanup failures", errors)


def _exception_metadata_warning(runtime_stages: Dict[str, Any] | None, attribute: str, error: BaseException) -> None:
    if not isinstance(runtime_stages, dict):
        return
    try:
        runtime_stages.setdefault("exceptionMetadataWarnings", []).append({
            "attribute": attribute,
            "error": f"{type(error).__name__}: {error}",
        })
    except BaseException:
        pass


def _best_effort_exception_metadata(error: BaseException, attribute: str, value: Any, runtime_stages: Dict[str, Any] | None) -> bool:
    try:
        setattr(error, attribute, value)
        return True
    except BaseException as metadata_error:
        _exception_metadata_warning(runtime_stages, attribute, metadata_error)
        return False


def _best_effort_exception_note(error: BaseException, note: str, runtime_stages: Dict[str, Any] | None) -> None:
    try:
        error.add_note(note)
    except BaseException as metadata_error:
        _exception_metadata_warning(runtime_stages, "__notes__", metadata_error)


def _exception_metadata_value(error: BaseException, attribute: str, fallback: Any, runtime_stages: Dict[str, Any] | None) -> Any:
    try:
        return getattr(error, attribute, fallback)
    except BaseException as metadata_error:
        _exception_metadata_warning(runtime_stages, f"{attribute}.read", metadata_error)
        return fallback


def _exception_runtime_stages(error: BaseException, fallback: Dict[str, Any]) -> Dict[str, Any]:
    value = _exception_metadata_value(error, "runtimeStages", None, fallback)
    return value if isinstance(value, dict) else fallback


def _run_renoise_lifecycle(guider: Any, deps: Dict[str, Callable[..., Any]], first_stage: Callable[[], Any], second_stage: Callable[[Any], Any]) -> Any:
    """Coordinate two inner stages with pristine conditions and unconditional restoration."""
    original_model_options = guider.model_options
    original_hook_mode = guider.model_patcher.hook_mode
    loaded_models: list[Any] = []
    multigpu_patchers: list[Any] = []
    primary_error: BaseException | None = None
    cleanup_errors: list[BaseException] = []
    result: Any = None
    try:
        guider.conds = _clone_condition_map(guider.original_conds)
        deps["preprocess"](guider.conds)
        guider.model_options = deps["clone_options"](original_model_options)
        if deps["hook_count"](guider.conds) <= 1:
            guider.model_patcher.hook_mode = deps["min_hook_mode"]
        deps["prepare_patcher"](guider.model_patcher, guider.conds, guider.model_options)
        deps["filter_hooks"](guider.conds, guider.model_options)
        guider.inner_model, guider.conds, loaded_models = deps["prepare_sampling"](guider.model_patcher, guider.conds, guider.model_options)
        guider.loaded_models = loaded_models
        pristine_conds = _clone_condition_map(guider.conds)
        multigpu_patchers = deps["prepare_multigpu"](guider.model_patcher, loaded_models, guider.model_options)
        deps["enter_runtime"]()
        deps["pre_run"](guider.model_patcher, multigpu_patchers, guider.model_options)
        guider.conds = _clone_condition_map(pristine_conds)
        first_output = first_stage()
        deps["cleanup_stage_conds"](guider.conds)
        guider.conds = _clone_condition_map(pristine_conds)
        result = second_stage(first_output)
    except BaseException as exc:
        primary_error = exc
    finally:
        actions = (
            lambda: deps["exit_runtime"](),
            lambda: deps["cleanup_models"](getattr(guider, "conds", {}), loaded_models),
            lambda: deps["cleanup_multigpu"](multigpu_patchers),
            lambda: deps["cleanup_patcher"](guider.model_patcher),
            lambda: deps["cast_offload"](guider.model_options, guider.model_patcher),
            lambda: setattr(guider, "model_options", original_model_options),
            lambda: setattr(guider.model_patcher, "hook_mode", original_hook_mode),
            lambda: deps["restore_patches"](guider.model_patcher),
            lambda: deps["delete_runtime"](guider),
        )
        for action in actions:
            try:
                action()
            except BaseException as exc:
                cleanup_errors.append(exc)
    if primary_error is not None:
        messages = [f"{type(error).__name__}: {error}" for error in cleanup_errors]
        if messages:
            runtime_stages = _exception_runtime_stages(primary_error, {})
            runtime_stages["resourceRecovery"] = {"failedStage": "resource_cleanup", "errors": messages}
            _best_effort_exception_metadata(primary_error, "cleanup_errors", messages, runtime_stages)
            _best_effort_exception_metadata(primary_error, "resourceRecoveryFailedStage", "resource_cleanup", runtime_stages)
            _best_effort_exception_metadata(primary_error, "runtimeStages", runtime_stages, runtime_stages)
            for message in messages:
                _best_effort_exception_note(primary_error, f"resource recovery failure: {message}", runtime_stages)
        raise primary_error
    if cleanup_errors:
        messages = [f"{type(error).__name__}: {error}" for error in cleanup_errors]
        failure = ExceptionGroup("renoise resource recovery failures", cleanup_errors)
        runtime_stages = {
            "failedStage": "resource_cleanup",
            "resourceRecovery": {"failedStage": "resource_cleanup", "errors": messages},
        }
        _best_effort_exception_metadata(failure, "failedStage", "resource_cleanup", runtime_stages)
        _best_effort_exception_metadata(failure, "resourceRecoveryFailedStage", "resource_cleanup", runtime_stages)
        _best_effort_exception_metadata(failure, "cleanup_errors", messages, runtime_stages)
        _best_effort_exception_metadata(failure, "runtimeStages", runtime_stages, runtime_stages)
        raise failure
    return result


def _av_components(value: Any, label: str) -> list[Any]:
    values = list(value.unbind()) if getattr(value, "is_nested", False) else [value]
    if not values or any(not hasattr(item, "shape") or not hasattr(item, "dtype") or not hasattr(item, "device") for item in values):
        raise H3RuntimeError(f"{label} must contain tensor components")
    return values


def _av_component_signature(value: Any) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), str(value.device)


def _validate_av_components(values: list[Any], latent_shapes: list[tuple[int, ...]], label: str) -> None:
    if len(values) != len(latent_shapes):
        raise H3RuntimeError(f"{label} component count mismatch: expected {len(latent_shapes)}, got {len(values)}")
    expected_dtype = str(values[0].dtype)
    expected_device = str(values[0].device)
    for index, (value, shape) in enumerate(zip(values, latent_shapes)):
        if tuple(value.shape) != tuple(shape):
            raise H3RuntimeError(f"{label} component {index} shape mismatch: expected {tuple(shape)}, got {tuple(value.shape)}")
        if int(value.numel()) != math.prod(tuple(shape)):
            raise H3RuntimeError(f"{label} component {index} element count mismatch")
        if str(value.dtype) != expected_dtype or str(value.device) != expected_device:
            raise H3RuntimeError(f"{label} component {index} dtype/device mismatch")


def _pack_av_components(values: list[Any], latent_shapes: list[tuple[int, ...]], utils: Any, label: str) -> Any:
    _validate_av_components(values, latent_shapes, label)
    expected_elements = sum(math.prod(tuple(shape)) for shape in latent_shapes)
    packed, packed_shapes = utils.pack_latents(values)
    if not hasattr(packed, "numel"):
        raise H3RuntimeError(f"{label} pack did not return a tensor")
    if len(packed_shapes) != len(latent_shapes) or [tuple(shape) for shape in packed_shapes] != [tuple(shape) for shape in latent_shapes]:
        raise H3RuntimeError(f"{label} pack shape receipt mismatch")
    if int(packed.numel()) != expected_elements:
        raise H3RuntimeError(f"{label} packed element count mismatch: expected {expected_elements}, got {int(packed.numel())}")
    return packed


def _unpack_av_components(packed: Any, latent_shapes: list[tuple[int, ...]], utils: Any, label: str) -> list[Any]:
    expected_elements = sum(math.prod(tuple(shape)) for shape in latent_shapes)
    if int(packed.numel()) != expected_elements:
        raise H3RuntimeError(f"{label} packed element count mismatch: expected {expected_elements}, got {int(packed.numel())}")
    values = list(utils.unpack_latents(packed, latent_shapes))
    if len(values) != len(latent_shapes):
        raise H3RuntimeError(f"{label} unpack component count mismatch: expected {len(latent_shapes)}, got {len(values)}")
    for index, (value, shape) in enumerate(zip(values, latent_shapes)):
        if tuple(value.shape) != tuple(shape):
            raise H3RuntimeError(f"{label} unpack component {index} shape mismatch: expected {tuple(shape)}, got {tuple(value.shape)}")
        if str(value.dtype) != str(packed.dtype) or str(value.device) != str(packed.device):
            raise H3RuntimeError(f"{label} unpack component {index} dtype/device mismatch")
    return values


def _prepare_av_noise_mask(mask: Any, latent_shapes: list[tuple[int, ...]], device: Any, sampler_helpers: Any, utils: Any, torch_module: Any) -> Any:
    if mask is None:
        return None
    masks = _av_components(mask, "noise_mask")
    if len(masks) != len(latent_shapes):
        raise H3RuntimeError(f"noise_mask component count mismatch: expected {len(latent_shapes)}, got {len(masks)}")
    prepared = []
    for index, (item, shape) in enumerate(zip(masks, latent_shapes)):
        if tuple(item.shape) != tuple(shape):
            raise H3RuntimeError(f"noise_mask component {index} shape mismatch: expected {tuple(shape)}, got {tuple(item.shape)}")
        value = sampler_helpers.prepare_mask(item, shape, device)
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(shape):
            raise H3RuntimeError(f"noise_mask component {index} prepared shape mismatch: expected {tuple(shape)}, got {getattr(value, 'shape', None)}")
        prepared.append(value)
    return _pack_av_components(prepared, latent_shapes, utils, "noise_mask").to(device=device, dtype=torch_module.float32)


def _tensor_component_stats(
    value: Any,
    max_samples: int = 4096,
    finite_scan_chunk_size: int = 65536,
    torch_module: Any | None = None,
    sync_to_list: Callable[[Any], list[Any]] | None = None,
) -> Dict[str, Any]:
    torch_module = torch_module or __import__("torch")
    sync_to_list = sync_to_list or (lambda tensor: tensor.tolist())
    total_count = int(value.numel())
    sample_count = min(total_count, int(max_samples))
    chunk_size = max(1, int(finite_scan_chunk_size))
    chunk_count = (total_count + chunk_size - 1) // chunk_size
    stats: Dict[str, Any] = {
        "shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
        "elements": total_count, "totalCount": total_count, "sampleCount": sample_count,
        "sampled": sample_count < total_count, "empty": total_count == 0,
        "finite": True, "nonfinite": 0,
        "finiteScanChunkSize": chunk_size, "finiteScanChunkCount": chunk_count,
    }
    if total_count == 0:
        return stats
    flat = value.detach().reshape(-1)
    finite_count = None
    for start in range(0, total_count, chunk_size):
        chunk_count_value = torch_module.isfinite(flat[start:start + chunk_size]).sum(dtype=torch_module.int64)
        finite_count = chunk_count_value if finite_count is None else finite_count + chunk_count_value
    if sample_count < total_count:
        indices = torch_module.arange(sample_count, device=flat.device, dtype=torch_module.int64)
        indices = (indices * total_count // sample_count).clamp_max(total_count - 1)
        sample = flat.index_select(0, indices)
    else:
        sample = flat
    reduced = sample.to(dtype=torch_module.float32)
    combined = torch_module.stack((
        finite_count.to(dtype=torch_module.float32),
        reduced.min(), reduced.max(), reduced.mean(),
        reduced.std(unbiased=False), reduced.square().mean().sqrt(),
    ))
    summary = sync_to_list(combined)
    nonfinite_count = total_count - int(summary[0])
    stats.update({"finite": nonfinite_count == 0, "nonfinite": nonfinite_count})
    if nonfinite_count:
        return stats
    stats.update({
        "min": float(summary[1]), "max": float(summary[2]), "mean": float(summary[3]),
        "std": float(summary[4]), "rms": float(summary[5]),
    })
    return stats


def _sampling_stage_error(message: str, runtime_stages: Dict[str, Any], failed_stage: str) -> H3RuntimeError:
    runtime_stages["failedStage"] = failed_stage
    error = H3RuntimeError(message)
    _best_effort_exception_metadata(error, "failedStage", failed_stage, runtime_stages)
    _best_effort_exception_metadata(error, "runtimeStages", runtime_stages, runtime_stages)
    return error


def _record_av_tensor_stats(runtime_stages: Dict[str, Any], name: str, value: Any, *, failed_stage: str | None = None) -> Any:
    components = [_tensor_component_stats(item) for item in _av_components(value, name)]
    runtime_stages[name] = {"components": components}
    if not all(component["finite"] for component in components):
        raise _sampling_stage_error(f"nonfinite {name} tensor", runtime_stages, failed_stage or name)
    return value


def _run_with_noise_scaling_capture(model_sampling: Any, run: Callable[[], Any], on_scaled: Callable[[Any], None], runtime_stages: Dict[str, Any], stage: str) -> Any:
    original = model_sampling.noise_scaling
    instance_dict = getattr(model_sampling, "__dict__", {})
    had_instance_value = "noise_scaling" in instance_dict
    instance_value = instance_dict.get("noise_scaling")
    calls = 0

    def tracked(sigma: Any, noise: Any, latent_image: Any, max_denoise: bool = False) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise _sampling_stage_error(f"{stage} called noise_scaling more than once", runtime_stages, stage)
        scaled = original(sigma, noise, latent_image, max_denoise)
        on_scaled(scaled)
        return scaled

    setattr(model_sampling, "noise_scaling", tracked)
    try:
        result = run()
    finally:
        if had_instance_value:
            setattr(model_sampling, "noise_scaling", instance_value)
        else:
            delattr(model_sampling, "noise_scaling")
    if calls != 1:
        raise _sampling_stage_error(f"{stage} must call noise_scaling exactly once, got {calls}", runtime_stages, stage)
    runtime_stages.setdefault("noiseScalingCalls", {})[stage] = calls
    return result


def _advance_sampling_callback(state: Dict[str, int], step: int, local_total: int, expected_total: int, runtime_stages: Dict[str, Any], stage: str) -> int:
    expected = int(state.get("next", 0))
    if int(local_total) != int(expected_total) or int(step) != expected or expected >= int(expected_total):
        raise _sampling_stage_error(
            f"{stage} callback sequence mismatch: step={step}, expected={expected}, total={local_total}",
            runtime_stages,
            stage,
        )
    state["next"] = expected + 1
    return expected


def _require_sampling_callbacks(state: Dict[str, int], expected_total: int, runtime_stages: Dict[str, Any], stage: str) -> None:
    if int(state.get("next", 0)) != int(expected_total):
        raise _sampling_stage_error(
            f"{stage} callback count mismatch: expected={expected_total}, got={state.get('next', 0)}",
            runtime_stages,
            stage,
        )


def _report_single_step_progress(
    progress_receipt: Dict[str, Any],
    publish: Callable[[Dict[str, Any]], Any] | None,
    stage: str,
    *,
    completed: bool,
    elapsed: float | None = None,
) -> None:
    now = time.time()
    if completed:
        phase_elapsed = max(0.000001, float(elapsed or 0.0))
        progress_receipt.update({
            "currentStage": stage,
            "lastMeaningfulStage": stage,
            "phaseEndedAt": now,
            "elapsed": phase_elapsed,
            "phaseStep": 1,
            "phaseTotal": 1,
            "stageElapsed": {
                **dict(progress_receipt.get("stageElapsed") or {}),
                stage: phase_elapsed,
            },
        })
    else:
        progress_receipt.update({
            "currentStage": stage,
            "lastMeaningfulStage": stage,
            "phaseStartedAt": now,
            "phaseEndedAt": None,
            "elapsed": 0.0,
            "phaseStep": 0,
            "phaseTotal": 1,
        })
    if callable(publish):
        publish(dict(progress_receipt))


def _sample_with_official_h3_contract(
    custom_sampler: Any, sample_module: Any, model_management: Any, model: Any,
    positive: Any, latent: Dict[str, Any], seed: int, steps: int,
    callback: Callable[..., Any], disable_pbar: bool = True,
    stage1_steps: int | None = None, stage2_steps: int | None = None,
    stage2_denoise: float = 0.30, target_latent_hw: tuple[int, int] | None = None,
    source_latent_hw: tuple[int, int] | None = None,
    prepare_stage2_video: Callable[[Any, Callable[[int, int], Any]], tuple[Any, Dict[str, Any]]] | None = None,
    stage2_positive: Any = None,
    phase_progress: Callable[[Dict[str, Any]], Any] | None = None,
    stage1_ffn_chunks: int = 1,
    stage2_ffn_chunks: int = 2,
    ffn_chunk_adapter: Callable[..., tuple[Any, Dict[str, Any]]] = apply_h3_ffn_chunk_diagnostic,
) -> tuple[Any, Dict[str, Any]]:
    """Run stage1, frame resize plus VAE re-encode, then independent stage2."""
    from latent_upscaler_adapter import split_av_latent
    stage1_total = int(stage1_steps or steps)
    stage2_total = int(stage2_steps or 0)
    stage1_ffn_chunks = int(stage1_ffn_chunks)
    stage2_ffn_chunks = int(stage2_ffn_chunks)
    if stage1_total < 1 or stage2_total < 1 or not 0 < float(stage2_denoise) <= 1:
        raise H3RuntimeError("invalid independent two-stage sampling contract")
    if stage1_ffn_chunks != 1 or stage2_ffn_chunks not in {2, 4}:
        raise H3RuntimeError("invalid stage-specific FFN chunk contract")
    if target_latent_hw is None or source_latent_hw is None:
        raise H3RuntimeError("compiled source and target latent H/W are required")
    source_h, source_w = int(source_latent_hw[0]), int(source_latent_hw[1])
    target_h, target_w = int(target_latent_hw[0]), int(target_latent_hw[1])
    if target_h <= source_h or target_w <= source_w:
        raise H3RuntimeError(
            f"stage2 target latent H/W must exceed stage1 source: {source_h}x{source_w} -> {target_h}x{target_w}"
        )
    if not callable(prepare_stage2_video):
        raise H3RuntimeError("stage2 frame resize and video VAE re-encode callback is required")
    stage_receipt: Dict[str, Any] = {
        "fallbackUsed": False,
        "failedStage": None,
        "stage1Steps": stage1_total,
        "stage2Steps": stage2_total,
        "stage2Denoise": float(stage2_denoise),
        "stage1FfnChunks": stage1_ffn_chunks,
        "stage2FfnChunks": stage2_ffn_chunks,
        "stage1FfnApplied": False,
        "stage2FfnApplied": False,
        "progressReceipt": {
            "schemaVersion": 1,
            "currentStage": "stage1_sampling",
            "lastMeaningfulStage": "stage1_sampling",
            "phaseStartedAt": None,
            "phaseEndedAt": None,
            "elapsed": None,
            "phaseStep": 0,
            "phaseTotal": stage1_total,
            "globalStep": 0,
            "globalTotal": stage1_total + stage2_total,
        },
    }
    progress_receipt = stage_receipt["progressReceipt"]
    stage_elapsed: Dict[str, float] = {}

    def report_phase() -> None:
        if callable(phase_progress):
            phase_progress(dict(progress_receipt))

    stage1_denoised: list[Any] = []
    def run_one(stage: str, stage_positive: Any, stage_latent: Dict[str, Any], local_steps: int, local_seed: int, denoise: float, offset: int) -> Any:
        receipt_stage = f"{stage}_sampling"
        phase_started_at = time.time()
        progress_receipt = stage_receipt["progressReceipt"]
        progress_receipt.update({
            "currentStage": receipt_stage,
            "lastMeaningfulStage": receipt_stage,
            "phaseStartedAt": phase_started_at,
            "phaseEndedAt": None,
            "elapsed": 0.0,
            "phaseStep": 0,
            "phaseTotal": local_steps,
            "globalStep": offset,
            "globalTotal": stage1_total + stage2_total,
        })
        report_phase()
        if stage == "stage2":
            sampler_name = "euler"
            scheduler_name = "beta"
        else:
            contract = _official_h3_sampler_contract(local_steps)
            sampler_name = contract["sampler"]
            scheduler_name = contract["scheduler"]
        stage_model = model
        if stage == "stage2":
            stage_model, stage2_ffn_receipt = _apply_stage_ffn_chunk(
                model, stage, stage2_ffn_chunks, adapter=ffn_chunk_adapter
            )
            stage_receipt["stage2FfnApplied"] = True
            stage_receipt["stage2FfnReceipt"] = dict(stage2_ffn_receipt)
        sigmas = _node_args(custom_sampler.BasicScheduler.execute(stage_model, scheduler_name, local_steps, denoise))[0]
        guider = _node_args(custom_sampler.BasicGuider.execute(stage_model, stage_positive))[0]
        sampler = _node_args(custom_sampler.KSamplerSelect.execute(sampler_name))[0]
        noise = _node_args(custom_sampler.RandomNoise.execute(int(local_seed)))[0]
        work = dict(stage_latent)
        image = sample_module.fix_empty_latent_channels(guider.model_patcher, work["samples"], work.get("downscale_ratio_spacial"), work.get("downscale_ratio_temporal"))
        local_state = {"next": 0}
        denoised_outputs: list[Any] = []
        def stage_callback(step: int, *values: Any, **kwargs: Any) -> Any:
            # SamplerCustomAdvanced reports positional values as (x0, sample,
            # total); use None checks because tensor truthiness is undefined.
            total = kwargs.get("total")
            if total is None:
                total = kwargs.get("total_steps")
            x0 = kwargs.get("x0")
            if x0 is None:
                x0 = kwargs.get("denoised")
            if x0 is None:
                x0 = kwargs.get("denoised_output")
            sample = kwargs.get("sample")
            if x0 is None:
                if len(values) < 3:
                    raise H3RuntimeError(f"{stage} sampler callback lacks x0/total")
                x0, sample, total = values[:3]
            elif total is None:
                if len(values) < 3:
                    raise H3RuntimeError(f"{stage} sampler callback lacks total")
                if sample is None:
                    sample = values[1]
                total = values[2]
            if total is None:
                raise H3RuntimeError(f"{stage} sampler callback lacks total")
            _advance_sampling_callback(local_state, step, int(total), local_steps, stage_receipt, f"{stage}_callback")
            denoised_outputs.append(x0)
            if stage == "stage1":
                stage1_denoised[:] = [x0]
            completed = int(step) + 1
            progress_receipt.update({
                "currentStage": receipt_stage,
                "lastMeaningfulStage": receipt_stage,
                "phaseEndedAt": time.time(),
                "elapsed": round(time.time() - phase_started_at, 6),
                "phaseStep": completed,
                "phaseTotal": local_steps,
                "globalStep": offset + completed,
                "globalTotal": stage1_total + stage2_total,
            })
            callback(int(step) + offset, x0, sample, stage1_total + stage2_total)
            report_phase()
        try:
            sampled = guider.sample(noise.generate_noise({**work, "samples": image}), image, sampler, sigmas, denoise_mask=work.get("noise_mask"), callback=stage_callback, disable_pbar=disable_pbar, seed=noise.seed)
        except BaseException as exc:
            stage_receipt["failedStage"] = stage
            _best_effort_exception_metadata(exc, "failedStage", stage, stage_receipt)
            _best_effort_exception_metadata(exc, "runtimeStages", stage_receipt, stage_receipt)
            raise
        _require_sampling_callbacks(local_state, local_steps, stage_receipt, f"{stage}_callback")
        stage_elapsed[stage] = round(time.time() - phase_started_at, 6)
        denoised_output = denoised_outputs[-1] if denoised_outputs else sampled
        if denoised_output is None:
            raise H3RuntimeError(f"{stage} sampler did not provide a denoised output")
        process_latent_out = getattr(getattr(guider, "model_patcher", None), "model", None)
        process_latent_out = getattr(process_latent_out, "process_latent_out", None)
        if not callable(process_latent_out):
            process_latent_out = getattr(getattr(guider, "model_patcher", None), "process_latent_out", None)
        if not callable(process_latent_out):
            process_latent_out = lambda value: value
        denoised_output = process_latent_out(denoised_output)
        if stage == "stage1":
            stage1_denoised[:] = [denoised_output]
        return denoised_output
    stage1 = run_one("stage1", positive, latent, stage1_total, int(seed), 1.0, 0)
    if not stage1_denoised:
        raise H3RuntimeError("stage1 sampler did not provide a denoised output")
    video, audio = split_av_latent(stage1_denoised[-1])
    if tuple(int(item) for item in video.shape[-2:]) != (source_h, source_w):
        raise H3RuntimeError(
            f"stage1 video latent shape does not match compiled source H/W: {tuple(video.shape[-2:])} != {(source_h, source_w)}"
        )
    resize_started_at = time.time()
    try:
        setattr(prepare_stage2_video, "_stage1_audio", audio)
    except Exception as exc:
        raise H3RuntimeError(f"stage2 audio handoff unavailable: {type(exc).__name__}: {exc}") from exc
    progress_receipt.update({
        "currentStage": "frame_resize_vae_reencode",
        "lastMeaningfulStage": "frame_resize_vae_reencode",
        "phaseStartedAt": resize_started_at,
        "phaseEndedAt": None,
        "elapsed": 0.0,
        "phaseStep": 0,
        "phaseTotal": 3,
        "globalStep": stage1_total,
        "globalTotal": stage1_total + stage2_total,
    })
    report_phase()
    def report_resize_step(step: int, total: int) -> None:
        progress_receipt.update({
            "currentStage": "frame_resize_vae_reencode",
            "lastMeaningfulStage": "frame_resize_vae_reencode",
            "phaseEndedAt": time.time(),
            "elapsed": round(time.time() - resize_started_at, 6),
            "phaseStep": int(step),
            "phaseTotal": int(total),
        })
        report_phase()

    stage2_video, resize_receipt = prepare_stage2_video(video, report_resize_step)
    if tuple(int(item) for item in stage2_video.shape[-2:]) != (target_h, target_w):
        raise H3RuntimeError(
            f"stage2 re-encoded video latent shape does not match target H/W: {tuple(stage2_video.shape[-2:])} != {(target_h, target_w)}"
        )
    if int(stage2_video.shape[2]) != int(video.shape[2]):
        raise H3RuntimeError("stage2 frame resize and VAE re-encode changed video latent T")
    resize_elapsed = round(time.time() - resize_started_at, 6)
    progress_receipt.update({
        "currentStage": "frame_resize_vae_reencode",
        "lastMeaningfulStage": "frame_resize_vae_reencode",
        "phaseEndedAt": time.time(),
        "elapsed": resize_elapsed,
        "phaseStep": 3,
        "phaseTotal": 3,
    })
    report_phase()
    stage2_starting_video = stage2_video
    stage2_input_audio = _align_latent_component_device(audio, stage2_starting_video)
    nested_module = importlib.import_module("comfy.nested_tensor")
    stage2_samples = nested_module.NestedTensor((stage2_starting_video, stage2_input_audio))
    stage2 = run_one(
        "stage2", stage2_positive if stage2_positive is not None else positive,
        {"samples": stage2_samples}, stage2_total, int(seed), float(stage2_denoise), stage1_total,
    )
    stage2_video, _stage2_audio = split_av_latent(stage2)
    stage1_video, stage1_audio = split_av_latent(stage1_denoised[-1])
    stage2 = nested_module.NestedTensor((stage2_video, stage1_audio))
    audio_identity_preserved = stage2_samples.unbind()[1] is audio
    audio_device_aligned = stage2_input_audio.device == stage2_starting_video.device
    if not audio_device_aligned:
        raise H3RuntimeError("stage2 video and audio latents must share one device")
    tensor_stats = {
        "stage1OutputVideo": _tensor_component_stats(video),
        "stage2StartingVideoLatent": _tensor_component_stats(stage2_starting_video),
        "stage2FinalVideoLatent": _tensor_component_stats(split_av_latent(stage2)[0]),
        "stage1Audio": _tensor_component_stats(stage1_audio),
        "stage2SampledAudio": _tensor_component_stats(_stage2_audio),
        "finalAudio": _tensor_component_stats(split_av_latent(stage2)[1]),
    }
    for name, stats in tensor_stats.items():
        if not stats["finite"]:
            raise H3RuntimeError(f"nonfinite {name}")
    final_audio = split_av_latent(stage2)[1]
    stage2_audio_was_replaced = _stage2_audio is not final_audio
    audio_comparison = {
        "stage1AudioShape": _shape(stage1_audio),
        "stage2SampledAudioShape": _shape(_stage2_audio),
        "finalAudioShape": _shape(final_audio),
        "finalUsesStage1DenoisedAudio": final_audio is stage1_audio,
        "stage2SampledAudioDiscarded": True,
        "stage2SampledAudioReplaced": stage2_audio_was_replaced,
        "statistics": {
            "stage1": tensor_stats["stage1Audio"],
            "stage2Sampled": tensor_stats["stage2SampledAudio"],
            "final": tensor_stats["finalAudio"],
        },
    }
    if not audio_comparison["finalUsesStage1DenoisedAudio"]:
        raise H3RuntimeError("stage2 audio receipt does not preserve stage1 denoised audio")
    sampling_contract = _official_h3_sampler_contract(stage1_total + stage2_total)
    progress_receipt = stage_receipt["progressReceipt"]
    progress_receipt.update({
        "currentStage": "stage2_sampling",
        "lastMeaningfulStage": "stage2_sampling",
        "phaseEndedAt": time.time(),
        "elapsed": max(0.000001, float(progress_receipt.get("elapsed") or 0.0)),
        "phaseStep": stage2_total,
        "phaseTotal": stage2_total,
        "globalStep": stage1_total + stage2_total,
        "globalTotal": stage1_total + stage2_total,
        "stageElapsed": {
            "stage1": stage_elapsed.get("stage1"),
            "frame_resize_vae_reencode": resize_elapsed,
            "stage2": stage_elapsed.get("stage2"),
            "video_decode": None,
            "audio_decode": None,
            "mux": None,
        },
    })
    return stage2, {
        **sampling_contract,
        "progressReceipt": progress_receipt,
        "lastMeaningfulStage": progress_receipt["lastMeaningfulStage"],
        "stage1Steps": stage1_total,
        "stage2Steps": stage2_total,
        "stage2Denoise": float(stage2_denoise),
        "stage1FfnChunks": stage1_ffn_chunks,
        "stage2FfnChunks": stage2_ffn_chunks,
        "stage1FfnApplied": False,
        "stage2FfnApplied": bool(stage_receipt["stage2FfnApplied"]),
        "stage2FfnReceipt": dict(stage_receipt.get("stage2FfnReceipt") or {}),
        "stage1Sampler": "res_multistep",
        "stage1Scheduler": "simple",
        "stage2Sampler": "euler",
        "stage2Scheduler": "beta",
        "stage1UpscalerSource": "denoised_output",
        "stage2DecodeSource": "denoised_output",
        "finalAudioSource": "stage1_denoised_output_audio",
        "stage2AudioSampled": True,
        "stage2AudioDiscarded": True,
        "samplerCalls": 2,
        "executionKind": (
            "latent_upscale_then_independent_resample"
            if str((resize_receipt or {}).get("routeMode") or "latent") == "latent"
            else "frame_resize_vae_reencode_then_independent_resample"
        ),
        "stage1VideoLatentShape": _shape(video),
        "stage2VideoLatentShape": _shape(stage2_video),
        "stage1AudioLatentShape": _shape(stage1_audio),
        "stage2AudioLatentShape": _shape(_stage2_audio),
        "finalAudioLatentShape": _shape(split_av_latent(stage2)[1]),
        "audioComparison": audio_comparison,
        "stage1AudioIdentityPreserved": audio_comparison["finalUsesStage1DenoisedAudio"],
        "stage2InputAudioIdentityPreserved": audio_identity_preserved,
        "stage2InputAudioDeviceAligned": audio_device_aligned,
        "stage2InputAudioDevice": str(stage2_input_audio.device),
        "stage2InputVideoDevice": str(stage2_starting_video.device),
        "stage2OutputAudioIdentityRequired": False,
        "sourceLatentHW": {"height": source_h, "width": source_w},
        "targetLatentHW": {"height": target_h, "width": target_w},
        "rawLbhOutputLatent": dict((resize_receipt or {}).get("rawLbhOutputLatent") or {}),
        "cropWindow": dict((resize_receipt or {}).get("cropWindow") or {}),
        "finalStage2Latent": dict((resize_receipt or {}).get("finalStage2Latent") or {}),
        "frameResizeVaeRoundTrip": {
            "videoLayout": "BCTHW",
            "videoChannels": 24,
            "preserveVideoT": True,
            "audioPolicy": "identity",
            **dict(resize_receipt),
            "nvidiaRtxVsr": bool(dict(resize_receipt).get("nvidiaRtxVsr", True)),
            "fallbackUsed": False,
        },
        "routeMode": str((resize_receipt or {}).get("routeMode") or "latent"),
        "reNoise": {
            "required": True,
            "construction": "stage2_sampler_noise_and_beta_sigmas",
            "denoise": float(stage2_denoise),
        },
        "tensorStats": tensor_stats,
        "latentShape": _shape(stage2),
    }


def _finalize_official_aimdo_sampling_lifecycle(
    model_management: Any,
    *,
    module_import: Callable[[str], Any] = importlib.import_module,
) -> Dict[str, Any]:
    """Mirror ComfyUI's AIMDO execution-finally cleanup after sampling.

    The direct runner owns a full sampling lifetime instead of going through
    ``execution.py``. This follows its official order once, outside the
    denoiser hot path. Cleanup failures are recorded and never replace a
    sampler exception already unwinding through the caller's ``finally``.
    """
    receipt: Dict[str, Any] = {
        "scope": "official_aimdo_execution_finally_after_sampling",
        "sequence": [
            "comfy.model_management.reset_cast_buffers",
            "comfy.model_prefetch.cleanup_prefetch_queues",
            "comfy_aimdo.model_vbar.vbars_reset_watermark_limits",
        ],
        "calls": [],
        "errors": [],
    }
    try:
        memory_management = module_import("comfy.memory_management")
        model_prefetch = module_import("comfy.model_prefetch")
        model_vbar = module_import("comfy_aimdo.model_vbar")
    except Exception as exc:
        receipt["available"] = False
        receipt["errors"].append({"stage": "module_import", "error": repr(exc)})
        return receipt

    receipt["available"] = True
    receipt["aimdoEnabled"] = bool(getattr(memory_management, "aimdo_enabled", False))
    if not receipt["aimdoEnabled"]:
        receipt["skipped"] = "aimdo_disabled"
        return receipt
    for label, callback in (
        ("comfy.model_management.reset_cast_buffers", getattr(model_management, "reset_cast_buffers", None)),
        ("comfy.model_prefetch.cleanup_prefetch_queues", getattr(model_prefetch, "cleanup_prefetch_queues", None)),
        ("comfy_aimdo.model_vbar.vbars_reset_watermark_limits", getattr(model_vbar, "vbars_reset_watermark_limits", None)),
    ):
        call_receipt: Dict[str, Any] = {"name": label, "called": False, "error": None}
        if not callable(callback):
            call_receipt["error"] = "unavailable"
            receipt["errors"].append({"stage": label, "error": "unavailable"})
        else:
            try:
                callback()
                call_receipt["called"] = True
            except H3RuntimeError:
                raise
            except Exception as exc:
                call_receipt["error"] = repr(exc)
                receipt["errors"].append({"stage": label, "error": repr(exc)})
        receipt["calls"].append(call_receipt)
    return receipt


def _shape(value: Any) -> Optional[list[int]]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def _device_text(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _cuda_memory_receipt(torch: Any) -> Dict[str, Any]:
    """Record allocator state without changing placement or cleanup policy."""

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {"available": False}
    try:
        free, total = cuda.mem_get_info()
        return {
            "available": True,
            "memoryAllocatedBytes": int(cuda.memory_allocated()),
            "memoryReservedBytes": int(cuda.memory_reserved()),
            "peakAllocatedBytes": int(cuda.max_memory_allocated()),
            "peakReservedBytes": int(cuda.max_memory_reserved()),
            "freeBytes": int(free),
            "totalBytes": int(total),
        }
    except Exception as exc:
        return {"available": True, "error": f"{type(exc).__name__}: {exc}"}


def _resource_cooperation_receipt(torch: Any) -> Dict[str, Any]:
    """Observe RAM, Windows commit and CUDA state without changing execution."""

    receipt: Dict[str, Any] = {"cuda": _cuda_memory_receipt(torch)}
    try:
        import ctypes
        import os

        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_process_memory_info = psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        get_process_memory_info.restype = ctypes.c_bool
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if get_process_memory_info(process, ctypes.byref(counters), counters.cb):
            receipt["process"] = {
                "pid": os.getpid(),
                "pageFaultCount": int(counters.PageFaultCount),
                "workingSetBytes": int(counters.WorkingSetSize),
                "peakWorkingSetBytes": int(counters.PeakWorkingSetSize),
                "privateBytes": int(counters.PrivateUsage),
                "pagefileUsageBytes": int(counters.PagefileUsage),
                "peakPagefileUsageBytes": int(counters.PeakPagefileUsage),
            }
        else:
            receipt["processTelemetryError"] = f"GetProcessMemoryInfo failed (winerror={ctypes.get_last_error()})"
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            receipt["systemCommit"] = {
                "totalPageFileBytes": int(status.ullTotalPageFile),
                "availablePageFileBytes": int(status.ullAvailPageFile),
                "committedBytes": int(status.ullTotalPageFile - status.ullAvailPageFile),
                "physicalAvailableBytes": int(status.ullAvailPhys),
            }
    except Exception as exc:
        receipt["processTelemetryError"] = f"{type(exc).__name__}: {exc}"
    return receipt


_SYSTEM_TIMES_PREVIOUS: Optional[Tuple[int, int, int, int]] = None


def _external_resource_receipt() -> Dict[str, Any]:
    """Read-only GPU/CPU utilization samples for performance receipts."""

    result: Dict[str, Any] = {"source": "nvidia-smi+GetSystemTimes"}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=0.5, check=False,
        )
        values = [item.strip() for item in query.stdout.splitlines()[0].split(",")]
        if query.returncode == 0 and len(values) == 4:
            result["gpu"] = {
                "available": True,
                "utilizationPercent": float(values[0]),
                "memoryUsedMiB": float(values[1]),
                "memoryTotalMiB": float(values[2]),
                "powerDrawW": float(values[3]),
            }
        else:
            result["gpu"] = {"available": False, "reason": "nvidia_smi_failed"}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired) as exc:
        result["gpu"] = {"available": False, "reason": f"nvidia_smi_unavailable:{type(exc).__name__}"}
    try:
        import ctypes
        from ctypes import wintypes
        global _SYSTEM_TIMES_PREVIOUS
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            def ticks(value: wintypes.FILETIME) -> int:
                return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)
            sample = (ticks(idle), ticks(kernel), ticks(user), time.monotonic_ns())
            if _SYSTEM_TIMES_PREVIOUS is not None:
                old_idle, old_kernel, old_user, _ = _SYSTEM_TIMES_PREVIOUS
                total = max(1, sample[1] - old_kernel + sample[2] - old_user)
                busy = max(0, total - (sample[0] - old_idle))
                result["cpu"] = {"available": True, "scope": "system", "loadPercent": round(100.0 * busy / total, 2)}
            else:
                result["cpu"] = {"available": False, "reason": "cpu_baseline_pending", "scope": "system"}
            _SYSTEM_TIMES_PREVIOUS = sample
        else:
            result["cpu"] = {"available": False, "reason": "GetSystemTimes_failed", "scope": "system"}
    except Exception as exc:
        result["cpu"] = {"available": False, "reason": f"cpu_telemetry_unavailable:{type(exc).__name__}", "scope": "system"}
    return result


_AIMDO_VBAR_PAGE_BYTES = 32 * 1024 * 1024


def _as_observed_int(value: Any) -> int | str:
    """Return a public numeric VBAR property without guessing private layout."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return "unavailable"


def _dynamic_vbar_module_manifest(model_patcher: Any) -> Dict[str, Any]:
    """Observe DynamicVRAM's actual load-list ordering without allocating pages.

    AIMDO does not expose a Python API that maps an allocated module handle back
    to a VBAR byte offset.  The manifest therefore records that mapping only if
    the installed binding makes it public; otherwise it explicitly says
    ``unavailable``.  Importantly, this function never creates a VBAR, faults a
    weight, pins a module, or changes the load list.
    """

    result: Dict[str, Any] = {
        "available": False,
        "allocationOrder": "ModelPatcherDynamic._load_list(for_dynamic=True); loading.sort()",
        "pageSizeBytes": _AIMDO_VBAR_PAGE_BYTES,
        "entries": [],
    }
    try:
        if not bool(getattr(model_patcher, "is_dynamic", lambda: False)()):
            result["reason"] = "primary_model_not_dynamic"
            return result
        load_list = getattr(model_patcher, "_load_list", None)
        if not callable(load_list):
            result["reason"] = "dynamic_load_list_unavailable"
            return result
        loading = list(load_list(for_dynamic=True, default_device=getattr(model_patcher, "load_device", None)))
        loading.sort()
        entries = []
        for order, entry in enumerate(loading):
            if len(entry) < 4:
                continue
            *sort_key, module_bytes, module_name, module, _params = entry
            handle = getattr(module, "_v", None)
            # No currently supported AIMDO Python binding promises these names.
            # Probe them read-only so a future binding becomes observable without
            # inventing a page mapping for today's binding.
            offset = "unavailable"
            for name in ("byte_offset", "offset", "start", "address"):
                candidate = getattr(handle, name, None)
                if candidate is not None:
                    offset = _as_observed_int(candidate)
                    break
            if isinstance(offset, int):
                page_start: int | str = offset // _AIMDO_VBAR_PAGE_BYTES
                page_end: int | str = max(page_start, (offset + int(module_bytes) - 1) // _AIMDO_VBAR_PAGE_BYTES)
            else:
                page_start = "unavailable"
                page_end = "unavailable"
            entries.append({
                "allocationOrder": order,
                "module": str(module_name),
                "moduleBytes": int(module_bytes),
                "loadListSortKey": [_as_observed_int(value) if not isinstance(value, bool) else value for value in sort_key],
                "vbarHandleType": (f"{type(handle).__module__}.{type(handle).__name__}" if handle is not None else "unallocated"),
                "vbarByteOffset": offset,
                "vbarPageStart": page_start,
                "vbarPageEnd": page_end,
            })
        result.update({"available": True, "entries": entries})
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _native_dynamic_vram_receipt(
    model_patcher: Any,
    task_watermark_limits: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Read native AIMDO DynamicVRAM state without influencing residency.

    ModelVBAR exposes loaded bytes, a page-level watermark and per-page residency.  It does
    not expose fault, unpin or prefetch counters through the public Python API,
    so those fields stay explicitly unavailable instead of being inferred from
    allocator values.  This function deliberately never allocates a VBAR and
    never calls prioritize/free/reset methods.
    """
    receipt: Dict[str, Any] = {
        "available": False,
        "isDynamic": False,
        "loadedWeightBytes": None,
        "offloadedWeightBytes": None,
        "lowvramPatchCount": None,
        "vbars": [],
        "faultCount": "unavailable",
        "unpinCount": "unavailable",
        "prefetchCount": "unavailable",
        "prioritizeCallCount": "unavailable",
        "watermarkResetCount": "unavailable",
    }
    try:
        is_dynamic = bool(getattr(model_patcher, "is_dynamic", lambda: False)())
        receipt["isDynamic"] = is_dynamic
        model = getattr(model_patcher, "model", None)
        if model is None:
            receipt["reason"] = "model_unavailable"
            return receipt
        model_size = int(getattr(model_patcher, "model_size", lambda: 0)() or 0)
        loaded_size = int(getattr(model_patcher, "loaded_size", lambda: 0)() or 0)
        receipt.update({
            "available": is_dynamic,
            "modelSizeBytes": model_size or None,
            "loadedWeightBytes": loaded_size,
            "offloadedWeightBytes": max(0, model_size - loaded_size) if model_size else None,
            "lowvramPatchCount": int(getattr(model, "lowvram_patch_counter", 0) or 0),
            "nativePriorityPolicy": (
                "ModelPatcherDynamic.load calls vbar.prioritize()"
                if is_dynamic else "not_dynamic"
            ),
        })
        for device, vbar in (getattr(model, "dynamic_vbars", {}) or {}).items():
            page_count = int(vbar.get_nr_pages())
            residency = list(vbar.get_residency())
            resident_pages = sum(1 for state in residency if int(state) & 1)
            pinned_pages = sum(1 for state in residency if int(state) & 2)
            watermark_pages = int(vbar.get_watermark())
            object_identity = f"{type(vbar).__module__}.{type(vbar).__name__}:{id(vbar):x}"
            configured_limit = (task_watermark_limits or {}).get(object_identity)
            receipt["vbars"].append({
                "device": _device_text(device),
                "objectIdentity": object_identity,
                "loadedBytes": int(vbar.loaded_size()),
                "pageSizeBytes": _AIMDO_VBAR_PAGE_BYTES,
                "pageCount": page_count,
                "watermarkPages": watermark_pages,
                "watermarkByteOffsetEstimate": watermark_pages * _AIMDO_VBAR_PAGE_BYTES,
                "residentPages": resident_pages,
                "residentBytesEstimate": resident_pages * _AIMDO_VBAR_PAGE_BYTES,
                "pinnedPages": pinned_pages,
                "offloadedPages": max(0, page_count - resident_pages),
                "watermarkLimitPages": (
                    configured_limit.get("watermarkLimitPages")
                    if configured_limit else "unavailable"
                ),
                "watermarkLimitByteOffset": (
                    configured_limit.get("watermarkLimitByteOffset")
                    if configured_limit else "unavailable"
                ),
                "limitDecisionSource": (
                    configured_limit.get("limitDecisionSource")
                    if configured_limit else "unavailable"
                ),
                "watermarkSemantics": "native_page_cutoff_after_eviction_not_a_residency_floor",
                "watermarkResetSource": "ModelPatcherDynamic.load -> ModelVBAR.prioritize",
                "temporaryFallbackCount": "unavailable",
            })
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    return receipt


def _observe_native_dynamic_vram_after_first_step(
    torch: Any,
    model_patcher: Any,
    residency_preflight: Dict[str, Any],
    task_watermark_limits: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Record the first real DynamicVRAM state without changing residency.

    This function intentionally does *not* call AIMDO ``set_watermark`` or
    ``set_watermark_limit``.  Their semantics need a page/module access manifest
    and a measured pressure trace first; observation must not become a hidden
    scheduling policy.
    """

    before = _native_dynamic_vram_receipt(model_patcher, task_watermark_limits)
    resources = _resource_cooperation_receipt(torch)
    result: Dict[str, Any] = {
        "strategy": "native_dynamicvram_sampler_owned",
        "status": "observed",
        "samplerOwnsLoad": True,
        "productWatermarkOverride": False,
        "productWatermarkLimit": False,
        "packedTokens": residency_preflight.get("packedTokens"),
        "workspaceEstimateBytes": residency_preflight.get("workspaceEstimateBytes"),
        "preflight": dict(residency_preflight.get("residencyBudget") or {}),
        "resourcesAtDecision": resources,
        "before": before,
        "native": before,
    }
    result.update({
        "status": "observed",
        "reason": "watermark_controls_disabled_pending_page_to_module_trace",
        "moduleManifest": _dynamic_vbar_module_manifest(model_patcher),
        "setWatermarkCalled": False,
        "setWatermarkLimitCalled": False,
    })
    return result


def _sampling_residency_telemetry_summary(stabilization: Dict[str, Any]) -> Dict[str, Any]:
    """Keep step receipts small; the full manifest remains terminal-only."""

    manifest = stabilization.get("moduleManifest") or {}
    return {
        "status": stabilization.get("status"),
        "strategy": stabilization.get("strategy"),
        "reason": stabilization.get("reason"),
        "samplerOwnsLoad": stabilization.get("samplerOwnsLoad"),
        "productWatermarkOverride": stabilization.get("productWatermarkOverride"),
        "productWatermarkLimit": stabilization.get("productWatermarkLimit"),
        "manifestEntryCount": len(manifest.get("entries", [])),
        "manifestAllocationOrder": manifest.get("allocationOrder"),
        "setWatermarkCalled": stabilization.get("setWatermarkCalled"),
        "setWatermarkLimitCalled": stabilization.get("setWatermarkLimitCalled"),
    }


def _record_first_native_step_residency(
    residency_receipt: Dict[str, Any],
    stabilization: Dict[str, Any],
    prefetch_trace: Optional[Dict[str, Any]],
) -> None:
    """Replace the pre-sampler snapshot with facts observed after native step one.

    ``_prepare_sequence_residency`` intentionally runs before Comfy owns the
    DynamicVRAM load, so its old ``awaiting_first_native_step`` value is not a
    runtime state after the callback.  Keep the preflight data, but update its
    phase once the official sampler has actually returned a denoiser step.
    This only repairs receipt semantics; it never changes VBAR residency or
    the official prefetch request.
    """

    budget = residency_receipt.get("residencyBudget")
    if not isinstance(budget, dict):
        return
    trace = prefetch_trace if isinstance(prefetch_trace, dict) else {}
    attempts = trace.get("makeQueueCalls")
    if not isinstance(attempts, list):
        attempts = []
    budget.update({
        "phase": "observed_after_first_native_step",
        "observedAfterFirstNativeStep": True,
        "prefetchRequestObserved": any(bool(item.get("requested")) for item in attempts if isinstance(item, dict)),
        "prefetchQueueCreated": any(bool(item.get("created")) for item in attempts if isinstance(item, dict)),
        "prefetchPopCount": int(trace.get("popCount", 0) or 0),
        "prefetchCleanupCount": int(trace.get("cleanupCount", 0) or 0),
    })
    stabilization["prefetch"] = {
        "requestObserved": budget["prefetchRequestObserved"],
        "queueCreated": budget["prefetchQueueCreated"],
        "popCount": budget["prefetchPopCount"],
        "cleanupCount": budget["prefetchCleanupCount"],
    }


def _release_completed_conditioning_memory(torch: Any, model_management: Any) -> Dict[str, Any]:
    """Finish the official conditioning-to-primary lifecycle boundary.

    Qwen and the reference VAEs have already been explicitly unloaded before
    this function runs.  Collecting their dead Python objects and requesting
    Comfy's own cache cleanup once prevents the next primary load from inheriting
    stale allocator segments.  It does not unload the positive conditioning or
    touch the forthcoming primary patcher.
    """
    import gc

    receipt: Dict[str, Any] = {
        "gcCollected": 0,
        "nativeSoftEmptyCacheCalled": False,
        "cudaSynchronizeCalled": False,
        "cudaEmptyCacheCalled": False,
        "ipcCollectCalled": False,
        "before": _resource_cooperation_receipt(torch),
    }
    try:
        receipt["gcCollected"] = int(gc.collect())
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.synchronize()
            receipt["cudaSynchronizeCalled"] = True
            soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
            if callable(soft_empty_cache):
                try:
                    soft_empty_cache(force=True)
                except TypeError:
                    soft_empty_cache()
                receipt["nativeSoftEmptyCacheCalled"] = True
            else:
                cuda.empty_cache()
                receipt["cudaEmptyCacheCalled"] = True
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
                receipt["ipcCollectCalled"] = True
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["after"] = _resource_cooperation_receipt(torch)
    return receipt


def _stage_latent_for_cpu_vae(value: Any, torch: Any) -> Tuple[Any, Dict[str, Any]]:
    """Queue a pinned CPU transfer when CUDA supports it, then let the caller sync.

    This preserves every latent value and only removes two back-to-back
    pageable transfer stalls between sampling and sequential official VAE
    decode.  The receipt makes the actual path explicit; unsupported systems
    retain the existing synchronous CPU copy.
    """

    detached = value.detach()
    receipt: Dict[str, Any] = {
        "sourceDevice": _device_text(getattr(detached, "device", None)),
        "bytes": int(getattr(detached, "numel", lambda: 0)() * getattr(detached, "element_size", lambda: 0)()),
        "pinned": False,
        "nonBlocking": False,
        "queued": False,
        "fallback": None,
    }
    if not getattr(torch.cuda, "is_available", lambda: False)() or not str(getattr(detached, "device", "")).startswith("cuda"):
        receipt["fallback"] = "source_not_cuda"
        return detached.cpu(), receipt
    try:
        host = torch.empty_like(detached, device="cpu", pin_memory=True)
        host.copy_(detached, non_blocking=True)
        receipt.update({"pinned": True, "nonBlocking": True, "queued": True})
        return host, receipt
    except Exception as exc:
        receipt["fallback"] = f"{type(exc).__name__}: {exc}"
        return detached.cpu(), receipt


def _tensor_receipt(value: Any) -> Dict[str, Any]:
    if isinstance(value, (list, tuple)):
        return {"components": [_tensor_receipt(item) for item in value]}
    return {
        "shape": _shape(value),
        "device": _device_text(getattr(value, "device", None)),
        "dtype": _device_text(getattr(value, "dtype", None)),
    }


def _is_cuda_out_of_memory(torch: Any, exc: BaseException) -> bool:
    """Recognize only an actual CUDA allocation failure for VAE recovery."""

    cuda_oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if cuda_oom is not None:
        try:
            if isinstance(exc, cuda_oom):
                return True
        except TypeError:
            pass
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and "out of memory" in text and "cuda" in text


def _vae_decode_device_preflight(torch: Any, vae: Any, latent: Any) -> Dict[str, Any]:
    """Choose VAE placement from live allocator state and official estimates.

    The H3 VAE wrapper exposes the official model-size and decode-memory
    estimates.  When a preceding denoiser leaves a substantial live CUDA
    allocation, putting another multi-gigabyte model on the same device is
    not a useful optimization: select the same official VAE on CPU before it
    starts loading.  This is a runtime resource decision, not a frame/time
    or product-quality limit; exact latent, canvas, audio, and frame counts
    remain unchanged.
    """

    receipt: Dict[str, Any] = {
        "requested": "auto",
        "selected": "gpu",
        "reason": None,
        "modelSizeBytes": None,
        "decodeEstimateBytes": None,
        "cudaFreeBytes": None,
        "cudaTotalBytes": None,
        "cudaAllocatedBytes": None,
        "residualReserveBytes": None,
    }
    if not getattr(torch.cuda, "is_available", lambda: False)():
        receipt.update({"selected": "cpu", "reason": "cuda_unavailable"})
        return receipt
    try:
        model_size = int(vae.model_size())
        decode_estimate = int(vae.memory_used_decode(latent.shape, vae.vae_dtype))
        free_bytes, total_bytes = (int(value) for value in torch.cuda.mem_get_info())
        allocated_bytes = int(torch.cuda.memory_allocated())
        residual_reserve = max(model_size // 2, 1)
        receipt.update({
            "modelSizeBytes": model_size,
            "decodeEstimateBytes": decode_estimate,
            "cudaFreeBytes": free_bytes,
            "cudaTotalBytes": total_bytes,
            "cudaAllocatedBytes": allocated_bytes,
            "residualReserveBytes": residual_reserve,
        })
        if allocated_bytes > residual_reserve:
            receipt.update({
                "selected": "cpu",
                "reason": "existing_cuda_residual_exceeds_vae_reserve",
            })
        elif free_bytes < model_size + decode_estimate + residual_reserve:
            receipt.update({
                "selected": "cpu",
                "reason": "insufficient_free_memory_for_vae_estimate",
            })
    except Exception as exc:
        # A missing estimate must not prevent a real request.  Keep the
        # normal GPU-first path and let the narrowly caught CUDA OOM handler
        # below select CPU if the runtime itself reports a failure.
        receipt["estimateError"] = f"{type(exc).__name__}: {exc}"
    return receipt


def _reference_vae_cpu_policy(torch: Any) -> Dict[str, Any]:
    """Select the official reference-VAE device from runtime availability.

    The official VAE encoder already moves each completed latent to its
    configured intermediate/output device and has its own regular-to-tiled
    memory fallback.  Do not turn a 32-GiB card into a minute-scale CPU encoder
    merely from total-capacity class; use the GPU path and let the official
    preflight choose tiled execution when live free memory requires it.
    """

    if torch is None or not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return {"selected": "cpu", "reason": "cuda_unavailable", "totalBytes": None}
    try:
        total = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:
        return {"selected": "gpu", "reason": f"capacity_probe_failed:{type(exc).__name__}", "totalBytes": None}
    return {
        "selected": "gpu",
        "reason": "cuda_available_official_vae_preflight_and_tiled_fallback",
        "totalBytes": total,
    }


def _primary_model_load_policy(
    torch: Any,
    packed_tokens: Optional[int],
    token_threshold: int = 65536,
) -> Dict[str, Any]:
    """Use the verified workflow's native DynamicVRAM model lifecycle.

    The 5090 reference workflow keeps DynamicVRAM enabled because the
    quantized primary model and NVFP4 text encoder cannot both be resident in
    32 GB.  This changes residency only; tokens, references, resolution,
    steps, and model weights remain untouched.
    """

    threshold = max(1, int(token_threshold))
    receipt: Dict[str, Any] = {
        "forceFullLoad": False,
        "requestedMode": "auto",
        "packedSequenceTokens": None if packed_tokens is None else int(packed_tokens),
        "dynamicVramTokenThreshold": threshold,
        "freeBytesBeforeLoad": None,
        "totalBytes": None,
        "reason": "workflow baseline: use native DynamicVRAM model management",
    }
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        receipt.update({"forceFullLoad": False, "reason": "cuda_unavailable"})
        return receipt
    try:
        free, total = cuda.mem_get_info()
        receipt.update({"freeBytesBeforeLoad": int(free), "totalBytes": int(total)})
    except Exception as exc:
        receipt["probeError"] = f"{type(exc).__name__}: {exc}"
    if packed_tokens is not None and int(packed_tokens) >= threshold:
        receipt["reason"] = "workflow baseline: long packed sequence still uses native DynamicVRAM"
    return receipt


def _prepare_sequence_residency(
    torch: Any,
    model_management: Any,
    model_patcher: Any,
    packed_tokens: Optional[int],
    token_threshold: int = 65536,
) -> Dict[str, Any]:
    """Describe the token-sized workspace while leaving native loading intact.

    The official sampler owns the only ``load_models_gpu`` invocation for the
    primary patcher.  Calling it a second time here changes AIMDO's VBAR timing
    before the sampler has established its real activation demand; long H3
    sequences showed that this extra pre-load can consume WDDM headroom and
    produce progressively slower steps.  We retain a transparent, continuous
    token/workspace estimate for receipts but never pre-load or reserve pages.
    After the sampler's first real step, the runner records AIMDO's public
    read-only residency state. It never sets a project watermark or requests
    an additional model load.
    """
    token_count = int(packed_tokens or 0)
    threshold = max(1, int(token_threshold))
    receipt: Dict[str, Any] = {
        "enabled": False,
        "strategy": "native_dynamicvram_sampler_owned_residency",
        "packedTokens": token_count or None,
        "tokenThreshold": threshold,
        "forceFullLoad": False,
        "workspaceEstimateFormula": "tokens * 5376 hidden * bf16(2 bytes) * 4 live activations",
        "workspaceEstimateBytes": None,
        "workspaceCappedByCurrentFreeBytes": None,
        "residencyBudget": {
            "phase": "awaiting_first_native_step",
            "policy": "native_sampler_owned_page_priority",
            "source": ["packedTokens", "torch_cuda", "windows_ram", "aimdo_vbar_after_first_step"],
            "productWatermarkOverride": False,
        },
        "freeBytesBefore": None,
        "freeBytesAfter": None,
        "reason": "native_sampler_owns_primary_residency",
    }
    if token_count <= 0:
        receipt["reason"] = "packed_token_count_unavailable"
        return receipt
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        receipt["reason"] = "cuda_unavailable"
        return receipt
    try:
        free_before, _total = cuda.mem_get_info()
        # H3's packed transformer width is fixed at 5376. Four BF16 live
        # activation views are a conservative continuous residency envelope,
        # not a duration branch, model limit, or quality setting. It is capped
        # by the real free budget.
        workspace = int(token_count) * 5376 * 2 * 4
        reserve = min(workspace, int(free_before))
        receipt.update({
            "workspaceEstimateBytes": workspace,
            "workspaceCappedByCurrentFreeBytes": reserve,
            "freeBytesBefore": int(free_before),
        })
        # Do not call load_models_gpu here.  SamplerCustomAdvanced is the
        # official owner of that native lifecycle and will invoke it with its
        # real model/input requirements immediately before the first forward.
        receipt.update({
            "enabled": True,
            "freeBytesAfter": int(free_before),
            "reason": "native_sampler_residency_deferred_until_real_forward",
        })
    except Exception as exc:
        # The normal sampler still owns the official load path. A failed
        # optional preflight never changes the generation contract.
        receipt.update({"reason": "native_residency_preflight_unavailable", "error": f"{type(exc).__name__}: {exc}"})
    return receipt


def _dynamic_vbar_block_snapshot(model_patcher: Any) -> Dict[str, Any]:
    """Small VBAR-only observation for selected block boundaries.

    Allocator, WDDM and process/RAM telemetry belongs at the sampler-step
    boundary.  Calling those APIs around all 50 blocks materially distorts a
    long H3 forward, so this helper intentionally reads only public VBAR page
    counters and never synchronizes CUDA.
    """

    native = _native_dynamic_vram_receipt(model_patcher)
    result = {
        "vbars": [
            {
                "objectIdentity": entry.get("objectIdentity"),
                "watermarkPages": entry.get("watermarkPages"),
                "residentPages": entry.get("residentPages"),
                "offloadedPages": entry.get("offloadedPages"),
                "pinnedPages": entry.get("pinnedPages"),
            }
            for entry in native.get("vbars", [])
        ],
        "loadedWeightBytes": native.get("loadedWeightBytes"),
        "offloadedWeightBytes": native.get("offloadedWeightBytes"),
    }
    return result


def _install_native_prefetch_trace(runtime_stages: Dict[str, Any]) -> Callable[[], None]:
    """Observe the official prefetch module without changing queue semantics."""

    receipt: Dict[str, Any] = {
        "available": False,
        "makeQueueCalls": [],
        "popCount": 0,
        "cleanupCount": 0,
        "popHostSecondsTotal": 0.0,
        # One compact summary per native H3 forward.  Pop callbacks update
        # this in memory only; the sampler callback or cancellation boundary
        # is responsible for persisting the enclosing runtime receipt.
        "forwardSummaries": [],
        "faultUnpinCorrelation": "unavailable_no_task_step_block_ids_in_aimdo_public_api",
    }
    runtime_stages["native_dynamic_vram_prefetch"] = receipt
    known_blocks: Dict[int, str] = {}
    forward_by_queue_id: Dict[int, Dict[str, Any]] = {}
    try:
        prefetch_module = importlib.import_module("comfy.model_prefetch")
        original_make = prefetch_module.make_prefetch_queue
        original_pop = prefetch_module.prefetch_queue_pop
        original_cleanup = prefetch_module.cleanup_prefetched_modules
        receipt["available"] = True

        def module_name(module: Any) -> str | None:
            if module is None:
                return None
            return known_blocks.get(id(module), type(module).__name__)

        def traced_make(queue: Any, device: Any, transformer_options: Dict[str, Any]) -> Any:
            for index, block in enumerate(queue or []):
                known_blocks[id(block)] = f"block_{index + 1}"
            requested = bool((transformer_options or {}).get("prefetch_dynamic_vbars", False))
            result = original_make(queue, device, transformer_options)
            queue_call = {
                "requested": requested,
                "created": result is not None,
                "queueLength": len(queue) if hasattr(queue, "__len__") else None,
                "device": _device_text(device),
                "nonBlockingEligible": "unavailable_native_predicate_not_exposed",
            }
            receipt["makeQueueCalls"].append(queue_call)
            summary = {
                "forwardIndex": len(receipt["forwardSummaries"]) + 1,
                "queueIdentity": (
                    f"{type(result).__module__}.{type(result).__name__}:{id(result):x}"
                    if result is not None else None
                ),
                "queueCreated": result is not None,
                "requested": requested,
                "queueLength": queue_call["queueLength"],
                "popCount": 0,
                "popHostSecondsTotal": 0.0,
                "popHostSecondsMax": 0.0,
                "slowestPop": None,
                # The runner surrounds one official function call only.  It
                # cannot truthfully split its time into stream waiting versus
                # VBAR casting without patching lower-level official code.
                "phaseBreakdown": "unavailable_single_official_pop_call",
            }
            receipt["forwardSummaries"].append(summary)
            if result is not None:
                forward_by_queue_id[id(result)] = summary
            return result

        def traced_pop(queue: Any, device: Any, module: Any) -> Any:
            target = None
            if queue is not None:
                try:
                    target = queue[1] if len(queue) > 1 else None
                except Exception:
                    target = None
            begin = time.perf_counter()
            try:
                return original_pop(queue, device, module)
            finally:
                receipt["popCount"] += 1
                elapsed = round(time.perf_counter() - begin, 6)
                receipt["popHostSecondsTotal"] = round(receipt["popHostSecondsTotal"] + elapsed, 6)
                summary = forward_by_queue_id.get(id(queue)) if queue is not None else None
                if summary is not None:
                    summary["popCount"] += 1
                    summary["popHostSecondsTotal"] = round(summary["popHostSecondsTotal"] + elapsed, 6)
                    if elapsed >= summary["popHostSecondsMax"]:
                        summary["popHostSecondsMax"] = elapsed
                        summary["slowestPop"] = {
                            "popIndex": summary["popCount"],
                            "currentBlock": module_name(module),
                            "targetBlock": module_name(target),
                            "elapsedHostSeconds": elapsed,
                        }

        def traced_cleanup(comfy_modules: Any) -> Any:
            # The official cleanup is where its temporary VBAR pins are
            # released.  Count only this lifecycle boundary; do not inspect
            # modules or pages in the per-block hot path.
            receipt["cleanupCount"] += 1
            return original_cleanup(comfy_modules)

        prefetch_module.make_prefetch_queue = traced_make
        prefetch_module.prefetch_queue_pop = traced_pop
        prefetch_module.cleanup_prefetched_modules = traced_cleanup

        def restore() -> None:
            prefetch_module.make_prefetch_queue = original_make
            prefetch_module.prefetch_queue_pop = original_pop
            prefetch_module.cleanup_prefetched_modules = original_cleanup

        return restore
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        return lambda: None


def _prefetch_trace_step_summary(prefetch_trace: Any) -> Dict[str, Any]:
    """Return a small immutable-by-convention snapshot for one sampler step."""

    trace = prefetch_trace if isinstance(prefetch_trace, dict) else {}
    summaries = trace.get("forwardSummaries")
    latest = summaries[-1] if isinstance(summaries, list) and summaries else None
    if not isinstance(latest, dict):
        return {
            "available": bool(trace.get("available")),
            "forwardIndex": None,
            "queueCreated": False,
            "popCount": 0,
            "popHostSecondsTotal": 0.0,
            "popHostSecondsMax": 0.0,
            "slowestPop": None,
            "phaseBreakdown": "unavailable_no_native_forward_observed",
        }
    return {
        "available": bool(trace.get("available")),
        "forwardIndex": latest.get("forwardIndex"),
        "queueIdentity": latest.get("queueIdentity"),
        "queueCreated": bool(latest.get("queueCreated")),
        "requested": bool(latest.get("requested")),
        "queueLength": latest.get("queueLength"),
        "popCount": int(latest.get("popCount", 0) or 0),
        "popHostSecondsTotal": float(latest.get("popHostSecondsTotal", 0.0) or 0.0),
        "popHostSecondsMax": float(latest.get("popHostSecondsMax", 0.0) or 0.0),
        "slowestPop": dict(latest["slowestPop"]) if isinstance(latest.get("slowestPop"), dict) else None,
        "phaseBreakdown": latest.get("phaseBreakdown"),
    }


def _sampling_step_telemetry_snapshot(
    *,
    step: int,
    total_steps: int,
    step_elapsed_seconds: float,
    prefetch_trace: Any,
    native_dynamic_vram: Any,
    resources: Any,
    sage: Any,
    ffn: Any,
    quantization: Any,
    quantization_previous: Any = None,
    denoiser_forward: Any = None,
    acceleration: Any = None,
) -> Dict[str, Any]:
    """Build an independent, compact, JSON-safe sampler-boundary receipt.

    Runtime-stage receipts intentionally remain mutable while a task is running.
    Telemetry history must not retain references to those objects: otherwise a
    later forward mutates evidence for an earlier step.  This function copies
    only scalar observations at the step boundary; it never scans VBARs,
    synchronizes CUDA, or persists anything from a transformer-block hot path.
    """
    prefetch = _prefetch_trace_step_summary(prefetch_trace)
    native = native_dynamic_vram if isinstance(native_dynamic_vram, dict) else {}
    resource = resources if isinstance(resources, dict) else {}
    cuda = resource.get("cuda") if isinstance(resource.get("cuda"), dict) else {}
    process = resource.get("process") if isinstance(resource.get("process"), dict) else {}
    vbars = native.get("vbars") if isinstance(native.get("vbars"), list) else []
    previous_quantization = quantization_previous if isinstance(quantization_previous, dict) else {}
    quantization = quantization if isinstance(quantization, dict) else {}
    executed_int8 = quantization.get("executedInt8LinearCalls")
    fallback_eager = quantization.get("fallbackEagerCalls")
    previous_executed = previous_quantization.get("executedInt8LinearCalls", 0)
    previous_fallback = previous_quantization.get("fallbackEagerCalls", 0)

    def counter_delta(current: Any, previous: Any) -> int | str:
        try:
            return int(current) - int(previous)
        except (TypeError, ValueError):
            return "unavailable"

    forward_trace = denoiser_forward if isinstance(denoiser_forward, dict) else {}
    acceleration = acceleration if isinstance(acceleration, dict) else {}
    forward_summaries = forward_trace.get("forwardSummaries")
    latest_forward = forward_summaries[-1] if isinstance(forward_summaries, list) and forward_summaries else {}
    latest_forward = latest_forward if isinstance(latest_forward, dict) else {}

    compact_vbars = []
    for vbar in vbars:
        if not isinstance(vbar, dict):
            continue
        compact_vbars.append({
            "objectIdentity": vbar.get("objectIdentity"),
            "watermarkPages": vbar.get("watermarkPages", "unavailable"),
            "residentPages": vbar.get("residentPages", "unavailable"),
            "offloadedPages": vbar.get("offloadedPages", "unavailable"),
            "pinnedPages": vbar.get("pinnedPages", "unavailable"),
            "faultCount": vbar.get("faultCount", "unavailable"),
            "unpinCount": vbar.get("unpinCount", "unavailable"),
        })

    def compact(source: Any, fields: tuple[str, ...]) -> Dict[str, Any]:
        source = source if isinstance(source, dict) else {}
        return {field: source.get(field) for field in fields}

    return {
        "step": int(step),
        "totalSteps": int(total_steps),
        "stepElapsedSeconds": float(step_elapsed_seconds),
        "prefetch": {
            "available": bool(prefetch.get("available")),
            "forwardIndex": prefetch.get("forwardIndex"),
            "queueIdentity": prefetch.get("queueIdentity"),
            "queueCreated": bool(prefetch.get("queueCreated")),
            "requested": bool(prefetch.get("requested")),
            "queueLength": prefetch.get("queueLength"),
            "popCount": int(prefetch.get("popCount", 0) or 0),
            "popHostSecondsTotal": float(prefetch.get("popHostSecondsTotal", 0.0) or 0.0),
            "popHostSecondsMax": float(prefetch.get("popHostSecondsMax", 0.0) or 0.0),
            "slowestPop": dict(prefetch["slowestPop"]) if isinstance(prefetch.get("slowestPop"), dict) else None,
            "offloadStreamWait": "unavailable_single_official_pop_call",
            "vbarCastSeconds": "unavailable_single_official_pop_call",
        },
        "dynamicVram": {
            "isDynamic": native.get("isDynamic"),
            "loadedWeightBytes": native.get("loadedWeightBytes"),
            "offloadedWeightBytes": native.get("offloadedWeightBytes"),
            "lowvramPatchCount": native.get("lowvramPatchCount"),
            "vbars": compact_vbars,
        },
        "resources": {
            "torchAllocatedBytes": cuda.get("memoryAllocatedBytes", "unavailable"),
            "torchReservedBytes": cuda.get("memoryReservedBytes", "unavailable"),
            "workerWorkingSetBytes": process.get("workingSetBytes", "unavailable"),
            "workerPrivateBytes": process.get("privateBytes", "unavailable"),
            "workerPageFaultCount": process.get("pageFaultCount", "unavailable"),
        },
        "blockTiming": {
            "forwardIndex": latest_forward.get("forwardIndex", "unavailable"),
            "blockCount": latest_forward.get("blockCount", "unavailable"),
            "totalHostSeconds": latest_forward.get("totalHostSeconds", "unavailable"),
            "maxHostSeconds": latest_forward.get("maxHostSeconds", "unavailable"),
            "slowestBlock": dict(latest_forward["slowestBlock"]) if isinstance(latest_forward.get("slowestBlock"), dict) else None,
            "topBlocks": [dict(item) for item in latest_forward.get("topBlocks", []) if isinstance(item, dict)],
            "attentionHostSeconds": "unavailable_block_boundary_only",
            "ffnHostSeconds": "unavailable_block_boundary_only",
            "weightCastHostSeconds": "unavailable_block_boundary_only",
        },
        "sage": {**compact(sage, ("hookApplied", "patchedBlocks", "backend")), "executionCount": "unavailable_no_sage_boundary_counter"},
        "ffn": {**compact(ffn, ("applied", "chunks", "minTokens")), "executionCount": "unavailable_no_ffn_boundary_counter"},
        "quantization": {
            "executedInt8LinearCalls": executed_int8 if executed_int8 is not None else "unavailable",
            "executedInt8LinearCallsDelta": counter_delta(executed_int8, previous_executed),
            "fallbackEagerCalls": fallback_eager if fallback_eager is not None else "unavailable",
            "fallbackEagerCallsDelta": counter_delta(fallback_eager, previous_fallback),
        },
        "acceleration": {
            "requestedMode": acceleration.get("requestedMode"),
            "actualMode": acceleration.get("actualMode"),
            "status": acceleration.get("status"),
            "stats": dict(acceleration.get("stats", {}) or {}),
        },
    }


def _install_denoiser_trace(model_patcher: Any, runtime_stages: Dict[str, Any], progress: Progress, cancel_event: Any = None):
    """Trace every real H3 denoiser call without changing its math.

    The sampler owns model loading and calls ``BaseModel.apply_model`` after
    model-management has selected a device.  This request-scoped trace makes
    that boundary observable and records the first DiT block/attention that
    does not return.  It is deliberately not a timeout or a test gate.
    """

    model = getattr(model_patcher, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    if model is None or diffusion is None:
        return lambda: None

    execution_receipt = runtime_stages.get("executionReceipt")
    if not isinstance(execution_receipt, dict):
        execution_receipt = {"firstDenoiserBlockAt": None}
        runtime_stages["executionReceipt"] = execution_receipt
    receipt: Dict[str, Any] = {
        "modelType": type(model).__name__,
        "diffusionModelType": type(diffusion).__name__,
        "loadDevice": _device_text(getattr(model_patcher, "load_device", None)),
        "offloadDevice": _device_text(getattr(model_patcher, "offload_device", None)),
        "modelDevice": _device_text(getattr(model, "device", None)),
        "isDynamic": bool(getattr(model_patcher, "is_dynamic", lambda: False)()),
        "manualCastDtype": str(getattr(model, "manual_cast_dtype", None)),
        "inferenceDtype": str(getattr(model, "get_dtype_inference", lambda: None)()),
        "forwardCalls": 0,
        "forwardSummaries": [],
        "attention": {},
    }
    runtime_stages["denoiser_forward"] = receipt
    state = {"active": False, "forwardCall": 0, "firstAttention": True}
    originals = []

    def heartbeat(value: int, message: str, payload: Dict[str, Any]) -> None:
        reporter = getattr(progress, "heartbeat", None)
        if callable(reporter):
            reporter(value, message, payload)
        else:
            progress(value, message)

    original_apply = getattr(model, "apply_model", None)
    original_forward = getattr(diffusion, "_forward", None)
    if original_apply is not None:
        def traced_apply(*args: Any, **kwargs: Any) -> Any:
            if cancel_event is not None and cancel_event.is_set():
                raise H3Cancelled("cancelled before denoiser forward")
            receipt["forwardCalls"] += 1
            first = receipt["forwardCalls"] == 1
            if first:
                value = args[0] if args else kwargs.get("x")
                receipt.update({
                    "inputShape": _shape(value),
                    "inputDevice": _device_text(getattr(value, "device", None)),
                    "inputDtype": str(getattr(value, "dtype", None)),
                    "started": True,
                    "startMonotonic": time.perf_counter(),
                })
                progress(60, "denoiser forward started")
            try:
                return original_apply(*args, **kwargs)
            finally:
                if first:
                    receipt["returned"] = True
                    receipt["elapsedSeconds"] = round(time.perf_counter() - receipt["startMonotonic"], 6)
                    progress(62, f"denoiser forward returned ({receipt['elapsedSeconds']:.3f}s)")

        object.__setattr__(model, "apply_model", traced_apply)
        originals.append((model, "apply_model", original_apply))

    if original_forward is not None:
        def traced_forward(*args: Any, **kwargs: Any) -> Any:
            if cancel_event is not None and cancel_event.is_set():
                raise H3Cancelled("cancelled before H3 core forward")
            value = args[0] if args else kwargs.get("x")
            context = args[2] if len(args) > 2 else kwargs.get("context")
            payload = kwargs.get("minimax_payload")
            state["forwardCall"] += 1
            state["active"] = True
            forward_summary = {
                "forwardIndex": state["forwardCall"],
                "blockCount": 0,
                "totalHostSeconds": 0.0,
                "maxHostSeconds": 0.0,
                "slowestBlock": None,
                "topBlocks": [],
            }
            state["currentForwardSummary"] = forward_summary
            receipt["forwardSummaries"].append(forward_summary)
            if state["forwardCall"] == 1:
                transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
                transformer_options = transformer_options if isinstance(transformer_options, dict) else {}
                receipt.update({
                    "coreForwardStarted": True,
                    "coreInputShape": _shape(value) if not isinstance(value, (list, tuple)) else [_shape(v) for v in value],
                    "coreInputDevice": _device_text(getattr(value, "device", None)) if not isinstance(value, (list, tuple)) else [_device_text(getattr(v, "device", None)) for v in value],
                    "contextShape": _shape(context),
                    "tokenCount": int(context.shape[1]) if getattr(context, "ndim", 0) >= 2 else None,
                    "referencePayloadBlocks": len((payload or {}).get("refs", [])) if isinstance(payload, dict) else None,
                    "coreForwardStartMonotonic": time.perf_counter(),
                    # This is the exact options object after BaseModel has
                    # applied its official DynamicVRAM request, immediately
                    # before MiniMaxH3Model constructs its native queue.
                    "officialPrefetchRequest": bool(transformer_options.get("prefetch_dynamic_vbars", False)),
                    "officialPrefetchRequestSource": "BaseModel._apply_model",
                })
                progress(60, f"H3 denoiser core forward: tokens={receipt.get('tokenCount')}")
            try:
                return original_forward(*args, **kwargs)
            finally:
                if receipt.get("coreForwardStarted") and not receipt.get("coreForwardReturned"):
                    receipt["coreForwardReturned"] = True
                    receipt["coreForwardElapsedSeconds"] = round(time.perf_counter() - receipt["coreForwardStartMonotonic"], 6)
                    state["active"] = False

        object.__setattr__(diffusion, "_forward", traced_forward)
        originals.append((diffusion, "_forward", original_forward))

    blocks = getattr(diffusion, "blocks", None)
    if blocks is not None:
        for index, block in enumerate(list(blocks)):
            block_forward = getattr(block, "forward", None)
            if block_forward is not None:
                def traced_block(*args: Any, _index=index, _original=block_forward, **kwargs: Any) -> Any:
                    if cancel_event is not None and cancel_event.is_set():
                        raise H3Cancelled(f"cancelled before H3 transformer block {_index + 1}")
                    begin = time.perf_counter()
                    forward_summary = state.get("currentForwardSummary")
                    if state["active"] and _index == 0 and not receipt.get("firstDenoiserBlockAt"):
                        first_block_at = _receipt_timestamp()
                        receipt["firstDenoiserBlockAt"] = first_block_at
                        execution_receipt = runtime_stages.get("executionReceipt")
                        if isinstance(execution_receipt, dict):
                            execution_receipt["firstDenoiserBlockAt"] = first_block_at
                        progress(61, f"denoiser block {_index + 1}/{len(blocks)} started")
                    try:
                        result = _original(*args, **kwargs)
                        return result
                    except Exception as exc:
                        if isinstance(forward_summary, dict):
                            forward_summary["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        elapsed = round(time.perf_counter() - begin, 6)
                        if isinstance(forward_summary, dict):
                            block_summary = {"index": _index + 1, "elapsedHostSeconds": elapsed}
                            forward_summary["blockCount"] += 1
                            forward_summary["totalHostSeconds"] = round(forward_summary["totalHostSeconds"] + elapsed, 6)
                            if elapsed >= forward_summary["maxHostSeconds"]:
                                forward_summary["maxHostSeconds"] = elapsed
                                forward_summary["slowestBlock"] = dict(block_summary)
                            top_blocks = list(forward_summary.get("topBlocks") or [])
                            top_blocks.append(block_summary)
                            top_blocks.sort(key=lambda item: item["elapsedHostSeconds"], reverse=True)
                            forward_summary["topBlocks"] = top_blocks[:3]
                        if state["active"]:
                            heartbeat(61, f"denoiser block {_index + 1}/{len(blocks)} returned ({elapsed:.3f}s)", {
                                "kind": "denoiser_block",
                                "forwardIndex": state.get("forwardCall"),
                                "block": _index + 1,
                                "totalBlocks": len(blocks),
                                "event": "returned",
                                "elapsedSeconds": elapsed,
                            })

                object.__setattr__(block, "forward", traced_block)
                originals.append((block, "forward", block_forward))

            attention = getattr(block, "attn", None)
            attention_forward = getattr(attention, "forward", None)
            if attention is not None and attention_forward is not None:
                def traced_attention(*args: Any, _index=index, _original=attention_forward, **kwargs: Any) -> Any:
                    first = state["active"] and state["firstAttention"]
                    if first:
                        state["firstAttention"] = False
                        receipt["attention"].update({
                            "firstBlock": _index + 1,
                            "started": True,
                            "inputShape": _shape(args[0] if args else kwargs.get("x")),
                            "startMonotonic": time.perf_counter(),
                        })
                        progress(61, f"attention started in denoiser block {_index + 1} (shape={receipt['attention']['inputShape']})")
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        if first:
                            receipt["attention"].update({
                                "returned": True,
                                "elapsedSeconds": round(time.perf_counter() - receipt["attention"]["startMonotonic"], 6),
                            })
                            progress(61, "attention returned")

                object.__setattr__(attention, "forward", traced_attention)
                originals.append((attention, "forward", attention_forward))

    def restore() -> None:
        for owner, name, original in reversed(originals):
            object.__setattr__(owner, name, original)

    return restore


def _install_quantization_trace(runtime_stages: Dict[str, Any]) -> Callable[[], None]:
    """Record the actual INT8/ConvRot backend selected by comfy-kitchen.

    This wraps only the registry resolver and returned ``int8_linear``
    implementation. It does not replace a kernel or alter dispatch priority;
    unsupported calls continue through the registry's normal eager fallback.
    Timings are host-side and intentionally unsynchronized so the receipt does
    not add a CUDA synchronization to every production linear layer.
    """

    import collections
    import importlib

    kitchen = importlib.import_module("comfy_kitchen")
    registry = kitchen.registry
    original_get_implementation = registry.get_implementation
    state: Dict[str, Any] = {
        "resolvedCalls": 0,
        "implementationCalls": 0,
        "selectionCounts": collections.Counter(),
        "callCounts": collections.Counter(),
        "functionSelectionCounts": collections.Counter(),
        "functionCallCounts": collections.Counter(),
        "uniqueShapes": collections.Counter(),
        "firstCallHostSeconds": {},
        "resolutionErrors": [],
    }

    def shape_of(value: Any) -> tuple[int, ...] | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            return None

    def backend_name(implementation: Any) -> str:
        explicit = getattr(implementation, "__h3_backend_name__", None)
        if explicit:
            return str(explicit)
        module = str(getattr(implementation, "__module__", ""))
        if "torch_int8_backend" in module:
            return "torch_int8_mm"
        if ".triton" in module:
            return "triton"
        if ".eager" in module:
            return "eager"
        if ".cuda" in module:
            return "cuda"
        return module or type(implementation).__name__

    def traced_get_implementation(func_name: str, backend: Any = None, kwargs: Any = None) -> Any:
        call_kwargs = kwargs or {}
        if func_name not in {"int8_linear", "w4a8_int8_linear"}:
            return original_get_implementation(func_name, backend=backend, kwargs=kwargs)
        try:
            implementation = original_get_implementation(func_name, backend=backend, kwargs=kwargs)
        except Exception as exc:
            state["resolutionErrors"].append(f"{type(exc).__name__}: {exc}")
            raise

        selected = backend_name(implementation)
        x_shape = shape_of(call_kwargs.get("x"))
        weight_shape = shape_of(call_kwargs.get("weight"))
        convrot = bool(call_kwargs.get("convrot", False))
        group_size = int(call_kwargs.get("convrot_groupsize", 256) or 256)
        shape_key = (x_shape, weight_shape, str(call_kwargs.get("out_dtype")), convrot, group_size)
        state["resolvedCalls"] += 1
        state["selectionCounts"][selected] += 1
        state["functionSelectionCounts"][func_name] += 1
        state["uniqueShapes"][shape_key] += 1

        def traced_implementation(*args: Any, **implementation_kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return implementation(*args, **implementation_kwargs)
            finally:
                elapsed = round(time.perf_counter() - started, 6)
                state["implementationCalls"] += 1
                state["callCounts"][selected] += 1
                state["functionCallCounts"][func_name] += 1
                state["firstCallHostSeconds"].setdefault(repr(shape_key), elapsed)

        return traced_implementation

    registry.get_implementation = traced_get_implementation

    def snapshot() -> Dict[str, Any]:
        return {
            "resolvedInt8LinearCalls": state["resolvedCalls"],
            "executedInt8LinearCalls": state["implementationCalls"],
            "selectionCounts": dict(state["selectionCounts"]),
            "callCounts": dict(state["callCounts"]),
            "functionSelectionCounts": dict(state["functionSelectionCounts"]),
            "functionCallCounts": dict(state["functionCallCounts"]),
            "fallbackEagerCalls": int(state["callCounts"].get("eager", 0)),
            "resolutionErrors": list(state["resolutionErrors"]),
        }

    def receipt() -> Dict[str, Any]:
        torch_backend = {}
        try:
            from embedded_h3_runtime.torch_int8_backend import torch_int8_backend_receipt

            torch_backend = torch_int8_backend_receipt()
        except Exception as exc:
            torch_backend = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
        runtime_stages["quantizationBackend"] = {
            "trace": "comfy_kitchen.registry.get_implementation",
            **snapshot(),
            "uniqueShapes": [
                {
                    "inputShape": key[0],
                    "weightShape": key[1],
                    "outputDtype": key[2],
                    "convrot": key[3],
                    "convrotGroupsize": key[4],
                    "resolvedCalls": count,
                }
                for key, count in state["uniqueShapes"].items()
            ],
            "firstCallHostSeconds": dict(state["firstCallHostSeconds"]),
            "torchInt8Backend": torch_backend,
            "note": "host-side timings are unsynchronized; block timings include actual CUDA completion boundaries",
        }
        receipt = runtime_stages["quantizationBackend"]
        execution_receipt = runtime_stages.get("executionReceipt")
        if isinstance(execution_receipt, dict):
            execution_receipt["quantizationDispatch"] = {
                "functionSelectionCounts": dict(state["functionSelectionCounts"]),
                "functionCallCounts": dict(state["functionCallCounts"]),
                "backendSelectionCounts": dict(state["selectionCounts"]),
                "backendCallCounts": dict(state["callCounts"]),
                "fallbackEagerCalls": int(state["callCounts"].get("eager", 0)),
            }
        return receipt

    def restore() -> None:
        registry.get_implementation = original_get_implementation
        receipt()

    restore.snapshot = snapshot  # type: ignore[attr-defined]

    runtime_stages["quantizationBackend"] = {
        "trace": "comfy_kitchen.registry.get_implementation",
        "status": "active",
    }
    return restore


def _quantized_linear_structure_receipt(patcher: Any) -> Dict[str, Any]:
    """Count model quantized linears from structure metadata only."""

    model = getattr(patcher, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    counts: Dict[str, int] = {}
    shapes: Dict[tuple[Any, ...], int] = {}
    total_linear = 0
    quantized = 0
    convrot = 0
    nvfp4 = 0
    if diffusion is None:
        return {"available": False}
    for _name, module in diffusion.named_modules():
        if type(module).__name__ != "Linear":
            continue
        total_linear += 1
        weight = getattr(module, "weight", None)
        quant_format = str(getattr(module, "quant_format", ""))
        params = getattr(weight, "_params", None)
        is_quantized = "QuantizedTensor" in type(weight).__name__ or bool(quant_format)
        if not is_quantized:
            continue
        quantized += 1
        label = quant_format or type(weight).__name__
        counts[label] = counts.get(label, 0) + 1
        if bool(getattr(params, "convrot", False)):
            convrot += 1
        if "nvfp4" in label.lower():
            nvfp4 += 1
        shape = tuple(int(item) for item in getattr(weight, "shape", ()))
        key = (label, str(getattr(weight, "dtype", None)), shape, bool(getattr(params, "convrot", False)), int(getattr(params, "convrot_groupsize", 0) or 0))
        shapes[key] = shapes.get(key, 0) + 1
    return {
        "available": True,
        "totalLinearModules": total_linear,
        "quantizedLinearModules": quantized,
        "convrotLinearModules": convrot,
        "nvfp4LinearModules": nvfp4,
        "quantFormatCounts": counts,
        "uniqueShapes": [
            {
                "quantFormat": key[0],
                "dtype": key[1],
                "weightShape": key[2],
                "convrot": key[3],
                "convrotGroupsize": key[4],
                "count": count,
            }
            for key, count in shapes.items()
        ],
    }


def _warmup_quantized_linear_kernels(
    patcher: Any,
    sequence_tokens: Optional[int],
    torch: Any,
    selected_backend: Optional[str],
) -> Dict[str, Any]:
    """Warm the selected ConvRot backend for the real packed row count.

    Triton's INT8 matmul autotune key includes ``m`` and the independent
    torch INT8 backend has a fused epilogue that also compiles on first use.
    The earlier shape checks exercised the right weight matrices with a tiny
    diagnostic row count, so a real request could still pay first-use
    compilation in block 1.
    This routine invokes the already-selected Linear modules once at the
    request's actual packed row count.  It never changes weights or outputs.
    If the row count would require an unsafe temporary allocation, the
    optimization is transparently skipped and the normal first-use path runs.
    """

    receipt: Dict[str, Any] = {
        "status": "skipped",
        "backend": selected_backend,
        "sequenceTokens": sequence_tokens,
        "source": "selected H3 ConvRot quantization backend",
    }
    if not sequence_tokens or sequence_tokens < 1:
        receipt["reason"] = "packed sequence length is unavailable"
        return receipt
    backend_name = str(selected_backend or "")
    if not (backend_name.startswith("triton") or backend_name == "h3_torch_int8_mm_convrot"):
        receipt["reason"] = "selected quantization backend has no registered warmup path"
        return receipt
    if not bool(getattr(torch, "cuda", None)) or not torch.cuda.is_available():
        receipt["reason"] = "CUDA unavailable"
        return receipt

    model = getattr(patcher, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    if diffusion is None:
        receipt["reason"] = "loaded model has no diffusion_model"
        return receipt
    device = getattr(model, "device", None) or getattr(patcher, "load_device", None)
    if device is None or not str(device).startswith("cuda"):
        receipt["reason"] = f"loaded model device is not CUDA: {device}"
        return receipt

    dtype = getattr(model, "get_dtype_inference", lambda: None)()
    if not isinstance(dtype, torch.dtype):
        dtype = torch.bfloat16

    unique: list[tuple[Any, tuple[int, ...], str]] = []
    seen: set[tuple[int, ...]] = set()
    for name, module in diffusion.named_modules():
        weight = getattr(module, "weight", None)
        params = getattr(weight, "_params", None)
        if params is None or not bool(getattr(params, "convrot", False)):
            continue
        shape = tuple(int(item) for item in getattr(weight, "shape", ()))
        if len(shape) != 2 or shape in seen:
            continue
        seen.add(shape)
        unique.append((module, shape, name))
    if not unique:
        receipt["reason"] = "no ConvRot Linear modules found"
        return receipt

    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes = 0
    # The largest temporary is the FC1 output.  Keep this optimization well
    # below available memory; full-length H3 requests remain fully supported
    # and simply compile kernels on their first actual invocation.
    largest_output = max(int(sequence_tokens) * shape[0] * torch.tensor([], dtype=dtype).element_size() for _, shape, _ in unique)
    safe_budget = min(512 * 1024 * 1024, int(free_bytes * 0.25)) if free_bytes else 512 * 1024 * 1024
    receipt["largestTemporaryBytes"] = largest_output
    receipt["safeTemporaryBudgetBytes"] = safe_budget
    if largest_output > safe_budget:
        receipt["reason"] = "exact-row warmup temporary exceeds safe available-memory budget"
        return receipt

    cache_key = (
        str(selected_backend), int(sequence_tokens), str(dtype),
        tuple(sorted(shape for _module, shape, _name in unique)),
    )
    warmup_identity = _quant_warmup_identity(
        torch,
        selected_backend,
        int(sequence_tokens),
        dtype,
        [shape for _module, shape, _name in unique],
    )
    if cache_key in _QUANT_WARMUP_CACHE:
        receipt.update({
            "status": "cache_hit",
            "cacheHit": True,
            "uniqueShapeCount": len(unique),
            "elapsedSeconds": 0.0,
        })
        return receipt
    persistent = _read_persistent_warmup(warmup_identity)
    if persistent is not None:
        # The manifest proves that compiled kernel metadata is reusable,
        # but it does not warm this Python process's module/kernel dispatch
        # state.  Keep the receipt and continue through the real four-shape
        # calls below; only _QUANT_WARMUP_CACHE is a valid same-process skip.
        receipt.update({
            "persistentCache": True,
            "persistentCachePath": str(_QUANT_WARMUP_MANIFEST),
            "uniqueShapeCount": len(unique),
            "persistentCachedResults": persistent.get("results", []),
        })

    started = time.perf_counter()
    results = []
    try:
        with torch.inference_mode():
            for module, shape, name in unique:
                input_tensor = torch.zeros((int(sequence_tokens), shape[1]), device=device, dtype=dtype)
                torch.cuda.synchronize(device)
                call_started = time.perf_counter()
                output = module(input_tensor)
                torch.cuda.synchronize(device)
                results.append({
                    "module": name,
                    "weightShape": list(shape),
                    "inputShape": [int(sequence_tokens), shape[1]],
                    "outputShape": _shape(output),
                    "elapsedSeconds": round(time.perf_counter() - call_started, 6),
                })
                del input_tensor, output
                gc.collect()
    except Exception as exc:
        receipt.update({
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsedSeconds": round(time.perf_counter() - started, 6),
            "results": results,
        })
        return receipt
    receipt.update({
        "status": "completed",
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "uniqueShapeCount": len(results),
        "results": results,
    })
    persistent_path = _write_persistent_warmup(warmup_identity, results)
    receipt["persistentCacheWritten"] = bool(persistent_path)
    if persistent_path:
        receipt["persistentCachePath"] = persistent_path
    _QUANT_WARMUP_CACHE.add(cache_key)
    return receipt


def _conditioning_receipt(positive: Any) -> Dict[str, Any]:
    """Summarize H3 reference blocks without retaining tensor payloads."""

    blocks: list[Dict[str, Any]] = []
    for entry in positive if isinstance(positive, list) else []:
        metadata = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else {}
        refs = metadata.get("minimax_refs", []) if isinstance(metadata, dict) else []
        for ref in refs if isinstance(refs, list) else []:
            if not isinstance(ref, dict):
                continue
            blocks.append({
                "kind": ref.get("kind"),
                "latentT": ref.get("latent_t"),
                "latentH": ref.get("latent_h"),
                "latentW": ref.get("latent_w"),
                "refAudioT": ref.get("ref_audio_t"),
                "latentShape": _shape(ref.get("latent")),
                "audioLatentShape": _shape(ref.get("audio_latent")),
            })
    return {"conditioningItems": len(positive) if isinstance(positive, list) else 0, "referenceBlocks": blocks}


def _h3_reference_order_receipt(ref_items: Any, ref_blocks: Any) -> Dict[str, Any]:
    """Record the official multimodal presentation and packed-block order."""
    items = [item for item in ref_items if isinstance(item, dict)] if isinstance(ref_items, list) else []
    blocks = [block for block in ref_blocks if isinstance(block, dict)] if isinstance(ref_blocks, list) else []
    presentation = []
    video_audio_pairs = []
    video_ordinal = 0
    audio_ordinal = 0
    image_ordinal = 0
    for item in items:
        kind = item.get("type")
        if kind == "image":
            image_ordinal += 1
            presentation.append({"type": "image", "ordinal": image_ordinal})
        elif kind == "audio":
            audio_ordinal += 1
            presentation.append({"type": "audio", "ordinal": audio_ordinal})
        elif kind == "video":
            video_ordinal += 1
            entry = {"type": "video", "ordinal": video_ordinal}
            if item.get("pairedAudio"):
                paired_ordinal = item.get("videoOrdinal", video_ordinal)
                video_audio_pairs.append({"videoOrdinal": paired_ordinal, "presentationOrder": [len(presentation) + 1]})
            presentation.append(entry)
    if len(items) != len(blocks):
        raise H3RuntimeError("H3 reference presentation and conditioning block counts differ")
    for index, item in enumerate(items):
        if item.get("type") == "video" and item.get("pairedAudio"):
            block = blocks[index]
            if block.get("kind") != "video_audio":
                raise H3RuntimeError(f"paired video item at index {index} must map to video_audio block")
            if item.get("videoOrdinal") != block.get("paired_video_ordinal"):
                raise H3RuntimeError(f"paired video ordinal mismatch at index {index}")
    return {
        "presentationOrder": presentation,
        "conditioningBlockOrder": [
            {"index": index, "kind": block.get("kind")}
            for index, block in enumerate(blocks, 1)
        ],
        "videoAudioPairing": video_audio_pairs,
        "officialOrder": "images_then_video_audio_pairs_then_standalone_audio",
    }


def _h3_packed_sequence_receipt(positive: Any, latent: Any, compiled: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the exact H3 packed sequence after real conditioning.

    H3 attention length is not the Qwen text length: it includes spatial
    patches for every reference and target video plus audio rows and any
    first/last-frame condition rows.  This helper instantiates the same
    low-level ``PackedLayout`` used by the denoiser, using shapes and
    conditioning metadata only.  It does not move or inspect tensor values.
    """

    receipt: Dict[str, Any] = {
        "available": False,
        "source": "comfy.ldm.minimax.model.PackedLayout",
    }
    if not isinstance(positive, list) or not isinstance(latent, dict):
        receipt["error"] = "conditioning or latent payload is not a list/dict"
        return receipt

    context = None
    refs: list[Dict[str, Any]] = []
    keyframes: list[Dict[str, Any]] = []
    frame_count = int((compiled.get("timing") or {}).get("frameCount") or 0)
    for entry in positive:
        if isinstance(entry, (list, tuple)) and entry:
            candidate = entry[0]
            if context is None and _shape(candidate) is not None and len(_shape(candidate) or []) >= 3:
                context = candidate
            metadata = entry[1] if len(entry) > 1 else {}
            if isinstance(metadata, dict):
                metadata_refs = metadata.get("minimax_refs") or []
                if isinstance(metadata_refs, list):
                    refs.extend(item for item in metadata_refs if isinstance(item, dict))
                metadata_keyframes = metadata.get("minimax_keyframes") or []
                if isinstance(metadata_keyframes, list):
                    keyframes.extend(item for item in metadata_keyframes if isinstance(item, dict))
                frame_count = int(metadata.get("minimax_frame_count") or frame_count)

    context_shape = _shape(context)
    samples = latent.get("samples")
    try:
        components = tuple(samples.unbind())
    except Exception as exc:
        receipt["error"] = f"latent samples cannot be unbound: {type(exc).__name__}: {exc}"
        return receipt
    if len(components) != 2:
        receipt["error"] = f"expected video/audio latent pair, got {len(components)} components"
        return receipt
    video_shape = _shape(components[0])
    audio_shape = _shape(components[1])
    if context_shape is None or len(context_shape) < 3:
        receipt["error"] = f"Qwen context shape is unavailable: {context_shape}"
        return receipt
    if video_shape is None or len(video_shape) < 5 or audio_shape is None or len(audio_shape) < 4:
        receipt["error"] = f"unexpected H3 latent shapes: video={video_shape}, audio={audio_shape}"
        return receipt

    try:
        minimax_model = importlib.import_module("comfy.ldm.minimax.model")
        layout = minimax_model.PackedLayout(
            int(context_shape[1]),
            int(video_shape[2]),
            int(video_shape[3]),
            int(video_shape[4]),
            int(audio_shape[-1]),
            keyframes=keyframes or None,
            refs=refs or None,
            frame_count=frame_count or None,
        )
    except Exception as exc:
        receipt["error"] = f"PackedLayout construction failed: {type(exc).__name__}: {exc}"
        return receipt

    receipt.update({
        "available": True,
        "contextShape": context_shape,
        "targetVideoShape": video_shape,
        "targetAudioShape": audio_shape,
        "referenceBlockCount": len(refs),
        "keyframeCount": len(keyframes),
        "frameCount": frame_count,
        "sequenceTokens": int(layout.seq_len),
        "layoutSignature": list(layout.signature),
        "segments": [
            {"start": int(start), "stop": int(stop), "kind": kind}
            for start, stop, kind in layout.segments
        ],
    })
    return receipt


def _build_ref2v_condition(
    h3: Any,
    clip: Any,
    video_vae: _H3VideoVAEProxy,
    audio_vae: _H3AudioVAEProxy,
    prompt: str,
    width: int,
    height: int,
    length: int,
    ref_images: Dict[str, Any],
    ref_videos: Dict[str, Any],
    ref_video_audios: Dict[str, Any],
    ref_audios: Dict[str, Any],
    progress: Progress,
    runtime_stages: Dict[str, Any],
    stop_before_qwen: bool = False,
    reference_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    torch_module: Any = None,
    reference_video_fps: int = 2,
) -> Any:
    """Build the official REF2VA payload with observable Qwen sub-stages.

    This is the direct equivalent of MiniMaxH3ReferenceToVideo.execute. It
    calls the same H3 helpers and preserves the exact reference ordering,
    24fps-to-2fps sampling, timestamps, VAE latents, and conditioning keys,
    while allowing the service to unload both VAEs before Qwen is loaded.
    """

    started = time.perf_counter()
    latent, frame_count = h3._empty_av_latent(width, height, length)
    ref_items: list[Dict[str, Any]] = []
    ref_blocks: list[Dict[str, Any]] = []
    stage_receipts: list[Dict[str, Any]] = []
    reference_metadata = reference_metadata or {}
    runtime_stages["referenceConditioningStages"] = stage_receipts
    media_total = max(1, len(ref_images) + len(ref_videos) + len(ref_audios))
    media_done = 0
    reference_device = str(getattr(getattr(video_vae, "_vae", None), "device", "unknown"))

    def inference_call(fn: Callable[[], Any]) -> Any:
        """Run reference encoders without retaining an autograd graph.

        Reference VAE outputs are inference artifacts.  Keeping autograd enabled
        here retains intermediate activations for every image/video/audio encode,
        which can exhaust VRAM and makes the CPU fallback needlessly expensive.
        This changes only execution bookkeeping; it does not alter model inputs,
        weights, precision, or output math.
        """

        if torch_module is None:
            return fn()
        inference_mode = getattr(torch_module, "inference_mode", None)
        if inference_mode is not None:
            with inference_mode():
                return fn()
        no_grad = getattr(torch_module, "no_grad", None)
        if no_grad is not None:
            with no_grad():
                return fn()
        return fn()

    def stage(name: str, fn: Callable[[], Any], metadata: Optional[Dict[str, Any]] = None) -> Any:
        nonlocal stage_receipts
        begin = time.perf_counter()
        before = _cuda_memory_receipt(torch_module)
        tiled_messages: list[str] = []

        class _TiledFallbackHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                message = record.getMessage()
                if "retrying with tiled VAE encoding" in message:
                    tiled_messages.append(message)

        handler = _TiledFallbackHandler()
        logging.getLogger().addHandler(handler)
        result = None
        failure: Optional[BaseException] = None
        try:
            result = fn()
        except BaseException as exc:
            failure = exc
        finally:
            logging.getLogger().removeHandler(handler)
        after = _cuda_memory_receipt(torch_module)
        observed_encoding_path = None
        observed_preflight = None
        for vae_proxy in (video_vae, audio_vae):
            candidate = getattr(vae_proxy, "last_encode_path", None)
            if candidate:
                observed_encoding_path = candidate
                observed_preflight = getattr(vae_proxy, "last_encode_preflight", None)
                vae_proxy.last_encode_path = None
                vae_proxy.last_encode_preflight = None
                break
        receipt: Dict[str, Any] = {
            "stage": name,
            "status": "error" if failure is not None else "completed",
            "elapsedSeconds": round(time.perf_counter() - begin, 6),
            "memoryBefore": before,
            "memoryAfter": after,
            "encodingPath": "tiled_fallback" if tiled_messages else (observed_encoding_path or "regular"),
            "output": None if failure is not None else _tensor_receipt(result),
        }
        if observed_preflight is not None:
            receipt["memoryPreflight"] = observed_preflight
        if metadata:
            receipt["reference"] = dict(metadata)
        if tiled_messages:
            receipt["tiledFallbackMessages"] = tiled_messages
        if failure is not None:
            receipt["errorType"] = type(failure).__name__
            receipt["error"] = str(failure)
        stage_receipts.append(receipt)
        runtime_stages["referenceConditioningStages"] = stage_receipts
        if failure is not None:
            _best_effort_exception_metadata(failure, "failedStage", name, runtime_stages)
            _best_effort_exception_metadata(failure, "runtimeStages", runtime_stages, runtime_stages)
            raise failure
        return result

    def complete_media(reference: Dict[str, Any]) -> None:
        nonlocal media_done
        media_done += 1
        percent = min(35, 27 + int(round(8 * media_done / media_total)))
        progress(percent, f"reference {reference.get('token', 'material')} encoding complete")

    def encode_ref_audio(audio: Dict[str, Any]) -> tuple[Any, int]:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        vae_rate = getattr(audio_vae, "audio_sample_rate", 32000)
        if sample_rate != vae_rate:
            waveform = h3.torchaudio.functional.resample(waveform, sample_rate, vae_rate)
        encoded = inference_call(lambda: audio_vae.encode(waveform[:1].movedim(1, -1)))
        return encoded, int(encoded.shape[-1])

    for image_name, image in ref_images.items():
        if image is None:
            continue
        h, w = image.shape[1], image.shape[2]
        metadata = dict(reference_metadata.get(image_name, {}))
        metadata.update({"sourceDimensions": {"width": int(w), "height": int(h)}})
        progress(min(35, 27 + int(round(8 * media_done / media_total))), f"reference {metadata.get('token', image_name)} VAE encoding ({reference_device})")
        planned = metadata.get("preprocessPlan") or {}
        resize_plan = planned.get("resize") if isinstance(planned.get("resize"), dict) else planned
        tw = int(resize_plan.get("width") or 0)
        th = int(resize_plan.get("height") or 0)
        if tw <= 0 or th <= 0:
            scale = min(1.0, math.sqrt((width * height) / (w * h)))
            tw = max(h3.CANVAS_MULTIPLE, round(w * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
            th = max(h3.CANVAS_MULTIPLE, round(h * scale / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        resized = stage("reference_image_prepare", lambda image=image, tw=tw, th=th: h3._resize(image[:1], tw, th, "disabled"))
        metadata["officialResize"] = {"width": int(tw), "height": int(th)}
        encoded = stage(
            "reference_image_video_vae_encode",
            lambda resized=resized: inference_call(lambda: video_vae.encode(resized)),
            metadata,
        )
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": encoded})
        ref_images[image_name] = None
        complete_media(metadata)

    for name, video_frames in ref_videos.items():
        if video_frames is None:
            continue
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        metadata = dict(reference_metadata.get(name, {}))
        metadata.update({"sourceDimensions": {"width": int(vw), "height": int(vh)}})
        progress(min(35, 27 + int(round(8 * media_done / media_total))), f"reference {metadata.get('token', name)} VAE encoding ({reference_device})")
        planned = metadata.get("preprocessPlan") or {}
        cw, ch = int(planned.get("width") or 0), int(planned.get("height") or 0)
        if cw <= 0 or ch <= 0:
            cw, ch = h3.adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(h3.CANVAS_MULTIPLE, round(vw / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
                ch = max(h3.CANVAS_MULTIPLE, round(vh / h3.CANVAS_MULTIPLE) * h3.CANVAS_MULTIPLE)
        frames = stage("reference_video_prepare_resize", lambda video_frames=video_frames, cw=cw, ch=ch: h3._resize(video_frames, cw, ch, "disabled"))
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise H3RuntimeError("MiniMax H3 reference videos need at least 5 frames")
        metadata["officialResize"] = {"width": int(cw), "height": int(ch)}
        metadata["legalFrameCount"] = int(n)
        reference_duration = float(metadata.get("referenceDurationSeconds") or metadata.get("sourceDurationSeconds") or (n / reference_video_fps))
        metadata["referenceTimeScale"] = round(reference_duration / max(n / h3.FPS, 1e-6), 6)
        metadata["referenceLoadedFps"] = round(float(metadata.get("referenceFps") or reference_video_fps), 6)
        encoded = stage(
            "reference_video_video_vae_encode",
            lambda frames=frames: inference_call(lambda: video_vae.encode(frames)),
            metadata,
        )
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_metadata = dict(reference_metadata.get("ref_video_audio_" + name.rsplit("_", 1)[-1], metadata))
            audio_metadata["sourceSampleRate"] = int(soundtrack.get("sample_rate", 0))
            audio_metadata["sourceSamples"] = int(soundtrack["waveform"].shape[-1])
            audio_latent, ref_audio_t = stage("reference_video_audio_vae_encode", lambda soundtrack=soundtrack: encode_ref_audio(soundtrack), audio_metadata)
        semantic_fps = int(reference_video_fps)
        if semantic_fps < 1 or semantic_fps > 24:
            raise H3RuntimeError("reference_video_fps must be between 1 and 24")
        source_timestamps = list(metadata.get("referenceTimestampsSeconds") or [])
        vae_timestamps = [round(float(value) - float(metadata.get("selectedStartSeconds") or 0.0), 6) for value in source_timestamps]
        if len(vae_timestamps) != frames.shape[0]:
            vae_timestamps = [round(index / float(metadata.get("referenceFps") or reference_video_fps), 6) for index in range(frames.shape[0])]
        sample_count = max(1, math.ceil(float(metadata.get("referenceDurationSeconds") or vae_timestamps[-1] or 0.0) * semantic_fps))
        target_times = [index / semantic_fps for index in range(sample_count)]
        sample_indices = sorted({min(len(vae_timestamps) - 1, min(range(len(vae_timestamps)), key=lambda pos: abs(vae_timestamps[pos] - target))) for target in target_times})
        qwen_frames = frames[sample_indices]
        timestamps = [vae_timestamps[index] for index in sample_indices]
        input_frame_count = int(frames.shape[0])
        qwen_frame_count = int(qwen_frames.shape[0])
        video_item = {"type": "video", "data": qwen_frames, "timestamps": timestamps}
        if ref_audio_t:
            video_item["pairedAudio"] = True
            video_item["videoOrdinal"] = int(metadata.get("ordinal") or 0)
        ref_items.append(video_item)
        ref_blocks.append({
            "kind": "video_audio" if ref_audio_t else "video",
            "paired_video_ordinal": int(metadata.get("ordinal") or 0),
            "latent_t": encoded.shape[2],
            "latent_h": ch // 16,
            "latent_w": cw // 16,
            "ref_audio_t": ref_audio_t,
            "reference_fps": metadata.get("referenceLoadedFps"),
            "reference_duration": metadata.get("referenceDurationSeconds"),
            "time_scale": metadata.get("referenceTimeScale", 1.0),
            "latent": encoded,
            "audio_latent": audio_latent,
        })
        if soundtrack is not None:
            complete_media(metadata)
        else:
            complete_media(metadata)
        ref_videos[name] = None
        del frames, encoded, qwen_frames
        stage_receipts.append({
            "stage": "reference_video_qwen_sampling_plan",
            "status": "completed",
            "elapsedSeconds": 0.0,
            "inputFps": h3.FPS,
            "qwenFps": semantic_fps,
            "inputFrameCount": input_frame_count,
            "qwenFrameCount": qwen_frame_count,
            "qwenFrameIndices": sample_indices,
            "timestampsSeconds": timestamps,
        })

    for audio_key, audio in ref_audios.items():
        if audio is None:
            continue
        metadata = dict(reference_metadata.get(audio_key, {}))
        metadata["sourceSampleRate"] = int(audio.get("sample_rate", 0))
        metadata["sourceSamples"] = int(audio["waveform"].shape[-1])
        progress(min(35, 27 + int(round(8 * media_done / media_total))), f"reference {metadata.get('token', audio_key)} VAE encoding ({reference_device})")
        audio_latent, ref_audio_t = stage("reference_audio_audio_vae_encode", lambda audio=audio: encode_ref_audio(audio), metadata)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
        complete_media(metadata)

    progress(35, "reference VAE encoding complete; moving ref_blocks to CPU to free VRAM")
    
    # Move all ref_block latents to CPU to free GPU memory before Qwen loads
    for block in ref_blocks:
        if "latent" in block and block["latent"] is not None:
            block["latent"] = block["latent"].cpu()
        if "audio_latent" in block and block["audio_latent"] is not None:
            block["audio_latent"] = block["audio_latent"].cpu()
    
    runtime_stages["reference_conditioning_pre_qwen"] = {
        "elapsedSeconds": round(time.perf_counter() - started, 6),
        "stages": stage_receipts,
        "referenceItemCount": len(ref_items),
        "referenceBlockCount": len(ref_blocks),
        "qwenVideoFrameCount": sum(int(item["data"].shape[0]) for item in ref_items if item.get("type") == "video"),
        "refBlocksMovedToCpu": True,
        **_h3_reference_order_receipt(ref_items, ref_blocks),
    }
    if stop_before_qwen:
        return {"latent": latent, "refItems": ref_items, "refBlocks": ref_blocks}

    return _complete_ref2v_qwen_condition(clip, prompt, latent, ref_items, ref_blocks, h3, runtime_stages, progress)


def _complete_ref2v_qwen_condition(
    clip: Any,
    prompt: str,
    latent: Dict[str, Any],
    ref_items: list[Dict[str, Any]],
    ref_blocks: list[Dict[str, Any]],
    h3: Any,
    runtime_stages: Dict[str, Any],
    progress: Progress,
) -> Any:
    """Run only the Qwen and conditioning half after VAE memory is released."""

    token_started = time.perf_counter()
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    runtime_stages["qwen_tokenize_multimodal_presentation"] = {
        "elapsedSeconds": round(time.perf_counter() - token_started, 6),
        "referenceItemCount": len(ref_items),
        "videoItems": [
            {
                "frameCount": int(item["data"].shape[0]),
                "timestampsSeconds": list(item.get("timestamps", [])),
            }
            for item in ref_items if item.get("type") == "video"
        ],
    }
    progress(41, f"Qwen tokenize/multimodal presentation complete ({runtime_stages['qwen_tokenize_multimodal_presentation']['elapsedSeconds']:.3f}s)")

    qwen_load_started = time.perf_counter()
    clip.load_model(tokens)
    runtime_stages["qwen_weight_mapping_loading"] = {"elapsedSeconds": round(time.perf_counter() - qwen_load_started, 6)}
    progress(43, f"Qwen weights ready ({runtime_stages['qwen_weight_mapping_loading']['elapsedSeconds']:.3f}s)")

    transformer = getattr(getattr(clip, "cond_stage_model", None), "transformer", None)
    visual_elapsed = 0.0
    original_preprocess = getattr(transformer, "preprocess_embed", None)
    if transformer is not None and original_preprocess is not None:
        def timed_preprocess(embed: Dict[str, Any], device: Any) -> Any:
            nonlocal visual_elapsed
            visual_started = time.perf_counter()
            result = original_preprocess(embed, device)
            visual_elapsed += time.perf_counter() - visual_started
            return result
        transformer.preprocess_embed = timed_preprocess
    try:
        qwen_started = time.perf_counter()
        cond = clip.encode_from_tokens_scheduled(tokens)
        qwen_elapsed = time.perf_counter() - qwen_started
    finally:
        if transformer is not None and original_preprocess is not None:
            transformer.preprocess_embed = original_preprocess
    runtime_stages["qwen_visual_encoding"] = {"elapsedSeconds": round(visual_elapsed, 6)}
    runtime_stages["qwen_encode_from_tokens_scheduled"] = {"elapsedSeconds": round(qwen_elapsed, 6)}
    progress(47, f"Qwen visual {visual_elapsed:.3f}s; language condition {qwen_elapsed:.3f}s")

    conditioning_started = time.perf_counter()
    if ref_blocks:
        # ref_blocks stay in CPU - let PackedLayout move them on-demand during sampling
        cond = h3.node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
    runtime_stages["conditioning_set_values"] = {
        "elapsedSeconds": round(time.perf_counter() - conditioning_started, 6),
        "referenceBlockCount": len(ref_blocks),
        "refBlocksKeptInCpu": True,
    }
    progress(49, f"reference conditioning attached ({runtime_stages['conditioning_set_values']['elapsedSeconds']:.3f}s)")
    return h3.io.NodeOutput(cond, latent)


class DirectH3Runner:
    def __init__(self, runtime: Optional[EmbeddedH3Runtime] = None) -> None:
        self.runtime = runtime or EmbeddedH3Runtime()

    def build_plan(self, compiled: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        _validate_compiled_route(compiled)
        model = compiled["routing"]["primaryModel"]
        reference_planning = compiled.get("referencePlanning") or {}
        total_reference_frames = int(reference_planning.get("totalBudgetFrames") or 0)
        advanced = compiled.get("advanced") or {}
        diagnostic_profile = str(
            advanced.get("executionProfile")
            or advanced.get("diagnosticExecutionProfile")
            or "production_kj_sage_ffn"
        )
        native_route_contract = native_acceleration_route_contract(diagnostic_profile)
        steps = int(advanced.get("steps") or 20)
        sampling_contract = _official_h3_sampler_contract(steps)
        sage = choose_sage_policy(
            compiled["timing"]["frameCount"],
            total_reference_frames,
            requested="off" if diagnostic_profile == "official_baseline" or native_route_contract else advanced.get("sageAttention", "auto"),
            threshold=advanced.get("sageThresholdFrames", 124),
        )
        return {
            "taskId": task_id,
            "dryRun": False,
            "backend": "embedded_h3_direct",
            "executionReceipt": initial_execution_receipt(compiled),
            "execution": {
                "model": model,
                "modelPath": str(self.runtime.model_path(model, "full_quality", model_quant=advanced.get("modelQuant", "int8"))),
                "steps": steps,
                "sampling": sampling_contract,
                "timeoutSeconds": compiled.get("advanced", {}).get("executionTimeoutSeconds"),
                "seed": compiled.get("advanced", {}).get("seed"),
                "outputFrameCount": compiled["timing"]["frameCount"],
                "exportFrameCount": compiled["timing"]["exportFrameCount"],
                "ffnChunk": advanced.get("ffnChunk", "auto"),
                "ffnChunks": advanced.get("ffnChunks"),
                "memoryStrategy": advanced.get("memoryStrategy", "auto"),
                "assetPrecision": advanced.get("assetPrecision", "official"),
                "modelQuant": advanced.get("modelQuant", "int8"),
            },
            "modelExclusion": {
                "loadedPrimaryModel": model,
                "notLoaded": "REF2VA" if model == "FL2VA" else "FL2VA",
                "simultaneousPrimaryModels": False,
            },
            "referencePlanning": reference_planning,
            "performance": {
                "sage": sage,
                "executionProfile": diagnostic_profile,
                "executionRouteContract": native_route_contract,
                "diagnosticExecutionProfile": diagnostic_profile if diagnostic_profile != "production_kj_sage_ffn" else None,
                "referenceFrames": total_reference_frames,
                "referencePolicyPreset": reference_planning.get("policyPreset"),
                "referencePolicyOverridden": reference_planning.get("policyOverridden", False),
            },
            "executionBaseline": {
                "name": "faithful_native",
                "preservesPrompt": True,
                "preservesReferenceInterval": True,
                "preservesResolutionAndFps": True,
                "preservesAudio": True,
                "nativeSteps": steps,
                "optimizationMayChangeBackendOnly": True,
                "executionProfile": diagnostic_profile,
            },
            "phases": [
                {"id": "reference_preprocess", "load": ["streaming 24fps reference decode", "2fps semantic samples"], "unload": []},
                {"id": "reference_encode", "load": ["video/audio VAE", "Qwen semantic condition"], "unload": ["reference decode buffers"]},
                {"id": "text_unload", "load": [], "unload": ["Qwen text encoder"]},
                {"id": "primary_sample", "load": [model], "unload": ["the other primary model", "text encoder after condition"]},
                {"id": "decode_export", "load": ["video/audio VAE"], "unload": [model], "sync": {"fps": 24, "exportFrames": compiled["timing"]["exportFrameCount"], "durationSeconds": compiled["timing"]["exportDurationSeconds"]}},
            ],
            "compiled": compiled,
        }

    def execute(
        self,
        plan: Dict[str, Any],
        progress: Optional[Progress] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        progress = progress or (lambda _value, _message: None)
        cancel_event = cancel_event or threading.Event()
        timeout_seconds = plan["execution"].get("timeoutSeconds")
        if timeout_seconds is not None:
            try:
                timeout_seconds = float(timeout_seconds)
            except (TypeError, ValueError) as exc:
                raise H3RuntimeError("executionTimeoutSeconds must be a positive number") from exc
            if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
                raise H3RuntimeError("executionTimeoutSeconds must be a positive number")
        deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None

        def ensure_budget() -> None:
            if cancel_event.is_set():
                raise H3Cancelled("cancelled during H3 execution")
            if deadline is not None and time.monotonic() > deadline:
                raise H3RuntimeError(f"H3 execution timeout exceeded configured budget ({timeout_seconds:g} seconds)")

        if cancel_event.is_set():
            return {"status": "cancelled", "realInference": False}

        model_name = plan["execution"]["model"]
        compiled = plan["compiled"]
        if compiled["references"] and any(not item.get("path") for item in compiled["references"]):
            raise H3RuntimeError("real execution requires staged media paths; metadata-only references cannot be sampled")
        ensure_budget()
        runtime_stages: Dict[str, Any] = {}
        runtime_stages["executionReceipt"] = dict(plan.get("executionReceipt") or initial_execution_receipt(compiled))
        runtime_stages["cancellation"] = {
            "requested": False,
            "status": "not_requested",
            "lastCompletedSamplingStep": 0,
        }
        current_stage = "bootstrap"
        is_r2v = compiled["mode"] == "R2V"
        progress(20, "loading reference/video/audio VAE and Qwen condition stage")
        try:
            if not is_r2v:
                current_stage = "text_encoder_load"
                runtime_stages[current_stage] = self.runtime.load_text_encoder()
                progress(21, "Qwen text encoder ready")
            current_stage = "video_vae_load"
            runtime_stages[current_stage] = self.runtime.load_video_vae()
            progress(23, "video VAE ready")
            current_stage = "audio_vae_load"
            runtime_stages[current_stage] = self.runtime.load_audio_vae()
            progress(25, "audio VAE ready")
            if is_r2v:
                current_stage = "reference_vae_conditioning"
                progress(27, "starting per-reference VAE encoding before Qwen")
            if cancel_event.is_set():
                runtime_stages["cancellation"].update({
                    "requested": True,
                    "status": "cancelled_before_sampling",
                    "observedAtMonotonic": time.monotonic(),
                })
                return {"status": "cancelled", "realInference": False, "outputAuthentic": False, "runtimeStages": runtime_stages}
            current_stage = "reference_conditioning_and_sampling"
            return self._sample_direct(
                plan,
                progress,
                cancel_event,
                deadline,
                timeout_seconds,
                runtime_stages,
            )
            failed_stage = _exception_metadata_value(exc, "failedStage", current_stage, runtime_stages)
            runtime_stages["failedStage"] = failed_stage
            runtime_stages["fallbackUsed"] = False
            execution_receipt = runtime_stages.get("executionReceipt")
            if isinstance(execution_receipt, dict):
                execution_receipt["failedStage"] = failed_stage
                execution_receipt["fallbackUsed"] = False
            error = H3RuntimeError(f"{failed_stage} failed: {exc}")
            error_stages = _exception_runtime_stages(exc, runtime_stages)
            _best_effort_exception_metadata(error, "failedStage", failed_stage, error_stages)
            _best_effort_exception_metadata(error, "runtimeStages", error_stages, error_stages)
            raise error from exc
        except Exception as exc:
            failed_stage = _exception_metadata_value(exc, "failedStage", current_stage, runtime_stages)
            runtime_stages["failedStage"] = failed_stage
            runtime_stages["fallbackUsed"] = False
            execution_receipt = runtime_stages.get("executionReceipt")
            if isinstance(execution_receipt, dict):
                execution_receipt["failedStage"] = failed_stage
                execution_receipt["fallbackUsed"] = False
            error = H3RuntimeError(f"{failed_stage} failed: {exc}")
            error_stages = _exception_runtime_stages(exc, runtime_stages)
            _best_effort_exception_metadata(error, "failedStage", failed_stage, error_stages)
            _best_effort_exception_metadata(error, "runtimeStages", error_stages, error_stages)
            raise error from exc
        finally:
            release_started = time.perf_counter()
            try:
                _restore_memory_strategy(runtime_stages.pop("_memoryManagementObject", None), runtime_stages.get("memoryStrategy"))
            except Exception as exc:
                runtime_stages.setdefault("memoryStrategy", {})["restoreError"] = f"{type(exc).__name__}: {exc}"
            # Persist component-scoped release receipts on every terminal
            # path.  Each runtime method delegates to Comfy's targeted
            # ``unload_model_and_clones`` API, never a broad process unload.
            lifecycle_release: Dict[str, Any] = {}
            if cancel_event.is_set():
                runtime_stages.setdefault("cancellation", {}).update({
                    "cleanupStartedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "cleanupOwnedBy": "spawned_h3_worker",
                })
            for name, release in (
                ("textEncoder", self.runtime.unload_text_encoder),
                ("primary", self.runtime.unload_primary),
                ("videoVae", self.runtime.unload_video_vae),
                ("audioVae", self.runtime.unload_audio_vae),
            ):
                try:
                    lifecycle_release[name] = release()
                except Exception as exc:
                    lifecycle_release[name] = {"error": f"{type(exc).__name__}: {exc}"}
            runtime_stages["final_component_release"] = lifecycle_release
            execution_receipt = runtime_stages.get("executionReceipt")
            if isinstance(execution_receipt, dict):
                execution_receipt["stageTimings"] = _stage_timings_receipt(runtime_stages)
            if cancel_event.is_set():
                runtime_stages["cancellation"].update({
                    "requested": True,
                    "status": "gpu_release_complete",
                    "cleanupFinishedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "releaseSeconds": round(time.perf_counter() - release_started, 6),
                    "componentRelease": lifecycle_release,
                })

    def _sample_direct(
        self,
        plan: Dict[str, Any],
        progress: Progress,
        cancel_event: threading.Event,
        deadline: Optional[float],
        timeout_seconds: Optional[float],
        runtime_stages: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the direct AV conditioning/sampling/export chain when requested."""

        objects = self.runtime.objects()
        modules = objects["modules"]
        torch = modules["torch"]
        advanced = plan["compiled"].get("advanced") or {}
        import gc
        resource_telemetry = runtime_stages.setdefault("resourceTelemetry", {})
        resource_scope: Dict[str, Any] = {"name": "reference_preprocess", "baseline": None}
        # Keep the shared receipt bound before any nested sampler callback can
        # observe it.  The callback updates this object during sampling, while
        # later lifecycle stages may re-read it; leaving the local unbound made
        # otherwise valid generations fail at the conditioning boundary.
        execution_receipt = runtime_stages.get("executionReceipt")
        if not isinstance(execution_receipt, dict):
            execution_receipt = initial_execution_receipt(plan["compiled"])
            runtime_stages["executionReceipt"] = execution_receipt

        def checkpoint_progress(value: int, message: str) -> None:
            """Report only after a Qwen/VAE CPU-side boundary has returned."""
            if cancel_event.is_set():
                raise H3Cancelled("cancelled at Qwen/VAE progress boundary")
            progress(value, message)

        def reset_resource_peaks(scope: str) -> None:
            # These counters are diagnostic only.  They reset PyTorch's
            # allocator accounting, never model placement or DynamicVRAM.
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            resource_scope["name"] = scope
            resource_scope["baseline"] = _resource_cooperation_receipt(torch)

        def record_resources(stage: str) -> None:
            # Observation only: neither the compiler nor model-management
            # reads this receipt to decide precision, residency or cancellation.
            receipt = _resource_cooperation_receipt(torch)
            receipt["measurementScope"] = resource_scope["name"]
            baseline = resource_scope.get("baseline") or {}
            start_faults = ((baseline.get("process") or {}).get("pageFaultCount"))
            end_faults = ((receipt.get("process") or {}).get("pageFaultCount"))
            receipt["pageFaultsSinceScopeStart"] = (
                int(end_faults) - int(start_faults)
                if start_faults is not None and end_faults is not None else None
            )
            if stage in {"primary_model_pre_sampler", "primary_sampling_complete", "mp4_export_complete"}:
                receipt["externalUtilization"] = _external_resource_receipt()
            resource_telemetry[stage] = receipt

        reset_resource_peaks("reference_preprocess")
        record_resources("reference_preprocess_enter")
        h3 = modules["h3_conditioning"]
        compiled = plan["compiled"]
        steps = int(plan["execution"].get("steps") or 20)
        target_width = compiled["canvas"]["stage2TargetCanvas"]["width"]
        target_height = compiled["canvas"]["stage2TargetCanvas"]["height"]
        width = compiled["canvas"]["stage1SourceCanvas"]["width"]
        height = compiled["canvas"]["stage1SourceCanvas"]["height"]
        length = compiled["timing"]["frameCount"]
        refs = compiled["references"]
        paths = {item["token"]: _verified_staged_path(item) for item in refs}
        runtime_stages["asset_integrity"] = {
            "status": "verified", "algorithm": "SHA256",
            "assets": [{"token": item["token"], "sha256": item["sha256"], "path": item["path"]} for item in refs],
        }
        mode = compiled["mode"]
        images = []
        if mode in {"I2V", "FIRST_LAST_FRAME"}:
            images = []
            for item in refs:
                if item["kind"] != "Picture":
                    continue
                pixels, image_receipt = _load_image(paths[item["token"]], torch, item.get("preprocessPlan"), return_receipt=True)
                item["imagePreprocessReceipt"] = image_receipt
                images.append((pixels, item))

        model_management = modules["model_management"]
        runtime_stages["memoryStrategy"] = _apply_memory_strategy(model_management, plan["execution"].get("memoryStrategy", "auto"))
        runtime_stages["_memoryManagementObject"] = model_management
        execution_receipt = runtime_stages.get("executionReceipt")
        if isinstance(execution_receipt, dict):
            execution_receipt["assetPrecision"] = (compiled.get("advanced") or {}).get("assetPrecision", "official")
            execution_receipt["memoryStrategy"] = runtime_stages["memoryStrategy"].get("requested", "auto")
            execution_receipt["memoryStrategyReceipt"] = {
                key: value for key, value in runtime_stages["memoryStrategy"].items()
                if key != "previous"
            }
        reference_vae_policy = _reference_vae_cpu_policy(torch) if mode == "R2V" else {"selected": "gpu", "reason": "not_r2v"}
        if mode == "R2V" and reference_vae_policy["selected"] == "cpu":
            runtime_stages["reference_video_vae_device"] = {
                "selected": "cpu",
                "reason": reference_vae_policy["reason"],
                "totalBytes": reference_vae_policy.get("totalBytes"),
                "unloadGpuWrapper": self.runtime.unload_video_vae(),
                "loadCpuWrapper": self.runtime.load_video_vae(device=torch.device("cpu")),
            }
            objects = self.runtime.objects()
        else:
            runtime_stages["reference_video_vae_device"] = reference_vae_policy
        video_vae = _H3VideoVAEProxy(
            objects["videoVae"], objects.get("videoVaePatcher"), model_management,
            torch_module=torch, prefer_tiled=(mode == "R2V" and reference_vae_policy["selected"] != "cpu"),
        )
        audio_vae = _H3AudioVAEProxy(objects["audioVae"], objects.get("audioVaePatcher"), model_management, torch_module=torch)
        conditioning_started = time.perf_counter()
        stage2_positive = None
        if mode in {"T2V", "I2V", "FIRST_LAST_FRAME"}:
            first = images[0][0] if mode in {"I2V", "FIRST_LAST_FRAME"} else None
            last = images[1][0] if mode == "FIRST_LAST_FRAME" else None
            output = h3.MiniMaxH3ImageToVideo.execute(
                objects["clip"], video_vae, compiled["prompt"], width, height, length,
                first_frame=first, last_frame=last,
            )
            if mode in {"I2V", "FIRST_LAST_FRAME"}:
                stage2_positive = _build_stage2_keyframe_conditioning(
                    h3, objects["clip"], video_vae, compiled["prompt"],
                    int(target_width), int(target_height), length, first, last,
                )
        else:
            ref_images = {}
            ref_videos = {}
            ref_video_audios = {}
            ref_audios = {}
            reference_cache_receipts = []
            reference_decode_receipts = []
            reference_metadata: Dict[str, Dict[str, Any]] = {}
            picture_ordinal = 0
            video_ordinal = 0
            audio_ordinal = 0
            for index, item in enumerate(refs, 1):
                if cancel_event.is_set():
                    raise H3Cancelled("cancelled before reference media batch")
                path = paths[item["token"]]
                if item["kind"] == "Picture":
                    picture_ordinal += 1
                    key = f"ref_image_{picture_ordinal}"
                    reference_metadata[key] = {"ordinal": picture_ordinal, "token": item["token"], "name": item["name"], "preprocessPlan": item.get("preprocessPlan") or {}}
                    pixels, image_receipt = _load_image(path, torch, item.get("preprocessPlan"), return_receipt=True)
                    reference_metadata[key]["imagePreprocess"] = image_receipt
                    ref_images[key] = pixels
                elif item["kind"] == "Video":
                    video_ordinal += 1
                    key = f"ref_video_{video_ordinal}"
                    video_plan = next(
                        item for item in (compiled.get("referencePlanning") or {}).get("videos", [])
                        if item["videoOrdinal"] == video_ordinal
                    )
                    reference_metadata[key] = {"ordinal": video_ordinal, "token": item["token"], "name": item["name"], "preprocessPlan": item.get("preprocessPlan") or {}}
                    video_plan = next(
                        item for item in (compiled.get("referencePlanning") or {}).get("videos", [])
                        if item["videoOrdinal"] == video_ordinal
                    )
                    cache_key = video_plan["cache"]["key"]
                    if video_plan["cache"].get("eligible"):
                        cached = read_manifest(cache_key)
                        reference_cache_receipts.append(
                            cached and {"cacheKey": cache_key, "cacheHit": True, "kind": "metadata_only"}
                            or write_manifest(cache_key, video_plan)
                        )
                    else:
                        reference_cache_receipts.append({"cacheKey": cache_key, "cacheHit": False, "kind": "metadata_only", "reason": video_plan["cache"].get("missReason")})
                    reference_fps = int((advanced.get("reference_video_vae_fps") or 12))
                    if reference_fps == h3.FPS:
                        decoded_video, decode_receipt = _load_video(path, torch, video_plan, return_receipt=True)
                        reference_object = {
                            "frames": decoded_video,
                            "source_fps": decode_receipt.get("sourceFps"),
                            "source_duration": float(video_plan["selectedInterval"]["durationSeconds"]),
                            "loaded_fps": h3.FPS,
                            "loaded_frame_count": int(decoded_video.shape[0]),
                            "loaded_duration": float(video_plan["selectedInterval"]["durationSeconds"]),
                            "timestamps": list(decode_receipt.get("targetTimestampsSeconds") or []),
                        }
                    else:
                        reference_object = _load_video_force_rate(
                            path, torch, reference_fps,
                            float(video_plan["selectedInterval"]["startSeconds"]),
                            float(video_plan["selectedInterval"]["endSeconds"]),
                            cancel_event,
                        )
                        decoded_video = reference_object["frames"]
                        decode_receipt = {
                            "sourceFps": reference_object["source_fps"],
                            "sourceDurationSeconds": reference_object["source_duration"],
                            "referenceFps": reference_object["loaded_fps"],
                            "referenceFrameCount": reference_object["loaded_frame_count"],
                            "referenceDurationSeconds": reference_object["loaded_duration"],
                            "referenceTimestampsSeconds": reference_object["timestamps"],
                            "timeMapping": "reference_fps_preserves_source_duration",
                        }
                    decoded_video = reference_object["frames"]
                    if reference_fps == h3.FPS:
                        decode_receipt = {
                            **decode_receipt,
                            "referenceFps": h3.FPS,
                            "referenceFrameCount": int(decoded_video.shape[0]),
                            "referenceDurationSeconds": float(video_plan["selectedInterval"]["durationSeconds"]),
                            "referenceTimestampsSeconds": list(reference_object["timestamps"]),
                            "timeMapping": "official_h3_reference_24fps",
                        }
                    reference_decode_receipts.append(decode_receipt)
                    reference_metadata[key].update(decode_receipt)
                    ref_videos[f"ref_video_{video_ordinal}"] = decoded_video
                    if _has_audio_stream(path):
                        # Official H3 pairs this soundtrack with the same video
                        # ordinal; it is not an additional uploaded Audio slot.
                        audio_key = f"ref_video_audio_{video_ordinal}"
                        audio, audio_receipt = _load_audio(
                            path, torch,
                            float(video_plan["selectedInterval"]["startSeconds"]),
                            float(video_plan["selectedInterval"]["endSeconds"]),
                            return_receipt=True,
                        )
                        reference_metadata[audio_key] = {"ordinal": video_ordinal, "token": item["token"], "name": item["name"], "pairedWithVideo": True, "audioWindow": audio_receipt}
                        ref_video_audios[audio_key] = audio
                else:
                    audio_ordinal += 1
                    key = f"ref_audio_{audio_ordinal}"
                    reference_metadata[key] = {"ordinal": audio_ordinal, "token": item["token"], "name": item["name"]}
                    audio, audio_receipt = _load_audio(path, torch, return_receipt=True)
                    reference_metadata[key]["audioWindow"] = audio_receipt
                    ref_audios[key] = audio
            if self.runtime.objects().get("clip") is not None:
                raise H3RuntimeError("R2V reference path expected Qwen to load after VAE conditioning")
            prepared = _build_ref2v_condition(
                h3, None, video_vae, audio_vae, compiled["prompt"], width, height, length,
                ref_images, ref_videos, ref_video_audios, ref_audios, checkpoint_progress, runtime_stages, stop_before_qwen=True,
                reference_metadata=reference_metadata, torch_module=torch,
                reference_video_fps=int((compiled.get("advanced") or {}).get("reference_video_fps") or 2),
            )
            reference_video_count = len(ref_videos)
            reference_video_audio_count = len(ref_video_audios)
            reference_audio_count = len(ref_audios)
            runtime_stages["reference_vae_unload"] = {
                "video": self.runtime.unload_video_vae(),
                "audio": self.runtime.unload_audio_vae(),
            }
            # Native targeted patcher release is complete before Qwen begins.
            # This receipt makes reference-VAE/Qwen residency overlap visible
            # without using resource values as a request gate.
            record_resources("reference_vae_released_before_qwen")
            del video_vae, audio_vae, ref_images, ref_videos, ref_video_audios, ref_audios
            # The initial objects snapshot also owns both official VAE wrappers;
            # drop it before loading Qwen so model-management can reclaim VRAM.
            del objects
            import gc
            gc.collect()
            reset_resource_peaks("qwen_conditioning")
            progress(39, "reference latents retained; loading Qwen text encoder")
            qwen_load = self.runtime.load_text_encoder()
            if cancel_event.is_set():
                raise H3Cancelled("cancelled after Qwen load boundary")
            runtime_stages["text_encoder_load"] = qwen_load
            record_resources("qwen_loaded")
            qwen_objects = self.runtime.objects()
            output = _complete_ref2v_qwen_condition(
                qwen_objects["clip"], compiled["prompt"], prepared["latent"], prepared["refItems"],
                prepared["refBlocks"], h3, runtime_stages, checkpoint_progress,
            )
            # ``objects()`` is a diagnostic snapshot, so it keeps the Qwen
            # wrapper alive even after the runtime clears its own reference.
            # Drop that snapshot before the primary model lifecycle begins.
            del qwen_objects
            # ``prepared`` is only the hand-off container used to construct
            # the final positive conditioning.  Keeping it through sampling
            # retains the reference latent/ref-block containers in addition
            # to the tensor graph that the sampler actually consumes.  Drop
            # this redundant owner now; the final ``positive`` and ``latent``
            # values below remain the sole conditioning inputs.
            del prepared
            gc.collect()
        positive, latent = _node_args(output)[:2]
        if stage2_positive is None:
            stage2_positive = positive
        latent = _prepare_low_fps_conditioning(compiled, latent, h3, torch, runtime_stages)
        reference_receipt = _conditioning_receipt(positive)
        reference_receipt.update({
            key: value
            for key, value in (runtime_stages.get("reference_conditioning_pre_qwen") or {}).items()
            if key in {"presentationOrder", "conditioningBlockOrder", "videoAudioPairing", "officialOrder"}
        })
        reference_receipt.update({
            "elapsedSeconds": round(time.perf_counter() - conditioning_started, 6),
            "referenceVideoCount": reference_video_count if mode == "R2V" else 0,
            "referenceAudioCount": reference_audio_count if mode == "R2V" else 0,
            "referenceVideoAudioCount": reference_video_audio_count if mode == "R2V" else 0,
            "latentSamplesShape": _shape(latent.get("samples")) if isinstance(latent, dict) else None,
        })
        runtime_stages["reference_conditioning"] = reference_receipt
        packed_sequence = _h3_packed_sequence_receipt(positive, latent, compiled)
        runtime_stages["packed_sequence"] = packed_sequence
        execution_receipt = runtime_stages.get("executionReceipt")
        if isinstance(execution_receipt, dict):
            execution_receipt["latentShapes"] = {
                "video": packed_sequence.get("targetVideoShape") or reference_receipt.get("latentSamplesShape"),
                "audio": packed_sequence.get("targetAudioShape"),
            }
            execution_receipt["packedTokens"] = packed_sequence.get("sequenceTokens")
        if packed_sequence.get("available"):
            reference_receipt["packedSequenceTokens"] = packed_sequence["sequenceTokens"]
            advanced = compiled.get("advanced") or {}
            diagnostic_profile = str(
                advanced.get("executionProfile")
                or advanced.get("diagnosticExecutionProfile")
                or "production_kj_sage_ffn"
            )
            native_route_contract = native_acceleration_route_contract(diagnostic_profile)
            requested_kernel = str(compiled.get("requestedKernel") or "kijai_fast")
            prior_sage = (plan.get("performance") or {}).get("sage") or {}
            updated_sage = choose_sage_policy(
                compiled["timing"]["frameCount"],
                int((compiled.get("referencePlanning") or {}).get("totalBudgetFrames") or 0),
                requested="off" if requested_kernel == "official_native" or native_route_contract else "on",
                threshold=1 if requested_kernel == "kijai_fast" else advanced.get("sageThresholdFrames", 124),
                available=prior_sage.get("available"),
                estimated_tokens=packed_sequence["sequenceTokens"],
                token_threshold=1 if requested_kernel == "kijai_fast" else advanced.get("sageTokenThresholdTokens", 2048),
            )
            plan.setdefault("performance", {})["sage"] = updated_sage
            plan["performance"]["executionProfile"] = diagnostic_profile
            plan["performance"]["executionRouteContract"] = native_route_contract
            plan["performance"]["diagnosticExecutionProfile"] = diagnostic_profile if diagnostic_profile != "production_kj_sage_ffn" else None
            plan["performance"]["packedSequence"] = packed_sequence
        if cancel_event.is_set():
            return {"status": "cancelled", "realInference": False, "outputAuthentic": False}
        if deadline is not None and time.monotonic() > deadline:
            raise H3RuntimeError(f"H3 execution timeout exceeded configured budget ({timeout_seconds:g} seconds) after conditioning")
        selected_backend = ((plan.get("performance") or {}).get("sage") or {}).get("selected", "torch_sdpa")
        sequence_text = (
            f"; packed sequence {packed_sequence['sequenceTokens']} tokens; attention {selected_backend}"
            if packed_sequence.get("available") else ""
        )
        progress(50, f"Qwen condition complete{sequence_text}; handing model lifecycle to native sampler")
        # The Qwen wrapper has finished producing the conditioning tensors.
        # Release it before REF2VA is loaded so the native DynamicVRAM path
        # can reserve the primary model's pages without carrying a second
        # large text-encoder resident set.  ``positive``/``latent`` are plain
        # conditioning outputs and remain intact; this changes only model
        # lifetime, not prompt, reference, token, or sampler semantics.
        qwen_unload_started = time.perf_counter()
        qwen_unload = self.runtime.unload_text_encoder()
        # Keep the unload receipt under the name used by the completed-task
        # result.  The prior path performed the unload correctly but referred
        # to an undefined local while assembling the final receipt, turning a
        # real MP4 into an error-state task after export.
        text_unload = qwen_unload
        gc.collect()
        runtime_stages["text_encoder_unload"] = {
            **qwen_unload,
            "deferred": False,
            "reason": "Qwen conditioning complete; release text encoder before native REF2VA load",
            "elapsedSeconds": round(time.perf_counter() - qwen_unload_started, 6),
        }
        record_resources("qwen_unloaded_before_primary")
        # Conditioning is now tensor-only.  Do not keep either VAE resident
        # while REF2VA/FL2VA is loaded; they are reloaded after primary
        # sampling for the decode/export phase.
        runtime_stages["conditioning_vae_unload"] = {
            "video": self.runtime.unload_video_vae(),
            "audio": self.runtime.unload_audio_vae(),
        }
        runtime_stages["conditioning_to_primary_memory_release"] = _release_completed_conditioning_memory(
            torch,
            model_management,
        )
        # This is the lifecycle boundary that matters to DynamicVRAM: all
        # completed conditioning-model patchers have been explicitly released
        # from Comfy's registry before the selected primary is constructed.
        # It is observation only and never rejects or changes a request.
        record_resources("conditioning_models_released_before_primary")
        reset_resource_peaks("primary_load_and_sampling")
        if mode != "R2V":
            del video_vae, audio_vae, objects
        gc.collect()
        progress(54, f"loading {plan['execution']['model']} after condition stage")
        model_load = self.runtime.load_primary(
            plan["execution"]["model"],
            "full_quality",
            model_quant=plan["execution"]["modelQuant"],
        )
        if cancel_event.is_set():
            raise H3Cancelled("cancelled after primary-model load boundary")
        runtime_stages["primary_model_load"] = model_load
        record_resources("primary_model_pre_sampler")
        objects = self.runtime.objects()
        shifted = _node_args(h3.MiniMaxH3SigmaShift.execute(objects["primary"], 12.0, 3.0))[0]
        sage_receipt = (plan.get("performance") or {}).get("sage") or {}
        requested_kernel = str(compiled.get("requestedKernel") or "kijai_fast")
        effective_kernel = requested_kernel
        kernel_fallback_reason: Optional[str] = None
        actual_hook_receipt: Optional[Dict[str, Any]] = None
        native_route_contract = native_acceleration_route_contract(diagnostic_profile)
        compiled_route = compiled.get("algorithmRoute") if isinstance(compiled.get("algorithmRoute"), dict) else {}
        if native_route_contract:
            _validate_native_experiment_route_identity(compiled, diagnostic_profile)
        if native_route_contract:
            # Native author routes start from the SigmaShift model and add
            # only their public node before the stock guider.  Project Sage,
            # FFN and DynamicVRAM helpers are deliberately not composed here.
            if str(native_route_contract.get("routeId") or "") == KJ_EXPERIMENT_ROUTE:
                if requested_kernel != KJ_EXPERIMENT_KERNEL:
                    raise H3RuntimeError("experimental kernel identity mismatch; fallback is forbidden")
                apply_isolated, sage_runtime_cls = _isolated_kj_adapter()
                shifted, isolated_receipt = apply_isolated(
                    shifted,
                    sage_runtime_cls(),
                    requested=KJ_EXPERIMENT_KERNEL,
                    actual=KJ_EXPERIMENT_KERNEL,
                    head_chunks=1,
                )
                sage_receipt.update({
                    "selected": "sage",
                    "hookApplied": True,
                    "hookImplementation": "isolated_native_kj_h3",
                    "hookReceipt": isolated_receipt,
                })
                actual_hook_receipt = isolated_receipt
            else:
                sage_receipt.update({
                    "selected": "off",
                    "hookApplied": False,
                    "disabledByExecutionRoute": native_route_contract["routeId"],
                })
            runtime_stages["execution_route_isolation"] = {
                "contract": native_route_contract,
                "verifiedAbsent": [] if native_route_contract.get("routeId") == KJ_EXPERIMENT_ROUTE else ["kjH3Sage", "ffnChunk", "projectDynamicVramInjection"],
            }
            execution_receipt = runtime_stages.get("executionReceipt")
            if isinstance(execution_receipt, dict):
                execution_receipt["executionRouteContract"] = dict(native_route_contract)
        elif requested_kernel == "kijai_fast":
            shifted, actual_hook_receipt, effective_kernel, kernel_fallback_reason = _apply_kijai_or_native_fallback(
                shifted, sage_receipt, apply_h3_memory_efficient_sage_patch
            )
            if effective_kernel == "official_native":
                runtime_stages["official_native_kernel"] = {
                    "backend": "comfyui_official_native_attention",
                    "kijaiPatchApplied": False,
                    "genericSageApplied": False,
                    "scope": "official_native",
                    "fallback": True,
                    "requestedKernel": requested_kernel,
                    "fallbackReason": kernel_fallback_reason,
                }
        else:
            sage_receipt.update({
                "selected": "official_native",
                "hookApplied": False,
                "hookImplementation": None,
                "hookReason": "explicit_official_native_selection",
                "fallback_reason": None,
            })
            runtime_stages["official_native_kernel"] = {
                "backend": "comfyui_official_native_attention",
                "kijaiPatchApplied": False,
                "genericSageApplied": False,
                "scope": "official_native",
                "fallback": False,
            }
        advanced = compiled.get("advanced") or {}
        # FFN chunking belongs only to the validated product and diagnostic
        # paths. The product patch is applied by the second sampler call, so
        # stage 1 never receives a wrapped model.
        ffn_chunk_receipt = None
        ffn_chunk_profiles = {
            "production_kj_sage_ffn",
            "ffn_chunk_diagnostic",
            "ffn_chunk_b2_calibration",
            "ffn_chunk_b2_candidate",
        }
        requested_ffn_chunks = advanced.get("ffnChunks")
        if diagnostic_profile in ffn_chunk_profiles and effective_kernel == "kijai_fast":
            if requested_ffn_chunks not in {1, 2, 4}:
                if requested_ffn_chunks is None:
                    requested_ffn_chunks = 2
                else:
                    raise H3RuntimeError("unsupported FFN chunk mapping")
            production_ffn = diagnostic_profile in {
                "production_kj_sage_ffn",
            }
            ffn_chunk_receipt = {
                "enabled": True,
                "applied": True,
                "chunks": int(requested_ffn_chunks),
                "minTokens": 4096,
                "scope": "production" if production_ffn else "diagnostic_only",
                "productionDefault": production_ffn,
                "executionProfile": diagnostic_profile,
                "verifiedProductRoute": False,
                "stage1FfnChunks": 1,
                "stage2FfnChunks": int(requested_ffn_chunks),
                "stage1FfnApplied": False,
                "stage2FfnApplied": False,
            }
            runtime_stages["ffn_chunk" if production_ffn else "ffn_chunk_diagnostic"] = ffn_chunk_receipt
            plan.setdefault("performance", {})["ffnChunk"] = ffn_chunk_receipt
            if production_ffn:
                plan["performance"]["productionAcceleration"] = {
                    "executionProfile": diagnostic_profile,
                    "sageSelected": (plan.get("performance") or {}).get("sage", {}).get("selected") == "sage",
                    "sageHookApplied": bool(sage_receipt.get("hookApplied")),
                    "ffnTwoWayChunkApplied": bool(ffn_chunk_receipt.get("applied")),
                    "dynamicVramReceipt": model_load,
                    "tritonReceiptPending": "reported after quantized layers execute",
                }
        low_fps_remap_state: Optional[Dict[str, Any]] = None
        actual_route_id = str(compiled_route.get("algorithmRoute") or "")
        if actual_route_id == LOW_FPS_TIME_REMAP_ROUTE:
            raise H3RuntimeError("non-24 FPS routes are retired; requested FPS must be exactly 24")
        elif actual_route_id not in {LATENT_UPSCALE_TWO_STAGE_ROUTE, KJ_EXPERIMENT_ROUTE}:
            raise H3RuntimeError(f"unsupported worker route: {actual_route_id or 'missing'}")
        if actual_route_id == KJ_EXPERIMENT_ROUTE:
            kernel_backend_receipt = {
                "contractVersion": native_route_contract["routeVersion"],
                "routeId": KJ_EXPERIMENT_ROUTE,
                "kernelBackend": KJ_EXPERIMENT_KERNEL,
                "requested": KJ_EXPERIMENT_KERNEL,
                "actual": KJ_EXPERIMENT_KERNEL,
                "patchedBlocks": 50,
                "scope": "denoiser_only",
                "fallback": False,
            }
            kernel_summary = dict(kernel_backend_receipt)
        else:
            kernel_backend_receipt = _verified_kernel_backend_receipt(effective_kernel, actual_hook_receipt, ffn_chunk_receipt, steps, requested_ffn_chunks)
            kernel_summary = finalize_kernel_backend_receipt(effective_kernel, actual_hook_receipt, ffn_chunk_receipt, steps, requested_ffn_chunks)
            if kernel_fallback_reason:
                kernel_backend_receipt = {
                    **kernel_backend_receipt,
                    "requestedKernel": requested_kernel,
                    "fallback": True,
                    "compatibilityFallback": True,
                    "fallbackReason": kernel_fallback_reason,
                }
                kernel_summary.update({
                    "requestedKernel": requested_kernel,
                    "fallback": True,
                    "fallbackReason": kernel_fallback_reason,
                })
        runtime_stages["kernelBackend"] = kernel_backend_receipt
        execution_receipt = runtime_stages.get("executionReceipt")
        if isinstance(execution_receipt, dict):
            execution_receipt.update(kernel_summary)
            execution_receipt["kernelBackendReceipt"] = kernel_backend_receipt
            execution_receipt["requestedFfnChunks"] = advanced.get("ffnChunks")
            execution_receipt["actualFfnChunks"] = (
                int((ffn_chunk_receipt or {}).get("chunks") or 0)
            )
            execution_receipt["ffnChunks"] = execution_receipt["actualFfnChunks"]
        acceleration_mode = str(advanced.get("accelerationMode") or "full_quality")
        reported_mode = actual_route_id if actual_route_id == LOW_FPS_TIME_REMAP_ROUTE else acceleration_mode
        acceleration_receipt: Dict[str, Any] = {
            "requestedMode": reported_mode,
            "actualMode": native_route_contract["routeId"] if native_route_contract else actual_route_id,
            "algorithmRoute": compiled_route.get("algorithmRoute"),
            "algorithmProfile": compiled_route.get("algorithmProfile"),
            "algorithmRouteFingerprint": compiled_route.get("algorithmRouteFingerprint"),
            "approximate": False,
            "status": "active" if actual_route_id in {LATENT_UPSCALE_TWO_STAGE_ROUTE, KJ_EXPERIMENT_ROUTE, LOW_FPS_TIME_REMAP_ROUTE} else "pending_preflight",
            "stats": {"completedSamplingSteps": 0},
            "fallbackReason": kernel_fallback_reason,
            "executionRouteContract": native_route_contract,
            "projectOptimizationIsolation": (
                native_route_contract.get("projectOptimizations") if native_route_contract else None
            ),
        }
        runtime_stages["acceleration"] = acceleration_receipt
        plan.setdefault("performance", {})["acceleration"] = acceleration_receipt
        teacache_b2_receipt = None
        b2_profiles = {
            "teacache_b2_calibration",
            "teacache_b2_candidate",
            "ffn_chunk_b2_calibration",
            "ffn_chunk_b2_candidate",
        }
        if diagnostic_profile in b2_profiles:
            b2_mode = "calibration" if diagnostic_profile in {"teacache_b2_calibration", "ffn_chunk_b2_calibration"} else "candidate"
            b2_calibration = advanced.get("teacacheB2Calibration") if isinstance(advanced.get("teacacheB2Calibration"), dict) else {}
            # B2 wraps the H3 block sequence after the upstream FFN wrappers
            # are attached.  Both preserve the current timestep and final AV
            # projection; neither is visible to normal product requests.
            shifted, teacache_b2_receipt = apply_h3_teacache_b2_diagnostic(
                shifted,
                mode=b2_mode,
                steps=int(plan["execution"].get("steps") or 20),
                calibration=b2_calibration,
            )
            teacache_b2_receipt["combinedWithFfnChunk"] = ffn_chunk_receipt is not None
            teacache_b2_receipt["patchOrder"] = "after_kj_h3_sage_after_ffn_chunk_before_basic_guider"
            runtime_stages["teacache_b2"] = teacache_b2_receipt
            plan.setdefault("performance", {})["teacacheB2"] = teacache_b2_receipt
        seed = int(plan["execution"].get("seed") or 0)
        progress(58, "sampling joint video/audio latent")
        sample_progress_floor = {"value": 58}
        residency_stabilization: Dict[str, Any] = {
            "strategy": "native_dynamicvram_sampler_owned",
            "status": "pending_first_native_step",
            "samplerOwnsLoad": True,
        }
        # Empty for observability only. Product code never writes a VBAR limit.
        task_watermark_limits: Dict[str, Dict[str, Any]] = {}

        def refresh_acceleration_receipt(completed_steps: int, requested_steps: int) -> Dict[str, Any]:
            """Refresh full-quality sampling progress without route adapters."""

            if acceleration_receipt.get("status") != "active":
                return acceleration_receipt
            stats = {"completedSamplingSteps": int(completed_steps), "requestedSamplingSteps": int(requested_steps)}
            acceleration_receipt["stats"] = stats
            runtime_stages["acceleration"] = acceleration_receipt
            return acceleration_receipt

        def sample_progress(value: int, message: str) -> None:
            # Denoiser/block telemetry and the sampler callback arrive from
            # different layers.  Keep the user-facing task progress monotonic
            # when a callback for step 0 arrives after block telemetry.
            sample_progress_floor["value"] = max(sample_progress_floor["value"], int(value))
            progress(sample_progress_floor["value"], message)

        heartbeat_hook = getattr(progress, "heartbeat", None)
        if callable(heartbeat_hook):
            sample_progress.heartbeat = heartbeat_hook  # type: ignore[attr-defined]

        def report_phase(progress_receipt: Dict[str, Any]) -> None:
            if not callable(heartbeat_hook):
                return
            stage = str(progress_receipt.get("currentStage") or "sampling")
            step = int(progress_receipt.get("phaseStep") or 0)
            total = int(progress_receipt.get("phaseTotal") or 0)
            heartbeat_hook(
                sample_progress_floor["value"],
                f"{stage} {step}/{total}",
                {"kind": "phase_progress", "progressReceipt": dict(progress_receipt)},
            )

        def callback(step: int, _x0: Any, _x: Any, total_steps: int) -> None:
            completed_at = time.perf_counter()
            cancellation = runtime_stages.setdefault("cancellation", {})
            cancellation["lastCompletedSamplingStep"] = max(
                int(cancellation.get("lastCompletedSamplingStep", 0) or 0), int(step) + 1
            )
            if cancel_event.is_set():
                cancellation.update({
                    "requested": True,
                    "status": "cancel_requested_at_sampler_callback",
                    "observedAtMonotonic": time.monotonic(),
                    "totalSamplingSteps": int(total_steps),
                })
                raise H3Cancelled("cancelled during H3 sampling")
            if deadline is not None and time.monotonic() > deadline:
                raise H3RuntimeError(f"H3 execution timeout exceeded configured budget ({timeout_seconds:g} seconds) during sampling")
            if int(step) == 0 and residency_stabilization.get("status") == "pending_first_native_step":
                # This runs after the official sampler has completed the first
                # real forward.  AIMDO's native ``prioritize`` call in the
                # sampler-owned load path already reset the watermark before
                # this point; capture native state plus the module/page manifest.
                # This observation never calls either AIMDO watermark control.
                residency_stabilization.clear()
                residency_stabilization.update(
                    _observe_native_dynamic_vram_after_first_step(
                        torch, shifted, residency_receipt, task_watermark_limits
                    )
                )
                runtime_stages["dynamic_vram_residency_stabilization"] = dict(residency_stabilization)
                _record_first_native_step_residency(
                    residency_receipt,
                    residency_stabilization,
                    runtime_stages.get("native_dynamic_vram_prefetch"),
                )
                runtime_stages["dynamic_vram_residency_stabilization"] = dict(residency_stabilization)
            prefetch_step = _prefetch_trace_step_summary(
                runtime_stages.get("native_dynamic_vram_prefetch")
            )
            refresh_acceleration_receipt(int(step) + 1, int(total_steps))
            # This small snapshot is emitted once per completed sampling step.
            # It intentionally contains no module/page scan and no CUDA call.
            residency_receipt["prefetchStep"] = dict(prefetch_step)
            quantization_snapshot = getattr(restore_quantization_trace, "snapshot", lambda: {})()
            forward_summaries = (runtime_stages.get("denoiser_forward") or {}).get("forwardSummaries") or []
            previous_forward_count = int(step_clock.get("forwardSummaryCount", 0) or 0)
            new_forward_summaries = [item for item in forward_summaries[previous_forward_count:] if isinstance(item, dict)]
            executed_blocks = sum(int(item.get("blockCount", 0) or 0) for item in new_forward_summaries)
            step_clock["forwardSummaryCount"] = len(forward_summaries)
            emit_runtime_telemetry("sampling_step", _sampling_step_telemetry_snapshot(
                step=int(step) + 1,
                total_steps=int(total_steps),
                step_elapsed_seconds=round(completed_at - step_clock["lastCompletedAt"], 6),
                prefetch_trace=runtime_stages.get("native_dynamic_vram_prefetch"),
                native_dynamic_vram=_native_dynamic_vram_receipt(shifted, task_watermark_limits),
                resources=_resource_cooperation_receipt(torch),
                sage=sage_runtime,
                ffn=ffn_runtime,
                quantization=quantization_snapshot,
                quantization_previous=step_clock.get("quantizationCumulative"),
                denoiser_forward=runtime_stages.get("denoiser_forward"),
                acceleration=acceleration_receipt,
            ))
            step_clock["quantizationCumulative"] = {
                "executedInt8LinearCalls": quantization_snapshot.get("executedInt8LinearCalls"),
                "fallbackEagerCalls": quantization_snapshot.get("fallbackEagerCalls"),
            }
            step_clock["lastCompletedAt"] = completed_at
            sample_progress(59 + int(31 * (step + 1) / max(1, total_steps)), f"sampling step {step + 1}/{total_steps}")

        restore_prefetch_trace = _install_native_prefetch_trace(runtime_stages)
        restore_denoiser_trace = _install_denoiser_trace(shifted, runtime_stages, sample_progress, cancel_event)
        advanced = compiled.get("advanced") or {}
        diagnostic_profile = str(
            advanced.get("executionProfile")
            or advanced.get("diagnosticExecutionProfile")
            or "production_kj_sage_ffn"
        )
        packed_tokens = (runtime_stages.get("packed_sequence") or {}).get("sequenceTokens")
        if native_route_contract:
            primary_load_policy = {
                "status": "native_managed_no_project_policy",
                "executionRoute": native_route_contract["routeId"],
            }
            residency_receipt = {
                "status": "native_managed_no_project_policy",
                "executionRoute": native_route_contract["routeId"],
                "residencyBudget": {},
            }
            residency_stabilization = {
                "strategy": "native_managed_no_project_policy",
                "status": "not_injected",
            }
            progress(59, "native ComfyUI model lifecycle preparing primary model")
        else:
            primary_load_policy = _primary_model_load_policy(
                torch,
                packed_tokens,
                advanced.get("dynamicVramTokenThresholdTokens", 65536),
            )
            residency_receipt = _prepare_sequence_residency(
                torch,
                model_management,
                shifted,
                packed_tokens,
                advanced.get("dynamicVramTokenThresholdTokens", 65536),
            )
            residency_stabilization["preflight"] = dict(residency_receipt.get("residencyBudget") or {})
            # Let the sampler call Comfy's native model-management path directly.
            # There is deliberately no load monkeypatch, force-full override, or
            # pre-sampler quantization warmup here: the workflow baseline's
            # DynamicVRAM/AIMDO lifecycle owns model residency and dispatch order.
            progress(59, "native DynamicVRAM model management preparing REF2VA")
            native_admission = native_managed_receipt()
            runtime_stages["memoryAdmission"] = {
                "state": NATIVE_MANAGED_DECISION,
                "lockedBeforeDenoise": False,
                "reason": native_admission["reason"],
                "historyLookup": "disabled",
                "decisionSource": NATIVE_DYNAMIC_VRAM_SOURCE,
                "resolutionBucket": native_admission["resolutionBucket"],
                "durationBucket": native_admission["durationBucket"],
                "offloadPlan": native_admission["offloadPlan"],
                "executionClass": "C",
                "executionClassScope": "current_execution_signature_only",
            }
            _update_memory_admission_receipt(runtime_stages["executionReceipt"], runtime_stages["memoryAdmission"])
        runtime_stages["primary_model_load_policy"] = primary_load_policy
        runtime_stages["sequence_residency"] = residency_receipt
        restore_quantization_trace = _install_quantization_trace(runtime_stages)
        runtime_stages["quantization_dispatch_policy"] = {
            "status": "native_comfy_kitchen_dispatch",
            "selectionSource": "model_quantization_contract",
            "requestedQuantization": str(plan["execution"].get("modelQuant") or "int8"),
            "packedSequenceTokens": int(packed_tokens),
            "automaticTokenThresholdSwitch": False,
        }
        model_quant = str(plan["execution"].get("modelQuant") or "int8").lower()
        quantization_scope = nullcontext()
        runtime_stages["quantization_dispatch_policy"]["backendPolicy"] = "native_default_for_requested_quantization"
        execution_receipt = runtime_stages.get("executionReceipt")
        if isinstance(execution_receipt, dict):
            execution_receipt.update({
                "requestedQuantization": str(plan["execution"].get("modelQuant") or "int8"),
                "actualQuantizationBackend": "observed_after_native_dispatch",
                "packedSequenceTokens": int(packed_tokens),
                "backendSelectionReason": "native ComfyUI/comfy-kitchen dispatch from the requested model quantization",
                "quantizationFallback": False,
            })
        telemetry_hook = getattr(progress, "telemetry", None)

        def emit_runtime_telemetry(kind: str, payload: Dict[str, Any]) -> None:
            if callable(telemetry_hook):
                telemetry_hook({"kind": kind, "observedAtMonotonic": time.monotonic(), **payload})

        sage_runtime = {
            "selected": sage_receipt.get("selected"),
            "hookApplied": bool(sage_receipt.get("hookApplied")),
            "implementation": sage_receipt.get("hookImplementation"),
            "patchedBlocks": (sage_receipt.get("hookReceipt") or {}).get("patchedBlocks"),
        }
        ffn_runtime = {
            "applied": bool((ffn_chunk_receipt or {}).get("applied")),
            "chunks": (ffn_chunk_receipt or {}).get("chunks"),
            "minTokens": (ffn_chunk_receipt or {}).get("minTokens"),
        }
        emit_runtime_telemetry("sampling_preflight", {
            "sage": sage_runtime,
            "ffn": ffn_runtime,
            "quantization": getattr(restore_quantization_trace, "snapshot", lambda: {})(),
            "dynamicVram": residency_receipt,
            "dynamicVramResidencyStabilization": dict(residency_stabilization),
            "dynamicVramNative": _native_dynamic_vram_receipt(shifted, task_watermark_limits),
            "resources": _resource_cooperation_receipt(torch),
        })
        def prepare_stage2_video(stage1_video_latent: Any, progress_step: Callable[[int, int], Any]) -> tuple[Any, Dict[str, Any]]:
            stage_started = time.perf_counter()
            latent_mode = _latent_upscale_mode()
            compiled_mode = str(
                (((compiled.get("canvas") or {}).get("latentUpscale") or {}).get("selectedMode"))
                or "latent"
            )
            if compiled_mode != latent_mode:
                raise H3RuntimeError(
                    f"compiled latent upscale mode mismatch: {compiled_mode} != {latent_mode}"
                )
            if latent_mode == "latent":
                try:
                    from latent_upscaler_adapter import split_av_latent
                    stage1_audio = getattr(prepare_stage2_video, "_stage1_audio", None)
                    if stage1_audio is None:
                        raise H3RuntimeError("stage2 audio handoff missing")
                except Exception as exc:
                    raise H3RuntimeError(f"latent upscaler audio boundary failed: {type(exc).__name__}: {exc}") from exc
                encoded, receipt = _latent_upscale_stage2(
                    stage1_video_latent,
                    stage1_audio,
                    int(compiled["canvas"]["stage2TargetLatent"]["height"]),
                    int(compiled["canvas"]["stage2TargetLatent"]["width"]),
                    progress_step,
                )
                receipt["elapsedSeconds"] = round(time.perf_counter() - stage_started, 6)
                return encoded, receipt
            load_receipt = self.runtime.load_video_vae()
            vae_objects = self.runtime.objects()
            stage_video_vae = _H3VideoVAEProxy(
                vae_objects["videoVae"], vae_objects.get("videoVaePatcher"), model_management,
                torch_module=torch, prefer_tiled=True,
            )
            resize_mode = _frame_resize_vae_mode()
            resize = _frame_resize_vae_handler(resize_mode)
            try:
                encoded, receipt = _frame_resize_vae_reencode(
                    stage1_video_latent, stage_video_vae, resize, int(target_width), int(target_height), progress_step,
                    route_mode=resize_mode, torch_module=torch,
                )
                receipt.update({
                    "videoVaeLoad": load_receipt,
                    "elapsedSeconds": round(time.perf_counter() - stage_started, 6),
                })
                return encoded, receipt
            finally:
                stage_video_vae = None
                vae_objects = None
                runtime_stages["stage1_to_stage2_video_vae_unload"] = self.runtime.unload_video_vae()

        sampling_started = time.perf_counter()
        step_clock = {"lastCompletedAt": sampling_started}
        official_aimdo_cleanup: Dict[str, Any] = {}
        try:
            with torch.inference_mode():
                with quantization_scope:
                    samples, sampler_contract = _sample_with_official_h3_contract(
                        modules["custom_sampler"], modules["sample"], model_management,
                        shifted, positive, latent, seed, steps, callback,
                        stage1_steps=int((compiled.get("advanced") or {}).get("stage1Steps") or steps),
                        stage2_steps=int((compiled.get("advanced") or {}).get("stage2Steps") or 1),
                        stage2_denoise=float((compiled.get("advanced") or {}).get("stage2Denoise") or 0.30),
                        target_latent_hw=(
                            int(compiled["canvas"]["stage2TargetLatent"]["height"]),
                            int(compiled["canvas"]["stage2TargetLatent"]["width"]),
                        ),
                        source_latent_hw=(
                            int(compiled["canvas"]["stage1SourceLatent"]["height"]),
                            int(compiled["canvas"]["stage1SourceLatent"]["width"]),
                        ),
                        prepare_stage2_video=prepare_stage2_video,
                        stage2_positive=stage2_positive,
                        phase_progress=report_phase,
                        stage1_ffn_chunks=1,
                        stage2_ffn_chunks=int(requested_ffn_chunks or 2),
                    )
        finally:
            restore_quantization_trace()
            restore_denoiser_trace()
            restore_prefetch_trace()
            official_aimdo_cleanup = _finalize_official_aimdo_sampling_lifecycle(model_management)
            runtime_stages["officialAimdoSamplingCleanup"] = official_aimdo_cleanup
            emit_runtime_telemetry("sampling_lifecycle_cleanup", official_aimdo_cleanup)
            # A cancelled diagnostic is still valuable calibration evidence.
            # Capture its actual real/reuse decisions before the exception
            # escapes to TaskStore; the product path has no TeaCache state.
            if teacache_b2_receipt is not None:
                teacache_b2_receipt.update(h3_teacache_b2_runtime_receipt(shifted))
            completed_sampling_steps = int((runtime_stages.get("cancellation", {}) or {}).get("lastCompletedSamplingStep", 0) or 0)
            refresh_acceleration_receipt(completed_sampling_steps, steps)
            emit_runtime_telemetry("acceleration_final", dict(acceleration_receipt))
        runtime_stages["sampler_prepare"] = _sampler_prepare_receipt(sampler_contract, latent)
        if ffn_chunk_receipt is not None:
            ffn_chunk_receipt.update({
                "stage1FfnChunks": sampler_contract["stage1FfnChunks"],
                "stage2FfnChunks": sampler_contract["stage2FfnChunks"],
                "stage1FfnApplied": sampler_contract["stage1FfnApplied"],
                "stage2FfnApplied": sampler_contract["stage2FfnApplied"],
                "stage2Receipt": sampler_contract.get("stage2FfnReceipt"),
                "verifiedProductRoute": bool(sampler_contract["stage2FfnApplied"]),
            })
        runtime_stages["twoStageSampling"] = {
            "progressReceipt": sampler_contract.get("progressReceipt"),
            "lastMeaningfulStage": (sampler_contract.get("progressReceipt") or {}).get("lastMeaningfulStage"),
            "stage1Steps": sampler_contract["stage1Steps"],
            "stage2Steps": sampler_contract["stage2Steps"],
            "stage2Denoise": sampler_contract["stage2Denoise"],
            "stage1FfnChunks": sampler_contract["stage1FfnChunks"],
            "stage2FfnChunks": sampler_contract["stage2FfnChunks"],
            "stage1FfnApplied": sampler_contract["stage1FfnApplied"],
            "stage2FfnApplied": sampler_contract["stage2FfnApplied"],
            "executionKind": sampler_contract["executionKind"],
            "samplerCalls": sampler_contract["samplerCalls"],
            "stage1UpscalerSource": sampler_contract.get("stage1UpscalerSource"),
            "stage2DecodeSource": sampler_contract.get("stage2DecodeSource"),
            "finalAudioSource": sampler_contract.get("finalAudioSource"),
            "stage2AudioSampled": sampler_contract.get("stage2AudioSampled"),
            "stage2AudioDiscarded": sampler_contract.get("stage2AudioDiscarded"),
            "stage1AudioLatentShape": sampler_contract.get("stage1AudioLatentShape"),
            "stage2AudioLatentShape": sampler_contract.get("stage2AudioLatentShape"),
            "finalAudioLatentShape": sampler_contract.get("finalAudioLatentShape"),
            "audioComparison": sampler_contract.get("audioComparison"),
            "tensorStats": sampler_contract.get("tensorStats"),
            "frameResizeVaeRoundTrip": sampler_contract["frameResizeVaeRoundTrip"],
            "latentShape": sampler_contract.get("latentShape"),
            "stage1VideoLatentShape": sampler_contract.get("stage1VideoLatentShape"),
            "stage2VideoLatentShape": sampler_contract.get("stage2VideoLatentShape"),
            "sourceLatentHW": sampler_contract.get("sourceLatentHW"),
            "targetLatentHW": sampler_contract.get("targetLatentHW"),
            "stage1SourceCanvas": dict(compiled["canvas"]["stage1SourceCanvas"]),
            "stage2TargetCanvas": dict(compiled["canvas"]["stage2TargetCanvas"]),
            "finalOutputCanvas": dict(compiled["canvas"]["finalOutputCanvas"]),
            "stage1Canvas": dict(compiled["canvas"]["stage1Canvas"]),
            "stage2InternalCanvas": dict(compiled["canvas"]["stage2InternalCanvas"]),
            "stage1AreaRatio": float(compiled["canvas"]["stage1AreaRatio"]),
            "stage2AreaScale": float(compiled["canvas"]["stage2AreaScale"]),
            "stage2LinearScale": dict(compiled["canvas"]["stage2LinearScale"]),
            "canvasMultiple": int(compiled["canvas"]["canvasMultiple"]),
            "internalScale": float(compiled["canvas"]["internalScale"]),
            "frameResize": dict(compiled["canvas"]["frameResize"]),
            "finalDownsample": dict(compiled["canvas"]["finalDownsample"]),
            "directStage2Output": bool(compiled["canvas"]["directStage2Output"]),
            "fallbackUsed": False,
        }
        if sampler_contract.get("runtimeStages") is not None:
            runtime_stages["renoiseTensorStats"] = sampler_contract["runtimeStages"]
        execution_receipt = runtime_stages.get("executionReceipt")
        if isinstance(execution_receipt, dict):
            execution_receipt.update(runtime_stages["twoStageSampling"])
        # ``sample.sample`` can return while CUDA work is still queued.  Do
        # not unload the denoiser or load either VAE until the real kernels
        # have completed; otherwise the next stage competes with in-flight
        # REF2VA work and can produce an invalid OOM or incomplete latent.
        cuda_sync_started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime_stages["primary_sampling"] = {
            "elapsedSeconds": round(time.perf_counter() - sampling_started, 6),
            "cudaSynchronizeSeconds": round(time.perf_counter() - cuda_sync_started, 6),
            "steps": steps,
        }
        if low_fps_remap_state is not None:
            if int(low_fps_remap_state.get("wrapperCalls") or 0) < 1 or low_fps_remap_state.get("status") != "active":
                raise H3RuntimeError("low-fps time-remap wrapper was not executed by the denoiser")
            timing = dict(compiled.get("timing") or {})
            required_actual = {
                "requestedFps": int(timing["requestedFps"]),
                "actualModelFps": int(timing["actualModelFps"]),
                "temporalDensityFps": int(timing["temporalDensityFps"]),
                "modelFrameCount": int(timing["modelFrameCount"]),
                "videoLatentT": int(timing["videoLatentT"]),
                "audioLatentT": int(timing["audioLatentT"]),
                "timeScale": float(timing["timeScale"]),
                "variantId": str(timing.get("variantId") or "continuous_direct"),
                "positionMapping": str(timing.get("positionMapping") or "continuous_time_scale"),
                "decodeMode": str(timing.get("decodeMode") or "direct"),
                "bridgedModelFps": timing.get("bridgedModelFps"),
                "bridgedModelFrameCount": timing.get("bridgedModelFrameCount"),
                "bridgedVideoLatentT": timing.get("bridgedVideoLatentT"),
                "fallback": False,
            }
            mismatches = {
                key: {"expected": value, "observed": low_fps_remap_state.get(key)}
                for key, value in required_actual.items()
                if low_fps_remap_state.get(key) != value
            }
            if mismatches:
                raise H3RuntimeError(f"low-fps worker actual receipt mismatch: {mismatches}")
            runtime_stages["low_fps_time_remap"] = dict(low_fps_remap_state)
            execution_receipt = runtime_stages.get("executionReceipt")
            if isinstance(execution_receipt, dict):
                execution_receipt.update(required_actual)
                execution_receipt["routeId"] = LOW_FPS_TIME_REMAP_ROUTE
                execution_receipt["timeRemapReceipt"] = dict(low_fps_remap_state)
        record_resources("primary_sampling_complete")
        primary_model = getattr(shifted, "model", None)
        primary_model_size = int(getattr(shifted, "model_size", lambda: 0)() or 0)
        primary_loaded_size = int(getattr(shifted, "loaded_size", lambda: getattr(primary_model, "model_loaded_weight_memory", 0))() or 0)
        runtime_stages["primary_model_device_load"] = {
            "observationPoint": "after native sampler model-management call",
            "modelType": type(shifted).__name__,
            "modelDevice": _device_text(getattr(primary_model, "device", None)),
            "loadDevice": _device_text(getattr(shifted, "load_device", None)),
            "offloadDevice": _device_text(getattr(shifted, "offload_device", None)),
            "isDynamic": bool(getattr(shifted, "is_dynamic", lambda: False)()),
            "loadedWeightBytes": int(getattr(primary_model, "model_loaded_weight_memory", primary_loaded_size) or 0),
            "offloadedWeightBytes": max(0, primary_model_size - primary_loaded_size) if primary_model_size else None,
            "lowvramPatchCount": int(getattr(primary_model, "lowvram_patch_counter", 0) or 0),
            "modelLowvram": bool(getattr(primary_model, "model_lowvram", False)),
            "forceFullLoadRequested": False,
            "dynamicVramPolicy": primary_load_policy,
            "dynamicVramResidencyStabilization": dict(residency_stabilization),
            "dynamicVramNative": _native_dynamic_vram_receipt(shifted, task_watermark_limits),
            "modelStructure": _quantized_linear_structure_receipt(shifted),
        }
        # Sampling has consumed the conditioning graph.  Release its large
        # tensor containers before the VAE lifecycle begins; the returned
        # latent sample is the only sampling result needed for decode/export.
        # This is a memory-lifetime optimization and does not alter the
        # prompt, reference payload, seed, steps, or latent values already
        # produced by the sampler.
        positive = None
        latent = None
        output = None
        prepared = None
        reference_receipt = None
        packed_sequence = None
        images = None
        gc.collect()
        # The runtime snapshot and the sigma-shift wrapper both retain the
        # primary ModelPatcher.  Release those Python references only after
        # the queued denoiser work has synchronized; unload_all_models()
        # cannot reclaim a live device object held by this frame.  This is
        # required for native 20-step output to enter VAE decode without an
        # avoidable OOM.
        primary_reference_release_started = time.perf_counter()
        # The restoration call has completed.  Drop the request-scoped trace
        # restorers before asking model-management to reclaim the primary
        # weights.
        restore_denoiser_trace = None
        restore_quantization_trace = None
        objects = None
        shifted = None
        gc.collect()
        runtime_stages["primary_python_reference_release"] = {
            "released": True,
            "elapsedSeconds": round(time.perf_counter() - primary_reference_release_started, 6),
        }
        # ModelPatcher keeps a separate entry in model_management's loaded
        # registry.  Dropping the local variable alone does not release that
        # entry, so release only this primary patcher (and its native clones)
        # before VAE decode.  This changes lifecycle/memory state only; it
        # never unloads unrelated components or changes samples.
        primary_release_started = time.perf_counter()
        primary_memory_release: Dict[str, Any] = {
            "method": "targeted_native_model_patcher_release",
            "unloadAllModelsCalled": False,
        }
        loaded_registry = getattr(model_management, "current_loaded_models", None)
        if loaded_registry is not None:
            try:
                primary_memory_release["loadedModelsBefore"] = len(loaded_registry)
                primary_memory_release["loadedModelTypesBefore"] = [
                    type(getattr(item, "model", None)).__name__ for item in loaded_registry
                ]
            except Exception:
                pass
        if deadline is not None and time.monotonic() > deadline:
            raise H3RuntimeError(f"H3 execution timeout exceeded configured budget ({timeout_seconds:g} seconds) before decode")
        primary_unload = self.runtime.unload_primary()
        runtime_stages["primary_model_unload"] = primary_unload
        primary_memory_release["runtimeRelease"] = primary_unload
        if loaded_registry is not None:
            try:
                primary_memory_release["loadedModelsAfterRuntimeUnload"] = len(loaded_registry)
            except Exception:
                pass
        shifted = None
        gc.collect()
        if torch.cuda.is_available():
            primary_memory_release["cudaAllocatedAfterRuntimeUnload"] = int(torch.cuda.memory_allocated())
            primary_memory_release["cudaReservedAfterRuntimeUnload"] = int(torch.cuda.memory_reserved())
            # Model-management normally performs this flush while unloading,
            # but the direct embedded path can finish an asynchronous device
            # transfer after that call.  Synchronize and flush once more at
            # the lifecycle boundary before constructing a VAE.
            try:
                torch.cuda.synchronize()
                soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
                if callable(soft_empty_cache):
                    try:
                        soft_empty_cache(force=True)
                    except TypeError:
                        soft_empty_cache()
                torch.cuda.empty_cache()
                ipc_collect = getattr(torch.cuda, "ipc_collect", None)
                if callable(ipc_collect):
                    ipc_collect()
                primary_memory_release["cacheFlushCalled"] = True
            except Exception as exc:
                primary_memory_release["cacheFlushError"] = f"{type(exc).__name__}: {exc}"
            primary_memory_release["cudaAllocatedAfterCacheFlush"] = int(torch.cuda.memory_allocated())
            primary_memory_release["cudaReservedAfterCacheFlush"] = int(torch.cuda.memory_reserved())
        primary_memory_release["elapsedSeconds"] = round(time.perf_counter() - primary_release_started, 6)
        runtime_stages["primary_model_memory_release"] = primary_memory_release
        record_resources("primary_unloaded_before_vae")
        reset_resource_peaks("video_vae_decode")
        registry_count = primary_memory_release.get("loadedModelsAfterRuntimeUnload", "unknown")
        allocated_mb = int(primary_memory_release.get("cudaAllocatedAfterRuntimeUnload", 0) or 0) // (1024 * 1024)
        reserved_mb = int(primary_memory_release.get("cudaReservedAfterCacheFlush", 0) or 0) // (1024 * 1024)
        _report_single_step_progress(
            execution_receipt["progressReceipt"], report_phase, "video_decode", completed=False
        )
        progress(91, f"primary model released; preparing VAE decode ({allocated_mb} MiB allocated, {reserved_mb} MiB reserved, {registry_count} registered)")
        loaded_registry = None
        progress(92, "preparing video/audio latents for VAE decode")
        video_latent, audio_latent = samples.unbind()
        # Make a host copy once.  The validated production path decodes with
        # the official VAE implementation in this process, sequentially.  The
        # isolated worker remains available as a diagnostic tool, but is not
        # selected by the normal product path because its child-process startup
        # can hang after a native DynamicVRAM denoiser run on Windows.
        transfer_started = time.perf_counter()
        video_latent, video_transfer = _stage_latent_for_cpu_vae(video_latent, torch)
        audio_latent, audio_transfer = _stage_latent_for_cpu_vae(audio_latent, torch)
        transfer_sync_started = time.perf_counter()
        if (video_transfer.get("queued") or audio_transfer.get("queued")) and torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime_stages["sampling_latent_cpu_transfer"] = {
            "elapsedSeconds": round(time.perf_counter() - transfer_started, 6),
            "cudaSynchronizeSeconds": round(time.perf_counter() - transfer_sync_started, 6),
            "video": video_transfer,
            "audio": audio_transfer,
            "strategy": "pinned_nonblocking_then_single_sync",
        }
        latent_dual_decode_bundle = None
        latent_dual_decode_persistence = None
        low_fps_bridge_bundle = None
        diagnostic_config = compiled.get("diagnostics")
        if diagnostic_config is not None:
            from latent_dual_decode_diagnostic import persist_latent_dual_decode, prepare_latent_dual_decode
            if diagnostic_config.get("latentDualDecode") is True:
                if os.environ.get("H3_ENABLE_LATENT_DUAL_DECODE") != "1":
                    raise H3RuntimeError("latent dual decode diagnostic is disabled in the worker")
                latent_dual_decode_bundle = prepare_latent_dual_decode(
                    video_latent,
                    audio_latent,
                    source_model_fps=int(diagnostic_config["sourceModelFps"]),
                    bridged_model_fps=int(diagnostic_config["bridgedModelFps"]),
                    bridged_video_latent_t=int(diagnostic_config["bridgedVideoLatentT"]),
                    export_fps=int(diagnostic_config["exportFps"]),
                    export_frame_count=int(diagnostic_config["exportFrameCount"]),
                )
                diagnostic_dir = (ROOT / "output" / plan["taskId"] / "latent_dual_decode").resolve()
                diagnostic_dir.relative_to((ROOT / "output").resolve())
                persist_started = time.perf_counter()
                latent_dual_decode_persistence = persist_latent_dual_decode(
                    latent_dual_decode_bundle,
                    diagnostic_dir,
                    path_base=ROOT,
                )
                latent_dual_decode_persistence["elapsedSeconds"] = round(time.perf_counter() - persist_started, 6)
                runtime_stages["latent_dual_decode_persistence"] = latent_dual_decode_persistence
            elif diagnostic_config.get("lowFpsVariant"):
                if os.environ.get("H3_ENABLE_LOW_FPS_VARIANTS") != "1":
                    raise H3RuntimeError("low-FPS diagnostic variants are disabled in the worker")
                if diagnostic_config.get("decodeMode") == "latent_bridge_to_official_24fps_density":
                    low_fps_bridge_bundle = prepare_latent_dual_decode(
                        video_latent,
                        audio_latent,
                        source_model_fps=int(diagnostic_config["sourceModelFps"]),
                        bridged_model_fps=int(diagnostic_config["bridgedModelFps"]),
                        bridged_video_latent_t=int(diagnostic_config["bridgedVideoLatentT"]),
                        export_fps=int(diagnostic_config["exportFps"]),
                        export_frame_count=int(diagnostic_config["exportFrameCount"]),
                    )
                    runtime_stages["low_fps_latent_bridge"] = dict(low_fps_bridge_bundle["receipt"])
            else:
                raise H3RuntimeError("unsupported worker diagnostic identity")
        samples = None
        runtime_stages["isolated_vae_worker"] = {
            "status": "not_selected",
            "reason": "production uses validated in-process official video/audio VAE sequential decode",
            "diagnosticEntryPoint": "embedded_h3_runtime.vae_decode_worker",
        }
        runtime_stages["decode_video_vae_load"] = self.runtime.load_video_vae()
        decode_objects = self.runtime.objects()
        video_vae = _H3VideoVAEProxy(
            decode_objects["videoVae"], decode_objects.get("videoVaePatcher"), model_management,
            torch_module=torch, prefer_tiled=torch.cuda.is_available(),
        )
        decode_started = time.perf_counter()
        video_decode_preflight = _vae_decode_device_preflight(torch, video_vae, video_latent)
        runtime_stages["decode_video_vae_preflight"] = video_decode_preflight
        video_decode_placement = video_decode_preflight.get("selected", "gpu")
        if video_decode_placement == "cpu":
            progress(93, "selecting official CPU VAE decode from live memory estimate")
            video_vae = None
            decode_objects = None
            gc.collect()
            runtime_stages["decode_video_vae_unload_before_cpu"] = self.runtime.unload_video_vae()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            runtime_stages["decode_video_vae_load_cpu"] = self.runtime.load_video_vae(device=torch.device("cpu"))
            decode_objects = self.runtime.objects()
            video_vae = _H3VideoVAEProxy(
                decode_objects["videoVae"], decode_objects.get("videoVaePatcher"), model_management,
                torch_module=torch, prefer_tiled=torch.cuda.is_available(),
            )
        # PyTorch 2.11/cu128 can select the cuDNN MHA graph for the official
        # H3 video VAE and return ``mha_graph.execute(...).is_good() == false``
        # on RTX 5090 for its tiled attention shape.  Keep the failure local
        # to VAE decode: disable only cuDNN SDP for the duration of the two
        # official VAE calls, then restore the process setting.  The H3
        # denoiser's Sage/SDPA choice is not changed by this boundary adapter.
        decode_attention_receipt: Dict[str, Any] = {
            "scope": "video_audio_vae_decode",
            "backend": "pytorch_sdpa_without_cudnn_mha",
            "cudnnSdpAvailable": False,
            "changed": False,
        }
        restore_cudnn_sdp = None
        try:
            progress(93, "decoding video VAE output")
            cudnn_sdp_enabled = getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)
            enable_cudnn_sdp = getattr(torch.backends.cuda, "enable_cudnn_sdp", None)
            if torch.cuda.is_available() and callable(cudnn_sdp_enabled) and callable(enable_cudnn_sdp):
                decode_attention_receipt["cudnnSdpAvailable"] = True
                previous_cudnn_sdp = bool(cudnn_sdp_enabled())
                decode_attention_receipt["previousEnabled"] = previous_cudnn_sdp
                if previous_cudnn_sdp:
                    enable_cudnn_sdp(False)
                    decode_attention_receipt["changed"] = True
                    restore_cudnn_sdp = lambda: enable_cudnn_sdp(previous_cudnn_sdp)
            runtime_stages["decode_attention_backend"] = decode_attention_receipt
            video_decode_started = time.perf_counter()
            video_decode_fallback = None
            try:
                decode_video_latent = (
                    low_fps_bridge_bundle["bridgedVideoLatent"]
                    if low_fps_bridge_bundle is not None else video_latent
                )
                video = video_vae.decode(decode_video_latent)
            except Exception as exc:
                if not _is_cuda_out_of_memory(torch, exc):
                    raise
                if latent_dual_decode_bundle is not None or low_fps_bridge_bundle is not None:
                    raise H3RuntimeError("latent dual decode direct VAE OOM; fallback is forbidden") from exc
                video_decode_fallback = {
                    "requested": "gpu",
                    "selected": "cpu",
                    "reason": "cuda_oom",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
                runtime_stages["decode_video_vae_fallback"] = video_decode_fallback
                progress(93, "GPU memory is insufficient for VAE decode; continuing with official CPU decode")
                # Drop the failed GPU patcher before constructing the CPU
                # VAE.  The CPU fallback uses the same official weights and
                # VAE code; it changes placement only, never the target.
                video_vae = None
                decode_objects = None
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                runtime_stages["decode_video_vae_unload_after_oom"] = self.runtime.unload_video_vae()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                runtime_stages["decode_video_vae_load_cpu"] = self.runtime.load_video_vae(device=torch.device("cpu"))
                decode_objects = self.runtime.objects()
                video_vae = _H3VideoVAEProxy(
                    decode_objects["videoVae"], decode_objects.get("videoVaePatcher"), model_management,
                    torch_module=torch, prefer_tiled=torch.cuda.is_available(),
                )
                try:
                    video = video_vae.decode(video_latent)
                except Exception as fallback_exc:
                    video_decode_fallback["fallbackErrorType"] = type(fallback_exc).__name__
                    video_decode_fallback["fallbackError"] = str(fallback_exc)
                    raise
            video = video.detach().cpu()
            runtime_stages["decode_video_vae"] = {
                "elapsedSeconds": round(time.perf_counter() - video_decode_started, 6),
                "outputShape": _shape(video),
                "outputDevice": _device_text(getattr(video, "device", None)),
                "deviceSelection": video_decode_placement,
                "fallback": None if video_decode_fallback is None else video_decode_fallback["selected"],
            }
            if isinstance(execution_receipt, dict) and isinstance(execution_receipt.get("progressReceipt"), dict):
                elapsed = max(0.000001, float(runtime_stages["decode_video_vae"]["elapsedSeconds"]))
                _report_single_step_progress(
                    execution_receipt["progressReceipt"], report_phase, "video_decode",
                    completed=True, elapsed=elapsed,
                )
            if low_fps_bridge_bundle is not None:
                expected_bridge_frames = int(low_fps_bridge_bundle["receipt"]["bridgedDecodedFrameCount"])
                observed_bridge_frames = int(video.shape[1] if video.ndim == 5 else video.shape[0])
                if observed_bridge_frames != expected_bridge_frames:
                    raise H3RuntimeError(
                        f"low-fps latent bridge frame mismatch: {observed_bridge_frames} != {expected_bridge_frames}"
                    )
                runtime_stages["low_fps_latent_bridge_decode"] = {
                    "outputShape": _shape(video),
                    "decodedFrameCount": observed_bridge_frames,
                    "fallback": False,
                }
            bridged_video = None
            if latent_dual_decode_bundle is not None:
                bridged_decode_started = time.perf_counter()
                try:
                    bridged_video = video_vae.decode(latent_dual_decode_bundle["bridgedVideoLatent"])
                except Exception as exc:
                    if _is_cuda_out_of_memory(torch, exc):
                        raise H3RuntimeError("latent dual decode bridged VAE OOM; fallback is forbidden") from exc
                    raise
                bridged_video = bridged_video.detach().cpu()
                expected_bridged_frames = int(latent_dual_decode_bundle["receipt"]["bridgedDecodedFrameCount"])
                bridged_frame_count = int(bridged_video.shape[1] if bridged_video.ndim == 5 else bridged_video.shape[0])
                if bridged_frame_count != expected_bridged_frames:
                    raise H3RuntimeError(
                        f"latent dual decode bridged frame mismatch: {bridged_frame_count} != {expected_bridged_frames}"
                    )
                runtime_stages["latent_dual_decode_bridged_video_vae"] = {
                    "elapsedSeconds": round(time.perf_counter() - bridged_decode_started, 6),
                    "outputShape": _shape(bridged_video),
                    "outputDevice": _device_text(getattr(bridged_video, "device", None)),
                    "fallback": False,
                }
            record_resources("video_vae_decode_complete")
            reset_resource_peaks("audio_vae_decode")
            # Keep the decoded video on CPU before releasing its official
            # patcher.  Load the audio VAE only after this release so the two
            # VAE models never compete for GPU residency.
            video_latent = None
            video_vae = None
            decode_objects = None
            gc.collect()
            runtime_stages["decode_video_vae_unload"] = self.runtime.unload_video_vae()
            _report_single_step_progress(
                execution_receipt["progressReceipt"], report_phase, "audio_decode", completed=False
            )
            runtime_stages["decode_audio_vae_load"] = self.runtime.load_audio_vae()
            decode_objects = self.runtime.objects()
            audio_vae = _H3AudioVAEProxy(decode_objects["audioVae"], decode_objects.get("audioVaePatcher"), model_management, torch_module=torch)
            progress(96, "decoding audio VAE output")
            audio_decode_preflight = _vae_decode_device_preflight(torch, audio_vae, audio_latent)
            runtime_stages["decode_audio_vae_preflight"] = audio_decode_preflight
            audio_decode_placement = audio_decode_preflight.get("selected", "gpu")
            if audio_decode_placement == "cpu":
                progress(96, "selecting official CPU audio VAE decode from live memory estimate")
                audio_vae = None
                decode_objects = None
                gc.collect()
                runtime_stages["decode_audio_vae_unload_before_cpu"] = self.runtime.unload_audio_vae()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                runtime_stages["decode_audio_vae_load_cpu"] = self.runtime.load_audio_vae(device=torch.device("cpu"))
                decode_objects = self.runtime.objects()
                audio_vae = _H3AudioVAEProxy(decode_objects["audioVae"], decode_objects.get("audioVaePatcher"), model_management, torch_module=torch)
            audio_decode_started = time.perf_counter()
            audio_decode_fallback = None
            try:
                audio = audio_vae.decode(audio_latent)
            except Exception as exc:
                if not _is_cuda_out_of_memory(torch, exc):
                    raise
                audio_decode_fallback = {
                    "requested": "gpu",
                    "selected": "cpu",
                    "reason": "cuda_oom",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
                runtime_stages["decode_audio_vae_fallback"] = audio_decode_fallback
                progress(96, "GPU memory is insufficient for audio decode; continuing with official CPU decode")
                audio_vae = None
                decode_objects = None
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                runtime_stages["decode_audio_vae_unload_after_oom"] = self.runtime.unload_audio_vae()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                runtime_stages["decode_audio_vae_load_cpu"] = self.runtime.load_audio_vae(device=torch.device("cpu"))
                decode_objects = self.runtime.objects()
                audio_vae = _H3AudioVAEProxy(
                    decode_objects["audioVae"], decode_objects.get("audioVaePatcher"), model_management,
                    torch_module=torch,
                )
                try:
                    audio = audio_vae.decode(audio_latent)
                except Exception as fallback_exc:
                    audio_decode_fallback["fallbackErrorType"] = type(fallback_exc).__name__
                    audio_decode_fallback["fallbackError"] = str(fallback_exc)
                    raise
            audio = audio.detach().cpu()
            runtime_stages["decode_audio_vae"] = {
                "elapsedSeconds": round(time.perf_counter() - audio_decode_started, 6),
                "outputShape": _shape(audio),
                "outputDevice": _device_text(getattr(audio, "device", None)),
                "deviceSelection": audio_decode_placement,
                "fallback": None if audio_decode_fallback is None else audio_decode_fallback["selected"],
            }
            if isinstance(execution_receipt, dict) and isinstance(execution_receipt.get("progressReceipt"), dict):
                elapsed = max(0.000001, float(runtime_stages["decode_audio_vae"]["elapsedSeconds"]))
                _report_single_step_progress(
                    execution_receipt["progressReceipt"], report_phase, "audio_decode",
                    completed=True, elapsed=elapsed,
                )
            record_resources("audio_vae_decode_complete")
            reset_resource_peaks("mp4_export")
            audio_latent = None
            audio_vae = None
            decode_objects = None
        finally:
            if restore_cudnn_sdp is not None:
                restore_cudnn_sdp()
        execution_receipt = runtime_stages.get("executionReceipt")
        _report_single_step_progress(
            execution_receipt["progressReceipt"], report_phase, "mux", completed=False
        )
        progress(98, "packing MP4")
        mux_started = time.perf_counter()
        export_compiled = compiled
        if low_fps_bridge_bundle is not None:
            export_compiled = copy.deepcopy(compiled)
            export_compiled["timing"]["modelFps"] = int(diagnostic_config["bridgedModelFps"])
        output_path = self._export_mp4(video, audio, plan["taskId"], export_compiled, execution_receipt)
        if low_fps_bridge_bundle is not None and isinstance(execution_receipt, dict):
            execution_receipt["modelFps"] = int(compiled["timing"]["modelFps"])
            execution_receipt["decodeTemporalDensityFps"] = int(diagnostic_config["bridgedModelFps"])
        runtime_stages["mp4_mux"] = {"elapsedSeconds": round(time.perf_counter() - mux_started, 6)}
        if isinstance(execution_receipt, dict) and isinstance(execution_receipt.get("progressReceipt"), dict):
            elapsed = max(0.000001, float(runtime_stages["mp4_mux"]["elapsedSeconds"]))
            _report_single_step_progress(
                execution_receipt["progressReceipt"], report_phase, "mux",
                completed=True, elapsed=elapsed,
            )
        diagnostic_outputs = None
        if latent_dual_decode_bundle is not None:
            bridged_compiled = copy.deepcopy(compiled)
            bridged_compiled["timing"]["modelFps"] = int(diagnostic_config["bridgedModelFps"])
            bridged_receipt: Dict[str, Any] = {}
            bridged_mux_started = time.perf_counter()
            bridged_output_path = self._export_mp4(
                bridged_video,
                audio,
                plan["taskId"],
                bridged_compiled,
                bridged_receipt,
                output_filename="h3_result_bridged_diagnostic.mp4",
            )
            runtime_stages["latent_dual_decode_bridged_mux"] = {
                "elapsedSeconds": round(time.perf_counter() - bridged_mux_started, 6),
                "outputPath": str(bridged_output_path.relative_to(ROOT).as_posix()),
            }
            diagnostic_outputs = {
                "directOutputPath": str(output_path.relative_to(ROOT).as_posix()),
                "bridgedOutputPath": str(bridged_output_path.relative_to(ROOT).as_posix()),
                "latentEvidence": latent_dual_decode_persistence,
                "bridgeReceipt": latent_dual_decode_bundle["receipt"],
                "bridgedExportReceipt": bridged_receipt,
                "fallback": False,
            }
            runtime_stages["latent_dual_decode"] = diagnostic_outputs
        runtime_stages["decode_export"] = {"elapsedSeconds": round(time.perf_counter() - decode_started, 6), "outputPath": str(output_path)}
        record_resources("mp4_export_complete")
        if isinstance(execution_receipt, dict):
            if low_fps_bridge_bundle is not None:
                execution_receipt["latentBridgeReceipt"] = dict(low_fps_bridge_bundle["receipt"])
            execution_receipt["stageTimings"] = _stage_timings_receipt(runtime_stages)
            execution_receipt["outputAuthentic"] = True
            execution_receipt["hardwareIdentity"] = _hardware_identity_receipt(torch)
            resource_telemetry = runtime_stages.get("resourceTelemetry") or {}
            primary_device_receipt = runtime_stages.get("primary_model_device_load") or {}
            dynamic_receipt = primary_device_receipt.get("dynamicVramNative") or {}
            execution_receipt["residencyPhases"] = {
                "conditioning": {
                    "referenceVae": "released_before_qwen",
                    "textEncoder": "released_before_primary",
                    "conditioningBuffers": "released_before_primary",
                    "releaseReceipt": runtime_stages.get("conditioning_to_primary_memory_release"),
                },
                "primaryModel": {
                    "loadReceipt": runtime_stages.get("primary_model_load"),
                    "samplingReceipt": primary_device_receipt,
                    "dynamicVram": dynamic_receipt,
                    "releaseBoundary": "unloaded_before_vae",
                },
                "decodeExport": {
                    "videoVae": "loaded_after_primary_unload",
                    "audioVae": "loaded_after_primary_unload",
                    "primaryModelReleased": "primary_unloaded_before_vae",
                },
            }
            execution_receipt["memoryBudget"] = _memory_budget_receipt(resource_telemetry, primary_device_receipt)
            execution_receipt["resourceTelemetry"] = {
                "primaryModelPreSampler": resource_telemetry.get("primary_model_pre_sampler"),
                "primarySamplingComplete": resource_telemetry.get("primary_sampling_complete"),
                "primaryUnloadedBeforeVae": resource_telemetry.get("primary_unloaded_before_vae"),
                "mp4ExportComplete": resource_telemetry.get("mp4_export_complete"),
                "perStep": "see task.samplingTelemetry and worker runtime snapshot",
            }
        progress(100, "MP4 export complete")
        peak_memory = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        result = {
            "status": "completed",
            "realInference": True,
            "outputAuthentic": True,
            "outputPath": str(output_path),
            "seed": seed,
            "steps": steps,
            "modelLoad": model_load,
            "primaryUnload": primary_unload,
            "textEncoderUnload": text_unload,
            "attentionBackend": sage_receipt.get("selected", "torch_sdpa"),
            "sage": sage_receipt,
            "peakMemoryBytes": peak_memory,
            "referenceCache": locals().get("reference_cache_receipts", []),
            "referenceDecode": locals().get("reference_decode_receipts", []),
            "runtimeStages": runtime_stages,
            "executionReceipt": dict(runtime_stages.get("executionReceipt") or {}),
        }
        if diagnostic_outputs is not None:
            result["diagnosticOutputs"] = diagnostic_outputs
        return result

    def _run_isolated_vae_worker(
        self,
        video_latent: Any,
        audio_latent: Any,
        task_id: str,
        compiled: Dict[str, Any],
        progress: Progress,
        cancel_event: threading.Event,
        deadline: Optional[float],
    ) -> Dict[str, Any]:
        """Decode sampled latents in a clean child process.

        The denoiser and Qwen are already released in the parent, but their
        CUDA allocator history can still fragment the process.  A dedicated
        VAE worker gives the official video/audio VAE a clean CUDA allocator,
        keeps the two VAEs sequential, and leaves cancellation/timeout under
        the product task state machine.  It does not start a server or an
        executor and it references the same read-only root weights.
        """

        worker = ROOT / "app" / "embedded_h3_runtime" / "vae_decode_worker.py"
        if not worker.is_file():
            raise H3RuntimeError(f"isolated VAE worker missing: {worker}")
        work_dir = (ROOT / "temp" / "vae_decode" / task_id).resolve()
        work_dir.relative_to(ROOT.resolve())
        work_dir.mkdir(parents=True, exist_ok=True)
        latent_path = work_dir / "sampled_latents.pt"
        manifest_path = work_dir / "manifest.json"
        result_path = work_dir / "result.json"
        torch = self.runtime.objects()["modules"]["torch"]
        torch.save(
            {
                "video": video_latent.detach().cpu(),
                "audio": audio_latent.detach().cpu(),
            },
            str(latent_path),
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "taskId": task_id,
                    "latentPath": str(latent_path),
                    "resultPath": str(result_path),
                    "compiled": {
                        "canvas": compiled["canvas"],
                        "timing": compiled["timing"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update({
            "TEMP": str(ROOT / "temp"),
            "TMP": str(ROOT / "temp"),
            "TORCH_HOME": str(ROOT / "cache" / "torch"),
            "HF_HOME": str(ROOT / "cache" / "huggingface"),
            "TRANSFORMERS_CACHE": str(ROOT / "cache" / "transformers"),
            "PYTHONPYCACHEPREFIX": str(ROOT / "cache" / "pycache"),
        })
        python_path = [
            str(ROOT / "app"),
            str(ROOT / "runtime" / "python_packages"),
            str(ROOT / "runtime" / "ComfyUI"),
        ]
        env["PYTHONPATH"] = os.pathsep.join(python_path + [env.get("PYTHONPATH", "")])
        started = time.perf_counter()
        process = subprocess.Popen(
            [sys.executable, str(worker), str(manifest_path)],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_receipt = _wait_for_isolated_vae_worker_exit(process, cancel_event, deadline)
        stdout, stderr = process.communicate()
        receipt = {
            "worker": str(worker),
            "returnCode": int(process.returncode or 0),
            "elapsedSeconds": round(time.perf_counter() - started, 6),
            "latentPath": str(latent_path),
            "manifestPath": str(manifest_path),
            "stdoutTail": stdout[-4000:],
            "stderrTail": stderr[-4000:],
            "safeExitWait": wait_receipt,
        }
        if wait_receipt["cancelObservedAtMonotonic"] is not None:
            raise H3Cancelled(
                "cancellation requested during isolated VAE decode; "
                "worker exited naturally without hard termination"
            )
        if wait_receipt["deadlineExceededAtMonotonic"] is not None:
            raise H3RuntimeError(
                "H3 execution timeout exceeded during isolated VAE decode; "
                "worker exited naturally without hard termination"
            )
        if process.returncode != 0 or not result_path.is_file():
            raise H3RuntimeError(f"isolated VAE worker failed: {json.dumps(receipt, ensure_ascii=False)}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        receipt["result"] = result
        if result.get("status") != "completed" or result.get("outputAuthentic") is not True:
            raise H3RuntimeError(f"isolated VAE worker returned non-authentic result: {json.dumps(receipt, ensure_ascii=False)}")
        return receipt

    @staticmethod
    def _protect_audio_pcm_peak(audio_samples: Any, safe_peak: float = 10 ** (-3.0 / 20.0)) -> tuple[Any, Dict[str, Any]]:
        """Apply deterministic whole-waveform attenuation before AAC encoding only when needed."""
        import numpy as np

        samples = np.asarray(audio_samples, dtype=np.float32)
        if not np.isfinite(samples).all():
            raise H3RuntimeError("decoded audio contains NaN or Inf; refusing AAC export")
        peak_before = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak_before > safe_peak:
            gain = safe_peak / peak_before
            samples = samples * gain
            applied = True
        else:
            gain = 1.0
            applied = False
        peak_after = float(np.max(np.abs(samples))) if samples.size else 0.0
        gain_db = 20.0 * math.log10(gain) if gain > 0.0 else float("-inf")
        return samples, {
            "audioPeakBefore": peak_before,
            "audioPeakAfter": peak_after,
            "gainDb": gain_db,
            "limiterApplied": applied,
        }


    @staticmethod
    def _prepare_final_video(video: Any, final_width: int, final_height: int) -> tuple[Any, Dict[str, Any]]:
        frames = video.detach().float().cpu()
        if frames.ndim == 5:
            frames = frames[0]
        if frames.ndim != 4:
            raise H3RuntimeError(f"unexpected decoded video shape for direct output: {tuple(frames.shape)}")
        if frames.shape[-1] == 3:
            result = frames
        elif frames.shape[1] == 3:
            result = frames.permute(0, 2, 3, 1).contiguous()
        else:
            raise H3RuntimeError(f"decoded video must have RGB channels for direct output: {tuple(frames.shape)}")
        source_t, source_height, source_width = int(result.shape[0]), int(result.shape[1]), int(result.shape[2])
        if (source_width, source_height) != (final_width, final_height):
            raise H3RuntimeError(
                f"stage2 decoded dimensions do not match direct output canvas: {source_width}x{source_height} != {final_width}x{final_height}"
            )
        return result, {
            "executed": False,
            "algorithm": None,
            "sourceCanvas": {"width": source_width, "height": source_height},
            "finalOutputCanvas": {"width": final_width, "height": final_height},
            "sourceFrameCount": source_t,
            "outputFrameCount": source_t,
            "preserveVideoT": True,
            "directStage2Output": True,
        }

    def _export_mp4(
        self,
        video: Any,
        audio: Any,
        task_id: str,
        compiled: Dict[str, Any],
        execution_receipt: Optional[Dict[str, Any]] = None,
        output_filename: str = "h3_result.mp4",
    ) -> Path:
        import av
        import numpy as np
        from fractions import Fraction

        is_experiment = str((compiled.get("advanced") or {}).get("experimentId") or "") == KJ_EXPERIMENT_ID
        output_root = (ROOT / "_isolated_research" / "outputs") if is_experiment else (ROOT / "output")
        output_dir = (output_root / task_id).resolve()
        output_dir.relative_to(output_root.resolve())
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_filename not in {"h3_result.mp4", "h3_result_bridged_diagnostic.mp4"}:
            raise H3RuntimeError("unsupported H3 output filename")
        output_path = output_dir / output_filename
        expected_width = int(compiled["canvas"]["export"]["width"])
        expected_height = int(compiled["canvas"]["export"]["height"])
        frames, downsample_receipt = self._prepare_final_video(video, expected_width, expected_height)
        actual_height, actual_width = int(frames.shape[1]), int(frames.shape[2])
        if isinstance(execution_receipt, dict):
            execution_receipt.update({
                "decodedResolution": dict(downsample_receipt["sourceCanvas"]),
                "resolutionMismatch": (actual_width, actual_height) != (expected_width, expected_height),
                "decodedFrameCount": int(downsample_receipt["sourceFrameCount"]),
                "finalDownsample": downsample_receipt,
                "fps": int(compiled["timing"]["fps"]),
                "modelFps": int(compiled["timing"].get("modelFps") or 24),
                "outputFps": int(compiled["timing"]["fps"]),
            })
        if (actual_width, actual_height) != (expected_width, expected_height):
            if isinstance(execution_receipt, dict):
                execution_receipt["outputAuthentic"] = False
            raise H3RuntimeError(
                "official VAE decoded dimensions do not match the native export canvas: "
                f"decoded={actual_width}x{actual_height}, expected={expected_width}x{expected_height}"
            )
        output_fps = int(compiled["timing"]["fps"])
        model_fps = int(compiled["timing"].get("modelFps") or 24)
        export_count = int(compiled["timing"]["exportFrameCount"])
        export_indices = _export_frame_indices(int(frames.shape[0]), export_count, model_fps, output_fps)
        frames = frames[export_indices].clamp(0, 1).numpy()
        if audio is None:
            raise H3RuntimeError("audio VAE returned no audio; refusing silent MP4 export")
        audio_samples = audio.detach().float().cpu()
        if audio_samples.ndim == 4:
            audio_samples = audio_samples[0]
        elif audio_samples.ndim == 3:
            # Official H3 audio VAE variants may return [B,T,2] or
            # [B,2,T].  Both are real stereo waveforms; remove only the
            # singleton batch dimension before normalizing channels below.
            if audio_samples.shape[0] == 1:
                audio_samples = audio_samples[0]
            elif audio_samples.shape[1] == 1:
                audio_samples = audio_samples[:, 0, :]
        if audio_samples.ndim != 2:
            raise H3RuntimeError(f"unexpected decoded audio shape: {tuple(audio_samples.shape)}")
        if audio_samples.shape[0] > 8:
            audio_samples = audio_samples.transpose(0, 1)
        if audio_samples.shape[0] == 1:
            audio_samples = audio_samples.repeat(2, 1)
        elif audio_samples.shape[0] > 2:
            audio_samples = audio_samples[:2]
        audio_rate = int(getattr(self.runtime.objects()["audioVae"], "audio_sample_rate", 32000))
        target_audio_samples = int(round(compiled["timing"]["exportDurationSeconds"] * audio_rate))
        audio_samples = audio_samples.numpy()
        if audio_samples.shape[1] < target_audio_samples:
            audio_samples = np.pad(
                audio_samples,
                ((0, 0), (0, target_audio_samples - audio_samples.shape[1])),
                mode="constant",
            )
        else:
            audio_samples = audio_samples[:, :target_audio_samples]
        audio_samples, audio_peak_receipt = self._protect_audio_pcm_peak(audio_samples)
        container = av.open(str(output_path), mode="w")
        try:
            stream = container.add_stream("libx264", rate=output_fps)
            stream.width = int(frames.shape[2])
            stream.height = int(frames.shape[1])
            stream.pix_fmt = "yuv420p"
            audio_stream = container.add_stream("aac", rate=audio_rate)
            audio_stream.layout = "stereo"
            for frame_index, frame in enumerate(frames):
                vf = av.VideoFrame.from_ndarray((frame * 255).astype(np.uint8), format="rgb24")
                vf.pts = frame_index
                vf.time_base = Fraction(1, output_fps)
                for packet in stream.encode(vf):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            chunk_size = 1024
            for offset in range(0, audio_samples.shape[1], chunk_size):
                chunk = audio_samples[:, offset:offset + chunk_size]
                af = av.AudioFrame.from_ndarray(chunk.astype(np.float32), format="fltp", layout="stereo")
                af.sample_rate = audio_rate
                af.pts = offset
                af.time_base = Fraction(1, audio_rate)
                for packet in audio_stream.encode(af):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
        finally:
            container.close()
        if isinstance(execution_receipt, dict):
            execution_receipt.update({
                "exportedResolution": {"width": int(frames.shape[2]), "height": int(frames.shape[1])},
                "exportedFrameCount": int(export_count),
                "actualOutputFps": output_fps,
                "outputPath": str(output_path),
                "outputAuthentic": True,
                **audio_peak_receipt,
            })
        return output_path
