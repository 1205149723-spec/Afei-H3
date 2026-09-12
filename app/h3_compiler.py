"""Deterministic, GPU-free H3 request compiler.

This module only compiles metadata into an execution manifest. It never opens
or copies model files and never performs inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from typing import Any, Dict, Iterable, List

from reference_planner import plan_reference_videos
from sage_policy import choose_sage_policy
from memory_admission import native_managed_receipt
from full_quality_contract import (
    FULL_QUALITY_SAMPLING_CONTRACT,
    full_quality_contract,
    full_quality_fingerprint,
)
from kernel_backend_contract import (
    expected_kernel_backend_contract,
    normalize_kernel_backend,
    serialize_kernel_backend_contract,
)


MODEL_FPS = 24
DEFAULT_FPS = 24
SUPPORTED_FPS = frozenset({24})
LOW_FPS_TIME_REMAP_ROUTE = "low_fps_time_remap_experimental"
LOW_FPS_TIME_REMAP_VERSION = "h3-low-fps-time-remap-experimental-v1"
LOW_FPS_DIAGNOSTIC_VARIANTS = {
    "continuous_direct": {
        "positionMapping": "continuous_time_scale",
        "decodeMode": "direct",
    },
    "continuous_latent_bridge": {
        "positionMapping": "continuous_time_scale",
        "decodeMode": "latent_bridge_to_official_24fps_density",
    },
    "integer_lattice_direct": {
        "positionMapping": "official_24fps_integer_lattice",
        "decodeMode": "direct",
    },
    "integer_lattice_latent_bridge": {
        "positionMapping": "official_24fps_integer_lattice",
        "decodeMode": "latent_bridge_to_official_24fps_density",
    },
}
AUDIO_LATENT_FPS = 40
# Product input range; this is not an execution deadline. The user may cancel
# a running task, and an execution timeout is optional per request.
MAX_PRODUCT_DURATION_SECONDS = 60.0
MIN_PRODUCT_DURATION_SECONDS = 0.5
REFERENCE_LIMITS = {"Picture": 9, "Video": 3, "Audio": 3}
MAX_REFERENCES = 12
MAX_PER_KIND = REFERENCE_LIMITS

MODE_CONFIG = {
    "T2V": {"label": "Text to Video", "model": "FL2VA"},
    "I2V": {"label": "Image to Video", "model": "FL2VA"},
    "FIRST_LAST_FRAME": {"label": "First / Last Frame", "model": "FL2VA"},
    "R2V": {"label": "Reference to Video", "model": "REF2VA"},
}

# The normal product path is a named, stable execution contract.  It retains
# the official H3 conditioning, sampler, VAE and MP4 pipeline while selecting
# the locally validated KJ H3 Sage + token-local FFN implementation.
DEFAULT_EXECUTION_PROFILE = "production_kj_sage_ffn"
KJ_EXPERIMENT_ID = "kj_h3_native_60cd6bc_sm120_experimental_v1"
KJ_EXPERIMENT_COMMIT = "60cd6bc1870db94c6eeb05fbe455147a8e91c4e9"
KJ_EXPERIMENT_PROFILE = "kj_h3_native_experimental"
KJ_EXPERIMENT_KERNEL = "kijai_kj_minimax_h3_native_sage_60cd6bc"
KJ_EXPERIMENT_ROUTE = "kj_h3_native_sage"
KJ_EXPERIMENT_VERSION = "h3-kj-native-sage-60cd6bc-v1"
ACCELERATION_MODE_BY_PROFILE = {
    "full_quality": DEFAULT_EXECUTION_PROFILE,
    KJ_EXPERIMENT_ROUTE: KJ_EXPERIMENT_PROFILE,
}
ACCELERATION_MODE_BY_PROFILE_REVERSE = {value: key for key, value in ACCELERATION_MODE_BY_PROFILE.items()}
ACCELERATION_ROUTE_METADATA = {
    "full_quality": {"approximate": False, "uiSelectable": True, "verification": "validated"},
    KJ_EXPERIMENT_ROUTE: {"approximate": False, "uiSelectable": False, "verification": "gpu_smoke_verified", "experimental": True},
}

# Removed product routes are rejected explicitly.  They are not migrated to
# full_quality and do not participate in execution-profile resolution.
REMOVED_ACCELERATION_ROUTES = {
    "spectrum_conservative", "easycache_conservative", "te_speed_conservative",
    "spectrum", "easycache", "te_speed",
    "turbo_lora_experimental", "larryvrh_author", "t8star_ema", "t8star_non_ema",
}
LEGACY_ACCELERATION_PROFILE_MIGRATIONS: Dict[str, str] = {}


def native_acceleration_route_contract(profile: str) -> Dict[str, Any] | None:
    if str(profile or "") != KJ_EXPERIMENT_PROFILE:
        return None
    return {
        "routeId": KJ_EXPERIMENT_ROUTE,
        "routeVersion": KJ_EXPERIMENT_VERSION,
        "experimentId": KJ_EXPERIMENT_ID,
        "sourceCommit": KJ_EXPERIMENT_COMMIT,
        "requested": KJ_EXPERIMENT_KERNEL,
        "actual": KJ_EXPERIMENT_KERNEL,
        "patchedBlocks": 50,
        "scope": "denoiser_only",
        "fallback": False,
        "tokenChunkFfn": "experimental_only",
        "cudaConstraint": "comfy_kitchen_cuda_backend_requires_cuda_13_plus; host_torch_is_cu128",
    }


def validate_native_route_asset_contract(route_id: str, asset_contract: Dict[str, Any]) -> Dict[str, Any]:
    """Keep unreachable legacy adapters importable while refusing execution."""

    normalized = str(route_id or "").lower()
    if normalized.startswith(("spectrum", "easycache", "te_speed")):
        raise CompilerError(f"route_removed/unsupported_route: {normalized}")
    raise CompilerError(f"route_removed/unsupported_route: {normalized or 'unknown'}")


def native_route_asset_contract(
    profile: str,
    model: str,
    references: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the stable asset contract for the sole product route."""

    if str(profile or "") == KJ_EXPERIMENT_PROFILE:
        return {
            "routeId": KJ_EXPERIMENT_ROUTE,
            "assetClass": "reference" if list(references or []) else "none",
            "status": "isolated_experimental",
            "model": str(model),
            "userSelectable": False,
            "experimentId": KJ_EXPERIMENT_ID,
        }
    if str(profile or "") != DEFAULT_EXECUTION_PROFILE:
        raise CompilerError(f"route_removed/unsupported_route: {profile}")
    refs = list(references or [])
    compiled = {
        "routeId": FULL_QUALITY_ROUTE_NAME,
        "assetClass": "reference" if refs else "none",
        "status": "native_managed",
        "model": str(model),
        "userSelectable": False,
    }

# This is the public full-quality H3 sampler contract. It is deliberately
# independent of canvas dimensions: a resolution may change only the canvas
# and its resulting H3 latent shape, never the denoiser route.
FULL_QUALITY_ROUTE_NAME = "full_quality"
FULL_QUALITY_ROUTE_VERSION = "h3-full-quality-route-v1"
OFFICIAL_H3_SAMPLER_FIELDS = {
    key: value for key, value in FULL_QUALITY_SAMPLING_CONTRACT.items() if key != "steps"
}
PRODUCT_EXECUTION_PROFILES = {
    DEFAULT_EXECUTION_PROFILE,
    # Kept for old persisted tasks only; the ordinary UI never submits it.
    "production_optimized",
    *ACCELERATION_MODE_BY_PROFILE_REVERSE,
    *LEGACY_ACCELERATION_PROFILE_MIGRATIONS,
}

# These values are deliberately diagnostics-only and must never become a UI
# default.  They remain accepted so historical test records can still replay.
DIAGNOSTIC_EXECUTION_PROFILES = {
    "official_baseline",
    # Hidden test-only candidate.  Spectrum is approximate and cannot alter
    # ordinary product execution until a controlled A/B is reviewed.
    "spectrum_candidate",
    # TeaCache B2 is a test-only H3 feature-residual experiment.  Neither
    # profile is accepted from the ordinary UI or used by product defaults.
    "teacache_b2_calibration",
    "teacache_b2_candidate",
    # MLP-only upstream chunking is a separate, diagnostics-only candidate.
    # It cannot enter normal product requests before numerical/visual A/B.
    "ffn_chunk_diagnostic",
    # Combined diagnostics keep both candidates out of the normal UI/API.
    # Calibration first executes every H3 block; candidate mode is permitted
    # only with the resulting H3-specific calibration receipt.
    "ffn_chunk_b2_calibration",
    "ffn_chunk_b2_candidate",
}
SUPPORTED_EXECUTION_PROFILES = PRODUCT_EXECUTION_PROFILES | DIAGNOSTIC_EXECUTION_PROFILES


class CompilerError(ValueError):
    """A user-correctable request validation error."""


def _verbatim_prompt_contract(prompt: str) -> Dict[str, Any]:
    """Record delivery integrity without interpreting the user's words."""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return {
        "contractVersion": "h3-verbatim-prompt-v1",
        "format": "verbatim_user_prompt",
        "localTransformation": False,
        "dialogueManifest": [],
        "rawPromptSha256": digest,
        "compiledPromptSha256": digest,
    }


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CompilerError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise CompilerError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str, maximum: int = 8192) -> int:
    number = _number(value, name)
    if number <= 0 or int(number) != number:
        raise CompilerError(f"{name} must be a positive integer")
    if number > maximum:
        raise CompilerError(f"{name} must be <= {maximum}")
    return int(number)


def _seed_int(value: Any, name: str = "seed") -> int:
    if isinstance(value, bool):
        raise CompilerError(f"{name} must be between 0 and 9223372036854775807")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str):
        text = value.strip()
        if not text or not text.isdecimal():
            raise CompilerError(f"{name} must be between 0 and 9223372036854775807")
        number = int(text, 10)
    else:
        raise CompilerError(f"{name} must be an integer or decimal string")
    if number < 0 or number > 9223372036854775807:
        raise CompilerError(f"{name} must be between 0 and 9223372036854775807")
    return number


def round_up_16(value: int) -> int:
    return int(math.ceil(value / 16.0) * 16)


def official_h3_sampler_contract(steps: Any) -> Dict[str, Any]:
    """Return the fixed H3 sampler fields shared by every canvas size."""

    return {**OFFICIAL_H3_SAMPLER_FIELDS, "steps": _positive_int(steps, "steps", maximum=200)}

SUPPORTED_STEPS = frozenset({20, 30, 40, 50, 60})
STAGE1_STEP_OPTIONS = {20, 30, 40, 50, 60}
STAGE2_STEP_OPTIONS = {3, 5, 7, 10}
DEFAULT_STAGE1_STEPS = 20
DEFAULT_STAGE2_STEPS = 3
MEMORY_STRATEGIES = {"auto"}
ASSET_PRECISION_MODES = {"official", "fixed_optimized"}
FIXED_ASSET_MAX_PIXELS = 960 * 544
LATENT_UPSCALE_TWO_STAGE_ROUTE = "latent_upscale_two_stage"
LATENT_UPSCALE_TWO_STAGE_VERSION = "h3-two-stage-8775-v5"
LATENT_SPATIAL_DIVISOR = 16
TWO_STAGE_CANVAS_MULTIPLE = 32
STAGE1_TARGET_AREA_RATIO = 0.50
STAGE2_DENOISE = 0.30
MODEL_QUANT = os.environ.get("H3_MODEL_QUANT", "w4a8").strip().lower()
if MODEL_QUANT not in {"int8", "w4a8"}:
    raise RuntimeError("H3_MODEL_QUANT must be int8 or w4a8")


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def _ceil_to_even(value: float) -> int:
    """Return the smallest even integer greater than or equal to ``value``."""

    return max(2, int(math.ceil(float(value) / 2.0)) * 2)


def _two_stage_canvas_contract(canvas: Dict[str, Any]) -> Dict[str, Any]:
    final = dict(canvas.get("export") or {})
    final_width, final_height = int(final.get("width") or 0), int(final.get("height") or 0)
    if final_width <= 0 or final_height <= 0 or final_width % LATENT_SPATIAL_DIVISOR or final_height % LATENT_SPATIAL_DIVISOR:
        raise CompilerError(
            f"two-stage final output canvas must be divisible by {LATENT_SPATIAL_DIVISOR}: {final_width}x{final_height}"
        )
    latent_mode = os.environ.get("H3_LATENT_UPSCALE_MODE", "latent").strip().lower()
    if latent_mode not in {"latent", "vsr_recovery"}:
        raise CompilerError(f"unsupported H3_LATENT_UPSCALE_MODE: {latent_mode}")
    linear_target = math.sqrt(STAGE1_TARGET_AREA_RATIO)
    if latent_mode == "latent":
        # LBH is strictly 2x in latent space. Round each stage1 latent axis
        # up to an even value so its pixel canvas is always a 32 multiple;
        # the 2x output is cropped back to the final latent target at runtime.
        final_latent_w = final_width // LATENT_SPATIAL_DIVISOR
        final_latent_h = final_height // LATENT_SPATIAL_DIVISOR
        stage1_latent_w = _ceil_to_even(final_latent_w / 2.0)
        stage1_latent_h = _ceil_to_even(final_latent_h / 2.0)
        stage1_width = stage1_latent_w * LATENT_SPATIAL_DIVISOR
        stage1_height = stage1_latent_h * LATENT_SPATIAL_DIVISOR
        canvas_multiple = LATENT_SPATIAL_DIVISOR * 2
    else:
        stage1_width = _nearest_multiple(final_width * linear_target, TWO_STAGE_CANVAS_MULTIPLE)
        stage1_height = _nearest_multiple(final_height * linear_target, TWO_STAGE_CANVAS_MULTIPLE)
        canvas_multiple = TWO_STAGE_CANVAS_MULTIPLE
    stage1_area = stage1_width * stage1_height
    final_area = final_width * final_height
    width_scale = final_width / stage1_width
    height_scale = final_height / stage1_height
    return {
        "finalOutputCanvas": {"width": final_width, "height": final_height},
        "stage1Canvas": {"width": stage1_width, "height": stage1_height},
        "stage2InternalCanvas": {"width": final_width, "height": final_height},
        "stage1SourceCanvas": {"width": stage1_width, "height": stage1_height},
        "stage2TargetCanvas": {"width": final_width, "height": final_height},
        "stage1SourceLatent": {"height": stage1_height // LATENT_SPATIAL_DIVISOR, "width": stage1_width // LATENT_SPATIAL_DIVISOR},
        "stage2TargetLatent": {"height": final_height // LATENT_SPATIAL_DIVISOR, "width": final_width // LATENT_SPATIAL_DIVISOR},
        "stage1AreaRatio": round(stage1_area / final_area, 6),
        "stage2AreaScale": round(final_area / stage1_area, 6),
        "stage2LinearScale": {"width": round(width_scale, 6), "height": round(height_scale, 6)},
        "internalScale": round(math.sqrt(width_scale * height_scale), 6),
        "canvasMultiple": canvas_multiple,
        "lbhOutputLatent": {
            "height": (stage1_height // LATENT_SPATIAL_DIVISOR) * 2,
            "width": (stage1_width // LATENT_SPATIAL_DIVISOR) * 2,
        } if latent_mode == "latent" else None,
        "lbhCrop": {
            "strategy": "top_left_edge",
            "top": 0,
            "left": 0,
            "height": final_height // LATENT_SPATIAL_DIVISOR,
            "width": final_width // LATENT_SPATIAL_DIVISOR,
        } if latent_mode == "latent" else None,
        "frameResize": ({
            "implementation": "LBH-123-AI.Comfyui_Minimax_h3_latent_Upscaler",
            "sourceCommit": "d7c01b9011f2e8439493f6c02c29995a27df276f",
            "algorithm": "minimax_h3_latent_upscaler_3d",
            "quality": "learned_latent_resize",
            "device": "native",
            "defaultRouteMode": "latent",
            "runtimeOverride": {"environmentVariable": "H3_LATENT_UPSCALE_MODE", "recoveryMode": "vsr_recovery", "fallbackAllowed": False},
            "nvidiaRtxVsr": False,
            "videoVaeRoundTrip": False,
            "fallbackUsed": False,
        } if latent_mode == "latent" else {
            "implementation": "Kijai.ComfyUI-KJNodes.ImageResizeKJv2",
            "sourceCommit": "60cd6bc1870db94c6eeb05fbe455147a8e91c4e9",
            "algorithm": "nvidia_rtx_vsr",
            "quality": "ULTRA",
            "package": "nvidia-vfx==0.1.0.1",
            "vfxSdkVersion": "1.2.0.0",
            "keepProportion": "crop",
            "cropPosition": "center",
            "device": "cpu",
            "defaultRouteMode": "vsr_recovery",
            "runtimeOverride": {"environmentVariable": "H3_LATENT_UPSCALE_MODE", "recoveryMode": "vsr_recovery", "fallbackAllowed": False},
            "nvidiaRtxVsr": True,
            "videoVaeRoundTrip": True,
            "fallbackUsed": False,
        }),
        "latentUpscale": {
            "environmentVariable": "H3_LATENT_UPSCALE_MODE",
            "mode": "latent",
            "selectedMode": latent_mode,
            "implementation": "LBH-123-AI.Comfyui_Minimax_h3_latent_Upscaler",
            "vendorCommit": "d7c01b9011f2e8439493f6c02c29995a27df276f",
            "weightRelativePath": "models/latent_upscaler/minimax_h3_latent_upscaler_3d_fp16.safetensors",
            "weightSha256": "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2",
            "exactLatentScale": 2.0,
            "videoVaeRoundTrip": False,
            "fallbackAllowed": False,
            "default": True,
        },
        "finalDownsample": {"executed": False, "algorithm": None, "preserveVideoT": True},
        "directStage2Output": True,
    }


def _normalize_tuning(
    advanced: Dict[str, Any], requested_kernel: str, execution_profile: str
) -> tuple[int, str, int | None, str]:
    legacy = {"single", "continuous", "renoise", "splitStep"}
    supplied = sorted(key for key in legacy if key in advanced)
    if supplied and "secondSamplingPreset" in advanced:
        for key in supplied:
            advanced.pop(key, None)
    elif supplied:
        raise CompilerError("旧二采字段已停用：" + ", ".join(supplied))
    if "secondSamplingPreset" in advanced:
        raise CompilerError("secondSamplingPreset 已停用；请提交 stage1Steps 和 stage2Steps")
    try:
        stage1_steps = int(advanced.get("stage1Steps", DEFAULT_STAGE1_STEPS))
        stage2_steps = int(advanced.get("stage2Steps", DEFAULT_STAGE2_STEPS))
    except (TypeError, ValueError) as exc:
        raise CompilerError("stage1Steps 和 stage2Steps 必须是整数") from exc
    if stage1_steps not in STAGE1_STEP_OPTIONS:
        raise CompilerError("stage1Steps must be one of 20, 30, 40, 50, 60")
    if stage2_steps not in STAGE2_STEP_OPTIONS:
        raise CompilerError("stage2Steps must be one of 3, 5, 7, 10")
    advanced["stage1Steps"] = stage1_steps
    advanced["stage2Steps"] = stage2_steps
    advanced["stage2Denoise"] = STAGE2_DENOISE
    advanced["modelQuant"] = MODEL_QUANT
    memory = str(advanced.get("memoryStrategy") or advanced.get("memoryMode") or "auto").strip().lower()
    if memory not in MEMORY_STRATEGIES:
        raise CompilerError("memoryStrategy must be auto; custom memory strategies are retired")
    requested_ffn = advanced.get("ffnChunks")
    if requested_ffn in {None, "", "auto"}:
        ffn_chunks = 2 if requested_kernel == "kijai_fast" and execution_profile == DEFAULT_EXECUTION_PROFILE else None
    else:
        try:
            ffn_chunks = int(requested_ffn)
        except (TypeError, ValueError) as exc:
            raise CompilerError("ffnChunks 必须是 1、2 或 4") from exc
        if ffn_chunks not in {1, 2, 4} or requested_kernel != "kijai_fast":
            raise CompilerError("ffnChunks 只适用于 Kijai 高速内核且必须是 1、2 或 4")
    return stage1_steps, "auto", ffn_chunks, memory


def _normalize_asset_precision(advanced: Dict[str, Any]) -> str:
    value = str(advanced.get("assetPrecision") or "official").strip().lower()
    if value not in ASSET_PRECISION_MODES:
        raise CompilerError("assetPrecision must be official or fixed_optimized")
    return value


def two_stage_route_contract(mode: str, requested_kernel: str, steps: int, ffn_chunks: Any, advanced: Dict[str, Any], canvas: Dict[str, Any]) -> Dict[str, Any]:
    base = json.loads(json.dumps(full_quality_contract(mode, requested_kernel, steps, ffn_chunks)))
    stage1_steps = int(advanced["stage1Steps"])
    stage2_steps = int(advanced["stage2Steps"])
    canvas_contract = _two_stage_canvas_contract(canvas)
    base.update({
        "contractVersion": LATENT_UPSCALE_TWO_STAGE_VERSION,
        "routeId": LATENT_UPSCALE_TWO_STAGE_ROUTE,
        "baseRouteId": "full_quality",
        "twoStage": {
            "stage1Steps": stage1_steps,
            "stage2Steps": stage2_steps,
            "stage2Denoise": STAGE2_DENOISE,
            "stage1Sampler": "res_multistep",
            "stage1Scheduler": "simple",
            "stage2Sampler": "euler",
            "stage2Scheduler": "beta",
            "modelQuant": MODEL_QUANT,
            "stage1FfnChunks": 1,
            "stage2FfnChunks": int(ffn_chunks or 1),
            "stage1FfnApplied": False,
            "stage2FfnApplied": requested_kernel == "kijai_fast" and int(ffn_chunks or 1) > 1,
            **canvas_contract,
            "frameResizeVaeRoundTrip": dict(canvas_contract["frameResize"]),
        },
    })
    return base


def algorithm_route_receipt(mode: str, model: str, advanced: Dict[str, Any], canvas: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the resolution-invariant execution route and fingerprint it."""

    profile = str(advanced.get("executionProfile") or DEFAULT_EXECUTION_PROFILE)
    acceleration_mode = str(advanced.get("accelerationMode") or FULL_QUALITY_ROUTE_NAME)
    requested_kernel = str(advanced.get("requestedKernel") or "kijai_fast")
    sampler = official_h3_sampler_contract(advanced.get("steps") or 20)
    stage1_steps = int(advanced["stage1Steps"])
    stage2_steps = int(advanced["stage2Steps"])

    native_contract = native_acceleration_route_contract(profile)
    route_id = native_contract["routeId"] if native_contract else LATENT_UPSCALE_TWO_STAGE_ROUTE
    route_version = native_contract.get("routeVersion") if native_contract else LATENT_UPSCALE_TWO_STAGE_VERSION
    route = {
        # The user-facing selector remains separate from the route that the
        # worker actually executes.  Native author routes must never be
        # reported as a legacy product profile or as full quality.
        "requestedAccelerationMode": acceleration_mode,
        "algorithmRoute": route_id,
        "algorithmRouteVersion": route_version,
        "model": str(model),
        "algorithmProfile": profile,
        "sampler": sampler["sampler"],
        "scheduler": sampler["scheduler"],
        "steps": sampler["steps"],
        "cfg": sampler["cfg"],
        "guider": sampler["guider"],
        "conditioningInputs": list(sampler["conditioningInputs"]),
        "usesNegativeConditioning": sampler["usesNegativeConditioning"],
        "noise": sampler["noise"],
        "denoise": sampler["denoise"],
        "executor": sampler["executor"],
        "routeContract": native_contract or two_stage_route_contract(mode, requested_kernel, sampler["steps"], advanced.get("ffnChunks"), advanced, canvas),
    }
    route.update({
        "stage1Steps": stage1_steps,
        "stage2Steps": stage2_steps,
        "stage2Denoise": STAGE2_DENOISE,
        "stage1Sampler": "res_multistep",
        "stage1Scheduler": "simple",
        "stage2Sampler": "euler",
        "stage2Scheduler": "beta",
        "executionKind": (
            "latent_upscale_then_independent_resample"
            if str((canvas.get("latentUpscale") or {}).get("selectedMode") or "latent") == "latent"
            else "frame_resize_vae_reencode_then_independent_resample"
        ),
        "samplerCalls": 2,
    })
    route["frameResizeVaeRoundTrip"] = dict(
        (route.get("routeContract") or {}).get("twoStage", {}).get("frameResizeVaeRoundTrip") or {}
    )
    if native_contract:
        route.update({"experimentId": KJ_EXPERIMENT_ID, "sourceCommit": KJ_EXPERIMENT_COMMIT})
    canonical = json.dumps(route, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    result = {
        **route,
        "algorithmRouteFingerprint": (
            f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
            if native_contract else f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        ),
    }
    migrated_from = str(advanced.get("executionProfileMigratedFrom") or "")
    if migrated_from:
        result["executionProfileMigratedFrom"] = migrated_from
    return result


def low_fps_time_remap_route_receipt(
    mode: str,
    model: str,
    advanced: Dict[str, Any],
    timing: Dict[str, Any],
) -> Dict[str, Any]:
    """Describe the isolated non-official temporal-remap experiment."""

    requested_kernel = str(advanced.get("requestedKernel") or "kijai_fast")
    sampler = official_h3_sampler_contract(advanced.get("steps") or 20)
    route_contract = {
        "contractVersion": LOW_FPS_TIME_REMAP_VERSION,
        "routeId": LOW_FPS_TIME_REMAP_ROUTE,
        "classification": "project_experimental_non_official_fps",
        "baseQualityRoute": FULL_QUALITY_ROUTE_NAME,
        "baseQualityFingerprint": full_quality_fingerprint(mode, requested_kernel),
        "mode": mode,
        "model": str(model),
        "requestedFps": timing["requestedFps"],
        "temporalDensityFps": timing["temporalDensityFps"],
        "officialModelTimebaseFps": MODEL_FPS,
        "modelFrameCount": timing["modelFrameCount"],
        "videoLatentT": timing["videoLatentT"],
        "audioLatentT": timing["audioLatentT"],
        "timeScale": timing["timeScale"],
        "promptTimeMapping": timing["promptTimeMapping"],
        "referenceTimeMapping": timing["referenceTimeMapping"],
        "variantId": timing.get("variantId", "continuous_direct"),
        "positionMapping": timing.get("positionMapping", "continuous_time_scale"),
        "decodeMode": timing.get("decodeMode", "direct"),
        "bridgedModelFps": timing.get("bridgedModelFps"),
        "bridgedModelFrameCount": timing.get("bridgedModelFrameCount"),
        "bridgedVideoLatentT": timing.get("bridgedVideoLatentT"),
        "fallback": False,
        "sampling": sampler,
    }
    route = {
        "requestedAccelerationMode": LOW_FPS_TIME_REMAP_ROUTE,
        "algorithmRoute": LOW_FPS_TIME_REMAP_ROUTE,
        "algorithmRouteVersion": LOW_FPS_TIME_REMAP_VERSION,
        "model": str(model),
        "algorithmProfile": str(advanced.get("executionProfile") or DEFAULT_EXECUTION_PROFILE),
        "sampler": sampler["sampler"],
        "scheduler": sampler["scheduler"],
        "steps": sampler["steps"],
        "cfg": sampler["cfg"],
        "guider": sampler["guider"],
        "conditioningInputs": list(sampler["conditioningInputs"]),
        "usesNegativeConditioning": sampler["usesNegativeConditioning"],
        "noise": sampler["noise"],
        "denoise": sampler["denoise"],
        "executor": sampler["executor"],
        "routeContract": route_contract,
    }
    canonical = json.dumps(route, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    receipt = {
        **route,
        "algorithmRouteFingerprint": f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
    }


def initial_execution_receipt(compiled: Dict[str, Any]) -> Dict[str, Any]:
    """Build the common receipt shell before the worker creates any tensors."""

    canvas = dict(compiled.get("canvas") or {})
    route = dict(compiled.get("algorithmRoute") or {})
    prompt_contract = dict(compiled.get("promptContract") or {})
    internal = dict(canvas.get("effectiveResolution") or {})
    memory_preview = native_managed_receipt()
    requested_kernel = str(compiled.get("requestedKernel") or "kijai_fast")
    timing = dict(compiled.get("timing") or {})
    advanced = dict(compiled.get("advanced") or {})
    stage1_steps = int(advanced["stage1Steps"])
    stage2_steps = int(advanced["stage2Steps"])
    sampler = official_h3_sampler_contract(stage1_steps)
    receipt = {
        "stage1Steps": stage1_steps,
        "stage2Steps": stage2_steps,
        "stage2Denoise": STAGE2_DENOISE,
        "executionKind": str(route.get("executionKind") or "latent_upscale_then_independent_resample"),
        "routeMode": str(((canvas.get("latentUpscale") or {}).get("selectedMode")) or "latent"),
        "reNoise": {"required": True, "construction": "stage2_sampler_noise_and_beta_sigmas", "denoise": STAGE2_DENOISE},
        "samplerCalls": 2,
        "lifecycleCalls": 1,
        "seed": int(advanced.get("seed")) if advanced.get("seed") is not None else None,
        "assetPrecision": advanced.get("assetPrecision", "official"),
        "ffnChunk": advanced.get("ffnChunk", "auto"),
        "ffnChunks": advanced.get("ffnChunks"),
        "stage1FfnChunks": 1,
        "stage2FfnChunks": int(advanced.get("ffnChunks") or 1),
        "stage1FfnApplied": False,
        "stage2FfnApplied": requested_kernel == "kijai_fast" and int(advanced.get("ffnChunks") or 1) > 1,
        "memoryStrategy": advanced.get("memoryStrategy", "auto"),
        "requestedKernel": requested_kernel,
        "actualKernel": None,
        "kernelIdentity": None,
        "patchedBlocks": None,
        "scope": None,
        "ffnEnabled": None,
        "fallback": False,
        "fallbackUsed": False,
        "failedStage": None,
        "frameResizeVaeRoundTrip": dict(route.get("frameResizeVaeRoundTrip") or {}),
        "requestedFps": timing.get("requestedFps", timing.get("fps", DEFAULT_FPS)),
        "modelFps": timing.get("modelFps", MODEL_FPS),
        "actualModelFps": timing.get("actualModelFps", timing.get("modelFps", MODEL_FPS)),
        "temporalDensityFps": timing.get("temporalDensityFps", timing.get("modelFps", MODEL_FPS)),
        "outputFps": timing.get("fps", DEFAULT_FPS),
        "modelFrameCount": timing.get("modelFrameCount", timing.get("frameCount")),
        "videoLatentT": timing.get("videoLatentT"),
        "audioLatentT": timing.get("audioLatentT"),
        "timeScale": timing.get("timeScale", 1.0),
        "promptTimeMapping": timing.get("promptTimeMapping"),
        "referenceTimeMapping": timing.get("referenceTimeMapping"),
        "variantId": timing.get("variantId"),
        "positionMapping": timing.get("positionMapping"),
        "decodeMode": timing.get("decodeMode"),
        "bridgedModelFps": timing.get("bridgedModelFps"),
        "bridgedModelFrameCount": timing.get("bridgedModelFrameCount"),
        "bridgedVideoLatentT": timing.get("bridgedVideoLatentT"),
        "requestedResolution": dict(canvas.get("requestedResolution") or {}),
        "effectiveResolution": dict(canvas.get("effectiveResolution") or {}),
        "resolutionMapping": dict(canvas.get("resolutionMapping") or {}),
        "stage1SourceCanvas": dict(canvas.get("stage1SourceCanvas") or {}),
        "stage2TargetCanvas": dict(canvas.get("stage2TargetCanvas") or {}),
        "finalOutputCanvas": dict(canvas.get("finalOutputCanvas") or {}),
        "stage1Canvas": dict(canvas.get("stage1Canvas") or {}),
        "stage2InternalCanvas": dict(canvas.get("stage2InternalCanvas") or {}),
        "stage1AreaRatio": float(canvas.get("stage1AreaRatio") or 0.0),
        "stage2AreaScale": float(canvas.get("stage2AreaScale") or 0.0),
        "stage2LinearScale": dict(canvas.get("stage2LinearScale") or {}),
        "canvasMultiple": int(canvas.get("canvasMultiple") or TWO_STAGE_CANVAS_MULTIPLE),
        "internalScale": float(canvas.get("internalScale") or 0.0),
        "frameResize": dict(canvas.get("frameResize") or {}),
        "finalDownsample": dict(canvas.get("finalDownsample") or {}),
        "directStage2Output": bool(canvas.get("directStage2Output")),
        "stage1SourceLatent": dict(canvas.get("stage1SourceLatent") or {}),
        "stage2TargetLatent": dict(canvas.get("stage2TargetLatent") or {}),
        "lbhOutputLatent": dict(canvas.get("lbhOutputLatent") or {}),
        "lbhCrop": dict(canvas.get("lbhCrop") or {}),
        # Memory admission is decided once by the worker after native capacity
        # values exist; these fields keep the receipt contract stable before and
        # after that decision.
        "historyLookup": "disabled",
        "resolutionBucket": memory_preview.get("resolutionBucket"),
        "durationBucket": memory_preview.get("durationBucket"),
        "decisionSource": memory_preview["decisionSource"],
        "offloadPlan": memory_preview["offloadPlan"],
        "latentShapes": {"video": None, "audio": None},
        "packedTokens": None,
        "routeAssetContract": dict(compiled.get("routeAssetContract") or {}),
        "promptContract": prompt_contract,
        "dialogueManifest": list(prompt_contract.get("dialogueManifest") or []),
        "executionRouteContract": dict(route.get("routeContract") or {}),
        **route,
        "stageTimings": {},
        "progressReceipt": {
            "schemaVersion": 1,
            "currentStage": "queue",
            "lastMeaningfulStage": "queue",
            "phaseStartedAt": None,
            "phaseEndedAt": None,
            "elapsed": None,
            "phaseStep": 0,
            "phaseTotal": 0,
            "globalStep": 0,
            "globalTotal": stage1_steps + stage2_steps,
        },
        "lastMeaningfulStage": "queue",
        "firstDenoiserBlockAt": None,
        "resolutionMismatch": False,
        "outputAuthentic": False,
        "performanceBaseline": {
            "status": "awaiting_independent_gpu_baseline",
            "comparable": False,
            "requiredFields": [
                "wallClockSeconds", "primarySamplingSeconds", "videoDecodeSeconds",
                "audioDecodeSeconds", "gpuUtilizationPercent", "gpuMemoryUsedMiB",
                "width", "height", "durationSeconds", "modelFrames", "exportFrames",
                "sampler", "scheduler", "steps", "algorithmRouteFingerprint",
            ],
            "inputScale": {
                "width": internal.get("width"),
                "height": internal.get("height"),
                "durationSeconds": (
                    compiled.get("timing", {}).get("durationSeconds")
                    if compiled.get("timing", {}).get("durationSeconds") is not None
                    else compiled.get("timing", {}).get("exportDurationSeconds")
                ),
                "modelFrames": compiled.get("timing", {}).get("frameCount"),
                "exportFrames": compiled.get("timing", {}).get("exportFrameCount"),
            },
        },
    }
    if route.get("experimentId") == KJ_EXPERIMENT_ID:
        receipt.update({
            "experimentId": KJ_EXPERIMENT_ID,
            "sourceCommit": KJ_EXPERIMENT_COMMIT,
            "requested": KJ_EXPERIMENT_KERNEL,
            "actual": KJ_EXPERIMENT_KERNEL,
            "requestedKernel": KJ_EXPERIMENT_KERNEL,
            "actualKernel": KJ_EXPERIMENT_KERNEL,
            "patchedBlocks": 50,
            "scope": "denoiser_only",
            "fallback": False,
        })
    return receipt


def frames_for_duration(duration_seconds: Any, fps: Any = DEFAULT_FPS) -> Dict[str, Any]:
    """Map seconds to a legal H3 frame count that can cover the export.

    H3 requires ``17*k+5`` frames.  The legal model count is rounded upward,
    never downward: an export must not ask the decoder for more frames than the
    model produced.  Thus 15 seconds (360 nominal frames) is 362 frames,
    i.e. ``17*21+5``.
    """

    duration = _number(duration_seconds, "durationSeconds")
    if duration < MIN_PRODUCT_DURATION_SECONDS or duration > MAX_PRODUCT_DURATION_SECONDS:
        raise CompilerError(
            f"durationSeconds must be between {MIN_PRODUCT_DURATION_SECONDS:g} and {MAX_PRODUCT_DURATION_SECONDS:g}"
        )
    output_fps = _positive_int(fps, "fps")
    if output_fps not in SUPPORTED_FPS:
        raise CompilerError("fps must be exactly 24")
    temporal_density_fps = MODEL_FPS if output_fps == MODEL_FPS else output_fps
    model_nominal = duration * temporal_density_fps
    k = max(1, math.ceil((model_nominal - 5) / 17))
    frame_count = 17 * k + 5
    export_frame_count = int(round(duration * output_fps))
    video_latent_t = 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2
    time_scale = MODEL_FPS / temporal_density_fps
    effective_duration = frame_count / temporal_density_fps
    audio_latent_t = round(effective_duration * AUDIO_LATENT_FPS)
    return {
        "durationSeconds": duration,
        "fps": output_fps,
        "requestedFps": output_fps,
        "modelFps": temporal_density_fps,
        "actualModelFps": temporal_density_fps,
        "temporalDensityFps": temporal_density_fps,
        "officialModelTimebaseFps": MODEL_FPS,
        "nominalFrames": model_nominal,
        "k": k,
        "frameCount": frame_count,
        "modelFrameCount": frame_count,
        "videoLatentT": video_latent_t,
        "audioLatentT": audio_latent_t,
        "timeScale": time_scale,
        "exportFrameCount": export_frame_count,
        "exportDurationSeconds": round(export_frame_count / output_fps, 6),
        "formula": f"17*{k}+5",
        "effectiveDurationSeconds": effective_duration,
        "promptTimeMapping": "target_wall_clock_seconds_verbatim",
        "referenceTimeMapping": "source_24fps_and_qwen_2fps_unchanged",
        "variantId": "continuous_direct" if output_fps != MODEL_FPS else None,
        "positionMapping": "continuous_time_scale" if output_fps != MODEL_FPS else None,
        "decodeMode": "direct" if output_fps != MODEL_FPS else None,
        "bridgedModelFps": None,
        "bridgedModelFrameCount": None,
        "bridgedVideoLatentT": None,
    }


def _kind_for_reference(reference: Dict[str, Any]) -> str:
    facts = reference.get("mediaFacts") if isinstance(reference.get("mediaFacts"), dict) else {}
    raw = str(facts.get("kind") or reference.get("kind") or reference.get("mediaType") or "").lower()
    aliases = {
        "image": "Picture",
        "picture": "Picture",
        "图片": "Picture",
        "video": "Video",
        "视频": "Video",
        "audio": "Audio",
        "音频": "Audio",
    }
    if raw not in aliases:
        raise CompilerError("each reference kind must be image, video, or audio")
    return aliases[raw]


def _official_asset_contract(kind: str, reference: Dict[str, Any], index: int) -> List[str]:
    facts = reference.get("mediaFacts") if isinstance(reference.get("mediaFacts"), dict) else {}
    if not facts:
        return [f"第 {index} 个素材缺少服务端媒体探测事实，真实任务必须重新探测"]
    size = int(reference.get("size") or 0)
    limits = {"Picture": 30 * 1024 * 1024, "Video": 50 * 1024 * 1024, "Audio": 15 * 1024 * 1024}
    if size > limits[kind]:
        raise CompilerError(f"第 {index} 个{kind}素材超过官方单文件大小限制")
    if kind == "Picture":
        video = facts.get("video") or {}
        codec = str(video.get("codec") or "").lower()
        if codec not in {"mjpeg", "png", "webp", "hevc"}:
            raise CompilerError(f"第 {index} 张图片格式不符合 JPG/JPEG/PNG/WEBP/HEIC/HEIF 官方合同")
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        if not (256 <= width <= 5760 and 256 <= height <= 5760 and 0.4 <= width / height <= 2.5):
            raise CompilerError(f"第 {index} 张图片尺寸或宽高比超出官方范围")
    elif kind == "Video":
        video = facts.get("video") or {}
        codec = str(video.get("codec") or "").lower()
        format_name = str(facts.get("formatName") or "").lower()
        if not ({"mov", "mp4"} & set(format_name.split(","))) or codec not in {"h264", "hevc"}:
            raise CompilerError(f"第 {index} 个视频必须是 MP4/MOV 且编码为 H.264/H.265")
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        fps = float(video.get("avgFps") or 0)
        duration = float(facts.get("durationSeconds") or video.get("durationSeconds") or 0)
        if not (256 <= width <= 5760 and 256 <= height <= 5760 and 0.4 <= width / height <= 2.5):
            raise CompilerError(f"第 {index} 个视频尺寸或宽高比超出官方范围")
        if not 23.976 <= fps <= 60:
            raise CompilerError(f"第 {index} 个视频 FPS 必须在 23.976 到 60 之间")
        start = float(reference.get("startSeconds") or 0)
        end = float(reference.get("endSeconds") if reference.get("endSeconds") not in {None, ""} else duration)
        if not 2 <= end - start <= 15:
            raise CompilerError(f"第 {index} 个视频实际选段必须在 2 到 15 秒之间")
    else:
        format_name = str(facts.get("formatName") or "").lower()
        audio = (facts.get("audioStreams") or [{}])[0]
        codec = str(audio.get("codec") or "").lower()
        if not ({"wav", "mp3"} & set(format_name.split(","))) and codec not in {"pcm_s16le", "pcm_s24le", "pcm_s32le", "mp3"}:
            raise CompilerError(f"第 {index} 个独立音频必须是 WAV 或 MP3")
        duration = float(facts.get("durationSeconds") or audio.get("durationSeconds") or 0)
        if not 2 <= duration <= 15:
            raise CompilerError(f"第 {index} 个独立音频时长必须在 2 到 15 秒之间")
    return []


def compile_references(raw_references: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    references = list(raw_references or [])
    if len(references) > MAX_REFERENCES:
        raise CompilerError(
            f"at most {MAX_REFERENCES} reference materials are allowed "
            f"(pictures {REFERENCE_LIMITS['Picture']}, videos {REFERENCE_LIMITS['Video']}, "
            f"audios {REFERENCE_LIMITS['Audio']})"
        )

    counts = {"Picture": 0, "Video": 0, "Audio": 0}
    compiled: List[Dict[str, Any]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            raise CompilerError(f"reference {index + 1} must be an object")
        kind = _kind_for_reference(reference)
        counts[kind] += 1
        if counts[kind] > MAX_PER_KIND[kind]:
            raise CompilerError(
                f"at most {MAX_PER_KIND[kind]} {kind.lower()} references are allowed "
                f"by the official H3 capacity"
            )
        name = str(reference.get("name") or f"reference-{index + 1}").strip()
        if not name:
            raise CompilerError(f"reference {index + 1} needs a name")
        token = f"<{kind} {counts[kind]}>"
        media_facts = dict(reference.get("mediaFacts") or {})
        warnings = _official_asset_contract(kind, reference, index + 1)
        video_facts = media_facts.get("video") or {}
        compiled.append(
            {
                "inputIndex": index,
                "token": token,
                "kind": kind,
                "name": name,
                "role": str(reference.get("role") or "general"),
                "size": int(reference.get("size") or 0),
                "mimeType": str(reference.get("mimeType") or ""),
                "path": str(reference.get("path") or ""),
                "assetId": str(reference.get("assetId") or ""),
                "sha256": str(reference.get("sha256") or reference.get("contentHash") or ""),
                "mediaFacts": media_facts,
                "assetPrecision": "official",
                "warnings": warnings,
                "originalDurationSeconds": media_facts.get("durationSeconds", reference.get("originalDurationSeconds")),
                "sourceFrameCount": video_facts.get("frameCount", reference.get("sourceFrameCount")),
                "sourceFps": video_facts.get("avgFps"),
                "startSeconds": reference.get("startSeconds"),
                "endSeconds": reference.get("endSeconds"),
            }
        )
    return compiled


def _compile_canvas(payload: Dict[str, Any]) -> Dict[str, Any]:
    ratio = str(payload.get("aspectRatio") or "16:9").replace(" ", "")
    requested_preset = str(payload.get("resolutionPreset") or "720p").lower()
    preset = requested_preset
    # Historical API value. Keep old saved requests working while exposing only
    # ordinary p labels in the product UI.
    if preset == "1024":
        preset = "576p"
    requested_width = _positive_int(payload.get("exportWidth", 1280), "exportWidth")
    requested_height = _positive_int(payload.get("exportHeight", 720), "exportHeight")
    width, height = requested_width, requested_height
    fixed_canvases = {
        "1080p": {
            "16:9": (1920, 1088), "9:16": (1088, 1920), "1:1": (1088, 1088),
            "4:3": (1456, 1088), "3:4": (1088, 1456),
        },
        "720p": {
            "16:9": (1248, 704), "9:16": (704, 1248), "1:1": (704, 704),
            "4:3": (928, 704), "3:4": (704, 928),
        },
        "768p": {
            "16:9": (1344, 768), "9:16": (768, 1344), "1:1": (768, 768),
            "4:3": (1024, 768), "3:4": (768, 1024),
        },
        "640p": {
            "16:9": (1152, 640), "9:16": (640, 1152), "1:1": (640, 640),
            "4:3": (864, 640), "3:4": (640, 864),
        },
        "576p": {
            "16:9": (1024, 576), "9:16": (576, 1024), "1:1": (576, 576),
            "4:3": (768, 576), "3:4": (576, 768),
        },
        "480p": {
            "16:9": (848, 480), "9:16": (480, 848), "1:1": (480, 480),
            "4:3": (640, 480), "3:4": (480, 640),
        },
    }
    dimensions = fixed_canvases.get(preset, {}).get(ratio)
    if dimensions:
        # Fixed presets are the source of truth. Ignore stale client dimensions
        # from an older draft so requested/effective receipts cannot disagree.
        requested_width, requested_height = dimensions
        width, height = dimensions
        internal_width, internal_height = width, height
        baseline = f"{ratio} · {preset} · actual generation and export {width}x{height}"
    else:
        if preset != "custom":
            raise CompilerError(f"未知分辨率档位：{requested_preset}。请选择工作台提供的分辨率。")
        width, height = round_up_16(width), round_up_16(height)
        internal_width, internal_height = width, height
        baseline = f"Custom · actual generation and export {width}x{height}"
    requested_resolution = {
        "preset": requested_preset,
        "aspectRatio": ratio,
        "width": requested_width,
        "height": requested_height,
    }
    effective_resolution = {
        "preset": preset,
        "aspectRatio": ratio,
        "width": width,
        "height": height,
    }
    resolution_mapping = {
        "kind": "fixed_preset" if dimensions else "round_up_16",
        "requestedPreset": requested_preset,
        "effectivePreset": preset,
        "aspectRatio": ratio,
        "requestedCanvas": {"width": requested_width, "height": requested_height},
        "effectiveCanvas": {"width": width, "height": height},
        "rule": f"{requested_preset}/{ratio} -> {width}x{height}",
    }
    return {
        "aspectRatio": ratio,
        "resolutionPreset": preset,
        "export": {"width": width, "height": height},
        "internal": {"width": internal_width, "height": internal_height},
        "internalMultiple": 32,
        "baseline": baseline,
        "requestedResolution": requested_resolution,
        "effectiveResolution": effective_resolution,
        "resolutionMapping": resolution_mapping,
    }


def _image_preprocess_plan(mode: str, item: Dict[str, Any], ordinal: int, canvas: Dict[str, Any]) -> Dict[str, Any]:
    facts = (item.get("mediaFacts") or {}).get("video") or {}
    source_width, source_height = int(facts.get("width") or 0), int(facts.get("height") or 0)
    target_width, target_height = int(canvas["internal"]["width"]), int(canvas["internal"]["height"])
    if source_width <= 0 or source_height <= 0:
        return {"mode": "等待服务端真实图片尺寸", "warning": "缺少尺寸事实，真实任务不得执行"}
    if mode == "R2V":
        budget = FIXED_ASSET_MAX_PIXELS if str(item.get("assetPrecision") or "official") == "fixed_optimized" else target_width * target_height
        scale = min(1.0, math.sqrt(budget / max(1, source_width * source_height)))
        width = max(32, int(round(source_width * scale / 32)) * 32)
        height = max(32, int(round(source_height * scale / 32)) * 32)
        return {"mode": "R2V match", "resize": {"width": width, "height": height, "filter": "Lanczos"}, "crop": "不裁剪", "upscale": False}
    if mode == "FIRST_LAST_FRAME" and ordinal == 2:
        scale = max(target_width / source_width, target_height / source_height)
        resized_width, resized_height = round(source_width * scale), round(source_height * scale)
        return {"mode": "尾帧 cover-center-crop", "resize": {"width": resized_width, "height": resized_height}, "crop": {"width": target_width, "height": target_height, "left": (resized_width - target_width) // 2, "top": (resized_height - target_height) // 2}}
    return {"mode": "首帧拉伸到画布", "resize": {"width": target_width, "height": target_height}, "crop": "不裁剪"}


def compile_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise CompilerError("request body must be a JSON object")
    mode = str(payload.get("mode") or "R2V").upper()
    if mode not in MODE_CONFIG:
        raise CompilerError("mode must be T2V, I2V, FIRST_LAST_FRAME, or R2V")

    raw_prompt = payload.get("prompt")
    prompt = "" if raw_prompt is None else str(raw_prompt)
    if not prompt.strip():
        raise CompilerError("prompt is required")

    references = compile_references(payload.get("references", []))
    timing = frames_for_duration(payload.get("durationSeconds", 15), payload.get("fps", DEFAULT_FPS))
    raw_diagnostics = payload.get("diagnostics")
    diagnostic_config = None
    if raw_diagnostics is not None:
        if raw_diagnostics == {"latentDualDecode": True}:
            if os.environ.get("H3_ENABLE_LATENT_DUAL_DECODE") != "1":
                raise CompilerError("latent dual decode diagnostic is disabled")
            if mode != "T2V" or timing["requestedFps"] != 16 or timing["durationSeconds"] != 15:
                raise CompilerError("latent dual decode diagnostic requires T2V at 16 fps for 15 seconds")
            if timing["modelFrameCount"] != 243 or timing["videoLatentT"] != 72 or timing["audioLatentT"] != 608:
                raise CompilerError("latent dual decode diagnostic timing identity mismatch")
            diagnostic_config = {
                "latentDualDecode": True,
                "sourceVideoLatentT": 72,
                "bridgedVideoLatentT": 107,
                "sourceModelFps": 16,
                "bridgedModelFps": 24,
                "exportFps": 16,
                "exportFrameCount": 240,
            }
        elif isinstance(raw_diagnostics, dict) and set(raw_diagnostics) == {"lowFpsVariant"}:
            variant_id = str(raw_diagnostics.get("lowFpsVariant") or "")
            variant = LOW_FPS_DIAGNOSTIC_VARIANTS.get(variant_id)
            if variant is None:
                raise CompilerError("unsupported low-FPS diagnostic variant")
            if os.environ.get("H3_ENABLE_LOW_FPS_VARIANTS") != "1":
                raise CompilerError("low-FPS diagnostic variants are disabled")
            if mode != "T2V" or timing["requestedFps"] != 16 or timing["durationSeconds"] != 5:
                raise CompilerError("low-FPS diagnostic variants require T2V at 16 fps for 5 seconds")
            bridged_timing = frames_for_duration(timing["durationSeconds"], MODEL_FPS)
            timing.update({
                "variantId": variant_id,
                "positionMapping": variant["positionMapping"],
                "decodeMode": variant["decodeMode"],
                "bridgedModelFps": MODEL_FPS if "latent_bridge" in variant_id else None,
                "bridgedModelFrameCount": bridged_timing["modelFrameCount"] if "latent_bridge" in variant_id else None,
                "bridgedVideoLatentT": bridged_timing["videoLatentT"] if "latent_bridge" in variant_id else None,
            })
            diagnostic_config = {
                "lowFpsVariant": variant_id,
                **variant,
                "sourceVideoLatentT": timing["videoLatentT"],
                "bridgedVideoLatentT": timing["bridgedVideoLatentT"],
                "sourceModelFps": 16,
                "bridgedModelFps": timing["bridgedModelFps"],
                "exportFps": 16,
                "exportFrameCount": timing["exportFrameCount"],
            }
        else:
            raise CompilerError("unsupported diagnostic request")
    canvas = _compile_canvas(payload)
    canvas.update(_two_stage_canvas_contract(canvas))
    pictures = [item for item in references if item["kind"] == "Picture"]
    videos = [item for item in references if item["kind"] == "Video"]
    audios = [item for item in references if item["kind"] == "Audio"]
    if mode == "T2V" and references:
        raise CompilerError("T2V does not accept reference materials")
    if mode == "I2V" and not pictures:
        raise CompilerError("I2V requires at least one picture reference")
    if mode == "I2V" and (len(pictures) != 1 or len(references) != 1):
        raise CompilerError("图生视频模式只接受一张首帧或尾帧图片")
    if mode == "FIRST_LAST_FRAME" and len(pictures) != 2:
        raise CompilerError("First / Last Frame requires exactly two picture references")
    if mode == "FIRST_LAST_FRAME" and len(references) != 2:
        raise CompilerError("First / Last Frame accepts only the two picture references")
    if mode == "R2V" and not references:
        raise CompilerError("R2V requires at least one reference material")
    if mode == "R2V" and audios and not (pictures or videos):
        raise CompilerError("多参考生成不能只使用纯音频，必须同时提供图片或视频")
    video_duration = sum(float((item.get("endSeconds") if item.get("endSeconds") not in {None, ""} else item.get("originalDurationSeconds")) or 0) - float(item.get("startSeconds") or 0) for item in videos)
    if video_duration > 15 + 1e-6:
        raise CompilerError("参考视频实际选段合计不能超过 15 秒")
    audio_duration = sum(float(item.get("originalDurationSeconds") or 0) for item in audios)
    if audio_duration > 15 + 1e-6:
        raise CompilerError("独立音频合计不能超过 15 秒")

    model = MODE_CONFIG[mode]["model"]
    advanced = dict(payload.get("advanced") or {})
    legacy_payload = {"single", "continuous", "renoise", "splitStep", "splitMode", "stage2Seed", "secondSamplingPreset"}
    supplied_legacy = sorted(key for key in legacy_payload if key in payload or key in advanced)
    if supplied_legacy:
        raise CompilerError("旧二采字段已停用：" + ", ".join(supplied_legacy))
    advanced["modelQuant"] = MODEL_QUANT
    raw_reference_fps = advanced.get("reference_video_fps", 2)
    raw_reference_vae_fps = advanced.get("reference_video_vae_fps", 12)
    try:
        reference_fps = int(raw_reference_fps)
        reference_vae_fps = int(raw_reference_vae_fps)
    except (TypeError, ValueError) as exc:
        raise CompilerError("reference FPS must be an integer") from exc
    if not 1 <= reference_fps <= 5:
        raise CompilerError("reference_video_fps must be between 1 and 5")
    if not 1 <= reference_vae_fps <= 24:
        raise CompilerError("reference_video_vae_fps must be between 1 and 24")
    advanced["reference_video_fps"] = reference_fps
    advanced["reference_video_vae_fps"] = reference_vae_fps
    raw_seed = advanced.get("seed")
    if raw_seed in {None, ""}:
        seed = secrets.randbelow(2**63)
    else:
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError) as exc:
            raise CompilerError("seed must be an integer") from exc
        if seed < 0 or seed >= 2**63:
            raise CompilerError("seed must be between 0 and 9223372036854775807")
    advanced["seed"] = seed
    top_level_kernel_present = "requestedKernel" in payload
    nested_kernel_present = "requestedKernel" in advanced
    if top_level_kernel_present and nested_kernel_present:
        raise CompilerError("内核选择只能提交一次，不能在请求与 advanced 中重复。")
    raw_kernel = payload.get("requestedKernel") if top_level_kernel_present else advanced.pop("requestedKernel", None)
    experiment_id = str(payload.get("experimentId") or advanced.pop("experimentId", "")).strip()
    experimental_route = str(payload.get("experimentalRoute") or advanced.pop("experimentalRoute", "")).strip()
    requested_experiment_kernel = str(payload.get("requestedKernel") or raw_kernel or "").strip()
    is_kj_experiment = bool(experiment_id or experimental_route)
    if is_kj_experiment:
        if experiment_id != KJ_EXPERIMENT_ID or experimental_route != KJ_EXPERIMENT_ROUTE:
            raise CompilerError("unknown experimental id or route; fail-closed")
        if requested_experiment_kernel != KJ_EXPERIMENT_KERNEL:
            raise CompilerError("unknown experimental kernel; fail-closed")
        raw_kernel = KJ_EXPERIMENT_KERNEL
    if is_kj_experiment:
        requested_kernel = KJ_EXPERIMENT_KERNEL
    else:
        try:
            requested_kernel = normalize_kernel_backend(raw_kernel)
        except ValueError as exc:
            raise CompilerError(str(exc)) from exc
    requested_steps = DEFAULT_STAGE1_STEPS
    if is_kj_experiment:
        kernel_contract = {
            "contractVersion": KJ_EXPERIMENT_VERSION,
            "routeId": KJ_EXPERIMENT_ROUTE,
            "kernelBackend": KJ_EXPERIMENT_KERNEL,
            "requested": KJ_EXPERIMENT_KERNEL,
            "actual": KJ_EXPERIMENT_KERNEL,
            "patchedBlocks": 50,
            "scope": "denoiser_only",
            "fallback": False,
        }
        kernel_serialized = json.dumps(kernel_contract, sort_keys=True, separators=(",", ":"))
    else:
        kernel_contract = expected_kernel_backend_contract(requested_kernel, requested_steps)
        kernel_serialized = serialize_kernel_backend_contract(requested_kernel, requested_steps)
    kernel_fingerprint = f"sha256:{hashlib.sha256(kernel_serialized.encode('utf-8')).hexdigest()}"
    # H3's official R2V path is BasicGuider with a single conditioning input,
    # not CFG. Preserve every supported advanced option but deliberately drop
    # stale UI/API guidance values so they cannot affect the sampling math.
    advanced.pop("guidance", None)
    forbidden_route_fields = {
        "turboLoraCandidate", "turboLoraStrength", "turboLowVram", "candidate",
        "candidateId", "provider", "providerId", "lora", "loraFile", "loraPath",
        "sampler", "scheduler", "fallbackRoute", "baseModel", "modelPath",
    }
    supplied_forbidden = sorted(key for key in forbidden_route_fields if key in advanced)
    if supplied_forbidden:
        raise CompilerError(
            "route_removed/unsupported_route: forbidden full_quality route fields: "
            + ", ".join(supplied_forbidden)
        )
    requested_acceleration = str(advanced.get("accelerationMode") or "").strip().lower()
    requested_route_id = str(advanced.get("routeId") or "").strip().lower()
    requested_candidate = str(advanced.get("turboLoraCandidate") or "").strip().lower()
    if requested_candidate:
        raise CompilerError(f"route_removed/unsupported_route: {requested_candidate}")
    if requested_route_id and requested_route_id != FULL_QUALITY_ROUTE_NAME:
        raise CompilerError(f"route_removed/unsupported_route: {requested_route_id}")
    if requested_route_id:
        if requested_acceleration and requested_acceleration != FULL_QUALITY_ROUTE_NAME:
            raise CompilerError("route_removed/unsupported_route: routeId and accelerationMode mismatch")
        requested_acceleration = FULL_QUALITY_ROUTE_NAME
    execution_profile = str(
        advanced.get("executionProfile")
        or advanced.get("diagnosticExecutionProfile")
        or DEFAULT_EXECUTION_PROFILE
    )
    if is_kj_experiment:
        execution_profile = KJ_EXPERIMENT_PROFILE
    elif "executionProfile" in advanced and execution_profile != DEFAULT_EXECUTION_PROFILE:
        raise CompilerError(f"route_removed/unsupported_route: {execution_profile}")
    migrated_from = LEGACY_ACCELERATION_PROFILE_MIGRATIONS.get(execution_profile)
    if migrated_from:
        execution_profile = migrated_from
    if requested_acceleration in REMOVED_ACCELERATION_ROUTES:
        raise CompilerError(f"route_removed/unsupported_route: {requested_acceleration}")
    if requested_acceleration and requested_acceleration not in ACCELERATION_MODE_BY_PROFILE:
        raise CompilerError(f"route_removed/unsupported_route: {requested_acceleration}")
    if execution_profile.lower() in REMOVED_ACCELERATION_ROUTES:
        raise CompilerError(f"route_removed/unsupported_route: {execution_profile}")
    if requested_acceleration:
        expected_profile = ACCELERATION_MODE_BY_PROFILE[requested_acceleration]
        if execution_profile not in {DEFAULT_EXECUTION_PROFILE, expected_profile}:
            raise CompilerError("executionProfile and accelerationMode must name the same mutually-exclusive route")
        execution_profile = expected_profile
    if execution_profile not in SUPPORTED_EXECUTION_PROFILES:
        raise CompilerError(
            "executionProfile must be a supported full-quality, conservative acceleration, or historical diagnostic profile"
        )
    advanced["executionProfile"] = execution_profile
    advanced["requestedKernel"] = requested_kernel
    advanced["accelerationMode"] = KJ_EXPERIMENT_ROUTE if is_kj_experiment else ACCELERATION_MODE_BY_PROFILE_REVERSE.get(execution_profile, "full_quality")
    if is_kj_experiment:
        advanced.update({
            "experimentId": KJ_EXPERIMENT_ID,
            "experimentalRoute": KJ_EXPERIMENT_ROUTE,
            "experimentVersion": KJ_EXPERIMENT_VERSION,
            "requestedKernel": KJ_EXPERIMENT_KERNEL,
            "requested": KJ_EXPERIMENT_KERNEL,
            "actual": KJ_EXPERIMENT_KERNEL,
            "fallback": False,
        })
    requested_steps, ffn_chunk, ffn_chunks, memory_strategy = _normalize_tuning(
        advanced, requested_kernel, execution_profile
    )
    requested_steps = int(advanced["stage1Steps"])
    if requested_kernel == "official_native" and ffn_chunks is not None:
        raise CompilerError("official_native does not support explicit FFN chunking; use auto")
    advanced["steps"] = requested_steps
    advanced.pop("splitMode", None)
    advanced.pop("splitStep", None)
    advanced.pop("stage2Seed", None)
    advanced["stage1Steps"] = int(advanced["stage1Steps"])
    advanced["stage2Steps"] = int(advanced["stage2Steps"])
    advanced["stage2Denoise"] = STAGE2_DENOISE
    advanced["ffnChunk"] = ffn_chunk
    advanced["ffnChunks"] = ffn_chunks
    # The backend contract must reflect the normalized W4A8/INT8 FFN choice.
    # It is initially built before tuning normalization for route validation.
    if not is_kj_experiment:
        kernel_contract = expected_kernel_backend_contract(
            requested_kernel, requested_steps, ffn_chunks
        )
        kernel_serialized = serialize_kernel_backend_contract(
            requested_kernel, requested_steps, ffn_chunks
        )
        kernel_fingerprint = f"sha256:{hashlib.sha256(kernel_serialized.encode('utf-8')).hexdigest()}"
    advanced["assetPrecision"] = _normalize_asset_precision(advanced)
    for item in references:
        item["assetPrecision"] = advanced["assetPrecision"]
    advanced["memoryStrategy"] = memory_strategy
    advanced["accelerationRoute"] = dict(ACCELERATION_ROUTE_METADATA[advanced["accelerationMode"]])
    if migrated_from:
        advanced["executionProfileMigratedFrom"] = next(
            legacy for legacy, native in LEGACY_ACCELERATION_PROFILE_MIGRATIONS.items()
            if native == execution_profile
        )
    # Preserve the legacy diagnostic field only for callers that explicitly
    # requested one.  New production requests carry no diagnostic selector.
    if "diagnosticExecutionProfile" in advanced:
        advanced["diagnosticExecutionProfile"] = execution_profile
    reference_planning = plan_reference_videos(
        references,
        timing["frameCount"],
        canvas,
        model,
        advanced,
    )
    picture_ordinal = 0
    for item in references:
        if item["kind"] == "Picture":
            picture_ordinal += 1
            item["preprocessPlan"] = _image_preprocess_plan(mode, item, picture_ordinal, canvas)
        elif item["kind"] == "Video":
            video_ordinal = sum(1 for prior in references[:item["inputIndex"] + 1] if prior["kind"] == "Video")
            item["preprocessPlan"] = reference_planning["videos"][video_ordinal - 1]["officialResize"]
    conditioning_references = pictures + videos + audios
    tokens = " ".join(item["token"] for item in conditioning_references)
    route = algorithm_route_receipt(mode, model, advanced, canvas)
    route_asset_contract = native_route_asset_contract(
        execution_profile,
        model,
        references,
    )
    compiled = {
        "schemaVersion": "h3-local-v1",
        "dryRun": True,
        "mode": mode,
        "modeLabel": MODE_CONFIG[mode]["label"],
        "requestedKernel": requested_kernel,
        "kernelBackendContract": kernel_contract,
        "kernelBackendFingerprint": kernel_fingerprint,
        "prompt": prompt,
        "promptContract": _verbatim_prompt_contract(prompt),
        **({"diagnostics": diagnostic_config} if diagnostic_config is not None else {}),
        "timing": timing,
        "canvas": canvas,
        "algorithmRoute": route,
        "qualityContract": {
            "fps": timing["fps"],
            "modelFps": MODEL_FPS,
            "requestedFps": timing["requestedFps"],
            "temporalDensityFps": timing["temporalDensityFps"],
            "officialModelTimebaseFps": MODEL_FPS,
            "modelFrameCount": timing["modelFrameCount"],
            "videoLatentT": timing["videoLatentT"],
            "audioLatentT": timing["audioLatentT"],
            "timeScale": timing["timeScale"],
            "routeId": route["algorithmRoute"],
            "fallback": False,
            "formal720PBaseline": canvas["resolutionPreset"] == "720p",
            "lowResolutionSpeedup": False,
            "exportSyncFrames": timing["exportFrameCount"],
        },
        "routing": {
            "primaryModel": model,
            "loadedModels": [model],
            "mutuallyExclusivePrimaryModels": ["FL2VA", "REF2VA"],
            "referenceTokens": tokens,
        },
        "routeAssetContract": route_asset_contract,
        "references": references,
        "conditioningOrder": [
            {
                "conditionIndex": index,
                "inputIndex": item["inputIndex"],
                "token": item["token"],
                "kind": item["kind"],
                "name": item["name"],
                "sha256": item["sha256"],
                "pairedEmbeddedAudio": bool(item["kind"] == "Video" and (item.get("mediaFacts") or {}).get("hasAudio")),
            }
            for index, item in enumerate(conditioning_references, 1)
        ],
        "referenceCapacity": {
            "total": MAX_REFERENCES,
            "byKind": dict(REFERENCE_LIMITS),
            "pairedVideoAudioDoesNotConsumeUploadSlot": True,
        },
        "referencePlanning": reference_planning,
        "advanced": advanced,
        "adapter": {
            "name": "DryRunH3Adapter",
            "next": "EmbeddedH3BackendAdapter",
            "realInference": False,
        },
    }
    compiled["executionReceipt"] = initial_execution_receipt(compiled)
    compiled["executionReceipt"]["dryRun"] = True
    return compiled


def build_execution_plan(compiled: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    model = compiled["routing"]["primaryModel"]
    refs = compiled["references"]
    reference_planning = compiled.get("referencePlanning") or {}
    advanced = compiled.get("advanced") or {}
    execution_profile = str(
        advanced.get("executionProfile")
        or advanced.get("diagnosticExecutionProfile")
        or DEFAULT_EXECUTION_PROFILE
    )
    acceleration_mode = str(advanced.get("accelerationMode") or ACCELERATION_MODE_BY_PROFILE_REVERSE.get(execution_profile, "full_quality"))
    native_contract = native_acceleration_route_contract(execution_profile)
    requested_kernel = str(compiled.get("requestedKernel") or "kijai_fast")
    sage = choose_sage_policy(
        compiled["timing"]["frameCount"],
        reference_planning.get("totalBudgetFrames", 0),
        requested="off" if requested_kernel == "official_native" or native_contract else "on",
        threshold=1 if requested_kernel == "kijai_fast" else advanced.get("sageThresholdFrames", 124),
        token_threshold=1 if requested_kernel == "kijai_fast" else advanced.get("sageTokenThresholdTokens", 2048),
    )
    return {
        "taskId": task_id,
        "requestedKernel": compiled["requestedKernel"],
        "kernelBackendContract": dict(compiled["kernelBackendContract"]),
        "kernelBackendFingerprint": compiled["kernelBackendFingerprint"],
        "dryRun": True,
        "status": "ready",
        "execution": {
            "model": model,
            "modelQuant": str(advanced.get("modelQuant") or "int8").strip() or "int8",
            "steps": int(advanced.get("steps") or 20),
            "memoryStrategy": str(advanced.get("memoryStrategy") or "auto"),
        },
        "executionReceipt": initial_execution_receipt(compiled),
        "modelExclusion": {
            "loadedPrimaryModel": model,
            "notLoaded": "REF2VA" if model == "FL2VA" else "FL2VA",
            "simultaneousPrimaryModels": False,
        },
        "referencePlanning": reference_planning,
        "performance": {
            "sage": sage,
            "executionProfile": execution_profile,
            "accelerationMode": acceleration_mode,
            "acceleration": {
                "requestedMode": acceleration_mode,
                "actualMode": "full_quality" if acceleration_mode == "full_quality" else "pending_preflight",
                "approximate": acceleration_mode != "full_quality",
                "status": "pending",
                "fallbackReason": None,
            },
            "executionRouteContract": native_contract,
            "diagnosticExecutionProfile": execution_profile if execution_profile in DIAGNOSTIC_EXECUTION_PROFILES else None,
            "referenceFrames": reference_planning.get("totalBudgetFrames", 0),
            "referencePolicyPreset": reference_planning.get("policyPreset"),
            "referencePolicyOverridden": reference_planning.get("policyOverridden", False),
        },
        "phases": [
            {"id": "reference_preprocess", "status": "ready", "detail": "stream 24fps legal frames and 2fps semantic samples"},
            {"id": "reference_encode", "status": "deferred", "detail": "reference VAE/Qwen condition; metadata cache only"},
            {"id": "text_unload", "status": "deferred", "detail": "unload Qwen before primary sampling"},
            {"id": "primary_sample", "status": "deferred", "detail": f"load {model} only; Sage status is explicit in performance.sage"},
            {"id": "decode_export", "status": "deferred", "detail": f"24fps export sync: {compiled['timing']['exportFrameCount']} frames / {compiled['timing']['exportDurationSeconds']}s"},
        ],
        "steps": [
            {"id": "validate", "status": "ready", "detail": "request validated"},
            {"id": "prepare_canvas", "status": "ready", "detail": compiled["canvas"]["baseline"]},
            {"id": "prepare_references", "status": "ready", "detail": f"{len(refs)} references; tokens: {compiled['routing']['referenceTokens'] or 'none'}; video budget: {compiled['referencePlanning']['totalBudgetFrames']} legal frames"},
            {"id": "load_primary_model", "status": "deferred", "detail": f"direct backend loads {model} only"},
            {"id": "inference", "status": "deferred", "detail": "dry-run does not execute GPU sampling"},
            {"id": "export_result", "status": "deferred", "detail": "dry-run manifest returned; no video was generated"},
        ],
        "compiled": compiled,
    }
