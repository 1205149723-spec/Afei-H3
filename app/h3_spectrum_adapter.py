"""Pinned upstream Spectrum node integration for MiniMax H3."""

from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Dict

from h3_compiler import CompilerError, validate_native_route_asset_contract


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECTRUM_ROOT = PROJECT_ROOT / "runtime" / "third_party" / "ComfyUI-Spectrum-MiniMax-H3-8bfc235"
SPECTRUM_COMMIT = "8bfc235cb3910c73964277e0316ec875f4b2c011"
SPECTRUM_LICENSE = SPECTRUM_ROOT / "LICENSE"
SPECTRUM_REQUIRED_COMFY_COMMIT = "e377e263"
LOCAL_PINNED_COMFY_COMMIT = "14b05228"
_SUPPORTED_PRIMARY_MODELS = frozenset({"FL2VA", "REF2VA"})
_DISABLED_PROJECT_OPTIMIZATIONS = {
    "kjH3Sage": "disabled",
    "ffnChunk": "disabled",
    "projectDynamicVramInjection": "disabled",
}

# Pinned upstream node defaults. No H3-specific tuning is applied here.
SPECTRUM_B_CONFIG = {
    "enabled": True,
    "blend_weight": 0.50,
    "degree": 4,
    "ridge_lambda": 0.10,
    "window_size": 2.0,
    "flex_window": 0.75,
    "warmup_steps": 5,
    "tail_actual_steps": 1,
    "max_history": 8,
    "history_storage": "system_ram",
    "debug": False,
}


class SpectrumPreflightError(RuntimeError):
    """A Spectrum contract rejection whose receipt is safe to persist on the task."""

    def __init__(self, message: str, receipt: Dict[str, Any]):
        super().__init__(message)
        self.receipt = receipt


def _add_spectrum_to_import_path() -> None:
    source = str(SPECTRUM_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)


def _preflight_failure(receipt: Dict[str, Any], status: str, check: str, message: str) -> None:
    receipt["status"] = status
    receipt["available"] = False
    receipt["fallback"] = False
    receipt.setdefault("failedChecks", []).append(check)
    receipt["compatibilityFailure"] = message
    raise SpectrumPreflightError(message, receipt)


def preflight_h3_spectrum(
    primary_model: str,
    algorithm_route: Dict[str, Any],
    route_contract: Dict[str, Any],
    asset_contract: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Verify the public Spectrum-to-embedded-runtime seam without loading a model.

    This is deliberately source/API inspection only.  It neither starts ComfyUI nor
    creates CUDA tensors, and it records the one audited host-memory difference
    between the author's required ComfyUI revision and this project's pinned tree.
    """

    primary_model = str(primary_model or "")
    route_contract = dict(route_contract or {})
    algorithm_route = dict(algorithm_route or {})
    receipt: Dict[str, Any] = {
        "requestedSource": "xmarre/ComfyUI-Spectrum-MiniMax-H3",
        "sourceCommit": SPECTRUM_COMMIT,
        "sourcePath": str(SPECTRUM_ROOT),
        "license": "GPL-3.0-or-later",
        "licensePath": str(SPECTRUM_LICENSE),
        "authorRequiredComfyCommit": SPECTRUM_REQUIRED_COMFY_COMMIT,
        "localPinnedComfyCommit": LOCAL_PINNED_COMFY_COMMIT,
        "authorVersionStatus": "pending",
        "adapterEntry": "comfyui_spectrum_h3.nodes.SpectrumApplyMiniMaxH3.apply",
        "startsComfyUiService": False,
        "usesSecondRuntime": False,
        "fallback": False,
        "available": False,
        "status": "pending_preflight",
        "failedChecks": [],
        "modelSupport": {
            "primaryModel": primary_model,
            "status": "static_contract_verified" if primary_model in _SUPPORTED_PRIMARY_MODELS else "unsupported",
            "runtimeSmoke": "required",
        },
        "routeAssetContract": dict(asset_contract or {}),
        "modelManagementPinnedMemoryDifference": {
            "authorCommit": SPECTRUM_REQUIRED_COMFY_COMMIT,
            "localCommit": LOCAL_PINNED_COMFY_COMMIT,
            "file": "comfy/model_management.py",
            "difference": "pinned host-memory policy",
            "compatibleApi": True,
            "runtimeDecision": "record_only_no_memory_policy_override",
        },
    }
    if primary_model not in _SUPPORTED_PRIMARY_MODELS:
        _preflight_failure(receipt, "blocked", "primaryModel", f"Spectrum has no static contract for primary model {primary_model!r}")

    try:
        receipt["routeAssetContract"] = validate_native_route_asset_contract("spectrum_native", asset_contract or {})
    except CompilerError as exc:
        _preflight_failure(receipt, "blocked", "assetContract", str(exc))

    required_route = "spectrum_native"
    observed_route = {
        "algorithmRoute": algorithm_route.get("algorithmRoute"),
        "algorithmProfile": algorithm_route.get("algorithmProfile"),
        "contractRouteId": route_contract.get("routeId"),
    }
    if any(value != required_route for value in observed_route.values()):
        _preflight_failure(receipt, "failed_preflight", "routeIdentity", f"Spectrum route contract mismatch: {observed_route}")
    receipt["routeIdentity"] = {
        **observed_route,
        "algorithmRouteFingerprint": algorithm_route.get("algorithmRouteFingerprint"),
    }
    observed_optimizations = route_contract.get("projectOptimizations") or {}
    if any(observed_optimizations.get(key) != value for key, value in _DISABLED_PROJECT_OPTIMIZATIONS.items()):
        _preflight_failure(
            receipt,
            "failed_preflight",
            "projectOptimizations",
            "Spectrum rejects KJ Sage, FFN chunking, or project DynamicVRAM injection",
        )
    receipt["projectOptimizationIsolation"] = dict(_DISABLED_PROJECT_OPTIMIZATIONS)

    if LOCAL_PINNED_COMFY_COMMIT != SPECTRUM_REQUIRED_COMFY_COMMIT:
        receipt["authorVersionStatus"] = "blocked_exact_commit_required"
        _preflight_failure(
            receipt,
            "blocked",
            "authorComfyCommit",
            "Spectrum 作者只验证其指定的 ComfyUI 提交；本地 pinned 提交不同，不能声称原生兼容。",
        )
    receipt["authorVersionStatus"] = "exact_commit"

    if not SPECTRUM_ROOT.joinpath("comfyui_spectrum_h3").is_dir() or not SPECTRUM_LICENSE.is_file():
        _preflight_failure(receipt, "failed_preflight", "pinnedSource", "pinned Spectrum source or license is missing")

    try:
        _add_spectrum_to_import_path()
        minimax_model = importlib.import_module("comfy.ldm.minimax.model")
        samplers = importlib.import_module("comfy.samplers")
        patcher_extension = importlib.import_module("comfy.patcher_extension")
        model_patcher = importlib.import_module("comfy.model_patcher")
        sampling = importlib.import_module("comfyui_spectrum_h3.sampling")
        minimax = importlib.import_module("comfyui_spectrum_h3.minimax_h3")
        nodes = importlib.import_module("comfyui_spectrum_h3.nodes")
        # ComfyUI exposes the public sampler boundary on CFGGuider; KSampler
        # only owns the sampler-function adapter in the pinned runtime.
        outer_sample = getattr(samplers.CFGGuider, "outer_sample")
        model_patcher_type = getattr(model_patcher, "ModelPatcher")
        wrapper_slots = ("OUTER_SAMPLE", "PREDICT_NOISE", "DIFFUSION_MODEL")
        required_patcher_api = ("get_wrappers", "add_wrapper_with_key", "get_callbacks", "add_callback_with_key")
        interface_checks = {
            "miniMaxH3Model": inspect.isclass(getattr(minimax_model, "MiniMaxH3Model", None)),
            "packedLayout": inspect.isclass(getattr(minimax_model, "PackedLayout", None)),
            "outerSampleAcceptsLatentShapes": "latent_shapes" in inspect.signature(outer_sample).parameters,
            "wrapperCallbackApi": all(callable(getattr(model_patcher_type, name, None)) for name in required_patcher_api),
            "wrapperSlots": all(hasattr(patcher_extension.WrappersMP, name) for name in wrapper_slots),
            "cloneCallback": hasattr(patcher_extension.CallbacksMP, "ON_CLONE"),
            "upstreamNode": callable(getattr(getattr(nodes, "SpectrumApplyMiniMaxH3"), "apply", None)),
            "upstreamInstallers": callable(getattr(sampling, "install_sampler_wrappers", None)) and callable(getattr(minimax, "install_h3_wrapper", None)),
        }
        supported_samplers = sorted(str(name).removeprefix("sample_") for name in sampling.SUPPORTED_SINGLE_CALL_SAMPLERS)
    except Exception as exc:
        _preflight_failure(receipt, "failed_preflight", "embeddedRuntimeImports", f"{type(exc).__name__}: {exc}")

    receipt["interfaceChecks"] = interface_checks
    receipt["supportedSamplers"] = supported_samplers
    if not all(interface_checks.values()):
        failed = [name for name, passed in interface_checks.items() if not passed]
        _preflight_failure(receipt, "failed_preflight", "embeddedRuntimeApi", f"Spectrum embedded-runtime API checks failed: {failed}")
    if "res_multistep" not in supported_samplers:
        _preflight_failure(receipt, "failed_preflight", "sampler", "Spectrum upstream does not expose res_multistep")
    receipt["available"] = True
    receipt["status"] = "static_ready"
    return receipt


def _new_boundary_telemetry() -> Dict[str, Any]:
    """Scalar-only adapter timings; never retain tensors or synchronize CUDA."""

    return {
        "instrumentation": "adapter_boundary_perf_counter",
        "actualArchiveCalls": 0,
        "actualArchiveSecondsTotal": 0.0,
        "actualArchiveSecondsMax": 0.0,
        "forecastCalls": 0,
        "forecastSecondsTotal": 0.0,
        "forecastSecondsMax": 0.0,
        "finalized": False,
        "partial": True,
    }


def _record_boundary(telemetry: Dict[str, Any], prefix: str, elapsed: float) -> None:
    telemetry[f"{prefix}Calls"] = int(telemetry.get(f"{prefix}Calls", 0) or 0) + 1
    telemetry[f"{prefix}SecondsTotal"] = float(telemetry.get(f"{prefix}SecondsTotal", 0.0) or 0.0) + elapsed
    telemetry[f"{prefix}SecondsMax"] = max(float(telemetry.get(f"{prefix}SecondsMax", 0.0) or 0.0), elapsed)


def _install_runtime_boundary_timing(runtime: Any, telemetry: Dict[str, Any]) -> None:
    """Time only upstream boundary calls, once per task-local runtime instance."""

    if getattr(runtime, "_h3_adapter_boundary_timing_installed", False):
        return
    for method_name, prefix in (("observe_actual", "actualArchive"), ("predict", "forecast")):
        original = getattr(runtime, method_name, None)
        if not callable(original):
            continue

        def timed(*args: Any, __original: Any = original, __prefix: str = prefix, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return __original(*args, **kwargs)
            finally:
                _record_boundary(telemetry, __prefix, time.perf_counter() - started)

        setattr(runtime, method_name, timed)
    setattr(runtime, "_h3_adapter_boundary_timing_installed", True)


def _install_spectrum_boundary_timing(model_patcher: Any) -> Dict[str, Any]:
    options = getattr(model_patcher, "model_options", {}) or {}
    telemetry = options.setdefault("h3_spectrum_adapter_telemetry", _new_boundary_telemetry())
    try:
        sampling = importlib.import_module("comfyui_spectrum_h3.sampling")
        binding = options.get(sampling.BINDING_KEY)
        runtime = getattr(binding, "runtime", None)
        if runtime is None:
            telemetry["instrumentationStatus"] = "unavailable_no_runtime_binding"
        else:
            _install_runtime_boundary_timing(runtime, telemetry)
            telemetry["instrumentationStatus"] = "installed"
    except Exception as exc:
        telemetry["instrumentationStatus"] = f"unavailable_{type(exc).__name__}"
    return telemetry


def h3_spectrum_runtime_receipt(model_patcher: Any, completed_sampling_steps: int, total_steps: int) -> Dict[str, Any]:
    """Return a scalar snapshot that remains honest for a cancelled partial run."""

    options = getattr(model_patcher, "model_options", {}) or {}
    state = dict(options.get("h3_spectrum_adapter_telemetry", {}) or {})
    stats: Any = None
    try:
        sampling = importlib.import_module("comfyui_spectrum_h3.sampling")
        binding = options.get(sampling.BINDING_KEY)
        stats = getattr(getattr(binding, "runtime", None), "stats", None)
    except Exception:
        stats = None
    completed = int(completed_sampling_steps or 0)
    total = int(total_steps or 0)
    finalized = total > 0 and completed >= total
    actual_steps = getattr(stats, "actual_steps", state.get("actualSteps", "unavailable" if finalized else "unavailable_partial_run"))
    forecast_steps = getattr(stats, "forecast_steps", state.get("forecastSteps", state.get("skippedSteps", "unavailable" if finalized else "unavailable_partial_run")))
    actual_forwards = getattr(stats, "actual_transformer_calls", state.get("actualTransformerForwards", "unavailable" if finalized else "unavailable_partial_run"))
    forecast_calls = getattr(stats, "forecast_model_calls", state.get("forecastModelCalls", "unavailable" if finalized else "unavailable_partial_run"))
    fallbacks = getattr(stats, "forecast_fallbacks", state.get("forecastFallbacks", "unavailable" if finalized else "unavailable_partial_run"))
    return {
        "executionState": "completed" if finalized else "partial_or_cancelled",
        "instrumentation": state.get("instrumentation", "unavailable"),
        "instrumentationStatus": state.get("instrumentationStatus", "unavailable"),
        "actualArchiveCalls": int(state.get("actualArchiveCalls", 0) or 0),
        "actualArchiveSecondsTotal": float(state.get("actualArchiveSecondsTotal", 0.0) or 0.0),
        "actualArchiveSecondsMax": float(state.get("actualArchiveSecondsMax", 0.0) or 0.0),
        "forecastCalls": int(state.get("forecastCalls", 0) or 0),
        "forecastSecondsTotal": float(state.get("forecastSecondsTotal", 0.0) or 0.0),
        "forecastSecondsMax": float(state.get("forecastSecondsMax", 0.0) or 0.0),
        "completedSamplingSteps": completed,
        "requestedSamplingSteps": total,
        "actualSteps": actual_steps,
        "actualTransformerForwards": actual_forwards,
        "forecastModelCalls": forecast_calls,
        "forecastFallbacks": fallbacks,
        "skippedSteps": forecast_steps,
        "forecastSteps": forecast_steps,
        "expectedRes20Plan": {"nativeSteps": 14, "forecastSteps": 6},
        "forecastExecutionContract": "forecast steps use upstream FinalLayer AV reconstruction and must execute zero H3 transformer blocks",
        "fallbackReason": getattr(stats, "disable_reason", state.get("fallbackReason")),
    }


def inspect_h3_spectrum() -> Dict[str, Any]:
    """Report the pinned candidate without claiming it is lossless."""

    package = SPECTRUM_ROOT / "comfyui_spectrum_h3"
    receipt: Dict[str, Any] = {
        "requestedSource": "xmarre/ComfyUI-Spectrum-MiniMax-H3",
        "sourceCommit": SPECTRUM_COMMIT,
        "sourcePath": str(SPECTRUM_ROOT),
        "license": "GPL-3.0-or-later",
        "licensePath": str(SPECTRUM_LICENSE),
        "packagePresent": package.is_dir(),
        "licensePresent": SPECTRUM_LICENSE.is_file(),
        "approximate": True,
        "productionDefault": False,
        "supportedSampler": "res_multistep",
        "candidateConfig": dict(SPECTRUM_B_CONFIG),
        "available": False,
        "compatibilityFailure": None,
    }
    if not receipt["packagePresent"]:
        receipt["compatibilityFailure"] = f"pinned Spectrum source is missing: {package}"
        return receipt
    _add_spectrum_to_import_path()
    try:
        from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3

        receipt["nodeClass"] = SpectrumApplyMiniMaxH3.__name__
        receipt["available"] = True
    except Exception as exc:
        receipt["compatibilityFailure"] = f"{type(exc).__name__}: {exc}"
    return receipt


def apply_h3_spectrum_b_candidate(
    model_patcher: Any,
    *,
    primary_model: str,
    algorithm_route: Dict[str, Any],
    route_contract: Dict[str, Any],
    asset_contract: Dict[str, Any],
) -> tuple[Any, Dict[str, Any]]:
    """Apply the upstream public Spectrum node with its upstream defaults."""

    receipt = preflight_h3_spectrum(primary_model, algorithm_route, route_contract, asset_contract)
    # Do not route through SpectrumApplyMiniMaxH3 (a node/UI façade).  These
    # are the original author's actual ModelPatcher-level installers.
    nodes = importlib.import_module("comfyui_spectrum_h3.nodes")
    patched = nodes.SpectrumApplyMiniMaxH3().apply(model_patcher, **SPECTRUM_B_CONFIG)[0]
    telemetry = _install_spectrum_boundary_timing(patched)
    receipt.update({
        "applied": True,
        "upstreamEntryPoint": "comfyui_spectrum_h3.nodes.SpectrumApplyMiniMaxH3.apply",
        "internalPatch": {"scope": "upstream_task_private_modelpatcher_clone"},
        "patchOrder": ["MiniMaxH3SigmaShift", "SpectrumApplyMiniMaxH3.apply", "BasicGuider", "SamplerCustomAdvanced"],
        "excludedProjectOptimizations": ["kjH3Sage", "ffnChunk", "projectDynamicVramInjection"],
        "fallbackPolicy": "fail_no_fallback_to_full_quality",
        "semanticWarning": "approximate transformer-feature forecasting; requires user A/B review",
        "boundaryTelemetry": dict(telemetry),
        "actualHook": {
            "samplerWrappers": ["OUTER_SAMPLE", "PREDICT_NOISE"],
            "modelWrapper": "DIFFUSION_MODEL",
            "cloneCallback": "ON_CLONE",
        },
        "predictedSteps": "reported after sampling",
        "actualSteps": "reported after sampling",
    })
    return patched, receipt
