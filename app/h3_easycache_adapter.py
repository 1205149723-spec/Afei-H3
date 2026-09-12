"""Pinned upstream ComfyUI EasyCache node integration for H3."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from h3_compiler import validate_native_route_asset_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PROJECT_ROOT / "runtime" / "ComfyUI"
COMFY_COMMIT = "14b05228cef127ce529bc0c08660770d4af3e9a8"
EASYCACHE_CONSERVATIVE_CONFIG = {"reuse_threshold": 0.20, "start_percent": 0.15, "end_percent": 0.95, "verbose": False}


def inspect_h3_easycache() -> Dict[str, Any]:
    source = COMFY_ROOT / "comfy_extras" / "nodes_easycache.py"
    return {
        "requestedSource": "Comfy-Org/ComfyUI EasyCache", "sourceCommit": COMFY_COMMIT,
        "sourcePath": str(source), "approximate": True, "experimental": True,
        "candidateConfig": dict(EASYCACHE_CONSERVATIVE_CONFIG), "available": source.is_file(),
        "compatibilityFailure": None if source.is_file() else "local ComfyUI EasyCache source is missing",
    }


def _new_telemetry() -> Dict[str, Any]:
    return {"candidateChecks": 0, "eligibleChecks": 0, "eligibleSteps": 0, "hitSteps": 0,
            "skippedSteps": 0, "nativeForwardCalls": 0, "lastDecision": "not_observed",
            "finalized": False, "cacheDiffUpdateCalls": 0, "cacheDiffApplyCalls": 0,
            "cacheDiffUpdateSecondsTotal": 0.0, "cacheDiffApplySecondsTotal": 0.0}


def _observed_holder(upstream_holder: Any, telemetry: Dict[str, Any]) -> Any:
    """Add scalar receipts only; every cache decision delegates to upstream."""
    from comfy_extras.nodes_easycache import EasyCacheHolder

    class ObservedEasyCacheHolder(EasyCacheHolder):
        def clone(self):
            return ObservedEasyCacheHolder(self.reuse_threshold, self.start_percent, self.end_percent,
                                           self.subsample_factor, self.offload_cache_diff, self.verbose,
                                           self.output_channels, telemetry)

        def __init__(self, reuse_threshold, start_percent, end_percent, subsample_factor, offload_cache_diff, verbose, output_channels, telemetry_ref):
            super().__init__(reuse_threshold, start_percent, end_percent, subsample_factor, offload_cache_diff, verbose, output_channels)
            self._h3_telemetry = telemetry_ref

        def should_do_easycache(self, timestep):
            result = super().should_do_easycache(timestep)
            self._h3_telemetry["lastDecision"] = "candidate" if result else "outside_configured_window"
            self._h3_telemetry["candidateChecks"] += int(bool(result))
            return result

        def can_apply_cache_diff(self, uuids):
            self._h3_telemetry["eligibleChecks"] += 1
            result = super().can_apply_cache_diff(uuids)
            self._h3_telemetry["eligibleSteps"] += int(bool(result))
            return result

        def apply_cache_diff(self, x, uuids, is_audio=False):
            started = time.perf_counter()
            try:
                return super().apply_cache_diff(x, uuids, is_audio=is_audio)
            finally:
                if not is_audio:
                    self._h3_telemetry["hitSteps"] += 1
                    self._h3_telemetry["skippedSteps"] += 1
                    self._h3_telemetry["lastDecision"] = "cache_hit"
                self._h3_telemetry["cacheDiffApplyCalls"] += 1
                self._h3_telemetry["cacheDiffApplySecondsTotal"] += time.perf_counter() - started

        def update_cache_diff(self, *args, **kwargs):
            started = time.perf_counter()
            try:
                return super().update_cache_diff(*args, **kwargs)
            finally:
                self._h3_telemetry["cacheDiffUpdateCalls"] += 1
                self._h3_telemetry["cacheDiffUpdateSecondsTotal"] += time.perf_counter() - started

        def reset(self):
            self._h3_telemetry["skippedSteps"] = int(getattr(self, "total_steps_skipped", 0) or 0)
            self._h3_telemetry["finalized"] = True
            return super().reset()

    return ObservedEasyCacheHolder(upstream_holder.reuse_threshold, upstream_holder.start_percent,
                                   upstream_holder.end_percent, upstream_holder.subsample_factor,
                                   upstream_holder.offload_cache_diff, upstream_holder.verbose,
                                   upstream_holder.output_channels, telemetry)


def apply_h3_easycache_conservative(model_patcher: Any, asset_contract: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    receipt = inspect_h3_easycache()
    receipt["routeAssetContract"] = validate_native_route_asset_contract("easycache_native", asset_contract)
    if not receipt["available"]:
        raise RuntimeError(str(receipt["compatibilityFailure"]))
    from comfy_extras.nodes_easycache import EasyCacheNode
    patched = EasyCacheNode.execute(model_patcher, **EASYCACHE_CONSERVATIVE_CONFIG)[0]
    options = patched.model_options.setdefault("transformer_options", {})
    telemetry = _new_telemetry()
    # The node, wrappers and decision algorithm are upstream. This replacement
    # only preserves scalar counters across the upstream holder.clone lifecycle.
    options["easycache"] = _observed_holder(options["easycache"], telemetry)
    options["h3_easycache_telemetry"] = telemetry
    receipt.update({"applied": True, "upstreamEntryPoint": "comfy_extras.nodes_easycache.EasyCacheNode.execute",
                    "internalPatch": {"scope": "upstream_node_with_scalar_receipt_observer"},
                    "patchOrder": ["MiniMaxH3SigmaShift", "EasyCacheNode.execute", "BasicGuider", "SamplerCustomAdvanced"],
                    "excludedProjectOptimizations": ["kjH3Sage", "ffnChunk", "projectDynamicVramInjection"],
                    "fallbackPolicy": "fail_no_fallback_to_full_quality"})
    return patched, receipt


def h3_easycache_runtime_receipt(model_patcher: Any, total_steps: int, completed_sampling_steps: int) -> Dict[str, Any]:
    options = getattr(model_patcher, "model_options", {}) or {}
    state = dict((options.get("transformer_options", {}) or {}).get("h3_easycache_telemetry", {}) or {})
    completed, requested = int(completed_sampling_steps or 0), int(total_steps or 0)
    finalized = bool(state.get("finalized")) and requested > 0 and completed >= requested
    skipped = int(state.get("skippedSteps", 0) or 0)
    return {"executionState": "completed" if finalized else "partial_or_cancelled",
            "completedSamplingSteps": completed, "requestedSamplingSteps": requested,
            "nativeSteps": max(0, completed - skipped), "skippedSteps": skipped,
            "candidateChecks": int(state.get("candidateChecks", 0) or 0),
            "eligibleChecks": int(state.get("eligibleChecks", 0) or 0),
            "eligibleSteps": int(state.get("eligibleSteps", 0) or 0),
            "hitSteps": int(state.get("hitSteps", 0) or 0), "skippedBlocks": "unavailable_step-cache",
            "rejectReasons": dict(state.get("rejectReasons", {}) or {}),
            "actualTransformerForwards": max(0, completed - skipped),
            "cacheDiffUpdateCalls": int(state.get("cacheDiffUpdateCalls", 0) or 0),
            "cacheDiffUpdateSecondsTotal": float(state.get("cacheDiffUpdateSecondsTotal", 0.0) or 0.0),
            "cacheDiffApplyCalls": int(state.get("cacheDiffApplyCalls", 0) or 0),
            "cacheDiffApplySecondsTotal": float(state.get("cacheDiffApplySecondsTotal", 0.0) or 0.0),
            "lastDecision": str(state.get("lastDecision") or "not_observed"),
            "hitExecutionContract": "an upstream EasyCache hit returns apply_cache_diff before the diffusion executor; the corresponding solver step runs zero H3 blocks",
            "fallbackReason": None}
