"""Pinned upstream TE-Speed public-node integration for MiniMax H3."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from h3_compiler import validate_native_route_asset_contract

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TE_ROOT = PROJECT_ROOT / "runtime" / "third_party" / "TE-Speed-MiniMaxH3-OSS-c1dacf47"
TE_COMMIT = "c1dacf47bc02cb9326f7b93c69280529b93d391b"
TE_UPSTREAM_CONFIG = {
    "processing_control_value": 0.12,
    "processing_percent_1": 0.10,
    "processing_percent_2": 0.90,
    "mcs": 2,
    "device": "auto",
    "cache_depth": 0.75,
}


def _find_minimax_h3(model_patcher: Any) -> Any:
    model = getattr(model_patcher, "model", None)
    for _ in range(12):
        if model is None:
            return None
        if type(model).__name__ == "MiniMaxH3Model":
            return model
        model = next((getattr(model, name, None) for name in ("model", "inner_model", "diffusion_model", "unet_model") if getattr(model, name, None) is not None), None)
    return None


def inspect_h3_te_speed(model_patcher: Any = None) -> Dict[str, Any]:
    source = TE_ROOT / "nodes.py"
    native_source = PROJECT_ROOT / "runtime" / "ComfyUI" / "comfy" / "ldm" / "minimax" / "model.py"
    hook_source = native_source.read_text(encoding="utf-8") if native_source.is_file() else ""
    has_hooks = "def _run_blocks(" in hook_source and '("block_loop", 0) in blocks_replace' in hook_source
    inner = _find_minimax_h3(model_patcher) if model_patcher is not None else None
    return {
        "requestedSource": "HELPMEEADICE/TE-Speed-MiniMaxH3-OSS", "sourceCommit": TE_COMMIT,
        "sourcePath": str(TE_ROOT), "approximate": True, "candidateConfig": dict(TE_UPSTREAM_CONFIG),
        "available": bool(source.is_file() and has_hooks),
        "requiredInternalHooks": ["MiniMaxH3Model._run_blocks", '("block_loop", 0)'],
        "modelResolved": inner is not None,
        "compatibilityFailure": None if source.is_file() and has_hooks else "pinned TE-Speed source or its required upstream model hooks are missing",
    }


def _load_upstream_module() -> Any:
    source = TE_ROOT / "nodes.py"
    spec = importlib.util.spec_from_file_location("h3_te_speed_upstream", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned TE-Speed source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_h3_te_speed_conservative(model_patcher: Any, asset_contract: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    """Apply the author-provided TESpeedMiniMaxH3.patch entry point exactly."""
    receipt = inspect_h3_te_speed(model_patcher)
    receipt["routeAssetContract"] = validate_native_route_asset_contract("te_speed_native", asset_contract)
    if not receipt["available"]:
        raise RuntimeError(str(receipt["compatibilityFailure"]))
    upstream = _load_upstream_module()
    patched = upstream.TESpeedMiniMaxH3().patch(model_patcher, **TE_UPSTREAM_CONFIG)[0]
    options = patched.model_options.setdefault("transformer_options", {})
    patches = (options.get("patches_replace", {}) or {}).get("dit", {}) or {}
    cache = patches.get(("block_loop", 0))
    if cache is None:
        raise RuntimeError("upstream TE-Speed node did not install its block_loop patch")
    options["h3_te_speed_state"] = cache
    receipt.update({
        "applied": True,
        "upstreamEntryPoint": "TESpeedMiniMaxH3.patch",
        "internalPatch": {"scope": "upstream_task_private_modelpatcher_clone", "hook": '("block_loop", 0)'},
        "patchOrder": ["MiniMaxH3SigmaShift", "TESpeedMiniMaxH3.patch", "BasicGuider", "SamplerCustomAdvanced"],
        "excludedProjectOptimizations": ["kjH3Sage", "ffnChunk", "projectDynamicVramInjection"],
        "fallbackPolicy": "fail_no_fallback_to_full_quality",
    })
    return patched, receipt


def h3_te_speed_runtime_receipt(model_patcher: Any, total_steps: int, completed_sampling_steps: int) -> Dict[str, Any]:
    """Scalar state reported by the upstream TE-Speed cache instance."""
    options = getattr(model_patcher, "model_options", {}) or {}
    cache = (options.get("transformer_options", {}) or {}).get("h3_te_speed_state")
    completed, requested = int(completed_sampling_steps or 0), int(total_steps or 0)
    if cache is None:
        return {"executionState": "unavailable", "completedSamplingSteps": completed, "requestedSamplingSteps": requested, "fallbackReason": "upstream TE-Speed state was not installed"}
    full = int(getattr(cache, "full_steps", 0) or 0)
    hits = int(getattr(cache, "cache_hits", 0) or 0)
    skipped = int(getattr(cache, "skipped_blocks", 0) or 0)
    total_blocks = int(getattr(cache, "total_blocks", 0) or 0)
    return {
        "executionState": "completed" if requested > 0 and completed >= requested else "partial_or_cancelled",
        "completedSamplingSteps": completed, "requestedSamplingSteps": requested,
        "actualTransformerForwards": full, "cacheHitSteps": hits, "skippedSteps": hits,
        "skippedBlocks": skipped, "executedBlocks": max(0, total_blocks - skipped),
        "totalBlockVisits": total_blocks, "lastMode": getattr(cache, "last_mode", "unavailable"),
        "warmBlocks": getattr(cache, "_warm_blocks", lambda _count: "unavailable")(50),
        "cacheExecutionContract": "upstream cache steps recompute only warm blocks and add the upstream residual; skippedBlocks must be positive before this route is effective",
        "fallbackReason": None,
    }
