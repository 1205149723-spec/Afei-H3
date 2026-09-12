"""Diagnostic-only H3 TeaCache research adapter.

This module is deliberately outside the product path.  It keeps the native
H3 BasicGuider / res_multistep contract intact and adds an opt-in wrapper at
the block-sequence boundary for controlled research.  H3 is not one of the
architectures with published TeaCache calibration coefficients, so this code
must never silently turn a measured trace into a production cache policy.

Unlike the rejected v0.1 experiment, the probe below is not raw ``t_emb`` and
the cache is not a prior denoiser output.  It samples the same *first-block
AdaLN-modulated attention input* that H3 actually feeds to its Transformer.
The final layer and the current timestep remain native on every step.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE = "ali-vilab/TeaCache"
OFFICIAL_LICENSE = "Apache-2.0"


def _relative_l1(current: Any, previous: Any) -> float:
    """Compute a tiny CPU-side relative-L1 comparison."""

    if current is None or previous is None or tuple(current.shape) != tuple(previous.shape):
        return float("inf")
    numerator = (current - previous).abs().mean()
    denominator = previous.abs().mean().clamp(min=1e-8)
    return float((numerator / denominator).item())


def _sample_indices(length: int, count: int, device: Any) -> Any:
    torch = __import__("torch")
    return torch.linspace(0, max(0, length - 1), steps=min(count, length), device=device).long()


def _small_probe(tensor: Any) -> Any:
    """Capture a deterministic 8x16 BF16/FP feature sample as CPU FP32."""

    torch = __import__("torch")
    rows = _sample_indices(int(tensor.shape[0]), 8, tensor.device)
    cols = _sample_indices(int(tensor.shape[1]), 16, tensor.device)
    return tensor.index_select(0, rows).index_select(1, cols).detach().to(
        device="cpu", dtype=torch.float32
    )


def _finite_values(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0.0 and math.isfinite(number):
            result.append(number)
    return result


def _fit_quadratic(xs: list[float], ys: list[float]) -> Optional[Dict[str, Any]]:
    """Fit/report a trace descriptor; it never enables reuse by itself."""

    if len(xs) < 6 or len(xs) != len(ys):
        return None
    try:
        import numpy as np

        coefficients = np.polyfit(np.asarray(xs), np.asarray(ys), deg=2)
        predicted = np.polyval(coefficients, np.asarray(xs))
        observed = np.asarray(ys)
        residual = observed - predicted
        ss_total = float(((observed - observed.mean()) ** 2).sum())
        r_squared = None if ss_total <= 1e-18 else round(1.0 - float((residual ** 2).sum()) / ss_total, 8)
        return {
            "kind": "quadratic_modulated_input_to_block_residual_transition",
            "coefficients": [round(float(value), 12) for value in coefficients.tolist()],
            "rmse": round(float(np.sqrt((residual ** 2).mean())), 12),
            "rSquared": r_squared,
            "sampleCount": len(xs),
        }
    except Exception as exc:  # diagnostics must not alter a real model path for report fitting.
        return {"kind": "unavailable", "reason": f"fit_failed:{type(exc).__name__}", "sampleCount": len(xs)}


def derive_h3_teacache_b2_calibration(events: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Return an empirical H3 calibration report, never an automatic policy.

    Official TeaCache implementations use architecture-trained coefficients
    and a quality-validated threshold.  A single all-real H3 trajectory can
    provide the former *measurement inputs*, but cannot prove a reuse budget.
    Therefore ``enabled`` stays false until a separately recorded validation
    supplies a model-specific policy.  This prevents threshold hunting.
    """

    pairs = [
        event for event in events
        if event.get("decision") == "real"
        and event.get("modulatedInputRelativeL1") is not None
        and event.get("residualTransitionRelativeL1") is not None
    ]
    xs = _finite_values([event.get("modulatedInputRelativeL1") for event in pairs])
    ys = _finite_values([event.get("residualTransitionRelativeL1") for event in pairs])
    count = min(len(xs), len(ys))
    xs, ys = xs[:count], ys[:count]
    report: Dict[str, Any] = {
        "schema": "h3_teacache_b2_calibration_v2",
        "enabled": False,
        "reason": "h3_specific_reuse_policy_not_quality_validated",
        "calibrationMethod": "first_block_adaln_modulated_input_probe",
        "observedTransitionCount": count,
        "requires": [
            "model_specific_rescale_curve",
            "quality_validated_reuse_budget",
            "same-contract_candidate_ab",
        ],
    }
    if count:
        report["modulatedInputRelativeL1"] = {
            "min": round(min(xs), 12), "median": round(statistics.median(xs), 12), "max": round(max(xs), 12)
        }
        report["residualTransitionRelativeL1"] = {
            "min": round(min(ys), 12), "median": round(statistics.median(ys), 12), "max": round(max(ys), 12)
        }
    fit = _fit_quadratic(xs, ys)
    if fit is not None:
        report["empiricalFit"] = fit
    return report


def _modulated_first_block_probe(block: Any, h: Any, t_emb: Any, mod_segments: Any) -> Any:
    """Sample the exact H3 first-block AdaLN attention input, not raw time.

    The operation only evaluates eight sequence rows and sixteen output
    features after the same RMSNorm + AdaLN scale/shift as ``DiTBlock``.
    It does not mutate the packed sequence and allocates no full-sequence
    clone.  Segment rows are preserved, which matters for joint text/ref/AV
    packed layouts.
    """

    torch = __import__("torch")
    started = time.perf_counter()
    rows = _sample_indices(int(h.shape[0]), 8, h.device)
    sampled = h.index_select(0, rows)
    normalized = block.norm1(sampled)
    shift_msa, scale_msa, *_ = block.adaln_proj(t_emb)
    mod_rows = torch.empty(rows.shape[0], dtype=torch.long, device=h.device)
    assigned = torch.zeros(rows.shape[0], dtype=torch.bool, device=h.device)
    for start, stop, mod_row in mod_segments:
        within = (rows >= int(start)) & (rows < int(stop))
        if bool(within.any()):
            mod_rows[within] = int(mod_row)
            assigned[within] = True
    if not bool(assigned.all()):
        raise RuntimeError("H3 TeaCache probe could not map sampled rows to modulation segments")
    modulated = normalized.mul(1.0 + scale_msa.index_select(0, mod_rows).to(normalized.dtype))
    modulated.add_(shift_msa.index_select(0, mod_rows).to(normalized.dtype))
    probe = _small_probe(modulated)
    return probe, round(time.perf_counter() - started, 6)


@dataclasses.dataclass
class H3TeaCacheB2State:
    total_steps: int
    mode: str
    calibration: Dict[str, Any]
    max_consecutive_reuse: int = 1
    step_index: int = 0
    real_forward_count: int = 0
    reused_forward_count: int = 0
    consecutive_reuse_count: int = 0
    previous_modulated_input_probe: Any = None
    previous_residual_probe: Any = None
    residual_cpu: Any = None
    events: list[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def reset(self) -> None:
        self.step_index = self.real_forward_count = self.reused_forward_count = self.consecutive_reuse_count = 0
        self.previous_modulated_input_probe = self.previous_residual_probe = self.residual_cpu = None
        self.events.clear()

    def should_reuse(self, modulated_input_probe: Any) -> tuple[bool, str, float]:
        if self.mode != "candidate":
            return False, "calibration_mode", float("inf")
        if self.step_index == 0 or self.step_index >= self.total_steps - 1:
            return False, "endpoint_real_step", float("inf")
        if self.residual_cpu is None or self.previous_modulated_input_probe is None:
            return False, "no_prior_residual", float("inf")
        if self.consecutive_reuse_count >= self.max_consecutive_reuse:
            return False, "consecutive_reuse_limit", float("inf")
        if not self.calibration.get("enabled"):
            return False, "missing_or_disabled_h3_calibration", float("inf")
        if self.calibration.get("schema") != "h3_teacache_b2_policy_v2":
            return False, "unversioned_h3_reuse_policy", float("inf")
        if self.calibration.get("qualityValidated") is not True:
            return False, "h3_reuse_policy_not_quality_validated", float("inf")
        limit = self.calibration.get("maxModulatedInputRelativeL1")
        if limit is None:
            return False, "missing_h3_modulated_input_policy", float("inf")
        delta = _relative_l1(modulated_input_probe, self.previous_modulated_input_probe)
        if delta > float(limit):
            return False, "calibrated_modulated_input_delta_exceeded", delta
        return True, "calibrated_block_residual_reuse", delta


def _sequence_wrapper(state: H3TeaCacheB2State, first_block: Any) -> Callable[..., Any]:
    """Return the opt-in native H3 block-sequence wrapper."""

    def wrapper(*, h: Any, t_emb: Any, mod_segments: Any, rope_freqs: Any, sigma: Any, execute_blocks: Callable[[Any], Any], transformer_options: Dict[str, Any]) -> Any:
        started = time.perf_counter()
        probe, probe_elapsed = _modulated_first_block_probe(first_block, h, t_emb, mod_segments)
        reuse, reason, delta = state.should_reuse(probe)
        observed_delta = delta
        if observed_delta == float("inf") and state.previous_modulated_input_probe is not None:
            observed_delta = _relative_l1(probe, state.previous_modulated_input_probe)
        event: Dict[str, Any] = {
            "step": int(state.step_index), "mode": state.mode,
            "sigma": float(sigma.detach().to(device="cpu", dtype=__import__("torch").float32).flatten()[0]),
            "modulatedInputRelativeL1": None if observed_delta == float("inf") else observed_delta,
            "decision": "reuse" if reuse else "real", "reason": reason,
            "modulatedInputProbeShape": list(probe.shape), "probeElapsedSeconds": probe_elapsed,
        }
        if reuse:
            residual = state.residual_cpu.to(device=h.device, dtype=h.dtype, non_blocking=False)
            h.add_(residual)
            del residual
            state.reused_forward_count += 1
            state.consecutive_reuse_count += 1
            event["blockCount"] = 0
        else:
            input_probe = _small_probe(h)
            # A candidate needs the complete block residual, but retaining a
            # second packed sequence on GPU would compete with the DiT's
            # actual working set.  Stage the pre-block state as BF16 on CPU;
            # after the real blocks return, form the CPU BF16 residual there.
            # This is diagnostic-only and deliberately records its own cost.
            before_cpu = h.to(device="cpu", dtype=h.dtype) if state.mode == "candidate" else None
            h = execute_blocks(h)
            output_probe = _small_probe(h)
            residual_probe = output_probe - input_probe
            event["blockCount"] = 50
            event["featureResidualProbeRelativeL1"] = _relative_l1(residual_probe, input_probe)
            if state.previous_residual_probe is not None:
                event["residualTransitionRelativeL1"] = _relative_l1(residual_probe, state.previous_residual_probe)
            state.previous_residual_probe = residual_probe
            if before_cpu is not None:
                output_cpu = h.to(device="cpu", dtype=h.dtype)
                state.residual_cpu = output_cpu.sub_(before_cpu)
                del before_cpu, output_cpu
            state.real_forward_count += 1
            state.consecutive_reuse_count = 0
        state.previous_modulated_input_probe = probe
        event["elapsedSeconds"] = round(time.perf_counter() - started, 6)
        state.events.append(event)
        state.step_index += 1
        return h

    return wrapper


def inspect_h3_teacache_b2() -> Dict[str, Any]:
    """Describe this diagnostic adapter without enabling it."""

    model_source = PROJECT_ROOT / "runtime" / "ComfyUI" / "comfy" / "ldm" / "minimax" / "model.py"
    source = model_source.read_text(encoding="utf-8") if model_source.is_file() else ""
    return {
        "requestedSource": OFFICIAL_SOURCE, "license": OFFICIAL_LICENSE,
        "scope": "diagnostic_only", "productionDefault": False, "approximate": True,
        "cacheKind": "packed_transformer_residual_cpu", "doesNotReuseDenoiserOutput": True,
        "keepsCurrentTimestepAndFinalLayer": True, "skippedBlocksAvoidPrefetch": True,
        "nativeSequenceHookPresent": "minimax_h3_block_sequence_wrapper" in source,
        "calibrationRequired": True,
        "calibrationMethod": "first_block_adaln_modulated_input_probe",
        "automaticReusePolicy": False,
    }


def apply_h3_teacache_b2_diagnostic(model_patcher: Any, *, mode: str, steps: int, calibration: Optional[Dict[str, Any]] = None) -> tuple[Any, Dict[str, Any]]:
    """Attach a test-only H3 residual-cache wrapper after KJ Sage is patched."""

    if mode not in {"calibration", "candidate"}:
        raise ValueError("H3 TeaCache B2 mode must be calibration or candidate")
    receipt = inspect_h3_teacache_b2()
    if not receipt["nativeSequenceHookPresent"]:
        raise RuntimeError("native H3 block sequence hook is missing")
    clone = model_patcher.clone()
    diffusion = clone.get_model_object("diffusion_model")
    first_block = diffusion.blocks[0]
    options = clone.model_options.setdefault("transformer_options", {})
    if "minimax_h3_block_sequence_wrapper" in options:
        raise RuntimeError("H3 block sequence wrapper is already installed")
    state = H3TeaCacheB2State(total_steps=int(steps), mode=mode, calibration=dict(calibration or {}))
    state.reset()
    options["minimax_h3_block_sequence_wrapper"] = _sequence_wrapper(state, first_block)
    clone.model_options["h3_teacache_b2_state"] = state
    receipt.update({
        "applied": True, "mode": mode, "steps": int(steps),
        "patchOrder": "after_kj_h3_sage_before_basic_guider", "cacheDevice": "cpu",
        "maxConsecutiveReuse": state.max_consecutive_reuse,
        "calibration": dict(calibration or {}),
        "firstBlockProbe": "RMSNorm+AdaLN shift/scale on 8x16 deterministic rows/features",
    })
    return clone, receipt


def h3_teacache_b2_runtime_receipt(model_patcher: Any) -> Dict[str, Any]:
    state = ((getattr(model_patcher, "model_options", {}) or {}).get("h3_teacache_b2_state"))
    if state is None:
        return {"stateAvailable": False}
    return {
        "stateAvailable": True, "mode": state.mode, "stepCount": int(state.step_index),
        "realForwardCount": int(state.real_forward_count), "reusedForwardCount": int(state.reused_forward_count),
        "events": list(state.events), "residualStoredOnCpu": state.residual_cpu is not None,
        "residualBytes": 0 if state.residual_cpu is None else int(state.residual_cpu.numel() * state.residual_cpu.element_size()),
        "derivedCalibration": derive_h3_teacache_b2_calibration(state.events) if state.mode == "calibration" else None,
    }
