"""Pre-denoise memory admission; no task-history or filesystem scanning."""

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Optional


class AdmissionState(str, Enum):
    FULL_RESIDENT = "FULL_RESIDENT"
    CANDIDATE_FULL_RESIDENT = "CANDIDATE_FULL_RESIDENT"
    CONTROLLED_OFFLOAD = "CONTROLLED_OFFLOAD"
    UNKNOWN_CANDIDATE = "UNKNOWN_CANDIDATE"


MEMORY_OFFLOAD_DECISION = "MEMORY_OFFLOAD"
NATIVE_MANAGED_DECISION = "NATIVE_MANAGED"
STATIC_LOCAL_EXPERIENCE_SOURCE = "static_local_experience"
OFFICIAL_CAPACITY_FALLBACK_SOURCE = "official_capacity_fallback"
NATIVE_DYNAMIC_VRAM_SOURCE = "native_dynamic_vram"


def native_managed_receipt(*, resolution_bucket: Optional[str] = None, duration_bucket: Optional[str] = None) -> dict:
    """Describe a receipt owned by native DynamicVRAM, without a project plan."""
    return {
        "status": "NATIVE_MANAGED",
        "decision": NATIVE_MANAGED_DECISION,
        "reason": "native_dynamic_vram_sampler_owned_residency",
        "decisionSource": NATIVE_DYNAMIC_VRAM_SOURCE,
        "resolutionBucket": resolution_bucket,
        "durationBucket": duration_bucket,
        "offloadPlan": {"status": "NATIVE_MANAGED", "decision": NATIVE_MANAGED_DECISION},
    }


@dataclass(frozen=True)
class AdmissionInputs:
    global_free_bytes: int
    model_full_resident_bytes: int
    peak_workspace_bytes: int
    driver_safety_margin_bytes: int = 0
    other_process_margin_bytes: int = 0
    calibrated: bool = True
    verified_full_resident: bool = False
    torch_allocated_bytes: Optional[int] = None
    torch_reserved_bytes: Optional[int] = None
    device_memory_used_bytes: Optional[int] = None
    route_id: Optional[str] = None
    primary_model: Optional[str] = None
    dimensions: Optional[dict] = None
    static_decision: Optional[str] = None
    decision_source: str = OFFICIAL_CAPACITY_FALLBACK_SOURCE
    resolution_bucket: Optional[str] = None
    duration_bucket: Optional[str] = None


@dataclass(frozen=True)
class AdmissionResult:
    state: AdmissionState
    required_bytes: int
    available_bytes: int
    guard_bytes: int
    locked_before_denoise: bool
    reason: str
    policyVersion: str = "h3-memory-policy-v1"
    evidence_class: str = "C"
    evidence_id: Optional[str] = None
    decision_source: str = OFFICIAL_CAPACITY_FALLBACK_SOURCE
    resolution_bucket: Optional[str] = None
    duration_bucket: Optional[str] = None


class MemoryAdmissionPolicy:
    def __init__(
        self, initial_guard_bytes: int = 1 << 30, policy_version: str = "h3-memory-policy-v1"
    ):
        if initial_guard_bytes < 0:
            raise ValueError("initial_guard_bytes must be non-negative")
        self.initial_guard_bytes = int(initial_guard_bytes)
        self.policy_version = policy_version

    def decide(self, inputs: AdmissionInputs) -> AdmissionResult:
        values = (
            inputs.global_free_bytes,
            inputs.model_full_resident_bytes,
            inputs.peak_workspace_bytes,
            inputs.driver_safety_margin_bytes,
            inputs.other_process_margin_bytes,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("memory admission values must be non-negative")
        required = (
            int(inputs.model_full_resident_bytes)
            + int(inputs.peak_workspace_bytes)
            + int(inputs.driver_safety_margin_bytes)
            + int(inputs.other_process_margin_bytes)
        )
        available = int(inputs.global_free_bytes)
        guard = self.initial_guard_bytes
        static_decision = str(inputs.static_decision or "")
        if static_decision == AdmissionState.FULL_RESIDENT.value:
            state = AdmissionState.FULL_RESIDENT
            reason = "static_local_experience_resolution_bucket"
        elif static_decision == MEMORY_OFFLOAD_DECISION:
            state = AdmissionState.CONTROLLED_OFFLOAD
            reason = "static_local_experience_duration_bucket"
        elif inputs.verified_full_resident and available >= required + guard:
            state = AdmissionState.FULL_RESIDENT
            reason = "official_capacity_proven_before_denoise"
        elif not inputs.calibrated or not inputs.verified_full_resident:
            state = AdmissionState.UNKNOWN_CANDIDATE
            reason = "official_capacity_unverified_and_history_lookup_disabled"
        else:
            state = AdmissionState.CONTROLLED_OFFLOAD
            reason = "official_capacity_not_proven_before_denoise"
        return AdmissionResult(
            state=state,
            required_bytes=required,
            available_bytes=available,
            guard_bytes=guard,
            locked_before_denoise=True,
            reason=reason,
            policyVersion=self.policy_version,
            decision_source=inputs.decision_source,
            resolution_bucket=inputs.resolution_bucket,
            duration_bucket=inputs.duration_bucket,
        )


def assert_residency_contract(state: AdmissionState, loaded_weight_bytes: int, offloaded_weight_bytes: int) -> None:
    """Fail closed if a full-resident contract observes any weight offload."""
    if state == AdmissionState.FULL_RESIDENT and int(offloaded_weight_bytes) > 0:
        raise RuntimeError("FULL_RESIDENT contract violated: model weights were offloaded")


def validate_full_resident_executor(is_dynamic_patcher: bool) -> None:
    """Dynamic ModelPatcher cannot honor ComfyUI's full_load contract."""
    if is_dynamic_patcher:
        raise RuntimeError("FULL_RESIDENT is unsupported by Dynamic ModelPatcher; refusing silent downgrade")


def official_aimdo_init_contract() -> dict:
    """Explicit pressure contract; distinguish official defaults from project overrides."""
    return {
        "simpleVramHeadroomBytes": 0,
        "simpleVramHeadroomSource": "projectOverride",
        "aimdoBaseHeadroomBytes": 256 << 20,
        "aimdoBaseHeadroomSource": "official_native_hard_guard",
        "nvmlPressure": True,
        "nvmlPressureSource": "official_comfy_default",
        "deviceExtraHeadroomBytes": 0,
        "deviceExtraHeadroomSource": "official_default",
        "source": "explicit_embedded_aimdo_contract",
    }


def load_memory_policy(path: Optional[Path] = None) -> dict:
    policy_path = path or Path(__file__).with_name("memory_policy.json")
    data = json.loads(policy_path.read_text(encoding="utf-8"))
    if data.get("schemaVersion") != "h3-memory-policy-v1":
        raise ValueError("unsupported memory policy schema")
    return data


def should_block_unknown(enforcement_enabled: bool) -> bool:
    return bool(enforcement_enabled)


def emergency_plan_for_event(safety_event: Optional[str]) -> Optional[dict]:
    """Only hard hardware events may create an emergency offload plan."""
    if safety_event not in {"cuda_oom", "wdmm_budget_exceeded", "make_resident_failed", "driver_sysmem_fallback"}:
        return None
    return {"route": "emergency_controlled_offload", "trigger": safety_event, "lockedBeforeDenoiseRestart": True}


def classify_execution(state: str, execution_signature: str) -> str:
    """Keep execution labels local; fuzzy envelopes are stored separately."""
    if state == "completed":
        return "A"
    if state == "error":
        return "B"
    return "C"


def match_envelope(policy: dict, *, route_id: Optional[str], primary_model: Optional[str], dimensions: Optional[dict]) -> dict:
    """Match only monotonic, same-route/model evidence from the small artifact."""
    dims = dimensions or {}
    required = ("internalWidth", "internalHeight", "outputFrames", "packedTokens", "referenceCount")
    if not route_id or not primary_model or any(dims.get(name) is None for name in required):
        return {"decision": AdmissionState.UNKNOWN_CANDIDATE.value, "evidenceClass": "C", "evidenceId": None, "reason": "missing_route_model_or_declared_dimension"}

    def compatible(item):
        return item.get("routeId") == route_id and item.get("primaryModel") == primary_model and all(item.get("dimensions", {}).get(name) is not None for name in required)

    danger = [item for item in policy.get("envelopes", {}).get("danger", []) if compatible(item) and all(int(dims[name]) >= int(item["dimensions"][name]) for name in required)]
    if danger:
        item = danger[0]
        return {"decision": AdmissionState.CONTROLLED_OFFLOAD.value, "evidenceClass": "B", "evidenceId": item.get("evidenceId"), "reason": "danger_lower_envelope_matched"}
    safe = [item for item in policy.get("envelopes", {}).get("safe", []) if compatible(item) and all(int(dims[name]) <= int(item["dimensions"][name]) for name in required)]
    if safe:
        item = safe[0]
        return {"decision": AdmissionState.FULL_RESIDENT.value, "evidenceClass": "A", "evidenceId": item.get("evidenceId"), "reason": "safe_upper_envelope_matched"}
    return {"decision": AdmissionState.UNKNOWN_CANDIDATE.value, "evidenceClass": "C", "evidenceId": None, "reason": "between_or_outside_known_envelopes"}


def match_static_5090_profile(policy: dict, *, route_id: Optional[str], primary_model: Optional[str], canvas: Optional[str], duration_seconds: Optional[float]) -> dict:
    """Apply the small explicit 5090 profile; never consult task history."""
    if not route_id or not primary_model or not canvas or duration_seconds is None:
        return {
            "decision": AdmissionState.UNKNOWN_CANDIDATE.value,
            "evidenceClass": "C",
            "evidenceId": None,
            "reason": "missing_static_profile_dimension",
            "decisionSource": OFFICIAL_CAPACITY_FALLBACK_SOURCE,
            "resolutionBucket": canvas,
            "durationBucket": None,
            "offloadPlan": None,
        }
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_seconds must be numeric") from exc
    if duration < 0:
        raise ValueError("duration_seconds must be non-negative")
    for rule in policy.get("staticRules", []):
        if rule.get("canvas") != canvas:
            continue
        minimum = float(rule.get("minDurationSeconds", 0))
        maximum = rule.get("maxDurationSeconds")
        maximum_exclusive = rule.get("maxDurationSecondsExclusive")
        in_range = duration >= minimum
        if maximum is not None:
            in_range = in_range and duration <= float(maximum)
        if maximum_exclusive is not None:
            in_range = in_range and duration < float(maximum_exclusive)
        if in_range:
            return {
                "decision": rule["decision"],
                "evidenceClass": rule.get("evidenceClass", "C"),
                "evidenceId": rule.get("evidenceId"),
                "reason": "static_rtx5090_profile_rule",
                "decisionSource": STATIC_LOCAL_EXPERIENCE_SOURCE,
                "resolutionBucket": rule.get("resolutionBucket", canvas),
                "durationBucket": rule.get("durationBucket"),
                "offloadPlanRequired": rule["decision"] == MEMORY_OFFLOAD_DECISION,
                "offloadPlan": "static_minimum_offload_prefetch" if rule["decision"] == MEMORY_OFFLOAD_DECISION else None,
            }
    return {
        "decision": AdmissionState.UNKNOWN_CANDIDATE.value,
        "evidenceClass": "C",
        "evidenceId": None,
        "reason": "canvas_or_duration_outside_static_profile",
        "decisionSource": OFFICIAL_CAPACITY_FALLBACK_SOURCE,
        "resolutionBucket": canvas,
        "durationBucket": "unknown",
        "offloadPlanRequired": False,
        "offloadPlan": None,
    }


def build_static_offload_plan(*, decision: str, free_bytes: Optional[int], workspace_bytes: Optional[int],
                              guard_bytes: int, model_bytes: Optional[int]) -> dict:
    """Build one deterministic minimum-offload plan; runtime replanning is forbidden."""
    if decision not in {"CONTROLLED_OFFLOAD", MEMORY_OFFLOAD_DECISION}:
        return {"status": "NOT_APPLICABLE", "decision": decision, "runtimeReplan": False, "transferLoop": False}
    values = (free_bytes, workspace_bytes, guard_bytes, model_bytes)
    if any(value is None for value in values):
        return {
            "status": "STATIC_PLAN_UNAVAILABLE",
            "decision": MEMORY_OFFLOAD_DECISION,
            "blocked": True,
            "reason": "capacity_inputs_missing",
            "runtimeReplan": False,
            "transferLoop": False,
        }
    if any(int(value) < 0 for value in values):
        raise ValueError("offload plan inputs must be non-negative")
    resident_budget = max(0, min(int(model_bytes), int(free_bytes) - int(workspace_bytes) - int(guard_bytes)))
    planned_offload = max(0, int(model_bytes) - resident_budget)
    return {
        "status": "STATIC_PLAN",
        "decision": MEMORY_OFFLOAD_DECISION,
        "blocked": False,
        "reason": "minimum_required_offload_static_plan",
        "residentBudgetBytes": resident_budget,
        "residentCutoff": "minimum_required_model_resident_prefix",
        "plannedOffloadBytes": planned_offload,
        "offloadScope": "minimum_required_weights_only",
        "prefetchDepth": 1,
        "prefetchOrder": ["resident_prefix", "single_offloaded_segment"],
        "transferLoop": False,
        "runtimeReplan": False,
        "planSource": STATIC_LOCAL_EXPERIENCE_SOURCE,
    }


def validate_static_offload_plan(plan: dict) -> None:
    """Fail closed if a locked MEMORY_OFFLOAD plan can be replanned or looped."""
    if not isinstance(plan, dict):
        raise ValueError("offload plan must be an object")
    if plan.get("decision") != MEMORY_OFFLOAD_DECISION:
        raise ValueError("offload plan decision must be MEMORY_OFFLOAD")
    if plan.get("runtimeReplan") is not False or plan.get("transferLoop") is not False:
        raise ValueError("MEMORY_OFFLOAD forbids runtime replanning and transfer loops")
    if plan.get("prefetchDepth") != 1 or plan.get("offloadScope") != "minimum_required_weights_only":
        raise ValueError("MEMORY_OFFLOAD requires one minimum weights-only prefetch plan")
