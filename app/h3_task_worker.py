"""One-task H3 GPU worker.

The HTTP/API process deliberately does not import or execute the model here.
Each real request receives its own spawned child process. Cancellation is
cooperative: this worker owns the CUDA context and must unwind naturally.
"""

from __future__ import annotations

import json
import os
import copy
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from project_paths import portable_project_record


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def safe_exception_metadata(exc: BaseException, attribute: str, default: Any, warnings: list[Dict[str, str]] | None = None) -> Any:
    try:
        return getattr(exc, attribute, default)
    except BaseException as metadata_error:
        if warnings is not None:
            warnings.append({
                "attribute": attribute,
                "error": f"{type(metadata_error).__name__}: {metadata_error}",
            })
        return default


def safe_exception_text(exc: BaseException) -> str:
    try:
        return str(exc)
    except BaseException as metadata_error:
        return f"<exception message unavailable: {type(metadata_error).__name__}: {metadata_error}>"


def exception_recovery_fields(exc: BaseException, warnings: list[Dict[str, str]] | None = None) -> Dict[str, Any]:
    raw_errors = safe_exception_metadata(exc, "cleanup_errors", [], warnings)
    try:
        errors = list(raw_errors or [])
    except BaseException as metadata_error:
        if warnings is not None:
            warnings.append({"attribute": "cleanup_errors.normalize", "error": f"{type(metadata_error).__name__}: {metadata_error}"})
        errors = []
    return {"resourceRecoveryErrors": errors} if errors else {}


def worker_exception_fields(exc: BaseException, execution_receipt: Dict[str, Any]) -> Dict[str, Any]:
    warnings: list[Dict[str, str]] = []
    runtime_value = safe_exception_metadata(exc, "runtimeStages", {}, warnings)
    try:
        runtime_stages = dict(runtime_value or {})
    except BaseException as metadata_error:
        warnings.append({"attribute": "runtimeStages.normalize", "error": f"{type(metadata_error).__name__}: {metadata_error}"})
        runtime_stages = {}
    failed_receipt = dict(runtime_stages.get("executionReceipt") or execution_receipt)
    recovery_fields = exception_recovery_fields(exc, warnings)
    cleanup_errors = list(recovery_fields.get("resourceRecoveryErrors") or [])
    fallback_stage = runtime_stages.get("failedStage") or failed_receipt.get("failedStage") or "worker_execution"
    failed_stage = safe_exception_metadata(exc, "failedStage", fallback_stage, warnings)
    if cleanup_errors:
        failed_receipt["resourceRecoveryErrors"] = cleanup_errors
        runtime_stages["resourceRecovery"] = {"status": "failed", "errors": cleanup_errors}
    failed_receipt["outputAuthentic"] = False
    if warnings:
        runtime_stages.setdefault("exceptionMetadataWarnings", []).extend(warnings)
        failed_receipt["exceptionMetadataWarnings"] = list(warnings)
    return {
        "error": safe_exception_text(exc),
        "exceptionType": type(exc).__name__,
        "failedStage": failed_stage,
        "runtimeStages": runtime_stages,
        "resourceRecoveryErrors": cleanup_errors,
        "executionReceipt": failed_receipt,
        "exceptionMetadataWarnings": warnings,
    }


TASK_RUNTIME_DIR = PROJECT_ROOT / "temp" / "task_runtime"


class TaskRuntimeSnapshot:
    """Atomic, task-local runtime evidence owned by one GPU worker.

    The parent API process receives the same events through its queue, but it
    cannot rely on that queue after a hard worker termination.  This compact
    receipt is deliberately independent of the model cache and contains only
    serialisable observations made at phase and sampler boundaries.
    """

    def __init__(self, task_id: str, *, root: Path = PROJECT_ROOT) -> None:
        safe_task_id = "".join(char for char in str(task_id) if char.isalnum() or char in {"-", "_"})
        if not safe_task_id:
            raise ValueError("task runtime snapshot requires a task id")
        self.task_id = safe_task_id
        self.directory = root / "temp" / "task_runtime"
        self.path = self.directory / f"{self.task_id}.json"
        self._lock = threading.Lock()
        self._receipt: Dict[str, Any] = {
            "schemaVersion": 1,
            "taskId": self.task_id,
            "workerPid": os.getpid(),
            "createdAt": self._timestamp(),
            "events": [],
        }

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    @staticmethod
    def _freeze_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
        """Detach receipt history from mutable runtime-stage dictionaries."""
        return copy.deepcopy(dict(payload or {}))

    def record(self, kind: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Append an observation and atomically replace only this task file."""
        event = {"kind": str(kind), "at": self._timestamp(), "payload": portable_project_record(self._freeze_payload(payload))}
        with self._lock:
            self._receipt["updatedAt"] = event["at"]
            self._receipt.setdefault("events", []).append(event)
            self._receipt["lastEvent"] = event
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_name(
                    f".{self.path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                temporary.write_text(json.dumps(self._receipt, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.path)
            except (OSError, TypeError, ValueError):
                # A receipt write must never alter inference or cancellation.
                pass
        return event["payload"]

    def update_fields(self, values: Dict[str, Any]) -> None:
        """Atomically expose stable execution receipt fields at the top level."""

        with self._lock:
            self._receipt.update(portable_project_record(self._freeze_payload(values)))
            self._receipt["updatedAt"] = self._timestamp()
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                temporary = self.path.with_name(
                    f".{self.path.stem}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                temporary.write_text(json.dumps(self._receipt, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(self.path)
            except (OSError, TypeError, ValueError):
                pass


def run_task(plan: Dict[str, Any], cancel_event: Any, messages: Any) -> None:
    """Run one direct H3 plan and stream serialisable status to the parent."""
    task_id = str(plan.get("taskId") or plan.get("compiled", {}).get("taskId") or "")
    snapshot = TaskRuntimeSnapshot(task_id)
    # Keep a plan receipt available even if a worker dependency fails during
    # startup.  Previously those imports ran before the exception boundary,
    # so the parent could only report "worker exited without a result".
    execution_receipt = dict(plan.get("executionReceipt") or {})
    last_persisted_progress: int | None = None

    def progress(value: int, message: str) -> None:
        nonlocal last_persisted_progress
        # The queue remains live for UI updates.  Persist only a changed coarse
        # progress value so repeated denoiser-block messages do not rewrite the
        # whole task receipt during a GPU hot path.
        if last_persisted_progress != int(value):
            snapshot.record("progress", {"progress": int(value), "message": str(message)})
            last_persisted_progress = int(value)
        messages.put({"type": "progress", "progress": int(value), "message": str(message)})

    def telemetry(payload: Dict[str, Any]) -> None:
        """Stream already-observed runtime evidence without changing inference."""
        observed = snapshot.record(str((payload or {}).get("kind") or "telemetry"), payload)
        messages.put({"type": "telemetry", "telemetry": copy.deepcopy(observed)})

    # DirectH3Runner probes this optional hook only at phase/step boundaries.
    # It remains absent for other adapter callers.
    progress.telemetry = telemetry  # type: ignore[attr-defined]

    def heartbeat(value: int, message: str, payload: Dict[str, Any]) -> None:
        """Keep the active UI current without synchronous task-file writes."""
        messages.put({
            "type": "heartbeat",
            "progress": int(value),
            "message": str(message),
            "heartbeat": copy.deepcopy(dict(payload or {})),
        })

    progress.heartbeat = heartbeat  # type: ignore[attr-defined]

    try:
        from h3_adapter import EmbeddedH3BackendAdapter
        from h3_compiler import initial_execution_receipt

        if not execution_receipt:
            execution_receipt = dict(initial_execution_receipt(plan.get("compiled") or {}))
        snapshot.update_fields(execution_receipt)
        snapshot.record("worker_started", {
            "executionProfile": plan.get("compiled", {}).get("advanced", {}).get("executionProfile"),
            "algorithmRouteFingerprint": execution_receipt.get("algorithmRouteFingerprint"),
            "workerPid": os.getpid(),
        })
        result = EmbeddedH3BackendAdapter().execute(
            plan,
            progress=progress,
            cancel_event=cancel_event,
        )
        runtime_stages = dict(result.get("runtimeStages") or {})
        final_receipt = dict(runtime_stages.get("executionReceipt") or result.get("executionReceipt") or execution_receipt)
        final_receipt["outputAuthentic"] = bool(result.get("outputAuthentic"))
        progress_receipt = final_receipt.get("progressReceipt")
        if isinstance(progress_receipt, dict):
            final_receipt["lastMeaningfulStage"] = progress_receipt.get("lastMeaningfulStage")
        result["executionReceipt"] = final_receipt
        snapshot.update_fields({"executionReceipt": final_receipt, **final_receipt})
        if result.get("status") == "cancelled":
            now = snapshot._timestamp()
            cancellation = dict((result.get("runtimeStages") or {}).get("cancellation") or {})
            cancellation.update({"cancelAcknowledgedAt": now, "safeBoundaryAt": now})
            result.setdefault("runtimeStages", {})["cancellation"] = cancellation
            acceleration = dict((result.get("runtimeStages") or {}).get("acceleration") or {})
            if acceleration:
                snapshot.record("acceleration_terminal", acceleration)
            snapshot.record("cancellation_acknowledged", cancellation)
        elif (result.get("runtimeStages") or {}).get("acceleration"):
            snapshot.record("acceleration_terminal", dict((result.get("runtimeStages") or {}).get("acceleration") or {}))
        snapshot.record("worker_result", {
            "status": result.get("status"),
            "algorithmRouteFingerprint": final_receipt.get("algorithmRouteFingerprint"),
            "resolutionMismatch": final_receipt.get("resolutionMismatch"),
            "outputAuthentic": final_receipt.get("outputAuthentic"),
        })
        result["workerRuntimeSnapshotPath"] = str(snapshot.path)
        messages.put({"type": "result", "result": result})
    except BaseException as exc:  # Parent owns the user-facing state record.
        failure = worker_exception_fields(exc, execution_receipt)
        snapshot.update_fields(failure["executionReceipt"])
        snapshot.record("worker_exception", {
            "error": failure["error"],
            "exceptionType": failure["exceptionType"],
            "failedStage": failure["failedStage"],
            "resourceRecoveryErrors": failure["resourceRecoveryErrors"],
            "exceptionMetadataWarnings": failure["exceptionMetadataWarnings"],
        })
        messages.put({
            "type": "exception",
            **failure,
            "workerRuntimeSnapshotPath": str(snapshot.path),
            "traceback": traceback.format_exc(),
        })
