"""Independent local H3 web service and task state machine."""

from __future__ import annotations

import json
import copy
import multiprocessing
import os
import queue
import ctypes
import subprocess
import threading
import time
import traceback
import uuid
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

from h3_adapter import DryRunH3Adapter, EmbeddedH3BackendAdapter
from h3_compiler import CompilerError, compile_request
from h3_task_worker import run_task
from media_staging import MAX_UPLOAD_BYTES, StagingError, claim_assets_for_task, stage_stream, validate_staged_references
from project_paths import portable_project_record
from performance_baseline import finalize_performance_baseline
from swiftvr_postprocess import SwiftVRPostprocessStore
from nvidia_vsr_postprocess import NvidiaVSRPostprocessStore


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
OUTPUT_DIR = APP_DIR.parent / "output"
GPU_ADMISSION_LOCK = threading.RLock()
TEMP_DIR = APP_DIR.parent / "temp"


def _validate_worker_result_identity(plan: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Reject a terminal low-FPS result unless the worker proves the compiled route."""

    compiled = plan.get("compiled") if isinstance(plan.get("compiled"), dict) else {}
    route = compiled.get("algorithmRoute") if isinstance(compiled.get("algorithmRoute"), dict) else {}
    expected_route = str(route.get("algorithmRoute") or "")
    if expected_route == "low_fps_time_remap_experimental":
        raise RuntimeError("non-24 FPS routes are retired; worker result rejected")
TASK_STATE_DIR = TEMP_DIR / "tasks"
TASK_RUNTIME_DIR = TEMP_DIR / "task_runtime"
HOST = os.environ.get("H3_HOST", "0.0.0.0")
PORT = int(os.environ.get("H3_PORT", "6006"))
# Give the worker one short window to stop at a sampler callback. If a CUDA
# kernel does not return in time, terminate only the isolated task process;
# the API service and persisted task evidence remain alive.
CANCEL_ACK_WAIT_SECONDS = 5.0
CANCEL_TERMINATE_JOIN_SECONDS = 5.0
HARDWARE_CACHE_SECONDS = 2.0


class HardwareStatusMonitor:
    """Read-only, process-wide hardware sampling with a short shared cache."""

    def __init__(self, cache_seconds: float = HARDWARE_CACHE_SECONDS) -> None:
        self._cache_seconds = max(0.5, float(cache_seconds))
        self._lock = threading.Lock()
        self._cached: Optional[Dict[str, Any]] = None
        self._cached_at = 0.0
        self._cpu_previous: Optional[tuple[int, int, int]] = None

    @staticmethod
    def _unavailable(reason: str) -> Dict[str, Any]:
        return {"available": False, "reason": reason}

    def _cpu(self) -> Dict[str, Any]:
        if os.name != "nt":
            return self._unavailable("windows_system_times_unavailable")

        class FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

        idle, kernel, user = FileTime(), FileTime(), FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return self._unavailable("GetSystemTimes_failed")
        convert = lambda value: (int(value.high) << 32) | int(value.low)
        current = (convert(idle), convert(kernel), convert(user))
        previous = self._cpu_previous
        self._cpu_previous = current
        if previous is None:
            return {"available": False, "reason": "first_sample_pending", "logicalCores": os.cpu_count()}
        idle_delta = current[0] - previous[0]
        total_delta = (current[1] - previous[1]) + (current[2] - previous[2])
        if total_delta <= 0:
            return self._unavailable("cpu_interval_unavailable")
        return {"available": True, "loadPercent": round(max(0.0, min(100.0, (total_delta - idle_delta) * 100.0 / total_delta)), 1), "logicalCores": os.cpu_count()}

    @staticmethod
    def _memory() -> Dict[str, Any]:
        if os.name != "nt":
            return HardwareStatusMonitor._unavailable("windows_memory_status_unavailable")

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_uint32), ("memoryLoad", ctypes.c_uint32), ("totalPhys", ctypes.c_uint64), ("availPhys", ctypes.c_uint64), ("totalPageFile", ctypes.c_uint64), ("availPageFile", ctypes.c_uint64), ("totalVirtual", ctypes.c_uint64), ("availVirtual", ctypes.c_uint64), ("availExtendedVirtual", ctypes.c_uint64)]

        value = MemoryStatus()
        value.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            return HardwareStatusMonitor._unavailable("GlobalMemoryStatusEx_failed")
        total, available = int(value.totalPhys), int(value.availPhys)
        return {"available": True, "usedGiB": round((total - available) / (1024 ** 3), 2), "totalGiB": round(total / (1024 ** 3), 2), "availableGiB": round(available / (1024 ** 3), 2), "usedPercent": round((total - available) * 100.0 / total, 1) if total else None}

    @staticmethod
    def _number(value: str) -> Optional[float]:
        try:
            return float(value.strip())
        except (TypeError, ValueError):
            return None

    def _gpu(self) -> Dict[str, Any]:
        fields = "temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,fan.speed,utilization.encoder,utilization.decoder"
        try:
            result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=1.0, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return self._unavailable("nvidia_smi_unavailable")
        if result.returncode != 0 or not result.stdout.strip():
            return self._unavailable("nvidia_smi_failed")
        values = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
        if len(values) != 9:
            return self._unavailable("nvidia_smi_schema_unavailable")
        temperature, load, memory_used, memory_total, power_draw, power_limit, fan, encoder, decoder = [self._number(item) for item in values]
        memory_percent = round(memory_used * 100.0 / memory_total, 1) if memory_used is not None and memory_total else None
        return {"available": True, "temperatureC": temperature, "coreLoadPercent": load, "memoryUsedMiB": memory_used, "memoryTotalMiB": memory_total, "memoryPercent": memory_percent, "powerDrawW": power_draw, "powerLimitW": power_limit, "fanPercent": fan, "encoderLoadPercent": encoder, "decoderLoadPercent": decoder}

    def snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            if self._cached is not None and now - self._cached_at < self._cache_seconds:
                result = dict(self._cached)
                result["cached"] = True
                result["cacheAgeMilliseconds"] = round((now - self._cached_at) * 1000)
                return result
            result = {"cached": False, "cacheAgeMilliseconds": 0, "sampledAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu": self._gpu(), "cpu": self._cpu(), "memory": self._memory()}
            self._cached, self._cached_at = result, now
            return dict(result)


class TaskStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._cancel_events: Dict[str, threading.Event] = {}
        self._process_context = multiprocessing.get_context("spawn")
        self._workers: Dict[str, Any] = {}
        self._worker_events: Dict[str, Any] = {}
        self._worker_messages: Dict[str, Any] = {}
        self._worker_cleanup: Dict[str, Dict[str, Any]] = {}
        self._cancel_monotonic: Dict[str, float] = {}
        self._dry_run = DryRunH3Adapter()
        self._direct = EmbeddedH3BackendAdapter()
        self._hardware_monitor = HardwareStatusMonitor()
        self._restore_persisted_tasks()

    def hardware_status(self) -> Dict[str, Any]:
        """Combine cached OS metrics with a cheap, current task/worker view."""
        with self._lock:
            active = next((task for task in self._tasks.values() if task.get("state") in {"queued", "compiling", "ready", "loading", "running", "cancelling"}), None)
            task_id = active.get("id") if active else None
            worker = self._workers.get(task_id) if task_id else None
            runtime = {
                "service": "ready", "taskState": active.get("state") if active else "idle",
                "taskStage": (active.get("timing") or {}).get("currentStage") if active else "idle",
                "taskId": task_id, "workerPid": getattr(worker, "pid", None) if worker else None,
                "workerAlive": bool(worker is not None and worker.is_alive()),
            }
        result = self._hardware_monitor.snapshot()
        result["runtime"] = runtime
        return result

    def gpu_busy(self) -> bool:
        with self._lock:
            active = any(task.get("state") in {"queued", "compiling", "ready", "loading", "running", "cancelling"} for task in self._tasks.values())
            worker_alive = any(worker.is_alive() for worker in self._workers.values())
            return active or worker_alive

    @staticmethod
    def _timestamp() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    @staticmethod
    def _stage_for(state: str, message: str) -> str:
        """Map observable runtime messages to stable, user-facing phases."""

        if state == "queued":
            return "queue"
        if state == "cancelling":
            return "stopping"
        if state == "completed":
            return "completed"
        if state == "cancelled":
            return "cancelled"
        if state == "error":
            return "failed"
        text = (message or "").lower()
        if "qwen" in text or "text encoder" in text:
            return "qwen"
        if "sampling" in text or "denoiser" in text or "dynamicvram" in text:
            return "sampling"
        if "audio vae" in text or "audio decode" in text:
            return "audio_vae"
        if "video vae" in text or "video decode" in text:
            return "video_vae"
        if "mp4" in text or "export" in text or "packing" in text:
            return "mp4_export"
        if "ref2va" in text or "fl2va" in text or "primary model" in text:
            return "model_load"
        if "reference" in text or "direct h3 runtime" in text:
            return "reference_preprocess"
        return "preparing"

    def _update_timing(self, task: Dict[str, Any], *, state: str, message: str, now: str, stage_override: Optional[str] = None) -> None:
        """Persist accepted-to-terminal and phase timestamps without timers in JS."""

        timing = task.setdefault("timing", {
            "acceptedAt": task.get("acceptedAt") or task.get("createdAt") or now,
            "currentStage": "queue",
            "stageStartedAt": task.get("acceptedAt") or task.get("createdAt") or now,
            "stageEvents": [],
            "retryEvents": [],
        })
        stage = str(stage_override or self._stage_for(state, message))
        prior = timing.get("currentStage")
        if stage != prior:
            events = timing.setdefault("stageEvents", [])
            if events and not events[-1].get("endedAt"):
                events[-1]["endedAt"] = now
            events.append({"stage": stage, "startedAt": now})
            timing["currentStage"] = stage
            timing["stageStartedAt"] = now
        elif not timing.get("stageEvents"):
            timing["stageEvents"] = [{"stage": stage, "startedAt": timing.get("stageStartedAt") or now}]
        timing["updatedAt"] = now
        if state in {"completed", "error", "cancelled", "dry_run_complete"}:
            # A terminal timestamp must describe the actual terminal state.
            # In particular, a user-cancelled GPU worker is not a completed
            # generation; retaining ``completedAt`` for it made API/UI timing
            # and postmortems ambiguous.
            terminal_at = task.get("terminalAt") or now
            task["terminalAt"] = terminal_at
            timing["terminalAt"] = terminal_at
            if state == "cancelled":
                task["cancelledAt"] = task.get("cancelledAt") or terminal_at
                timing["cancelledAt"] = task["cancelledAt"]
            elif state in {"completed", "dry_run_complete"}:
                task["completedAt"] = task.get("completedAt") or terminal_at
                timing["completedAt"] = task["completedAt"]
            else:
                task["failedAt"] = task.get("failedAt") or terminal_at
                timing["failedAt"] = task["failedAt"]
            events = timing.setdefault("stageEvents", [])
            if events and not events[-1].get("endedAt"):
                events[-1]["endedAt"] = terminal_at

    def _persist_task(self, task: Dict[str, Any]) -> None:
        """Persist only a task created and completed by this server."""
        try:
            TASK_STATE_DIR.mkdir(parents=True, exist_ok=True)
            target = (TASK_STATE_DIR / f"{task['id']}.json").resolve()
            target.relative_to(TASK_STATE_DIR.resolve())
            # A cancellation request and a sampling callback may both update a
            # task.  Write a per-thread temporary file and atomically replace
            # the record so a browser restart never sees half a JSON object.
            temporary = target.with_name(f".{target.stem}.{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps(portable_project_record(task), ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(target)
        except (OSError, TypeError, ValueError):
            # Persistence must not crash or alter the running model task.
            pass

    @staticmethod
    def _worker_snapshot_path(task_id: str) -> Path:
        safe_task_id = "".join(char for char in str(task_id) if char.isalnum() or char in {"-", "_"})
        return (TASK_RUNTIME_DIR / f"{safe_task_id}.json").resolve()

    def _read_worker_snapshot(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Read only the matching task's atomic worker receipt, never old task state."""
        try:
            path = self._worker_snapshot_path(task_id)
            path.relative_to(TASK_RUNTIME_DIR.resolve())
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            if str(snapshot.get("taskId") or "") != str(task_id):
                return None
            events = snapshot.get("events")
            if not isinstance(events, list):
                return None
            return snapshot
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _persist_worker_final_baseline(self, task_id: str, baseline: Dict[str, Any]) -> None:
        """Append the parent-verified terminal baseline to the matching worker receipt."""

        snapshot = self._read_worker_snapshot(task_id)
        if snapshot is None:
            return
        try:
            path = self._worker_snapshot_path(task_id)
            path.relative_to(TASK_RUNTIME_DIR.resolve())
            snapshot["performanceBaseline"] = json.loads(json.dumps(baseline))
            snapshot["updatedAt"] = self._timestamp()
            temporary = path.with_name(f".{path.stem}.{threading.get_ident()}.tmp")
            temporary.write_text(json.dumps(portable_project_record(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(path)
        except (OSError, TypeError, ValueError):
            pass

    @staticmethod
    def _snapshot_telemetry(snapshot: Optional[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if not snapshot:
            return []
        telemetry = []
        for event in snapshot.get("events", []):
            if not isinstance(event, dict):
                continue
            payload = event.get("payload")
            if isinstance(payload, dict) and event.get("kind") not in {"progress", "worker_started", "worker_result", "worker_exception"}:
                telemetry.append(dict(payload))
        return telemetry

    @staticmethod
    def _terminal_states() -> set[str]:
        return {"completed", "error", "cancelled", "dry_run_complete"}

    @staticmethod
    def _pid_exists(pid: Any) -> bool:
        try:
            value = int(pid)
            if value <= 0:
                return False
            if os.name == "nt":
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, value)
                if not handle:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            os.kill(value, 0)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def _worker_pid_for_task(self, task: Dict[str, Any], worker_snapshot: Optional[Dict[str, Any]]) -> Optional[int]:
        pid = task.get("workerPid")
        if pid is None and worker_snapshot is not None:
            pid = worker_snapshot.get("workerPid")
        try:
            return int(pid) if pid is not None else None
        except (TypeError, ValueError):
            return None

    def _mark_worker_lost(self, task: Dict[str, Any], worker_snapshot: Optional[Dict[str, Any]], *, reason: str) -> None:
        now = self._timestamp()
        worker_pid = self._worker_pid_for_task(task, worker_snapshot)
        task.update({
            "state": "error",
            "progress": 100,
            "message": "H3 task worker disappeared before returning a terminal result",
            "updatedAt": now,
            "workerExitedAt": now,
            "terminalAt": now,
            "failedAt": now,
            "result": {
                "status": "error",
                "realInference": False,
                "outputAuthentic": False,
                "failedStage": "worker_lost",
                "error": reason,
                "workerRuntimeSnapshot": worker_snapshot,
                "samplingTelemetry": self._snapshot_telemetry(worker_snapshot),
                "workerLifecycle": {
                    "workerExited": True,
                    "workerExitedAt": now,
                    "cleanupEvidence": {
                        "workerPid": worker_pid,
                        "workerPresent": False,
                        "reason": reason,
                    },
                },
            },
        })
        self._update_timing(task, state="error", message=str(task["message"]), now=now)
        task["assetCleanup"] = {
            **(task.get("assetCleanup") or {}),
            "released": False,
            "automaticCleanup": "disabled",
        }

    def _restore_persisted_tasks(self) -> None:
        """Restore terminal records written by this server, never old outputs."""
        if not TASK_STATE_DIR.is_dir():
            return
        restored: Dict[str, Dict[str, Any]] = {}
        for path in TASK_STATE_DIR.glob("*.json"):
            try:
                task = portable_project_record(json.loads(path.read_text(encoding="utf-8")))
                task_id = str(task.get("id") or "")
                if not task_id or task_id != path.stem:
                    continue
                if task.get("state") not in self._terminal_states():
                    # A restarted service cannot supervise or adopt an old CUDA
                    # worker. Record an explicit terminal failure when its PID is
                    # already gone, and the same failure class when a live PID is
                    # unmanaged by this new process.
                    worker_snapshot = self._read_worker_snapshot(task_id)
                    worker_pid = self._worker_pid_for_task(task, worker_snapshot)
                    worker_present = self._pid_exists(worker_pid)
                    reason = "orphaned_worker_after_service_recovery" if not worker_present else "orphaned_worker_unmanaged_after_service_recovery"
                    self._mark_worker_lost(task, worker_snapshot, reason=reason)
                    task["result"]["workerLifecycle"]["cleanupEvidence"]["workerPresentAfterRecovery"] = worker_present
                    self._persist_task(task)
                # Records written before explicit timing support have no
                # terminal timestamp.  The persisted record's modification
                # time is a truthful terminal bound, unlike a browser clock;
                # do not fabricate missing per-phase history for them.
                accepted_at = task.get("acceptedAt") or task.get("createdAt")
                terminal_at = task.get("terminalAt") or task.get("completedAt") or task.get("cancelledAt") or task.get("failedAt") or time.strftime(
                    "%Y-%m-%dT%H:%M:%S%z", time.localtime(path.stat().st_mtime)
                )
                task.setdefault("acceptedAt", accepted_at)
                task.setdefault("updatedAt", terminal_at)
                task.setdefault("terminalAt", terminal_at)
                if task.get("state") == "cancelled":
                    task.setdefault("cancelledAt", terminal_at)
                elif task.get("state") in {"completed", "dry_run_complete"}:
                    task.setdefault("completedAt", terminal_at)
                elif task.get("state") == "error":
                    task.setdefault("failedAt", terminal_at)
                task.setdefault("timing", {
                    "acceptedAt": accepted_at,
                    "terminalAt": terminal_at,
                    "currentStage": "completed" if task.get("state") in {"completed", "dry_run_complete"} else task.get("state"),
                    "stageStartedAt": terminal_at,
                    "stageEvents": [],
                    "retryEvents": [],
                    "legacyRecord": True,
                })
                restored[task_id] = task
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        with self._lock:
            self._tasks.update(restored)
            for task_id in restored:
                self._cancel_events[task_id] = threading.Event()

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            if any(worker.is_alive() for worker in self._workers.values()):
                raise CompilerError("A previous H3 worker is still safely stopping; wait for it to exit before starting another task")
        execution_mode = str(payload.get("executionMode") or "dry-run").lower()
        if execution_mode not in {"dry-run", "real"}:
            raise CompilerError("executionMode must be dry-run or real")
        if execution_mode == "real":
            payload = copy.deepcopy(payload)
            validated = validate_staged_references(payload.get("references"))
            for reference, facts in zip(payload.get("references") or [], validated):
                reference.update(facts)
        compiled = compile_request(payload)
        is_experiment = str((compiled.get("advanced") or {}).get("experimentId") or "")
        task_id = ("kjexp-" if is_experiment else "") + uuid.uuid4().hex[:12]
        adapter = self._direct if execution_mode != "dry-run" else self._dry_run
        plan = adapter.build_plan(compiled, task_id)
        asset_claims = claim_assets_for_task(payload.get("references"), task_id)
        if execution_mode != "dry-run":
            plan["compiled"]["dryRun"] = False
            plan["compiled"]["adapter"] = {
                "name": "EmbeddedH3BackendAdapter",
                "realInference": True,
                "server": False,
            }
        accepted_at = self._timestamp()
        task = {
            "id": task_id,
            "createdAt": accepted_at,
            "acceptedAt": accepted_at,
            "updatedAt": accepted_at,
            "state": "queued",
            "progress": 0,
            "message": "Queued for direct H3 runtime" if execution_mode != "dry-run" else "Queued for dry-run compilation",
            "realInference": execution_mode != "dry-run",
            "executionMode": execution_mode,
            "postprocess1080p": bool(payload.get("postprocess1080p")),
            "experimental": {
                "enabled": bool(is_experiment),
                "experimentId": is_experiment or None,
                "route": (compiled.get("advanced") or {}).get("experimentalRoute") if is_experiment else None,
                "status": "queued" if is_experiment else None,
                "outputRoot": "_isolated_research/outputs" if is_experiment else None,
            },
            "plan": plan,
            "assetCleanup": {
                "claims": asset_claims,
                "released": False,
                "automaticCleanup": "disabled",
                "retained": [claim.get("path") for claim in asset_claims],
            },
            "result": None,
            "timing": {
                "acceptedAt": accepted_at,
                "currentStage": "queue",
                "stageStartedAt": accepted_at,
                "stageEvents": [{"stage": "queue", "startedAt": accepted_at}],
                "retryEvents": [],
            },
        }
        with self._lock:
            self._tasks[task_id] = task
            self._cancel_events[task_id] = threading.Event()
        self._persist_task(task)
        if execution_mode == "dry-run":
            threading.Thread(target=self._run_dry_run, args=(task_id,), daemon=True, name=f"h3-dry-{task_id}").start()
        else:
            self._start_real_worker(task_id)
        return self.get(task_id)

    def _set(self, task_id: str, **values: Any) -> None:
        snapshot = None
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id]["state"] not in {"cancelled", "completed", "error", "dry_run_complete"}:
                current_state = str(self._tasks[task_id].get("state") or "queued")
                requested_state = values.get("state")
                # Cancellation is a monotonic lifecycle transition. Progress
                # callbacks may arrive after the user clicked stop, but they
                # must never resurrect the task as running or hide its stop
                # request. Only a terminal worker result may close it.
                if current_state == "cancelling" and requested_state not in {"cancelled", "completed", "error", "dry_run_complete"}:
                    values.pop("state", None)
                    values.pop("message", None)
                if "progress" in values:
                    # Denoiser telemetry and sampler callbacks arrive through
                    # different layers.  Never expose a numeric regression to
                    # API clients or a refreshed browser.
                    values["progress"] = max(
                        int(self._tasks[task_id].get("progress", 0) or 0),
                        int(values["progress"] or 0),
                    )
                self._tasks[task_id].update(values)
                experimental = self._tasks[task_id].get("experimental")
                if isinstance(experimental, dict) and experimental.get("enabled"):
                    experimental["status"] = str(self._tasks[task_id].get("state") or "queued")
                now = self._timestamp()
                self._tasks[task_id]["updatedAt"] = now
                self._update_timing(
                    self._tasks[task_id],
                    state=str(self._tasks[task_id].get("state") or "queued"),
                    message=str(self._tasks[task_id].get("message") or ""),
                    now=now,
                )
                if self._tasks[task_id].get("state") in {"completed", "error", "cancelled", "dry_run_complete"}:
                    snapshot = json.loads(json.dumps(self._tasks[task_id]))
                elif values.get("state") is not None:
                    snapshot = json.loads(json.dumps(self._tasks[task_id]))
        if snapshot is not None:
            self._persist_task(snapshot)

    def _is_cancelled(self, task_id: str) -> bool:
        with self._lock:
            return self._cancel_events.get(task_id, threading.Event()).is_set() or self._tasks.get(task_id, {}).get("state") == "cancelled"

    def _run_dry_run(self, task_id: str) -> None:
        self._set(task_id, state="compiling", progress=20, message="Compiling deterministic execution plan")
        time.sleep(0.12)
        if self._is_cancelled(task_id):
            return
        self._set(task_id, state="ready", progress=60, message="Plan ready; model execution remains deferred")
        time.sleep(0.12)
        if self._is_cancelled(task_id):
            return
        with self._lock:
            plan = self._tasks[task_id]["plan"]
        result = self._dry_run.execute(plan, cancel_event=self._cancel_events[task_id])
        self._set(task_id, state="dry_run_complete", progress=100, message="Dry-run complete; no model was loaded", result=result)

    def _start_real_worker(self, task_id: str) -> None:
        with self._lock:
            plan = self._tasks[task_id]["plan"]
            cancel_event = self._process_context.Event()
            messages = self._process_context.Queue()
            worker = self._process_context.Process(
                target=run_task,
                args=(plan, cancel_event, messages),
                name=f"h3-gpu-{task_id}",
            )
            self._worker_events[task_id] = cancel_event
            self._worker_messages[task_id] = messages
            self._workers[task_id] = worker
        worker.start()
        self._set(
            task_id,
            state="loading",
            progress=2,
            message="Starting isolated H3 GPU worker",
            workerPid=worker.pid,
            workerRuntimeSnapshotPath=str(self._worker_snapshot_path(task_id)),
        )
        threading.Thread(target=self._monitor_real_worker, args=(task_id,), daemon=True, name=f"h3-monitor-{task_id}").start()

    def _cleanup_evidence(self, task_id: str, worker: Any) -> Dict[str, Any]:
        with self._lock:
            prior = dict(self._worker_cleanup.get(task_id) or {})
        worker_pid = int(getattr(worker, "pid", 0) or 0)
        prior.update({
            "workerPid": worker_pid or None,
            "workerExitCode": getattr(worker, "exitcode", None),
            "workerProcessExists": bool(worker.is_alive()),
            "gpuDedicatedMiBBefore": "unavailable_parent_does_not_query_gpu",
            "gpuDedicatedMiBAfter": "unavailable_parent_does_not_query_gpu",
            "apiServiceRemainsInProcess": True,
            "servicePort": PORT,
        })
        if task_id in self._cancel_monotonic:
            prior["stopLatencySeconds"] = round(time.monotonic() - self._cancel_monotonic[task_id], 6)
        return prior

    def _finish_worker(self, task_id: str, worker: Any, message: Optional[Dict[str, Any]]) -> None:
        cancelled = self._is_cancelled(task_id)
        worker_exited_at = self._timestamp()
        evidence = self._cleanup_evidence(task_id, worker)
        evidence["workerExitedAt"] = worker_exited_at
        self._worker_cleanup[task_id] = evidence
        with self._lock:
            sampling_telemetry = list(self._tasks.get(task_id, {}).get("samplingTelemetry") or [])
        worker_snapshot = self._read_worker_snapshot(task_id)
        snapshot_telemetry = self._snapshot_telemetry(worker_snapshot)
        # The direct queue is lower latency; the worker-owned snapshot is the
        # durable source when a CUDA worker is terminated before queue flush.
        durable_telemetry = snapshot_telemetry or sampling_telemetry
        receipt = {
            "workerRuntimeSnapshot": worker_snapshot,
            "samplingTelemetry": durable_telemetry,
            "workerLifecycle": {
                "workerExited": not bool(worker.is_alive()),
                "workerExitedAt": worker_exited_at,
                "cleanupEvidence": evidence,
            },
        }
        terminal_worker_values = {"workerExitedAt": worker_exited_at}
        if message and message.get("type") == "result":
            result = dict(message.get("result") or {})
            result.update(receipt)
            if cancelled or result.get("status") == "cancelled":
                cancellation_stage = dict((result.get("runtimeStages") or {}).get("cancellation") or {})
                for field in ("cancelAcknowledgedAt", "safeBoundaryAt", "cleanupStartedAt", "cleanupFinishedAt"):
                    if cancellation_stage.get(field):
                        terminal_worker_values[field] = cancellation_stage[field]
                result.update({"status": "cancelled", "outputAuthentic": False, "cancellation": evidence})
                self._set(task_id, state="cancelled", message="Generation stopped; task worker exited and resources were released", result=result, **terminal_worker_values)
            else:
                with self._lock:
                    baseline_task = json.loads(json.dumps(self._tasks.get(task_id) or {}))
                try:
                    _validate_worker_result_identity(baseline_task.get("plan") or {}, result)
                except RuntimeError as exc:
                    result.update({"status": "error", "outputAuthentic": False, "error": str(exc)})
                    self._set(
                        task_id,
                        state="error",
                        progress=100,
                        message=f"Direct H3 worker receipt rejected: {exc}",
                        result=result,
                        **terminal_worker_values,
                    )
                else:
                    baseline_task.update({"state": "completed", "terminalAt": worker_exited_at, "completedAt": worker_exited_at})
                    result = finalize_performance_baseline(
                        baseline_task,
                        result,
                        worker_exit_code=evidence.get("workerExitCode"),
                        project_root=APP_DIR.parent,
                    )
                    baseline = dict((result.get("executionReceipt") or {}).get("performanceBaseline") or {})
                    if baseline:
                        self._persist_worker_final_baseline(task_id, baseline)
                    self._set(task_id, state="completed", progress=100, message="Direct H3 execution complete; task worker exited and result is persisted", result=result, **terminal_worker_values)
        elif message and message.get("type") == "exception":
            if cancelled:
                self._set(task_id, state="cancelled", message="Generation stopped; task worker exited and resources were released", result={
                    "status": "cancelled", "realInference": False, "outputAuthentic": False,
                    "runtimeStages": message.get("runtimeStages") or {}, "cancellation": evidence, **receipt,
                }, **terminal_worker_values)
            else:
                self._set(task_id, state="error", progress=100, message=f"Direct H3 worker error: {message.get('error')}", result={
                    "status": "error", "realInference": False, "outputAuthentic": False,
                    "error": message.get("error"), "exceptionType": message.get("exceptionType"),
                    "traceback": message.get("traceback"), "failedStage": message.get("failedStage"),
                    "runtimeStages": message.get("runtimeStages") or {},
                    "executionReceipt": message.get("executionReceipt") or {}, **receipt,
                }, **terminal_worker_values)
        elif cancelled:
            self._set(task_id, state="cancelled", message="Generation stopped; task worker exited and resources were released", result={
                "status": "cancelled", "realInference": False, "outputAuthentic": False,
                "cancellation": evidence, **receipt,
            }, **terminal_worker_values)
        else:
            self._set(task_id, state="error", progress=100, message="H3 task worker disappeared before returning a terminal result", result={
                "status": "error", "realInference": False, "outputAuthentic": False,
                "failedStage": "worker_lost", "error": "worker_lost_before_terminal_result", "worker": evidence,
                **receipt,
            }, **terminal_worker_values)
        with self._lock:
            self._workers.pop(task_id, None)
            self._worker_events.pop(task_id, None)
            messages = self._worker_messages.pop(task_id, None)
        # The worker owns CUDA.  It is already joined above, so closing only
        # releases parent-side process/queue handles; task receipts and MP4
        # paths stay persisted for UI playback without keeping a model alive.
        try:
            if messages is not None:
                messages.close()
                messages.join_thread()
            worker.close()
        except (AttributeError, OSError, ValueError):
            pass

    def _record_worker_telemetry(self, task_id: str, telemetry: Dict[str, Any]) -> None:
        """Persist worker-produced phase/step evidence even if it is later killed."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.get("state") in {"completed", "cancelled", "error", "dry_run_complete"}:
                return
            entries = task.setdefault("samplingTelemetry", [])
            entries.append(dict(telemetry))
            # Production has at most 20 sampling callbacks plus a small number
            # of phase receipts. Keep all of them rather than silently sampling.
            task["updatedAt"] = self._timestamp()
            snapshot = json.loads(json.dumps(task))
        self._persist_task(snapshot)

    def _record_worker_heartbeat(self, task_id: str, heartbeat: Dict[str, Any], progress: int, message: str) -> None:
        """Refresh active block status without persisting every denoiser boundary."""

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.get("state") in {"completed", "cancelled", "error", "dry_run_complete"}:
                return
            is_cancelling = task.get("state") == "cancelling"
            if not is_cancelling:
                task["state"] = "running"
            task["progress"] = max(int(task.get("progress", 0) or 0), int(progress or 0))
            if not is_cancelling:
                task["message"] = str(message)
            task["runtimeHeartbeat"] = dict(heartbeat or {})
            progress_receipt = heartbeat.get("progressReceipt") if isinstance(heartbeat, dict) else None
            if isinstance(progress_receipt, dict):
                task["liveProgressReceipt"] = json.loads(json.dumps(progress_receipt))
            now = self._timestamp()
            task["updatedAt"] = now
            # Force microsecond precision to ensure every heartbeat has unique timestamp
            if "." not in task["updatedAt"]:
                task["updatedAt"] = f"{task['updatedAt']}.{int(time.time() * 1000000) % 1000000:06d}"
            self._update_timing(
                task,
                state=str(task.get("state") or "running"),
                message=str(task.get("message") or ""),
                now=now,
                stage_override=str(progress_receipt.get("currentStage") or "") if isinstance(progress_receipt, dict) else None,
            )

    def _monitor_real_worker(self, task_id: str) -> None:
        terminal: Optional[Dict[str, Any]] = None
        while True:
            with self._lock:
                worker = self._workers.get(task_id)
                messages = self._worker_messages.get(task_id)
            if worker is None or messages is None:
                return
            try:
                item = messages.get(timeout=0.2)
                if item.get("type") == "progress":
                    self._set(task_id, state="running", progress=int(item.get("progress") or 0), message=str(item.get("message") or ""))
                elif item.get("type") == "telemetry":
                    self._record_worker_telemetry(task_id, dict(item.get("telemetry") or {}))
                elif item.get("type") == "heartbeat":
                    self._record_worker_heartbeat(
                        task_id,
                        dict(item.get("heartbeat") or {}),
                        int(item.get("progress") or 0),
                        str(item.get("message") or ""),
                    )
                else:
                    terminal = item
            except queue.Empty:
                pass
            if terminal is not None and not worker.is_alive():
                worker.join(timeout=0.2)
                self._finish_worker(task_id, worker, terminal)
                return
            if not worker.is_alive():
                worker.join(timeout=0.2)
                self._finish_worker(task_id, worker, terminal)
                return

    def _observe_worker_cancel(self, task_id: str) -> None:
        """Enforce a bounded stop when a worker cannot reach a cooperative callback."""
        time.sleep(CANCEL_ACK_WAIT_SECONDS)
        with self._lock:
            worker = self._workers.get(task_id)
            task = self._tasks.get(task_id)
            if worker is None or task is None or not worker.is_alive() or task.get("state") != "cancelling":
                return
            now = self._timestamp()
            task.update({
                "cancellationTimedOut": True,
                "cancelWaitExceededAt": now,
                "forceStopRequestedAt": now,
                "message": "Stopping unresponsive generation worker and releasing GPU resources",
                "updatedAt": now,
            })
            self._update_timing(task, state="cancelling", message=str(task["message"]), now=now)
            self._worker_cleanup.setdefault(task_id, {}).update({
                "workerPid": worker.pid,
                "cooperativeOnly": False,
                "hardTermination": True,
                "terminationRequestedAt": now,
            })
            snapshot = json.loads(json.dumps(task))
        self._persist_task(snapshot)

        termination_error = None
        kill_fallback_used = False
        worker_alive_after_termination = True
        try:
            worker.terminate()
            worker.join(timeout=CANCEL_TERMINATE_JOIN_SECONDS)
            worker_alive_after_termination = worker.is_alive()
            if worker_alive_after_termination:
                kill_fallback_used = True
                worker.kill()
                worker.join(timeout=CANCEL_TERMINATE_JOIN_SECONDS)
                worker_alive_after_termination = worker.is_alive()
        except (AttributeError, OSError, ValueError) as exc:
            termination_error = f"{type(exc).__name__}: {exc}"

        with self._lock:
            cleanup = self._worker_cleanup.setdefault(task_id, {})
            cleanup["killFallbackUsed"] = kill_fallback_used
            cleanup["workerAliveAfterTermination"] = worker_alive_after_termination
            if termination_error:
                cleanup["terminationError"] = termination_error

    def cancel(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(task_id)
            if task["state"] in {"dry_run_complete", "completed", "error", "cancelled"}:
                return dict(task)
            self._cancel_events[task_id].set()
            worker_event = self._worker_events.get(task_id)
            if worker_event is not None:
                worker_event.set()
            if task["state"] in {"queued", "compiling", "ready"}:
                task.update({
                    "state": "cancelled",
                    "message": "Cancelled before model execution",
                    "cancelRequestedAt": self._timestamp(),
                })
            else:
                task.update({
                    "state": "cancelling",
                    "message": "Stopping generation and releasing CPU/RAM/VRAM",
                    "cancelRequestedAt": self._timestamp(),
                })
            # A CUDA step can be uninterruptible until the worker boundary.
            # Persist the last worker-owned atomic snapshot with the cancel
            # request itself so the API immediately exposes honest evidence,
            # even if the subsequent terminate prevents another queue flush.
            worker_snapshot = self._read_worker_snapshot(task_id)
            if worker_snapshot is not None:
                task["lastWorkerRuntimeSnapshot"] = worker_snapshot
                task["lastKnownRuntimeTelemetry"] = self._snapshot_telemetry(worker_snapshot)
            now = self._timestamp()
            task["updatedAt"] = now
            self._update_timing(task, state=str(task["state"]), message=str(task["message"]), now=now)
            self._persist_task(json.loads(json.dumps(task)))
            snapshot = dict(task)
            should_enforce = task["state"] == "cancelling" and task_id in self._workers and task_id not in self._cancel_monotonic
            if should_enforce:
                self._cancel_monotonic[task_id] = time.monotonic()
                worker = self._workers[task_id]
                self._worker_cleanup[task_id] = {"workerPid": worker.pid, "cooperativeOnly": True, "hardTermination": False}
        if should_enforce:
            threading.Thread(target=self._observe_worker_cancel, args=(task_id,), daemon=True, name=f"h3-cancel-{task_id}").start()
        return snapshot

    def get(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                raise KeyError(task_id)
            return portable_project_record(json.loads(json.dumps(task)))

    @staticmethod
    def _receipt_summary(receipt: Any) -> Optional[Dict[str, Any]]:
        """Return only the receipt fields needed by a task card."""

        if not isinstance(receipt, dict):
            return None
        fields = (
            "requestedKernel", "actualKernel", "kernelIdentity", "patchedBlocks",
            "scope", "ffnEnabled", "ffnChunks", "requestedFfnChunks", "actualFfnChunks", "ffnChunk", "seed", "steps", "memoryStrategy", "fallback", "kernelBackendReceipt",
            "requestedResolution", "effectiveResolution", "resolutionMapping",
            "stage1SourceCanvas", "stage2TargetCanvas", "finalOutputCanvas", "stage1Canvas", "stage2InternalCanvas",
            "stage1AreaRatio", "stage2AreaScale", "stage2LinearScale", "canvasMultiple", "internalScale",
            "frameResize", "frameResizeVaeRoundTrip", "finalDownsample", "directStage2Output",
            "historyLookup", "resolutionBucket", "durationBucket", "decisionSource", "offloadPlan",
            "algorithmRouteFingerprint", "algorithmRoute", "algorithmProfile",
            "model", "sampler", "scheduler", "steps", "stageTimings",
            "modelFps", "outputFps", "actualOutputFps", "requestedFps", "actualModelFps",
            "temporalDensityFps", "officialModelTimebaseFps", "modelFrameCount",
            "videoLatentT", "audioLatentT", "timeScale", "routeId", "timeRemapReceipt",
            "variantId", "positionMapping", "decodeMode", "bridgedModelFps",
            "bridgedModelFrameCount", "bridgedVideoLatentT", "latentBridgeReceipt",
            "loraCandidateId", "provider", "loraRelativePath", "loraSha256",
            "loraStrength", "loraStrengthSource", "baseModel",
            "firstDenoiserBlockAt", "resolutionMismatch", "outputAuthentic",
            "performanceBaseline", "hardwareIdentity",
            "totalSteps", "splitMode", "splitStep", "executionKind", "samplerCalls", "lifecycleCalls",
            "stage1Steps", "stage2Steps", "stage1NoiseMode", "stage2NoiseMode",
            "stage1SigmaIndices", "stage2SigmaIndices", "stage1SigmaBounds", "stage2SigmaBounds",
            "stage1Seed", "stage2Seed", "stage1Output", "stage2Input", "denoisedOutputUsed",
            "latentShape", "resourceRecoveryErrors",
            "stage1UpscalerSource", "stage2DecodeSource", "finalAudioSource", "stage2AudioSampled", "stage2AudioDiscarded",
            "stage1AudioLatentShape", "stage2AudioLatentShape", "finalAudioLatentShape", "audioComparison", "tensorStats", "frameResizeVaeRoundTrip",
            "progressReceipt", "lastMeaningfulStage", "phaseStartedAt", "phaseEndedAt", "elapsed", "phaseStep", "phaseTotal", "globalStep", "globalTotal",
        )
        return {
            field: json.loads(json.dumps(receipt[field]))
            for field in fields
            if field in receipt
        }

    @classmethod
    def _task_summary(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """Build the bounded public record used by the polling task list.

        Full worker receipts can be megabytes each. A task list is refreshed
        repeatedly by the workbench, so it must expose card state rather than
        serialising every historical diagnostic on every request.
        """

        plan = task.get("plan") if isinstance(task.get("plan"), dict) else {}
        compiled = plan.get("compiled") if isinstance(plan.get("compiled"), dict) else {}
        execution = plan.get("execution") if isinstance(plan.get("execution"), dict) else {}
        result = task.get("result") if isinstance(task.get("result"), dict) else {}
        receipt = (
            result.get("executionReceipt")
            or plan.get("executionReceipt")
            or compiled.get("executionReceipt")
        )
        receipt_summary = cls._receipt_summary(receipt)
        plan_summary = {
            "compiled": {
                field: compiled[field]
                for field in ("modeLabel", "outputPath")
                if field in compiled
            },
            "execution": {
                field: execution[field]
                for field in ("steps",)
                if field in execution
            },
        }
        plan_summary["assetSummary"] = {
            "conditioningOrder": compiled.get("conditioningOrder") or [],
            "references": [
                {
                    field: item.get(field)
                    for field in ("inputIndex", "token", "kind", "name", "size", "sha256", "mediaFacts", "preprocessPlan", "warnings")
                }
                for item in (compiled.get("references") or [])
                if isinstance(item, dict)
            ],
            "referencePlanning": compiled.get("referencePlanning") or {},
        }
        if receipt_summary is not None:
            plan_summary["executionReceipt"] = receipt_summary
        result_summary = {
            field: result[field]
            for field in ("outputAuthentic", "outputPath", "error", "failedStage", "outputWidth", "outputHeight")
            if field in result
        }
        if receipt_summary is not None:
            result_summary["executionReceipt"] = receipt_summary
        summary = {
            field: task[field]
            for field in (
                "id", "createdAt", "acceptedAt", "updatedAt", "state", "progress", "message",
                "realInference", "executionMode", "postprocess1080p", "workerPid", "workerExitedAt", "terminalAt",
                "cancelRequestedAt", "cancelledAt", "completedAt", "runtimeHeartbeat", "liveProgressReceipt", "timing",
            )
            if field in task
        } | {"plan": plan_summary, "result": result_summary}
        experimental = task.get("experimental")
        if isinstance(experimental, dict) and experimental.get("enabled"):
            summary["experimental"] = json.loads(json.dumps(experimental))
            summary["experimental"]["status"] = str(task.get("state") or "queued")
        return summary

    def list(self) -> list[Dict[str, Any]]:
        with self._lock:
            tasks = [portable_project_record(self._task_summary(item)) for item in self._tasks.values()]
        return sorted(tasks, key=lambda item: item["createdAt"], reverse=True)

    def backend_health(self) -> Dict[str, Any]:
        return self._direct.health()

    def backend_capabilities(self) -> Dict[str, Any]:
        return self._direct.capabilities()

    def backend_models(self) -> Dict[str, Any]:
        return self._direct.discover_models()


STORE = TaskStore()
SWIFTVR_STORE = SwiftVRPostprocessStore(
    project_root=APP_DIR.parent,
    parent_lookup=STORE.get,
    gpu_busy=lambda: STORE.gpu_busy(),
    start_worker=False,
)
NVIDIA_VSR_STORE = NvidiaVSRPostprocessStore(
    project_root=APP_DIR.parent,
    parent_lookup=STORE.get,
    gpu_busy=lambda: STORE.gpu_busy() or SWIFTVR_STORE.is_busy(),
    start_worker=False,
)


def _multipart_boundary(content_type: str) -> bytes:
    message = BytesParser(policy=policy.default).parsebytes(
        (f"Content-Type: {content_type}\r\n\r\n").encode("utf-8")
    )
    boundary = message.get_boundary()
    if not boundary:
        raise StagingError("multipart 上传缺少边界参数")
    encoded = boundary.encode("ascii", "strict")
    if len(encoded) > 200:
        raise StagingError("multipart 上传边界过长")
    return encoded


class _LimitedRequestStream:
    """Prevent multipart parsing from reading beyond the declared request."""

    def __init__(self, source: Any, remaining: int) -> None:
        self.source = source
        self.remaining = int(remaining)

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        wanted = self.remaining if size < 0 else min(self.remaining, size)
        data = self.source.read(wanted)
        self.remaining -= len(data)
        return data

    def readline(self, limit: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        wanted = self.remaining if limit < 0 else min(self.remaining, limit)
        data = self.source.readline(wanted)
        self.remaining -= len(data)
        return data


class _MultipartPartStream:
    """Expose one multipart file body while retaining only a boundary-sized tail."""

    def __init__(self, source: _LimitedRequestStream, boundary: bytes) -> None:
        self.source = source
        self.marker = b"\r\n--" + boundary
        self.buffer = bytearray()
        self.finished = False

    def read(self, size: int = -1) -> bytes:
        if self.finished and not self.buffer:
            return b""
        wanted = 1024 * 1024 if size < 0 else max(1, int(size))
        while not self.finished and len(self.buffer) < wanted + len(self.marker):
            chunk = self.source.read(min(64 * 1024, max(1, wanted + len(self.marker) - len(self.buffer))))
            if not chunk:
                raise StagingError("上传在 multipart 结束边界前中断")
            self.buffer.extend(chunk)
            position = self.buffer.find(self.marker)
            if position >= 0:
                del self.buffer[position:]
                self.finished = True
                break
        if not self.finished:
            available = max(0, len(self.buffer) - len(self.marker) + 1)
            count = min(wanted, available)
        else:
            count = min(wanted, len(self.buffer))
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result


class Handler(BaseHTTPRequestHandler):
    server_version = "H3LocalUI/0.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        print("[h3] " + format % args)

    def _send_json(self, status: int, payload: Any) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            return json.loads(body.decode("utf-8")) if body else {}
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CompilerError("request body must contain valid JSON") from exc

    def _upload_asset(self) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise StagingError("upload must use multipart/form-data")
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise StagingError("上传请求大小无效或超过 2 GiB 限制")
        boundary = _multipart_boundary(content_type)
        limited = _LimitedRequestStream(self.rfile, length)
        opening = limited.readline(len(boundary) + 8)
        if opening.rstrip(b"\r\n") != b"--" + boundary:
            raise StagingError("上传请求的 multipart 起始边界无效")
        header_lines = []
        header_bytes = 0
        while True:
            line = limited.readline(8193)
            if not line or len(line) > 8192:
                raise StagingError("上传附件头部不完整或过大")
            if line in {b"\r\n", b"\n"}:
                break
            header_bytes += len(line)
            if header_bytes > 65536:
                raise StagingError("上传附件头部超过 64 KiB")
            header_lines.append(line)
        headers = BytesParser(policy=policy.default).parsebytes(b"".join(header_lines) + b"\r\n")
        disposition = headers.get("Content-Disposition")
        filename = disposition.params.get("filename") if disposition and getattr(disposition, "params", None) else None
        if not filename:
            raise StagingError("multipart 请求中没有文件附件")
        part = _MultipartPartStream(limited, boundary)
        return stage_stream(filename, part, None, headers.get_content_type(), probe=True)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/events":
            self._serve_events()
            return
        if path == "/api/health":
            self._send_json(200, {"ok": True, "service": "h3-local-ui", "realInference": True, "server": False})
            return
        if path == "/api/hardware-status":
            self._send_json(200, STORE.hardware_status())
            return
        if path == "/api/backend/health":
            self._send_json(200, STORE.backend_health())
            return
        if path == "/api/backend/capabilities":
            self._send_json(200, STORE.backend_capabilities())
            return
        if path == "/api/backend/models":
            self._send_json(200, {"models": STORE.backend_models()})
            return
        if path == "/api/tasks":
            self._send_json(200, {"tasks": STORE.list()})
            return
        if path == "/api/postprocess/tasks":
            self._send_json(200, {"tasks": SWIFTVR_STORE.list()})
            return
        if path == "/api/nvidia-vsr/tasks":
            self._send_json(200, {"tasks": NVIDIA_VSR_STORE.list()})
            return
        if path.startswith("/api/nvidia-vsr/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            try:
                self._send_json(200, NVIDIA_VSR_STORE.get(task_id))
            except KeyError:
                self._send_json(404, {"error": "nvidia vsr task not found"})
            return
        if path.startswith("/api/postprocess/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            try:
                self._send_json(200, SWIFTVR_STORE.get(task_id))
            except KeyError:
                self._send_json(404, {"error": "postprocess task not found"})
            return
        if path.startswith("/api/output/"):
            self._serve_output(path[len("/api/output/"):])
            return
        if path.startswith("/api/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            try:
                self._send_json(200, STORE.get(task_id))
            except KeyError:
                self._send_json(404, {"error": "task not found"})
            return
        self._serve_static(path)

    def _serve_events(self) -> None:
        """Push lightweight state deltas; the browser never polls while connected."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last_task_id = None
        last_task_updated = None
        last_task_progress = None
        last_task_message = None
        last_task_state = None
        last_heartbeat_key = None
        last_hardware_key = None
        last_hardware_at = 0.0
        try:
            while True:
                now = time.monotonic()
                tasks = STORE.list()
                active = next((item for item in tasks if item.get("state") in {"queued", "compiling", "ready", "loading", "running", "cancelling"}), None)
                # Keep the last task visible for one final terminal event so
                # the browser can mount the finished video without a refresh.
                terminal_candidate = tasks[0] if tasks and not active else None
                candidate = active or terminal_candidate
                task = STORE.get(candidate["id"]) if candidate else None
                if task:
                    task_id = task.get("id")
                    task_updated = task.get("updatedAt")
                    task_progress = task.get("progress")
                    task_message = task.get("message")
                    task_state = task.get("state")
                    # Also check runtimeHeartbeat for denoiser block updates
                    heartbeat = task.get("runtimeHeartbeat")
                    heartbeat_key = json.dumps(heartbeat, ensure_ascii=False, sort_keys=True) if heartbeat else None
                    # Send update if any key field changed
                    should_send = (
                        task_id != last_task_id
                        or task_updated != last_task_updated
                        or task_progress != last_task_progress
                        or task_message != last_task_message
                        or task_state != last_task_state
                        or heartbeat_key != last_heartbeat_key
                    )
                    if should_send:
                        last_task_id = task_id
                        last_task_updated = task_updated
                        last_task_progress = task_progress
                        last_task_message = task_message
                        last_task_state = task_state
                        last_heartbeat_key = heartbeat_key
                        telemetry = task.get("samplingTelemetry") if isinstance(task.get("samplingTelemetry"), list) else []
                        progress_events = task.get("progressEvents") if isinstance(task.get("progressEvents"), list) else []
                        latest_telemetry = telemetry[-1:] if telemetry else []
                        latest_progress = progress_events[-1:] if progress_events else []
                        delta = {
                            key: task.get(key)
                            for key in ("id", "updatedAt", "state", "progress", "message", "timing", "runtimeHeartbeat", "liveProgressReceipt")
                        }
                        if task.get("state") in {"completed", "error", "cancelled", "dry_run_complete"}:
                            delta["result"] = task.get("result")
                        delta["samplingTelemetry"] = latest_telemetry
                        delta["progressEvents"] = latest_progress
                        payload = json.dumps(delta, ensure_ascii=False, separators=(",", ":"))
                        self.wfile.write(f"event: task\ndata: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                if now - last_hardware_at >= 1.5:
                    hardware = STORE.hardware_status()
                    hardware_key = json.dumps(hardware, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    if hardware_key != last_hardware_key:
                        last_hardware_key = hardware_key
                        self.wfile.write(f"event: hardware\ndata: {hardware_key}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    last_hardware_at = now
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def do_HEAD(self) -> None:  # noqa: N802
        """Expose the same local MP4 metadata contract without reading its body."""
        path = urlparse(self.path).path
        if path.startswith("/api/output/"):
            self._serve_output(path[len("/api/output/"):])
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/compile":
                self._send_json(200, {"compiled": compile_request(self._read_json())})
                return
            if parsed.path == "/api/assets":
                self._send_json(201, {"asset": self._upload_asset()})
                return
            if parsed.path == "/api/tasks":
                payload = self._read_json()
                with GPU_ADMISSION_LOCK:
                    if str(payload.get("executionMode") or "dry-run").lower() == "real" and SWIFTVR_STORE.is_busy():
                        raise CompilerError("SwiftVR post-processing owns the single GPU queue")
                    task = STORE.create(payload)
                self._send_json(202, task)
                return
            if parsed.path.startswith("/api/tasks/") and parsed.path.endswith("/enhance"):
                task_id = parsed.path.split("/")[-2]
                payload = self._read_json()
                with GPU_ADMISSION_LOCK:
                    task = SWIFTVR_STORE.create(task_id, payload.get("targetResolution") or "1080p")
                self._send_json(202, task)
                return
            if parsed.path.startswith("/api/postprocess/tasks/") and parsed.path.endswith("/cancel"):
                task_id = parsed.path.split("/")[-2]
                self._send_json(200, SWIFTVR_STORE.cancel(task_id))
                return
            if parsed.path == "/api/nvidia-vsr/tasks":
                payload = self._read_json()
                with GPU_ADMISSION_LOCK:
                    task = NVIDIA_VSR_STORE.create(payload.get("sourceTaskId") or payload.get("taskId"))
                self._send_json(202, task)
                return
            if parsed.path.startswith("/api/nvidia-vsr/tasks/") and parsed.path.endswith("/cancel"):
                task_id = parsed.path.split("/")[-2]
                self._send_json(200, NVIDIA_VSR_STORE.cancel(task_id))
                return
            if parsed.path.startswith("/api/tasks/") and parsed.path.endswith("/cancel"):
                task_id = parsed.path.split("/")[-2]
                self._send_json(200, STORE.cancel(task_id))
                return
            self._send_json(404, {"error": "endpoint not found"})
        except CompilerError as exc:
            self._send_json(400, {"error": str(exc)})
        except StagingError as exc:
            self._send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except KeyError:
            self._send_json(404, {"error": "task not found"})
        except Exception as exc:  # keep the local server responsive for bad input
            self._send_json(500, {"error": f"server error: {exc}"})

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        if relative.startswith("api/") or "/" in relative and relative.split("/", 1)[0] == "api":
            self._send_json(404, {"error": "endpoint not found"})
            return
        file_path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR not in file_path.parents and file_path != STATIC_DIR:
            self._send_json(403, {"error": "forbidden"})
            return
        if not file_path.is_file():
            self._send_json(404, {"error": "asset not found"})
            return
        content_types = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}
        raw = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_types.get(file_path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _serve_output(self, relative: str) -> None:
        """Serve only task MP4 outputs, never arbitrary project files."""
        relative = unquote(relative)
        if not relative.lower().endswith(".mp4"):
            self._send_json(404, {"error": "output not found"})
            return
        parts = Path(relative).parts
        task = STORE.get(parts[0]) if parts else None
        if task and (task.get("experimental") or {}).get("enabled"):
            file_path = (ROOT / "_isolated_research" / "outputs" / parts[0] / Path(*parts[1:])).resolve()
            allowed_root = (ROOT / "_isolated_research" / "outputs").resolve()
        else:
            file_path = (OUTPUT_DIR / relative).resolve()
            allowed_root = OUTPUT_DIR.resolve()
        if allowed_root not in file_path.parents or not file_path.is_file():
            self._send_json(404, {"error": "output not found"})
            return
        total_size = file_path.stat().st_size
        requested_range = self.headers.get("Range")
        start, end = 0, total_size - 1
        partial = False
        if requested_range:
            if not requested_range.startswith("bytes=") or "," in requested_range:
                self.send_response(416)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{total_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start_text, separator, end_text = requested_range[6:].strip().partition("-")
            try:
                if not separator:
                    raise ValueError("range has no separator")
                if not start_text:
                    suffix_length = int(end_text)
                    if suffix_length <= 0:
                        raise ValueError("suffix must be positive")
                    start = max(0, total_size - suffix_length)
                    end = total_size - 1
                else:
                    start = int(start_text)
                    end = total_size - 1 if not end_text else min(int(end_text), total_size - 1)
                if start < 0 or start >= total_size or start > end:
                    raise ValueError("range is outside the file")
            except ValueError:
                self.send_response(416)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes */{total_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            partial = True
        content_length = end - start + 1 if total_size else 0
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total_size}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command == "HEAD":
            return
        with file_path.open("rb") as output_file:
            output_file.seek(start)
            remaining = content_length
            while remaining:
                chunk = output_file.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"H3 independent local UI: http://{HOST}:{PORT}")
    print("Direct backend is available; UI defaults to real generation, with explicit parameter preview available.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping H3 local UI")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
