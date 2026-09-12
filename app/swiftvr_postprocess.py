"""Independent SwiftVR post-processing task contract.

This module never imports SwiftVR or torch. The web service owns lightweight
validation and lifecycle state; a separately installed Python 3.10 process
owns the 20 GB model and all GPU work.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional


SWIFTVR_CODE_COMMIT = "5ca168cef6ca7200f135fdfea85e5e13d12c5b53"
SWIFTVR_MODEL_REVISION = "743ed2530c550764905400f38eb6cc41af5abc80"
SWIFTVR_MODEL_FILES = {
    "prompt_embedding.safetensors": "cc4cf7b9aa9def4026bb5952b8aaec846ffc83eee43cafff0d3796b7e9fdf922",
    "reae.safetensors": "c915205d1833677b6887e2fdf675499d3fc781af0c644c99330f7d22fd855514",
    "transformer/config.json": "dc00d9866e72cf77db6b531aaa33be4dc7148fef9338442bd0ae9181f7075e9b",
    "transformer/diffusion_pytorch_model.safetensors": "f7ade5b8f7f4ff8b4e26a581772ebe5bcfb6a619ece2dd3483c5395c2d7e1a31",
}
TARGETS = {
    "1080p": {"id": "1080p", "width": 1920, "height": 1088},
    "2k": {"id": "2k", "width": 2560, "height": 1440},
}
ACTIVE_STATES = {"queued", "preflight", "loading", "restoring", "encoding"}
TERMINAL_STATES = {"completed", "failed", "cancelled"}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


class SwiftVRPostprocessStore:
    def __init__(
        self,
        *,
        project_root: Path,
        parent_lookup: Callable[[str], Dict[str, Any]],
        gpu_busy: Callable[[], bool],
        start_worker: bool = True,
        python_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(project_root).resolve()
        self.parent_lookup = parent_lookup
        self.gpu_busy = gpu_busy
        self.start_worker = bool(start_worker)
        self.python_path = Path(python_path) if python_path else self.root / "runtime" / "swiftvr_env" / "python.exe"
        self.state_dir = self.root / "temp" / "swiftvr_tasks"
        self.runtime_dir = self.root / "temp" / "swiftvr_runtime"
        self.output_dir = self.root / "output"
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._restore()

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _persist(self, task: Dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        target = self.state_dir / f"{task['id']}.json"
        temporary = target.with_name(f".{target.stem}.{threading.get_ident()}.tmp")
        temporary.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _restore(self) -> None:
        if not self.state_dir.is_dir():
            return
        for path in self.state_dir.glob("*.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"))
                if task.get("id") != path.stem:
                    continue
                if task.get("state") in ACTIVE_STATES:
                    task["state"] = "cancelled"
                    task["message"] = "服务恢复时 SwiftVR worker 已不存在"
                    task["terminalAt"] = self._now()
                    task["receipt"]["workerExit"] = {"code": None, "reason": "service_recovery_missing_worker"}
                    self._persist(task)
                self._tasks[task["id"]] = task
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue

    def is_busy(self) -> bool:
        with self._lock:
            return any(task.get("state") in ACTIVE_STATES for task in self._tasks.values())

    def _validated_source(self, source_task_id: str) -> tuple[Dict[str, Any], Path, str]:
        try:
            parent = self.parent_lookup(source_task_id)
        except KeyError as exc:
            raise ValueError("source task not found") from exc
        result = parent.get("result") or {}
        if parent.get("state") != "completed":
            raise ValueError("source task must be completed")
        if result.get("outputAuthentic") is not True:
            raise ValueError("source task must have authentic output")
        probe = result.get("mediaProbe") or result.get("outputMedia") or {}
        relative = str(result.get("outputPath") or "")
        if not relative:
            raise ValueError("source task has no recorded output path")
        source = (self.root / relative).resolve()
        try:
            source.relative_to(self.output_dir.resolve())
        except ValueError as exc:
            raise ValueError("source MP4 is outside project output") from exc
        if not source.is_file() or source.suffix.lower() != ".mp4":
            raise ValueError("source MP4 is missing")
        actual = sha256_file(source)
        expected = str(result.get("outputSha256") or result.get("sha256") or "").lower()
        if expected and actual != expected:
            raise ValueError("source MP4 SHA256 changed")
        if probe.get("decodable") is not True or len(expected) != 64:
            ffprobe = shutil.which("ffprobe")
            if not ffprobe:
                raise ValueError("source task is missing evidence and ffprobe is unavailable")
            command = [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(source)]
            checked = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if checked.returncode:
                raise ValueError("source task is missing decodable media evidence")
            streams = json.loads(checked.stdout).get("streams") or []
            if not streams or int(streams[0].get("nb_read_frames") or 0) <= 0:
                raise ValueError("source task is missing decodable media evidence")
        return parent, source, actual

    def create(self, source_task_id: str, target_resolution: str) -> Dict[str, Any]:
        target_id = str(target_resolution or "").lower()
        if target_id not in TARGETS:
            raise ValueError("targetResolution must be 1080p or 2k")
        with self._lock:
            if self.gpu_busy() or self.is_busy():
                raise RuntimeError("GPU queue is busy")
        parent, source, source_hash = self._validated_source(str(source_task_id))
        task_id = f"svr{uuid.uuid4().hex[:12]}"
        now = self._now()
        output = self.output_dir / task_id / "swiftvr_restored.mp4"
        task = {
            "id": task_id,
            "taskType": "swiftvr_postprocess",
            "parentTaskId": parent["id"],
            "sourceTaskId": parent["id"],
            "sourceMp4Sha256": source_hash,
            "sourcePath": _portable_path(self.root, source),
            "targetResolution": dict(TARGETS[target_id]),
            "outputPath": _portable_path(self.root, output),
            "state": "queued",
            "progress": 0,
            "message": "高清修复任务已进入单 GPU 队列",
            "createdAt": now,
            "acceptedAt": now,
            "updatedAt": now,
            "outputAuthentic": False,
            "receipt": {
                "taskId": task_id,
                "parentTaskId": parent["id"],
                "sourceTaskId": parent["id"],
                "sourceMp4Sha256": source_hash,
                "targetResolution": dict(TARGETS[target_id]),
                "swiftvrCodeCommit": SWIFTVR_CODE_COMMIT,
                "swiftvrModelRevision": SWIFTVR_MODEL_REVISION,
                "modelFilesSha256": dict(SWIFTVR_MODEL_FILES),
                "attentionBackend": "sdpa",
                "dtype": "bfloat16",
                "fallback": False,
                "audioPolicy": "source_stream_copy_preferred",
                "sourceEvidence": "fresh_sha256_and_ffprobe" if not ((parent.get("result") or {}).get("outputSha256") and ((parent.get("result") or {}).get("mediaProbe") or {}).get("decodable") is True) else "parent_receipt",
                "outputAuthentic": False,
                "stageTimingsSeconds": {},
                "workerExit": None,
            },
        }
        with self._lock:
            self._tasks[task_id] = task
            self._persist(task)
        if self.start_worker:
            self._launch(task_id)
        return self.get(task_id)

    def _launch(self, task_id: str) -> None:
        if not self.python_path.is_file():
            self._finish_failed(task_id, "SwiftVR 隔离环境不存在", worker_exit=None)
            return
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.runtime_dir / f"{task_id}.json"
        task = self.get(task_id)
        manifest.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
        worker_script = self.root / "app" / "swiftvr_worker.py"
        process = subprocess.Popen(
            [str(self.python_path), str(worker_script), "--manifest", str(manifest)],
            cwd=str(self.root),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self._lock:
            self._processes[task_id] = process
            self._tasks[task_id]["workerPid"] = process.pid
            self._persist(self._tasks[task_id])
        threading.Thread(target=self._monitor, args=(task_id, process, manifest), daemon=True, name=f"swiftvr-{task_id}").start()

    def _monitor(self, task_id: str, process: subprocess.Popen, manifest: Path) -> None:
        last_payload = None
        while process.poll() is None:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                if payload != last_payload:
                    self._merge_worker(task_id, payload)
                    last_payload = payload
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            time.sleep(0.25)
        try:
            self._merge_worker(task_id, json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError):
            self._finish_failed(task_id, "SwiftVR worker 未写出有效回执", worker_exit=process.returncode)
        with self._lock:
            self._processes.pop(task_id, None)

    def _merge_worker(self, task_id: str, payload: Dict[str, Any]) -> None:
        if payload.get("id") != task_id:
            return
        with self._lock:
            current = self._tasks.get(task_id)
            if not current or current.get("state") in TERMINAL_STATES:
                return
            current.update(payload)
            current["updatedAt"] = self._now()
            self._persist(current)

    def _finish_failed(self, task_id: str, message: str, worker_exit: Optional[int]) -> None:
        with self._lock:
            task = self._tasks[task_id]
            task.update({"state": "failed", "message": message, "progress": task.get("progress", 0), "terminalAt": self._now(), "outputAuthentic": False})
            task["receipt"]["workerExit"] = {"code": worker_exit, "reason": message}
            task["receipt"]["outputAuthentic"] = False
            self._persist(task)

    def cancel(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            task = self._tasks[task_id]
            if task.get("state") in TERMINAL_STATES:
                return json.loads(json.dumps(task))
            process = self._processes.get(task_id)
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            task.update({"state": "cancelled", "message": "高清修复已取消，原始视频未受影响", "terminalAt": self._now(), "outputAuthentic": False})
            task["receipt"]["workerExit"] = {"code": getattr(process, "returncode", None), "reason": "user_cancelled"}
            task["receipt"]["outputAuthentic"] = False
            self._persist(task)
            return json.loads(json.dumps(task))

    def get(self, task_id: str) -> Dict[str, Any]:
        with self._lock:
            if task_id not in self._tasks:
                raise KeyError(task_id)
            return json.loads(json.dumps(self._tasks[task_id]))

    def list(self) -> list[Dict[str, Any]]:
        with self._lock:
            return sorted((json.loads(json.dumps(task)) for task in self._tasks.values()), key=lambda task: task.get("createdAt", ""), reverse=True)
