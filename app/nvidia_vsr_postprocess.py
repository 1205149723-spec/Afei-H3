"""Independent NVIDIA Video Super Resolution post-processing store."""
from __future__ import annotations
import hashlib, json, os, subprocess, threading, time, uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

ACTIVE_STATES = {"queued", "decoding", "loading", "processing", "encoding"}
TERMINAL_STATES = {"completed", "failed", "cancelled"}
TARGET = {"id": "1080p", "width": 1920, "height": 1088}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _portable(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        output_root = (root / "output").resolve()
        return (Path("output") / path.resolve().relative_to(output_root)).as_posix()

class NvidiaVSRPostprocessStore:
    def __init__(self, *, project_root: Path, parent_lookup: Callable[[str], Dict[str, Any]], gpu_busy: Callable[[], bool], start_worker: bool = True, python_path: Optional[Path] = None) -> None:
        self.root = Path(project_root).resolve(); self.parent_lookup = parent_lookup; self.gpu_busy = gpu_busy
        self.start_worker = bool(start_worker); self.python_path = Path(python_path) if python_path else Path(os.environ.get("NVIDIA_VSR_PYTHON", os.sys.executable))
        self.state_dir = self.root / "temp" / "nvidia_vsr_tasks"; self.runtime_dir = self.root / "temp" / "nvidia_vsr_runtime"; self.output_dir = self.root / "output"
        self._lock = threading.RLock(); self._tasks: Dict[str, Dict[str, Any]] = {}; self._processes: Dict[str, subprocess.Popen] = {}; self._restore()
    @staticmethod
    def _now() -> str: return time.strftime("%Y-%m-%dT%H:%M:%S%z")
    def _persist(self, task: Dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True); target = self.state_dir / f"{task['id']}.json"; tmp = target.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(target)
    def _restore(self) -> None:
        if not self.state_dir.is_dir(): return
        for path in self.state_dir.glob("*.json"):
            try:
                task = json.loads(path.read_text(encoding="utf-8"));
                if task.get("id") != path.stem: continue
                if task.get("state") in ACTIVE_STATES:
                    task.update(state="cancelled", message="服务恢复时 NVIDIA VSR worker 已不存在", terminalAt=self._now()); task["receipt"]["workerExit"] = {"code": None, "reason": "service_recovery_missing_worker"}; self._persist(task)
                self._tasks[task["id"]] = task
            except (OSError, ValueError, TypeError, json.JSONDecodeError): pass
    def is_busy(self) -> bool:
        with self._lock: return any(t.get("state") in ACTIVE_STATES for t in self._tasks.values())
    def _source(self, source_task_id: str):
        try: parent = self.parent_lookup(source_task_id)
        except KeyError as exc: raise ValueError("source task not found") from exc
        result = parent.get("result") or {}
        if parent.get("state") not in {"completed", "dry_run_complete"} or result.get("outputAuthentic") is not True: raise ValueError("source task must have authentic completed output")
        rel = str(result.get("outputPath") or ""); source = (self.root / rel).resolve()
        try: source.relative_to(self.output_dir.resolve())
        except ValueError as exc: raise ValueError("source MP4 is outside project output") from exc
        if not source.is_file() or source.suffix.lower() != ".mp4": raise ValueError("source MP4 is missing")
        actual = sha256_file(source); expected = str(result.get("outputSha256") or result.get("sha256") or "").lower()
        if expected and (len(expected) != 64 or actual != expected): raise ValueError("source MP4 SHA256 evidence mismatch")
        return parent, source, actual
    def create(self, source_task_id: str) -> Dict[str, Any]:
        with self._lock:
            if self.gpu_busy() or self.is_busy(): raise RuntimeError("GPU queue is busy")
        parent, source, source_hash = self._source(str(source_task_id)); task_id = f"nvsr{uuid.uuid4().hex[:12]}"; output = self.output_dir / task_id / "nvidia_vsr.mp4"; now = self._now()
        task = {"id": task_id, "taskType": "nvidia_vsr_postprocess", "parentTaskId": parent["id"], "sourceTaskId": parent["id"], "sourcePath": _portable(self.root, source), "sourceMp4Sha256": source_hash, "targetResolution": dict(TARGET), "outputPath": _portable(self.root, output), "state": "queued", "progress": 0, "message": "NVIDIA VSR 已进入单 GPU 队列", "createdAt": now, "acceptedAt": now, "updatedAt": now, "outputAuthentic": False, "receipt": {"taskId": task_id, "parentTaskId": parent["id"], "sourceMp4Sha256": source_hash, "targetResolution": dict(TARGET), "quality": "ULTRA", "fallback": False, "audioPolicy": "source_stream_copy_preferred", "outputAuthentic": False, "stageTimingsSeconds": {}, "workerExit": None}}
        with self._lock: self._tasks[task_id] = task; self._persist(task)
        if self.start_worker: self._launch(task_id)
        return self.get(task_id)
    def _launch(self, task_id: str) -> None:
        if not self.python_path.is_file(): return self._finish_failed(task_id, "NVIDIA VSR Python 环境不存在", None)
        self.runtime_dir.mkdir(parents=True, exist_ok=True); manifest = self.runtime_dir / f"{task_id}.json"; manifest.write_text(json.dumps(self.get(task_id), ensure_ascii=False, indent=2), encoding="utf-8")
        process = subprocess.Popen([str(self.python_path), str(self.root / "app" / "nvidia_vsr_worker.py"), "--manifest", str(manifest)], cwd=str(self.root), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        with self._lock: self._processes[task_id] = process; self._tasks[task_id]["workerPid"] = process.pid; self._persist(self._tasks[task_id])
        threading.Thread(target=self._monitor, args=(task_id, process, manifest), daemon=True).start()
    def _monitor(self, task_id, process, manifest):
        while process.poll() is None:
            try: self._merge(task_id, json.loads(manifest.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError): pass
            time.sleep(.25)
        try: self._merge(task_id, json.loads(manifest.read_text(encoding="utf-8")))
        except (OSError, ValueError, json.JSONDecodeError): self._finish_failed(task_id, "NVIDIA VSR worker 未写出有效回执", process.returncode)
        with self._lock: self._processes.pop(task_id, None)
    def _merge(self, task_id, payload):
        if payload.get("id") != task_id: return
        with self._lock:
            if task_id not in self._tasks or self._tasks[task_id].get("state") in TERMINAL_STATES: return
            self._tasks[task_id].update(payload, updatedAt=self._now()); self._persist(self._tasks[task_id])
    def _finish_failed(self, task_id, message, code):
        with self._lock:
            task = self._tasks[task_id]; task.update(state="failed", message=message, terminalAt=self._now(), outputAuthentic=False); task["receipt"].update(workerExit={"code": code, "reason": message}, outputAuthentic=False); self._persist(task)
    def get(self, task_id):
        with self._lock:
            if task_id not in self._tasks: raise KeyError(task_id)
            return json.loads(json.dumps(self._tasks[task_id]))
    def list(self):
        with self._lock: return sorted((json.loads(json.dumps(t)) for t in self._tasks.values()), key=lambda t: t.get("createdAt", ""), reverse=True)
    def cancel(self, task_id):
        with self._lock:
            task = self._tasks[task_id]; process = self._processes.get(task_id)
            if task.get("state") in TERMINAL_STATES: return json.loads(json.dumps(task))
            if process and process.poll() is None: process.terminate()
            task.update(state="cancelled", message="NVIDIA VSR 已取消，原始视频未受影响", terminalAt=self._now(), outputAuthentic=False); task["receipt"].update(workerExit={"code": getattr(process, "returncode", None), "reason": "user_cancelled"}, outputAuthentic=False); self._persist(task); return json.loads(json.dumps(task))
