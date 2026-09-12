"""Application adapters for dry-run and the independent direct H3 runtime."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from embedded_h3_runtime import EmbeddedH3Runtime
from embedded_h3_runtime.runner import DirectH3Runner


class H3InferenceAdapter(ABC):
    @abstractmethod
    def build_plan(self, compiled_request: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        """Return a complete execution plan without changing task state."""

    @abstractmethod
    def execute(
        self,
        plan: Dict[str, Any],
        progress: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Execute a plan or return an honest dry-run result."""


class DryRunH3Adapter(H3InferenceAdapter):
    """Safe compiler adapter; it never opens a model file."""

    def build_plan(self, compiled_request: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        from h3_compiler import build_execution_plan

        return build_execution_plan(compiled_request, task_id)

    def execute(
        self,
        plan: Dict[str, Any],
        progress: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "dry_run_complete",
            "realInference": False,
            "message": "Dry-run only: no H3 model was loaded and no video was generated.",
            "plan": plan,
        }


class EmbeddedH3BackendAdapter(H3InferenceAdapter):
    """Independent direct adapter backed by embedded low-level H3 modules."""

    def __init__(self, runtime: Optional[EmbeddedH3Runtime] = None) -> None:
        self.runtime = runtime or EmbeddedH3Runtime()
        self.runner = DirectH3Runner(self.runtime)

    def build_plan(self, compiled_request: Dict[str, Any], task_id: str) -> Dict[str, Any]:
        return self.runner.build_plan(compiled_request, task_id)

    def execute(
        self,
        plan: Dict[str, Any],
        progress: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        return self.runner.execute(plan, progress=progress, cancel_event=cancel_event)

    def health(self) -> Dict[str, Any]:
        return self.runtime.health()

    def capabilities(self) -> Dict[str, Any]:
        return self.runtime.capabilities()

    def discover_models(self) -> Dict[str, Any]:
        return self.runtime.discover_models()


# Compatibility name for earlier callers; it now points to the real direct
# backend rather than a future ComfyUI adapter.
FutureH3BackendAdapter = EmbeddedH3BackendAdapter
