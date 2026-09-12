"""Independent H3 runtime bridge.

This package imports only the low-level model/quantization modules from the
project-local read-only source tree under ``runtime``. It never starts a
ComfyUI server, imports its web application, or executes a workflow graph.
"""

from .loader import EmbeddedH3Runtime, H3RuntimeError

__all__ = ["EmbeddedH3Runtime", "H3RuntimeError"]
