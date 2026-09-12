"""Diagnostic bridge to Saganaki22's MiniMax H3 token-local FFN chunker.

This deliberately imports the upstream Apache-2.0 source in-place rather
than copying its implementation.  It is never selected by ordinary product
requests: a real same-input numerical/visual comparison is required before
any promotion.  The sparse Sol-Attn nodes in the same repository are not
loaded here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "runtime" / "third_party" / "ComfyUI-sol-attn"
SOURCE_FILE = SOURCE_ROOT / "minimax.py"
SOURCE_COMMIT = "0cd77c63fcb362392c8248e7e5f90f7ae35013b3"
SOURCE_URL = "https://github.com/Saganaki22/ComfyUI-sol-attn"
LICENSE = "Apache-2.0"


def _load_upstream_module() -> Any:
    if not SOURCE_FILE.is_file():
        raise RuntimeError(f"H3 FFN diagnostic source missing: {SOURCE_FILE}")
    spec = importlib.util.spec_from_file_location("hailuo_h3_sol_attn_minimax", SOURCE_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load H3 FFN diagnostic source: {SOURCE_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_h3_ffn_chunk() -> Dict[str, Any]:
    """Return auditable source identity without activating any patch."""

    available = SOURCE_FILE.is_file()
    license_path = SOURCE_ROOT / "LICENSE"
    return {
        "requestedSource": SOURCE_URL,
        "sourceCommit": SOURCE_COMMIT,
        "sourcePath": str(SOURCE_FILE),
        "license": LICENSE,
        "licensePath": str(license_path),
        "sourcePresent": available,
        "licensePresent": license_path.is_file(),
        "scope": "diagnostic_only",
        "productionDefault": False,
        "attentionAlgorithmChanged": False,
        "tokenLocalMlpOnly": True,
        "requiresSameInputNumericalAndVisualAB": True,
    }


def apply_h3_ffn_chunk_diagnostic(
    model: Any,
    *,
    chunks: int = 2,
    min_tokens: int = 4096,
) -> Tuple[Any, Dict[str, Any]]:
    """Apply the upstream MLP-only patch to a diagnostic model clone.

    The upstream node clones the incoming ModelPatcher and installs wrappers
    only on ``*.mlp.forward``.  It does not patch attention, scheduler,
    references, prompt conditioning, sample count, or AV VAE handling.
    """

    chunks = int(chunks)
    min_tokens = int(min_tokens)
    if chunks not in {1, 2, 4}:
        raise ValueError("H3 FFN chunk count must be 1, 2, or 4")
    if min_tokens < 1:
        raise ValueError("H3 FFN diagnostic minimum token count must be positive")
    if chunks == 1:
        patched = model
    else:
        module = _load_upstream_module()
        node = module.MiniMaxH3ChunkFeedForward()
        patched = node.patch(model, True, chunks, min_tokens)[0]
    receipt = {
        **inspect_h3_ffn_chunk(),
        "applied": patched is not model,
        "enabled": True,
        "chunks": chunks,
        "minTokens": min_tokens,
        "patchOrder": "after_kj_h3_sage_before_basic_guider",
        "sourceNode": "MiniMaxH3ChunkFeedForward",
        "mathClaim": "upstream token-local MLP partition; local GPU numerical A/B still required",
    }
    return patched, receipt
