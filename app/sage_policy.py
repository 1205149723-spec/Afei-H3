"""Conservative SageAttention selection for H3 long-sequence requests."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_LONG_SEQUENCE_THRESHOLD = 124
# The frame threshold remains the user-facing long-video heuristic.  H3's
# packed transformer sequence also grows with spatial tokens, references, and
# audio, so the direct runner may supply an exact token count after
# conditioning.  This is an attention-backend heuristic, not a model or
# product limit.
DEFAULT_LONG_SEQUENCE_TOKEN_THRESHOLD = 2048
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGE_RECEIPT_PATH = PROJECT_ROOT / "cache" / "sage_kernel_self_test.json"


def _relative_project_path(path: Path) -> Optional[str]:
    """Return an ASCII relative path when the process is inside this project."""

    cwd = Path.cwd().resolve()
    try:
        return os.path.relpath(path.resolve(), cwd)
    except ValueError:
        return None


def configure_sage_environment() -> Dict[str, Any]:
    """Configure process-local Linux compiler and cache paths for Sage/Triton."""

    compiler = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if compiler:
        os.environ.setdefault("CC", compiler)
    os.environ.setdefault("TEMP", str(PROJECT_ROOT / "temp"))
    os.environ.setdefault("TMP", str(PROJECT_ROOT / "temp"))
    os.environ.setdefault("TRITON_HOME", str(PROJECT_ROOT / "cache" / "triton_home"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / "cache" / "triton"))
    return {
        "bundledTcc": None,
        "configured": bool(compiler),
        "compiler": os.environ.get("CC"),
    }


def _compiler_info() -> Dict[str, Any]:
    configured = configure_sage_environment()
    configured_cc = os.environ.get("CC")
    if configured_cc:
        candidate = Path(configured_cc)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if candidate.is_file():
            name = candidate.name.lower()
            return {
                "name": "bundled-tcc" if name == "tcc.exe" else candidate.name,
                "path": str(candidate),
                "source": "triton-windows bundled runtime/tcc",
            }
    for name in ("cl", "gcc", "clang"):
        found = shutil.which(name)
        if found:
            return {"name": name, "path": found, "source": "process PATH"}
    if configured.get("bundledTcc"):
        return {"name": "bundled-tcc", "path": configured["bundledTcc"], "source": "D-drive package"}
    return {"name": None, "path": None, "source": None}


def _read_kernel_receipt() -> Optional[Dict[str, Any]]:
    try:
        receipt = json.loads(SAGE_RECEIPT_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if receipt.get("ok") is not True or receipt.get("kernelFunction") != "sageattention.core.sageattn":
        return None
    return receipt


def run_sage_kernel_self_test() -> Dict[str, Any]:
    """Run one tiny real CUDA kernel test and persist its D-drive receipt."""

    configure_sage_environment()
    started = time.time()
    try:
        import torch
        from sageattention import sageattn

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.reset_peak_memory_stats()
        q = torch.randn((1, 2, 32, 64), device="cuda", dtype=torch.float16)
        k = torch.randn((1, 2, 32, 64), device="cuda", dtype=torch.float16)
        v = torch.randn((1, 2, 32, 64), device="cuda", dtype=torch.float16)
        first_started = time.time()
        first = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()
        first_seconds = time.time() - first_started
        second_started = time.time()
        second = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        torch.cuda.synchronize()
        second_seconds = time.time() - second_started
        receipt = {
            "ok": True,
            "kernelFunction": f"{sageattn.__module__}.sageattn",
            "firstCallSeconds": round(first_seconds, 4),
            "secondCallSeconds": round(second_seconds, 4),
            "shape": list(first.shape),
            "device": str(first.device),
            "torch": getattr(torch, "__version__", "unknown"),
            "triton": getattr(__import__("triton"), "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(0),
            "maxMemoryBytes": int(torch.cuda.max_memory_allocated()),
            "elapsedSeconds": round(time.time() - started, 4),
            "compiler": _compiler_info(),
            "allowCompile": False,
            "patchScope": "denoiser_only",
        }
    except Exception as exc:
        receipt = {
            "ok": False,
            "kernelFunction": "sageattention.core.sageattn",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsedSeconds": round(time.time() - started, 4),
            "compiler": _compiler_info(),
        }
        SAGE_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SAGE_RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    SAGE_RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAGE_RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def detect_sage() -> Dict[str, Any]:
    """Report package availability without changing global attention hooks."""

    configure_sage_environment()
    package_available = importlib.util.find_spec("sageattention") is not None
    triton_available = importlib.util.find_spec("triton") is not None
    hook_available = False
    hook_source = None
    kernel_source = None
    hook_error = None
    try:
        from sageattention import sageattn

        kernel_source = f"{sageattn.__module__}.sageattn"
        import comfy.ldm.modules.attention as comfy_attention

        hook = getattr(comfy_attention, "attention_sage", None)
        hook_available = callable(hook)
        hook_source = f"{hook.__module__}.attention_sage" if hook_available else None
    except Exception as exc:
        hook_error = f"{type(exc).__name__}: {exc}"

    compiler = _compiler_info()
    host_compiler = compiler["name"]
    kernel_receipt = _read_kernel_receipt()
    try:
        from h3_kj_adapter import inspect_h3_memory_efficient_sage

        h3_memory_efficient = inspect_h3_memory_efficient_sage()
    except Exception as exc:
        h3_memory_efficient = {
            "available": False,
            "compatibilityFailure": f"H3 KJ adapter probe failed: {type(exc).__name__}: {exc}",
        }
    # nvcc is deliberately not counted as a Triton host compiler: the actual
    # Triton Windows command line contains MSVC/GCC flags that nvcc rejects.
    compiler_error = None if host_compiler else "no bundled-tcc/cl/gcc/clang compiler; nvcc is not Triton-compatible here"
    verified_hook = bool(package_available and triton_available and hook_available)
    usable = bool(verified_hook and kernel_receipt)
    if not package_available or not triton_available:
        failure = "SageAttention or Triton package is not importable"
    elif not verified_hook:
        failure = hook_error or "H3 denoiser hook is unavailable"
    elif not kernel_receipt:
        failure = compiler_error or "Sage kernel self-test has not succeeded"
    else:
        failure = None
    return {
        "packageAvailable": package_available,
        "tritonAvailable": triton_available,
        "verifiedDenoiserHook": verified_hook,
        "available": usable,
        "kernelFunction": kernel_source,
        "denoiserHook": hook_source,
        "hostCompiler": host_compiler,
        "compilerPath": compiler["path"],
        "kernelSelfTest": kernel_receipt,
        "compatibilityFailure": failure,
        "h3MemoryEfficientSage": h3_memory_efficient,
        "allowCompile": False,
        "patchScope": "denoiser_only",
    }


def apply_sage_denoiser_hook(model_patcher: Any) -> Dict[str, Any]:
    """Install Sage on one ModelPatcher's denoiser options only.

    The hook is the same function-level implementation used by the embedded
    low-level attention module. No global attention function is replaced.
    """

    from sageattention import sageattn
    import comfy.ldm.modules.attention as comfy_attention

    hook = getattr(comfy_attention, "attention_sage", None)
    if not callable(hook):
        raise RuntimeError("Comfy low-level attention_sage hook is unavailable")
    options = getattr(model_patcher, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("denoiser does not expose model_options")
    transformer_options = options.setdefault("transformer_options", {})
    if "optimized_attention_override" in transformer_options:
        raise RuntimeError("denoiser already has optimized_attention_override; refusing duplicate patch")

    def override(_original: Any, *args: Any, **kwargs: Any) -> Any:
        # attention_sage is wrapped by the low-level module. The wrapper sees
        # _inside_attn_wrapper and therefore does not recurse into this hook.
        return hook(*args, **kwargs)

    transformer_options["optimized_attention_override"] = override
    transformer_options["h3_sage_attention"] = {
        "kernel": f"{sageattn.__module__}.sageattn",
        "hook": f"{hook.__module__}.attention_sage",
        "allow_compile": False,
        "scope": "denoiser_only",
        "globalPatch": False,
    }
    return transformer_options["h3_sage_attention"]


def apply_chunked_sdpa_denoiser_hook(model_patcher: Any, query_chunk_tokens: int = 2048) -> Dict[str, Any]:
    """Use exact query-row chunks with PyTorch's memory-efficient SDPA.

    H3 denoiser attention is non-causal and unmasked.  Splitting only the
    query rows computes the same attention over the complete key/value set;
    it does not shorten or window the sequence.  This is a memory strategy for
    packed long references, not a quality or capability limit.
    """

    import torch
    import torch.nn.functional as F

    options = getattr(model_patcher, "model_options", None)
    if not isinstance(options, dict):
        raise RuntimeError("denoiser does not expose model_options")
    transformer_options = options.setdefault("transformer_options", {})
    if "optimized_attention_override" in transformer_options:
        raise RuntimeError("denoiser already has optimized_attention_override; refusing duplicate patch")
    chunk = max(256, int(query_chunk_tokens))

    def override(original: Any, q: Any, k: Any, v: Any, heads: int, mask: Any = None,
                 attn_precision: Any = None, skip_reshape: bool = False,
                 skip_output_reshape: bool = False, **kwargs: Any) -> Any:
        if mask is not None or not skip_reshape or q.ndim != 4 or q.device.type != "cuda":
            return original(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
        scale = kwargs.get("scale", q.shape[-1] ** -0.5)
        sdpa_kwargs = {"dropout_p": 0.0, "is_causal": False}
        if scale is not None:
            sdpa_kwargs["scale"] = scale
        if kwargs.get("enable_gqa", False):
            sdpa_kwargs["enable_gqa"] = True
        out = torch.empty_like(q)
        for start in range(0, q.shape[2], chunk):
            q_part = q[:, :, start:start + chunk, :]
            try:
                part = F.scaled_dot_product_attention(q_part, k, v, **sdpa_kwargs)
            except TypeError:
                sdpa_kwargs.pop("scale", None)
                sdpa_kwargs.pop("enable_gqa", None)
                part = F.scaled_dot_product_attention(q_part, k, v, **sdpa_kwargs)
            out[:, :, start:start + q_part.shape[2], :].copy_(part)
            del part, q_part
        if skip_output_reshape:
            return out
        return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], heads * q.shape[-1])

    transformer_options["optimized_attention_override"] = override
    transformer_options["h3_chunked_sdpa_attention"] = {
        "backend": "torch_scaled_dot_product_attention",
        "queryChunkTokens": chunk,
        "scope": "denoiser_only",
        "globalPatch": False,
        "maskedAttention": False,
        "fullKeyValueSequence": True,
    }
    return transformer_options["h3_chunked_sdpa_attention"]


def choose_sage_policy(
    output_frames: int,
    reference_frames: int = 0,
    requested: Any = "auto",
    threshold: int = DEFAULT_LONG_SEQUENCE_THRESHOLD,
    available: Optional[bool] = None,
    estimated_tokens: Optional[int] = None,
    token_threshold: int = DEFAULT_LONG_SEQUENCE_TOKEN_THRESHOLD,
) -> Dict[str, Any]:
    """Return an auditable requested/selected/available/fallback decision."""

    mode = str(requested or "auto").lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError("sageAttention must be auto, on, or off")
    threshold = int(threshold)
    if threshold < 1:
        raise ValueError("sage threshold must be positive")
    token_threshold = int(token_threshold)
    if token_threshold < 1:
        raise ValueError("sage token threshold must be positive")
    effective = int(output_frames) + int(reference_frames)
    estimated = None if estimated_tokens is None else int(estimated_tokens)
    if estimated is not None and estimated < 1:
        raise ValueError("estimated Sage sequence tokens must be positive")
    detected = detect_sage()
    usable = detected["available"] if available is None else bool(available)
    selected = False
    reason = ""
    frame_long = effective >= threshold
    token_long = estimated is not None and estimated >= token_threshold
    if mode == "off":
        reason = "disabled by request"
    elif not usable:
        reason = "SageAttention is unavailable or lacks a verified H3 denoiser hook"
    elif not frame_long and not token_long:
        if estimated is None:
            reason = f"effective sequence {effective} is below threshold {threshold}"
        else:
            reason = (
                f"effective sequence {effective} is below threshold {threshold} and "
                f"packed sequence {estimated} tokens is below threshold {token_threshold}"
            )
    else:
        selected = True

    return {
        "requested": mode != "off",
        "requestedMode": mode,
        "selected": "sage" if selected else "torch_sdpa",
        "available": usable,
        "fallback_reason": reason or None,
        "effectiveSequenceFrames": effective,
        "thresholdFrames": threshold,
        "estimatedSequenceTokens": estimated,
        "tokenThresholdTokens": token_threshold,
        "selectionReason": (
            "packed token sequence reached the attention backend threshold"
            if token_long and not frame_long else
            "effective frame sequence reached the attention backend threshold"
            if frame_long else None
        ),
        "allow_compile": False,
        "patchScope": "denoiser_only",
        "globalPatch": False,
        "packageAvailable": detected["packageAvailable"],
        "tritonAvailable": detected["tritonAvailable"],
        "verifiedDenoiserHook": detected["verifiedDenoiserHook"],
        "kernelFunction": detected["kernelFunction"],
        "denoiserHook": detected["denoiserHook"],
        "hostCompiler": detected["hostCompiler"],
        "compatibilityFailure": detected["compatibilityFailure"],
        "h3MemoryEfficientSage": detected.get("h3MemoryEfficientSage"),
    }
