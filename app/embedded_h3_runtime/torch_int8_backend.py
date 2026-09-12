"""Independent H3 INT8 ConvRot backend using Torch's native INT8 GEMM.

The H3 weights are already INT8 and the existing comfy-kitchen code provides
the exact ConvRot Hadamard transform and row-wise activation quantizer.  This
adapter keeps those operations and replaces only the INT8 matrix multiply with
``torch._int_mm``.  It is registered in the app process as a small backend;
the ComfyUI source tree and the installed package remain read-only.

The non-contiguous ``weight.T`` view is intentional.  Torch 2.11/cu128 on the
target RTX 5090 accepts it and avoids making a full transposed copy of every
20-GB model's weights on each call.  If the runtime rejects that layout, the
adapter reports a fallback and calls the existing Triton implementation.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Callable, Dict

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - the reliable Torch path remains usable without Triton
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _h3_int32_dequant_epilogue_kernel(
        acc_ptr,
        row_scale_ptr,
        col_scale_ptr,
        bias_ptr,
        out_ptr,
        m,
        n,
        acc_stride_m,
        acc_stride_n,
        out_stride_m,
        out_stride_n,
        HAS_COLUMN_SCALE: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Exact INT32 -> FP32 dequantization and BF16/FP32 store.

        This is only the epilogue after Torch's native ``_int_mm``.  It uses
        the same operation order as the existing comfy-kitchen INT8 kernel:
        accumulator -> float32, row activation scale, weight scale, bias,
        then store to the requested output dtype.  It does not quantize or
        alter weights.
        """

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < m
        col_mask = cols < n
        mask = row_mask[:, None] & col_mask[None, :]
        acc = tl.load(
            acc_ptr + rows[:, None] * acc_stride_m + cols[None, :] * acc_stride_n,
            mask=mask,
            other=0,
        )
        row_scale = tl.load(row_scale_ptr + rows, mask=row_mask, other=0.0)
        if HAS_COLUMN_SCALE:
            col_scale = tl.load(col_scale_ptr + cols, mask=col_mask, other=0.0)
        else:
            col_scale = tl.load(col_scale_ptr)
        result = acc.to(tl.float32) * row_scale[:, None]
        result = result * col_scale[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
            result = result + bias[None, :]
        tl.store(
            out_ptr + rows[:, None] * out_stride_m + cols[None, :] * out_stride_n,
            result,
            mask=mask,
        )


def _torch_int8_epilogue(
    torch: Any,
    acc: Any,
    x_scale: Any,
    weight_scale: Any,
    bias: Any,
    out_dtype: Any,
) -> Any:
    """Reference epilogue retained as the exact fallback for the fused path."""

    m, n = int(acc.shape[0]), int(acc.shape[1])
    chunk_size = max(1, min(m, (256 * 1024 * 1024) // max(1, n * 4)))
    result = torch.empty((m, n), device=acc.device, dtype=out_dtype)
    row_scales = x_scale.reshape(-1)
    scale_cols = weight_scale.reshape(1, -1)
    bias_value = None
    if bias is not None:
        bias_value = bias.to(device=acc.device, dtype=torch.float32).reshape(1, -1)
    for start in range(0, m, chunk_size):
        stop = min(start + chunk_size, m)
        scaled = acc[start:stop].to(torch.float32) * row_scales[start:stop, None].to(torch.float32)
        if weight_scale.numel() == 1:
            scaled = scaled * scale_cols[0, 0]
        else:
            scaled = scaled * scale_cols
        if bias_value is not None:
            scaled = scaled + bias_value
        result[start:stop] = scaled.to(dtype=out_dtype)
    return result


def _triton_int8_epilogue(
    torch: Any,
    acc: Any,
    x_scale: Any,
    weight_scale: Any,
    bias: Any,
    out_dtype: Any,
) -> Any:
    """Run the exact epilogue with the bundled Triton runtime."""

    if triton is None or tl is None:
        raise RuntimeError("Triton is unavailable for the INT8 epilogue")
    if not acc.is_cuda or acc.dtype != torch.int32:
        raise TypeError(f"expected CUDA int32 accumulator, got {acc.dtype}/{acc.device}")
    m, n = int(acc.shape[0]), int(acc.shape[1])
    x_scale = x_scale.reshape(-1).contiguous()
    weight_scale = weight_scale.reshape(-1).contiguous()
    if x_scale.numel() != m:
        raise ValueError(f"activation scale count {x_scale.numel()} does not match rows {m}")
    if weight_scale.numel() not in (1, n):
        raise ValueError(f"weight scale count {weight_scale.numel()} does not match columns {n}")
    if bias is not None:
        bias = bias.to(device=acc.device, dtype=torch.float32).reshape(-1).contiguous()
        if bias.numel() != n:
            raise ValueError(f"bias count {bias.numel()} does not match columns {n}")
    output = torch.empty((m, n), device=acc.device, dtype=out_dtype)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 128))
    _h3_int32_dequant_epilogue_kernel[grid](
        acc,
        x_scale,
        weight_scale,
        bias if bias is not None else acc,
        output,
        m,
        n,
        acc.stride(0),
        acc.stride(1),
        output.stride(0),
        output.stride(1),
        HAS_COLUMN_SCALE=weight_scale.numel() != 1,
        HAS_BIAS=bias is not None,
        BLOCK_M=64,
        BLOCK_N=128,
        num_warps=4,
        num_stages=2,
    )
    return output


class TorchInt8BackendState:
    """Request-process receipt for the native Torch INT8 path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.successes = 0
        self.fallbacks = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.shapes: Counter[str] = Counter()
        self.epilogue_calls = 0
        self.epilogue_fused = 0
        self.epilogue_fallbacks = 0
        self.epilogue_fallback_reasons: Counter[str] = Counter()

    def record_success(self, x_shape: Any, weight_shape: Any) -> None:
        with self._lock:
            self.calls += 1
            self.successes += 1
            self.shapes[f"{tuple(x_shape)}->{tuple(weight_shape)}"] += 1

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self.calls += 1
            self.fallbacks += 1
            self.fallback_reasons[str(reason)] += 1

    def record_epilogue(self, fused: bool, reason: str | None = None) -> None:
        with self._lock:
            self.epilogue_calls += 1
            if fused:
                self.epilogue_fused += 1
            else:
                self.epilogue_fallbacks += 1
                if reason:
                    self.epilogue_fallback_reasons[str(reason)] += 1

    def receipt(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": "torch_int8_mm_convrot",
                "calls": self.calls,
                "successfulCalls": self.successes,
                "fallbackCalls": self.fallbacks,
                "fallbackReasons": dict(self.fallback_reasons),
                "shapes": dict(self.shapes),
                "epilogueCalls": self.epilogue_calls,
                "epilogueFusedCalls": self.epilogue_fused,
                "epilogueFallbackCalls": self.epilogue_fallbacks,
                "epilogueFallbackReasons": dict(self.epilogue_fallback_reasons),
                "math": "existing comfy-kitchen ConvRot rotation + row quantization; torch._int_mm INT8 accumulation; BF16 dequant",
            }


def _safe_backend_constraints(torch: Any) -> Any:
    from comfy_kitchen.constraints import FunctionConstraints, ParamConstraint

    floats = frozenset({torch.float16, torch.bfloat16, torch.float32})
    return FunctionConstraints(
        params={
            "x": ParamConstraint(dtypes=floats),
            "weight": ParamConstraint(dtypes=frozenset({torch.int8})),
            "weight_scale": ParamConstraint(dtypes=floats),
            "out_dtype": ParamConstraint(dtypes=floats),
            "convrot": ParamConstraint(dtypes=frozenset({bool})),
            "convrot_groupsize": ParamConstraint(dtypes=frozenset({int})),
            "input_act": ParamConstraint(dtypes=frozenset({str, type(None)})),
        },
        default_devices=frozenset({"cuda"}),
        min_compute_capability=(7, 5),
    )


def _rotate_activation_flat(torch: Any, x: Any, h: Any, group_size: int) -> Any:
    """Apply the same grouped Hadamard matmul with one GEMM launch.

    comfy-kitchen's reference spelling uses ``[M, groups, K] @ [K, K]``.
    Flattening the independent groups into ``[M*groups, K]`` selects the
    same dense GEMM math while avoiding one small batched GEMM per group.  On
    the target Torch/cu128 build this is bit-identical for BF16 and FP32 and
    removes the dominant FC2 ConvRot cost.
    """

    original_shape = tuple(x.shape)
    features = int(original_shape[-1])
    if features % int(group_size) != 0:
        raise ValueError(f"features {features} not divisible by group size {group_size}")
    groups = features // int(group_size)
    grouped = x.reshape(-1, groups, int(group_size))
    # Keep the native Comfy grouped operation shape.  Chunk only the logical
    # row dimension, so each launch remains [rows, groups, K] @ [K, K].  This
    # is the same dense Hadamard multiplication and operand order as the
    # official eager ConvRot implementation, while bounding the cuBLAS
    # workspace needed by a 106k-token H3 request.
    chunk_rows = 256
    rotated = torch.empty_like(grouped)
    for start in range(0, int(grouped.shape[0]), chunk_rows):
        stop = min(start + chunk_rows, int(grouped.shape[0]))
        rotated[start:stop] = torch.matmul(grouped[start:stop], h)
    return rotated.reshape(original_shape)


def _build_int8_linear(
    torch: Any,
    fallback: Callable[..., Any] | None,
    state: TorchInt8BackendState,
    *,
    use_triton_epilogue: bool = True,
) -> Callable[..., Any]:
    """Build the backend callable without changing model weights."""

    from comfy_kitchen.backends._activations import apply_input_act
    from comfy_kitchen.backends.triton.quantization import triton_quantize_rowwise
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    def int8_linear(
        x: Any,
        weight: Any,
        weight_scale: Any,
        bias: Any = None,
        out_dtype: Any = None,
        convrot: bool = False,
        convrot_groupsize: int = 256,
        input_act: str | None = None,
    ) -> Any:
        if out_dtype is None:
            out_dtype = torch.bfloat16
        try:
            if not torch.cuda.is_available() or not hasattr(torch, "_int_mm"):
                raise RuntimeError("Torch native INT8 GEMM is unavailable")
            if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8:
                raise TypeError(f"expected raw INT8 weight, got {type(weight).__name__}/{getattr(weight, 'dtype', None)}")
            if not isinstance(x, torch.Tensor) or not x.is_cuda:
                raise RuntimeError("Torch INT8 backend requires CUDA activations")
            x = apply_input_act(x, input_act)
            # The H3 MLP down projection passes input_act="swiglu".  That
            # official activation halves the logical K before the packed INT8
            # weight is consumed, so validate dimensions only after applying
            # it (matching comfy-kitchen's existing implementation).
            if x.shape[-1] != weight.shape[-1]:
                raise ValueError(f"input/weight K mismatch: {x.shape[-1]} vs {weight.shape[-1]}")
            if convrot and x.shape[-1] % int(convrot_groupsize) != 0:
                raise ValueError(f"ConvRot group size {convrot_groupsize} does not divide K={x.shape[-1]}")

            original_shape = tuple(x.shape)
            x_2d = x.reshape(-1, x.shape[-1])
            if convrot:
                h = _build_hadamard(int(convrot_groupsize), device=x.device, dtype=x.dtype)
                x_2d = _rotate_activation_flat(torch, x_2d, h, int(convrot_groupsize))
            x_int8, x_scale = triton_quantize_rowwise(x_2d)

            # ModelPatcher normally supplies the weight on the active device.
            # A device transfer is allowed for a prefetched block, but never
            # copy the transposed 20-GB logical model representation.
            if weight.device != x.device:
                weight = weight.to(device=x.device)
            if not weight.is_contiguous():
                weight = weight.contiguous()
            acc = torch._int_mm(x_int8, weight.t())
            n = int(acc.shape[1])

            scales = weight_scale.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
            if scales.numel() not in (1, weight.shape[0]):
                raise ValueError(
                    f"INT8 scale count {scales.numel()} does not match output channels {weight.shape[0]}"
                )
            x_scale = x_scale.reshape(-1).contiguous()
            if use_triton_epilogue:
                try:
                    result = _triton_int8_epilogue(torch, acc, x_scale, scales, bias, out_dtype)
                    state.record_epilogue(True)
                except Exception as epilogue_exc:
                    # Keep the same FP32 operation order as the fused kernel when
                    # a local Triton compiler/kernel is unavailable.
                    state.record_epilogue(False, f"{type(epilogue_exc).__name__}: {epilogue_exc}")
                    result = _torch_int8_epilogue(torch, acc, x_scale, scales, bias, out_dtype)
            else:
                result = _torch_int8_epilogue(torch, acc, x_scale, scales, bias, out_dtype)
                state.record_epilogue(False, "strict_backend_disables_triton_epilogue")
            output = result.reshape(*original_shape[:-1], n)
            state.record_success(tuple(x.shape), tuple(weight.shape))
            return output
        except Exception as exc:
            state.record_fallback(f"{type(exc).__name__}: {exc}")
            if fallback is None:
                raise RuntimeError(f"strict Torch INT8 backend failed: {type(exc).__name__}: {exc}") from exc
            return fallback(
                x=x,
                weight=weight,
                weight_scale=weight_scale,
                bias=bias,
                out_dtype=out_dtype,
                convrot=convrot,
                convrot_groupsize=convrot_groupsize,
                input_act=input_act,
            )

    return int8_linear


def _dequant_int4_grouped_to_int8(
    torch: Any,
    qdata: Any,
    s_rel: Any,
    codebook: Any,
    group_size: int,
) -> Any:
    """Decode packed W4A8 weights without using the Triton W4A8 kernel."""

    n, k_half = (int(qdata.shape[0]), int(qdata.shape[1]))
    k = k_half * 2
    groups = k // int(group_size)
    packed = qdata.to(dtype=torch.int32) & 0xFF
    unpacked = torch.empty((n, k), dtype=torch.int32, device=qdata.device)
    unpacked[:, 0::2] = packed & 0xF
    unpacked[:, 1::2] = (packed >> 4) & 0xF
    if codebook is not None:
        values = codebook.to(device=qdata.device, dtype=torch.float32)[unpacked]
    else:
        values = unpacked.to(dtype=torch.float32) - 8.0
    values = values.reshape(n, groups, int(group_size)) * s_rel.float().unsqueeze(-1)
    return values.reshape(n, k).round().clamp_(-127, 127).to(dtype=torch.int8)


def _build_w4a8_int8_linear(
    torch: Any,
    strict_int8_linear: Callable[..., Any],
    eager_fallback: Callable[..., Any],
) -> Callable[..., Any]:
    """Route packed W4A8 through the same native Torch INT8 GEMM."""

    from comfy_kitchen.backends.eager.w4a8_int8 import validate_w4a8_operands

    def w4a8_int8_linear(
        x: Any,
        qdata: Any,
        s_rel: Any,
        s_channel: Any,
        codebook: Any = None,
        correction: Any = None,
        bias: Any = None,
        group_size: int = 16,
        convrot_groupsize: int = 256,
        out_dtype: Any = None,
    ) -> Any:
        validate_w4a8_operands(
            qdata, s_rel, s_channel, codebook, correction,
            group_size, convrot_groupsize,
        )
        if correction is not None:
            return eager_fallback(
                x, qdata, s_rel, s_channel, codebook=codebook,
                correction=correction, bias=bias, group_size=group_size,
                convrot_groupsize=convrot_groupsize, out_dtype=out_dtype or torch.bfloat16,
            )
        int8_weight = _dequant_int4_grouped_to_int8(
            torch, qdata, s_rel, codebook, int(group_size),
        )
        return strict_int8_linear(
            x, int8_weight, s_channel, bias=bias,
            out_dtype=out_dtype or torch.bfloat16,
            convrot=True, convrot_groupsize=convrot_groupsize,
        )

    return w4a8_int8_linear


def install_torch_int8_backend(torch: Any, *, prefer: bool = True) -> Dict[str, Any]:
    """Register the native Torch INT8 backend, optionally as the preferred path.

    The independent H3 runtime prefers the verified Triton ConvRot kernel when
    it is available for the full model.  This backend remains a registered,
    numerically checked fallback for environments where Triton is unavailable.
    """

    receipt: Dict[str, Any] = {
        "requested": True,
        "selected": None,
        "available": False,
        "source": "embedded_h3_runtime.torch_int8_backend",
    }
    if not bool(torch.cuda.is_available()) or not hasattr(torch, "_int_mm"):
        receipt["reason"] = "Torch CUDA native INT8 GEMM is unavailable"
        return receipt
    try:
        import comfy_kitchen as kitchen
        from comfy_kitchen.registry import registry

        fallback = registry.get_implementation("int8_linear", backend="triton")
        state = TorchInt8BackendState()

        class BackendModule:
            pass

        module = BackendModule()
        module.int8_linear = _build_int8_linear(torch, fallback, state)
        module.int8_linear.__h3_backend_name__ = "h3_torch"
        strict_module = BackendModule()
        strict_module.int8_linear = _build_int8_linear(torch, None, state, use_triton_epilogue=False)
        strict_module.int8_linear.__h3_backend_name__ = "h3_torch_strict"
        eager_w4a8 = registry.get_implementation("w4a8_int8_linear", backend="eager")
        module.w4a8_int8_linear = _build_w4a8_int8_linear(
            torch, strict_module.int8_linear, eager_w4a8,
        )
        module.w4a8_int8_linear.__h3_backend_name__ = "h3_torch"
        strict_module.w4a8_int8_linear = _build_w4a8_int8_linear(
            torch, strict_module.int8_linear, eager_w4a8,
        )
        strict_module.w4a8_int8_linear.__h3_backend_name__ = "h3_torch_strict"
        int8_constraints = _safe_backend_constraints(torch)
        w4a8_constraints = registry.get_constraints("eager", "w4a8_int8_linear")
        if w4a8_constraints is None:
            raise RuntimeError("Comfy Kitchen eager W4A8 constraints are unavailable")
        registry.register(
            "h3_torch", module,
            {"int8_linear": int8_constraints, "w4a8_int8_linear": w4a8_constraints},
        )
        registry.register(
            "h3_torch_strict", strict_module,
            {"int8_linear": int8_constraints, "w4a8_int8_linear": w4a8_constraints},
        )
        if prefer:
            priority = ["h3_torch"] + [name for name in ("cuda", "triton", "eager") if name != "h3_torch"]
        else:
            priority = ["triton", "h3_torch"] + [name for name in ("cuda", "eager") if name not in {"triton", "h3_torch"}]
        registry.set_priority(priority)
        kitchen._h3_torch_int8_state = state
        receipt.update({
            "available": True,
            "selected": "h3_torch_int8_mm_convrot",
            "strictBackend": "h3_torch_strict",
            "preferred": bool(prefer),
            "role": "preferred" if prefer else "fallback",
            "priority": priority,
            "capability": "torch._int_mm",
            "fallback": "existing comfy-kitchen Triton int8_linear",
            "computeCapability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
        })
        return receipt
    except Exception as exc:
        receipt["reason"] = f"native Torch INT8 backend registration failed: {type(exc).__name__}: {exc}"
        return receipt


def torch_int8_backend_receipt() -> Dict[str, Any]:
    """Return the current process dispatch receipt, if installed."""

    try:
        import comfy_kitchen as kitchen
        state = getattr(kitchen, "_h3_torch_int8_state", None)
        if state is None:
            return {"installed": False}
        return {"installed": True, **state.receipt()}
    except Exception as exc:
        return {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
