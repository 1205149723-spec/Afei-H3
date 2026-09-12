"""Pure contract helpers for the full-quality attention backend selector."""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Mapping, Optional


KERNEL_BACKEND_CONTRACT_VERSION = "h3-full-quality-kernel-backend-v2"
DEFAULT_KERNEL_BACKEND = "official_native"
KERNEL_BACKENDS = ("kijai_fast", "official_native")

KERNEL_BACKEND_COPY = {
    "kijai_fast": {
        "labelZh": "Kijai 高速内核",
        "qualityZh": "保持 full_quality 的模型与采样合同，但低精度计算可能产生数值差异；画质尚未由严格同输入 GPU A/B 证明等价。",
        "speedZh": "预期减少 attention 计算与显存搬运；当前没有严格同输入 GPU A/B，不能承诺普遍提速幅度。",
    },
    "official_native": {
        "labelZh": "官方原生内核（默认）",
        "qualityZh": "使用 ComfyUI 官方原生 attention；仍需真实成片验收，不能仅凭合同宣称画质结果。",
        "speedZh": "作为原生对照，可能比 Kijai 路径慢；普通速度建议只提示，不阻断任务。",
    },
}

REQUIRED_KIJAI_SYMBOLS = (
    "get_cuda_arch_versions",
    "per_thread_int8_triton",
    "per_warp_int8_cuda",
    "per_block_int8_triton",
    "per_channel_fp8",
)

_SHARED_FULL_QUALITY = {
    "routeId": "full_quality",
    "qualityRoute": "full_quality",
    "sampling": {"steps": 20, "sampler": "res_multistep", "scheduler": "simple"},
    "preservedInputs": [
        "model",
        "weights",
        "frames",
        "duration",
        "assetContract",
    ],
    "selectionPolicy": {
        "automaticSwitching": False,
        "switchDimensions": [],
        "speedAdviceBlocksExecution": False,
    },
}

_BACKEND_CONTRACTS = {
    "kijai_fast": {
        "backend": "kijai_kj_minimax_h3_memory_efficient_sage",
        "classification": "official_contract_compatible_third_party_optimization",
        "attentionProvider": "Kijai",
        "source": "kijai/ComfyUI-KJNodes",
        "sourceCommit": "ab8f90f02ad6ec3a4900b1b4df9c03cded7b4690",
        "sagePackageVisibility": "private",
        "sagePackageVersion": "2.2.0",
        "cudaArchitecturePolicy": "current_device_must_be_supported",
        "requiredSageSymbols": list(REQUIRED_KIJAI_SYMBOLS),
        "patchedBlocks": 50,
        "scope": "denoiser_only",
        "globalPatch": False,
        "fallbackAllowed": False,
        "fallback": False,
        "ffnChunk": {
            "enabled": True,
            "chunks": 2,
            "minTokens": 4096,
            "belongsToKernelBackend": True,
        },
    },
    "official_native": {
        "backend": "comfyui_official_native_attention",
        "classification": "comfyui_official_native",
        "attentionProvider": "ComfyUI",
        "requiresOfficialOptimizedAttention": True,
        "kijaiAllowed": False,
        "genericSageAllowed": False,
        "ffnChunkAllowed": False,
        "fallbackAllowed": False,
        "fallback": False,
        "ffnChunk": {
            "enabled": False,
            "belongsToKernelBackend": True,
        },
    },
}


def normalize_kernel_backend(value: Optional[str] = None) -> str:
    """Return the explicit backend ID; omission alone selects the default."""

    if value is None:
        return DEFAULT_KERNEL_BACKEND
    if not isinstance(value, str):
        raise ValueError("内核选择必须是字符串：kijai_fast 或 official_native。")
    normalized = value.strip()
    if normalized not in KERNEL_BACKENDS:
        raise ValueError(
            f"未知内核选择“{value}”；只允许 kijai_fast 或 official_native，不能自动回退。"
        )
    return normalized


def kernel_backend_copy(value: Optional[str] = None) -> Dict[str, str]:
    """Return stable Chinese UI copy for a validated selection."""

    return copy.deepcopy(KERNEL_BACKEND_COPY[normalize_kernel_backend(value)])


def expected_kernel_backend_contract(value: Optional[str] = None, steps: int = 20, ffn_chunks: Optional[int] = None) -> Dict[str, Any]:
    """Build the immutable expected contract for compiler/worker wiring."""

    selected = normalize_kernel_backend(value)
    if int(steps) not in {20, 30, 40, 50, 60}:
        raise ValueError(f"unsupported full_quality steps: {steps}")
    contract = copy.deepcopy(_SHARED_FULL_QUALITY)
    contract["sampling"]["steps"] = int(steps)
    contract.update({
        "contractVersion": KERNEL_BACKEND_CONTRACT_VERSION,
        "kernelBackend": selected,
        "labelZh": KERNEL_BACKEND_COPY[selected]["labelZh"],
        "attentionKernelOnlyChange": selected == "official_native",
        "expectedBackendReceipt": copy.deepcopy(_BACKEND_CONTRACTS[selected]),
    })
    if selected == "kijai_fast" and ffn_chunks is not None:
        if int(ffn_chunks) not in {1, 2, 4}:
            raise ValueError("unsupported FFN chunks: expected 1, 2, or 4")
        contract["expectedBackendReceipt"]["ffnChunk"]["chunks"] = int(ffn_chunks)
    contract["ffnChunk"] = contract["expectedBackendReceipt"].pop("ffnChunk")
    return contract


def expected_kernel_backend_receipt(value: Optional[str] = None, steps: int = 20, ffn_chunks: Optional[int] = None) -> Dict[str, Any]:
    """Return the exact backend identity the runtime must report."""

    selected = normalize_kernel_backend(value)
    if int(steps) not in {20, 30, 40, 50, 60}:
        raise ValueError(f"unsupported full_quality steps: {steps}")
    backend = copy.deepcopy(_BACKEND_CONTRACTS[selected])
    ffn = backend.pop("ffnChunk")
    if selected == "kijai_fast" and ffn_chunks is not None:
        if int(ffn_chunks) not in {1, 2, 4}:
            raise ValueError("unsupported FFN chunks: expected 1, 2, or 4")
        ffn["chunks"] = int(ffn_chunks)
    receipt = {
        "contractVersion": KERNEL_BACKEND_CONTRACT_VERSION,
        "routeId": "full_quality",
        "kernelBackend": selected,
        **backend,
        "ffnChunkEnabled": ffn["enabled"],
    }
    if ffn["enabled"]:
        receipt.update({"ffnChunks": ffn["chunks"], "ffnMinTokens": ffn["minTokens"]})
    return receipt


def _fail(field: str, expected: Any, actual: Any) -> None:
    raise ValueError(f"内核真实回执不匹配：{field} 应为 {expected!r}，实际为 {actual!r}；已阻断，不能回退。")


def verify_kernel_backend_receipt(
    value: Optional[str], actual_receipt: Mapping[str, Any], steps: int = 20, ffn_chunks: Optional[int] = None
) -> Dict[str, Any]:
    """Fail closed unless the actual receipt proves the selected identity."""

    selected = normalize_kernel_backend(value)
    if not isinstance(actual_receipt, Mapping):
        raise ValueError("内核真实回执必须是对象；已阻断，不能回退。")
    actual = dict(actual_receipt)
    expected = expected_kernel_backend_receipt(selected, steps, ffn_chunks)
    common_fields = (
        "contractVersion",
        "routeId",
        "kernelBackend",
        "backend",
        "classification",
        "attentionProvider",
    )
    for field in common_fields:
        if actual.get(field) != expected[field]:
            _fail(field, expected[field], actual.get(field))
    if actual.get("fallback") is not False:
        _fail("fallback", False, actual.get("fallback"))
    if actual.get("ffnChunkEnabled") is not expected["ffnChunkEnabled"]:
        _fail("ffnChunkEnabled", expected["ffnChunkEnabled"], actual.get("ffnChunkEnabled"))

    if selected == "kijai_fast":
        scalar_fields = (
            "source",
            "sourceCommit",
            "sagePackageVisibility",
            "sagePackageVersion",
            "cudaArchitecturePolicy",
            "patchedBlocks",
            "scope",
            "globalPatch",
            "fallbackAllowed",
        )
        for field in scalar_fields:
            if field == "sagePackageVersion" and str(actual.get(field) or "").startswith("2.2.0"):
                continue
            if actual.get(field) != expected[field]:
                _fail(field, expected[field], actual.get(field))
        for field in ("ffnChunks", "ffnMinTokens"):
            if actual.get(field) != expected[field]:
                _fail(field, expected[field], actual.get(field))
        symbols = actual.get("requiredSageSymbols")
        if isinstance(symbols, Mapping):
            missing = [name for name in REQUIRED_KIJAI_SYMBOLS if symbols.get(name) is not True]
            extras = [name for name in symbols if name not in REQUIRED_KIJAI_SYMBOLS]
            if missing or extras:
                _fail("requiredSageSymbols", list(REQUIRED_KIJAI_SYMBOLS), symbols)
        elif list(symbols or []) != list(REQUIRED_KIJAI_SYMBOLS):
            _fail("requiredSageSymbols", list(REQUIRED_KIJAI_SYMBOLS), symbols)
    else:
        official_fields = (
            "requiresOfficialOptimizedAttention",
            "kijaiAllowed",
            "genericSageAllowed",
            "ffnChunkAllowed",
            "fallbackAllowed",
        )
        for field in official_fields:
            if actual.get(field) != expected[field]:
                _fail(field, expected[field], actual.get(field))
        forbidden = ("sourceCommit", "sagePackageVersion", "patchedBlocks", "scope")
        present = [field for field in forbidden if field in actual]
        if present:
            raise ValueError(f"官方原生内核回执混入第三方字段：{', '.join(present)}；已阻断。")
    return copy.deepcopy(actual)


def serialize_kernel_backend_contract(
    value: Optional[str] = None, steps: int = 20, ffn_chunks: Optional[int] = None
) -> str:
    """Return a deterministic JSON representation for fingerprints/receipts."""

    return json.dumps(
        expected_kernel_backend_contract(value, steps, ffn_chunks),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
