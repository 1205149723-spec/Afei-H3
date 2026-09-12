"""Thin adapter for the audited KJNodes MiniMax H3 Sage patch.

The vendor implementation remains in ``runtime/third_party`` at the pinned
upstream commit.  This module only discovers it, checks its dependencies, and
applies the vendor ``minimax_sageattn_forward`` function to a ModelPatcher.
It never starts a ComfyUI server or executes a workflow.
"""

from __future__ import annotations

import importlib.util
import json
import os
from email.parser import Parser
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KJ_SOURCE = PROJECT_ROOT / "runtime" / "third_party" / "KJNodes_ltxv_nodes_ab8f90f0.py"
KJ_LICENSE = PROJECT_ROOT / "runtime" / "third_party" / "KJNodes_LICENSE_GPL-3.0.txt"
SAGE_PRIVATE_ROOT = Path(os.environ.get("H3_SAGE_PRIVATE_ROOT", PROJECT_ROOT / "runtime" / "python_packages" / "sageattention-2.2.0-cu130-torch211-py312-linux"))
KJ_COMMIT = "ab8f90f02ad6ec3a4900b1b4df9c03cded7b4690"
_MODULE_NAME = "h3_kj_vendor_ltxv_ab8f90f0"
_REQUIRED_SAGE_SYMBOLS = (
    "get_cuda_arch_versions",
    "per_thread_int8_triton",
    "per_warp_int8_cuda",
    "per_block_int8_triton",
    "per_channel_fp8",
)
_MODULE: Optional[Any] = None
_MODULE_ERROR: Optional[str] = None
_KERNEL_SELF_TEST: Optional[Dict[str, Any]] = None
_LOCK = threading.RLock()


def _current_cuda_architecture() -> tuple[Optional[str], Optional[str]]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, "CUDA is unavailable"
        major, minor = torch.cuda.get_device_capability()
        return f"sm{major}{minor}", None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _configure_vendor_sage_backend(module: Any, variant: str) -> Dict[str, Any]:
    """Select one real Sage kernel implementation for KJ's H3 forward."""

    if variant == "vendor_default":
        return {"kernelVariant": variant, "vendorHelperOverride": False}

    if variant == "public_fp16_cuda":
        from sageattention import sageattn_qk_int8_pv_fp16_cuda

        def stable_helper(qkv: list[Any], dtype: Any) -> Any:
            q, k, v = qkv
            qkv.clear()
            out = sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout="NHD", is_causal=False)
            if out.dtype != dtype:
                out = out.to(dtype)
            return out

        module._sageattn_int8_fp8_nhd = stable_helper
        return {"kernelVariant": variant, "vendorHelperOverride": True}

    if variant == "public_auto":
        from sageattention import sageattn

        def stable_helper(qkv: list[Any], dtype: Any) -> Any:
            q, k, v = qkv
            qkv.clear()
            out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
            if out.dtype != dtype:
                out = out.to(dtype)
            return out

        module._sageattn_int8_fp8_nhd = stable_helper
        return {"kernelVariant": variant, "vendorHelperOverride": True}

    raise RuntimeError(f"unknown Kijai Sage kernel variant: {variant}")


def _kernel_candidates(current_architecture: Optional[str]) -> list[str]:
    # Ada's FP8 path is known to be unstable in SageAttention 2.2 on real
    # workloads. Prefer the validated FP16 CUDA path there. Other devices try
    # KJ's architecture-specific implementation first, then public Sage APIs.
    if current_architecture == "sm89":
        return ["public_fp16_cuda", "vendor_default"]
    return ["vendor_default", "public_auto", "public_fp16_cuda"]


def _run_kijai_kernel_self_test(current_architecture: Optional[str]) -> Dict[str, Any]:
    """Probe candidate kernels in isolated CUDA processes and cache the winner."""

    global _KERNEL_SELF_TEST
    with _LOCK:
        if _KERNEL_SELF_TEST is not None:
            return dict(_KERNEL_SELF_TEST)

    script = r'''
import json, os, torch
from h3_kj_adapter import _configure_vendor_sage_backend, _load_vendor_module
variant = os.environ["H3_KJ_TEST_VARIANT"]
module = _load_vendor_module()
config = _configure_vendor_sage_backend(module, variant)
checks = []
for dtype in (torch.float16, torch.bfloat16):
    q = torch.randn((1, 512, 56, 128), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    out = module._sageattn_int8_fp8_nhd([q, k, v], dtype)
    torch.cuda.synchronize()
    if out.shape != q.shape or not torch.isfinite(out.float()).all().item():
        raise RuntimeError(f"invalid Sage output for {dtype}: shape={tuple(out.shape)}")
    checks.append(str(dtype))
print(json.dumps({"ok": True, **config, "dtypes": checks}))
'''
    attempts = []
    for variant in _kernel_candidates(current_architecture):
        env = os.environ.copy()
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["H3_KJ_TEST_VARIANT"] = variant
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
                env=env,
            )
        except Exception as exc:
            attempts.append({"variant": variant, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout.strip().splitlines()[-1])
            except (json.JSONDecodeError, IndexError) as exc:
                attempts.append({"variant": variant, "ok": False, "error": f"invalid self-test output: {exc}"})
                continue
            payload.update({"currentDeviceArchitecture": current_architecture, "attempts": attempts + [{"variant": variant, "ok": True}]})
            with _LOCK:
                _KERNEL_SELF_TEST = dict(payload)
            return dict(payload)
        attempts.append({
            "variant": variant,
            "ok": False,
            "error": (result.stderr or result.stdout or f"exit {result.returncode}")[-4000:],
        })

    payload = {
        "ok": False,
        "kernelVariant": None,
        "currentDeviceArchitecture": current_architecture,
        "attempts": attempts,
    }
    with _LOCK:
        _KERNEL_SELF_TEST = dict(payload)
    return dict(payload)


def _prefer_private_sage() -> bool:
    """Prefer the audited D-drive Sage build without mutating system Python."""

    if not SAGE_PRIVATE_ROOT.is_dir():
        return False
    private = str(SAGE_PRIVATE_ROOT)
    sys.path[:] = [item for item in sys.path if item != private]
    sys.path.insert(0, private)
    return True


def _path_is_private_sage(path: Any) -> bool:
    try:
        return Path(path).resolve().is_relative_to(SAGE_PRIVATE_ROOT.resolve())
    except (OSError, TypeError, ValueError):
        return False


def _activate_private_sage() -> Dict[str, Any]:
    """Promote the locked package and discard a prematurely cached generic one."""

    present = _prefer_private_sage()
    reset = False
    cached_paths: Dict[str, Optional[str]] = {}
    for name, module in list(sys.modules.items()):
        if name != "sageattention" and not name.startswith("sageattention."):
            continue
        module_path = getattr(module, "__file__", None)
        cached_paths[name] = str(module_path) if module_path else None
        if module_path is None or not _path_is_private_sage(module_path):
            reset = True
    if reset:
        for name in list(sys.modules):
            if name == "sageattention" or name.startswith("sageattention."):
                sys.modules.pop(name, None)
    return {
        "privatePathPromoted": present,
        "moduleCacheReset": reset,
        "cachedModulePaths": cached_paths,
    }


_SAGE_ACTIVATION = _activate_private_sage()


def _inspect_private_sage_distribution(core_path: Path, root: Path = SAGE_PRIVATE_ROOT, metadata_paths: Optional[list[Path]] = None) -> Dict[str, Any]:
    resolved_root = root.resolve()
    resolved_core = core_path.resolve()
    if not resolved_core.is_relative_to(resolved_root):
        raise RuntimeError("sageattention.core path escapes SAGE_PRIVATE_ROOT")
    metadata_files = list(metadata_paths) if metadata_paths is not None else list(resolved_root.glob("sageattention-*.dist-info/METADATA"))
    if len(metadata_files) != 1:
        raise RuntimeError(f"expected one locked sageattention METADATA, found {len(metadata_files)}")
    metadata_path = metadata_files[0].resolve()
    distribution_path = metadata_path.parent
    if not metadata_path.is_relative_to(resolved_root) or not distribution_path.is_relative_to(resolved_root):
        raise RuntimeError("sageattention distribution path escapes SAGE_PRIVATE_ROOT")
    document = Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    name = str(document.get("Name") or "")
    version = str(document.get("Version") or "")
    if name != "sageattention":
        raise RuntimeError(f"locked SageAttention distribution name mismatch: {name or 'missing'}")
    if not version.startswith("2.2.0"):
        raise RuntimeError(f"locked SageAttention distribution version mismatch: {version or 'missing'}")
    return {
        "sageDistributionName": name,
        "sagePackageVersion": version,
        "sageMetadataPath": str(metadata_path),
        "sageDistributionPath": str(distribution_path),
        "sageDistributionPathVerified": True,
        "sagePackageVersionVerified": True,
    }


def _inspect_sage_core_identity(sage_core: Any, root: Path = SAGE_PRIVATE_ROOT, metadata_paths: Optional[list[Path]] = None) -> Dict[str, Any]:
    symbols = {name: hasattr(sage_core, name) for name in _REQUIRED_SAGE_SYMBOLS}
    sage_path = Path(sage_core.__file__).resolve()
    identity: Dict[str, Any] = {
        "requiredSageSymbols": symbols,
        "sageModulePath": str(sage_path),
        "sageModulePathVerified": sage_path.is_relative_to(root.resolve()),
    }
    identity.update(_inspect_private_sage_distribution(sage_path, root, metadata_paths))
    architectures = sage_core.get_cuda_arch_versions() if symbols["get_cuda_arch_versions"] else []
    normalized = {str(item).lower().replace("_", "") for item in (architectures or [])}
    current_architecture, current_architecture_error = _current_cuda_architecture()
    kernel_self_test = _run_kijai_kernel_self_test(current_architecture) if current_architecture else {"ok": False, "kernelVariant": None, "attempts": []}
    current_architecture_verified = kernel_self_test.get("ok") is True
    identity.update({
        "cudaArchitectures": architectures,
        "sm120Verified": bool({"sm120", "120"} & normalized),
        "currentDeviceArchitecture": current_architecture,
        "currentDeviceArchitectureVerified": current_architecture_verified,
        "currentDeviceArchitectureError": current_architecture_error,
        "kernelSelfTest": kernel_self_test,
        "kernelVariant": kernel_self_test.get("kernelVariant"),
    })
    return identity


def _load_vendor_module() -> Any:
    """Load only the pinned vendor module, without importing Comfy's server."""

    global _MODULE, _MODULE_ERROR
    with _LOCK:
        if _MODULE is not None:
            return _MODULE
        if _MODULE_ERROR is not None:
            raise RuntimeError(_MODULE_ERROR)
        if not KJ_SOURCE.is_file():
            _MODULE_ERROR = f"pinned KJ source is missing: {KJ_SOURCE}"
            raise RuntimeError(_MODULE_ERROR)

        # KJNodes imports ``server`` only for its preview-node registration.
        # The H3 function itself does not use it.  A temporary inert module
        # keeps this embedded adapter independent of a ComfyUI server.
        class _PromptServer:
            instance = object()

        inert_server = types.ModuleType("server")
        inert_server.PromptServer = _PromptServer
        previous_server = sys.modules.get("server")
        sys.modules["server"] = inert_server
        try:
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, KJ_SOURCE)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot create import spec for {KJ_SOURCE}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAME] = module
            spec.loader.exec_module(module)
            _MODULE = module
            return module
        except Exception as exc:
            _MODULE_ERROR = f"{type(exc).__name__}: {exc}"
            sys.modules.pop(_MODULE_NAME, None)
            raise RuntimeError(_MODULE_ERROR) from exc
        finally:
            if previous_server is None:
                sys.modules.pop("server", None)
            else:
                sys.modules["server"] = previous_server


def inspect_h3_memory_efficient_sage() -> Dict[str, Any]:
    """Return an auditable compatibility receipt; never claims availability optimistically."""

    receipt: Dict[str, Any] = {
        "classification": "official_contract_compatible_third_party_optimization",
        "author": "Kijai",
        "requestedSource": "kijai/ComfyUI-KJNodes",
        "sourceCommit": KJ_COMMIT,
        "sourcePath": str(KJ_SOURCE),
        "license": "GPL-3.0",
        "licensePath": str(KJ_LICENSE),
        "sagePrivateRoot": str(SAGE_PRIVATE_ROOT),
        "sagePrivatePresent": SAGE_PRIVATE_ROOT.is_dir(),
        "sourcePresent": KJ_SOURCE.is_file(),
        "licensePresent": KJ_LICENSE.is_file(),
        "patchClass": "MiniMaxH3MemoryEfficientSageAttentionPatch",
        "forwardFunction": "minimax_sageattn_forward",
        "vendorImport": False,
        "requiredSageSymbols": {name: False for name in _REQUIRED_SAGE_SYMBOLS},
        "requiredSagePackageVersion": "2.2.0",
        "requiredCudaArchitecture": "current_device",
        "sageModulePathVerified": False,
        "sageDistributionPathVerified": False,
        "sagePackageVersionVerified": False,
        "sm120Verified": False,
        "currentDeviceArchitecture": None,
        "currentDeviceArchitectureVerified": False,
        "currentDeviceArchitectureError": None,
        "pathActivation": dict(_SAGE_ACTIVATION),
        "cudaArchitectures": None,
        "comfyKitchen": False,
        "miniMaxH3Model": False,
        "available": False,
        "compatibilityFailure": None,
    }
    if not receipt["sourcePresent"]:
        receipt["compatibilityFailure"] = f"pinned KJ source is missing: {KJ_SOURCE}"
        return receipt
    # Static source inspection keeps the health endpoint fast and avoids
    # importing KJ's optional preview-registration surface on every request.
    # The exact vendor module is imported only by apply_* after prerequisites
    # pass, in the real denoiser path.
    source_text = KJ_SOURCE.read_text(encoding="utf-8", errors="replace")
    receipt["patchClassPresent"] = f"class {receipt['patchClass']}" in source_text
    receipt["forwardFunctionPresent"] = f"def {receipt['forwardFunction']}" in source_text
    receipt["vendorImport"] = False
    receipt["vendorImportDeferred"] = True

    try:
        activation = _activate_private_sage()
        receipt["pathActivation"] = activation
        import sageattention.core as sage_core

        receipt.update(_inspect_sage_core_identity(sage_core))
    except Exception as exc:
        receipt["sageImportError"] = f"{type(exc).__name__}: {exc}"

    try:
        import comfy_kitchen  # noqa: F401

        receipt["comfyKitchen"] = True
    except Exception as exc:
        receipt["comfyKitchenError"] = f"{type(exc).__name__}: {exc}"
    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: F401

        receipt["miniMaxH3Model"] = True
    except Exception as exc:
        receipt["miniMaxH3ModelError"] = f"{type(exc).__name__}: {exc}"

    missing = [name for name, present in receipt["requiredSageSymbols"].items() if not present]
    receipt["available"] = bool(
        receipt.get("patchClassPresent")
        and receipt.get("forwardFunctionPresent")
        and receipt["sourcePresent"]
        and receipt["licensePresent"]
        and receipt["sageModulePathVerified"]
        and receipt["sageDistributionPathVerified"]
        and receipt["sagePackageVersionVerified"]
        and receipt["currentDeviceArchitectureVerified"]
        and not missing
        and receipt["comfyKitchen"]
        and receipt["miniMaxH3Model"]
    )
    if not receipt["available"] and not receipt["compatibilityFailure"]:
        if not receipt["sageModulePathVerified"]:
            receipt["compatibilityFailure"] = "locked private SageAttention module path mismatch"
        elif not receipt["sagePackageVersionVerified"]:
            receipt["compatibilityFailure"] = "locked SageAttention package version mismatch"
        elif not receipt["currentDeviceArchitectureVerified"]:
            current_arch = receipt.get("currentDeviceArchitecture") or "unknown"
            attempts = (receipt.get("kernelSelfTest") or {}).get("attempts") or []
            receipt["compatibilityFailure"] = f"no validated Kijai/Sage kernel for current GPU architecture: {current_arch}; attempts={attempts}"
        elif missing:
            receipt["compatibilityFailure"] = "Kijai H3 Sage extension symbols missing: " + ", ".join(missing)
        else:
            receipt["compatibilityFailure"] = "Kijai H3 patch prerequisites are incomplete"
    return receipt


def apply_h3_memory_efficient_sage_patch(model_patcher: Any) -> tuple[Any, Dict[str, Any]]:
    """Apply the exact pinned KJ forward function to all H3 attention blocks."""

    receipt = inspect_h3_memory_efficient_sage()
    if not receipt["available"]:
        raise RuntimeError(str(receipt["compatibilityFailure"] or "KJ H3 Sage patch is unavailable"))
    module = _load_vendor_module()
    kernel_variant = str(receipt.get("kernelVariant") or "vendor_default")
    kernel_config = _configure_vendor_sage_backend(module, kernel_variant)
    clone = model_patcher.clone()
    diffusion_model = clone.get_model_object("diffusion_model")
    model_type = getattr(module, "_MiniMaxH3Model", None)
    if model_type is not None and not isinstance(diffusion_model, model_type):
        raise RuntimeError("pinned KJ H3 patch refuses a non-MiniMaxH3Model diffusion model")
    forward = getattr(module, "minimax_sageattn_forward")
    block_count = 0
    for index, block in enumerate(diffusion_model.blocks):
        clone.add_object_patch(
            f"diffusion_model.blocks.{index}.attn.forward",
            forward.__get__(block.attn, block.attn.__class__),
        )
        block_count += 1
    if block_count != 50:
        raise RuntimeError(f"pinned Kijai H3 patch requires exactly 50 denoiser blocks, observed {block_count}")
    receipt.update({
        "applied": True,
        "patchedBlocks": block_count,
        "backend": "kijai_kj_minimax_h3_memory_efficient_sage",
        "allowCompile": False,
        "scope": "denoiser_only",
        "globalPatch": False,
        **kernel_config,
    })
    return clone, receipt
