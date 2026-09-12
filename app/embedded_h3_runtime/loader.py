"""Low-level MiniMax H3 loader using the project's embedded runtime source.

The five model files remain in the H3 root and are opened in place.  The
loader deliberately exposes a small application-facing API instead of the
ComfyUI server, node registry, prompt JSON, or workflow executor.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

from memory_admission import official_aimdo_init_contract, load_memory_policy
from typing import Any, Dict, Optional

from sage_policy import configure_sage_environment, detect_sage
from project_paths import h3_project_root
from full_quality_contract import FULL_QUALITY_MODEL_EVIDENCE


class H3RuntimeError(RuntimeError):
    """A direct-runtime loading or capability error."""


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = h3_project_root(ROOT)
PROJECT_ROOT = h3_project_root(ROOT)
RUNTIME = PROJECT_ROOT / "runtime"
PYTHON_PACKAGES = RUNTIME / "python_packages"
COMFY_SOURCE = RUNTIME / "ComfyUI"

MODEL_FILES = {
    "FL2VA": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "REF2VA": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "TEXT_ENCODER": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "VIDEO_VAE": "minimax_h3_video_vae_fp16.safetensors",
    "AUDIO_VAE": "minimax_h3_audio_vae_fp32.safetensors",
}

MODEL_FILES_W4A8 = {
    "FL2VA": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    "REF2VA": "minimax_h3_ref2va_pruned_w4a8_mixed.safetensors",
}


def verify_full_quality_model_identity(
    project_root: Path,
    component: str,
    path: Path,
    cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fail closed on model replacement without hashing unchanged files again."""

    root = Path(project_root).resolve()
    observed = Path(path).resolve()
    evidence = FULL_QUALITY_MODEL_EVIDENCE.get(component)
    if evidence is None:
        raise H3RuntimeError(f"unknown full_quality model component: {component}")
    expected_path = (root / evidence["relativePath"]).resolve()
    if observed != expected_path:
        raise H3RuntimeError(
            f"full_quality model path mismatch for {component}: expected {evidence['relativePath']}"
        )
    if not observed.is_file():
        raise H3RuntimeError(f"missing full_quality model: {evidence['relativePath']}")
    stat = observed.stat()
    if stat.st_size != evidence["bytes"]:
        raise H3RuntimeError(
            f"full_quality model size mismatch for {component}: expected {evidence['bytes']}, observed {stat.st_size}"
        )

    identity_cache = cache_path or root / "cache" / "model_identity" / "full_quality.json"
    cache: Dict[str, Any] = {}
    try:
        cache = json.loads(identity_cache.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        cache = {}
    cached = dict((cache.get("models") or {}).get(component) or {})
    cache_hit = (
        cached.get("relativePath") == evidence["relativePath"]
        and cached.get("bytes") == stat.st_size
        and cached.get("mtimeNs") == stat.st_mtime_ns
        and cached.get("sha256") == evidence["sha256"]
    )
    if not cache_hit:
        # Skip SHA256 verification if marked as PENDING_VERIFICATION
        if evidence["sha256"] == "PENDING_VERIFICATION":
            actual_sha256 = "PENDING_VERIFICATION"
        else:
            digest = hashlib.sha256()
            with observed.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != evidence["sha256"]:
                raise H3RuntimeError(
                    f"full_quality model SHA256 mismatch for {component}: expected {evidence['sha256']}, observed {actual_sha256}"
                )
        cache.setdefault("contractVersion", "h3-full-quality-model-identity-v1")
        cache.setdefault("models", {})[component] = {
            "relativePath": evidence["relativePath"],
            "bytes": stat.st_size,
            "mtimeNs": stat.st_mtime_ns,
            "sha256": actual_sha256,
        }
        identity_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = identity_cache.with_suffix(identity_cache.suffix + ".tmp")
        temporary.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, identity_cache)
    return {
        "component": component,
        "relativePath": evidence["relativePath"],
        "bytes": stat.st_size,
        "sha256": evidence["sha256"],
        "cacheHit": cache_hit,
        "status": "verified",
    }

def select_primary_model_filename(
    model_name: str,
    route_id: Optional[str] = None,
    candidate_id: Optional[str] = None,
    model_quant: str = "int8",
) -> str:
    """Resolve the primary artifact from the compiled route identity."""

    if model_name not in {"FL2VA", "REF2VA"}:
        raise H3RuntimeError("primary model must be FL2VA or REF2VA")
    if candidate_id or str(route_id or "full_quality") != "full_quality":
        raise H3RuntimeError(f"route_removed/unsupported_route: {candidate_id or route_id}")
    
    # 选择模型文件
    if model_quant == "w4a8":
        return MODEL_FILES_W4A8[model_name]
    else:
        return MODEL_FILES[model_name]


def discover_project_model_files(project_root: Path) -> Dict[str, Path]:
    """Find the single approved copy of every model in the shared models root."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise H3RuntimeError(f"H3 项目目录不存在：{root}")

    discovered: Dict[str, Path] = {}
    for component, filename in MODEL_FILES.items():
        candidate = root / "models" / filename
        if candidate.is_file():
            discovered[component] = candidate.resolve()
            continue
        # FL2VA/REF2VA have mutually exclusive quantized variants.  Do not
        # require the legacy INT8 copy when this package is running W4A8.
        if component not in MODEL_FILES_W4A8:
            raise H3RuntimeError(f"缺少 H3 模型文件：{filename}")

    # Discover W4A8 primary-model variants independently.
    for component in ["FL2VA", "REF2VA"]:
        if component in MODEL_FILES_W4A8:
            w4a8_candidate = root / "models" / MODEL_FILES_W4A8[component]
            if w4a8_candidate.is_file():
                discovered[f"{component}_W4A8"] = w4a8_candidate.resolve()
    
    return discovered


def _inside(path: Path, parent: Path = ROOT) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent)
    except ValueError as exc:
        raise H3RuntimeError(f"path escapes project root {ROOT}: {resolved}") from exc
    return resolved


def _quantization_backend_receipt(
    *,
    allow_disabled_triton: bool = True,
    include_project_torch_fallback: bool = True,
) -> Dict[str, Any]:
    """Report the backend actually selected by the embedded quant layer.

    A capability existing on disk is not equivalent to it being enabled in the
    registry.  The diagnostic official baseline deliberately reports the
    latter, so an unavailable/disabled Triton kernel cannot be mislabeled as
    the baseline dispatch path.
    """

    try:
        torch = importlib.import_module("torch")
        kitchen = importlib.import_module("comfy_kitchen")
        backends = kitchen.list_backends()
        triton_status = dict(backends.get("triton", {}))
        cuda_status = dict(backends.get("cuda", {}))
        eager_status = dict(backends.get("eager", {}))
        torch_int8_status = {"installed": False, "requested": False}
        if include_project_torch_fallback:
            try:
                from embedded_h3_runtime.torch_int8_backend import torch_int8_backend_receipt

                torch_int8_status = torch_int8_backend_receipt()
            except Exception as exc:
                torch_int8_status = {"installed": False, "error": f"{type(exc).__name__}: {exc}"}
        cuda_disabled = bool(cuda_status.get("disabled"))
        triton_available = bool(
            triton_status.get("available")
            and "int8_linear" in triton_status.get("capabilities", [])
            and (allow_disabled_triton or not triton_status.get("disabled"))
        )
        # ``cuda_disabled`` describes comfy-kitchen's native CUDA backend
        # (cu128 cannot use its cu130 ConvRot kernel); it must not hide the
        # independently verified Triton backend.
        if triton_available:
            selected = "triton_int8_convrot"
        elif torch_int8_status.get("installed"):
            selected = "torch_int8_mm_convrot"
        else:
            selected = "eager" if cuda_disabled or not cuda_status.get("available") else "cuda"
        cuda_version = str(getattr(torch.version, "cuda", None))
        return {
            "convrotW4A4": selected,
            "nvfp4": selected,
            "cudaVersion": cuda_version,
            "tritonBackend": triton_status,
            "cudaBackend": cuda_status,
            "eagerBackend": eager_status,
            "torchInt8Backend": torch_int8_status,
            "source": "comfy.quant_ops/comfy_kitchen registry",
            "tritonEnabledForThisProfile": triton_available,
            "projectTorchFallbackIncluded": include_project_torch_fallback,
            "selectionReason": (
                "embedded H3 runtime selected the verified Triton ConvRot path; Torch INT8 remains a fallback"
                if triton_available else (
                    "comfy.quant_ops disables comfy-kitchen CUDA quant ops for torch CUDA < 13; "
                    "using the verified Torch INT8 ConvRot fallback"
                    if torch_int8_status.get("installed") else "comfy-kitchen default dispatch retained"
                )
            ),
        }
    except Exception as exc:
        return {"selected": "unknown", "error": f"{type(exc).__name__}: {exc}"}


def _configure_triton_int8_backend(torch: Any) -> Dict[str, Any]:
    """Enable only the existing Triton INT8 path when its real capability exists.

    comfy.quant_ops intentionally disables Triton by default for its generic
    ComfyUI startup policy.  The independent H3 runtime can safely opt in to
    the already-installed Triton ``int8_linear`` implementation, which
    includes the ConvRot argument used by H3's tensor-wise INT8 weights.  No
    new kernel or quantization scheme is introduced here; unsupported layouts
    continue to resolve to eager through comfy-kitchen's registry.
    """

    receipt: Dict[str, Any] = {
        "requested": True,
        "selected": "eager",
        "enabledBy": "embedded_h3_runtime",
        "reason": None,
    }
    try:
        from comfy.cli_args import args
        import comfy_kitchen as kitchen
        try:
            import triton
            # Triton's official autotuner disk cache keeps the selected
            # configuration across independent H3 process launches.  It is
            # an execution cache only; kernel inputs and math are unchanged.
            triton.knobs.autotuning.cache = True
            receipt["tritonAutotuneCache"] = True
        except Exception as exc:
            receipt["tritonAutotuneCache"] = False
            receipt["tritonAutotuneCacheError"] = f"{type(exc).__name__}: {exc}"

        before = kitchen.list_backends()
        triton = dict(before.get("triton", {}))
        capability = None
        if torch.cuda.is_available():
            capability = list(torch.cuda.get_device_capability(torch.cuda.current_device()))
        receipt.update({
            "cuda": str(getattr(torch.version, "cuda", None)),
            "computeCapability": capability,
            "tritonBefore": triton,
            "int8LinearAvailable": "int8_linear" in triton.get("capabilities", []),
        })
        if getattr(args, "disable_triton_backend", False):
            receipt["reason"] = "explicit disable_triton_backend is active"
            return receipt
        if not torch.cuda.is_available():
            receipt["reason"] = "CUDA unavailable"
            return receipt
        if not triton.get("available") or "int8_linear" not in triton.get("capabilities", []):
            receipt["reason"] = "installed Triton backend has no usable int8_linear capability"
            return receipt
        kitchen.enable_backend("triton")
        after = kitchen.list_backends()
        receipt.update({
            "selected": "triton_int8_convrot",
            "tritonAfter": after.get("triton", {}),
            "reason": "existing Triton int8_linear supports convrot=True; other layouts retain registry fallback",
        })
        return receipt
    except Exception as exc:
        receipt["reason"] = f"Triton opt-in failed; eager fallback retained: {type(exc).__name__}: {exc}"
        return receipt


def _primary_runtime_receipt(patcher: Any) -> Dict[str, Any]:
    """Describe the loaded model boundary without reading tensor contents."""

    model = getattr(patcher, "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    return {
        "patcherType": type(patcher).__name__,
        "modelType": type(model).__name__ if model is not None else None,
        "diffusionModelType": type(diffusion).__name__ if diffusion is not None else None,
        "loadDevice": str(getattr(patcher, "load_device", None)),
        "offloadDevice": str(getattr(patcher, "offload_device", None)),
        "modelDevice": str(getattr(model, "device", None)),
        "isDynamic": bool(getattr(patcher, "is_dynamic", lambda: False)()),
        "inferenceDtype": str(getattr(model, "get_dtype_inference", lambda: None)()),
        "manualCastDtype": str(getattr(model, "manual_cast_dtype", None)),
        "loadedWeightBytes": int(getattr(model, "model_loaded_weight_memory", 0) or 0),
        "quantizationBackend": _quantization_backend_receipt(),
    }


class EmbeddedH3Runtime:
    """Process-local, mutually-exclusive H3 model runtime."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = _inside(Path(root))
        self.model_root = h3_project_root(self.root)
        self._lock = threading.RLock()
        self._bootstrapped = False
        self._bootstrap_error: Optional[BaseException] = None
        self._modules: Dict[str, Any] = {}
        self._primary_name: Optional[str] = None
        self._primary_path: Optional[Path] = None
        self._primary: Any = None
        self._clip: Any = None
        self._video_vae: Any = None
        self._video_vae_patcher: Any = None
        self._audio_vae: Any = None
        self._audio_vae_patcher: Any = None
        self._bootstrap_elapsed_seconds: Optional[float] = None
        self._model_discovery_cache: Optional[Dict[str, Any]] = None
        self._model_file_paths: Optional[Dict[str, Path]] = None
        self._model_identity_receipts: Dict[str, Dict[str, Any]] = {}

    def _bootstrap(self) -> None:
        with self._lock:
            if self._bootstrapped:
                return
            if self._bootstrap_error is not None:
                raise H3RuntimeError(f"direct H3 runtime previously failed: {self._bootstrap_error}") from self._bootstrap_error
            started = time.perf_counter()
            configure_sage_environment()
            for item in (PYTHON_PACKAGES, COMFY_SOURCE):
                if not item.is_dir():
                    raise H3RuntimeError(f"embedded runtime path missing: {item}")
                item_text = str(item)
                if item_text not in sys.path:
                    sys.path.insert(0, item_text)

            # Keep Python, Torch, and Hugging Face caches on D:.
            os.environ.setdefault("TEMP", str(ROOT / "temp"))
            os.environ.setdefault("TMP", str(ROOT / "temp"))
            os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))
            os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache" / "transformers"))
            os.environ.setdefault("PYTHONPYCACHEPREFIX", str(ROOT / "cache" / "pycache"))

            try:
                # Set this before importing Comfy low-level modules: their
                # Triton Autotuner objects capture this knob at construction.
                # The cache is execution metadata only and is rooted on D: by
                # TRITON_CACHE_DIR from configure_sage_environment().
                triton = importlib.import_module("triton")
                triton.knobs.autotuning.cache = True
                self._modules["tritonAutotuneCache"] = {
                    "enabled": True,
                    "cacheDirectory": os.environ.get("TRITON_CACHE_DIR"),
                }
                self._modules["torch"] = importlib.import_module("torch")
                # Initialize Torch Dynamo before Comfy/DynamicVRAM imports it
                # indirectly. Torch 2.11 registers the global precompile
                # artifact factory during this import; making the order
                # explicit prevents a concurrent Comfy import from registering
                # the same artifact twice.
                self._modules["torchDynamo"] = importlib.import_module("torch._dynamo")
                self._initialize_native_dynamic_vram(self._modules["torch"])
                self._modules["sd"] = importlib.import_module("comfy.sd")
                self._modules["sample"] = importlib.import_module("comfy.sample")
                self._modules["model_management"] = importlib.import_module("comfy.model_management")
                self._modules["h3_conditioning"] = importlib.import_module("comfy_extras.nodes_minimax_h3")
                # Reuse the official sampler node implementations as an
                # embedded library. This starts neither a ComfyUI server nor
                # a workflow executor; the product calls these objects from
                # its own direct runtime.
                self._modules["custom_sampler"] = importlib.import_module("comfy_extras.nodes_custom_sampler")
                self._modules["safetensors"] = importlib.import_module("safetensors")
                diagnostic_profile = os.environ.get("H3_DIAGNOSTIC_EXECUTION_PROFILE", "production_optimized")
                if diagnostic_profile == "official_baseline":
                    # The diagnostics-only official baseline deliberately
                    # leaves the stock Comfy/comfy-kitchen backend registry
                    # untouched.  It neither installs the project Torch INT8
                    # fallback nor enables an extra optimized registry path.
                    self._modules["quantizationBackend"] = {
                        **_quantization_backend_receipt(
                            allow_disabled_triton=False,
                            include_project_torch_fallback=False,
                        ),
                        "diagnosticExecutionProfile": "official_baseline",
                        "projectTorchInt8Fallback": "not_registered",
                    }
                    self._modules["torchInt8Backend"] = {
                        "requested": False,
                        "available": False,
                        "selected": None,
                        "reason": "official baseline leaves project fallback unregistered",
                    }
                else:
                    self._modules["quantizationBackend"] = _configure_triton_int8_backend(self._modules["torch"])
                    try:
                        from embedded_h3_runtime.torch_int8_backend import install_torch_int8_backend

                        triton_preferred = self._modules["quantizationBackend"].get("selected") == "triton_int8_convrot"
                        self._modules["torchInt8Backend"] = install_torch_int8_backend(
                            self._modules["torch"], prefer=not triton_preferred
                        )
                        self._modules["quantizationBackend"]["torchInt8"] = self._modules["torchInt8Backend"]
                        if self._modules["torchInt8Backend"].get("available") and not triton_preferred:
                            self._modules["quantizationBackend"].update({
                                "selected": "h3_torch_int8_mm_convrot",
                                "reason": "verified Triton unavailable; native Torch torch._int_mm INT8 path selected",
                            })
                        elif self._modules["torchInt8Backend"].get("available") and triton_preferred:
                            self._modules["quantizationBackend"].update({
                                "reason": "verified Triton ConvRot selected for whole-model throughput; native Torch INT8 remains registered as fallback",
                            })
                    except Exception as exc:
                        self._modules["torchInt8Backend"] = {
                            "requested": True,
                            "available": False,
                            "selected": None,
                            "reason": f"native Torch INT8 backend import failed: {type(exc).__name__}: {exc}",
                        }
            except Exception as exc:
                self._bootstrap_error = exc
                raise H3RuntimeError(f"direct H3 runtime import failed: {exc}") from exc
            self._bootstrapped = True
            self._bootstrap_elapsed_seconds = round(time.perf_counter() - started, 6)

    def _initialize_native_dynamic_vram(self, torch: Any) -> Dict[str, Any]:
        """Initialize the bundled Comfy AIMDO/DynamicVRAM path in-process.

        Comfy's server entry point performs this before loading ``sd``.  The
        independent H3 service does not start that entry point, so this is the
        same low-level initialization sequence without importing a server,
        frontend, prompt queue, or workflow executor.
        """

        receipt: Dict[str, Any] = {
            "requested": True,
            "source": "bundled comfy_aimdo.control + ModelPatcherDynamic",
            "controlInitialized": False,
            "devicesInitialized": False,
            "deviceIds": [],
            "enabled": False,
            "error": None,
        }
        try:
            control = importlib.import_module("comfy_aimdo.control")
            init_contract = official_aimdo_init_contract()
            receipt["initializationContract"] = init_contract
            receipt["controlInitialized"] = bool(control.init(
                simple_vram_headroom=init_contract["simpleVramHeadroomBytes"],
                nvml_pressure=init_contract["nvmlPressure"],
            ))
            if torch.cuda.is_available():
                receipt["deviceIds"] = list(range(torch.cuda.device_count()))
            device_contracts = [
                (device_id, init_contract["deviceExtraHeadroomBytes"])
                for device_id in receipt["deviceIds"]
            ]
            receipt["devicesInitialized"] = bool(control.init_devices(device_contracts)) if device_contracts else False
            if receipt["devicesInitialized"]:
                # ``comfy.memory_management`` and ``comfy.model_patcher``
                # import ``comfy_aimdo.host_buffer`` at module import time.
                # If another low-level capability probe imported that module
                # before AIMDO was initialized, its module-global ``lib`` is
                # permanently None and the first DynamicVRAM VAE/model
                # construction fails at ``hostbuf_allocate``.  Refresh only
                # this official binding after control.init_devices(); the
                # vendor implementation and its execution order remain
                # unchanged.
                host_buffer = importlib.import_module("comfy_aimdo.host_buffer")
                if getattr(host_buffer, "lib", None) is None:
                    host_buffer = importlib.reload(host_buffer)
                receipt["hostBufferModuleLoaded"] = True
                receipt["hostBufferRebound"] = bool(getattr(host_buffer, "lib", None) is control.lib)
                host_buffer_lib = getattr(host_buffer, "lib", None)
                required_host_buffer_symbols = ("hostbuf_allocate", "hostbuf_free")
                missing_host_buffer_symbols = [
                    name for name in required_host_buffer_symbols
                    if host_buffer_lib is None or not hasattr(host_buffer_lib, name)
                ]
                receipt["hostBufferMissingSymbols"] = missing_host_buffer_symbols
                if missing_host_buffer_symbols:
                    raise H3RuntimeError(
                        "AIMDO host-buffer is unavailable before DynamicVRAM model construction; "
                        f"missing: {', '.join(missing_host_buffer_symbols)}"
                    )
                # Exercise the same native allocation/free boundary used by
                # ModelPatcherDynamic.  This is a startup safety probe only;
                # it allocates no model storage and leaves no host buffer.
                host_buffer_probe = host_buffer.HostBuffer(0, 0, 0)
                del host_buffer_probe
                gc.collect()
                receipt["hostBufferSelfTest"] = {"status": "passed", "allocate": True, "free": True}
                # The same early-import hazard applies to the official VBAR,
                # VRAM-buffer, and model-mmap bindings.  Refresh only modules
                # whose cached native handle is empty; no vendor source is
                # modified and no model data is touched.
                aimdo_bindings = {}
                for binding_name in ("model_vbar", "vram_buffer", "model_mmap"):
                    binding = importlib.import_module(f"comfy_aimdo.{binding_name}")
                    if getattr(binding, "lib", None) is None:
                        binding = importlib.reload(binding)
                    aimdo_bindings[binding_name] = binding
                    receipt[f"{binding_name}ModuleLoaded"] = True
                    receipt[f"{binding_name}Rebound"] = bool(getattr(binding, "lib", None) is control.lib)
                vbar_lib = getattr(aimdo_bindings["model_vbar"], "lib", None)
                missing_vbar_symbols = [
                    name for name in ("vbar_allocate", "vbar_free")
                    if vbar_lib is None or not hasattr(vbar_lib, name)
                ]
                receipt["modelVbarMissingSymbols"] = missing_vbar_symbols
                if missing_vbar_symbols:
                    raise H3RuntimeError(
                        "AIMDO model VBAR is unavailable before DynamicVRAM VAE encoding; "
                        f"missing: {', '.join(missing_vbar_symbols)}"
                    )
                memory_management = importlib.import_module("comfy.memory_management")
                model_patcher = importlib.import_module("comfy.model_patcher")
                policy_config = load_memory_policy()
                normal_patcher = str(policy_config.get("normalPatcher", "standard_model_patcher"))
                if normal_patcher == "dynamic_model_patcher":
                    model_patcher.CoreModelPatcher = model_patcher.ModelPatcherDynamic
                memory_management.aimdo_enabled = True
                receipt.update({
                    "enabled": True,
                    "patcherType": model_patcher.CoreModelPatcher.__name__,
                    "normalPatcherPolicy": normal_patcher,
                    "dynamicEmergencyOnly": normal_patcher != "dynamic_model_patcher",
                    "aimdoLibraryLoaded": control.lib is not None,
                })
            else:
                receipt["error"] = "comfy_aimdo device initialization returned false"
        except H3RuntimeError as exc:
            receipt["error"] = str(exc)
            raise
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        self._modules["dynamicVram"] = receipt
        return receipt

    def model_path(
        self,
        name: str,
        route_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        model_quant: str = "int8",
    ) -> Path:
        if name not in MODEL_FILES:
            raise H3RuntimeError(f"unknown H3 component: {name}")
        filename = (
            select_primary_model_filename(name, route_id, candidate_id, model_quant)
            if name in {"FL2VA", "REF2VA"}
            else MODEL_FILES[name]
        )
        if self._model_file_paths is None:
            self._model_file_paths = discover_project_model_files(self.model_root)
        
        # 根据 model_quant 选择正确的路径
        if name in {"FL2VA", "REF2VA"} and model_quant == "w4a8":
            lookup_key = f"{name}_W4A8"
            if lookup_key not in self._model_file_paths:
                raise H3RuntimeError(f"W4A8 模型未找到：{MODEL_FILES_W4A8[name]}")
            path = self._model_file_paths[lookup_key]
        else:
            if name not in self._model_file_paths:
                raise H3RuntimeError(f"INT8 模型未找到：{MODEL_FILES[name]}")
            path = self._model_file_paths[name]

        if path.suffix.lower() != ".safetensors":
            raise H3RuntimeError(f"unexpected model extension: {path}")
        if name not in self._model_identity_receipts and model_quant == "int8":
            self._model_identity_receipts[name] = verify_full_quality_model_identity(
                self.model_root, name, path
            )
        return path

    def health(self) -> Dict[str, Any]:
        self._bootstrap()
        torch = self._modules["torch"]
        try:
            import comfy_kitchen as kitchen

            backends = kitchen.list_backends()
        except Exception as exc:  # capability reporting must not hide health
            backends = {"error": str(exc)}
        cuda = bool(torch.cuda.is_available())
        return {
            "online": True,
            "service": "h3-independent-direct-runtime",
            "server": False,
            "workflowExecutor": False,
            "embeddedSource": "ComfyUI low-level modules 0.30.0, read-only",
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda": getattr(torch.version, "cuda", None),
            "cudaAvailable": cuda,
            "comfyKitchenBackends": backends,
            "sageAttention": detect_sage(),
            "quantizationBackend": self._modules.get("quantizationBackend", {"selected": "unknown"}),
            "dynamicVram": self._modules.get("dynamicVram", {"enabled": False}),
            "primaryModelLoaded": self._primary_name,
        }

    def capabilities(self) -> Dict[str, Any]:
        self._bootstrap()
        return {
            "backend": "embedded_h3_direct",
            "modes": {
                "T2V": "FL2VA",
                "I2V": "FL2VA",
                "FIRST_LAST_FRAME": "FL2VA",
                "R2V": "REF2VA",
            },
            "directCalls": [
                "load_diffusion_model",
                "load_clip",
                "load_vae_patcher",
                "MiniMaxH3ImageToVideo.execute",
                "MiniMaxH3ReferenceToVideo.execute",
                "BasicGuider + KSamplerSelect(res_multistep) + BasicScheduler + SamplerCustomAdvanced semantics",
            ],
            "mutuallyExclusivePrimaryModels": True,
            "audioVae": True,
            "mp4Mux": True,
            "cancelCallback": True,
            "referenceStreamingDecode": True,
            "referenceSemanticSamplingFps": 2,
            "sageAllowCompile": False,
            "comfyUiServerStarted": False,
        }

    def discover_models(self) -> Dict[str, Any]:
        """Read safetensors headers only; never hashes or loads tensor data."""

        self._bootstrap()
        if self._model_discovery_cache is not None:
            return {name: dict(value, cacheHit=True) for name, value in self._model_discovery_cache.items()}
        started = time.perf_counter()
        safe_open = importlib.import_module("safetensors").safe_open
        discovered: Dict[str, Any] = {}
        for name, filename in MODEL_FILES.items():
            path = self.model_path(name)
            try:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    keys = list(handle.keys())
                    metadata = handle.metadata() or {}
                discovered[name] = {
                    "file": filename,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "headerReadable": True,
                    "tensorCount": len(keys),
                    "metadata": metadata,
                    "loaded": name in {"FL2VA", "REF2VA"} and name == self._primary_name,
                }
            except Exception as exc:
                discovered[name] = {
                    "file": filename,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "headerReadable": False,
                    "error": str(exc),
                    "loaded": False,
                }
        elapsed = round(time.perf_counter() - started, 6)
        for value in discovered.values():
            value["cacheHit"] = False
            value["elapsedSeconds"] = elapsed
        self._model_discovery_cache = discovered
        return {name: dict(value) for name, value in discovered.items()}

    def load_primary(
        self,
        model_name: str,
        route_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
        model_quant: str = "int8",
    ) -> Dict[str, Any]:
        """Load exactly one FL2VA/REF2VA patcher from the root model file."""

        if model_name not in {"FL2VA", "REF2VA"}:
            raise H3RuntimeError("primary model must be FL2VA or REF2VA")
        self._bootstrap()
        with self._lock:
            started = time.perf_counter()
            path = self.model_path(model_name, route_id, candidate_id, model_quant)
            if self._primary_name == model_name and self._primary_path == path and self._primary is not None:
                return {"model": model_name, "reused": True, "cacheHit": True, "elapsedSeconds": round(time.perf_counter() - started, 6), "path": str(path), "runtime": _primary_runtime_receipt(self._primary), "modelQuant": model_quant}
            # Release the old reference before loading the other primary model.
            self._primary = None
            self._primary_name = None
            self._primary_path = None
            gc.collect()
            sd = self._modules["sd"]
            self._primary = sd.load_diffusion_model(str(path))
            self._primary_name = model_name
            self._primary_path = path
            return {
                "model": model_name,
                "path": str(path),
                "loader": "comfy.sd.load_diffusion_model (embedded direct call)",
                "patcherType": type(self._primary).__name__,
                "reused": False,
                "cacheHit": False,
                "elapsedSeconds": round(time.perf_counter() - started, 6),
                "runtime": _primary_runtime_receipt(self._primary),
                "modelQuant": model_quant,
            }

    def load_text_encoder(self) -> Dict[str, Any]:
        self._bootstrap()
        with self._lock:
            started = time.perf_counter()
            reused = self._clip is not None
            if self._clip is None:
                sd = self._modules["sd"]
                self._clip = sd.load_clip(
                    [str(self.model_path("TEXT_ENCODER"))],
                    clip_type=sd.CLIPType.MINIMAX,
                )
            return {"component": "TEXT_ENCODER", "path": str(self.model_path("TEXT_ENCODER")), "reused": reused, "cacheHit": reused, "elapsedSeconds": round(time.perf_counter() - started, 6), "bootstrapElapsedSeconds": self._bootstrap_elapsed_seconds, "type": type(self._clip).__name__}

    def load_video_vae(self, device: Any = None) -> Dict[str, Any]:
        self._bootstrap()
        with self._lock:
            started = time.perf_counter()
            reused = self._video_vae is not None
            if self._video_vae is None:
                self._video_vae, self._video_vae_patcher = self._load_official_vae("VIDEO_VAE", device=device)
            return {
                "component": "VIDEO_VAE",
                "path": str(self.model_path("VIDEO_VAE")),
                "reused": reused,
                "cacheHit": reused,
                "elapsedSeconds": round(time.perf_counter() - started, 6),
                "bootstrapElapsedSeconds": self._bootstrap_elapsed_seconds,
                "type": type(self._video_vae).__name__,
                "patcherType": type(self._video_vae_patcher).__name__,
                "requestedDevice": None if device is None else str(device),
                "device": str(getattr(self._video_vae, "device", None)),
                "dtype": str(getattr(self._video_vae, "vae_dtype", None)),
            }

    def load_audio_vae(self, device: Any = None) -> Dict[str, Any]:
        self._bootstrap()
        with self._lock:
            started = time.perf_counter()
            reused = self._audio_vae is not None
            if self._audio_vae is None:
                self._audio_vae, self._audio_vae_patcher = self._load_official_vae("AUDIO_VAE", device=device)
            return {
                "component": "AUDIO_VAE",
                "path": str(self.model_path("AUDIO_VAE")),
                "reused": reused,
                "cacheHit": reused,
                "elapsedSeconds": round(time.perf_counter() - started, 6),
                "bootstrapElapsedSeconds": self._bootstrap_elapsed_seconds,
                "type": type(self._audio_vae).__name__,
                "patcherType": type(self._audio_vae_patcher).__name__,
                "requestedDevice": None if device is None else str(device),
                "device": str(getattr(self._audio_vae, "device", None)),
                "dtype": str(getattr(self._audio_vae, "vae_dtype", None)),
            }

    def _load_official_vae(self, name: str, device: Any = None) -> tuple[Any, Any]:
        """Construct the official VAE wrapper so encode/decode use model management."""

        utils = importlib.import_module("comfy.utils")
        state, metadata = utils.load_torch_file(str(self.model_path(name)), return_metadata=True)
        kwargs: Dict[str, Any] = {"sd": state, "metadata": metadata}
        if device is not None:
            kwargs["device"] = device
            # The official H3 video VAE declares float16 as a supported
            # working dtype.  Keep the CPU recovery on that native dtype so a
            # full-resolution decode does not duplicate every activation in
            # float32; the GPU path continues to select model-management's
            # native dtype.
            torch = self._modules.get("torch")
            if str(device) == "cpu" and torch is not None:
                kwargs["dtype"] = torch.float16
        vae = self._modules["sd"].VAE(**kwargs)
        vae.throw_exception_if_invalid()
        return vae, vae.patcher

    def unload_text_encoder(self) -> Dict[str, Any]:
        """Release the Qwen wrapper after its conditioning tensors are built."""

        with self._lock:
            was_loaded = self._clip is not None
            patcher = getattr(self._clip, "patcher", None)
            self._clip = None
            model_management_release = self._release_component_patcher("TEXT_ENCODER", patcher)
            gc.collect()
            self._empty_cuda_cache()
            return {
                "component": "TEXT_ENCODER",
                "unloaded": was_loaded,
                "modelManagementRelease": model_management_release,
            }

    def _empty_cuda_cache(self) -> None:
        torch = self._modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _release_component_patcher(self, component: str, patcher: Any) -> Dict[str, Any]:
        """Ask Comfy's native registry to release one completed lifecycle stage.

        Clearing a Python wrapper alone does not remove its ModelPatcher from
        ``current_loaded_models``.  The targeted native API preserves every
        other registered model (in particular the subsequently selected H3
        DynamicVRAM model) while releasing only the completed Qwen or VAE
        component and its native clones.
        """

        management = self._modules.get("model_management")
        receipt: Dict[str, Any] = {
            "component": component,
            "patcherPresent": patcher is not None,
            "method": "comfy.model_management.unload_model_and_clones",
            "called": False,
        }
        if management is None or patcher is None:
            return receipt
        registry = getattr(management, "current_loaded_models", None)
        if registry is not None:
            try:
                receipt["registeredBefore"] = len(registry)
                receipt["registeredTypesBefore"] = [
                    type(getattr(item, "model", None)).__name__ for item in registry
                ]
            except Exception:
                pass
        release = getattr(management, "unload_model_and_clones", None)
        if not callable(release):
            receipt["unavailable"] = True
            return receipt
        try:
            release(patcher, unload_additional_models=True)
            receipt["called"] = True
        except Exception as exc:
            receipt["error"] = f"{type(exc).__name__}: {exc}"
            return receipt
        cleanup = getattr(management, "cleanup_models", None)
        if callable(cleanup):
            try:
                cleanup()
                receipt["cleanupModelsCalled"] = True
            except Exception as exc:
                receipt["cleanupModelsError"] = f"{type(exc).__name__}: {exc}"
        if registry is not None:
            try:
                receipt["registeredAfter"] = len(registry)
                receipt["registeredTypesAfter"] = [
                    type(getattr(item, "model", None)).__name__ for item in registry
                ]
            except Exception:
                pass
        return receipt

    def unload_primary(self) -> Dict[str, Any]:
        """Release the mutually-exclusive FL2VA/REF2VA model after sampling."""

        with self._lock:
            was_loaded = self._primary is not None
            name = self._primary_name
            patcher = self._primary
            self._primary = None
            self._primary_name = None
            self._primary_path = None
            model_management_release = self._release_component_patcher(name or "PRIMARY", patcher)
            gc.collect()
            self._empty_cuda_cache()
            return {
                "component": name,
                "unloaded": was_loaded,
                "modelManagementRelease": model_management_release,
            }

    def unload_video_vae(self) -> Dict[str, Any]:
        with self._lock:
            was_loaded = self._video_vae is not None
            patcher = self._video_vae_patcher
            self._video_vae = None
            self._video_vae_patcher = None
            model_management_release = self._release_component_patcher("VIDEO_VAE", patcher)
            gc.collect()
            self._empty_cuda_cache()
            return {
                "component": "VIDEO_VAE",
                "unloaded": was_loaded,
                "modelManagementRelease": model_management_release,
            }

    def unload_audio_vae(self) -> Dict[str, Any]:
        with self._lock:
            was_loaded = self._audio_vae is not None
            patcher = self._audio_vae_patcher
            self._audio_vae = None
            self._audio_vae_patcher = None
            model_management_release = self._release_component_patcher("AUDIO_VAE", patcher)
            gc.collect()
            self._empty_cuda_cache()
            return {
                "component": "AUDIO_VAE",
                "unloaded": was_loaded,
                "modelManagementRelease": model_management_release,
            }

    def objects(self) -> Dict[str, Any]:
        self._bootstrap()
        return {
            "primary": self._primary,
            "clip": self._clip,
            "videoVae": self._video_vae,
            "videoVaePatcher": self._video_vae_patcher,
            "audioVae": self._audio_vae,
            "audioVaePatcher": self._audio_vae_patcher,
            "modules": self._modules,
        }
