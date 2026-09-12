"""Immutable product contract for the only supported H3 route."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict


FULL_QUALITY_CONTRACT_VERSION = "h3-full-quality-product-v4"
FULL_QUALITY_MODE_CONTRACTS = {
    "T2V": {"primary": "FL2VA", "conditioningBranch": "MiniMaxH3ImageToVideo.text"},
    "I2V": {"primary": "FL2VA", "conditioningBranch": "MiniMaxH3ImageToVideo.first_frame"},
    "FIRST_LAST_FRAME": {"primary": "FL2VA", "conditioningBranch": "MiniMaxH3ImageToVideo.first_last_frame"},
    "R2V": {"primary": "REF2VA", "conditioningBranch": "MiniMaxH3ReferenceToVideo.reference_media"},
}
FULL_QUALITY_MODEL_EVIDENCE = {
    "FL2VA": {
        "relativePath": "models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "bytes": 20_970_379_616,
        "sha256": "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a",
        "pruned": True,
    },
    "REF2VA": {
        "relativePath": "models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "bytes": 20_970_379_616,
        "sha256": "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779",
        "pruned": True,
    },
    "TEXT_ENCODER": {
        "relativePath": "models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "bytes": 15_687_142_551,
        "sha256": "PENDING_VERIFICATION",
    },
    "VIDEO_VAE": {
        "relativePath": "models/minimax_h3_video_vae_fp16.safetensors",
        "bytes": 5_207_808_496,
        "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522",
    },
    "AUDIO_VAE": {
        "relativePath": "models/minimax_h3_audio_vae_fp32.safetensors",
        "bytes": 605_254_808,
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
    },
}
FULL_QUALITY_SAMPLING_CONTRACT = {
    "sampler": "res_multistep",
    "scheduler": "simple",
    "steps": 20,
    "denoise": 1.0,
    "guider": "BasicGuider",
    "conditioningInputs": ["positive"],
    "usesNegativeConditioning": False,
    "cfg": None,
    "noise": "RandomNoise",
    "executor": "SamplerCustomAdvanced",
}
FULL_QUALITY_EXECUTION_SWITCHES = {
    "executionProfile": "production_kj_sage_ffn",
    "kjH3Sage": {
        "enabled": True,
        "policy": "existing_auto_policy",
        "dependencyPolicy": "fail_closed",
        "classification": "official_contract_compatible_third_party_optimization",
        "author": "Kijai",
        "source": "kijai/ComfyUI-KJNodes",
        "commit": "ab8f90f02ad6ec3a4900b1b4df9c03cded7b4690",
        "license": "GPL-3.0",
        "sageAttentionVersion": "2.2.0",
        "cudaArchitecture": "sm120",
        "requiredPatchedBlocks": 50,
        "scope": "denoiser_only",
        "evidenceLabels": {
            "REAL_RUN_VERIFIED": True,
            "OWNER_ACCEPTED": True,
            "DEFAULT_ENABLED": True,
            "PERFORMANCE_AB_VALIDATED": False,
        },
    },
    "ffnChunk": {"enabled": True, "chunks": 2, "minTokens": 4096},
    "dynamicVram": {
        "decisionSource": "native_dynamic_vram",
        "offloadPlanStatus": "NATIVE_MANAGED",
        "projectWatermarkInjection": False,
        "forceFullLoad": False,
    },
    "sharedMechanisms": ["video_vae", "audio_vae", "mp4_mux"],
}


def full_quality_contract(mode: str, kernel_backend: str = "kijai_fast", steps: int = 20, ffn_chunks: int | None = None) -> Dict[str, Any]:
    normalized = str(mode or "").upper()
    if normalized not in FULL_QUALITY_MODE_CONTRACTS:
        raise ValueError(f"unsupported full_quality mode: {normalized}")
    mode_contract = FULL_QUALITY_MODE_CONTRACTS[normalized]
    primary = mode_contract["primary"]
    if kernel_backend not in {"kijai_fast", "official_native"}:
        raise ValueError(f"unsupported full_quality kernel backend: {kernel_backend}")
    if int(steps) not in {20, 30, 40, 50, 60}:
        raise ValueError(f"unsupported full_quality steps: {steps}")
    execution_switches = copy.deepcopy(FULL_QUALITY_EXECUTION_SWITCHES)
    sampling = copy.deepcopy(FULL_QUALITY_SAMPLING_CONTRACT)
    sampling["steps"] = int(steps)
    if kernel_backend == "kijai_fast" and ffn_chunks not in {None, 1, 2, 4}:
        raise ValueError("full_quality FFN chunking must be 1, 2, or 4")
    if kernel_backend == "official_native" and ffn_chunks is not None:
        raise ValueError("official_native does not use FFN chunking")
    execution_switches["selectedKernel"] = kernel_backend
    if kernel_backend == "kijai_fast" and ffn_chunks is not None:
        execution_switches["ffnChunk"] = {
            "enabled": True,
            "chunks": int(ffn_chunks),
            "minTokens": 4096,
        }
    if kernel_backend == "official_native":
        execution_switches["kjH3Sage"] = {
            "enabled": False,
            "policy": "explicit_official_native",
            "dependencyPolicy": "not_loaded",
        }
        execution_switches["ffnChunk"] = {"enabled": False, "requestedChunks": ffn_chunks}
        execution_switches["officialNativeAttention"] = {
            "enabled": True,
            "backend": "comfyui_official_native_attention",
            "fallback": False,
        }
    return copy.deepcopy({
        "contractVersion": FULL_QUALITY_CONTRACT_VERSION,
        "routeId": "full_quality",
        "mode": normalized,
        "primaryModel": FULL_QUALITY_MODEL_EVIDENCE[primary],
        "sharedModels": {
            key: FULL_QUALITY_MODEL_EVIDENCE[key]
            for key in ("TEXT_ENCODER", "VIDEO_VAE", "AUDIO_VAE")
        },
        "conditioningBranch": mode_contract["conditioningBranch"],
        "sampling": sampling,
        "executionSwitches": execution_switches,
    })


def full_quality_fingerprint(mode: str, kernel_backend: str = "kijai_fast", steps: int = 20, ffn_chunks: int | None = None) -> str:
    canonical = json.dumps(full_quality_contract(mode, kernel_backend, steps, ffn_chunks), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
