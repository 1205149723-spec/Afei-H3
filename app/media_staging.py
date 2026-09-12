"""Safe local upload assets under input/, with legacy staging compatibility."""

from __future__ import annotations

import mimetypes
import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from project_paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
INPUT_ROOT = ROOT / "input"
LEGACY_STAGING_ROOT = ROOT / "staging"
# Kept as an import-compatible name for callers/tests; the product default is input/.
STAGING_ROOT = INPUT_ROOT
MANIFEST_NAME = ".h3-asset.json"
METADATA_ROOT = ROOT / "temp" / "input_asset_meta"
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
KIND_SIZE_LIMITS = {"image": 30 * 1024 * 1024, "video": 50 * 1024 * 1024, "audio": 15 * 1024 * 1024}
ALLOWED_PREFIXES = ("image/", "video/", "audio/")
STREAM_CHUNK_BYTES = 1024 * 1024


class StagingError(ValueError):
    pass


def _sniff_media(header: bytes, filename: str = "") -> tuple[str, str]:
    """Identify the supported media family from bytes, never client metadata."""

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if header[:6] in {b"GIF87a", b"GIF89a"}:
        return "image", "image/gif"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "image", "image/tiff"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image", "image/webp"
    if header.startswith(b"RIFF") and header[8:12] == b"WAVE":
        return "audio", "audio/wav"
    if header.startswith(b"fLaC"):
        return "audio", "audio/flac"
    if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0):
        return "audio", "audio/mpeg"
    if header.startswith(b"OggS"):
        return ("video", "video/ogg") if b"theora" in header.lower() else ("audio", "audio/ogg")
    if len(header) >= 12 and header[4:8] == b"ftyp":
        brands = header[8:64].lower()
        if any(brand in brands for brand in (b"heic", b"heix", b"hevc", b"hevx", b"heif", b"mif1", b"msf1")):
            return "image", "image/heic"
        return "video", "video/mp4"
    if header.startswith(b"\x1aE\xdf\xa3"):
        return "video", "video/webm"
    guessed = (mimetypes.guess_type(filename)[0] or "").lower()
    if guessed.startswith(ALLOWED_PREFIXES):
        return guessed.split("/", 1)[0], guessed
    raise StagingError("无法从文件真实内容识别图片、视频或音频格式")


def _safe_name(name: str) -> str:
    raw = Path(str(name or "")).name
    raw = re.sub(r"[^0-9A-Za-z._()\-\u4e00-\u9fff ]+", "_", raw).strip(" .")
    if not raw or raw in {".", ".."}:
        raise StagingError("upload needs a safe filename")
    return raw[:180]


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().relative_to(STAGING_ROOT.resolve().parent).as_posix()


def stage_bytes(filename: str, payload: bytes, mime_type: str = "") -> Dict[str, Any]:
    from io import BytesIO

    return stage_stream(filename, BytesIO(payload), len(payload), mime_type)


def stage_stream(
    filename: str,
    source: BinaryIO,
    content_length: int | None,
    client_mime_type: str = "",
    *,
    chunk_size: int = STREAM_CHUNK_BYTES,
    probe: bool = False,
) -> Dict[str, Any]:
    """Persist one original upload with bounded memory and atomic publication."""

    try:
        expected = int(content_length) if content_length is not None else None
    except (TypeError, ValueError) as exc:
        raise StagingError("上传大小无效") from exc
    if expected is not None and (expected <= 0 or expected > MAX_UPLOAD_BYTES):
        raise StagingError(f"上传大小必须在 1 到 {MAX_UPLOAD_BYTES} 字节之间")
    if chunk_size <= 0:
        raise StagingError("上传分块大小无效")
    asset_id = uuid.uuid4().hex[:16]
    safe_name = _safe_name(filename)
    pending_root = (STAGING_ROOT / ".uploading").resolve()
    pending_root.mkdir(parents=True, exist_ok=True)
    partial = pending_root / f"{asset_id}.partial"
    digest = hashlib.sha256()
    total = 0
    try:
        with partial.open("xb") as handle:
            while expected is None or total < expected:
                read_size = chunk_size if expected is None else min(chunk_size, expected - total)
                block = source.read(read_size)
                if not block:
                    if expected is None:
                        break
                    raise StagingError("上传在文件接收完成前中断")
                total += len(block)
                if (expected is not None and total > expected) or total > MAX_UPLOAD_BYTES:
                    raise StagingError("上传内容超过声明大小")
                digest.update(block)
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
        if total <= 0:
            raise StagingError("上传文件为空")
        with partial.open("rb") as handle:
            header = handle.read(512)
        kind, mime = _sniff_media(header, safe_name)
        if total > KIND_SIZE_LIMITS[kind]:
            raise StagingError(f"{kind} 文件超过当前锁定官方单文件大小限制")
        media_facts = None
        if probe:
            from media_probe import MediaProbeError, probe_media
            try:
                media_facts = probe_media(partial)
            except MediaProbeError as exc:
                raise StagingError(str(exc)) from exc
            if media_facts["kind"] != kind:
                kind = str(media_facts["kind"])
        asset_dir = (STAGING_ROOT / kind / asset_id).resolve()
        asset_dir.relative_to(STAGING_ROOT.resolve())
        asset_dir.mkdir(parents=True, exist_ok=False)
        target = (asset_dir / safe_name).resolve()
        target.relative_to(asset_dir)
        os.replace(partial, target)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        asset_dir_value = locals().get("asset_dir")
        if isinstance(asset_dir_value, Path) and asset_dir_value.exists():
            shutil.rmtree(asset_dir_value, ignore_errors=True)
        if isinstance(exc, StagingError):
            raise
        raise StagingError(f"上传写入失败：{exc}") from exc
    try:
        portable_path = _portable_path(target)
        portable_root = STAGING_ROOT.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        portable_path = target.relative_to(STAGING_ROOT.resolve().parent).as_posix()
        portable_root = STAGING_ROOT.name
    manifest = {
        "assetId": asset_id,
        "createdBy": "h3-upload",
        "cleanupEligible": True,
        "owners": [],
        "name": safe_name,
        "kind": kind,
        "mimeType": mime,
        "clientMimeType": str(client_mime_type or "").lower(),
        "size": total,
        "sha256": digest.hexdigest(),
        "path": portable_path,
        "mediaFacts": media_facts,
    }
    try:
        METADATA_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = METADATA_ROOT / f"{asset_id}.json"
        temporary_manifest = METADATA_ROOT / f".{asset_id}.tmp"
        temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
    except Exception as exc:
        shutil.rmtree(asset_dir, ignore_errors=True)
        raise StagingError(f"上传资产记录写入失败：{exc}") from exc
    return {
        "assetId": asset_id,
        "name": safe_name,
        "kind": kind,
        "mimeType": mime,
        "clientMimeType": str(client_mime_type or "").lower(),
        "size": total,
        "sha256": digest.hexdigest(),
        "path": portable_path,
        "root": portable_root,
        "mediaFacts": media_facts,
    }


def resolve_staged_path(value: str) -> Path:
    """Resolve input/ or legacy staging/ paths without trusting them."""
    raw = str(value or "").strip()
    if not raw:
        raise StagingError("reference path is required")
    path = Path(raw)
    root = STAGING_ROOT.resolve()
    legacy = LEGACY_STAGING_ROOT.resolve()
    if not path.is_absolute():
        # Persisted records may use either ``staging/...`` (nested project
        # root) or ``HailuoH3/staging/...`` (outer portable root).
        parts = path.parts
        if parts and parts[0].casefold() == root.parent.name.casefold():
            path = root.parent.parent.joinpath(*parts)
        else:
            path = root.parent.joinpath(*parts)
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        try:
            resolved.relative_to(legacy)
        except ValueError as exc:
            raise StagingError("reference must remain inside project input") from exc
    return resolved


def validate_staged_references(references: Any) -> list[Dict[str, Any]]:
    """Validate real-task references before a worker is accepted."""
    validated: list[Dict[str, str]] = []
    for index, reference in enumerate(references if isinstance(references, list) else []):
        if not isinstance(reference, dict):
            continue
        raw_path = str(reference.get("path") or "")
        path = resolve_staged_path(raw_path)
        if not path.is_file():
            raise StagingError(f"第 {index + 1} 个参考素材在项目输入目录中不存在")
        expected_sha = str(reference.get("sha256") or reference.get("contentHash") or "").strip().lower()
        if not expected_sha:
            raise StagingError(f"第 {index + 1} 个参考素材缺少上传 SHA256，真实任务不能启动")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(STREAM_CHUNK_BYTES), b""):
                digest.update(block)
        actual_sha = digest.hexdigest()
        if expected_sha and actual_sha != expected_sha:
            raise StagingError(f"第 {index + 1} 个参考素材 SHA256 校验失败，文件内容已变化")
        try:
            portable = path.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            portable = path.relative_to(STAGING_ROOT.resolve().parent).as_posix()
        from media_probe import MediaProbeError, probe_media
        try:
            media_facts = probe_media(path)
        except MediaProbeError as exc:
            raise StagingError(str(exc)) from exc
        validated.append({
            "path": portable,
            "sha256": actual_sha,
            "size": path.stat().st_size,
            "kind": media_facts["kind"],
            "mediaFacts": media_facts,
        })
    return validated


def claim_assets_for_task(references: Any, task_id: str) -> list[Dict[str, str]]:
    """Claim only newly-created upload assets for one task.

    References outside the project staging root are caller-owned and are left
    untouched. Legacy staging directories without our manifest are also left
    untouched, which makes the transition non-destructive.
    """
    claims: list[Dict[str, str]] = []
    root = STAGING_ROOT.resolve()
    for reference in references if isinstance(references, list) else []:
        if not isinstance(reference, dict):
            continue
        raw_path = str(reference.get("path") or "")
        if not raw_path:
            continue
        try:
            path = resolve_staged_path(raw_path)
        except StagingError:
            continue
        try:
            path.relative_to(root)
        except ValueError:
            continue
        asset_dir = path.parent
        manifest_path = METADATA_ROOT / f"{asset_dir.name}.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if manifest.get("createdBy") != "h3-upload" or not manifest.get("cleanupEligible"):
            continue
        owners = {str(owner) for owner in manifest.get("owners", []) if owner}
        owners.add(str(task_id))
        manifest["owners"] = sorted(owners)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        claims.append({"assetId": str(manifest.get("assetId") or asset_dir.name), "path": _portable_path(path)})
    return claims


def release_task_assets(claims: Any, task_id: str) -> Dict[str, Any]:
    """Retain input assets; task completion never deletes user inputs."""
    retained: list[str] = []
    for claim in claims if isinstance(claims, list) else []:
        if not isinstance(claim, dict):
            continue
        try:
            retained.append(_portable_path(resolve_staged_path(str(claim.get("path") or ""))))
        except StagingError:
            continue
    return {
        "taskId": str(task_id),
        "released": [],
        "retained": retained,
        "automaticCleanup": "disabled",
    }
