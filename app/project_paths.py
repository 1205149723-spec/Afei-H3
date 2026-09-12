"""Portable paths rooted at the HailuoH3 project directory."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


PROGRAM_ROOT = Path(__file__).resolve().parent.parent


def h3_project_root(program_root: Path = PROGRAM_ROOT) -> Path:
    """Return the outer portable HailuoH3 folder for the agreed nested layout."""
    package_root = os.environ.get("H3_PACKAGE_ROOT")
    if package_root:
        return Path(package_root).resolve()
    program = Path(program_root).resolve()
    # Candidate code shares the project's immutable models/input/output via
    # junctions. Resolve the shared project boundary before validating paths.
    for parent in (program, *program.parents):
        if parent.name.casefold() == "hailuoh3":
            return parent
    parent = program.parent
    if program.name.casefold() == "hailuoh3" and parent.name.casefold() == program.name.casefold():
        return parent
    return program


PROJECT_ROOT = h3_project_root()


def portable_project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> str:
    """Return a project-relative path when a value belongs to this H3 project."""
    text = str(value)
    root = Path(project_root).resolve()
    candidate = Path(text)
    if not candidate.is_absolute():
        return text
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except ValueError:
        # A task archive may have been written before the project moved to a
        # new disk. Its old absolute prefix is irrelevant once HailuoH3 is the
        # portability boundary, so retain only the portion below that root.
        windows = PureWindowsPath(text)
        positions = [index for index, part in enumerate(windows.parts) if part.casefold() == root.name.casefold()]
        if positions:
            remainder = windows.parts[positions[-1] + 1:]
            if remainder:
                return PurePosixPath(*remainder).as_posix()
        return text


def portable_project_record(value: Any, project_root: Path = PROJECT_ROOT) -> Any:
    """Copy a task/receipt payload while removing project-root absolute paths."""
    if isinstance(value, dict):
        portable = {}
        for key, item in value.items():
            normalized = portable_project_record(item, project_root)
            if isinstance(item, (str, Path)) and str(key).casefold().endswith(("path", "directory")):
                normalized = portable_project_path(item, project_root)
            portable[key] = normalized
        return portable
    if isinstance(value, list):
        return [portable_project_record(item, project_root) for item in value]
    return value
