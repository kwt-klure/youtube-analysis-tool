from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
from typing import Any

PROJECT_NAME = "youtube-analysis-tool"
UNKNOWN_VERSION = "0+unknown"

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = None  # type: ignore[assignment]


def package_version() -> str:
    source_version = pyproject_version()
    if source_version is not None:
        return source_version
    try:
        return metadata.version(PROJECT_NAME)
    except metadata.PackageNotFoundError:
        return UNKNOWN_VERSION


def add_version_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--version", action="version", version=f"{PROJECT_NAME} {package_version()}")
    return parser


def pyproject_version() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return None

    if tomllib is not None:
        data: dict[str, Any] = tomllib.loads(pyproject_text)
        version = (data.get("project") or {}).get("version")
        return str(version) if version else None

    return pyproject_version_from_text(pyproject_text)


def pyproject_version_from_text(pyproject_text: str) -> str | None:
    in_project_section = False
    for raw_line in pyproject_text.splitlines():
        line = raw_line.strip()
        if line == "[project]":
            in_project_section = True
            continue
        if in_project_section and line.startswith("["):
            return None
        if in_project_section and line.startswith("version"):
            key, separator, value = line.partition("=")
            if separator and key.strip() == "version":
                return value.strip().strip('"').strip("'") or None
    return None
