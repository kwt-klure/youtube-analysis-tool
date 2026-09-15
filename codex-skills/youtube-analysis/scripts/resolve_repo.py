"""Print one runnable checkout path without importing the analysis package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


OVERRIDES = ("YOUTUBE_ANALYSIS_REPO", "MIRA_YOUTUBE_ANALYSIS_REPO")
WRAPPERS = ("scripts/youtube_analyze.py", "scripts/youtube_bundle_check.py")
PROBE_TIMEOUT_SECONDS = 15


class ResolutionError(RuntimeError):
    pass


def probe(repo: Path, command: list[str], label: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=repo,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise ResolutionError(f"{label} timed out in {repo}") from None
    except OSError as exc:
        raise ResolutionError(f"Cannot run {label} in {repo}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr.strip() or result.stdout.strip())[-1000:]
        raise ResolutionError(f"{label} failed (exit {result.returncode}) in {repo}: {detail}")
    return result


def validate_repo(repo: Path) -> Path:
    if not repo.is_dir():
        raise ResolutionError(f"Not a checkout directory: {repo}")
    for relative in (".venv/bin/python", *WRAPPERS):
        if not (repo / relative).is_file():
            raise ResolutionError(f"Not a runnable checkout: missing {relative} in {repo}")
    python = str(repo / ".venv/bin/python")
    token = "youtube-analysis-python-ready"
    result = probe(
        repo,
        [python, "-c", f"import sys; assert sys.version_info >= (3, 10); print({token!r})"],
        ".venv/bin/python interpreter",
    )
    if result.stdout.strip() != token:
        raise ResolutionError(f".venv/bin/python is not a usable Python interpreter in {repo}")
    for wrapper in WRAPPERS:
        result = probe(repo, [python, str(repo / wrapper), "--help"], f"{wrapper} --help")
        if not (result.stdout.strip() or result.stderr.strip()):
            raise ResolutionError(f"{wrapper} --help produced no help output in {repo}")
    return repo


def resolve_repo() -> Path:
    overrides = {}
    for name in OVERRIDES:
        if name in os.environ:
            value = os.environ[name]
            if not value.strip():
                raise ResolutionError(f"{name} is set but empty")
            overrides[name] = Path(value).expanduser().resolve()
    if len(set(overrides.values())) > 1:
        detail = ", ".join(f"{name}={str(path)!r}" for name, path in overrides.items())
        raise ResolutionError(f"Conflicting repo overrides: {detail}")
    if overrides:
        return validate_repo(next(iter(overrides.values())))

    cwd = Path.cwd().resolve()
    markers = (*WRAPPERS, "codex-skills/youtube-analysis/SKILL.md")
    for candidate in (cwd, *cwd.parents):
        if any((candidate / marker).is_file() for marker in markers):
            # A broken nearest checkout must not silently select a different tree.
            return validate_repo(candidate)
    raise ResolutionError("No youtube-analysis-tool checkout found in current directory or parents")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve YOUTUBE_ANALYSIS_REPO / MIRA_YOUTUBE_ANALYSIS_REPO, or the nearest "
            "checkout in cwd/parents. Check its Python and both wrapper --help commands, "
            "then print only the absolute repo path."
        )
    )
    parser.parse_args()
    try:
        repo = resolve_repo()
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"Repo resolution failed: {exc}. Set YOUTUBE_ANALYSIS_REPO or "
            "MIRA_YOUTUBE_ANALYSIS_REPO to a runnable checkout; if both are set, "
            "make them resolve to the same directory or unset one. No fallback was used.",
            file=sys.stderr,
        )
        return 1
    print(repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
