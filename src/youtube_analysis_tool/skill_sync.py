from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .version import add_version_argument


IGNORED_TREE_NAMES = {".DS_Store", "__pycache__"}


def default_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "codex-skills" / "youtube-analysis"


def default_target_dir() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return codex_home / "skills" / "youtube-analysis"


def tree_manifest(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    manifest: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in IGNORED_TREE_NAMES for part in path.parts):
            continue
        relative_path = path.relative_to(root).as_posix()
        manifest[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def manifest_digest(manifest: dict[str, str]) -> str | None:
    if not manifest:
        return None
    digest = hashlib.sha256()
    for relative_path, file_digest in sorted(manifest.items()):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def check_skill_sync(source: Path, target: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    source_manifest = tree_manifest(source)
    target_manifest = tree_manifest(target)
    source_files = set(source_manifest)
    target_files = set(target_manifest)
    changed_files = sorted(
        path
        for path in source_files & target_files
        if source_manifest[path] != target_manifest[path]
    )
    result = {
        "source": str(source),
        "target": str(target),
        "source_exists": source.is_dir(),
        "target_exists": target.is_dir(),
        "source_hash": manifest_digest(source_manifest),
        "target_hash": manifest_digest(target_manifest),
        "missing_files": sorted(source_files - target_files),
        "extra_files": sorted(target_files - source_files),
        "changed_files": changed_files,
    }
    result["in_sync"] = bool(result["source_exists"]) and (
        result["source_hash"] == result["target_hash"]
        and not result["missing_files"]
        and not result["extra_files"]
        and not result["changed_files"]
    )
    return result


def next_backup_path(backup_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = backup_root / f"youtube-analysis-skill-{timestamp}"
    suffix = 2
    while candidate.exists():
        candidate = backup_root / f"youtube-analysis-skill-{timestamp}-{suffix}"
        suffix += 1
    return candidate


def install_skill(source: Path, target: Path, backup_root: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    backup_root = backup_root.expanduser().resolve()
    before = check_skill_sync(source, target)
    if not before["source_exists"] or before["source_hash"] is None:
        raise FileNotFoundError(f"Canonical skill tree is missing or empty: {source}")

    backup_path: Path | None = None
    if target.exists():
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_path = next_backup_path(backup_root)
        shutil.copytree(target, backup_path)
        if manifest_digest(tree_manifest(backup_path)) != before["target_hash"]:
            raise RuntimeError("Runtime skill backup hash did not match the preimage.")

    target.parent.mkdir(parents=True, exist_ok=True)
    nonce = uuid.uuid4().hex
    staging_path = target.parent / f".{target.name}.staging-{nonce}"
    previous_path = target.parent / f".{target.name}.previous-{nonce}"
    shutil.copytree(source, staging_path)
    staged_hash = manifest_digest(tree_manifest(staging_path))
    if staged_hash != before["source_hash"]:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise RuntimeError("Staged runtime skill hash did not match the canonical tree.")

    moved_previous = False
    try:
        if target.exists():
            target.rename(previous_path)
            moved_previous = True
        staging_path.rename(target)
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)
        if moved_previous and not target.exists() and previous_path.exists():
            previous_path.rename(target)
        raise

    after = check_skill_sync(source, target)
    if not after["in_sync"]:
        raise RuntimeError(
            "Installed runtime skill failed post-write verification; "
            f"preserved preimage: {backup_path}"
        )
    if previous_path.exists():
        shutil.rmtree(previous_path)

    return {
        **after,
        "runtime_before_hash": before["target_hash"],
        "runtime_after_hash": after["target_hash"],
        "backup_path": str(backup_path) if backup_path is not None else None,
        "installed": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or explicitly install the repo-owned youtube-analysis skill."
    )
    add_version_argument(parser)
    parser.add_argument("--source", type=Path, default=default_source_dir())
    parser.add_argument("--target", type=Path, default=default_target_dir())
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=default_target_dir().parents[1] / "backups",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check for drift without writing (default)",
    )
    mode.add_argument(
        "--install",
        action="store_true",
        help="Explicitly back up and replace the runtime skill tree",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.install:
        result = install_skill(args.source, args.target, args.backup_root)
    else:
        result = check_skill_sync(args.source, args.target)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        status = "in sync" if result["in_sync"] else "out of sync"
        print(f"youtube-analysis skill: {status}")
        if result.get("backup_path"):
            print(f"backup: {result['backup_path']}")
    return 0 if result["in_sync"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
