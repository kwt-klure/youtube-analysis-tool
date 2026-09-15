from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from .version import add_version_argument


IGNORED_TREE_NAMES = {".DS_Store", "__pycache__"}


def default_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "codex-skills" / "youtube-analysis"


def default_target_dir() -> Path:
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    return codex_home / "skills" / "youtube-analysis"


def _checked_path(path: Path) -> Path:
    path = path.expanduser().absolute()
    # Inspect before resolve(), including aliases hidden by a later '..'.
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError(f"Symlinks are not supported in skill paths: {component}")
    return path.resolve()


def _tree_entries(root: Path) -> list[Path]:
    root = _checked_path(root)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError(f"Skill tree must be a directory: {root}")
    entries: list[Path] = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"Symlinks are not supported in skill trees: {path}")
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"Unsupported skill tree entry: {path}")
            entries.append(path)
            if stat.S_ISDIR(mode):
                visit(path)

    visit(root)
    return entries


@contextmanager
def _open_regular(path: Path) -> Iterator[BinaryIO]:
    path = _checked_path(path)
    expected = path.lstat()
    if not stat.S_ISREG(expected.st_mode):
        raise ValueError(f"Expected a regular file, not a symlink or special file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        actual = os.fstat(stream.fileno())
        if not stat.S_ISREG(actual.st_mode) or (
            expected.st_dev, expected.st_ino
        ) != (actual.st_dev, actual.st_ino):
            raise RuntimeError(f"File changed while opening: {path}")
        yield stream


def _manifest(root: Path, *, complete: bool = False) -> dict[str, str]:
    root = _checked_path(root)
    manifest: dict[str, str] = {}
    # Validate the entire tree first, even entries excluded from the public hash.
    for path in _tree_entries(root):
        relative = path.relative_to(root)
        if not complete and any(part in IGNORED_TREE_NAMES for part in relative.parts):
            continue
        if path.is_dir():
            if complete:
                manifest[relative.as_posix() + "/"] = "directory"
            continue
        with _open_regular(path) as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest[relative.as_posix()] = digest.hexdigest()
    return manifest


def tree_manifest(root: Path) -> dict[str, str]:
    return _manifest(root)


def _target_state(root: Path) -> tuple[int, int, dict[str, str]] | None:
    root = _checked_path(root)
    if not root.exists():
        return None
    identity = root.stat()
    return identity.st_dev, identity.st_ino, _manifest(root, complete=True)


def _has_entrypoint(root: Path) -> bool:
    entrypoint = root / "SKILL.md"
    if not entrypoint.is_file():
        return False
    with _open_regular(entrypoint) as stream:
        return bool(stream.read().strip())


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
    source = _checked_path(source)
    target = _checked_path(target)
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
    result["in_sync"] = bool(result["source_exists"] and result["target_exists"]) and (
        _has_entrypoint(source)
        and _has_entrypoint(target)
        and result["source_hash"] is not None
        and result["target_hash"] is not None
        and result["source_hash"] == result["target_hash"]
        and not result["missing_files"]
        and not result["extra_files"]
        and not result["changed_files"]
    )
    return result


def next_backup_path(backup_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = backup_root / f"youtube-analysis-skill-{timestamp}"
    suffix = 2
    while os.path.lexists(candidate):
        candidate = backup_root / f"youtube-analysis-skill-{timestamp}-{suffix}"
        suffix += 1
    return candidate


def _private_directory(path: Path) -> None:
    _checked_path(path)
    if not path.parent.exists():
        _private_directory(path.parent)
    path.mkdir(mode=0o700, exist_ok=True)
    path.chmod(0o700)


def _copy_regular_file(source: str, target: str) -> str:
    with _open_regular(Path(source)) as reader, open(target, "xb") as writer:
        shutil.copyfileobj(reader, writer)
    shutil.copystat(source, target, follow_symlinks=False)
    return target


def _copy_tree(source: Path, target: Path, *, private: bool = False) -> None:
    _tree_entries(source)
    _checked_path(target)
    # Preserve newly discovered links instead of following them; readback rejects
    # them. File opens also recheck ancestors and never follow a leaf symlink.
    try:
        shutil.copytree(source, target, symlinks=True, copy_function=_copy_regular_file)
    finally:
        if private and target.is_dir() and not target.is_symlink():
            _checked_path(target).chmod(0o700)
    entries = _tree_entries(target)
    if private:
        for path in entries:
            if path.is_dir():
                path.chmod(0o700)


@contextmanager
def _install_lock(target: Path) -> Iterator[None]:
    lock_path = target.parent / f".{target.name}.install.lock"
    _checked_path(lock_path)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lock_path, flags, 0o600)
    with os.fdopen(fd, "r+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise ValueError(f"Install lock is not a regular file: {lock_path}")
        if os.name == "nt":
            import msvcrt

            if os.fstat(lock.fileno()).st_size == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    # Keep the lock inode: unlinking it would split waiting/new installers.


def install_skill(source: Path, target: Path, backup_root: Path) -> dict[str, Any]:
    """Serialize installers and retain moved inodes for late external writers.

    This is not a filesystem CAS or an adversarial-filesystem sandbox: directory
    ancestors must remain stable, and publication briefly leaves target absent.
    """
    source, target, backup_root = map(_checked_path, (source, target, backup_root))
    paths = (source, target, backup_root)
    for index, left in enumerate(paths):
        for right in paths[index + 1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"Skill source, target and backup paths must not overlap: {left}, {right}"
                )
    target.parent.mkdir(parents=True, exist_ok=True)
    with _install_lock(target):
        return _install_locked(source, target, backup_root)


def _install_locked(source: Path, target: Path, backup_root: Path) -> dict[str, Any]:
    before = check_skill_sync(source, target)
    if not before["source_exists"] or not _has_entrypoint(source):
        raise FileNotFoundError(f"Canonical skill tree has no nonempty SKILL.md: {source}")
    expected_target = _target_state(target)
    expected_source = _manifest(source, complete=True)
    expected_visible = {
        name: digest for name, digest in (expected_target[2] if expected_target else {}).items()
        if not name.endswith("/") and not any(part in IGNORED_TREE_NAMES for part in Path(name).parts)
    }
    if manifest_digest(expected_visible) != before["target_hash"]:
        raise RuntimeError("Runtime skill changed while recording its expected preimage.")

    backup_path: Path | None = None
    if expected_target is not None:
        _private_directory(backup_root)
        backup_path = next_backup_path(backup_root)
        _copy_tree(target, backup_path, private=True)
        if _manifest(backup_path, complete=True) != expected_target[2]:
            raise RuntimeError("Runtime skill backup hash did not match the preimage.")

    nonce = uuid.uuid4().hex
    staging_path = target.parent / f".{target.name}.staging-{nonce}"
    previous_container = target.parent / f".{target.name}.previous-{nonce}"
    previous_path = previous_container / "tree"

    moved_previous = False
    try:
        _copy_tree(source, staging_path)
        if (
            _manifest(staging_path, complete=True) != expected_source
            or manifest_digest(tree_manifest(staging_path)) != before["source_hash"]
        ):
            raise RuntimeError("Staged runtime skill hash did not match the canonical tree.")
        if expected_target is not None:
            _private_directory(previous_container)
        # The lock serializes installers; this check also catches external edits.
        if _target_state(target) != expected_target:
            raise RuntimeError(
                "Runtime skill changed since its expected preimage; replacement refused."
            )
        if expected_target is not None:
            target.rename(previous_path)
            moved_previous = True
            if _target_state(previous_path) != expected_target:
                raise RuntimeError(
                    "Runtime skill changed while moving its preimage; replacement refused."
                )
        if os.path.lexists(target):
            raise RuntimeError("Runtime skill target appeared during replacement; refusing to overwrite it.")
        staging_path.rename(target)
    except Exception as exc:
        if staging_path.exists():
            shutil.rmtree(staging_path, ignore_errors=True)
        if moved_previous and not os.path.lexists(target):
            try:
                previous_path.rename(target)
            except OSError as restore_error:
                raise RuntimeError(
                    f"Skill installation and restore failed; preserved moved preimage: {previous_path}"
                ) from restore_error
            moved_previous = False
        if moved_previous:
            raise RuntimeError(f"Skill installation failed; preserved moved preimage: {previous_path}") from exc
        raise
    finally:
        if not moved_previous and previous_container.exists():
            previous_container.rmdir()

    try:
        after = check_skill_sync(source, target)
    except Exception as exc:
        raise RuntimeError(
            f"Installed runtime skill could not be verified; preserved preimage: {backup_path}; "
            f"moved tree: {previous_path if moved_previous else None}"
        ) from exc
    if not after["in_sync"]:
        raise RuntimeError(
            "Installed runtime skill failed post-write verification; "
            f"preserved preimage: {backup_path}; moved tree: {previous_path if moved_previous else None}"
        )
    # A non-cooperating writer may still hold this inode/open files. Never delete
    # the moved tree, even after a matching readback; a final check is not a CAS.

    return {
        **after,
        "runtime_before_hash": before["target_hash"],
        "runtime_after_hash": after["target_hash"],
        "backup_path": str(backup_path) if backup_path is not None else None,
        "preserved_previous_path": str(previous_path) if moved_previous else None,
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
        if result.get("preserved_previous_path"):
            print(f"preserved previous tree: {result['preserved_previous_path']}")
    return 0 if result["in_sync"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
