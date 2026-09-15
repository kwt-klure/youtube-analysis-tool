from __future__ import annotations

import contextlib
import io
import json
import multiprocessing
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from youtube_analysis_tool import skill_sync
from youtube_analysis_tool.skill_sync import check_skill_sync, install_skill


def run_paused_install(source, target, backups, ready, staged, release, results):
    copytree = shutil.copytree

    def pause_after_staging(src, dst, *args, **kwargs):
        result = copytree(src, dst, *args, **kwargs)
        if Path(src) == source:
            staged.set()
            if not release.wait(10):
                raise RuntimeError("Test staging release timed out")
        return result

    ready.set()
    try:
        with mock.patch.object(skill_sync.shutil, "copytree", side_effect=pause_after_staging):
            results.put(install_skill(source, target, backups))
    except Exception as exc:
        results.put({"error": str(exc)})


class SkillSyncTests(unittest.TestCase):
    def write_skill(self, root: Path, text: str, *, extra: bool = False) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "SKILL.md").write_text(text, encoding="utf-8")
        references = root / "references"
        references.mkdir()
        (references / "execution-modes.md").write_text(f"modes: {text}", encoding="utf-8")
        if extra:
            (root / "stale.md").write_text("stale", encoding="utf-8")

    def test_check_reports_changed_and_extra_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source = root / "source"
            target = root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "runtime", extra=True)

            result = check_skill_sync(source, target)

        self.assertFalse(result["in_sync"])
        self.assertEqual(["SKILL.md", "references/execution-modes.md"], result["changed_files"])
        self.assertEqual(["stale.md"], result["extra_files"])

    def test_install_backs_up_preimage_and_verifies_postimage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source = root / "source"
            target = root / "runtime" / "youtube-analysis"
            backup_root = root / "backups"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime", extra=True)

            result = install_skill(source, target, backup_root)
            backup_path = Path(str(result["backup_path"]))

            checked = check_skill_sync(source, target)
            backup_text = (backup_path / "SKILL.md").read_text(encoding="utf-8")
            target_text = (target / "SKILL.md").read_text(encoding="utf-8")

        self.assertTrue(result["installed"])
        self.assertTrue(result["in_sync"])
        self.assertTrue(checked["in_sync"])
        self.assertNotEqual(result["runtime_before_hash"], result["runtime_after_hash"])
        self.assertEqual("old runtime", backup_text)
        self.assertEqual("canonical", target_text)

    def test_install_to_missing_target_needs_no_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source = root / "source"
            target = root / "runtime" / "youtube-analysis"
            self.write_skill(source, "canonical")

            result = install_skill(source, target, root / "backups")

        self.assertTrue(result["in_sync"])
        self.assertIsNone(result["backup_path"])

    def test_incomplete_trees_fail_sync_and_cli_check(self) -> None:
        for source_kind, target_kind in (
            ("empty", "missing"), ("empty", "empty"), ("missing", "missing"),
            ("no-entrypoint", "no-entrypoint"), ("blank", "blank"),
            ("whitespace", "whitespace"), ("valid", "missing"), ("missing", "valid"),
            ("valid", "empty"), ("valid", "blank"), ("valid", "no-entrypoint"),
        ):
            with self.subTest(source=source_kind, target=target_kind), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir).resolve()
                source, target = root / "source", root / "target"
                for tree, kind in ((source, source_kind), (target, target_kind)):
                    if kind != "missing":
                        tree.mkdir()
                    if kind in {"no-entrypoint", "blank", "whitespace", "valid"}:
                        (tree / "reference.md").write_text("reference")
                    if kind in {"blank", "whitespace", "valid"}:
                        (tree / "SKILL.md").write_text({"blank": "", "whitespace": " \n", "valid": "skill"}[kind])
                self.assertFalse(check_skill_sync(source, target)["in_sync"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    code = skill_sync.main(["--source", str(source), "--target", str(target), "--check", "--json"])
                self.assertEqual(1, code)
                self.assertFalse(json.loads(output.getvalue())["in_sync"])
                self.assertFalse(list(root.glob(".*install*")))

    def test_invalid_source_is_not_installed(self) -> None:
        for text in (None, "", " \n"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir).resolve()
                source, target = root / "source", root / "target"
                source.mkdir()
                (source / "reference.md").write_text("not an entrypoint")
                if text is not None:
                    (source / "SKILL.md").write_text(text)
                self.write_skill(target, "old runtime")
                with self.assertRaises((ValueError, FileNotFoundError)):
                    install_skill(source, target, root / "backups")
                self.assertEqual("old runtime", (target / "SKILL.md").read_text())

    def test_install_to_empty_target_preserves_empty_preimage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            target.mkdir()
            result = install_skill(source, target, root / "backups")
            self.assertTrue(result["in_sync"])
            self.assertIsNone(result["runtime_before_hash"])
            for key in ("backup_path", "preserved_previous_path"):
                preserved = Path(result[key])
                self.assertTrue(preserved.is_dir())
                self.assertEqual([], list(preserved.iterdir()))

    def test_source_change_after_initial_check_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            check = check_skill_sync

            def mutate_after_check(src, dst):
                result = check(src, dst)
                (source / "SKILL.md").write_text("unexpected canonical edit")
                return result

            with mock.patch.object(skill_sync, "check_skill_sync", side_effect=mutate_after_check):
                with self.assertRaisesRegex(RuntimeError, "changed|canonical"):
                    install_skill(source, target, root / "backups")
            self.assertEqual("old runtime", (target / "SKILL.md").read_text())

    def test_staging_hook_drift_aborts_without_losing_changed_target(self) -> None:
        for drift in ("entrypoint", "ignored-file", "empty-directory", "appeared"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir).resolve()
                source, target = root / "source", root / "target"
                self.write_skill(source, "canonical")
                if drift != "appeared":
                    self.write_skill(target, "old runtime")
                copytree = shutil.copytree

                def mutate_after_staging(src, dst, *args, **kwargs):
                    copied = copytree(src, dst, *args, **kwargs)
                    if Path(src) == source:
                        if drift == "appeared":
                            self.write_skill(target, "concurrent")
                        elif drift == "entrypoint":
                            (target / "SKILL.md").write_text("concurrent")
                        elif drift == "ignored-file":
                            (target / "__pycache__").mkdir()
                            (target / "__pycache__" / "new.pyc").write_text("concurrent")
                        else:
                            (target / "concurrent-empty-directory").mkdir()
                    return copied

                with mock.patch.object(skill_sync.shutil, "copytree", side_effect=mutate_after_staging):
                    with self.assertRaisesRegex(RuntimeError, "changed|drift|preimage"):
                        install_skill(source, target, root / "backups")
                if drift in {"appeared", "entrypoint"}:
                    self.assertEqual("concurrent", (target / "SKILL.md").read_text())
                elif drift == "ignored-file":
                    self.assertEqual("concurrent", (target / "__pycache__" / "new.pyc").read_text())
                else:
                    self.assertTrue((target / "concurrent-empty-directory").is_dir())
                self.assertFalse(list(root.glob(".target.staging-*")))

    def test_change_between_final_check_and_rename_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            rename = Path.rename

            def mutate_before_move(path, destination):
                if path == target:
                    (path / "SKILL.md").write_text("last-moment edit")
                return rename(path, destination)

            with mock.patch.object(Path, "rename", new=mutate_before_move):
                with self.assertRaisesRegex(RuntimeError, "changed|drift|preimage"):
                    install_skill(source, target, root / "backups")
            self.assertEqual("last-moment edit", (target / "SKILL.md").read_text())

    def test_late_writer_to_moved_tree_is_never_cleaned_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            rename = Path.rename
            previous = []

            def mutate_after_publish(path, destination):
                result = rename(path, destination)
                if path == target:
                    previous.append(Path(destination))
                elif Path(destination) == target and previous:
                    (previous[0] / "SKILL.md").write_text("late writer")
                return result

            with mock.patch.object(Path, "rename", new=mutate_after_publish):
                result = install_skill(source, target, root / "backups")
            self.assertTrue(result["in_sync"])
            self.assertEqual("late writer", (previous[0] / "SKILL.md").read_text())
            self.assertEqual(str(previous[0]), result["preserved_previous_path"])
            self.assertEqual(0o700, stat.S_IMODE(previous[0].parent.stat().st_mode))

    def test_publish_failure_restores_original_and_releases_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            rename = Path.rename

            def fail_publish(path, destination):
                if ".staging-" in path.name:
                    raise OSError("injected publish failure")
                return rename(path, destination)

            with mock.patch.object(Path, "rename", new=fail_publish):
                with self.assertRaisesRegex(OSError, "injected publish failure"):
                    install_skill(source, target, root / "backups")
            self.assertEqual("old runtime", (target / "SKILL.md").read_text())
            self.assertFalse(list(root.glob(".target.staging-*")))
            self.assertTrue(install_skill(source, target, root / "backups")["in_sync"])

    def test_postwrite_drift_is_not_rolled_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            rename = Path.rename
            previous = []

            def mutate_after_publish(path, destination):
                result = rename(path, destination)
                if path == target:
                    previous.append(Path(destination))
                elif Path(destination) == target:
                    (target / "SKILL.md").write_text("postwrite edit")
                return result

            with mock.patch.object(Path, "rename", new=mutate_after_publish):
                with self.assertRaisesRegex(RuntimeError, "post-write verification"):
                    install_skill(source, target, root / "backups")
            self.assertEqual("postwrite edit", (target / "SKILL.md").read_text())
            self.assertEqual("old runtime", (previous[0] / "SKILL.md").read_text())

    def test_target_recreated_during_move_is_not_overwritten_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            rename = Path.rename
            previous = []

            def recreate_after_move(path, destination):
                result = rename(path, destination)
                if path == target:
                    previous.append(Path(destination))
                    self.write_skill(target, "new external tree")
                return result

            with mock.patch.object(Path, "rename", new=recreate_after_move):
                with self.assertRaisesRegex(RuntimeError, "preserved moved preimage"):
                    install_skill(source, target, root / "backups")
            self.assertEqual("new external tree", (target / "SKILL.md").read_text())
            self.assertEqual("old runtime", (previous[0] / "SKILL.md").read_text())

    def test_root_swapped_to_symlink_at_copy_hook_does_not_copy_external_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target, backups = root / "source", root / "target", root / "backups"
            self.write_skill(source, "canonical")
            self.write_skill(target, "old runtime")
            external = root / "external"
            external.mkdir()
            (external / "secret").write_text("external payload")
            copytree = shutil.copytree

            def swap_root_before_copy(src, dst, *args, **kwargs):
                if Path(src) == target:
                    target.rename(root / "saved-runtime")
                    target.symlink_to(external, target_is_directory=True)
                return copytree(src, dst, *args, **kwargs)

            with mock.patch.object(skill_sync.shutil, "copytree", side_effect=swap_root_before_copy):
                with self.assertRaises((ValueError, RuntimeError, shutil.Error)):
                    install_skill(source, target, backups)
            self.assertFalse(list(backups.rglob("secret")))
            self.assertEqual("external payload", (external / "secret").read_text())

    def test_symlink_hidden_by_dotdot_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            source, target = root / "source", root / "target"
            self.write_skill(source, "canonical")
            (root / "alias").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                install_skill(root / "alias" / ".." / "source", target, root / "backups")

    def test_symlinks_fail_before_any_tree_is_copied(self) -> None:
        for location in ("source-file", "target-directory", "ignored", "dangling", "source-root", "target-root", "backup-root", "source-parent", "target-parent", "backup-parent"):
            with self.subTest(location=location), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir).resolve()
                source, target, backups = root / "source", root / "target", root / "backups"
                self.write_skill(source, "canonical")
                self.write_skill(target, "old runtime")
                external = root / "external"
                external.mkdir()
                (external / "secret").write_text("never copy this")
                if location == "source-file":
                    (source / "link").symlink_to(external / "secret")
                elif location in {"target-directory", "ignored", "dangling"}:
                    (target / ("__pycache__" if location == "ignored" else "link")).symlink_to(external / "missing" if location == "dangling" else external, target_is_directory=True)
                elif location.endswith("-root"):
                    if location == "backup-root":
                        backups.symlink_to(external, target_is_directory=True)
                    else:
                        tree = source if location == "source-root" else target
                        real_tree = root / "real-tree"
                        tree.rename(real_tree)
                        tree.symlink_to(real_tree, target_is_directory=True)
                else:
                    alias = root / "alias"
                    alias.symlink_to(root, target_is_directory=True)
                    if location == "source-parent":
                        source = alias / "source"
                    elif location == "target-parent":
                        target = alias / "target"
                    else:
                        backups = alias / "backups"
                with mock.patch.object(skill_sync.shutil, "copytree", wraps=shutil.copytree) as copier:
                    with self.assertRaisesRegex(ValueError, "[Ss]ymlink"):
                        install_skill(source, target, backups)
                copier.assert_not_called()
                self.assertEqual(["secret"], sorted(p.name for p in external.iterdir()))

    def test_overlapping_paths_are_rejected_before_copy(self) -> None:
        for relationship in ("same", "source-under-target", "target-under-source", "backup-under-target", "backup-under-source", "backup-contains-target"):
            with self.subTest(relationship=relationship), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir).resolve()
                source, target, backups = root / "source", root / "target", root / "backups"
                if relationship == "same":
                    target = source
                elif relationship == "source-under-target":
                    source = target / "source"
                elif relationship == "target-under-source":
                    target = source / "target"
                elif relationship == "backup-under-target":
                    backups = target / "backups"
                elif relationship == "backup-under-source":
                    backups = source / "backups"
                else:
                    backups = root
                self.write_skill(source, "canonical")
                if target != source:
                    self.write_skill(target, "old runtime")
                with mock.patch.object(skill_sync.shutil, "copytree", wraps=shutil.copytree) as copier:
                    with self.assertRaisesRegex(ValueError, "overlap"):
                        install_skill(source, target, backups)
                copier.assert_not_called()

    def test_backup_directories_are_private_even_with_permissive_umask(self) -> None:
        for preexisting in (False, True):
            with self.subTest(preexisting=preexisting), tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir).resolve()
                source, target, backups = root / "source", root / "target", root / "private" / "backups"
                self.write_skill(source, "canonical")
                self.write_skill(target, "old runtime")
                if preexisting:
                    backups.mkdir(parents=True)
                    backups.chmod(0o755)
                previous_umask = os.umask(0o022)
                try:
                    result = install_skill(source, target, backups)
                finally:
                    os.umask(previous_umask)
                backup_path = Path(result["backup_path"])
                for directory in (backups, backup_path, backup_path / "references"):
                    self.assertEqual(0o700, stat.S_IMODE(directory.stat().st_mode), directory)

    def test_concurrent_process_installs_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            target, backups = root / "target", root / "backups"
            sources = [root / "source-1", root / "source-2"]
            for index, source in enumerate(sources):
                self.write_skill(source, f"canonical-{index}")
            self.write_skill(target, "old runtime")
            ctx = multiprocessing.get_context("spawn")
            ready = [ctx.Event(), ctx.Event()]
            staged = [ctx.Event(), ctx.Event()]
            release = [ctx.Event(), ctx.Event()]
            results = [ctx.Queue(), ctx.Queue()]
            processes = [ctx.Process(target=run_paused_install, args=(sources[i], target, backups, ready[i], staged[i], release[i], results[i])) for i in range(2)]
            try:
                processes[0].start()
                self.assertTrue(staged[0].wait(10))
                processes[1].start()
                self.assertTrue(ready[1].wait(10))
                self.assertFalse(staged[1].wait(0.3), "second installer entered staging while first held the lock")
                release[0].set()
                self.assertTrue(staged[1].wait(10))
                release[1].set()
                returned = [queue.get(timeout=10) for queue in results]
                for result in returned:
                    self.assertNotIn("error", result)
                    self.assertTrue(result["in_sync"])
                self.assertEqual("canonical-0", (Path(returned[1]["backup_path"]) / "SKILL.md").read_text())
                self.assertEqual("canonical-1", (target / "SKILL.md").read_text())
            finally:
                for event in release:
                    event.set()
                for process in processes:
                    if process.pid is not None:
                        process.join(10)
                        if process.is_alive():
                            process.terminate()
                            process.join(10)
                for queue in results:
                    queue.close()


if __name__ == "__main__":
    unittest.main()
