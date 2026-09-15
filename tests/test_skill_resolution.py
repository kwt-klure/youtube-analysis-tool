from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RESOLVER = ROOT / "codex-skills/youtube-analysis/scripts/resolve_repo.py"
OVERRIDES = ("YOUTUBE_ANALYSIS_REPO", "MIRA_YOUTUBE_ANALYSIS_REPO")
WRAPPERS = ("youtube_analyze.py", "youtube_bundle_check.py")


class SkillResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.cwd = self.root / "unrelated cwd"
        self.cwd.mkdir()
        self.resolver = self.root / "installed skill/scripts/resolve_repo.py"
        self.resolver.parent.mkdir(parents=True)
        shutil.copyfile(RESOLVER, self.resolver)

    def make_repo(self, name: str = "checkout") -> Path:
        repo = self.root / name
        (repo / ".venv/bin").mkdir(parents=True)
        (repo / ".venv/bin/python").symlink_to(sys.executable)
        (repo / "scripts").mkdir()
        for wrapper in WRAPPERS:
            (repo / "scripts" / wrapper).write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "assert sys.argv[1:] == ['--help']\n"
                "assert Path.cwd() == Path(__file__).resolve().parents[1]\n"
                "print('usage: ' + Path(__file__).name)\n",
                encoding="utf-8",
            )
        return repo

    def resolve(
        self, *, cwd: Path | None = None, overrides: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for name in OVERRIDES:
            env.pop(name, None)
        env.update(overrides or {})
        return subprocess.run(
            [sys.executable, str(self.resolver)],
            cwd=cwd or self.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def assert_resolved(self, result: subprocess.CompletedProcess[str], repo: Path) -> None:
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(f"{repo}\n", result.stdout)
        self.assertEqual("", result.stderr)

    def assert_rejected(self, result: subprocess.CompletedProcess[str], reason: str) -> None:
        self.assertNotEqual(0, result.returncode)
        self.assertEqual("", result.stdout)
        self.assertIn(reason, result.stderr)
        self.assertIn("YOUTUBE_ANALYSIS_REPO", result.stderr)
        self.assertIn("MIRA_YOUTUBE_ANALYSIS_REPO", result.stderr)

    def test_each_override_works_from_installed_skill_in_stale_cwd_with_spaces(self) -> None:
        repo = self.make_repo("valid repo with spaces")
        stale = self.make_repo("old worktree")
        (stale / ".venv/bin/python").unlink()
        for name in OVERRIDES:
            with self.subTest(name=name):
                self.assert_resolved(self.resolve(cwd=stale, overrides={name: str(repo)}), repo)

    def test_matching_overrides_accept_normalized_and_symlink_paths(self) -> None:
        repo = self.make_repo()
        alias = self.root / "repo alias"
        alias.symlink_to(repo, target_is_directory=True)
        self.assert_resolved(
            self.resolve(
                overrides={OVERRIDES[0]: str(repo), OVERRIDES[1]: str(alias / "scripts/..")}
            ),
            repo,
        )

    def test_relative_override_resolves_against_callers_cwd(self) -> None:
        repo = self.make_repo()
        self.assert_resolved(self.resolve(overrides={OVERRIDES[0]: "../checkout"}), repo)

    def test_conflicting_overrides_fail_before_running_checkout_code(self) -> None:
        first = self.make_repo("first")
        second = self.make_repo("second")
        (first / ".venv/bin/python").unlink()
        self.assert_rejected(
            self.resolve(overrides={OVERRIDES[0]: str(first), OVERRIDES[1]: str(second)}),
            "Conflicting",
        )

    def test_invalid_override_does_not_fall_back_to_valid_cwd(self) -> None:
        repo = self.make_repo()
        for name in OVERRIDES:
            with self.subTest(name=name):
                self.assert_rejected(
                    self.resolve(cwd=repo, overrides={name: str(self.root / "missing")}),
                    "directory",
                )

    def test_empty_override_does_not_fall_back(self) -> None:
        repo = self.make_repo()
        for name in OVERRIDES:
            for value in ("", "   "):
                with self.subTest(name=name, value=value):
                    self.assert_rejected(self.resolve(cwd=repo, overrides={name: value}), "empty")

    def test_current_and_parent_discovery_from_installed_skill(self) -> None:
        repo = self.make_repo()
        nested = repo / "nested working directory"
        nested.mkdir()
        for cwd in (repo, nested):
            with self.subTest(cwd=cwd):
                self.assert_resolved(self.resolve(cwd=cwd), repo)

    def test_missing_interpreter_rejects_stale_discovered_checkout(self) -> None:
        repo = self.make_repo()
        (repo / ".venv/bin/python").unlink()
        self.assert_rejected(self.resolve(cwd=repo), ".venv/bin/python")

    def test_missing_wrapper_or_checker_fails_explicit_and_discovered_resolution(self) -> None:
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                repo = self.make_repo(wrapper)
                (repo / "scripts" / wrapper).unlink()
                self.assert_rejected(self.resolve(cwd=repo), wrapper)
                self.assert_rejected(self.resolve(overrides={OVERRIDES[0]: str(repo)}), wrapper)

    def test_stale_nested_checkout_does_not_fall_back_to_valid_parent(self) -> None:
        parent = self.make_repo("parent")
        stale = self.make_repo("parent/old worktree")
        (stale / ".venv/bin/python").unlink()
        self.assert_rejected(self.resolve(cwd=stale), ".venv/bin/python")
        self.assert_resolved(self.resolve(cwd=stale, overrides={OVERRIDES[0]: str(parent)}), parent)

    def test_no_checkout_does_not_select_the_installed_skills_location(self) -> None:
        self.assert_rejected(self.resolve(), "No youtube-analysis-tool checkout")

    def test_nonexecutable_interpreter_is_rejected(self) -> None:
        repo = self.make_repo()
        python = repo / ".venv/bin/python"
        python.unlink()
        python.write_text("not executable\n", encoding="utf-8")
        self.assert_rejected(self.resolve(cwd=repo), "interpreter")

    def test_successful_noop_is_not_a_python_interpreter(self) -> None:
        repo = self.make_repo()
        python = repo / ".venv/bin/python"
        python.unlink()
        python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        python.chmod(0o700)
        self.assert_rejected(self.resolve(cwd=repo), "interpreter")

    def test_broken_help_is_rejected_for_each_wrapper(self) -> None:
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper):
                repo = self.make_repo(wrapper)
                (repo / "scripts" / wrapper).write_text(
                    "import sys\nprint('broken import', file=sys.stderr)\nsys.exit(7)\n",
                    encoding="utf-8",
                )
                result = self.resolve(cwd=repo)
                self.assert_rejected(result, f"{wrapper} --help")
                self.assertIn("broken import", result.stderr)

    def test_silent_help_is_not_usable(self) -> None:
        repo = self.make_repo()
        (repo / "scripts/youtube_bundle_check.py").write_text("pass\n", encoding="utf-8")
        self.assert_rejected(self.resolve(cwd=repo), "help output")

    def test_probe_timeout_stops_a_hanging_process(self) -> None:
        namespace = runpy.run_path(str(self.resolver))
        probe = namespace["probe"]
        with mock.patch.dict(probe.__globals__, PROBE_TIMEOUT_SECONDS=0.1):
            with self.assertRaisesRegex(namespace["ResolutionError"], "timed out"):
                probe(
                    self.cwd,
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    "hanging helper",
                )


if __name__ == "__main__":
    unittest.main()
