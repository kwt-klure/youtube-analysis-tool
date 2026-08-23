from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from youtube_analysis_tool.skill_sync import check_skill_sync, install_skill


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
            root = Path(tmpdir)
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
            root = Path(tmpdir)
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
            root = Path(tmpdir)
            source = root / "source"
            target = root / "runtime" / "youtube-analysis"
            self.write_skill(source, "canonical")

            result = install_skill(source, target, root / "backups")

        self.assertTrue(result["in_sync"])
        self.assertIsNone(result["backup_path"])


if __name__ == "__main__":
    unittest.main()
