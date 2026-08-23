from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from youtube_analysis_tool.bundle_check import check_bundle, main


class BundleCheckTests(unittest.TestCase):
    def write_bundle(self, root: Path, payload: object) -> Path:
        path = root / "output.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_completed_bundle_without_errors_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_bundle(
                Path(tmpdir),
                {
                    "processing": {"run_status": "completed"},
                    "transcript": {"source": "subtitle_manual"},
                    "visual_sampling": {"status": "skipped"},
                    "errors": [],
                },
            )

            result = check_bundle(path)

        self.assertTrue(result["valid"])
        self.assertTrue(result["parseable"])
        self.assertEqual("completed", result["run_status"])
        self.assertEqual("subtitle_manual", result["transcript_source"])
        self.assertEqual("skipped", result["visual_status"])

    def test_missing_run_status_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_bundle(Path(tmpdir), {"processing": {}, "errors": []})

            result = check_bundle(path)

        self.assertFalse(result["valid"])
        self.assertIn("run_status_missing", result["failure_reasons"])

    def test_failed_bundle_with_error_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_bundle(
                Path(tmpdir),
                {
                    "processing": {"run_status": "failed"},
                    "errors": [{"stage": "analyze", "message": "boom"}],
                },
            )

            result = check_bundle(path)

        self.assertFalse(result["valid"])
        self.assertEqual(1, result["fatal_error_count"])
        self.assertIn("run_status_failed", result["failure_reasons"])
        self.assertIn("fatal_errors_present", result["failure_reasons"])

    def test_warning_entry_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_bundle(
                Path(tmpdir),
                {
                    "processing": {"run_status": "completed"},
                    "errors": [{"severity": "warning", "message": "advisory"}],
                },
            )

            result = check_bundle(path)

        self.assertTrue(result["valid"])
        self.assertEqual(1, result["error_count"])
        self.assertEqual(0, result["fatal_error_count"])

    def test_malformed_json_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.json"
            path.write_text("{not-json", encoding="utf-8")

            result = check_bundle(path)

        self.assertFalse(result["parseable"])
        self.assertEqual(["bundle_not_parseable"], result["failure_reasons"])

    def test_parseable_non_object_is_invalid_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_bundle(Path(tmpdir), [])

            result = check_bundle(path)

        self.assertTrue(result["parseable"])
        self.assertFalse(result["valid"])
        self.assertEqual(["bundle_not_object"], result["failure_reasons"])

    def test_json_cli_returns_machine_readable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self.write_bundle(Path(tmpdir), {"processing": {}, "errors": []})
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                exit_code = main([str(path), "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(1, exit_code)
        self.assertFalse(payload["valid"])
        self.assertIn("run_status_missing", payload["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
