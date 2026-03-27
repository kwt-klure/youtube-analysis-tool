from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from youtube_analysis_tool import batch, library, pipeline
from youtube_analysis_tool.artifacts import write_json


class BatchQueueTests(unittest.TestCase):
    def test_read_source_list_ignores_comments_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_list = Path(tmpdir) / "sources.txt"
            source_list.write_text(
                "\n# comment\nhttps://youtu.be/abc123\n\n/tmp/demo.mp4\n",
                encoding="utf-8",
            )

            sources = batch.read_source_list(source_list)

        self.assertEqual(["https://youtu.be/abc123", "/tmp/demo.mp4"], sources)

    def test_batch_skips_existing_youtube_output_without_running_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "youtube"
            batch_root = root / "batches"
            existing = output_root / "my-great-talk-abc-123"
            existing.mkdir(parents=True, exist_ok=True)
            (existing / "output.json").write_text("{}", encoding="utf-8")
            source_list = root / "sources.txt"
            source_list.write_text("https://www.youtube.com/watch?v=AbC_123\n", encoding="utf-8")
            stdout = io.StringIO()

            with mock.patch.object(batch.constants, "BATCH_REPORT_ROOT", batch_root), mock.patch(
                "youtube_analysis_tool.batch.analyze_source"
            ) as analyze_mock, mock.patch("sys.stdout", stdout):
                exit_code = batch.main(
                    ["--source-list", str(source_list), "--root", str(output_root)]
                )
                report_path = Path(stdout.getvalue().strip())
                payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        analyze_mock.assert_not_called()
        self.assertEqual({"queued": 1, "completed": 0, "skipped": 1, "failed": 0}, payload["totals"])
        self.assertEqual("skipped", payload["items"][0]["status"])
        self.assertEqual(str(existing), payload["items"][0]["output_path"])

    def test_batch_continues_after_failure_and_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "youtube"
            batch_root = root / "batches"
            source_list = root / "sources.txt"
            source_list.write_text("/tmp/one.mp4\n/tmp/two.mp4\n", encoding="utf-8")
            stdout = io.StringIO()

            def fake_analyze(source: str, **kwargs):
                del kwargs
                if source.endswith("one.mp4"):
                    completed = output_root / "one"
                    completed.mkdir(parents=True, exist_ok=True)
                    return completed
                raise RuntimeError("boom")

            with mock.patch.object(batch.constants, "BATCH_REPORT_ROOT", batch_root), mock.patch(
                "youtube_analysis_tool.batch.analyze_source",
                side_effect=fake_analyze,
            ), mock.patch("sys.stdout", stdout):
                exit_code = batch.main(
                    ["--source-list", str(source_list), "--root", str(output_root)]
                )
                report_path = Path(stdout.getvalue().strip())
                payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(1, exit_code)
        self.assertEqual({"queued": 2, "completed": 1, "skipped": 0, "failed": 1}, payload["totals"])
        self.assertEqual("completed", payload["items"][0]["status"])
        self.assertEqual("failed", payload["items"][1]["status"])
        self.assertEqual("boom", payload["items"][1]["error"])

    def test_batch_passes_custom_root_into_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "custom-youtube"
            batch_root = root / "batches"
            source_list = root / "sources.txt"
            source_list.write_text("/tmp/demo.mp4\n", encoding="utf-8")
            stdout = io.StringIO()

            with mock.patch.object(batch.constants, "BATCH_REPORT_ROOT", batch_root), mock.patch(
                "youtube_analysis_tool.batch.analyze_source",
                return_value=output_root / "demo",
            ) as analyze_mock, mock.patch("sys.stdout", stdout):
                exit_code = batch.main(
                    ["--source-list", str(source_list), "--root", str(output_root)]
                )

        self.assertEqual(0, exit_code)
        _, kwargs = analyze_mock.call_args
        self.assertEqual(output_root, kwargs["output_root_base"])


class LibraryIndexTests(unittest.TestCase):
    def test_library_derives_high_trust_for_manual_subtitles(self) -> None:
        payload = {
            "source": {"input": "https://youtu.be/demo"},
            "metadata": {"title": "Demo", "channel": "Channel"},
            "transcript": {"source": "subtitle_manual", "language": "zh-hant"},
            "processing": {"visuals_mode": "off"},
            "errors": [],
        }
        record = library.derive_record(Path("/tmp/output.json"), payload)

        self.assertEqual("high", record["trust"])
        self.assertIsNone(record["read_mode"])

    def test_library_filters_and_grep_match_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first" / "output.json"
            second = root / "second" / "output.json"
            write_json(
                first,
                {
                    "source": {"input": "https://youtu.be/a"},
                    "metadata": {
                        "title": "Prompt Engineering Deep Dive",
                        "channel": "Anthropic Lab",
                        "chapters": [{"title": "Introduction"}],
                    },
                    "transcript": {
                        "source": "whisper",
                        "language": "zh",
                        "full_text": "Prompt engineering basics and examples.",
                        "interpretation": {
                            "trust": "medium",
                            "read_mode": "verify_entities",
                            "caution": ["names", "numbers", "exact_wording"],
                        },
                    },
                    "processing": {"visuals_mode": "off"},
                    "errors": [],
                },
            )
            write_json(
                second,
                {
                    "source": {"input": "https://youtu.be/b"},
                    "metadata": {"title": "Cooking Demo", "channel": "Kitchen"},
                    "transcript": {
                        "source": "subtitle_manual",
                        "language": "zh-hant",
                        "full_text": "Chicken soup in ten minutes.",
                    },
                    "processing": {"visuals_mode": "off"},
                    "errors": [],
                },
            )
            stdout = io.StringIO()

            with mock.patch("sys.stdout", stdout):
                exit_code = library.main(
                    [
                        "--root",
                        str(root),
                        "--trust",
                        "medium",
                        "--grep",
                        "prompt",
                    ]
                )

        self.assertEqual(0, exit_code)
        rows = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(1, len(rows))
        self.assertEqual("whisper", rows[0]["transcript_source"])
        self.assertEqual(["title", "full_text"], rows[0]["match_fields"])

    def test_library_skips_unreadable_json_without_failing_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            broken = root / "broken" / "output.json"
            broken.parent.mkdir(parents=True, exist_ok=True)
            broken.write_text("{bad json", encoding="utf-8")
            good = root / "good" / "output.json"
            write_json(
                good,
                {
                    "source": {"input": "https://youtu.be/demo"},
                    "metadata": {"title": "Demo", "channel": "Channel"},
                    "transcript": {"source": "subtitle_manual", "language": "en"},
                    "processing": {"visuals_mode": "off"},
                    "errors": [],
                },
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with mock.patch("sys.stdout", stdout), mock.patch("sys.stderr", stderr):
                exit_code = library.main(["--root", str(root)])

        self.assertEqual(0, exit_code)
        rows = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(1, len(rows))
        self.assertIn("Skipping unreadable bundle", stderr.getvalue())


class SharedHelperTests(unittest.TestCase):
    def test_pipeline_extracts_youtube_ids_from_common_url_shapes(self) -> None:
        self.assertEqual("abc123", pipeline.youtube_video_id_from_source("https://youtu.be/abc123"))
        self.assertEqual(
            "abc123",
            pipeline.youtube_video_id_from_source("https://www.youtube.com/watch?v=abc123&feature=share"),
        )
        self.assertEqual(
            "abc123",
            pipeline.youtube_video_id_from_source("https://www.youtube.com/shorts/abc123?si=demo"),
        )

    def test_default_output_dir_supports_custom_root(self) -> None:
        output = pipeline.default_output_dir_for_source(
            "https://youtu.be/abc123",
            "abc123",
            "My Great Talk",
            output_root=Path("/tmp/custom"),
        )
        self.assertEqual(Path("/tmp/custom/my-great-talk-abc123"), output)
