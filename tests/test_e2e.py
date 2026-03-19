from __future__ import annotations

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

from youtube_analysis_tool import pipeline, triage as triage_module
from youtube_analysis_tool.artifacts import write_json


def detailed_matrix() -> list[list[float]]:
    return [
        [0.0, 255.0, 0.0, 255.0, 0.0],
        [255.0, 0.0, 255.0, 0.0, 255.0],
        [0.0, 255.0, 0.0, 255.0, 0.0],
        [255.0, 0.0, 255.0, 0.0, 255.0],
        [0.0, 255.0, 0.0, 255.0, 0.0],
    ]


def alternate_matrix() -> list[list[float]]:
    return [
        [0.0, 255.0, 0.0, 255.0, 0.0],
        [255.0, 0.0, 255.0, 0.0, 255.0],
        [0.0, 255.0, 255.0, 255.0, 0.0],
        [255.0, 0.0, 255.0, 0.0, 255.0],
        [0.0, 255.0, 0.0, 255.0, 0.0],
    ]


class EndToEndPipelineTests(unittest.TestCase):
    def make_stubbed_run(
        self,
        *,
        keyframes: list[dict[str, object]],
        ocr_rows: list[dict[str, object]],
        transcript_segments: list[dict[str, object]] | None = None,
        matrix_by_filename: dict[str, list[list[float]]] | None = None,
        phash_by_filename: dict[str, str] | None = None,
        gpt_report_title: str = "影片分析報告",
    ) -> tuple[Path, Path]:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        source_path = root / "input.mp4"
        source_path.write_bytes(b"video")
        out_dir = root / "output"
        matrix_by_filename = matrix_by_filename or {}
        original_compute_phash = triage_module.compute_phash
        self.gpt_call_count = 0

        def fake_materialize_local_input(source_file: Path, paths: pipeline.AnalysisPaths):
            metadata = {"format": {"duration": "120"}, "streams": [{"codec_type": "video"}]}
            write_json(paths.metadata_path, metadata)
            return metadata, source_file

        def fake_extract_audio(_: Path, audio_dir: Path) -> Path:
            audio_path = audio_dir / "source.wav"
            audio_path.write_bytes(b"audio")
            return audio_path

        def fake_transcript_strategy_auto(_: Path, paths: pipeline.AnalysisPaths) -> dict[str, object]:
            transcript = pipeline.transcript_from_segments(
                transcript_segments or [{"start": 0.0, "end": 4.0, "text": "Demo transcript"}],
                source="subtitle",
                language="en",
            )
            pipeline.write_transcript(paths, transcript)
            return transcript

        def fake_create_keyframes(
            video_path: Path | None,
            metadata: dict[str, object],
            paths: pipeline.AnalysisPaths,
            *,
            mode: str,
            interval_seconds: int,
            threshold: float,
        ) -> list[dict[str, object]]:
            del video_path, metadata, mode, interval_seconds, threshold
            rows = []
            for row in keyframes:
                frame_path = paths.keyframes_dir / str(row["filename"])
                frame_path.write_bytes(str(row["filename"]).encode("utf-8"))
                rows.append(dict(row))
            pipeline.write_keyframe_index(paths.keyframe_index_path, rows)
            return rows

        def fake_optional_ocr(
            paths: pipeline.AnalysisPaths,
            _: list[dict[str, object]],
        ) -> tuple[list[dict[str, object]], str | None]:
            rows = [dict(row) for row in ocr_rows]
            pipeline.write_ocr_index(paths.ocr_index_path, rows)
            return rows, None

        def fake_read_grayscale_image(path: Path):
            return matrix_by_filename.get(path.name, detailed_matrix())

        def fake_compute_phash(matrix, image_path: Path):
            if phash_by_filename and image_path.name in phash_by_filename:
                return phash_by_filename[image_path.name]
            return original_compute_phash(matrix, image_path)

        def fake_gpt_analyze_segments(
            output_root: Path,
            manifest_entries: list[dict[str, object]],
            *,
            model: str,
            transcript: dict[str, object] | None,
            metadata: dict[str, object],
            report_language: str,
            client=None,
        ):
            del model, transcript, metadata, report_language, client
            self.gpt_call_count += 1
            analyses = [
                {"source_segment_id": entry["segment_id"], "segment_summary": "summary"}
                for entry in manifest_entries
                if entry.get("approved_for_gpt")
            ]
            report = {
                "title": gpt_report_title,
                "executive_summary": "summary",
                "main_sections": [],
                "key_visuals": [],
                "speaker_points": [],
                "open_questions": [],
                "source_segment_ids": [entry["segment_id"] for entry in manifest_entries if entry.get("approved_for_gpt")],
            }
            write_json(output_root / "gpt" / "analyses.json", {"segment_analyses": analyses, "final_report": report})
            return analyses, report

        patches = [
            mock.patch.object(pipeline, "materialize_local_input", side_effect=fake_materialize_local_input),
            mock.patch.object(pipeline, "extract_audio", side_effect=fake_extract_audio),
            mock.patch.object(pipeline, "transcript_strategy_auto", side_effect=fake_transcript_strategy_auto),
            mock.patch.object(pipeline, "create_keyframes", side_effect=fake_create_keyframes),
            mock.patch.object(pipeline, "try_run_optional_ocr", side_effect=fake_optional_ocr),
            mock.patch("youtube_analysis_tool.triage.read_grayscale_image", side_effect=fake_read_grayscale_image),
            mock.patch("youtube_analysis_tool.triage.compute_phash", side_effect=fake_compute_phash),
            mock.patch("youtube_analysis_tool.gpt.analyze_segments", side_effect=fake_gpt_analyze_segments),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

        return source_path, out_dir

    def test_slide_heavy_input_collapses_into_single_segment(self) -> None:
        source_path, out_dir = self.make_stubbed_run(
            keyframes=[
                {"kind": "scene", "filename": "slide-1.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00"},
                {"kind": "scene", "filename": "slide-2.jpg", "timestamp_seconds": 5.0, "timestamp_hms": "00:00:05"},
                {"kind": "scene", "filename": "slide-3.jpg", "timestamp_seconds": 9.0, "timestamp_hms": "00:00:09"},
            ],
            ocr_rows=[
                {"filename": "slide-1.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": "Revenue revenue revenue revenue revenue revenue"},
                {"filename": "slide-2.jpg", "timestamp_seconds": 5.0, "timestamp_hms": "00:00:05", "text": "Revenue revenue revenue revenue revenue revenue"},
                {"filename": "slide-3.jpg", "timestamp_seconds": 9.0, "timestamp_hms": "00:00:09", "text": "Revenue revenue revenue revenue revenue revenue"},
            ],
        )

        output_root = pipeline.analyze_source(
            str(source_path),
            out_dir=out_dir,
            gpt_mode="off",
            artifacts_mode="debug",
        )

        segments = json.loads((output_root / "triage" / "segments.json").read_text(encoding="utf-8"))["segments"]
        self.assertEqual(1, len(segments))
        self.assertEqual("slides", segments[0]["heuristic_label"])
        analysis = json.loads((output_root / "output.json").read_text(encoding="utf-8"))
        self.assertEqual("Revenue revenue revenue revenue revenue revenue", analysis["visuals"]["slides"][0]["ocr_summary"])
        self.assertNotIn("gpt", analysis)
        self.assertEqual("subtitle", analysis["transcript"]["source"])
        self.assertEqual("Demo transcript", analysis["transcript"]["full_text"])
        self.assertEqual(1, len(analysis["transcript"]["segments"]))
        self.assertEqual(1, len(analysis["visuals"]["slides"]))
        self.assertEqual("slide", analysis["visuals"]["slides"][0]["effective_label"])
        self.assertEqual(1, len(analysis["visuals"]["slides"][0]["images"]))
        self.assertEqual(0, self.gpt_call_count)

    def test_chart_heavy_input_saves_durable_chart_visuals(self) -> None:
        source_path, out_dir = self.make_stubbed_run(
            keyframes=[
                {"kind": "scene", "filename": "chart-1.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00"},
                {"kind": "scene", "filename": "chart-2.jpg", "timestamp_seconds": 4.0, "timestamp_hms": "00:00:04"},
            ],
            ocr_rows=[
                {"filename": "chart-1.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": "Q1 revenue 42% growth table chart"},
                {"filename": "chart-2.jpg", "timestamp_seconds": 4.0, "timestamp_hms": "00:00:04", "text": "Q2 revenue 57% growth table chart"},
            ],
            phash_by_filename={"chart-1.jpg": "0f0f0f0f0f0f0f0f", "chart-2.jpg": "f0f0f0f0f0f0f0f0"},
        )

        output_root = pipeline.analyze_source(
            str(source_path),
            out_dir=out_dir,
            gpt_mode="off",
            artifacts_mode="debug",
        )

        analysis = json.loads((output_root / "output.json").read_text(encoding="utf-8"))
        manifest = json.loads((output_root / "visuals" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(1, len(analysis["visuals"]["charts"]))
        self.assertEqual("chart", analysis["visuals"]["charts"][0]["effective_label"])
        self.assertEqual(1, len(analysis["visuals"]["charts"][0]["images"]))
        self.assertEqual(1, len(manifest["charts"]))
        self.assertEqual(2, len(manifest["charts"][0]["image_paths"]))
        for path in manifest["charts"][0]["image_paths"]:
            self.assertTrue((output_root / path).exists())

    def test_talking_head_input_yields_no_gpt_candidates(self) -> None:
        source_path, out_dir = self.make_stubbed_run(
            keyframes=[
                {"kind": "scene", "filename": "talk-1.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00"},
                {"kind": "scene", "filename": "talk-2.jpg", "timestamp_seconds": 6.0, "timestamp_hms": "00:00:06"},
            ],
            ocr_rows=[
                {"filename": "talk-1.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": ""},
                {"filename": "talk-2.jpg", "timestamp_seconds": 6.0, "timestamp_hms": "00:00:06", "text": ""},
            ],
        )

        output_root = pipeline.analyze_source(
            str(source_path),
            out_dir=out_dir,
            gpt_mode="on",
            review_mode="off",
            artifacts_mode="debug",
        )

        manifest = json.loads((output_root / "routing" / "manifest.json").read_text(encoding="utf-8"))["segments"]
        self.assertTrue(manifest)
        self.assertFalse(any(entry["approved_for_gpt"] for entry in manifest))
        self.assertTrue((output_root / "report" / "report.md").exists())
        analysis = json.loads((output_root / "output.json").read_text(encoding="utf-8"))
        self.assertIn("gpt", analysis)
        self.assertEqual("影片分析報告", analysis["gpt"]["final_report"]["title"])
        self.assertEqual([], analysis["visuals"]["slides"])
        self.assertEqual([], analysis["visuals"]["charts"])

    def test_uncertain_content_stops_before_gpt_until_reviewed(self) -> None:
        source_path, out_dir = self.make_stubbed_run(
            keyframes=[
                {"kind": "scene", "filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00"},
                {"kind": "scene", "filename": "uncertain.jpg", "timestamp_seconds": 20.0, "timestamp_hms": "00:00:20"},
            ],
            ocr_rows=[
                {"filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": "Revenue revenue revenue revenue revenue revenue"},
                {
                    "filename": "uncertain.jpg",
                    "timestamp_seconds": 20.0,
                    "timestamp_hms": "00:00:20",
                    "text": "context maybe important context maybe",
                },
            ],
            matrix_by_filename={"slide.jpg": detailed_matrix(), "uncertain.jpg": alternate_matrix()},
            phash_by_filename={"slide.jpg": "0f0f0f0f0f0f0f0f", "uncertain.jpg": "f0f0f0f0f0f0f0f0"},
        )

        output_root = pipeline.analyze_source(
            str(source_path),
            out_dir=out_dir,
            gpt_mode="on",
            review_mode="off",
            artifacts_mode="debug",
        )

        manifest = json.loads((output_root / "routing" / "manifest.json").read_text(encoding="utf-8"))["segments"]
        uncertain_entries = [entry for entry in manifest if entry["effective_label"] == "uncertain"]
        self.assertEqual(1, len(uncertain_entries))
        self.assertEqual("pending_review", uncertain_entries[0]["routing_disposition"])
        self.assertFalse(uncertain_entries[0]["approved_for_gpt"])

    def test_successful_run_writes_only_output_json_in_minimal_mode(self) -> None:
        source_path, out_dir = self.make_stubbed_run(
            keyframes=[
                {"kind": "scene", "filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00"},
            ],
            ocr_rows=[
                {"filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": "Revenue revenue revenue revenue revenue revenue"},
            ],
        )

        output_root = pipeline.analyze_source(str(source_path), out_dir=out_dir, gpt_mode="off")

        self.assertFalse((output_root / "audio").exists())
        self.assertFalse((output_root / "video").exists())
        self.assertFalse((output_root / "subtitles").exists())
        self.assertFalse((output_root / "keyframes").exists())
        self.assertFalse((output_root / "ocr").exists())
        self.assertFalse((output_root / "triage").exists())
        self.assertFalse((output_root / "routing").exists())
        self.assertFalse((output_root / "visuals").exists())
        self.assertFalse((output_root / "report").exists())
        self.assertFalse((output_root / "gpt").exists())
        self.assertFalse((output_root / "metadata.json").exists())
        self.assertFalse((output_root / "transcript.json").exists())
        analysis = json.loads((output_root / "output.json").read_text(encoding="utf-8"))
        self.assertEqual("minimal", analysis["processing"]["artifact_mode"])
        self.assertEqual("auto", analysis["processing"]["ocr_mode"])
        self.assertEqual("completed", analysis["processing"]["ocr_status"])
        self.assertEqual(1, analysis["processing"]["counts"]["slide_count"])
        self.assertNotIn("routing", analysis["processing"])
        self.assertNotIn("artifacts", analysis)
        self.assertEqual(1, len(analysis["visuals"]["slides"]))
        self.assertEqual(1, len(analysis["visuals"]["slides"][0]["images"]))
        self.assertEqual("Revenue revenue revenue revenue revenue revenue", analysis["visuals"]["slides"][0]["ocr_summary"])
        self.assertEqual("base64", analysis["visuals"]["slides"][0]["images"][0]["encoding"])

    def test_debug_artifacts_mode_preserves_trace_outputs(self) -> None:
        source_path, out_dir = self.make_stubbed_run(
            keyframes=[
                {"kind": "scene", "filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00"},
            ],
            ocr_rows=[
                {"filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": "Revenue revenue revenue revenue revenue revenue"},
            ],
        )

        output_root = pipeline.analyze_source(
            str(source_path),
            out_dir=out_dir,
            gpt_mode="off",
            artifacts_mode="debug",
        )

        self.assertTrue((output_root / "output.json").exists())
        self.assertTrue((output_root / "triage" / "segments.json").exists())
        self.assertTrue((output_root / "routing" / "manifest.json").exists())
        self.assertTrue((output_root / "visuals" / "manifest.json").exists())
        self.assertTrue((output_root / "metadata.json").exists())
        self.assertTrue((output_root / "transcript.json").exists())

    def test_failed_run_also_cleans_intermediate_dirs(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        source_path = root / "input.mp4"
        source_path.write_bytes(b"video")
        out_dir = root / "output"

        def fake_materialize_local_input(source_file: Path, paths: pipeline.AnalysisPaths):
            return {"format": {"duration": "120"}, "streams": [{"codec_type": "video"}]}, source_file

        def fake_extract_audio(_: Path, audio_dir: Path) -> Path:
            audio_path = audio_dir / "source.wav"
            audio_path.write_bytes(b"audio")
            return audio_path

        with mock.patch.object(pipeline, "materialize_local_input", side_effect=fake_materialize_local_input), mock.patch.object(
            pipeline, "extract_audio", side_effect=fake_extract_audio
        ), mock.patch.object(
            pipeline, "transcript_strategy_auto", side_effect=RuntimeError("boom")
        ):
            with self.assertRaises(RuntimeError):
                pipeline.analyze_source(str(source_path), out_dir=out_dir, gpt_mode="off")

        self.assertFalse((out_dir / "audio").exists())
        self.assertFalse((out_dir / "video").exists())
        self.assertFalse((out_dir / "subtitles").exists())
        self.assertFalse((out_dir / "keyframes").exists())
        self.assertFalse((out_dir / "ocr").exists())
        self.assertFalse((out_dir / "error.json").exists())
        self.assertFalse((out_dir / "metadata.json").exists())
        self.assertFalse((out_dir / "transcript.json").exists())
        analysis = json.loads((out_dir / "output.json").read_text(encoding="utf-8"))
        self.assertTrue(analysis["errors"])
        self.assertEqual("boom", analysis["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
