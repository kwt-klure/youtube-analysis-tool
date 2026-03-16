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

from youtube_analysis_tool import constants, gpt, pipeline, reporting, review, routing, triage
from youtube_analysis_tool.artifacts import write_json


class TriageHelperTests(unittest.TestCase):
    def test_blur_score_distinguishes_flat_and_detailed_frames(self) -> None:
        flat = [[10.0 for _ in range(5)] for _ in range(5)]
        detailed = [
            [0.0, 255.0, 0.0, 255.0, 0.0],
            [255.0, 0.0, 255.0, 0.0, 255.0],
            [0.0, 255.0, 0.0, 255.0, 0.0],
            [255.0, 0.0, 255.0, 0.0, 255.0],
            [0.0, 255.0, 0.0, 255.0, 0.0],
        ]

        self.assertLess(triage.compute_blur_score(flat), triage.compute_blur_score(detailed))

    def test_motion_proxy_detects_change(self) -> None:
        left = [[0.0 for _ in range(4)] for _ in range(4)]
        right = [[255.0 for _ in range(4)] for _ in range(4)]

        self.assertEqual(0.0, triage.compute_motion_proxy(left, left))
        self.assertGreater(triage.compute_motion_proxy(left, right), 0.9)

    def test_heuristic_label_reads_thresholds_from_constants(self) -> None:
        frame = {
            "ocr_char_count": 6,
            "motion_proxy": 0.01,
            "blur_score": 250.0,
            "numeric_token_ratio": 0.0,
            "chart_hint_score": 0.0,
        }

        with mock.patch.object(constants, "DEFAULT_SLIDE_OCR_CHAR_COUNT", 5), mock.patch.object(
            constants, "DEFAULT_TALKING_HEAD_OCR_CHAR_MAX", 0
        ):
            label, confidence, _ = triage.choose_heuristic_label(frame)

        self.assertEqual("slides", label)
        self.assertGreater(confidence, 0.55)

    def test_duplicate_groups_keep_sharpest_representative(self) -> None:
        frames = [
            {"frame_id": "frame-1", "timestamp_seconds": 0.0, "phash": "0f0f0f0f0f0f0f0f", "blur_score": 150.0, "ocr_char_count": 5},
            {"frame_id": "frame-2", "timestamp_seconds": 1.0, "phash": "0f0f0f0f0f0f0f0e", "blur_score": 300.0, "ocr_char_count": 5},
        ]

        triage.assign_duplicate_groups(frames)

        self.assertEqual(frames[0]["duplicate_group"], frames[1]["duplicate_group"])
        self.assertFalse(frames[0]["is_duplicate_representative"])
        self.assertTrue(frames[1]["is_duplicate_representative"])

    def test_merge_segments_binds_transcript_window(self) -> None:
        frames = [
            {
                "frame_id": "frame-1",
                "timestamp_seconds": 0.0,
                "timestamp_hms": "00:00:00",
                "frame_path": "keyframes/scene-1.jpg",
                "heuristic_label": "slides",
                "heuristic_confidence": 0.9,
                "ocr_text": "Quarterly revenue",
                "ocr_char_count": 16,
                "numeric_token_ratio": 0.25,
                "chart_hint_score": 0.0,
                "keep_drop_recommendation": "keep",
                "is_duplicate_representative": True,
            },
            {
                "frame_id": "frame-2",
                "timestamp_seconds": 8.0,
                "timestamp_hms": "00:00:08",
                "frame_path": "keyframes/scene-2.jpg",
                "heuristic_label": "slides",
                "heuristic_confidence": 0.88,
                "ocr_text": "Quarterly revenue",
                "ocr_char_count": 16,
                "numeric_token_ratio": 0.25,
                "chart_hint_score": 0.0,
                "keep_drop_recommendation": "keep",
                "is_duplicate_representative": True,
            },
        ]
        transcript = {
            "segments": [
                {"start": 0.0, "end": 3.0, "text": "Intro"},
                {"start": 6.0, "end": 10.0, "text": "Revenue is rising"},
            ]
        }

        segments = triage.merge_frames_to_segments(frames, transcript)

        self.assertEqual(1, len(segments))
        self.assertIn("Revenue is rising", segments[0]["transcript_window"]["text"])
        self.assertEqual(["keyframes/scene-1.jpg", "keyframes/scene-2.jpg"], segments[0]["representative_frame_paths"])


class RoutingAndReviewTests(unittest.TestCase):
    def test_manifest_requires_review_for_uncertain_or_high_detail(self) -> None:
        segment = {
            "segment_id": "segment-0001",
            "heuristic_label": "chart_table",
            "heuristic_confidence": 0.9,
            "representative_frame_paths": ["keyframes/chart.jpg"],
            "ocr_summary": "Q1 2024 revenue table",
            "transcript_window": {"text": "The chart shows growth."},
            "ocr_char_count": 150,
        }
        manifest = [routing.manifest_entry_from_segment(segment, "gpt-4o")]

        queue = review.build_review_queue(Path(tempfile.mkdtemp()), manifest)

        self.assertEqual("high", manifest[0]["detail"])
        self.assertTrue(manifest[0]["review_required"])
        self.assertEqual(1, len(queue))

    def test_pending_review_blocks_gpt_until_approved(self) -> None:
        manifest = [
            {
                "segment_id": "segment-0001",
                "heuristic_label": "uncertain",
                "effective_label": "uncertain",
                "heuristic_confidence": 0.4,
                "representative_frame_paths": ["keyframes/frame.jpg"],
                "ocr_summary": "",
                "transcript_window": {"text": ""},
                "review_required": True,
                "review_status": "pending",
                "routing_disposition": "candidate",
                "approved_for_gpt": False,
                "prompt_family": "uncertain",
                "detail": "low",
                "reason": [],
                "review_note": "",
            }
        ]

        routing.finalize_manifest_entries(manifest, review_enabled=False)

        self.assertEqual("pending_review", manifest[0]["routing_disposition"])
        self.assertFalse(manifest[0]["approved_for_gpt"])

    def test_interactive_review_persists_note_and_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "review").mkdir(parents=True, exist_ok=True)
            (root / "keyframes").mkdir(parents=True, exist_ok=True)
            (root / "keyframes" / "frame.jpg").write_bytes(b"frame")
            queue = [
                {
                    "segment_id": "segment-0001",
                    "effective_label": "uncertain",
                    "heuristic_label": "uncertain",
                    "heuristic_confidence": 0.5,
                    "detail": "low",
                    "representative_frame_paths": ["keyframes/frame.jpg"],
                    "ocr_summary": "",
                    "transcript_text": "",
                    "routing_disposition": "candidate",
                }
            ]
            decisions = {}
            responses = iter(["note needs review", "keep"])

            review.interactive_review(root, queue, decisions, input_func=lambda _: next(responses))

            payload = json.loads((root / "review" / "decisions.json").read_text(encoding="utf-8"))
            self.assertEqual("approved", payload["decisions"]["segment-0001"]["status"])
            self.assertEqual("needs review", payload["decisions"]["segment-0001"]["note"])


class GptAndReportTests(unittest.TestCase):
    def test_analyze_segments_uses_mock_client_and_writes_artifact(self) -> None:
        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.output_text = text

        class FakeResponses:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **_: object) -> FakeResponse:
                self.calls += 1
                if self.calls == 1:
                    return FakeResponse(
                        json.dumps(
                            {
                                "segment_summary": "這是一張投影片。",
                                "key_evidence": ["營收圖"],
                                "inferred_visual_type": "slides",
                                "importance": "high",
                                "confidence": 0.9,
                                "uncertainties": [],
                                "transcript_linkage": "講者正在解釋這張圖。",
                                "source_segment_id": "segment-0001",
                            },
                            ensure_ascii=False,
                        )
                    )
                return FakeResponse(
                    json.dumps(
                        {
                            "title": "影片分析報告",
                            "executive_summary": "重點是營收成長。",
                            "main_sections": [{"heading": "財務重點", "summary": "營收向上。", "source_segment_ids": ["segment-0001"]}],
                            "key_visuals": [{"summary": "營收圖表", "source_segment_id": "segment-0001"}],
                            "speaker_points": ["營收持續成長"],
                            "open_questions": [],
                            "source_segment_ids": ["segment-0001"],
                        },
                        ensure_ascii=False,
                    )
                )

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "gpt").mkdir(parents=True, exist_ok=True)
            (root / "keyframes").mkdir(parents=True, exist_ok=True)
            (root / "keyframes" / "slide.jpg").write_bytes(b"jpeg-bytes")
            manifest = [
                {
                    "segment_id": "segment-0001",
                    "effective_label": "slides",
                    "prompt_family": "slides",
                    "approved_for_gpt": True,
                    "detail": "low",
                    "representative_frame_paths": ["keyframes/slide.jpg"],
                    "transcript_window": {"text": "The revenue trend is up."},
                    "ocr_summary": "Revenue trend",
                }
            ]

            segment_analyses, report_payload = gpt.analyze_segments(
                root,
                manifest,
                model="gpt-5.4",
                transcript={"text": "The revenue trend is up."},
                metadata={"title": "Demo"},
                report_language="zh-TW",
                client=FakeClient(),
            )

            self.assertEqual(1, len(segment_analyses))
            self.assertEqual("影片分析報告", report_payload["title"])
            artifact = json.loads((root / "gpt" / "analyses.json").read_text(encoding="utf-8"))
            self.assertEqual("segment-0001", artifact["segment_analyses"][0]["source_segment_id"])


class AnalysisBundleTests(unittest.TestCase):
    def test_analysis_json_is_written_without_gpt_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = pipeline.analysis_paths(root)
            for path in (
                paths.triage_dir,
                paths.review_dir,
                paths.routing_dir,
                paths.gpt_dir,
                paths.report_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)
            write_json(paths.metadata_path, {"id": "demo", "title": "Demo"})
            write_json(paths.transcript_json_path, {"source": "subtitle"})
            paths.transcript_text_path.write_text("demo\n", encoding="utf-8")
            write_json(paths.triage_segments_path, {"segments": []})
            write_json(paths.review_queue_path, {"queue": []})
            write_json(paths.review_decisions_path, {"decisions": {}})
            write_json(paths.routing_manifest_path, {"segments": []})
            paths.triage_frames_path.write_text("", encoding="utf-8")
            segments = [
                {
                    "segment_id": "segment-0001",
                    "heuristic_label": "slides",
                    "heuristic_confidence": 0.9,
                    "start_seconds": 0.0,
                    "end_seconds": 5.0,
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:05",
                    "frame_ids": ["frame-00001"],
                    "representative_frame_paths": ["keyframes/slide.jpg"],
                    "ocr_summary": "Revenue trend",
                    "ocr_char_count": 80,
                    "numeric_token_ratio": 0.2,
                    "chart_hint_score": 0.0,
                    "transcript_window": {"text": "Revenue is up."},
                    "review_required": False,
                    "review_status": "auto_approved",
                    "routing_disposition": "candidate",
                }
            ]
            manifest = [
                {
                    "segment_id": "segment-0001",
                    "effective_label": "slides",
                    "review_required": False,
                    "review_status": "auto_approved",
                    "routing_disposition": "candidate",
                    "approved_for_gpt": False,
                    "detail": "low",
                    "prompt_family": "slides",
                    "review_note": "",
                }
            ]

            payload = reporting.write_analysis_file(
                paths,
                source_input="/tmp/demo.mp4",
                is_url=False,
                is_youtube_url=False,
                metadata={"id": "demo", "title": "Demo"},
                transcript={"source": "subtitle", "text": "Revenue is up.", "segments": []},
                ocr={"mode": "auto", "status": "completed", "attempted": True, "frame_count": 1, "artifact_path": "ocr/index.csv", "error": None},
                segments=segments,
                manifest_entries=manifest,
                errors=[],
                cleanup_intermediates=True,
                transcript_mode="auto",
                keyframe_mode="scene+interval",
                ocr_mode="auto",
                triage_mode="on",
                gpt_mode="off",
                review_mode="interactive",
                interval_seconds=60,
                scene_threshold=0.35,
            )

            written = json.loads(paths.analysis_json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["analysis_version"], written["analysis_version"])
        self.assertEqual("Revenue trend", written["segments"][0]["ocr_summary"])
        self.assertIn("routing_manifest_json", written["artifacts"])
        self.assertNotIn("gpt", written)
        self.assertFalse(written["artifacts"]["report_json"]["exists"])

    def test_analysis_json_can_embed_gpt_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = pipeline.analysis_paths(root)
            for path in (
                paths.triage_dir,
                paths.review_dir,
                paths.routing_dir,
                paths.gpt_dir,
                paths.report_dir,
            ):
                path.mkdir(parents=True, exist_ok=True)

            payload = reporting.write_analysis_file(
                paths,
                source_input="https://youtu.be/demo",
                is_url=True,
                is_youtube_url=True,
                metadata={"id": "demo", "title": "Demo"},
                transcript={"source": "subtitle", "text": "Revenue is up.", "segments": []},
                ocr={"mode": "auto", "status": "completed", "attempted": True, "frame_count": 1, "artifact_path": "ocr/index.csv", "error": None},
                segments=[],
                manifest_entries=[],
                errors=[],
                cleanup_intermediates=True,
                transcript_mode="auto",
                keyframe_mode="scene+interval",
                ocr_mode="auto",
                triage_mode="on",
                gpt_mode="on",
                review_mode="off",
                interval_seconds=60,
                scene_threshold=0.35,
                gpt_payload={
                    "model": "gpt-5.4",
                    "report_language": "zh-TW",
                    "segment_analyses": [],
                    "final_report": {"title": "影片分析報告"},
                },
            )

        self.assertEqual("gpt-5.4", payload["gpt"]["model"])
        self.assertEqual("影片分析報告", payload["gpt"]["final_report"]["title"])


if __name__ == "__main__":
    unittest.main()
