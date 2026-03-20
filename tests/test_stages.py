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

from youtube_analysis_tool import constants, gpt, pipeline, reporting, review, routing, triage, visuals


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


class OutputBundleTests(unittest.TestCase):
    def test_output_json_embeds_full_transcript_and_visual_gallery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = pipeline.analysis_paths(root)
            (root / "keyframes").mkdir(parents=True, exist_ok=True)
            (root / "keyframes" / "slide.jpg").write_bytes(b"slide-bytes")

            payload = reporting.write_output_file(
                paths,
                source_input="/tmp/demo.mp4",
                is_url=False,
                is_youtube_url=False,
                metadata={"id": "demo", "title": "Demo", "webpage_url": "https://youtu.be/demo", "chapters": [{"title": "Intro"}]},
                transcript={
                    "source": "subtitle",
                    "language": "zh-tw",
                    "text": "Revenue is up.",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "Revenue is up."}],
                },
                ocr={"mode": "auto", "status": "completed", "attempted": True, "frame_count": 1, "error": None},
                burned_subtitles={"mode": "auto", "status": "not_attempted", "attempted": False, "probe_passed": False, "ocr_event_count": 0, "error": None},
                visuals_payload={
                    "slides": [
                        {
                            "segment_id": "segment-0001",
                            "effective_label": "slide",
                            "heuristic_confidence": 0.91,
                            "start_seconds": 0.0,
                            "end_seconds": 5.0,
                            "start_hms": "00:00:00",
                            "end_hms": "00:00:05",
                            "ocr_text": "Revenue trend",
                            "ocr_summary": "Revenue trend",
                            "ocr_char_count": 80,
                            "transcript_excerpt": "Revenue is up.",
                            "images": [
                                {
                                    "filename": "slide.jpg",
                                    "mime_type": "image/jpeg",
                                    "encoding": "base64",
                                    "data": "c2xpZGUtYnl0ZXM=",
                                }
                            ],
                            "primary_image_index": 0,
                        }
                    ],
                    "charts": [],
                },
                errors=[],
                cleanup_intermediates=True,
                transcript_mode="auto",
                visuals_mode="on",
                ocr_mode="auto",
                gpt_mode="off",
                artifacts_mode="minimal",
            )

            written = json.loads(paths.output_json_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["output_version"], written["output_version"])
        self.assertEqual("https://youtu.be/demo", written["metadata"]["url"])
        self.assertEqual([{"title": "Intro"}], written["metadata"]["chapters"])
        self.assertEqual("subtitle", written["transcript"]["source"])
        self.assertEqual("Revenue is up.", written["transcript"]["full_text"])
        self.assertEqual(1, len(written["transcript"]["segments"]))
        self.assertEqual(1, written["transcript"]["segment_count"])
        self.assertEqual("text_track", written["transcript"]["provenance"]["extraction_kind"])
        self.assertEqual(1, len(written["visuals"]["slides"]))
        self.assertEqual("base64", written["visuals"]["slides"][0]["images"][0]["encoding"])
        self.assertEqual("slide", written["visuals"]["slides"][0]["effective_label"])
        self.assertEqual(
            "segment-0001",
            written["visuals"]["slides"][0]["source_segment_ref"]["segment_id"],
        )
        self.assertEqual(
            "heuristic_segment_promotion",
            written["visuals"]["slides"][0]["provenance"]["selection_kind"],
        )
        self.assertEqual("minimal", written["processing"]["artifact_mode"])
        self.assertEqual("auto", written["processing"]["transcript_mode"])
        self.assertEqual("auto", written["processing"]["ocr_mode"])
        self.assertEqual("auto", written["processing"]["burned_subtitles_mode"])
        self.assertEqual("not_attempted", written["processing"]["burned_subtitles_status"])
        self.assertIsNone(written["processing"]["burned_subtitles_reason"])
        self.assertEqual(0, written["processing"]["burned_subtitles_probe_hits"])
        self.assertEqual(0, written["processing"]["burned_subtitles_ocr_events"])
        self.assertFalse(written["processing"]["gpt_enabled"])
        self.assertEqual("direct_metadata_extract", written["provenance"]["metadata"]["extraction_kind"])
        self.assertTrue(written["provenance"]["transcript"]["is_direct_text_track"])
        self.assertEqual(1, written["provenance"]["visuals"]["slides"]["count"])
        self.assertEqual(["segment-0001"], written["provenance"]["visuals"]["slides"]["segment_ids"])
        self.assertNotIn("gpt", written)

    def test_output_json_can_embed_gpt_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = pipeline.analysis_paths(root)

            payload = reporting.write_output_file(
                paths,
                source_input="https://youtu.be/demo",
                is_url=True,
                is_youtube_url=True,
                metadata={"id": "demo", "title": "Demo"},
                transcript={"source": "subtitle", "text": "Revenue is up.", "segments": []},
                ocr={"mode": "auto", "status": "completed", "attempted": True, "frame_count": 1, "error": None},
                burned_subtitles={"mode": "auto", "status": "not_attempted", "attempted": False, "probe_passed": False, "ocr_event_count": 0, "error": None},
                visuals_payload={"slides": [], "charts": []},
                errors=[],
                cleanup_intermediates=True,
                transcript_mode="auto",
                visuals_mode="on",
                ocr_mode="auto",
                gpt_mode="on",
                artifacts_mode="minimal",
                gpt_payload={
                    "model": "gpt-5.4",
                    "report_language": "zh-TW",
                    "segment_analyses": [],
                    "final_report": {"title": "影片分析報告"},
                },
            )

        self.assertEqual("gpt-5.4", payload["gpt"]["model"])
        self.assertEqual("影片分析報告", payload["gpt"]["final_report"]["title"])


class VisualPayloadTests(unittest.TestCase):
    def test_build_embedded_visuals_exports_only_slides_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keyframes").mkdir(parents=True, exist_ok=True)
            for filename in ("slide.jpg", "chart.jpg", "uncertain.jpg"):
                (root / "keyframes" / filename).write_bytes(filename.encode("utf-8"))
            (root / "keyframes" / "chart-2.jpg").write_bytes(b"chart-2")

            frames = [
                {
                    "frame_id": "frame-1",
                    "frame_path": "keyframes/slide.jpg",
                    "timestamp_seconds": 0.0,
                    "blur_score": 200.0,
                    "ocr_text": "Slide OCR",
                },
                {
                    "frame_id": "frame-2",
                    "frame_path": "keyframes/chart.jpg",
                    "timestamp_seconds": 1.0,
                    "blur_score": 250.0,
                    "ocr_text": "Chart OCR",
                },
                {
                    "frame_id": "frame-4",
                    "frame_path": "keyframes/chart-2.jpg",
                    "timestamp_seconds": 1.5,
                    "blur_score": 150.0,
                    "ocr_text": "Chart OCR continued",
                },
                {
                    "frame_id": "frame-3",
                    "frame_path": "keyframes/uncertain.jpg",
                    "timestamp_seconds": 2.0,
                    "blur_score": 150.0,
                    "ocr_text": "Uncertain OCR",
                },
            ]
            segments = [
                {
                    "segment_id": "segment-0001",
                    "heuristic_label": "slides",
                    "heuristic_confidence": 0.9,
                    "frame_ids": ["frame-1"],
                    "representative_frame_paths": ["keyframes/slide.jpg"],
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:00",
                    "ocr_summary": "Slide",
                    "ocr_char_count": 5,
                    "transcript_window": {"text": "Slide excerpt"},
                },
                {
                    "segment_id": "segment-0002",
                    "heuristic_label": "chart_table",
                    "heuristic_confidence": 0.8,
                    "frame_ids": ["frame-2", "frame-4"],
                    "representative_frame_paths": ["keyframes/chart.jpg"],
                    "start_seconds": 1.0,
                    "end_seconds": 1.0,
                    "start_hms": "00:00:01",
                    "end_hms": "00:00:01",
                    "ocr_summary": "Chart",
                    "ocr_char_count": 5,
                    "transcript_window": {"text": "Chart excerpt"},
                },
                {
                    "segment_id": "segment-0003",
                    "heuristic_label": "uncertain",
                    "heuristic_confidence": 0.5,
                    "frame_ids": ["frame-3"],
                    "representative_frame_paths": ["keyframes/uncertain.jpg"],
                    "start_seconds": 2.0,
                    "end_seconds": 2.0,
                    "start_hms": "00:00:02",
                    "end_hms": "00:00:02",
                    "ocr_summary": "Uncertain",
                    "ocr_char_count": 9,
                    "transcript_window": {"text": "Uncertain excerpt"},
                },
            ]
            manifest_entries = [
                {"segment_id": "segment-0001", "effective_label": "slides"},
                {"segment_id": "segment-0002", "effective_label": "chart_table"},
                {"segment_id": "segment-0003", "effective_label": "uncertain"},
            ]

            payload = visuals.build_embedded_visuals(
                root,
                frames=frames,
                segments=segments,
                manifest_entries=manifest_entries,
            )

        self.assertEqual(1, len(payload["slides"]))
        self.assertEqual(1, len(payload["charts"]))
        self.assertEqual("slide", payload["slides"][0]["effective_label"])
        self.assertEqual("chart", payload["charts"][0]["effective_label"])
        self.assertEqual("Slide OCR", payload["slides"][0]["ocr_text"])
        self.assertEqual("base64", payload["slides"][0]["images"][0]["encoding"])
        self.assertEqual(1, len(payload["charts"][0]["images"]))
        self.assertIn("Chart OCR continued", payload["charts"][0]["ocr_text"])
        self.assertEqual(
            "segment-0001",
            payload["slides"][0]["source_segment_ref"]["segment_id"],
        )
        self.assertEqual(
            "heuristic_segment_promotion",
            payload["charts"][0]["provenance"]["selection_kind"],
        )

    def test_save_durable_visuals_exports_only_slides_and_charts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "keyframes").mkdir(parents=True, exist_ok=True)
            for filename in ("slide.jpg", "chart.jpg", "uncertain.jpg"):
                (root / "keyframes" / filename).write_bytes(filename.encode("utf-8"))

            frames = [
                {
                    "frame_id": "frame-1",
                    "frame_path": "keyframes/slide.jpg",
                    "timestamp_seconds": 0.0,
                    "blur_score": 200.0,
                    "ocr_text": "Slide OCR",
                },
                {
                    "frame_id": "frame-2",
                    "frame_path": "keyframes/chart.jpg",
                    "timestamp_seconds": 1.0,
                    "blur_score": 250.0,
                    "ocr_text": "Chart OCR",
                },
                {
                    "frame_id": "frame-3",
                    "frame_path": "keyframes/uncertain.jpg",
                    "timestamp_seconds": 2.0,
                    "blur_score": 150.0,
                    "ocr_text": "Uncertain OCR",
                },
            ]
            segments = [
                {
                    "segment_id": "segment-0001",
                    "heuristic_label": "slides",
                    "frame_ids": ["frame-1"],
                    "representative_frame_paths": ["keyframes/slide.jpg"],
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:00",
                    "ocr_summary": "Slide",
                    "ocr_char_count": 5,
                    "transcript_window": {"text": "Slide excerpt"},
                },
                {
                    "segment_id": "segment-0002",
                    "heuristic_label": "chart_table",
                    "frame_ids": ["frame-2"],
                    "representative_frame_paths": ["keyframes/chart.jpg"],
                    "start_seconds": 1.0,
                    "end_seconds": 1.0,
                    "start_hms": "00:00:01",
                    "end_hms": "00:00:01",
                    "ocr_summary": "Chart",
                    "ocr_char_count": 5,
                    "transcript_window": {"text": "Chart excerpt"},
                },
                {
                    "segment_id": "segment-0003",
                    "heuristic_label": "uncertain",
                    "frame_ids": ["frame-3"],
                    "representative_frame_paths": ["keyframes/uncertain.jpg"],
                    "start_seconds": 2.0,
                    "end_seconds": 2.0,
                    "start_hms": "00:00:02",
                    "end_hms": "00:00:02",
                    "ocr_summary": "Uncertain",
                    "ocr_char_count": 9,
                    "transcript_window": {"text": "Uncertain excerpt"},
                },
            ]
            manifest_entries = [
                {"segment_id": "segment-0001", "effective_label": "slides"},
                {"segment_id": "segment-0002", "effective_label": "chart_table"},
                {"segment_id": "segment-0003", "effective_label": "uncertain"},
            ]

            payload = visuals.save_durable_visuals(
                root,
                frames=frames,
                segments=segments,
                manifest_entries=manifest_entries,
            )

            manifest = json.loads((root / "visuals" / "manifest.json").read_text(encoding="utf-8"))
            slide_exists = (root / "visuals" / "slides" / "segment-0001" / "slide.jpg").exists()
            chart_exists = (root / "visuals" / "charts" / "segment-0002" / "chart.jpg").exists()
            uncertain_exists = (root / "visuals" / "slides" / "segment-0003").exists()

        self.assertEqual(1, len(payload["slides"]))
        self.assertEqual(1, len(payload["charts"]))
        self.assertEqual(payload, manifest)
        self.assertEqual(
            "segment-0001",
            payload["slides"][0]["source_segment_ref"]["segment_id"],
        )
        self.assertEqual(
            "heuristic_segment_promotion",
            payload["charts"][0]["provenance"]["selection_kind"],
        )
        self.assertTrue(slide_exists)
        self.assertTrue(chart_exists)
        self.assertFalse(uncertain_exists)


if __name__ == "__main__":
    unittest.main()
