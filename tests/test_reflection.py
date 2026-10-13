from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from youtube_analysis_tool import reflection


class RunReflectionTests(unittest.TestCase):
    def test_default_cheap_run_is_limited_when_rich_layers_are_disabled(self) -> None:
        payload = reflection.build_run_reflection(
            transcript={
                "source": "subtitle_manual",
                "segments": [{"start": 0.0, "end": 3.0, "text": "Plain transcript"}],
                "full_text": "Plain transcript",
            },
            visuals={"slides": [], "charts": []},
            visual_sampling={"status": "skipped", "candidate_frame_count": 0},
            audio_features={"status": "disabled"},
            comments={"status": "disabled", "items": []},
        )

        self.assertEqual("limited", payload["status"])
        self.assertEqual("limited", payload["summary"]["overall_confidence"])
        self.assertEqual(["transcript"], payload["summary"]["available_evidence_layers"])
        self.assertIn("visuals_skipped", payload["summary"]["limitations"])
        self.assertIn("audio_features_disabled", payload["summary"]["limitations"])
        self.assertIn("comments_disabled", payload["summary"]["limitations"])

    def test_visual_reference_without_nearby_retained_visual_suggests_denser_sampling(self) -> None:
        payload = reflection.build_run_reflection(
            transcript={
                "source": "whisper",
                "segments": [
                    {
                        "start": 20.0,
                        "end": 24.0,
                        "text": "這張截圖可以看到工具漏掉了完整畫面",
                    }
                ],
                "full_text": "這張截圖可以看到工具漏掉了完整畫面",
            },
            visuals={"slides": [], "charts": []},
            visual_sampling={
                "status": "extracted",
                "density": "medium",
                "interval_seconds": 30,
                "candidate_frame_count": 12,
                "retained_visual_count": 0,
            },
            audio_features={"status": "disabled"},
            comments={"status": "disabled", "items": []},
        )

        self.assertEqual("limited", payload["status"])
        self.assertEqual("visual_reference_without_retained_visual", payload["uncertainties"][0]["type"])
        self.assertEqual(20.0, payload["uncertainties"][0]["start_seconds"])
        self.assertIn(
            "increase_visual_density",
            [item["action"] for item in payload["next_run_adjustments"]],
        )
        self.assertEqual("visual_sampling", payload["skill_patch_candidates"][0]["area"])
        self.assertTrue(payload["skill_patch_candidates"][0]["requires_human_review"])

    def test_audio_large_change_becomes_time_anchor_not_semantic_claim(self) -> None:
        payload = reflection.build_run_reflection(
            transcript={"source": "subtitle_manual", "segments": [], "full_text": "Transcript"},
            visuals={"slides": [], "charts": []},
            visual_sampling={"status": "skipped", "candidate_frame_count": 0},
            audio_features={
                "status": "extracted",
                "large_changes": [
                    {
                        "start": 60.0,
                        "end": 75.0,
                        "from_dbfs": -12.0,
                        "to_dbfs": -20.0,
                        "delta_db": -8.0,
                    }
                ],
                "quiet_segments": [],
                "loud_segments": [],
            },
            comments={"status": "disabled", "items": []},
        )

        adjustment = payload["next_run_adjustments"][0]
        self.assertEqual("inspect_audio_change_anchor", adjustment["action"])
        self.assertEqual(60.0, adjustment["start_seconds"])
        self.assertEqual("audio features are routing signals, not semantic conclusions", adjustment["caution"])

    def test_method_question_comments_are_classified_as_critique_signals(self) -> None:
        payload = reflection.build_run_reflection(
            transcript={"source": "subtitle_manual", "segments": [], "full_text": "Transcript"},
            visuals={"slides": [], "charts": []},
            visual_sampling={"status": "skipped", "candidate_frame_count": 0},
            audio_features={"status": "disabled"},
            comments={
                "status": "extracted",
                "items": [
                    {
                        "id": "comment-1",
                        "text": "你怎麼判斷 8 分鐘伏筆和 82 分鐘 payoff 有連上？是依據抽幀還是 dB？",
                    }
                ],
            },
        )

        self.assertEqual("method_question_from_comments", payload["uncertainties"][0]["type"])
        self.assertEqual("comment-1", payload["uncertainties"][0]["comment_id"])
        self.assertIn(
            "review_comment_method_challenge",
            [item["action"] for item in payload["next_run_adjustments"]],
        )

    def test_low_confidence_retained_visual_is_reported(self) -> None:
        payload = reflection.build_run_reflection(
            transcript={"source": "subtitle_manual", "segments": [], "full_text": "Transcript"},
            visuals={
                "slides": [
                    {
                        "segment_id": "segment-0001",
                        "heuristic_confidence": 0.42,
                        "start_seconds": 30.0,
                        "end_seconds": 45.0,
                    }
                ],
                "charts": [],
            },
            visual_sampling={
                "status": "extracted",
                "density": "dense",
                "interval_seconds": 15,
                "candidate_frame_count": 10,
                "retained_visual_count": 1,
            },
            audio_features={"status": "disabled"},
            comments={"status": "disabled", "items": []},
        )

        self.assertEqual("low_confidence_retained_visual", payload["uncertainties"][0]["type"])
        self.assertEqual("segment-0001", payload["uncertainties"][0]["segment_id"])
        self.assertIn(
            "inspect_debug_artifacts",
            [item["action"] for item in payload["next_run_adjustments"]],
        )


class SparseTranscriptTests(unittest.TestCase):
    def reflect(self, transcript: dict | None, duration: object = 189.0) -> dict:
        return reflection.build_run_reflection(
            transcript=transcript,
            media_duration_seconds=duration,
            visuals=None,
            visual_sampling=None,
            audio_features=None,
            comments=None,
        )

    def sparse_signals(self, payload: dict) -> list[dict]:
        return [item for item in payload["uncertainties"] if item["type"] == "sparse_transcript"]

    def test_sparse_result_contains_observations_without_changing_transcript(self) -> None:
        transcript = {
            "source": "whisper",
            "full_text": "Went to check this out.",
            "segments": [{"start": 120.0, "end": 134.36, "text": "Went to check this out."}],
            "segment_count": 99,
        }
        before = deepcopy(transcript)

        payload = self.reflect(transcript)

        signal, = self.sparse_signals(payload)
        self.assertEqual(189.0, signal["media_duration_seconds"])
        self.assertEqual(18, signal["text_character_count"])
        self.assertEqual(1, signal["nonempty_segment_count"])
        self.assertEqual(["transcript"], signal["evidence_layers"])
        self.assertEqual("limited", payload["status"])
        self.assertIn("sparse_transcript", payload["summary"]["limitations"])
        self.assertNotIn("transcript_unavailable", payload["summary"]["limitations"])
        self.assertEqual(["assess_transcript_sufficiency"],
                         [item["action"] for item in payload["next_run_adjustments"]])
        self.assertEqual([], payload["skill_patch_candidates"])
        self.assertEqual(before, transcript)

    def test_duration_and_character_boundaries(self) -> None:
        for duration, characters, expected in (
            (119.999, 79, False), (120.0, 79, True),
            (120.0, 80, False), (3600.0, 80, False),
        ):
            with self.subTest(duration=duration, characters=characters):
                payload = self.reflect({"source": "whisper", "full_text": "x" * characters}, duration)
                self.assertEqual(expected, bool(self.sparse_signals(payload)))

    def test_unicode_count_ignores_spacing_and_punctuation_and_normalizes_accents(self) -> None:
        payload = self.reflect({"source": "subtitle_manual", "text": "中文 日本語 café e\u0301 12!\n"})

        signal, = self.sparse_signals(payload)
        self.assertEqual(12, signal["text_character_count"])
        self.assertEqual(0, signal["nonempty_segment_count"])
        self.assertIn("transcript", payload["summary"]["available_evidence_layers"])
        self.assertNotIn("transcript_unavailable", payload["summary"]["limitations"])

    def test_text_fallbacks_and_counts_do_not_double_count_full_text(self) -> None:
        cases = [
            {"full_text": "x" * 79, "segments": [{"text": "x" * 79}]},
            {"full_text": " \n", "text": "x" * 79},
            {"full_text": " \n", "segments": [{"text": "x" * 79}, {"text": None}, {"text": "  "}]},
            {"segments": [{"text": "x" * 79}]},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                payload = self.reflect({"source": "whisper", **fields})
                signal, = self.sparse_signals(payload)
                self.assertEqual(79, signal["text_character_count"])
                self.assertEqual(1 if fields.get("segments") else 0, signal["nonempty_segment_count"])

    def test_long_single_segment_and_many_short_segments_are_not_sparse(self) -> None:
        cases = [
            {"segments": [{"start": 0, "end": 189, "text": "x" * 80}]},
            {"segments": [{"text": "x" * 10} for _ in range(8)]},
            {"full_text": "x" * 80, "segments": [{"text": "short"}]},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                self.assertEqual([], self.sparse_signals(self.reflect({"source": "whisper", **fields})))

    def test_missing_skipped_and_empty_transcripts_are_unavailable_not_sparse(self) -> None:
        for transcript in (
            None, {}, {"source": "skipped", "full_text": "old text"},
            {"source": "whisper", "full_text": " \n", "segments": [{"text": None}, {"text": " "}]},
        ):
            with self.subTest(transcript=transcript):
                payload = self.reflect(transcript)
                self.assertEqual([], self.sparse_signals(payload))
                self.assertIn("transcript_unavailable", payload["summary"]["limitations"])
                self.assertNotIn("transcript", payload["summary"]["available_evidence_layers"])

    def test_invalid_or_unknown_duration_does_not_trigger_warning(self) -> None:
        for duration in (None, "", "unknown", -1, 0, True, float("nan"), float("inf"),
                         float("-inf"), "NaN", "Infinity", 10 ** 400, {}, []):
            with self.subTest(duration=duration):
                payload = self.reflect({"source": "whisper", "full_text": "short"}, duration)
                self.assertEqual([], self.sparse_signals(payload))

    def test_punctuation_only_and_reused_evidence_remain_advisory(self) -> None:
        for source in ("whisper", "subtitle_manual", "subtitle_auto", "burned_subtitle_ocr", "openai"):
            with self.subTest(source=source):
                payload = self.reflect({
                    "source": source, "full_text": "...!", "provenance": {"status": "reused"},
                }, "120.5")
                signal, = self.sparse_signals(payload)
                self.assertEqual(0, signal["text_character_count"])
                self.assertEqual(120.5, signal["media_duration_seconds"])


if __name__ == "__main__":
    unittest.main()
