from __future__ import annotations

import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()
