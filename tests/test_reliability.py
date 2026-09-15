from __future__ import annotations

import json
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from youtube_analysis_tool import batch, pipeline as p, triage
from youtube_analysis_tool.bundle_check import check_bundle


class InputAndCleanupTests(unittest.TestCase):
    def test_reuse_overlap_preserves_original_before_any_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "out"
            output.mkdir()
            original = output / "output.json"
            original.write_text(json.dumps({"transcript": {"text": "original"},
                                            "processing": {"run_status": "completed"}, "errors": []}))
            before = original.read_bytes()
            with mock.patch.object(p, "materialize_local_input") as media:
                with self.assertRaisesRegex(ValueError, "overlap"):
                    p.analyze_source(str(root / "source.mp4"), out_dir=output,
                                     reuse_transcript=original)
            media.assert_not_called()
            self.assertEqual(before, original.read_bytes())

    def test_local_source_inside_output_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source = output / "video" / "source.mp4"
            source.parent.mkdir()
            source.write_bytes(b"keep this input")
            with self.assertRaisesRegex(ValueError, "overlap"):
                p.analyze_source(str(source), out_dir=output, transcript_mode="off")
            self.assertEqual(b"keep this input", source.read_bytes())

    def test_bad_reuse_fails_before_media_or_output_mutation(self):
        invalid = [
            {}, {"text": 123}, {"text": "x", "segments": {}},
            {"segments": [{"start": 2, "end": 1, "text": "x"}]},
            {"segments": [{"start": float("nan"), "end": 1, "text": "x"}]},
            {"transcript": {"text": "partial"}, "processing": {"run_status": "failed"}, "errors": []},
            {"transcript": {"text": "partial"}},
            {"source": "whisper", "status": "failed", "text": "partial failed ASR"},
            {"source": "whisper", "status": "aborted", "text": "interrupted ASR"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, payload in enumerate(invalid):
                with self.subTest(payload=payload):
                    reuse = root / "reuse.json"
                    reuse.write_text(json.dumps(payload))
                    output = root / str(index)
                    with mock.patch.object(p, "materialize_local_input") as media:
                        with self.assertRaises(ValueError):
                            p.analyze_source(str(root / "source.mp4"), out_dir=output,
                                             reuse_transcript=reuse)
                    media.assert_not_called()
                    self.assertFalse(output.exists())

    def test_reused_bundle_source_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reuse = root / "reuse.json"
            reuse.write_text(json.dumps({"transcript": {"text": "wrong source"},
                                        "processing": {"run_status": "completed"}, "errors": [],
                                        "source": {"input": str(root / "other.mp4")}}))
            with self.assertRaisesRegex(ValueError, "source"):
                p.analyze_source(str(root / "requested.mp4"), out_dir=root / "out",
                                 reuse_transcript=reuse)

    def test_raw_reuse_without_identity_is_explicitly_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reuse = root / "reuse.json"
            reuse.write_text(json.dumps({"text": "raw text"}))
            transcript = p.load_reused_transcript(reuse, p.analysis_paths(root / "out"))
            self.assertEqual("unverified", transcript["reuse_source_identity"])

    def test_reuse_matches_youtube_alias_but_preserves_prior_uncertainty(self):
        payload = {"transcript": {"text": "text", "provenance": {}},
                   "processing": {"run_status": "completed"}, "errors": [],
                   "source": {"input": "https://www.youtube.com/watch?v=abcdefghijk"}}
        normalized = p.normalize_reused_transcript_payload(payload, Path("reuse.json"),
                                                          source_input="https://youtu.be/abcdefghijk?t=2")
        self.assertEqual("matched", normalized["reuse_source_identity"])
        payload["transcript"]["provenance"]["reuse_source_identity"] = "unverified"
        normalized = p.normalize_reused_transcript_payload(payload, Path("reuse.json"),
                                                          source_input="https://youtu.be/abcdefghijk")
        self.assertEqual("unverified", normalized["reuse_source_identity"])

    def test_legacy_relative_source_cannot_claim_identity_match(self):
        payload = {"transcript": {"text": "text"}, "processing": {"run_status": "completed"},
                   "errors": [], "source": {"input": "clip.mp4"}}
        normalized = p.normalize_reused_transcript_payload(payload, Path("reuse.json"), source_input="clip.mp4")
        self.assertEqual("unverified", normalized["reuse_source_identity"])
        payload["source"]["resolved_input"] = str(Path("other/clip.mp4").resolve())
        with self.assertRaisesRegex(ValueError, "source"):
            p.normalize_reused_transcript_payload(payload, Path("reuse.json"), source_input="clip.mp4")

    def test_unlink_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.mp4"
            source.write_bytes(b"input")
            output = root / "out"
            real_unlink = os.unlink

            def materialize(source_path, paths):
                media = paths.video_dir / "source.mp4"
                media.write_bytes(b"media")
                return {"streams": [{"codec_type": "video"}]}, media

            def deny_unlink(path, *args, **kwargs):
                if str(path) == "source.mp4" and kwargs.get("dir_fd") is not None:
                    raise PermissionError("injected unlink failure")
                return real_unlink(path, *args, **kwargs)

            with mock.patch.object(p, "materialize_local_input", side_effect=materialize), \
                    mock.patch("os.unlink", side_effect=deny_unlink):
                with self.assertRaisesRegex(RuntimeError, "cleanup"):
                    p.analyze_source(str(source), out_dir=output, transcript_mode="off")
            bundle = json.loads((output / "output.json").read_text())
            self.assertEqual("failed", bundle["processing"]["run_status"])
            self.assertFalse(bundle["processing"]["cleanup_applied"])
            self.assertTrue((output / "video" / "source.mp4").exists())
            self.assertFalse(check_bundle(output / "output.json")["valid"])


class BatchCompletionTests(unittest.TestCase):
    def test_invalid_and_legacy_bundles_are_not_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.json"
            for payload in [[], {"transcript": {"source": "whisper"}, "errors": []}]:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload))
                    self.assertFalse(batch.is_completed_output_bundle(path))

    def test_unreadable_cache_does_not_abort_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = root / "sources.txt"
            sources.write_text("/example/one.mp4\n/example/two.mp4\n")
            with mock.patch.object(batch, "find_existing_output", side_effect=[OSError("unreadable"), None]), \
                    mock.patch.object(batch, "analyze_source", return_value=root / "out") as run, \
                    mock.patch.object(batch, "batch_report_path", return_value=root / "report.json"), \
                    mock.patch("builtins.print"):
                self.assertEqual(0, batch.main(["--source-list", str(sources), "--root", str(root)]))
            self.assertEqual(2, run.call_count)


class VisualEvidenceTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV optional dependency")
    def test_same_layout_different_text_survives_prepass_and_triage(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            paths = p.analysis_paths(Path(tmp))
            p.ensure_dirs(paths)
            rows = []
            for index, text in enumerate(["Revenue increased 100 percent", "Revenue decreased 90 percent"]):
                image = np.full((720, 1280), 255, dtype=np.uint8)
                cv2.putText(image, text, (55, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
                name = f"slide-{index}.png"
                cv2.imwrite(str(paths.keyframes_dir / name), image)
                rows.append({"filename": name, "timestamp_seconds": index * 60,
                             "timestamp_hms": p.hms_from_seconds(index * 60), "kind": "scene"})
            selection = p.select_keyframes_for_processing(rows, paths, max_frames=36)
            self.assertEqual(2, selection.selected_count)
            frames = triage.build_frame_records(paths.root, rows, [])
            self.assertTrue(all(frame["is_duplicate_representative"] for frame in frames))

    def test_uneven_candidates_cover_elapsed_time(self):
        times = list(range(100)) + [900, 1800, 2700, 3599]
        indices = p.time_spaced_indices(times, 5)
        self.assertEqual([0, 900, 1800, 2700, 3599], [times[index] for index in indices])


class AcquisitionTests(unittest.TestCase):
    def test_subtitle_only_run_does_not_fetch_media_or_asr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def subtitles(url, paths, **kwargs):
                (paths.subtitles_dir / "video.en.manual.vtt").write_text(
                    "WEBVTT\n\n00:00.000 --> 00:01.000\nUsable captions\n")
            with mock.patch.object(p, "fetch_youtube_metadata", return_value={"id": "abcdefghijk"}), \
                    mock.patch.object(p, "download_youtube_subtitles", side_effect=subtitles), \
                    mock.patch.object(p, "_run_yt_dlp_extract_info", side_effect=AssertionError("unexpected media")), \
                    mock.patch.object(p, "extract_audio", side_effect=AssertionError("unexpected audio")):
                p.analyze_source("https://youtu.be/abcdefghijk", out_dir=root / "out")
            bundle = json.loads((root / "out" / "output.json").read_text())
            self.assertEqual("completed", bundle["processing"]["run_status"])
            self.assertIn("Usable captions", bundle["transcript"]["full_text"])

    def test_burned_fallback_does_not_fetch_video_when_captions_suffice(self):
        with tempfile.TemporaryDirectory() as tmp:
            def subtitles(url, paths, **kwargs):
                (paths.subtitles_dir / "video.en.manual.vtt").write_text(
                    "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nUsable captions\n")
            with mock.patch.object(p, "fetch_youtube_metadata", return_value={"id": "abcdefghijk"}), \
                    mock.patch.object(p, "download_youtube_subtitles", side_effect=subtitles), \
                    mock.patch.object(p, "_run_yt_dlp_extract_info", side_effect=AssertionError("unneeded video")):
                p.analyze_source("https://youtu.be/abcdefghijk", out_dir=Path(tmp) / "out",
                                 burned_subtitles_mode="auto")

    def test_burned_fallback_gets_video_when_captions_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = p.analysis_paths(Path(tmp))
            p.ensure_dirs(paths)
            def download(options, url, **kwargs):
                self.assertIn("merge_output_format", options)
                (paths.video_dir / "source.mp4").write_bytes(b"video")
                return {}
            with mock.patch.object(p, "download_youtube_subtitles"), \
                    mock.patch.object(p, "download_subtitle_from_metadata"), \
                    mock.patch.object(p, "_run_yt_dlp_extract_info", side_effect=download):
                _, media = p.download_youtube_media("https://youtu.be/abcdefghijk", paths,
                                                   media_kind="transcript", transcript_fallback_kind="video")
            self.assertEqual(paths.video_dir, media.parent)

    def test_download_selects_audio_when_captions_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = p.analysis_paths(Path(tmp))
            p.ensure_dirs(paths)
            def download(options, url, **kwargs):
                self.assertEqual("bestaudio/best", options["format"])
                target = Path(options["outtmpl"]["default"].replace("%(ext)s", "m4a"))
                target.write_bytes(b"audio")
                return {"id": "abcdefghijk"}
            with mock.patch.object(p, "download_youtube_subtitles"), \
                    mock.patch.object(p, "download_subtitle_from_metadata"), \
                    mock.patch.object(p, "_run_yt_dlp_extract_info", side_effect=download):
                _, media = p.download_youtube_media("https://youtu.be/abcdefghijk", paths,
                                                   media_kind="transcript")
            self.assertEqual(paths.audio_dir, media.parent)

    def test_requested_visual_lane_still_downloads_video_with_captions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(p, "fetch_youtube_metadata", return_value={"id": "abcdefghijk"}), \
                    mock.patch.object(p, "download_youtube_media", return_value=({}, root / "video.mp4")) as download, \
                    mock.patch.object(p, "create_keyframes", return_value=[]):
                p.analyze_source("https://youtu.be/abcdefghijk", out_dir=root / "out",
                                 visuals_mode="on", transcript_mode="off")
            self.assertEqual("video", download.call_args.kwargs["media_kind"])


if __name__ == "__main__":
    unittest.main()
