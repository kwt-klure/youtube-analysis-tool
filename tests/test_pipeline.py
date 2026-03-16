from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from youtube_analysis_tool.pipeline import (
    analysis_paths,
    cleanup_intermediate_artifacts,
    choose_subtitle_file,
    choose_subtitle_track_from_metadata,
    default_ocr_state,
    default_output_dir_for_source,
    download_youtube_media,
    download_youtube_subtitles,
    duration_seconds,
    extract_interval_keyframes,
    parse_args,
    preferred_subtitle_languages,
    parse_srt_or_vtt,
    run_ocr_stage,
    transcript_strategy_auto,
    transcript_from_segments,
)


class SubtitleParsingTests(unittest.TestCase):
    def test_parse_vtt_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path = Path(tmpdir) / "sample.en.vtt"
            subtitle_path.write_text(
                "WEBVTT\n\n"
                "00:00:00.000 --> 00:00:02.000\n"
                "Hello world\n\n"
                "00:00:02.500 --> 00:00:04.000\n"
                "<i>Second line</i>\n",
                encoding="utf-8",
            )
            segments = parse_srt_or_vtt(subtitle_path)

        self.assertEqual(2, len(segments))
        self.assertEqual("Hello world", segments[0]["text"])
        self.assertEqual("Second line", segments[1]["text"])

    def test_choose_subtitle_file_prefers_chinese_then_english(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "video.en.vtt").write_text("", encoding="utf-8")
            (root / "video.zh-tw.vtt").write_text("", encoding="utf-8")
            choice = choose_subtitle_file(root)

        self.assertIsNotNone(choice)
        self.assertEqual("video.zh-tw.vtt", choice.name)

    def test_transcript_from_segments_keeps_text_and_count(self) -> None:
        transcript = transcript_from_segments(
            [{"start": 0.0, "end": 1.0, "text": "A"}, {"start": 1.0, "end": 2.0, "text": "B"}],
            source="subtitle",
            language="en",
        )

        self.assertEqual("A\nB", transcript["text"])
        self.assertEqual(2, transcript["segment_count"])


class OutputPathTests(unittest.TestCase):
    def test_local_file_uses_stem(self) -> None:
        output = default_output_dir_for_source("/tmp/My Demo Video.mp4")
        self.assertEqual(Path("output/youtube/my-demo-video"), output)

    def test_url_with_id_uses_id(self) -> None:
        output = default_output_dir_for_source("https://youtu.be/abc123", "AbC_123")
        self.assertEqual(Path("output/youtube/abc-123"), output)


class MetadataShapeTests(unittest.TestCase):
    def test_duration_seconds_supports_ffprobe_shape(self) -> None:
        self.assertEqual(12.5, duration_seconds({"format": {"duration": "12.5"}}))

    def test_duration_seconds_supports_ytdlp_shape(self) -> None:
        self.assertEqual(42.0, duration_seconds({"duration": 42}))

    def test_extract_interval_keyframes_skips_exact_end_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keyframes_dir = Path(tmpdir)
            calls = []

            def fake_run_command(command, **kwargs):
                del kwargs
                calls.append(command)
                target = Path(command[-1])
                target.write_bytes(b"frame")
                return None

            with mock.patch("youtube_analysis_tool.pipeline.run_command", side_effect=fake_run_command):
                rows = extract_interval_keyframes(
                    Path("/tmp/demo.mp4"),
                    keyframes_dir,
                    duration=180.0,
                    interval_seconds=60,
                    existing_rows=[],
                )

        self.assertEqual([0, 60, 120], [int(row["timestamp_seconds"]) for row in rows])
        self.assertEqual(3, len(calls))


class CliArgumentTests(unittest.TestCase):
    def test_new_cli_flags_have_expected_defaults(self) -> None:
        args = parse_args(["--source", "/tmp/demo.mp4"])

        self.assertEqual("auto", args.ocr)
        self.assertEqual("on", args.triage)
        self.assertEqual("off", args.gpt)
        self.assertEqual("interactive", args.review)
        self.assertEqual("gpt-5.4", args.gpt_model)
        self.assertEqual("zh-TW", args.report_language)
        self.assertFalse(args.review_reset)
        self.assertFalse(args.keep_intermediates)


class TranscriptPolicyTests(unittest.TestCase):
    def test_transcript_auto_prefers_subtitles_before_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            subtitle_path = paths.subtitles_dir / "demo.zh-tw.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle", "text": "字幕"}

            with mock.patch("youtube_analysis_tool.pipeline.choose_subtitle_file", return_value=subtitle_path), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ) as subtitle_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper"
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill"
            ) as openai_mock:
                result = transcript_strategy_auto(Path("/tmp/audio.wav"), paths)

        self.assertEqual(transcript, result)
        subtitle_mock.assert_called_once()
        whisper_mock.assert_not_called()
        openai_mock.assert_not_called()

    def test_transcript_auto_falls_back_to_openai_after_whisper_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            transcript = {"source": "openai", "text": "API"}

            with mock.patch("youtube_analysis_tool.pipeline.choose_subtitle_file", return_value=None), mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper",
                side_effect=RuntimeError("whisper unavailable"),
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill",
                return_value=transcript,
            ) as openai_mock:
                result = transcript_strategy_auto(Path("/tmp/audio.wav"), paths)

        self.assertEqual(transcript, result)
        whisper_mock.assert_called_once()
        openai_mock.assert_called_once()


class OcrModeTests(unittest.TestCase):
    def test_default_ocr_state_uses_requested_mode(self) -> None:
        state = default_ocr_state("auto")
        self.assertEqual("auto", state["mode"])
        self.assertEqual("not_attempted", state["status"])

    def test_run_ocr_stage_off_skips_without_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            rows, state = run_ocr_stage(paths, [{"filename": "frame.jpg"}], ocr_mode="off")

        self.assertEqual([], rows)
        self.assertFalse(state["attempted"])
        self.assertEqual("disabled", state["status"])

    def test_run_ocr_stage_auto_records_failure_non_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            with mock.patch("youtube_analysis_tool.pipeline.run_ocr", side_effect=RuntimeError("tesseract missing")):
                rows, state = run_ocr_stage(paths, [{"filename": "frame.jpg"}], ocr_mode="auto")

        self.assertEqual([], rows)
        self.assertTrue(state["attempted"])
        self.assertEqual("failed", state["status"])
        self.assertIn("tesseract missing", state["error"])

    def test_run_ocr_stage_on_raises_when_ocr_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            with mock.patch("youtube_analysis_tool.pipeline.run_ocr", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    run_ocr_stage(paths, [{"filename": "frame.jpg"}], ocr_mode="on")


class YoutubeDownloadFallbackTests(unittest.TestCase):
    def test_preferred_subtitle_languages_follow_constants(self) -> None:
        self.assertEqual("zh-tw", preferred_subtitle_languages()[0])

    def test_choose_subtitle_track_from_metadata_prefers_language_and_vtt(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "subtitles": {
                    "en": [{"ext": "srt", "url": "https://example.com/en.srt"}],
                    "zh-TW": [
                        {"ext": "json3", "url": "https://example.com/zh.json3"},
                        {"ext": "vtt", "url": "https://example.com/zh.vtt"},
                    ],
                }
            }
        )

        self.assertIsNotNone(selection)
        language, item = selection
        self.assertEqual("zh-TW", language)
        self.assertEqual("vtt", item["ext"])

    def test_download_youtube_subtitles_uses_narrow_official_subtitle_policy(self) -> None:
        captured = {}

        class FakeYoutubeDL:
            def __init__(self, opts):
                captured["opts"] = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                del url, download
                return {}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL):
                download_youtube_subtitles("https://youtu.be/demo", paths)

        opts = captured["opts"]
        self.assertTrue(opts["writesubtitles"])
        self.assertFalse(opts["writeautomaticsub"])
        self.assertEqual(preferred_subtitle_languages(), opts["subtitleslangs"])

    def test_download_youtube_media_continues_when_subtitle_download_fails(self) -> None:
        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                del url, download
                if self.opts.get("skip_download"):
                    raise RuntimeError("subtitle 429")
                default_template = self.opts["outtmpl"]["default"]
                video_path = Path(default_template.replace("%(ext)s", "mp4"))
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"video")
                return {"id": "demo-video"}

        class FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return self.payload

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "id": "demo-video",
                "subtitles": {
                    "zh-TW": [{"ext": "vtt", "url": "https://example.com/demo.vtt"}]
                },
            }
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL), mock.patch(
                "youtube_analysis_tool.pipeline.urlopen",
                return_value=FakeResponse(b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n"),
            ):
                info, video_path = download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    metadata_hint=metadata,
                )
            self.assertEqual("demo-video", info["id"])
            self.assertTrue(video_path.exists())
            subtitle_path = paths.subtitles_dir / "demo-video.zh-TW.vtt"
            self.assertTrue(subtitle_path.exists())


class CleanupTests(unittest.TestCase):
    def test_cleanup_intermediate_artifacts_removes_media_like_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            for directory in (
                paths.audio_dir,
                paths.video_dir,
                paths.subtitles_dir,
                paths.keyframes_dir,
                paths.ocr_dir,
                paths.root / "tmp-whisper",
            ):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "artifact.bin").write_bytes(b"x")

            cleanup_intermediate_artifacts(paths)

            for directory in (
                paths.audio_dir,
                paths.video_dir,
                paths.subtitles_dir,
                paths.keyframes_dir,
                paths.ocr_dir,
                paths.root / "tmp-whisper",
            ):
                self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
