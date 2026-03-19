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
    analyze_source,
    analysis_paths,
    burned_subtitle_quality_is_insufficient,
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
    parse_subtitle_file,
    preferred_subtitle_languages,
    parse_srt_or_vtt,
    run_ocr_stage,
    transcript_from_subtitles,
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

    def test_parse_json3_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subtitle_path = Path(tmpdir) / "sample.ja.auto.json3"
            subtitle_path.write_text(
                (
                    '{"events": ['
                    '{"tStartMs": 0, "dDurationMs": 1200, "segs": [{"utf8": "Hello "}, {"utf8": "world"}]},'
                    '{"tStartMs": 1500, "dDurationMs": 500, "segs": [{"utf8": "Again"}]}'
                    "]} "
                ),
                encoding="utf-8",
            )
            segments = parse_subtitle_file(subtitle_path)

        self.assertEqual(2, len(segments))
        self.assertEqual("Hello world", segments[0]["text"])
        self.assertEqual(0.0, segments[0]["start"])
        self.assertEqual(1.2, segments[0]["end"])
        self.assertEqual("Again", segments[1]["text"])

    def test_transcript_from_manual_subtitles_uses_manual_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            subtitle_path = paths.subtitles_dir / "video.en.manual.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
            transcript = transcript_from_subtitles(subtitle_path, paths)

        self.assertEqual("subtitle_manual", transcript["source"])
        self.assertEqual("en", transcript["language"])

    def test_transcript_from_auto_subtitles_uses_auto_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            subtitle_path = paths.subtitles_dir / "video.ja.auto.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
            transcript = transcript_from_subtitles(subtitle_path, paths)

        self.assertEqual("subtitle_auto", transcript["source"])
        self.assertEqual("ja", transcript["language"])

    def test_choose_subtitle_file_prefers_chinese_then_english(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "video.en.vtt").write_text("", encoding="utf-8")
            (root / "video.zh-tw.vtt").write_text("", encoding="utf-8")
            choice = choose_subtitle_file(root)

        self.assertIsNotNone(choice)
        self.assertEqual("video.zh-tw.vtt", choice.name)

    def test_choose_subtitle_file_prefers_manual_before_auto_for_same_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "video.ja.auto.vtt").write_text("", encoding="utf-8")
            (root / "video.ja.manual.vtt").write_text("", encoding="utf-8")
            choice = choose_subtitle_file(root)

        self.assertIsNotNone(choice)
        self.assertEqual("video.ja.manual.vtt", choice.name)

    def test_choose_subtitle_file_accepts_json3_when_no_vtt_or_srt_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "video.ja.auto.json3").write_text("{}", encoding="utf-8")
            choice = choose_subtitle_file(root)

        self.assertIsNotNone(choice)
        self.assertEqual("video.ja.auto.json3", choice.name)

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

    def test_url_with_title_and_id_uses_title_id(self) -> None:
        output = default_output_dir_for_source("https://youtu.be/abc123", "AbC_123", "My Great Talk")
        self.assertEqual(Path("output/youtube/my-great-talk-abc-123"), output)

    def test_url_with_unicode_title_and_id_uses_title_id(self) -> None:
        output = default_output_dir_for_source(
            "https://youtu.be/8LrniR6db-k",
            "8LrniR6db-k",
            "『荒野のコトブキ飛行隊』イジツ見聞録～戦闘機編～",
        )
        self.assertEqual(
            Path("output/youtube/荒野のコトブキ飛行隊-イジツ見聞録-戦闘機編-8lrnir6db-k"),
            output,
        )

    def test_url_without_title_falls_back_to_id(self) -> None:
        output = default_output_dir_for_source("https://youtu.be/abc123", "AbC_123", None)
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
        self.assertEqual("auto", args.burned_subtitles)
        self.assertEqual("on", args.triage)
        self.assertEqual("off", args.gpt)
        self.assertEqual("interactive", args.review)
        self.assertEqual("gpt-5.4", args.gpt_model)
        self.assertEqual("zh-TW", args.report_language)
        self.assertEqual("minimal", args.artifacts)
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

    def test_whisper_mode_still_prefers_subtitles_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            subtitle_path = output_root / "subtitles" / "demo.en.manual.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle_manual", "language": "en", "text": "subtitle text", "segments": []}

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ), mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=subtitle_path,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ) as subtitle_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper"
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ) as empty_stage_mock:
                result = analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="whisper",
                    cleanup_intermediates=False,
                )

        self.assertEqual(output_root, result)
        subtitle_mock.assert_called_once()
        whisper_mock.assert_not_called()
        empty_stage_mock.assert_called_once()

    def test_whisper_mode_with_subtitles_skips_audio_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            subtitle_path = output_root / "subtitles" / "demo.en.auto.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle_auto", "language": "en", "text": "subtitle text", "segments": []}

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=subtitle_path,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper"
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="whisper",
                    cleanup_intermediates=False,
                )

        extract_audio_mock.assert_not_called()
        whisper_mock.assert_not_called()

    def test_api_mode_still_prefers_subtitles_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            subtitle_path = output_root / "subtitles" / "demo.en.auto.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle_auto", "language": "en", "text": "subtitle text", "segments": []}

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=subtitle_path,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ) as subtitle_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill"
            ) as openai_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                result = analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="api",
                    cleanup_intermediates=False,
                )

        self.assertEqual(output_root, result)
        subtitle_mock.assert_called_once()
        extract_audio_mock.assert_not_called()
        openai_mock.assert_not_called()

    def test_subtitles_mode_accepts_auto_captions_without_audio_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            subtitle_path = output_root / "subtitles" / "demo.ja.auto.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle_auto", "language": "ja", "text": "subtitle text", "segments": []}

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=subtitle_path,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ) as subtitle_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                result = analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="subtitles",
                    cleanup_intermediates=False,
                )

        self.assertEqual(output_root, result)
        subtitle_mock.assert_called_once()
        extract_audio_mock.assert_not_called()

    def test_subtitles_mode_does_not_run_burned_subtitle_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            subtitle_path = output_root / "subtitles" / "demo.ja.auto.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle_auto", "language": "ja", "text": "subtitle text", "segments": []}

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=subtitle_path,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_burned_subtitles_stage"
            ) as burned_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="subtitles",
                    cleanup_intermediates=False,
                )

        burned_mock.assert_not_called()

    def test_auto_mode_uses_burned_subtitle_ocr_before_audio_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            burned_transcript = {
                "source": "burned_subtitle_ocr",
                "language": None,
                "text": "燒錄字幕",
                "segments": [{"start": 0.0, "end": 1.0, "text": "燒錄字幕"}],
            }
            burned_state = {
                "mode": "auto",
                "status": "completed",
                "attempted": True,
                "probe_passed": True,
                "ocr_event_count": 12,
                "error": None,
            }

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({"streams": [{"codec_type": "video"}]}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=None,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_burned_subtitles_stage",
                return_value=(burned_transcript, burned_state),
            ) as burned_mock, mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcript_strategy_auto"
            ) as strategy_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="auto",
                    cleanup_intermediates=False,
                )

        burned_mock.assert_called_once()
        extract_audio_mock.assert_not_called()
        strategy_mock.assert_not_called()

    def test_whisper_mode_falls_back_to_whisper_when_burned_ocr_quality_is_poor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            whisper_transcript = {
                "source": "whisper",
                "language": "zh",
                "text": "Whisper transcript",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Whisper transcript"}],
            }
            burned_state = {
                "mode": "auto",
                "status": "fallback_to_whisper",
                "attempted": True,
                "probe_passed": True,
                "ocr_event_count": 5,
                "error": None,
            }

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({"streams": [{"codec_type": "video"}]}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=None,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_burned_subtitles_stage",
                return_value=(None, burned_state),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio",
                return_value=Path(tmpdir) / "audio.wav",
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper",
                return_value=whisper_transcript,
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="whisper",
                    cleanup_intermediates=False,
                )

        extract_audio_mock.assert_called_once()
        whisper_mock.assert_called_once()

    def test_api_mode_uses_burned_subtitle_ocr_before_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            burned_transcript = {
                "source": "burned_subtitle_ocr",
                "language": None,
                "text": "燒錄字幕",
                "segments": [{"start": 0.0, "end": 1.0, "text": "燒錄字幕"}],
            }
            burned_state = {
                "mode": "on",
                "status": "completed",
                "attempted": True,
                "probe_passed": True,
                "ocr_event_count": 9,
                "error": None,
            }

            with mock.patch("youtube_analysis_tool.pipeline.materialize_local_input", return_value=({"streams": [{"codec_type": "video"}]}, source_path)), mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=None,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_burned_subtitles_stage",
                return_value=(burned_transcript, burned_state),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill"
            ) as openai_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="api",
                    cleanup_intermediates=False,
                )

        extract_audio_mock.assert_not_called()
        openai_mock.assert_not_called()

    def test_burned_subtitle_quality_gate_detects_insufficient_signal(self) -> None:
        self.assertTrue(
            burned_subtitle_quality_is_insufficient(
                ocr_event_count=10,
                nonempty_hits=3,
                cjk_char_count=80,
            )
        )
        self.assertFalse(
            burned_subtitle_quality_is_insufficient(
                ocr_event_count=12,
                nonempty_hits=8,
                cjk_char_count=140,
            )
        )


class OcrModeTests(unittest.TestCase):
    def test_default_ocr_state_uses_requested_mode(self) -> None:
        state = default_ocr_state("auto")
        self.assertEqual("auto", state["mode"])
        self.assertEqual("not_attempted", state["status"])
        self.assertNotIn("artifact_path", state)

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

    def test_choose_subtitle_track_from_metadata_prefers_manual_before_auto(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "subtitles": {
                    "en": [{"ext": "srt", "url": "https://example.com/en.srt"}],
                },
                "automatic_captions": {
                    "zh-TW": [
                        {"ext": "json3", "url": "https://example.com/zh.json3"},
                        {"ext": "vtt", "url": "https://example.com/zh.vtt"},
                    ]
                }
            }
        )

        self.assertIsNotNone(selection)
        bucket_name, language, item = selection
        self.assertEqual("subtitles", bucket_name)
        self.assertEqual("en", language)
        self.assertEqual("srt", item["ext"])

    def test_choose_subtitle_track_from_metadata_prefers_language_and_vtt_within_bucket(self) -> None:
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
        bucket_name, language, item = selection
        self.assertEqual("subtitles", bucket_name)
        self.assertEqual("zh-TW", language)
        self.assertEqual("vtt", item["ext"])

    def test_choose_subtitle_track_from_metadata_accepts_auto_json3_when_needed(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "automatic_captions": {
                    "ja": [
                        {"ext": "srv3", "url": "https://example.com/ja.srv3"},
                        {"ext": "json3", "url": "https://example.com/ja.json3"},
                    ]
                }
            }
        )

        self.assertIsNotNone(selection)
        bucket_name, language, item = selection
        self.assertEqual("automatic_captions", bucket_name)
        self.assertEqual("ja", language)
        self.assertEqual("json3", item["ext"])

    def test_choose_subtitle_track_from_metadata_prefers_original_auto_caption_before_translated(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "automatic_captions": {
                    "zh-Hant": [
                        {
                            "ext": "vtt",
                            "url": "https://example.com/caption.vtt?tlang=zh-Hant&lang=ja",
                        }
                    ],
                    "ja": [
                        {
                            "ext": "vtt",
                            "url": "https://example.com/caption.vtt?lang=ja",
                        }
                    ],
                }
            }
        )

        self.assertIsNotNone(selection)
        bucket_name, language, item = selection
        self.assertEqual("automatic_captions", bucket_name)
        self.assertEqual("ja", language)
        self.assertNotIn("tlang=", item["url"])

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
        self.assertIn(".manual.", opts["outtmpl"]["subtitle"])

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
            subtitle_path = paths.subtitles_dir / "demo-video.zh-TW.manual.vtt"
            self.assertTrue(subtitle_path.exists())

    def test_download_youtube_media_falls_back_to_post_download_info_for_auto_captions(self) -> None:
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
                    return {}
                default_template = self.opts["outtmpl"]["default"]
                video_path = Path(default_template.replace("%(ext)s", "mp4"))
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"video")
                return {
                    "id": "demo-video",
                    "automatic_captions": {
                        "zh-Hant": [{"ext": "vtt", "url": "https://example.com/auto.vtt"}]
                    },
                }

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
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL), mock.patch(
                "youtube_analysis_tool.pipeline.urlopen",
                return_value=FakeResponse(b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n"),
            ):
                info, video_path = download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    metadata_hint={"id": "demo-video"},
                )
            self.assertEqual("demo-video", info["id"])
            self.assertTrue(video_path.exists())
            subtitle_path = paths.subtitles_dir / "demo-video.zh-Hant.auto.vtt"
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
