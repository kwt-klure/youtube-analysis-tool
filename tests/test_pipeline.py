from __future__ import annotations

import json
import io
import os
import tempfile
import unittest
from pathlib import Path
import sys
from unittest import mock

import numpy


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from youtube_analysis_tool.pipeline import (
    analyze_source,
    analysis_paths,
    burned_subtitle_detection_roi,
    burned_subtitle_quality_is_insufficient,
    burned_subtitle_text_metrics,
    cleanup_intermediate_artifacts,
    choose_subtitle_file,
    choose_burned_subtitle_tesseract_languages,
    choose_subtitle_track_from_metadata,
    default_ocr_state,
    default_output_dir_for_source,
    download_subtitle_from_metadata,
    download_youtube_media,
    download_youtube_subtitles,
    duration_seconds,
    extract_interval_keyframes,
    fetch_youtube_metadata,
    find_dotenv_path,
    load_dotenv_file,
    load_local_env,
    parse_args,
    parse_subtitle_file,
    preferred_subtitle_languages,
    parse_srt_or_vtt,
    preprocess_burned_subtitle_image,
    run_ocr_stage,
    main,
    transcribe_burned_subtitles,
    transcript_from_subtitles,
    transcript_strategy_auto,
    transcript_from_segments,
    is_effective_burned_subtitle_text,
)
from youtube_analysis_tool import constants


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

        self.assertEqual("off", args.visuals)
        self.assertEqual("auto", args.ocr)
        self.assertEqual("off", args.burned_subtitles)
        self.assertEqual("on", args.triage)
        self.assertEqual("off", args.gpt)
        self.assertEqual("interactive", args.review)
        self.assertEqual("gpt-5.4", args.gpt_model)
        self.assertEqual("zh-TW", args.report_language)
        self.assertEqual("minimal", args.artifacts)
        self.assertFalse(args.review_reset)
        self.assertFalse(args.keep_intermediates)

    def test_main_wires_progress_to_stderr_without_polluting_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()

        def fake_analyze_source(*args, **kwargs):
            progress_callback = kwargs.get("progress_callback")
            self.assertIsNotNone(progress_callback)
            progress_callback("transcript", "Running local Whisper transcription")
            return Path("/tmp/out")

        with mock.patch("youtube_analysis_tool.pipeline.analyze_source", side_effect=fake_analyze_source), mock.patch(
            "sys.stdout",
            stdout,
        ), mock.patch("sys.stderr", stderr):
            exit_code = main(["--source", "/tmp/demo.mp4"])

        self.assertEqual(0, exit_code)
        self.assertIn("[transcript] Running local Whisper transcription", stderr.getvalue())
        self.assertEqual("/tmp/out\n", stdout.getvalue())


class DotenvLoadingTests(unittest.TestCase):
    def test_find_dotenv_path_walks_up_parents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "a" / "b"
            nested.mkdir(parents=True)
            dotenv_path = root / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")

            found = find_dotenv_path(nested)

        self.assertEqual(dotenv_path.resolve(), found.resolve())

    def test_load_dotenv_file_sets_missing_env_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text(
                '# comment\nexport OPENAI_API_KEY="from-dotenv"\nEMPTY_OK=\n',
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                loaded = load_dotenv_file(dotenv_path)
                self.assertEqual("from-dotenv", os.environ["OPENAI_API_KEY"])
                self.assertEqual("", os.environ["EMPTY_OK"])
                self.assertEqual("from-dotenv", loaded["OPENAI_API_KEY"])

    def test_load_local_env_does_not_override_exported_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
            nested = root / "work"
            nested.mkdir()

            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "already-set"}, clear=True):
                loaded = load_local_env(nested)

                self.assertEqual({}, loaded)
                self.assertEqual("already-set", os.environ["OPENAI_API_KEY"])


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
                "status": "fallback_to_whisper_quality",
                "attempted": True,
                "probe_passed": True,
                "reason": "fallback_to_whisper_quality",
                "probe_hits": 3,
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
                average_cjk_ratio=0.45,
            )
        )
        self.assertFalse(
            burned_subtitle_quality_is_insufficient(
                ocr_event_count=12,
                nonempty_hits=8,
                cjk_char_count=140,
                average_cjk_ratio=0.6,
            )
        )

    def test_burned_subtitle_text_metrics_penalize_garbage_cjk_mix(self) -> None:
        metrics = burned_subtitle_text_metrics("中A!@#")

        self.assertEqual(5.0, metrics["visible_char_count"])
        self.assertEqual(1.0, metrics["cjk_char_count"])
        self.assertLess(metrics["cjk_ratio"], 0.45)
        self.assertGreater(metrics["noise_ratio"], 0.35)
        self.assertFalse(is_effective_burned_subtitle_text("中A!@#"))

    def test_choose_burned_subtitle_languages_requires_both_chinese_packs(self) -> None:
        with mock.patch(
            "youtube_analysis_tool.pipeline.available_tesseract_languages",
            return_value={"chi_sim", "eng"},
        ):
            self.assertIsNone(choose_burned_subtitle_tesseract_languages())

        with mock.patch(
            "youtube_analysis_tool.pipeline.available_tesseract_languages",
            return_value={"chi_sim", "chi_tra", "eng"},
        ):
            self.assertEqual("chi_sim+chi_tra+eng", choose_burned_subtitle_tesseract_languages())

    def test_detection_roi_is_smaller_than_full_subtitle_band(self) -> None:
        image = numpy.zeros((160, 900), dtype=numpy.uint8)

        roi = burned_subtitle_detection_roi(image)

        self.assertEqual((96, 630), roi.shape)

    def test_preprocess_burned_subtitle_image_only_upscales_small_inputs(self) -> None:
        small = numpy.zeros((120, 240), dtype=numpy.uint8)
        large = numpy.zeros((220, 240), dtype=numpy.uint8)

        small_processed = preprocess_burned_subtitle_image(small)
        large_processed = preprocess_burned_subtitle_image(large)

        self.assertEqual((180, 360), small_processed.shape)
        self.assertEqual((220, 240), large_processed.shape)

    def test_transcribe_burned_subtitles_fast_rejects_bad_shorts_case(self) -> None:
        metadata = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920}]}
        frames = [
            {"timestamp_seconds": 0.0, "image": "frame-0"},
            {"timestamp_seconds": 1.0, "image": "frame-1"},
            {"timestamp_seconds": 2.0, "image": "frame-2"},
            {"timestamp_seconds": 3.0, "image": "frame-3"},
        ]
        captured_sample_fps: list[float] = []

        def fake_iter(_video_path, _metadata, *, sample_fps, duration_limit=None):
            del duration_limit
            captured_sample_fps.append(sample_fps)
            return iter(frames)

        with mock.patch(
            "youtube_analysis_tool.pipeline.iter_subtitle_band_frames",
            side_effect=fake_iter,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.preprocess_burned_subtitle_image",
            side_effect=lambda image: image,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.burned_subtitle_detection_roi",
            side_effect=lambda image: image,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.subtitle_band_diff",
            return_value=1.0,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.ocr_burned_subtitle_image",
            side_effect=["亂A!", "", "中-", ""],
        ):
            result = transcribe_burned_subtitles(Path("/tmp/demo.mp4"), metadata, tesseract_langs="chi_sim+chi_tra+eng")

        self.assertEqual([1.0], captured_sample_fps)
        self.assertEqual("fast_reject", result["status"])
        self.assertEqual(4, result["ocr_event_count"])

    def test_transcribe_burned_subtitles_auto_mode_is_more_conservative(self) -> None:
        metadata = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920}]}
        frames = [
            {"timestamp_seconds": 0.0, "image": "frame-0"},
            {"timestamp_seconds": 2.0, "image": "frame-1"},
        ]
        captured_sample_fps: list[float] = []

        def fake_iter(_video_path, _metadata, *, sample_fps, duration_limit=None):
            del duration_limit
            captured_sample_fps.append(sample_fps)
            return iter(frames)

        with mock.patch(
            "youtube_analysis_tool.pipeline.iter_subtitle_band_frames",
            side_effect=fake_iter,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.preprocess_burned_subtitle_image",
            side_effect=lambda image: image,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.burned_subtitle_detection_roi",
            side_effect=lambda image: image,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.subtitle_band_diff",
            return_value=1.0,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.ocr_burned_subtitle_image",
            side_effect=["亂A!", ""],
        ):
            result = transcribe_burned_subtitles(
                Path("/tmp/demo.mp4"),
                metadata,
                tesseract_langs="chi_sim+chi_tra+eng",
                mode="auto",
            )

        self.assertEqual([0.5], captured_sample_fps)
        self.assertEqual("fast_reject", result["status"])
        self.assertEqual(2, result["ocr_event_count"])

    def test_transcribe_burned_subtitles_can_fall_back_at_early_gate(self) -> None:
        metadata = {"streams": [{"codec_type": "video", "width": 1080, "height": 1920}]}
        frames = [
            {"timestamp_seconds": float(index), "image": f"frame-{index}"}
            for index in range(12)
        ]
        ocr_texts = ["这是字幕内容"] * 4 + ["", "", "", "", "", "", "", ""]

        with mock.patch(
            "youtube_analysis_tool.pipeline.iter_subtitle_band_frames",
            return_value=iter(frames),
        ), mock.patch(
            "youtube_analysis_tool.pipeline.preprocess_burned_subtitle_image",
            side_effect=lambda image: image,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.burned_subtitle_detection_roi",
            side_effect=lambda image: image,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.subtitle_band_diff",
            return_value=1.0,
        ), mock.patch(
            "youtube_analysis_tool.pipeline.ocr_burned_subtitle_image",
            side_effect=ocr_texts,
        ):
            result = transcribe_burned_subtitles(Path("/tmp/demo.mp4"), metadata, tesseract_langs="chi_sim+chi_tra+eng")

        self.assertEqual("fallback_to_whisper_quality", result["status"])
        self.assertEqual(12, result["ocr_event_count"])


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


class VisualsModeTests(unittest.TestCase):
    def test_visuals_off_forces_keyframes_off_and_marks_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"
            subtitle_path = output_root / "subtitles" / "demo.en.manual.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n", encoding="utf-8")
            transcript = {"source": "subtitle_manual", "language": "en", "text": "subtitle text", "segments": []}

            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=({}, source_path),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.choose_subtitle_file",
                return_value=subtitle_path,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcript_from_subtitles",
                return_value=transcript,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ) as keyframes_mock, mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts"
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="auto",
                    visuals_mode="off",
                    cleanup_intermediates=False,
                )

                written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))

        self.assertEqual("off", keyframes_mock.call_args.kwargs["mode"])
        self.assertEqual("off", written["processing"]["visuals_mode"])
        self.assertEqual([], written["visuals"]["slides"])
        self.assertEqual([], written["visuals"]["charts"])

    def test_write_empty_stage_artifacts_creates_missing_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            paths = analysis_paths(output_root)

            from youtube_analysis_tool.pipeline import write_empty_stage_artifacts

            write_empty_stage_artifacts(paths)

            self.assertTrue(paths.triage_frames_path.exists())
            self.assertTrue(paths.triage_segments_path.exists())
            self.assertTrue(paths.review_queue_path.exists())
            self.assertTrue(paths.review_decisions_path.exists())
            self.assertTrue(paths.routing_manifest_path.exists())


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
        self.assertIn("socket_timeout", opts)

    def test_fetch_youtube_metadata_retries_once_then_succeeds(self) -> None:
        attempts = {"count": 0}
        events = []
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
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("metadata stalled")
                return {"id": "demo-video"}

        with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL), mock.patch(
            "youtube_analysis_tool.pipeline.time.sleep",
            return_value=None,
        ):
            metadata = fetch_youtube_metadata(
                "https://youtu.be/demo",
                progress_callback=lambda phase, message: events.append((phase, message)),
            )

        self.assertEqual("demo-video", metadata["id"])
        self.assertEqual(2, attempts["count"])
        self.assertEqual("metadata", events[-1][0])
        self.assertIn("retrying (2/2)", events[-1][1])
        self.assertEqual(
            constants.DEFAULT_YTDLP_METADATA_SOCKET_TIMEOUT_SECONDS,
            captured["opts"]["socket_timeout"],
        )

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

    def test_download_youtube_media_retries_download_once_then_succeeds(self) -> None:
        download_attempts = {"count": 0}
        events = []
        video_opts = {}

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                if "default" in self.opts.get("outtmpl", {}):
                    video_opts.update(opts)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                del url, download
                if self.opts.get("skip_download"):
                    return {}
                download_attempts["count"] += 1
                if download_attempts["count"] == 1:
                    raise RuntimeError("download stalled")
                for hook in self.opts.get("progress_hooks", []):
                    hook({"status": "downloading"})
                default_template = self.opts["outtmpl"]["default"]
                video_path = Path(default_template.replace("%(ext)s", "mp4"))
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"video")
                for hook in self.opts.get("progress_hooks", []):
                    hook({"status": "finished"})
                return {"id": "demo-video"}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL), mock.patch(
                "youtube_analysis_tool.pipeline.time.sleep",
                return_value=None,
            ):
                info, video_path = download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    metadata_hint={"id": "demo-video"},
                    progress_callback=lambda phase, message: events.append((phase, message)),
                )

            self.assertEqual("demo-video", info["id"])
            self.assertTrue(video_path.exists())

        self.assertEqual(2, download_attempts["count"])
        self.assertEqual(("transcript", "Fetching subtitle tracks"), events[0])
        self.assertTrue(any(message == "Starting source media download" for _, message in events))
        self.assertTrue(any(phase == "download" and "Media transfer finished" in message for phase, message in events))
        self.assertTrue(any("retrying (2/2)" in message for _, message in events))
        self.assertTrue(any("Media transfer started" in message for _, message in events))
        self.assertEqual(
            constants.DEFAULT_YTDLP_MEDIA_SOCKET_TIMEOUT_SECONDS,
            video_opts["socket_timeout"],
        )

    def test_download_youtube_media_reports_metadata_subtitle_fallback_before_download(self) -> None:
        events = []

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
                    raise RuntimeError("subtitle fetch failed")
                for hook in self.opts.get("progress_hooks", []):
                    hook({"status": "downloading"})
                default_template = self.opts["outtmpl"]["default"]
                video_path = Path(default_template.replace("%(ext)s", "mp4"))
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"video")
                for hook in self.opts.get("progress_hooks", []):
                    hook({"status": "finished"})
                return {"id": "demo-video"}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL), mock.patch(
                "youtube_analysis_tool.pipeline.download_subtitle_from_metadata",
                return_value=None,
            ):
                download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    metadata_hint={"id": "demo-video"},
                    progress_callback=lambda phase, message: events.append((phase, message)),
                )

        self.assertIn(("transcript", "Trying metadata subtitle fallback"), events)
        self.assertIn(("download", "Starting source media download"), events)

    def test_download_subtitle_from_metadata_uses_timeout_and_retry(self) -> None:
        call_timeouts = []

        class FakeResponse:
            def __init__(self, payload: bytes) -> None:
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return self.payload

        def fake_urlopen(url, timeout=None):
            call_timeouts.append(timeout)
            if len(call_timeouts) == 1:
                raise RuntimeError("temporary timeout")
            return FakeResponse(b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n")

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            metadata = {
                "id": "demo-video",
                "subtitles": {
                    "zh-TW": [{"ext": "vtt", "url": "https://example.com/demo.vtt"}]
                },
            }
            with mock.patch("youtube_analysis_tool.pipeline.urlopen", side_effect=fake_urlopen), mock.patch(
                "youtube_analysis_tool.pipeline.time.sleep",
                return_value=None,
            ):
                subtitle_path = download_subtitle_from_metadata(metadata, paths)

            self.assertIsNotNone(subtitle_path)
            self.assertTrue(subtitle_path.exists())

        self.assertEqual(2, len(call_timeouts))
        self.assertTrue(all(timeout is not None for timeout in call_timeouts))


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
