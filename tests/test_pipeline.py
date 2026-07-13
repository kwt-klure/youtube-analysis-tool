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
    analysis_kwargs_from_args,
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
    extract_audio,
    extract_interval_keyframes,
    fetch_youtube_metadata,
    find_dotenv_path,
    load_reused_transcript,
    load_dotenv_file,
    load_local_env,
    link_local_source,
    parse_args,
    parse_subtitle_file,
    preferred_subtitle_languages,
    parse_srt_or_vtt,
    preprocess_burned_subtitle_image,
    run_ocr_stage,
    run_comments_stage,
    main,
    transcribe_burned_subtitles,
    transcribe_with_whisper,
    transcript_from_subtitles,
    transcript_strategy_auto,
    transcript_from_segments,
    is_effective_burned_subtitle_text,
    extract_audio_features,
    fetch_youtube_comments,
    youtube_format_selector,
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

    def test_transcript_from_auto_subtitles_with_translated_suffix_keeps_auto_source_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            subtitle_path = paths.subtitles_dir / "video.en-US.auto.en.vtt"
            subtitle_path.parent.mkdir(parents=True, exist_ok=True)
            subtitle_path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")
            transcript = transcript_from_subtitles(subtitle_path, paths)

        self.assertEqual("subtitle_auto", transcript["source"])
        self.assertEqual("en", transcript["language"])

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

    def test_choose_subtitle_file_treats_translated_auto_suffix_as_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "video.en-US.auto.en.vtt").write_text("", encoding="utf-8")
            (root / "video.en.manual.vtt").write_text("", encoding="utf-8")
            choice = choose_subtitle_file(root)

        self.assertIsNotNone(choice)
        self.assertEqual("video.en.manual.vtt", choice.name)

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


class AudioFeatureTests(unittest.TestCase):
    def test_extract_audio_uses_a_distinct_path_for_local_wav_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.wav"
            source_path.write_bytes(b"original audio")
            audio_dir = Path(tmpdir) / "out" / "audio"
            audio_dir.mkdir(parents=True)
            linked_source = link_local_source(source_path, audio_dir)

            with mock.patch("youtube_analysis_tool.pipeline.run_command") as run_command:
                normalized_audio = extract_audio(source_path, audio_dir)
            source_bytes_after = source_path.read_bytes()

        self.assertEqual(audio_dir / "normalized.wav", normalized_audio)
        self.assertNotEqual(linked_source, normalized_audio)
        self.assertEqual(str(normalized_audio), run_command.call_args.args[0][-1])
        self.assertEqual(b"original audio", source_bytes_after)

    def test_extract_audio_features_builds_loudness_windows_and_silence_segments(self) -> None:
        samples = numpy.array(
            ([12000] * 10) + ([0] * 20) + ([24000] * 10),
            dtype=numpy.int16,
        )

        completed = mock.Mock()
        completed.stdout = samples.tobytes()

        with mock.patch("youtube_analysis_tool.pipeline.subprocess.run", return_value=completed):
            payload = extract_audio_features(
                Path("/tmp/demo.mp4"),
                metadata={"format": {"duration": "4"}},
                sample_rate=10,
                window_seconds=1,
                silence_min_duration=1.0,
            )

        self.assertEqual("extracted", payload["status"])
        self.assertEqual("local_ffmpeg_python_rms", payload["source"])
        self.assertEqual(4.0, payload["summary"]["duration_seconds"])
        self.assertAlmostEqual(0.5, payload["summary"]["silence_ratio"])
        self.assertEqual(
            [{"start": 1.0, "end": 3.0, "duration": 2.0}],
            payload["silence_segments"],
        )
        self.assertEqual(4, len(payload["windows"]))
        self.assertLessEqual(payload["windows"][1]["rms_db"], -99.0)
        self.assertAlmostEqual(payload["summary"]["mean_loudness_db"], payload["summary"]["mean_dbfs"])
        self.assertEqual("wide_or_high_contrast", payload["summary"]["dynamic_range_hint"])
        self.assertTrue(payload["quiet_segments"])
        self.assertTrue(payload["loud_segments"])
        self.assertGreaterEqual(len(payload["large_changes"]), 2)
        self.assertEqual(0.0, payload["large_changes"][0]["start"])
        self.assertEqual(2.0, payload["large_changes"][0]["end"])
        self.assertLess(payload["large_changes"][0]["to_dbfs"], payload["large_changes"][0]["from_dbfs"])
        self.assertIn("structural evidence", payload["interpretation_warning"])
        self.assertEqual("structural_signal_not_semantics", payload["provenance"]["trust"])


class CliArgumentTests(unittest.TestCase):
    def test_version_flag_prints_project_version_and_exits(self) -> None:
        expected_version = next(
            line.partition("=")[2].strip().strip('"')
            for line in (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
            if line.startswith("version =")
        )
        stdout = io.StringIO()

        with self.assertRaises(SystemExit) as context, mock.patch("sys.stdout", stdout):
            parse_args(["--version"])

        self.assertEqual(0, context.exception.code)
        self.assertEqual(f"youtube-analysis-tool {expected_version}\n", stdout.getvalue())

    def test_new_cli_flags_have_expected_defaults(self) -> None:
        args = parse_args(["--source", "/tmp/demo.mp4"])

        self.assertEqual("default", args.intake_profile)
        self.assertIsNone(args.visuals)
        self.assertIsNone(args.visual_density)
        self.assertEqual("auto", args.ocr)
        self.assertEqual("off", args.burned_subtitles)
        self.assertIsNone(args.audio_features)
        self.assertIsNone(args.comments)
        self.assertEqual("on", args.triage)
        self.assertEqual("off", args.gpt)
        self.assertEqual("interactive", args.review)
        self.assertEqual("gpt-5.4", args.gpt_model)
        self.assertEqual("zh-TW", args.report_language)
        self.assertEqual("minimal", args.artifacts)
        self.assertFalse(args.review_reset)
        self.assertFalse(args.keep_intermediates)
        self.assertIsNone(args.max_video_height)
        self.assertEqual(constants.DEFAULT_LOCAL_ASR_BACKEND, args.local_asr_backend)

    def test_transcript_off_and_reuse_transcript_flags_parse(self) -> None:
        args = parse_args(
            [
                "--source",
                "/tmp/demo.mp4",
                "--transcript",
                "off",
                "--reuse-transcript",
                "/tmp/transcript.json",
            ]
        )

        self.assertEqual("off", args.transcript)
        self.assertEqual(Path("/tmp/transcript.json"), args.reuse_transcript)

        kwargs = analysis_kwargs_from_args(args)
        self.assertEqual("off", kwargs["transcript_mode"])
        self.assertEqual(Path("/tmp/transcript.json"), kwargs["reuse_transcript"])

    def test_max_video_height_flag_parses_and_flows_to_analysis(self) -> None:
        args = parse_args(
            ["--source", "/tmp/demo.mp4", "--max-video-height", "720"]
        )

        self.assertEqual(720, args.max_video_height)
        self.assertEqual(720, analysis_kwargs_from_args(args)["max_video_height"])

    def test_local_asr_backend_flag_parses_and_flows_to_analysis(self) -> None:
        args = parse_args(
            ["--source", "/tmp/demo.mp4", "--local-asr-backend", "mlx-whisper"]
        )

        self.assertEqual("mlx-whisper", args.local_asr_backend)
        self.assertEqual(
            "mlx-whisper",
            analysis_kwargs_from_args(args)["local_asr_backend"],
        )

    def test_default_profile_keeps_cheap_visuals_audio_and_comments_off(self) -> None:
        args = parse_args(["--source", "/tmp/demo.mp4"])

        kwargs = analysis_kwargs_from_args(args)

        self.assertEqual("default", kwargs["intake_profile"])
        self.assertEqual("off", kwargs["visuals_mode"])
        self.assertEqual("default", kwargs["visual_density"])
        self.assertEqual(constants.DEFAULT_INTERVAL_SECONDS, kwargs["interval_seconds"])
        self.assertEqual("off", kwargs["audio_features_mode"])
        self.assertEqual(0, kwargs["comments_count"])

    def test_rich_profile_sets_dense_visuals_audio_and_comments(self) -> None:
        args = parse_args(["--source", "/tmp/demo.mp4", "--intake-profile", "rich"])

        kwargs = analysis_kwargs_from_args(args)

        self.assertEqual("rich", kwargs["intake_profile"])
        self.assertEqual("on", kwargs["visuals_mode"])
        self.assertEqual("dense", kwargs["visual_density"])
        self.assertEqual(15, kwargs["interval_seconds"])
        self.assertEqual("on", kwargs["audio_features_mode"])
        self.assertEqual(5, kwargs["comments_count"])

    def test_explicit_density_and_interval_override_rich_profile(self) -> None:
        args = parse_args(
            [
                "--source",
                "/tmp/demo.mp4",
                "--intake-profile",
                "rich",
                "--visual-density",
                "medium",
                "--interval-seconds",
                "10",
                "--comments",
                "0",
                "--audio-features",
                "off",
            ]
        )

        kwargs = analysis_kwargs_from_args(args)

        self.assertEqual("medium", kwargs["visual_density"])
        self.assertEqual(10, kwargs["interval_seconds"])
        self.assertEqual("off", kwargs["audio_features_mode"])
        self.assertEqual(0, kwargs["comments_count"])

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


class LocalAsrBackendTests(unittest.TestCase):
    def test_default_dispatch_uses_openai_whisper_adapter(self) -> None:
        expected = {"source": "whisper", "text": "hello", "segments": []}

        with mock.patch(
            "youtube_analysis_tool.pipeline.transcribe_with_openai_whisper",
            return_value=expected,
        ) as openai_adapter, mock.patch(
            "youtube_analysis_tool.pipeline.transcribe_with_mlx_whisper"
        ) as mlx_adapter:
            result = transcribe_with_whisper(
                Path("/tmp/audio.wav"),
                analysis_paths(Path("/tmp/out")),
            )

        self.assertEqual(expected, result)
        openai_adapter.assert_called_once()
        mlx_adapter.assert_not_called()

    def test_mlx_adapter_reads_json_and_records_backend_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir) / "out")
            paths.audio_dir.mkdir(parents=True)
            audio_path = paths.audio_dir / "normalized.wav"
            audio_path.write_bytes(b"audio")
            captured_command: list[str] = []

            def fake_run_command(command, env=None):
                del env
                captured_command.extend(command)
                temp_dir = paths.root / "tmp-whisper"
                temp_dir.mkdir(parents=True, exist_ok=True)
                (temp_dir / "normalized.json").write_text(
                    json.dumps(
                        {
                            "text": "MLX transcript",
                            "language": "en",
                            "segments": [
                                {"start": 0.0, "end": 1.0, "text": "MLX transcript"}
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            with mock.patch(
                "youtube_analysis_tool.pipeline.resolve_command",
                return_value=("/tmp/mlx_whisper", None),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_command",
                side_effect=fake_run_command,
            ):
                result = transcribe_with_whisper(
                    audio_path,
                    paths,
                    backend="mlx-whisper",
                )

        self.assertEqual("whisper", result["source"])
        self.assertEqual("mlx-whisper", result["backend"])
        self.assertEqual("mlx-community/whisper-base-mlx", result["model"])
        self.assertEqual("MLX transcript", result["text"])
        self.assertIn("--output-format", captured_command)
        self.assertIn("--output-dir", captured_command)
        self.assertIn("mlx-community/whisper-base-mlx", captured_command)

    def test_missing_mlx_backend_fails_clearly(self) -> None:
        with mock.patch(
            "youtube_analysis_tool.pipeline.resolve_command",
            return_value=(None, None),
        ):
            with self.assertRaisesRegex(FileNotFoundError, "mlx_whisper command is not available"):
                transcribe_with_whisper(
                    Path("/tmp/audio.wav"),
                    analysis_paths(Path("/tmp/out")),
                    backend="mlx-whisper",
                )

    def test_explicit_mlx_auto_mode_does_not_fall_back_to_remote_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir) / "out")
            paths.subtitles_dir.mkdir(parents=True)
            with mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper",
                side_effect=FileNotFoundError("mlx missing"),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill"
            ) as remote_adapter:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "mlx missing.*Remote fallback is disabled",
                ):
                    transcript_strategy_auto(
                        Path("/tmp/audio.wav"),
                        paths,
                        local_asr_backend="mlx-whisper",
                    )

        remote_adapter.assert_not_called()

    def test_reused_output_preserves_nested_local_asr_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "output.json"
            paths = analysis_paths(Path(tmpdir) / "reused")
            source_path.write_text(
                json.dumps(
                    {
                        "transcript": {
                            "source": "whisper",
                            "full_text": "MLX transcript",
                            "segments": [],
                            "provenance": {
                                "backend": "mlx-whisper",
                                "model": "mlx-community/whisper-base-mlx",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            transcript = load_reused_transcript(source_path, paths)

        self.assertEqual("mlx-whisper", transcript["backend"])
        self.assertEqual("mlx-community/whisper-base-mlx", transcript["model"])


class CommentsStageTests(unittest.TestCase):
    def test_fetch_youtube_comments_uses_yt_dlp_python_api_with_comment_limits(self) -> None:
        captured = {}

        class FakeYoutubeDL:
            def __init__(self, opts):
                captured["opts"] = opts

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                captured["url"] = url
                captured["download"] = download
                return {
                    "comments": [
                        {
                            "id": "comment-1",
                            "text": "Useful context",
                            "like_count": 12,
                            "reply_count": 2,
                            "timestamp": 1710000000,
                        }
                    ]
                }

        with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL):
            payload = fetch_youtube_comments("https://youtu.be/demo", requested_count=5)

        self.assertEqual("https://youtu.be/demo", captured["url"])
        self.assertFalse(captured["download"])
        self.assertTrue(captured["opts"]["getcomments"])
        self.assertEqual(["5"], captured["opts"]["extractor_args"]["youtube"]["max_comments"])
        self.assertEqual(["top"], captured["opts"]["extractor_args"]["youtube"]["comment_sort"])
        self.assertEqual("extracted", payload["status"])
        self.assertEqual(1, payload["returned_count"])
        self.assertEqual("Useful context", payload["items"][0]["text"])

    def test_comments_stage_failure_is_non_fatal(self) -> None:
        with mock.patch(
            "youtube_analysis_tool.pipeline.fetch_youtube_comments",
            side_effect=RuntimeError("comments unavailable"),
        ):
            payload = run_comments_stage("https://youtu.be/demo", requested_count=5)

        self.assertEqual("failed", payload["status"])
        self.assertEqual(5, payload["requested_count"])
        self.assertEqual(0, payload["returned_count"])
        self.assertIn("comments unavailable", payload["error"])
        self.assertIn("top comments are contextual signals", payload["interpretation_notes"][0])


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
                    local_asr_backend="mlx-whisper",
                    cleanup_intermediates=False,
                )

        extract_audio_mock.assert_called_once()
        whisper_mock.assert_called_once()
        self.assertEqual("mlx-whisper", whisper_mock.call_args.kwargs["backend"])

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
    def test_transcript_off_runs_visual_debug_without_audio_or_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"

            def fake_create_keyframes(video_path, metadata, paths, **kwargs):
                del video_path, metadata, kwargs
                frame_path = paths.keyframes_dir / "slide.jpg"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(b"slide-bytes")
                return [
                    {
                        "kind": "scene",
                        "filename": "slide.jpg",
                        "timestamp_seconds": 0.0,
                        "timestamp_hms": "00:00:00",
                    }
                ]

            def fake_triage(output_root_path, keyframe_rows, ocr_rows, transcript):
                del keyframe_rows, ocr_rows
                self.assertEqual("skipped", transcript["source"])
                frame = {
                    "frame_id": "frame-00001",
                    "frame_path": "keyframes/slide.jpg",
                    "timestamp_seconds": 0.0,
                    "timestamp_hms": "00:00:00",
                    "ocr_text": "Slide text",
                    "ocr_char_count": 10,
                    "blur_score": 100.0,
                }
                segment = {
                    "segment_id": "segment-0001",
                    "heuristic_label": "slides",
                    "heuristic_confidence": 0.9,
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "start_hms": "00:00:00",
                    "end_hms": "00:00:00",
                    "frame_ids": ["frame-00001"],
                    "representative_frame_paths": ["keyframes/slide.jpg"],
                    "ocr_summary": "Slide text",
                    "ocr_char_count": 10,
                    "transcript_window": {"text": None, "segments": []},
                    "review_status": "auto_approved",
                }
                self.assertTrue((output_root_path / "keyframes" / "slide.jpg").exists())
                return [frame], [segment]

            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=({"format": {"duration": "1"}, "streams": [{"codec_type": "video"}]}, source_path),
            ), mock.patch("youtube_analysis_tool.pipeline.transcript_from_preferred_subtitles") as subtitle_mock, mock.patch(
                "youtube_analysis_tool.pipeline.run_burned_subtitles_stage"
            ) as burned_mock, mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper"
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill"
            ) as openai_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                side_effect=fake_create_keyframes,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_ocr_stage",
                return_value=(
                    [{"filename": "slide.jpg", "timestamp_seconds": 0.0, "timestamp_hms": "00:00:00", "text": "Slide text"}],
                    {"mode": "auto", "status": "completed", "attempted": True, "frame_count": 1, "error": None},
                ),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.run_local_triage",
                side_effect=fake_triage,
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    transcript_mode="off",
                    visuals_mode="on",
                    artifacts_mode="debug",
                )

                written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))
                manifest = json.loads((output_root / "visuals" / "manifest.json").read_text(encoding="utf-8"))

        subtitle_mock.assert_not_called()
        burned_mock.assert_not_called()
        extract_audio_mock.assert_not_called()
        whisper_mock.assert_not_called()
        openai_mock.assert_not_called()
        self.assertEqual("skipped", written["transcript"]["source"])
        self.assertEqual("skipped", written["transcript"]["provenance"]["status"])
        self.assertEqual("skipped", written["provenance"]["transcript"]["status"])
        self.assertEqual("completed", written["processing"]["run_status"])
        self.assertEqual("off", written["processing"]["transcript_mode"])
        self.assertIsNone(written["visuals"]["slides"][0]["transcript_excerpt"])
        self.assertEqual(1, len(manifest["slides"]))
        self.assertFalse((output_root / "keyframes").exists())

    def test_reuse_transcript_uses_existing_artifact_without_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "demo.mp4"
            source_path.write_bytes(b"video")
            transcript_path = root / "transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "source": "whisper",
                        "language": "en",
                        "text": "Reused transcript line",
                        "segments": [{"start": 0.0, "end": 5.0, "text": "Reused transcript line"}],
                    }
                ),
                encoding="utf-8",
            )
            output_root = root / "out"

            def fake_create_keyframes(video_path, metadata, paths, **kwargs):
                del video_path, metadata, kwargs
                frame_path = paths.keyframes_dir / "slide.jpg"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(b"slide-bytes")
                return [
                    {
                        "kind": "scene",
                        "filename": "slide.jpg",
                        "timestamp_seconds": 1.0,
                        "timestamp_hms": "00:00:01",
                    }
                ]

            def fake_triage(output_root_path, keyframe_rows, ocr_rows, transcript):
                del output_root_path, keyframe_rows, ocr_rows
                self.assertEqual("Reused transcript line", transcript["text"])
                return [
                    {
                        "frame_id": "frame-00001",
                        "frame_path": "keyframes/slide.jpg",
                        "timestamp_seconds": 1.0,
                        "timestamp_hms": "00:00:01",
                        "ocr_text": "Slide text",
                        "ocr_char_count": 10,
                        "blur_score": 100.0,
                    }
                ], [
                    {
                        "segment_id": "segment-0001",
                        "heuristic_label": "slides",
                        "heuristic_confidence": 0.9,
                        "start_seconds": 1.0,
                        "end_seconds": 1.0,
                        "start_hms": "00:00:01",
                        "end_hms": "00:00:01",
                        "frame_ids": ["frame-00001"],
                        "representative_frame_paths": ["keyframes/slide.jpg"],
                        "ocr_summary": "Slide text",
                        "ocr_char_count": 10,
                        "transcript_window": {"text": "Reused transcript line", "segments": transcript["segments"]},
                        "review_status": "auto_approved",
                    }
                ]

            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=({"format": {"duration": "1"}, "streams": [{"codec_type": "video"}]}, source_path),
            ), mock.patch("youtube_analysis_tool.pipeline.transcript_from_preferred_subtitles") as subtitle_mock, mock.patch(
                "youtube_analysis_tool.pipeline.extract_audio"
            ) as extract_audio_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_whisper"
            ) as whisper_mock, mock.patch(
                "youtube_analysis_tool.pipeline.transcribe_with_openai_skill"
            ) as openai_mock, mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                side_effect=fake_create_keyframes,
            ), mock.patch(
                "youtube_analysis_tool.pipeline.run_ocr_stage",
                return_value=(
                    [{"filename": "slide.jpg", "timestamp_seconds": 1.0, "timestamp_hms": "00:00:01", "text": "Slide text"}],
                    {"mode": "auto", "status": "completed", "attempted": True, "frame_count": 1, "error": None},
                ),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.triage.run_local_triage",
                side_effect=fake_triage,
            ):
                analyze_source(
                    str(source_path),
                    out_dir=output_root,
                    reuse_transcript=transcript_path,
                    visuals_mode="on",
                    artifacts_mode="debug",
                )

                written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))

        subtitle_mock.assert_not_called()
        extract_audio_mock.assert_not_called()
        whisper_mock.assert_not_called()
        openai_mock.assert_not_called()
        self.assertEqual("whisper", written["transcript"]["source"])
        self.assertEqual("Reused transcript line", written["transcript"]["full_text"])
        self.assertEqual("reused", written["transcript"]["provenance"]["status"])
        self.assertEqual(str(transcript_path.resolve()), written["transcript"]["provenance"]["reused_from"])
        self.assertEqual("reused", written["provenance"]["transcript"]["status"])
        self.assertEqual("Reused transcript line", written["visuals"]["slides"][0]["transcript_excerpt"])

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
                    max_video_height=720,
                    cleanup_intermediates=False,
                )

                written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))

        self.assertEqual("off", keyframes_mock.call_args.kwargs["mode"])
        self.assertEqual("off", written["processing"]["visuals_mode"])
        self.assertEqual(720, written["processing"]["requested_max_video_height"])
        self.assertFalse(written["provenance"]["media"]["height_limit_applied"])
        self.assertEqual([], written["visuals"]["slides"])
        self.assertEqual([], written["visuals"]["charts"])

    def test_stage_failure_writes_failed_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"

            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=({"streams": [{"codec_type": "video"}]}, source_path),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                side_effect=RuntimeError("visual stage failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "visual stage failed"):
                    analyze_source(
                        str(source_path),
                        out_dir=output_root,
                        transcript_mode="off",
                        visuals_mode="on",
                        cleanup_intermediates=False,
                    )

            written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))

        self.assertEqual("failed", written["processing"]["run_status"])
        self.assertEqual(1, written["processing"]["counts"]["error_count"])
        self.assertEqual("visual stage failed", written["errors"][0]["message"])

    def test_keyboard_interrupt_writes_aborted_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"

            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=({"streams": [{"codec_type": "video"}]}, source_path),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                side_effect=KeyboardInterrupt,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    analyze_source(
                        str(source_path),
                        out_dir=output_root,
                        transcript_mode="off",
                        visuals_mode="on",
                        cleanup_intermediates=False,
                    )

            written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))

        self.assertEqual("aborted", written["processing"]["run_status"])
        self.assertEqual(1, written["processing"]["counts"]["error_count"])
        self.assertEqual("interrupted", written["errors"][0]["kind"])
        self.assertEqual("Analysis interrupted by user.", written["errors"][0]["message"])

    def test_cleanup_failure_writes_failed_partial_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "demo.mp4"
            source_path.write_bytes(b"video")
            output_root = Path(tmpdir) / "out"

            with mock.patch(
                "youtube_analysis_tool.pipeline.materialize_local_input",
                return_value=({"streams": [{"codec_type": "video"}]}, source_path),
            ), mock.patch(
                "youtube_analysis_tool.pipeline.create_keyframes",
                return_value=[],
            ), mock.patch(
                "youtube_analysis_tool.pipeline.write_empty_stage_artifacts",
            ), mock.patch(
                "youtube_analysis_tool.pipeline.cleanup_intermediate_artifacts",
                side_effect=RuntimeError("cleanup failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                    analyze_source(
                        str(source_path),
                        out_dir=output_root,
                        transcript_mode="off",
                        visuals_mode="off",
                    )

            written = json.loads((output_root / "output.json").read_text(encoding="utf-8"))

        self.assertEqual("failed", written["processing"]["run_status"])
        self.assertEqual("cleanup", written["errors"][0]["stage"])
        self.assertEqual("cleanup failed", written["errors"][0]["message"])

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
    def test_youtube_format_selector_is_optional_and_height_bounded(self) -> None:
        self.assertIsNone(youtube_format_selector(None))
        self.assertEqual(
            "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            youtube_format_selector(720),
        )

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

    def test_choose_subtitle_track_from_metadata_accepts_nonpreferred_manual_language_when_only_option(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "subtitles": {
                    "cs": [{"ext": "vtt", "url": "https://example.com/cs.vtt"}]
                }
            }
        )

        self.assertIsNotNone(selection)
        bucket_name, language, item = selection
        self.assertEqual("subtitles", bucket_name)
        self.assertEqual("cs", language)
        self.assertEqual("vtt", item["ext"])

    def test_choose_subtitle_track_from_metadata_accepts_nonpreferred_auto_language_when_only_option(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "automatic_captions": {
                    "cs": [{"ext": "json3", "url": "https://example.com/cs.json3"}]
                }
            }
        )

        self.assertIsNotNone(selection)
        bucket_name, language, item = selection
        self.assertEqual("automatic_captions", bucket_name)
        self.assertEqual("cs", language)
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

    def test_choose_subtitle_track_from_metadata_rejects_translated_auto_caption_when_it_is_only_option(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "automatic_captions": {
                    "zh-Hant": [
                        {
                            "ext": "vtt",
                            "url": "https://example.com/caption.vtt?tlang=zh-Hant&lang=ja",
                        }
                    ]
                }
            }
        )

        self.assertIsNone(selection)

    def test_choose_subtitle_track_from_metadata_rejects_translated_subtitles_bucket_track(self) -> None:
        selection = choose_subtitle_track_from_metadata(
            {
                "subtitles": {
                    "en": [
                        {
                            "ext": "vtt",
                            "url": "https://example.com/caption.vtt?tlang=en&lang=ja",
                        }
                    ]
                }
            }
        )

        self.assertIsNone(selection)

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

    def test_download_youtube_media_can_skip_subtitle_fetch(self) -> None:
        calls = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                calls.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                del url, download
                default_template = self.opts["outtmpl"]["default"]
                video_path = Path(default_template.replace("%(ext)s", "mp4"))
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"video")
                return {"id": "demo-video"}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL):
                _, video_path = download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    fetch_subtitles=False,
                )
                video_exists = video_path.exists()

        self.assertEqual(1, len(calls))
        self.assertNotIn("skip_download", calls[0])
        self.assertNotIn("format", calls[0])
        self.assertTrue(video_exists)

    def test_download_youtube_media_applies_max_video_height(self) -> None:
        calls = []

        class FakeYoutubeDL:
            def __init__(self, opts):
                self.opts = opts
                calls.append(opts)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def extract_info(self, url, download):
                del url, download
                default_template = self.opts["outtmpl"]["default"]
                video_path = Path(default_template.replace("%(ext)s", "mp4"))
                video_path.parent.mkdir(parents=True, exist_ok=True)
                video_path.write_bytes(b"video")
                return {"id": "demo-video"}

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL):
                download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    fetch_subtitles=False,
                    max_video_height=720,
                )

        self.assertEqual(
            "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            calls[0]["format"],
        )

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

    def test_download_youtube_media_falls_back_to_post_download_info_for_nonpreferred_auto_caption_language(self) -> None:
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
                        "cs": [{"ext": "vtt", "url": "https://example.com/cs-auto.vtt"}]
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
                return_value=FakeResponse(b"WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nahoj\n"),
            ):
                info, video_path = download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    metadata_hint={"id": "demo-video"},
                )
            self.assertEqual("demo-video", info["id"])
            self.assertTrue(video_path.exists())
            subtitle_path = paths.subtitles_dir / "demo-video.cs.auto.vtt"
            self.assertTrue(subtitle_path.exists())

    def test_download_youtube_media_ignores_translated_auto_captions_in_post_download_info(self) -> None:
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
                        "zh-Hant": [
                            {
                                "ext": "vtt",
                                "url": "https://example.com/translated.vtt?tlang=zh-Hant&lang=ja",
                            }
                        ]
                    },
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = analysis_paths(Path(tmpdir))
            paths.video_dir.mkdir(parents=True, exist_ok=True)
            with mock.patch("youtube_analysis_tool.pipeline.load_yt_dlp", return_value=FakeYoutubeDL):
                info, video_path = download_youtube_media(
                    "https://youtu.be/demo",
                    paths,
                    metadata_hint={"id": "demo-video"},
                )

            self.assertEqual("demo-video", info["id"])
            self.assertTrue(video_path.exists())
            self.assertFalse(any(paths.subtitles_dir.glob("*")))

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
