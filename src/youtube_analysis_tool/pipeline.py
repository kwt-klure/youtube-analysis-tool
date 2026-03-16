from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen
from urllib.parse import urlparse

from . import constants, gpt, reporting, review, routing, triage
from .artifacts import write_json


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    analysis_json_path: Path
    audio_dir: Path
    video_dir: Path
    subtitles_dir: Path
    keyframes_dir: Path
    transcript_text_path: Path
    transcript_json_path: Path
    keyframe_index_path: Path
    metadata_path: Path
    source_path: Path
    error_path: Path
    ocr_dir: Path
    ocr_index_path: Path
    triage_dir: Path
    triage_frames_path: Path
    triage_segments_path: Path
    review_dir: Path
    review_queue_path: Path
    review_decisions_path: Path
    routing_dir: Path
    routing_manifest_path: Path
    gpt_dir: Path
    gpt_analyses_path: Path
    report_dir: Path
    report_markdown_path: Path
    report_json_path: Path


def looks_like_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def is_youtube_url(source: str) -> bool:
    parsed = urlparse(source)
    host = parsed.netloc.lower()
    return any(marker in host for marker in constants.YOUTUBE_HOST_MARKERS)


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized or "video"


def hms_from_seconds(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def default_output_dir_for_source(source: str, video_id: str | None = None) -> Path:
    if video_id:
        return constants.DEFAULT_OUTPUT_ROOT / slugify(video_id)
    if looks_like_url(source):
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        return constants.DEFAULT_OUTPUT_ROOT / f"url-{digest}"
    return constants.DEFAULT_OUTPUT_ROOT / slugify(Path(source).stem)


def analysis_paths(root: Path) -> AnalysisPaths:
    return AnalysisPaths(
        root=root,
        analysis_json_path=root / "analysis.json",
        audio_dir=root / "audio",
        video_dir=root / "video",
        subtitles_dir=root / "subtitles",
        keyframes_dir=root / "keyframes",
        transcript_text_path=root / "transcript.txt",
        transcript_json_path=root / "transcript.json",
        keyframe_index_path=root / "keyframes" / "index.csv",
        metadata_path=root / "metadata.json",
        source_path=root / "source.txt",
        error_path=root / "error.json",
        ocr_dir=root / "ocr",
        ocr_index_path=root / "ocr" / "index.csv",
        triage_dir=root / "triage",
        triage_frames_path=root / "triage" / "frames.jsonl",
        triage_segments_path=root / "triage" / "segments.json",
        review_dir=root / "review",
        review_queue_path=root / "review" / "queue.json",
        review_decisions_path=root / "review" / "decisions.json",
        routing_dir=root / "routing",
        routing_manifest_path=root / "routing" / "manifest.json",
        gpt_dir=root / "gpt",
        gpt_analyses_path=root / "gpt" / "analyses.json",
        report_dir=root / "report",
        report_markdown_path=root / "report" / "report.md",
        report_json_path=root / "report" / "report.json",
    )


def ensure_dirs(paths: AnalysisPaths, with_ocr: bool = False) -> None:
    for path in (
        paths.root,
        paths.audio_dir,
        paths.video_dir,
        paths.subtitles_dir,
        paths.keyframes_dir,
        paths.triage_dir,
        paths.review_dir,
        paths.routing_dir,
        paths.gpt_dir,
        paths.report_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if with_ocr:
        paths.ocr_dir.mkdir(parents=True, exist_ok=True)


def clear_stale_output_files(paths: AnalysisPaths) -> None:
    for path in (
        paths.error_path,
        paths.gpt_analyses_path,
        paths.report_json_path,
        paths.report_markdown_path,
    ):
        path.unlink(missing_ok=True)


def write_keyframe_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "filename", "timestamp_seconds", "timestamp_hms"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_ocr_index(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "timestamp_seconds", "timestamp_hms", "text"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_command(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
        env=env,
    )


def save_error(paths: AnalysisPaths, stage: str, message: str) -> None:
    write_json(paths.error_path, {"stage": stage, "message": message})


def ffprobe_json(media_path: Path) -> dict[str, Any]:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(media_path),
        ]
    )
    return json.loads(result.stdout)


def has_video_stream(metadata: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "video" for stream in metadata.get("streams", []))


def duration_seconds(metadata: dict[str, Any]) -> float:
    format_value = metadata.get("format", {})
    raw = None
    if isinstance(format_value, dict):
        raw = format_value.get("duration")
    if raw is None:
        raw = metadata.get("duration")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def link_local_source(source_path: Path, destination_dir: Path) -> Path:
    destination = destination_dir / f"source{source_path.suffix.lower()}"
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source_path.resolve())
    return destination


def detect_language_from_filename(path: Path) -> str | None:
    suffixes = path.name.split(".")
    if len(suffixes) < 3:
        return None
    return suffixes[-2].lower()


def subtitle_rank(path: Path) -> tuple[int, str]:
    language = detect_language_from_filename(path) or "zz"
    try:
        rank = constants.SUBTITLE_LANGUAGE_PREFERENCE.index(language)
    except ValueError:
        rank = len(constants.SUBTITLE_LANGUAGE_PREFERENCE)
    return (rank, language)


def subtitle_language_rank(language: str | None) -> int:
    normalized = (language or "zz").lower()
    try:
        return constants.SUBTITLE_LANGUAGE_PREFERENCE.index(normalized)
    except ValueError:
        return len(constants.SUBTITLE_LANGUAGE_PREFERENCE)


def subtitle_ext_rank(ext: str | None) -> int:
    preferred = ("vtt", "srt", "ttml", "srv3", "srv2", "srv1", "json3")
    normalized = (ext or "").lower()
    try:
        return preferred.index(normalized)
    except ValueError:
        return len(preferred)


def choose_subtitle_file(subtitles_dir: Path) -> Path | None:
    candidates = [
        path
        for path in subtitles_dir.glob("*")
        if path.is_file()
        and path.suffix.lower() in {".vtt", ".srt"}
        and "live_chat" not in path.name.lower()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=subtitle_rank)[0]


def choose_subtitle_track_from_metadata(metadata: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    for bucket_name in ("subtitles", "automatic_captions"):
        bucket = metadata.get(bucket_name) or {}
        for language, items in bucket.items():
            for item in items:
                ext = str(item.get("ext", "")).lower()
                if ext not in {"vtt", "srt"}:
                    continue
                candidates.append(
                    (
                        subtitle_language_rank(language),
                        subtitle_ext_rank(ext),
                        language,
                        item,
                    )
                )
    if not candidates:
        return None
    _, _, language, item = sorted(candidates, key=lambda row: row[:2])[0]
    return language, item


def preferred_subtitle_languages() -> list[str]:
    return [language for language in constants.SUBTITLE_LANGUAGE_PREFERENCE]


def download_subtitle_from_metadata(metadata: dict[str, Any], paths: AnalysisPaths) -> Path | None:
    selection = choose_subtitle_track_from_metadata(metadata)
    if selection is None:
        return None
    language, item = selection
    url = item.get("url")
    if not url:
        return None
    ext = str(item.get("ext", "vtt")).lower()
    subtitle_path = paths.subtitles_dir / f"{metadata.get('id', 'video')}.{language}.{ext}"
    with urlopen(str(url)) as response:
        payload = response.read()
    subtitle_path.parent.mkdir(parents=True, exist_ok=True)
    subtitle_path.write_bytes(payload)
    return subtitle_path


def parse_timestamp(raw: str) -> float:
    token = raw.replace(",", ".")
    hours, minutes, seconds = token.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def clean_caption_text(lines: list[str]) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_srt_or_vtt(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if lines and lines[0].strip().upper().startswith("WEBVTT"):
        lines = lines[1:]
    segments: list[dict[str, Any]] = []
    block: list[str] = []
    for line in lines + [""]:
        if line.strip():
            block.append(line)
            continue
        if not block:
            continue
        timing_line_index = 0
        if "-->" not in block[0] and len(block) > 1:
            timing_line_index = 1
        timing_line = block[timing_line_index]
        if "-->" not in timing_line:
            block = []
            continue
        start_raw, end_raw = [part.strip().split(" ")[0] for part in timing_line.split("-->")]
        text = clean_caption_text(block[timing_line_index + 1 :])
        if text:
            segments.append(
                {
                    "start": round(parse_timestamp(start_raw), 3),
                    "end": round(parse_timestamp(end_raw), 3),
                    "text": text,
                }
            )
        block = []
    return segments


def transcript_from_segments(
    segments: list[dict[str, Any]],
    *,
    source: str,
    language: str | None = None,
    source_path: str | None = None,
) -> dict[str, Any]:
    text = "\n".join(segment["text"] for segment in segments if segment["text"])
    return {
        "source": source,
        "language": language,
        "source_path": source_path,
        "segment_count": len(segments),
        "text": text,
        "segments": segments,
    }


def write_transcript(paths: AnalysisPaths, transcript: dict[str, Any]) -> None:
    write_json(paths.transcript_json_path, transcript)
    paths.transcript_text_path.write_text(transcript.get("text", "").strip() + "\n", encoding="utf-8")


def transcript_from_subtitles(subtitle_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    segments = parse_srt_or_vtt(subtitle_path)
    language = detect_language_from_filename(subtitle_path)
    transcript = transcript_from_segments(
        segments,
        source="subtitle",
        language=language,
        source_path=str(subtitle_path.relative_to(paths.root)),
    )
    write_transcript(paths, transcript)
    return transcript


def extract_audio(input_media_path: Path, audio_dir: Path) -> Path:
    audio_path = audio_dir / "source.wav"
    run_command(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_media_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )
    return audio_path


def codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))


def transcribe_skill_script() -> Path:
    return codex_home() / "skills" / "transcribe" / "scripts" / "transcribe_diarize.py"


def extract_text_from_transcript_payload(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"].strip()
    segments = payload.get("segments") or []
    return "\n".join(segment.get("text", "").strip() for segment in segments if segment.get("text"))


def transcribe_with_openai_skill(audio_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    script_path = transcribe_skill_script()
    if not script_path.exists():
        raise FileNotFoundError(f"Transcribe skill is not installed: {script_path}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set for API transcription.")
    temp_json = paths.root / "transcript.openai.json"
    run_command(
        [
            sys.executable,
            str(script_path),
            str(audio_path),
            "--response-format",
            "json",
            "--out",
            str(temp_json),
        ]
    )
    payload = json.loads(temp_json.read_text(encoding="utf-8"))
    temp_json.unlink(missing_ok=True)
    transcript = {
        "source": "openai",
        "language": payload.get("language"),
        "source_path": str(audio_path.relative_to(paths.root)),
        "segment_count": len(payload.get("segments") or []),
        "text": extract_text_from_transcript_payload(payload),
        "segments": payload.get("segments") or [],
        "raw": payload,
    }
    write_transcript(paths, transcript)
    return transcript


def transcribe_with_whisper(audio_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    whisper_bin = shutil.which("whisper")
    if whisper_bin is None:
        raise FileNotFoundError("whisper command is not available.")
    temp_dir = paths.root / "tmp-whisper"
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    command = [
        whisper_bin,
        str(audio_path),
        "--model",
        "base",
        "--task",
        "transcribe",
        "--output_format",
        "json",
        "--output_dir",
        str(temp_dir),
    ]
    run_command(command, env=env)
    json_path = temp_dir / f"{audio_path.stem}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    transcript = {
        "source": "whisper",
        "language": payload.get("language"),
        "source_path": str(audio_path.relative_to(paths.root)),
        "segment_count": len(payload.get("segments") or []),
        "text": extract_text_from_transcript_payload(payload),
        "segments": payload.get("segments") or [],
        "raw": payload,
    }
    write_transcript(paths, transcript)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return transcript


def select_filter_expression(threshold: float) -> str:
    return f"select='gt(scene\\,{threshold})',showinfo"


def extract_scene_keyframes(
    video_path: Path,
    keyframes_dir: Path,
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    output_pattern = keyframes_dir / "scene-%05d.jpg"
    result = run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            select_filter_expression(threshold),
            "-fps_mode",
            "vfr",
            str(output_pattern),
        ]
    )
    timestamps = [
        float(match.group(1))
        for match in re.finditer(r"pts_time:(\d+(?:\.\d+)?)", result.stderr)
    ]
    frames = sorted(keyframes_dir.glob("scene-*.jpg"))
    rows: list[dict[str, Any]] = []
    for frame_path, timestamp in zip(frames, timestamps):
        rows.append(
            {
                "kind": "scene",
                "filename": frame_path.name,
                "timestamp_seconds": round(timestamp, 3),
                "timestamp_hms": hms_from_seconds(timestamp),
            }
        )
    return rows


def extract_interval_keyframes(
    video_path: Path,
    keyframes_dir: Path,
    *,
    duration: float,
    interval_seconds: int,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if interval_seconds <= 0:
        return []
    existing_timestamps = [float(row["timestamp_seconds"]) for row in existing_rows]
    rows: list[dict[str, Any]] = []
    interval = max(interval_seconds, 1)
    stop = max(int(duration), 1)
    for timestamp in range(0, stop, interval):
        if any(abs(timestamp - existing) < 2.0 for existing in existing_timestamps):
            continue
        frame_path = keyframes_dir / f"interval-{timestamp:06d}.jpg"
        try:
            run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    str(timestamp),
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(frame_path),
                ]
            )
        except subprocess.CalledProcessError:
            continue
        if frame_path.exists():
            rows.append(
                {
                    "kind": "interval",
                    "filename": frame_path.name,
                    "timestamp_seconds": float(timestamp),
                    "timestamp_hms": hms_from_seconds(float(timestamp)),
                }
            )
    return rows


def run_ocr(paths: AnalysisPaths, keyframe_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import cv2
    import pytesseract

    ensure_dirs(paths, with_ocr=True)
    rows: list[dict[str, Any]] = []
    for row in keyframe_rows:
        image_path = paths.keyframes_dir / row["filename"]
        image = cv2.imread(str(image_path))
        text = pytesseract.image_to_string(image).strip() if image is not None else ""
        rows.append(
            {
                "filename": row["filename"],
                "timestamp_seconds": row["timestamp_seconds"],
                "timestamp_hms": row["timestamp_hms"],
                "text": text,
            }
        )
    write_ocr_index(paths.ocr_index_path, rows)
    return rows


def default_ocr_state(ocr_mode: str) -> dict[str, Any]:
    return {
        "mode": ocr_mode,
        "status": "not_attempted",
        "attempted": False,
        "frame_count": 0,
        "artifact_path": "ocr/index.csv",
        "error": None,
    }


def try_run_optional_ocr(paths: AnalysisPaths, keyframe_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    try:
        return run_ocr(paths, keyframe_rows), None
    except Exception as exc:
        write_ocr_index(paths.ocr_index_path, [])
        return [], str(exc)


def run_ocr_stage(
    paths: AnalysisPaths,
    keyframe_rows: list[dict[str, Any]],
    *,
    ocr_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    state = default_ocr_state(ocr_mode)
    if ocr_mode == "off":
        state["status"] = "disabled"
        return [], state
    if not keyframe_rows:
        write_ocr_index(paths.ocr_index_path, [])
        state["status"] = "not_applicable"
        return [], state
    if ocr_mode == "on":
        try:
            rows = run_ocr(paths, keyframe_rows)
        except Exception as exc:
            write_ocr_index(paths.ocr_index_path, [])
            state["status"] = "failed"
            state["attempted"] = True
            state["error"] = str(exc)
            raise
        state["status"] = "completed"
        state["attempted"] = True
        state["frame_count"] = len(rows)
        return rows, state
    rows, error = try_run_optional_ocr(paths, keyframe_rows)
    state["attempted"] = True
    state["frame_count"] = len(rows)
    if error is None:
        state["status"] = "completed"
    else:
        state["status"] = "failed"
        state["error"] = error
    return rows, state


def materialize_local_input(source_path: Path, paths: AnalysisPaths) -> tuple[dict[str, Any], Path | None]:
    metadata = ffprobe_json(source_path)
    write_json(paths.metadata_path, metadata)
    if has_video_stream(metadata):
        linked = link_local_source(source_path, paths.video_dir)
        return metadata, linked
    linked = link_local_source(source_path, paths.audio_dir)
    return metadata, None


def load_yt_dlp():
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError(
            "yt-dlp is not installed. Install the package or optional youtube extras."
        ) from exc
    return YoutubeDL


def fetch_youtube_metadata(url: str) -> dict[str, Any]:
    YoutubeDL = load_yt_dlp()
    with YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        return ydl.extract_info(url, download=False)


def download_youtube_subtitles(url: str, paths: AnalysisPaths) -> None:
    YoutubeDL = load_yt_dlp()
    subtitle_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": preferred_subtitle_languages(),
        "subtitlesformat": "vtt",
        "outtmpl": {
            "subtitle": str(paths.subtitles_dir / "%(id)s.%(language)s.%(ext)s"),
        },
    }
    with YoutubeDL(subtitle_opts) as ydl:
        ydl.extract_info(url, download=True)


def download_youtube_media(
    url: str,
    paths: AnalysisPaths,
    *,
    metadata_hint: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    YoutubeDL = load_yt_dlp()
    try:
        download_youtube_subtitles(url, paths)
    except Exception:
        # Subtitle retrieval is best-effort. If YouTube rate limits or rejects
        # caption download, keep the pipeline alive and allow later fallback to
        # Whisper after the video is downloaded.
        if metadata_hint is not None:
            try:
                download_subtitle_from_metadata(metadata_hint, paths)
            except Exception:
                pass
    video_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "outtmpl": {
            "default": str(paths.video_dir / "source.%(ext)s"),
        },
    }
    with YoutubeDL(video_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    write_json(paths.metadata_path, info)
    candidates = [
        path
        for path in paths.video_dir.glob("source.*")
        if path.is_file() and not path.name.endswith(".part")
    ]
    if not candidates:
        raise FileNotFoundError("yt-dlp did not leave a downloaded video file in the video directory.")
    return info, sorted(candidates)[0]


def ensure_source_file(paths: AnalysisPaths, source: str) -> None:
    paths.source_path.write_text(source + "\n", encoding="utf-8")


def transcript_strategy_auto(audio_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    subtitle_file = choose_subtitle_file(paths.subtitles_dir)
    if subtitle_file is not None:
        return transcript_from_subtitles(subtitle_file, paths)
    try:
        return transcribe_with_whisper(audio_path, paths)
    except Exception:
        return transcribe_with_openai_skill(audio_path, paths)


def create_keyframes(
    video_path: Path | None,
    metadata: dict[str, Any],
    paths: AnalysisPaths,
    *,
    mode: str,
    interval_seconds: int,
    threshold: float,
) -> list[dict[str, Any]]:
    if video_path is None or mode == "off":
        write_keyframe_index(paths.keyframe_index_path, [])
        return []
    rows: list[dict[str, Any]] = []
    if mode in {"scene", "scene+interval"}:
        rows.extend(extract_scene_keyframes(video_path, paths.keyframes_dir, threshold=threshold))
    if mode in {"interval", "scene+interval"}:
        rows.extend(
            extract_interval_keyframes(
                video_path,
                paths.keyframes_dir,
                duration=duration_seconds(metadata),
                interval_seconds=interval_seconds,
                existing_rows=rows,
            )
        )
    rows = sorted(rows, key=lambda row: float(row["timestamp_seconds"]))
    write_keyframe_index(paths.keyframe_index_path, rows)
    return rows


def write_empty_stage_artifacts(paths: AnalysisPaths) -> None:
    paths.triage_frames_path.write_text("", encoding="utf-8")
    write_json(paths.triage_segments_path, {"segments": []})
    write_json(paths.review_queue_path, {"queue": []})
    write_json(paths.review_decisions_path, {"decisions": {}})
    write_json(paths.routing_manifest_path, {"segments": []})


def cleanup_intermediate_artifacts(paths: AnalysisPaths) -> None:
    for directory in (
        paths.audio_dir,
        paths.video_dir,
        paths.subtitles_dir,
        paths.keyframes_dir,
        paths.ocr_dir,
        paths.root / "tmp-whisper",
    ):
        shutil.rmtree(directory, ignore_errors=True)


def analyze_source(
    source: str,
    *,
    transcript_mode: str = "auto",
    keyframe_mode: str = "scene+interval",
    ocr_mode: str = constants.DEFAULT_OCR_MODE,
    out_dir: Path | None = None,
    interval_seconds: int = constants.DEFAULT_INTERVAL_SECONDS,
    scene_threshold: float = constants.DEFAULT_SCENE_THRESHOLD,
    triage_mode: str = constants.DEFAULT_TRIAGE_MODE,
    gpt_mode: str = constants.DEFAULT_GPT_MODE,
    review_mode: str = constants.DEFAULT_REVIEW_MODE,
    gpt_model: str = constants.DEFAULT_GPT_MODEL,
    report_language: str = constants.DEFAULT_REPORT_LANGUAGE,
    review_reset: bool = False,
    cleanup_intermediates: bool = True,
    review_input: Callable[[str], str] = input,
) -> Path:
    source_path = Path(source).expanduser()
    metadata_hint: dict[str, Any] | None = None
    if looks_like_url(source):
        metadata_hint = fetch_youtube_metadata(source)
    output_root = out_dir or default_output_dir_for_source(source, (metadata_hint or {}).get("id"))
    paths = analysis_paths(output_root)
    ensure_dirs(paths)
    clear_stale_output_files(paths)
    ensure_source_file(paths, source)
    metadata: dict[str, Any] = {}
    transcript: dict[str, Any] | None = None
    ocr_state = default_ocr_state(ocr_mode)
    segments: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    gpt_payload: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []

    try:
        if looks_like_url(source):
            metadata = metadata_hint or fetch_youtube_metadata(source)
            write_json(paths.metadata_path, metadata)
            metadata, video_path = download_youtube_media(source, paths, metadata_hint=metadata)
            audio_input = video_path
        else:
            if not source_path.exists():
                raise FileNotFoundError(f"Source file does not exist: {source_path}")
            metadata, video_path = materialize_local_input(source_path.resolve(), paths)
            audio_input = video_path or source_path.resolve()

        normalized_audio = extract_audio(audio_input, paths.audio_dir)

        if transcript_mode == "auto":
            transcript = transcript_strategy_auto(normalized_audio, paths)
        elif transcript_mode == "subtitles":
            subtitle_file = choose_subtitle_file(paths.subtitles_dir)
            if subtitle_file is None:
                raise FileNotFoundError("No subtitle file is available for transcript_mode=subtitles.")
            transcript = transcript_from_subtitles(subtitle_file, paths)
        elif transcript_mode == "api":
            transcript = transcribe_with_openai_skill(normalized_audio, paths)
        elif transcript_mode == "whisper":
            transcript = transcribe_with_whisper(normalized_audio, paths)
        else:
            raise ValueError(f"Unsupported transcript mode: {transcript_mode}")

        keyframe_rows = create_keyframes(
            video_path,
            metadata,
            paths,
            mode=keyframe_mode,
            interval_seconds=interval_seconds,
            threshold=scene_threshold,
        )

        try:
            ocr_rows, ocr_state = run_ocr_stage(paths, keyframe_rows, ocr_mode=ocr_mode)
        except Exception as exc:
            ocr_state = default_ocr_state(ocr_mode)
            ocr_state["attempted"] = bool(keyframe_rows) and ocr_mode != "off"
            ocr_state["status"] = "failed"
            ocr_state["error"] = str(exc)
            raise

        if triage_mode == "on" and keyframe_rows:
            _, segments = triage.run_local_triage(paths.root, keyframe_rows, ocr_rows, transcript)
            manifest_entries = [
                routing.manifest_entry_from_segment(segment, gpt_model)
                for segment in segments
            ]
            queue = review.build_review_queue(paths.root, manifest_entries)
            decisions = review.load_review_decisions(paths.root, reset=review_reset)
            if gpt_mode == "on" and review_mode == "interactive" and queue:
                decisions = review.interactive_review(
                    paths.root,
                    queue,
                    decisions,
                    input_func=review_input,
                )
            review.apply_review_decisions(manifest_entries, decisions)
            for entry in manifest_entries:
                entry["detail"] = routing.resolve_detail_for_model(entry["detail"], gpt_model)
            routing.finalize_manifest_entries(
                manifest_entries,
                review_enabled=(gpt_mode == "on" and review_mode == "interactive"),
            )
            write_json(paths.review_decisions_path, {"decisions": decisions})
            write_json(paths.routing_manifest_path, {"segments": manifest_entries})
        else:
            write_empty_stage_artifacts(paths)

        if gpt_mode == "on":
            approved_entries = [entry for entry in manifest_entries if entry.get("approved_for_gpt")]
            if approved_entries or transcript.get("text", "").strip():
                segment_analyses, final_report = gpt.analyze_segments(
                    paths.root,
                    manifest_entries,
                    model=gpt_model,
                    transcript=transcript,
                    metadata=metadata,
                    report_language=report_language,
                )
                reporting.write_report_files(paths.root, final_report)
                gpt_payload = {
                    "model": gpt_model,
                    "report_language": report_language,
                    "segment_analyses": segment_analyses,
                    "final_report": final_report,
                }

        return paths.root
    except Exception as exc:
        error = {"stage": "analyze", "message": str(exc)}
        errors.append(error)
        save_error(paths, error["stage"], error["message"])
        raise
    finally:
        if cleanup_intermediates:
            cleanup_intermediate_artifacts(paths)
        reporting.write_analysis_file(
            paths,
            source_input=source,
            is_url=looks_like_url(source),
            is_youtube_url=is_youtube_url(source),
            metadata=metadata,
            transcript=transcript,
            ocr=ocr_state,
            segments=segments,
            manifest_entries=manifest_entries,
            errors=errors,
            cleanup_intermediates=cleanup_intermediates,
            transcript_mode=transcript_mode,
            keyframe_mode=keyframe_mode,
            ocr_mode=ocr_mode,
            triage_mode=triage_mode,
            gpt_mode=gpt_mode,
            review_mode=review_mode,
            interval_seconds=interval_seconds,
            scene_threshold=scene_threshold,
            gpt_payload=gpt_payload,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a YouTube URL or local media file into transcripts, keyframes, triage, and reports."
    )
    parser.add_argument("--source", required=True, help="YouTube URL or local media path")
    parser.add_argument(
        "--transcript",
        default="auto",
        choices=["auto", "subtitles", "api", "whisper"],
        help="Transcript strategy",
    )
    parser.add_argument(
        "--keyframes",
        default="scene+interval",
        choices=["off", "scene", "interval", "scene+interval"],
        help="Keyframe extraction strategy",
    )
    parser.add_argument(
        "--ocr",
        default=constants.DEFAULT_OCR_MODE,
        choices=["auto", "off", "on"],
        help="Run local OCR on keyframes (`auto` is best-effort and non-fatal)",
    )
    parser.add_argument(
        "--triage",
        default=constants.DEFAULT_TRIAGE_MODE,
        choices=["off", "on"],
        help="Run local triage and routing artifact generation",
    )
    parser.add_argument(
        "--gpt",
        default=constants.DEFAULT_GPT_MODE,
        choices=["off", "on"],
        help="Run GPT analysis and final report synthesis",
    )
    parser.add_argument(
        "--review",
        default=constants.DEFAULT_REVIEW_MODE,
        choices=["off", "interactive"],
        help="Review mode before GPT routing",
    )
    parser.add_argument(
        "--gpt-model",
        default=constants.DEFAULT_GPT_MODEL,
        help="GPT model used for vision analysis and final synthesis",
    )
    parser.add_argument(
        "--report-language",
        default=constants.DEFAULT_REPORT_LANGUAGE,
        help="Human-facing report language",
    )
    parser.add_argument(
        "--review-reset",
        action="store_true",
        help="Ignore any saved review decisions and start review from scratch",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep downloaded media and other intermediate files instead of cleaning them after the run",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Override output directory (default: output/youtube/<video-id-or-stem>)",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=constants.DEFAULT_INTERVAL_SECONDS,
        help=f"Interval seconds for supplemental keyframes (default: {constants.DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=constants.DEFAULT_SCENE_THRESHOLD,
        help=f"Scene detection threshold (default: {constants.DEFAULT_SCENE_THRESHOLD})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = analyze_source(
        args.source,
        transcript_mode=args.transcript,
        keyframe_mode=args.keyframes,
        ocr_mode=args.ocr,
        out_dir=args.out_dir,
        interval_seconds=args.interval_seconds,
        scene_threshold=args.scene_threshold,
        triage_mode=args.triage,
        gpt_mode=args.gpt,
        review_mode=args.review,
        gpt_model=args.gpt_model,
        report_language=args.report_language,
        review_reset=args.review_reset,
        cleanup_intermediates=not args.keep_intermediates,
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
