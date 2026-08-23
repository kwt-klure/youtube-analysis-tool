from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen
from urllib.parse import parse_qs, urlparse
from xml.etree import ElementTree

from . import constants, gpt, reporting, review, routing, triage, visuals
from .artifacts import write_json
from .version import add_version_argument


HAN_CHARACTER_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_CHARACTER_PATTERN = re.compile(r"[A-Za-z]")
DIGIT_CHARACTER_PATTERN = re.compile(r"\d")
VISIBLE_CHARACTER_PATTERN = re.compile(r"\S")
PREFERRED_COMMAND_DIRS = (
    "/opt/homebrew/bin",
    "/opt/homebrew/sbin",
    "/usr/local/bin",
    "/usr/local/sbin",
)


def _strip_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def find_dotenv_path(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None


def load_dotenv_file(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue
        if key in os.environ:
            continue
        resolved = _strip_env_value(value)
        os.environ[key] = resolved
        loaded[key] = resolved
    return loaded


def load_local_env(start: Path | None = None) -> dict[str, str]:
    dotenv_path = find_dotenv_path(start)
    if dotenv_path is None:
        return {}
    return load_dotenv_file(dotenv_path)


def build_command_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if overrides:
        env.update(overrides)
    raw_path = env.get("PATH", "")
    entries = [entry for entry in raw_path.split(os.pathsep) if entry]
    for directory in reversed(PREFERRED_COMMAND_DIRS):
        if directory not in entries and Path(directory).exists():
            entries.insert(0, directory)
    interpreter_dir = str(Path(sys.executable).parent)
    if interpreter_dir not in entries:
        entries.insert(0, interpreter_dir)
    env["PATH"] = os.pathsep.join(entries)
    return env


def resolve_command(name: str) -> tuple[str | None, dict[str, str]]:
    env = build_command_env()
    resolved = shutil.which(name, path=env.get("PATH"))
    return resolved, env


@dataclass(frozen=True)
class AnalysisPaths:
    root: Path
    output_json_path: Path
    visuals_dir: Path
    visuals_manifest_path: Path
    visuals_slides_dir: Path
    visuals_charts_dir: Path
    audio_dir: Path
    video_dir: Path
    subtitles_dir: Path
    keyframes_dir: Path
    transcript_text_path: Path
    transcript_json_path: Path
    keyframe_index_path: Path
    visual_selection_path: Path
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


def youtube_video_id_from_source(source: str) -> str | None:
    if not is_youtube_url(source):
        return None

    parsed = urlparse(source)
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]

    if host.endswith("youtu.be"):
        return path_parts[0] if path_parts else None

    query_video_id = parse_qs(parsed.query).get("v", [None])[0]
    if query_video_id:
        return query_video_id

    for marker in ("shorts", "embed", "live"):
        if len(path_parts) >= 2 and path_parts[0] == marker:
            return path_parts[1]

    return None


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    chunks: list[str] = []
    separator_open = False
    for character in normalized:
        if character.isalnum():
            chunks.append(character.casefold())
            separator_open = False
            continue
        if chunks and not separator_open:
            chunks.append("-")
            separator_open = True
    slug = "".join(chunks).strip("-")
    return slug or "video"


def hms_from_seconds(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_progress_line(phase: str, message: str) -> str:
    return f"[{phase}] {message}"


class StderrProgressReporter:
    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stderr

    def __call__(self, phase: str, message: str) -> None:
        self.stream.write(format_progress_line(phase, message) + "\n")
        self.stream.flush()


def report_progress(progress_callback: Callable[[str, str], None] | None, phase: str, message: str) -> None:
    if progress_callback is None:
        return
    progress_callback(phase, message)


def make_download_progress_hook(
    progress_callback: Callable[[str, str], None] | None,
) -> Callable[[dict[str, Any]], None]:
    has_reported_start = False

    def hook(status: dict[str, Any]) -> None:
        nonlocal has_reported_start
        state = str(status.get("status") or "")
        if state == "downloading" and not has_reported_start:
            has_reported_start = True
            report_progress(progress_callback, "download", "Media transfer started")
        elif state == "finished":
            report_progress(progress_callback, "download", "Media transfer finished")

    return hook


def default_output_dir_for_source(
    source: str,
    video_id: str | None = None,
    video_title: str | None = None,
    output_root: Path | None = None,
) -> Path:
    base_root = output_root or constants.DEFAULT_OUTPUT_ROOT
    if video_id:
        video_id_slug = slugify(video_id)
        raw_title = str(video_title or "").strip()
        title_slug = slugify(raw_title) if raw_title else ""
        if title_slug:
            return base_root / f"{title_slug}-{video_id_slug}"
        return base_root / video_id_slug
    if looks_like_url(source):
        digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        return base_root / f"url-{digest}"
    return base_root / slugify(Path(source).stem)


def analysis_paths(root: Path) -> AnalysisPaths:
    return AnalysisPaths(
        root=root,
        output_json_path=root / "output.json",
        visuals_dir=root / "visuals",
        visuals_manifest_path=root / "visuals" / "manifest.json",
        visuals_slides_dir=root / "visuals" / "slides",
        visuals_charts_dir=root / "visuals" / "charts",
        audio_dir=root / "audio",
        video_dir=root / "video",
        subtitles_dir=root / "subtitles",
        keyframes_dir=root / "keyframes",
        transcript_text_path=root / "transcript.txt",
        transcript_json_path=root / "transcript.json",
        keyframe_index_path=root / "keyframes" / "index.csv",
        visual_selection_path=root / "visuals" / "selection.json",
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
    ):
        path.mkdir(parents=True, exist_ok=True)
    if with_ocr:
        paths.ocr_dir.mkdir(parents=True, exist_ok=True)


def clear_stale_output_files(paths: AnalysisPaths) -> None:
    for directory in (
        paths.audio_dir,
        paths.video_dir,
        paths.subtitles_dir,
        paths.keyframes_dir,
        paths.ocr_dir,
        paths.triage_dir,
        paths.review_dir,
        paths.routing_dir,
        paths.gpt_dir,
        paths.report_dir,
        paths.visuals_dir,
        paths.root / "tmp-whisper",
    ):
        shutil.rmtree(directory, ignore_errors=True)
    for path in (
        paths.output_json_path,
        paths.root / "analysis.json",
        paths.metadata_path,
        paths.source_path,
        paths.transcript_json_path,
        paths.transcript_text_path,
        paths.error_path,
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
    effective_env = build_command_env(env)
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
        env=effective_env,
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


def rms_db(samples: Any) -> float:
    import numpy

    if len(samples) == 0:
        return -100.0
    rms = float(numpy.sqrt(numpy.mean(numpy.square(samples.astype("float64")))))
    if rms <= 0.0:
        return -100.0
    return round(max(-100.0, 20.0 * math.log10(rms)), 3)


def audio_window_records(
    samples: Any,
    *,
    sample_rate: int,
    window_seconds: int,
) -> list[dict[str, Any]]:
    window_size = max(1, int(sample_rate * max(1, window_seconds)))
    records: list[dict[str, Any]] = []
    for start_index in range(0, len(samples), window_size):
        end_index = min(len(samples), start_index + window_size)
        start_seconds = round(start_index / float(sample_rate), 3)
        end_seconds = round(end_index / float(sample_rate), 3)
        records.append(
            {
                "start": start_seconds,
                "end": end_seconds,
                "rms_db": rms_db(samples[start_index:end_index]),
            }
        )
    return records


def audio_dynamic_range_hint(windows: list[dict[str, Any]]) -> str | None:
    if not windows:
        return None
    values = [float(window["rms_db"]) for window in windows]
    spread = max(values) - min(values)
    if spread < 6.0:
        return "compressed_or_flat"
    if spread < 12.0:
        return "moderate_variation"
    return "wide_or_high_contrast"


def audio_extreme_windows(
    windows: list[dict[str, Any]],
    *,
    quiet: bool,
    limit: int = constants.DEFAULT_AUDIO_TOP_SEGMENT_COUNT,
) -> list[dict[str, Any]]:
    sorted_windows = sorted(windows, key=lambda window: float(window["rms_db"]), reverse=not quiet)
    return [
        {
            "start": window.get("start"),
            "end": window.get("end"),
            "rms_db": window.get("rms_db"),
        }
        for window in sorted_windows[: max(0, limit)]
    ]


def audio_large_changes(
    windows: list[dict[str, Any]],
    *,
    threshold_db: float = constants.DEFAULT_AUDIO_LARGE_CHANGE_DB,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for previous, current in zip(windows, windows[1:]):
        from_dbfs = float(previous["rms_db"])
        to_dbfs = float(current["rms_db"])
        delta = round(to_dbfs - from_dbfs, 3)
        if abs(delta) < threshold_db:
            continue
        changes.append(
            {
                "start": previous.get("start"),
                "end": current.get("end"),
                "from_dbfs": from_dbfs,
                "to_dbfs": to_dbfs,
                "delta_db": delta,
                "interpretation_hint": "possible location, scene, or production-context change",
            }
        )
    return changes


def detect_silence_segments(
    samples: Any,
    *,
    sample_rate: int,
    silence_threshold_db: float,
    min_duration_seconds: float,
) -> list[dict[str, Any]]:
    chunk_size = max(1, int(sample_rate * 0.1))
    segments: list[dict[str, Any]] = []
    active_start: float | None = None
    active_end: float | None = None
    for start_index in range(0, len(samples), chunk_size):
        end_index = min(len(samples), start_index + chunk_size)
        chunk_db = rms_db(samples[start_index:end_index])
        chunk_start = start_index / float(sample_rate)
        chunk_end = end_index / float(sample_rate)
        if chunk_db <= silence_threshold_db:
            if active_start is None:
                active_start = chunk_start
            active_end = chunk_end
            continue
        if active_start is not None and active_end is not None:
            duration = active_end - active_start
            if duration >= min_duration_seconds:
                segments.append(
                    {
                        "start": round(active_start, 3),
                        "end": round(active_end, 3),
                        "duration": round(duration, 3),
                    }
                )
            active_start = None
            active_end = None
    if active_start is not None and active_end is not None:
        duration = active_end - active_start
        if duration >= min_duration_seconds:
            segments.append(
                {
                    "start": round(active_start, 3),
                    "end": round(active_end, 3),
                    "duration": round(duration, 3),
                }
            )
    return segments


def default_audio_features_payload(mode: str) -> dict[str, Any]:
    return {
        "status": "disabled" if mode == "off" else "not_attempted",
        "source": None,
        "summary": {},
        "silence_segments": [],
        "windows": [],
        "quiet_segments": [],
        "loud_segments": [],
        "large_changes": [],
        "interpretation_warning": "audio features are structural evidence and routing signals, not semantic conclusions",
        "top_quiet_windows": [],
        "top_loud_windows": [],
        "provenance": {
            "method": None,
            "threshold_db": constants.DEFAULT_AUDIO_SILENCE_THRESHOLD_DB,
            "trust": "structural_signal_not_semantics",
            "quality_notes": [
                "audio features are evidence, not interpretation",
                "similar silence or loudness patterns can mean different things in different video genres",
            ],
        },
    }


def extract_audio_features(
    media_path: Path,
    metadata: dict[str, Any],
    *,
    sample_rate: int = constants.DEFAULT_AUDIO_FEATURE_SAMPLE_RATE,
    window_seconds: int = constants.DEFAULT_AUDIO_FEATURE_WINDOW_SECONDS,
    silence_threshold_db: float = constants.DEFAULT_AUDIO_SILENCE_THRESHOLD_DB,
    silence_min_duration: float = constants.DEFAULT_AUDIO_SILENCE_MIN_DURATION_SECONDS,
) -> dict[str, Any]:
    import numpy

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        env=build_command_env(),
    )
    samples = numpy.frombuffer(result.stdout, dtype=numpy.int16).astype("float32") / 32768.0
    measured_duration = len(samples) / float(sample_rate) if sample_rate > 0 else 0.0
    duration = duration_seconds(metadata) or measured_duration
    windows = audio_window_records(
        samples,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
    )
    silence_segments = detect_silence_segments(
        samples,
        sample_rate=sample_rate,
        silence_threshold_db=silence_threshold_db,
        min_duration_seconds=silence_min_duration,
    )
    silence_duration = sum(float(segment["duration"]) for segment in silence_segments)
    silence_ratio = round(silence_duration / duration, 3) if duration > 0 else 0.0
    quiet_windows = sorted(windows, key=lambda window: float(window["rms_db"]))[:3]
    loud_windows = sorted(windows, key=lambda window: float(window["rms_db"]), reverse=True)[:3]
    mean_dbfs = rms_db(samples)
    return {
        "status": "extracted",
        "source": "local_ffmpeg_python_rms",
        "summary": {
            "duration_seconds": round(duration, 3),
            "silence_ratio": silence_ratio,
            "mean_loudness_db": mean_dbfs,
            "mean_dbfs": mean_dbfs,
            "dynamic_range_hint": audio_dynamic_range_hint(windows),
            "window_seconds": window_seconds,
        },
        "silence_segments": silence_segments,
        "windows": windows,
        "quiet_segments": audio_extreme_windows(windows, quiet=True),
        "loud_segments": audio_extreme_windows(windows, quiet=False),
        "large_changes": audio_large_changes(windows),
        "interpretation_warning": "audio features are structural evidence and routing signals, not semantic conclusions",
        "top_quiet_windows": quiet_windows,
        "top_loud_windows": loud_windows,
        "provenance": {
            "method": "ffmpeg_pcm_decode_plus_python_rms_windows",
            "threshold_db": silence_threshold_db,
            "sample_rate": sample_rate,
            "trust": "structural_signal_not_semantics",
            "quality_notes": [
                "audio features are evidence, not interpretation",
                "silence and loudness patterns need genre and context before semantic claims",
            ],
        },
    }


def run_audio_features_stage(
    media_path: Path | None,
    metadata: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode == "off":
        return default_audio_features_payload(mode)
    if media_path is None:
        payload = default_audio_features_payload(mode)
        payload["status"] = "not_applicable"
        payload["error"] = "no_media_path_available"
        return payload
    try:
        return extract_audio_features(media_path, metadata)
    except Exception as exc:
        payload = default_audio_features_payload(mode)
        payload["status"] = "failed"
        payload["error"] = str(exc)
        return payload


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
    marker_index = None
    for index in range(len(suffixes) - 2, 0, -1):
        if suffixes[index].lower() in {"manual", "auto"}:
            marker_index = index
            break
    if marker_index is not None:
        if marker_index + 1 < len(suffixes) - 1:
            return suffixes[marker_index + 1].lower()
        if marker_index - 1 >= 1:
            return suffixes[marker_index - 1].lower()
    return suffixes[-2].lower()


def detect_subtitle_source_from_filename(path: Path) -> str:
    suffixes = path.name.split(".")
    for index in range(len(suffixes) - 2, 0, -1):
        token = suffixes[index].lower()
        if token == "auto":
            return "subtitle_auto"
        if token == "manual":
            return "subtitle_manual"
    return "subtitle_manual"


def subtitle_rank(path: Path) -> tuple[int, int, int, str]:
    language = detect_language_from_filename(path)
    source_kind = detect_subtitle_source_from_filename(path)
    ext = path.suffix.lower().lstrip(".")
    return (
        0 if source_kind == "subtitle_manual" else 1,
        subtitle_language_rank(language),
        subtitle_ext_rank(ext),
        path.name,
    )


def subtitle_language_rank(language: str | None) -> int:
    normalized = (language or "zz").lower()
    try:
        return constants.SUBTITLE_LANGUAGE_PREFERENCE.index(normalized)
    except ValueError:
        return len(constants.SUBTITLE_LANGUAGE_PREFERENCE)


def subtitle_ext_rank(ext: str | None) -> int:
    preferred = ("vtt", "srt", "json3", "ttml", "srv3", "srv2", "srv1")
    normalized = (ext or "").lower()
    try:
        return preferred.index(normalized)
    except ValueError:
        return len(preferred)


def subtitle_bucket_rank(bucket_name: str | None) -> int:
    if bucket_name == "subtitles":
        return 0
    if bucket_name == "automatic_captions":
        return 1
    return 2


def subtitle_bucket_marker(bucket_name: str | None) -> str:
    if bucket_name == "automatic_captions":
        return "auto"
    return "manual"


def is_translated_subtitle_track(item: dict[str, Any]) -> bool:
    url = str(item.get("url") or "")
    return "tlang=" in url


def choose_subtitle_file(subtitles_dir: Path) -> Path | None:
    candidates = [
        path
        for path in subtitles_dir.glob("*")
        if path.is_file()
        and path.suffix.lower() in {".vtt", ".srt", ".json3", ".ttml", ".srv3", ".srv2", ".srv1"}
        and "live_chat" not in path.name.lower()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=subtitle_rank)[0]


def choose_subtitle_track_from_metadata(metadata: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    candidates: list[tuple[int, int, int, str, str, dict[str, Any]]] = []
    allowed_extensions = {"vtt", "srt", "json3", "ttml", "srv3", "srv2", "srv1"}
    for bucket_name in ("subtitles", "automatic_captions"):
        bucket = metadata.get(bucket_name) or {}
        for language, items in bucket.items():
            for item in items:
                if is_translated_subtitle_track(item):
                    continue
                ext = str(item.get("ext", "")).lower()
                if ext not in allowed_extensions:
                    continue
                candidates.append(
                    (
                        subtitle_bucket_rank(bucket_name),
                        subtitle_language_rank(language),
                        subtitle_ext_rank(ext),
                        bucket_name,
                        language,
                        item,
                    )
                )
    if not candidates:
        return None
    _, _, _, bucket_name, language, item = sorted(candidates, key=lambda row: row[:3])[0]
    return bucket_name, language, item


def preferred_subtitle_languages() -> list[str]:
    return [language for language in constants.SUBTITLE_LANGUAGE_PREFERENCE]


def download_selected_subtitle_track(
    source_url: str,
    paths: AnalysisPaths,
    *,
    bucket_name: str,
    language: str,
    ext: str,
    progress_callback: Callable[[str, str], None] | None = None,
) -> Path | None:
    marker = subtitle_bucket_marker(bucket_name)
    before = {path.name for path in paths.subtitles_dir.glob("*")}
    subtitle_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": constants.DEFAULT_YTDLP_MEDIA_SOCKET_TIMEOUT_SECONDS,
        "skip_download": True,
        "writesubtitles": bucket_name == "subtitles",
        "writeautomaticsub": bucket_name == "automatic_captions",
        "subtitleslangs": [language],
        "subtitlesformat": ext,
        "outtmpl": {
            "subtitle": str(paths.subtitles_dir / f"%(id)s.%(language)s.{marker}.%(ext)s"),
        },
    }
    _run_yt_dlp_extract_info(
        subtitle_opts,
        source_url,
        download=True,
        progress_callback=progress_callback,
        phase="transcript",
        retry_message="Subtitle track fetch stalled",
    )
    after = {
        path.name
        for path in paths.subtitles_dir.glob("*")
        if path.is_file() and "live_chat" not in path.name.lower()
    }
    new_files = sorted(after - before)
    if new_files:
        return paths.subtitles_dir / new_files[0]
    return choose_subtitle_file(paths.subtitles_dir)


def download_subtitle_from_metadata(
    metadata: dict[str, Any],
    paths: AnalysisPaths,
    *,
    source_url: str | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> Path | None:
    selection = choose_subtitle_track_from_metadata(metadata)
    if selection is None:
        return None
    bucket_name, language, item = selection
    ext = str(item.get("ext", "vtt")).lower()
    if source_url is not None:
        try:
            result = download_selected_subtitle_track(
                source_url,
                paths,
                bucket_name=bucket_name,
                language=language,
                ext=ext,
                progress_callback=progress_callback,
            )
            if result is not None:
                return result
        except Exception:
            pass
    url = item.get("url")
    if not url:
        return None
    marker = subtitle_bucket_marker(bucket_name)
    subtitle_path = paths.subtitles_dir / f"{metadata.get('id', 'video')}.{language}.{marker}.{ext}"
    payload = _read_url_bytes_with_retry(
        str(url),
        progress_callback=progress_callback,
        phase="transcript",
        retry_message="Metadata subtitle fetch stalled",
    )
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


def parse_json3(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    segments: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        segs = event.get("segs") or []
        text = "".join(str(part.get("utf8", "")) for part in segs if isinstance(part, dict))
        text = clean_caption_text([html.unescape(text)])
        if not text:
            continue
        start_ms = event.get("tStartMs")
        duration_ms = event.get("dDurationMs")
        if start_ms is None or duration_ms is None:
            continue
        start = round(float(start_ms) / 1000.0, 3)
        end = round(start + (float(duration_ms) / 1000.0), 3)
        segments.append({"start": start, "end": end, "text": text})
    return segments


def parse_xml_timestamp(raw: str | None) -> float | None:
    if raw is None:
        return None
    token = str(raw).strip()
    if not token:
        return None
    if token.endswith("ms"):
        return float(token[:-2]) / 1000.0
    if token.endswith("s"):
        return float(token[:-1])
    if ":" in token:
        return parse_timestamp(token)
    return float(token)


def parse_xml_subtitles(path: Path) -> list[dict[str, Any]]:
    root = ElementTree.fromstring(path.read_text(encoding="utf-8", errors="ignore"))
    segments: list[dict[str, Any]] = []
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()
        if tag not in {"text", "p"}:
            continue
        raw_text = "".join(element.itertext())
        text = clean_caption_text([html.unescape(raw_text)])
        if not text:
            continue
        start = parse_xml_timestamp(
            element.attrib.get("start") or element.attrib.get("begin") or element.attrib.get("t")
        )
        end = parse_xml_timestamp(element.attrib.get("end"))
        dur = parse_xml_timestamp(element.attrib.get("dur") or element.attrib.get("d"))
        if start is None:
            continue
        if end is None:
            if dur is None:
                continue
            end = start + dur
        segments.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return segments


def parse_subtitle_file(path: Path) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext in {".vtt", ".srt"}:
        return parse_srt_or_vtt(path)
    if ext == ".json3":
        return parse_json3(path)
    if ext in {".ttml", ".srv3", ".srv2", ".srv1"}:
        return parse_xml_subtitles(path)
    raise ValueError(f"Unsupported subtitle format: {path.suffix}")


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


def skipped_transcript(reason: str = "transcript_mode_off") -> dict[str, Any]:
    return {
        "source": "skipped",
        "language": None,
        "source_path": None,
        "segment_count": 0,
        "text": "",
        "segments": [],
        "status": "skipped",
        "skip_reason": reason,
    }


def normalize_reused_transcript_payload(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    transcript_payload = payload.get("transcript") if isinstance(payload.get("transcript"), dict) else payload
    segments = transcript_payload.get("segments") or []
    text = transcript_payload.get("text")
    if text is None:
        text = transcript_payload.get("full_text", "")
    transcript = dict(transcript_payload)
    transcript["text"] = text or ""
    transcript["segments"] = segments
    transcript["segment_count"] = transcript_payload.get("segment_count", len(segments))
    provenance = transcript_payload.get("provenance")
    if isinstance(provenance, dict):
        for key in ("backend", "model"):
            if not transcript.get(key) and provenance.get(key):
                transcript[key] = provenance[key]
    transcript["status"] = "reused"
    transcript["reused_from"] = str(source_path)
    return transcript


def load_reused_transcript(path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    source_path = path.expanduser().resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Reusable transcript must be a JSON object: {source_path}")
    transcript = normalize_reused_transcript_payload(payload, source_path)
    write_transcript(paths, transcript)
    return transcript


def transcript_from_subtitles(subtitle_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    segments = parse_subtitle_file(subtitle_path)
    language = detect_language_from_filename(subtitle_path)
    transcript = transcript_from_segments(
        segments,
        source=detect_subtitle_source_from_filename(subtitle_path),
        language=language,
        source_path=str(subtitle_path.relative_to(paths.root)),
    )
    write_transcript(paths, transcript)
    return transcript


def default_burned_subtitles_state(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "status": "not_attempted",
        "reason": None,
        "attempted": False,
        "probe_passed": False,
        "probe_hits": 0,
        "ocr_event_count": 0,
        "error": None,
    }


def video_dimensions(metadata: dict[str, Any]) -> tuple[int, int] | None:
    for stream in metadata.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        width = stream.get("width")
        height = stream.get("height")
        if width and height:
            return int(width), int(height)
    return None


def burned_subtitle_crop_geometry(metadata: dict[str, Any]) -> tuple[int, int, int, int] | None:
    dimensions = video_dimensions(metadata)
    if dimensions is None:
        return None
    width, height = dimensions
    crop_width = max(1, int(round(width * (1.0 - (constants.DEFAULT_BURNED_SUBTITLE_SIDE_MARGIN_RATIO * 2.0)))))
    crop_height = max(1, int(round(height * constants.DEFAULT_BURNED_SUBTITLE_BAND_HEIGHT_RATIO)))
    crop_x = max(0, int(round(width * constants.DEFAULT_BURNED_SUBTITLE_SIDE_MARGIN_RATIO)))
    crop_y = max(0, height - crop_height)
    return crop_width, crop_height, crop_x, crop_y


def count_cjk_characters(text: str) -> int:
    return len(HAN_CHARACTER_PATTERN.findall(str(text or "")))


def normalize_burned_subtitle_text(text: str) -> str:
    normalized = clean_caption_text([str(text or "").replace("\u3000", " ")])
    return normalized.strip()


def burned_subtitle_text_metrics(text: str) -> dict[str, float]:
    normalized = str(text or "").strip()
    visible_char_count = len(VISIBLE_CHARACTER_PATTERN.findall(normalized))
    cjk_char_count = count_cjk_characters(normalized)
    latin_char_count = len(LATIN_CHARACTER_PATTERN.findall(normalized))
    digit_char_count = len(DIGIT_CHARACTER_PATTERN.findall(normalized))
    symbol_count = max(0, visible_char_count - cjk_char_count - latin_char_count - digit_char_count)
    if visible_char_count <= 0:
        cjk_ratio = 0.0
        noise_ratio = 1.0
    else:
        cjk_ratio = cjk_char_count / visible_char_count
        noise_ratio = symbol_count / visible_char_count
    return {
        "visible_char_count": float(visible_char_count),
        "cjk_char_count": float(cjk_char_count),
        "latin_char_count": float(latin_char_count),
        "digit_char_count": float(digit_char_count),
        "symbol_count": float(symbol_count),
        "cjk_ratio": float(cjk_ratio),
        "noise_ratio": float(noise_ratio),
    }


def is_effective_burned_subtitle_text(text: str) -> bool:
    metrics = burned_subtitle_text_metrics(text)
    return (
        metrics["visible_char_count"] >= constants.DEFAULT_BURNED_SUBTITLE_MIN_VISIBLE_CHARS
        and metrics["cjk_char_count"] >= 2
        and metrics["cjk_ratio"] >= constants.DEFAULT_BURNED_SUBTITLE_MIN_CJK_RATIO
        and metrics["noise_ratio"] <= constants.DEFAULT_BURNED_SUBTITLE_MAX_NOISE_RATIO
    )


def available_tesseract_languages() -> set[str]:
    tesseract_bin, env = resolve_command("tesseract")
    if tesseract_bin is None:
        return set()
    result = run_command([tesseract_bin, "--list-langs"], env=env)
    languages: set[str] = set()
    for line in result.stdout.splitlines():
        token = line.strip()
        if not token or token.lower().startswith("list of available languages"):
            continue
        languages.add(token)
    return languages


def choose_burned_subtitle_tesseract_languages() -> str | None:
    available = available_tesseract_languages()
    required_languages = [language for language in constants.BURNED_SUBTITLE_TESSERACT_LANGS if language in available]
    if len(required_languages) != len(constants.BURNED_SUBTITLE_TESSERACT_LANGS):
        return None
    optional_languages = [
        language
        for language in constants.BURNED_SUBTITLE_OPTIONAL_TESSERACT_LANGS
        if language in available
    ]
    return "+".join(required_languages + optional_languages)


def burned_subtitle_policy(mode: str) -> dict[str, float | int]:
    if mode == "auto":
        return {
            "sample_fps": constants.DEFAULT_BURNED_SUBTITLE_AUTO_SAMPLE_FPS,
            "probe_min_hits": constants.DEFAULT_BURNED_SUBTITLE_AUTO_PROBE_MIN_HITS,
            "probe_min_distinct_texts": constants.DEFAULT_BURNED_SUBTITLE_AUTO_PROBE_MIN_DISTINCT_TEXTS,
            "quick_reject_seconds": constants.DEFAULT_BURNED_SUBTITLE_AUTO_QUICK_REJECT_SECONDS,
            "quick_reject_events": constants.DEFAULT_BURNED_SUBTITLE_AUTO_QUICK_REJECT_EVENTS,
            "early_gate_seconds": constants.DEFAULT_BURNED_SUBTITLE_AUTO_EARLY_GATE_SECONDS,
            "early_gate_events": constants.DEFAULT_BURNED_SUBTITLE_AUTO_EARLY_GATE_EVENTS,
            "min_nonempty_hits": constants.DEFAULT_BURNED_SUBTITLE_AUTO_MIN_NONEMPTY_HITS,
            "min_hit_rate": constants.DEFAULT_BURNED_SUBTITLE_AUTO_MIN_HIT_RATE,
            "min_cjk_chars": constants.DEFAULT_BURNED_SUBTITLE_AUTO_MIN_CJK_CHARS,
            "min_cjk_ratio": constants.DEFAULT_BURNED_SUBTITLE_AUTO_MIN_CJK_RATIO,
        }
    return {
        "sample_fps": constants.DEFAULT_BURNED_SUBTITLE_SAMPLE_FPS,
        "probe_min_hits": constants.DEFAULT_BURNED_SUBTITLE_PROBE_MIN_HITS,
        "probe_min_distinct_texts": constants.DEFAULT_BURNED_SUBTITLE_PROBE_MIN_DISTINCT_TEXTS,
        "quick_reject_seconds": constants.DEFAULT_BURNED_SUBTITLE_QUICK_REJECT_SECONDS,
        "quick_reject_events": constants.DEFAULT_BURNED_SUBTITLE_QUICK_REJECT_EVENTS,
        "early_gate_seconds": constants.DEFAULT_BURNED_SUBTITLE_EARLY_GATE_SECONDS,
        "early_gate_events": constants.DEFAULT_BURNED_SUBTITLE_EARLY_GATE_EVENTS,
        "min_nonempty_hits": constants.DEFAULT_BURNED_SUBTITLE_MIN_NONEMPTY_HITS,
        "min_hit_rate": constants.DEFAULT_BURNED_SUBTITLE_MIN_HIT_RATE,
        "min_cjk_chars": constants.DEFAULT_BURNED_SUBTITLE_MIN_CJK_CHARS,
        "min_cjk_ratio": constants.DEFAULT_BURNED_SUBTITLE_MIN_CJK_RATIO,
    }


def iter_subtitle_band_frames(
    video_path: Path,
    metadata: dict[str, Any],
    *,
    sample_fps: float,
    duration_limit: float | None = None,
):
    import numpy

    geometry = burned_subtitle_crop_geometry(metadata)
    if geometry is None:
        raise RuntimeError("Unable to determine subtitle band geometry from video metadata.")
    crop_width, crop_height, crop_x, crop_y = geometry
    frame_size = crop_width * crop_height
    if frame_size <= 0:
        raise RuntimeError("Computed subtitle band geometry is invalid.")

    filter_parts = [
        f"fps={sample_fps}",
        f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}",
        "format=gray",
    ]
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
    ]
    if duration_limit is not None and duration_limit > 0:
        command.extend(["-t", str(duration_limit)])
    command.extend(
        [
            "-vf",
            ",".join(filter_parts),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=build_command_env(),
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to open ffmpeg pipe for burned subtitle OCR.")

    sample_interval = 1.0 / sample_fps
    frame_index = 0
    stderr_text = ""
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk:
                break
            if len(chunk) < frame_size:
                break
            image = numpy.frombuffer(chunk, dtype=numpy.uint8).reshape((crop_height, crop_width))
            yield {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index * sample_interval, 3),
                "image": image,
            }
            frame_index += 1
    finally:
        process.stdout.close()
        stderr_text = process.stderr.read().decode("utf-8", errors="ignore")
        process.stderr.close()
        return_code = process.wait()
        if return_code not in {0, None}:
            message = stderr_text.strip() or f"ffmpeg exited with code {return_code}"
            if "Broken pipe" in message:
                return
            raise RuntimeError(f"Burned subtitle frame extraction failed: {message}")


def preprocess_burned_subtitle_image(image):
    import cv2

    if image is None:
        raise RuntimeError("Subtitle band image is empty.")
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    height = thresholded.shape[0]
    if height < constants.DEFAULT_BURNED_SUBTITLE_UPSCALE_HEIGHT_THRESHOLD and height > 0:
        upscale = min(
            constants.DEFAULT_BURNED_SUBTITLE_MAX_UPSCALE,
            constants.DEFAULT_BURNED_SUBTITLE_UPSCALE_HEIGHT_THRESHOLD / float(height),
        )
        if upscale > 1.0:
            return cv2.resize(thresholded, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    return thresholded


def burned_subtitle_detection_roi(image):
    if image is None:
        return None
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image
    roi_width = max(1, int(round(width * constants.DEFAULT_BURNED_SUBTITLE_DETECTION_WIDTH_RATIO)))
    roi_height = max(1, int(round(height * constants.DEFAULT_BURNED_SUBTITLE_DETECTION_HEIGHT_RATIO)))
    roi_x = max(0, int(round((width - roi_width) / 2.0)))
    roi_y = max(0, height - roi_height)
    return image[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]


def subtitle_band_diff(left, right) -> float:
    import numpy

    if left is None or right is None:
        return 1.0
    if left.shape != right.shape:
        return 1.0
    return float(numpy.mean(numpy.abs(left.astype("float32") - right.astype("float32"))) / 255.0)


def ocr_burned_subtitle_image(image, *, lang: str) -> str:
    import pytesseract

    text = pytesseract.image_to_string(image, lang=lang, config="--psm 6")
    return normalize_burned_subtitle_text(text)


def probe_burned_subtitles(
    video_path: Path,
    metadata: dict[str, Any],
    *,
    tesseract_langs: str,
    mode: str = "on",
) -> dict[str, Any]:
    hits = 0
    distinct_texts: list[str] = []
    sample_count = 0
    policy = burned_subtitle_policy(mode)
    probe_fps = constants.DEFAULT_BURNED_SUBTITLE_PROBE_SAMPLES / constants.DEFAULT_BURNED_SUBTITLE_PROBE_DURATION_SECONDS
    for frame in iter_subtitle_band_frames(
        video_path,
        metadata,
        sample_fps=probe_fps,
        duration_limit=constants.DEFAULT_BURNED_SUBTITLE_PROBE_DURATION_SECONDS,
    ):
        sample_count += 1
        text = ocr_burned_subtitle_image(
            preprocess_burned_subtitle_image(frame["image"]),
            lang=tesseract_langs,
        )
        if not is_effective_burned_subtitle_text(text):
            continue
        hits += 1
        if text not in distinct_texts:
            distinct_texts.append(text)
    passed = (
        hits >= int(policy["probe_min_hits"])
        and len(distinct_texts) >= int(policy["probe_min_distinct_texts"])
    )
    return {
        "passed": passed,
        "sample_count": sample_count,
        "hits": hits,
        "distinct_text_count": len(distinct_texts),
    }


def burned_subtitle_quality_is_insufficient(
    *,
    ocr_event_count: int,
    nonempty_hits: int,
    cjk_char_count: int,
    average_cjk_ratio: float = 0.0,
    min_nonempty_hits: int = constants.DEFAULT_BURNED_SUBTITLE_MIN_NONEMPTY_HITS,
    min_hit_rate: float = constants.DEFAULT_BURNED_SUBTITLE_MIN_HIT_RATE,
    min_cjk_chars: int = constants.DEFAULT_BURNED_SUBTITLE_MIN_CJK_CHARS,
    min_cjk_ratio: float = constants.DEFAULT_BURNED_SUBTITLE_MIN_CJK_RATIO,
) -> bool:
    hit_rate = 0.0 if ocr_event_count <= 0 else (nonempty_hits / ocr_event_count)
    return (
        nonempty_hits < min_nonempty_hits
        or hit_rate < min_hit_rate
        or cjk_char_count < min_cjk_chars
        or average_cjk_ratio < min_cjk_ratio
    )


def transcribe_burned_subtitles(
    video_path: Path,
    metadata: dict[str, Any],
    *,
    tesseract_langs: str,
    mode: str = "on",
) -> dict[str, Any]:
    segments: list[dict[str, Any]] = []
    current_text: str | None = None
    current_start: float | None = None
    current_end: float | None = None
    previous_image = None
    policy = burned_subtitle_policy(mode)
    sample_interval = 1.0 / float(policy["sample_fps"])
    ocr_event_count = 0
    nonempty_hits = 0
    cjk_char_count = 0
    visible_char_total = 0.0
    cjk_ratio_total = 0.0
    noise_ratio_total = 0.0
    quick_reject_checked = False
    early_gate_checked = False

    def close_segment() -> None:
        nonlocal current_text, current_start, current_end
        if current_text and current_start is not None and current_end is not None:
            segments.append(
                {
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                    "text": current_text,
                }
            )
        current_text = None
        current_start = None
        current_end = None

    def should_fail_quick_reject(force: bool, timestamp_seconds: float) -> bool:
        nonlocal quick_reject_checked
        if quick_reject_checked:
            return False
        if not force and (
            timestamp_seconds < float(policy["quick_reject_seconds"])
            and ocr_event_count < int(policy["quick_reject_events"])
        ):
            return False
        quick_reject_checked = True
        average_visible_chars = 0.0 if ocr_event_count <= 0 else (visible_char_total / ocr_event_count)
        average_cjk_ratio = 0.0 if ocr_event_count <= 0 else (cjk_ratio_total / ocr_event_count)
        average_noise_ratio = 1.0 if ocr_event_count <= 0 else (noise_ratio_total / ocr_event_count)
        return (
            nonempty_hits < constants.DEFAULT_BURNED_SUBTITLE_QUICK_MIN_NONEMPTY_HITS
            or average_visible_chars < constants.DEFAULT_BURNED_SUBTITLE_QUICK_MIN_VISIBLE_CHARS
            or average_cjk_ratio < constants.DEFAULT_BURNED_SUBTITLE_QUICK_MIN_CJK_RATIO
            or average_noise_ratio > constants.DEFAULT_BURNED_SUBTITLE_QUICK_MAX_NOISE_RATIO
        )

    def should_fail_quality_gate(force: bool, timestamp_seconds: float) -> bool:
        nonlocal early_gate_checked
        if early_gate_checked:
            return False
        if not force and (
            timestamp_seconds < float(policy["early_gate_seconds"])
            and ocr_event_count < int(policy["early_gate_events"])
        ):
            return False
        early_gate_checked = True
        average_cjk_ratio = 0.0 if ocr_event_count <= 0 else (cjk_ratio_total / ocr_event_count)
        return burned_subtitle_quality_is_insufficient(
            ocr_event_count=ocr_event_count,
            nonempty_hits=nonempty_hits,
            cjk_char_count=cjk_char_count,
            average_cjk_ratio=average_cjk_ratio,
            min_nonempty_hits=int(policy["min_nonempty_hits"]),
            min_hit_rate=float(policy["min_hit_rate"]),
            min_cjk_chars=int(policy["min_cjk_chars"]),
            min_cjk_ratio=float(policy["min_cjk_ratio"]),
        )

    for frame in iter_subtitle_band_frames(
        video_path,
        metadata,
        sample_fps=float(policy["sample_fps"]),
    ):
        timestamp_seconds = float(frame["timestamp_seconds"])
        processed_image = preprocess_burned_subtitle_image(frame["image"])
        detection_image = burned_subtitle_detection_roi(processed_image)
        if previous_image is not None:
            difference = subtitle_band_diff(previous_image, detection_image)
            if difference < constants.DEFAULT_BURNED_SUBTITLE_DIFF_THRESHOLD:
                if current_text:
                    current_end = round(timestamp_seconds + sample_interval, 3)
                continue
        previous_image = detection_image
        ocr_event_count += 1
        text = ocr_burned_subtitle_image(processed_image, lang=tesseract_langs)
        metrics = burned_subtitle_text_metrics(text)
        visible_char_total += metrics["visible_char_count"]
        cjk_ratio_total += metrics["cjk_ratio"]
        noise_ratio_total += metrics["noise_ratio"]
        effective_text = text if is_effective_burned_subtitle_text(text) else ""
        if effective_text:
            nonempty_hits += 1
            cjk_char_count += count_cjk_characters(effective_text)

        if effective_text == current_text and current_text:
            current_end = round(timestamp_seconds + sample_interval, 3)
        else:
            close_segment()
            if effective_text:
                current_text = effective_text
                current_start = timestamp_seconds
                current_end = round(timestamp_seconds + sample_interval, 3)

        if should_fail_quick_reject(False, timestamp_seconds):
            return {
                "segments": [],
                "ocr_event_count": ocr_event_count,
                "status": "fast_reject",
            }
        if should_fail_quality_gate(False, timestamp_seconds):
            return {
                "segments": [],
                "ocr_event_count": ocr_event_count,
                "status": "fallback_to_whisper_quality",
            }

    if should_fail_quick_reject(True, float(policy["quick_reject_seconds"])):
        return {
            "segments": [],
            "ocr_event_count": ocr_event_count,
            "status": "fast_reject",
        }
    if should_fail_quality_gate(True, float(policy["early_gate_seconds"])):
        return {
            "segments": [],
            "ocr_event_count": ocr_event_count,
            "status": "fallback_to_whisper_quality",
        }

    close_segment()
    return {
        "segments": segments,
        "ocr_event_count": ocr_event_count,
        "status": "completed",
    }


def run_burned_subtitles_stage(
    video_path: Path | None,
    metadata: dict[str, Any],
    paths: AnalysisPaths,
    *,
    mode: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    state = default_burned_subtitles_state(mode)
    if mode == "off":
        state["status"] = "disabled"
        state["reason"] = "disabled"
        return None, state
    resolved_metadata = metadata
    if video_path is not None and (
        not has_video_stream(resolved_metadata) or video_dimensions(resolved_metadata) is None
    ):
        try:
            resolved_metadata = ffprobe_json(video_path)
        except Exception:
            pass
    if video_path is None or not has_video_stream(resolved_metadata):
        state["status"] = "not_applicable"
        state["reason"] = "not_applicable"
        return None, state

    state["attempted"] = True
    tesseract_langs = choose_burned_subtitle_tesseract_languages()
    if tesseract_langs is None:
        state["status"] = "missing_language_pack"
        state["reason"] = "missing_language_pack"
        return None, state

    try:
        if mode == "auto":
            probe = probe_burned_subtitles(video_path, resolved_metadata, tesseract_langs=tesseract_langs, mode=mode)
            state["probe_passed"] = bool(probe["passed"])
            state["probe_hits"] = int(probe["hits"])
            if not probe["passed"]:
                state["status"] = "probe_rejected"
                state["reason"] = "probe_rejected"
                return None, state
        else:
            state["probe_passed"] = True

        result = transcribe_burned_subtitles(video_path, resolved_metadata, tesseract_langs=tesseract_langs, mode=mode)
        state["ocr_event_count"] = int(result["ocr_event_count"])
        if result["status"] != "completed":
            state["status"] = str(result["status"])
            state["reason"] = str(result["status"])
            return None, state

        transcript = transcript_from_segments(
            result["segments"],
            source="burned_subtitle_ocr",
            language=None,
        )
        write_transcript(paths, transcript)
        state["status"] = "completed"
        state["reason"] = "completed"
        return transcript, state
    except Exception as exc:
        state["status"] = "failed"
        state["reason"] = "failed"
        state["error"] = str(exc)
        return None, state


def extract_audio(input_media_path: Path, audio_dir: Path) -> Path:
    audio_path = audio_dir / "normalized.wav"
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


def extract_text_from_transcript_payload(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"].strip()
    segments = payload.get("segments") or []
    return "\n".join(segment.get("text", "").strip() for segment in segments if segment.get("text"))


def create_openai_client(api_key: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The optional `openai` package is required for API transcription."
        ) from exc
    return OpenAI(api_key=api_key)


def openai_response_payload(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
    elif isinstance(response, dict):
        payload = response
    else:
        raise TypeError("OpenAI transcription returned an unsupported response type.")
    if not isinstance(payload, dict):
        raise TypeError("OpenAI transcription response did not serialize to an object.")
    return payload


def transcribe_with_openai_api(audio_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set for API transcription.")
    client = create_openai_client(api_key)
    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model=constants.DEFAULT_REMOTE_ASR_MODEL,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    payload = openai_response_payload(response)
    transcript = {
        "source": "openai",
        "backend": "openai-api",
        "model": constants.DEFAULT_REMOTE_ASR_MODEL,
        "language": payload.get("language"),
        "source_path": str(audio_path.relative_to(paths.root)),
        "segment_count": len(payload.get("segments") or []),
        "text": extract_text_from_transcript_payload(payload),
        "segments": payload.get("segments") or [],
        "raw": payload,
    }
    write_transcript(paths, transcript)
    return transcript


def command_failure_detail(exc: subprocess.CalledProcessError) -> str:
    detail = ""
    for value in (exc.stderr, exc.stdout):
        if isinstance(value, str) and value.strip():
            detail = value.strip()
            break
    if len(detail) > 2000:
        detail = detail[-2000:]
    return detail or str(exc)


def transcript_from_local_asr_payload(
    payload: dict[str, Any],
    audio_path: Path,
    paths: AnalysisPaths,
    *,
    backend: str,
    model: str,
) -> dict[str, Any]:
    transcript = {
        "source": "whisper",
        "backend": backend,
        "model": model,
        "language": payload.get("language"),
        "source_path": str(audio_path.relative_to(paths.root)),
        "segment_count": len(payload.get("segments") or []),
        "text": extract_text_from_transcript_payload(payload),
        "segments": payload.get("segments") or [],
        "raw": payload,
    }
    write_transcript(paths, transcript)
    return transcript


def transcribe_with_openai_whisper(audio_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    whisper_bin, command_env = resolve_command("whisper")
    if whisper_bin is None:
        raise FileNotFoundError("whisper command is not available.")
    model = constants.LOCAL_ASR_MODELS[constants.DEFAULT_LOCAL_ASR_BACKEND]
    temp_dir = paths.root / "tmp-whisper"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    env = build_command_env(command_env)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    command = [
        whisper_bin,
        str(audio_path),
        "--model",
        model,
        "--task",
        "transcribe",
        "--output_format",
        "json",
        "--output_dir",
        str(temp_dir),
    ]
    try:
        run_command(command, env=env)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"OpenAI Whisper command failed: {command_failure_detail(exc)}"
        ) from exc
    json_path = temp_dir / f"{audio_path.stem}.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    transcript = transcript_from_local_asr_payload(
        payload,
        audio_path,
        paths,
        backend=constants.DEFAULT_LOCAL_ASR_BACKEND,
        model=model,
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    return transcript


def transcribe_with_mlx_whisper(audio_path: Path, paths: AnalysisPaths) -> dict[str, Any]:
    mlx_whisper_bin, command_env = resolve_command("mlx_whisper")
    if mlx_whisper_bin is None:
        raise FileNotFoundError(
            "mlx_whisper command is not available. Install the optional `mlx-whisper` dependency."
        )
    model = constants.LOCAL_ASR_MODELS["mlx-whisper"]
    temp_dir = paths.root / "tmp-whisper"
    temp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        mlx_whisper_bin,
        str(audio_path),
        "--model",
        model,
        "--task",
        "transcribe",
        "--output-format",
        "json",
        "--output-dir",
        str(temp_dir),
        "--verbose",
        "False",
    ]
    run_command(command, env=build_command_env(command_env))
    json_path = temp_dir / f"{audio_path.stem}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"mlx_whisper did not write the expected JSON transcript: {json_path}"
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    transcript = transcript_from_local_asr_payload(
        payload,
        audio_path,
        paths,
        backend="mlx-whisper",
        model=model,
    )
    shutil.rmtree(temp_dir, ignore_errors=True)
    return transcript


def transcribe_with_whisper(
    audio_path: Path,
    paths: AnalysisPaths,
    *,
    backend: str = constants.DEFAULT_LOCAL_ASR_BACKEND,
) -> dict[str, Any]:
    if backend == constants.DEFAULT_LOCAL_ASR_BACKEND:
        return transcribe_with_openai_whisper(audio_path, paths)
    if backend == "mlx-whisper":
        return transcribe_with_mlx_whisper(audio_path, paths)
    raise ValueError(f"Unsupported local ASR backend: {backend}")


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


def _retry_message(
    progress_callback: Callable[[str, str], None] | None,
    phase: str,
    message: str,
    attempt: int,
    attempts: int,
) -> None:
    if progress_callback is None or attempt >= attempts:
        return
    report_progress(progress_callback, phase, f"{message}; retrying ({attempt + 1}/{attempts})")


def _run_yt_dlp_extract_info(
    opts: dict[str, Any],
    url: str,
    *,
    download: bool,
    attempts: int = constants.DEFAULT_YTDLP_RETRY_ATTEMPTS,
    progress_callback: Callable[[str, str], None] | None = None,
    phase: str = "metadata",
    retry_message: str = "yt-dlp request failed",
) -> dict[str, Any]:
    YoutubeDL = load_yt_dlp()
    total_attempts = max(1, attempts)
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            with YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=download)
        except Exception as exc:
            last_error = exc
            _retry_message(progress_callback, phase, retry_message, attempt, total_attempts)
            if attempt < total_attempts:
                time.sleep(constants.DEFAULT_YTDLP_RETRY_SLEEP_SECONDS)
    assert last_error is not None
    raise last_error


def _read_url_bytes_with_retry(
    url: str,
    *,
    attempts: int = constants.DEFAULT_REMOTE_SUBTITLE_RETRY_ATTEMPTS,
    progress_callback: Callable[[str, str], None] | None = None,
    phase: str = "transcript",
    retry_message: str = "Remote fetch stalled",
) -> bytes:
    total_attempts = max(1, attempts)
    last_error: Exception | None = None
    for attempt in range(1, total_attempts + 1):
        try:
            with urlopen(str(url), timeout=constants.DEFAULT_REMOTE_SUBTITLE_TIMEOUT_SECONDS) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            _retry_message(progress_callback, phase, retry_message, attempt, total_attempts)
            if attempt < total_attempts:
                time.sleep(constants.DEFAULT_YTDLP_RETRY_SLEEP_SECONDS)
    assert last_error is not None
    raise last_error


def fetch_youtube_metadata(
    url: str,
    *,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": constants.DEFAULT_YTDLP_METADATA_SOCKET_TIMEOUT_SECONDS,
    }
    return _run_yt_dlp_extract_info(
        opts,
        url,
        download=False,
        progress_callback=progress_callback,
        phase="metadata",
        retry_message="YouTube metadata fetch stalled",
    )


def default_comments_payload(requested_count: int, *, status: str = "disabled") -> dict[str, Any]:
    return {
        "status": status,
        "requested_count": requested_count,
        "returned_count": 0,
        "source": None,
        "items": [],
        "interpretation_notes": [
            "top comments are contextual signals, not representative sampling",
            "comments are hints about visible audience context, not ground truth",
        ],
        "provenance": {
            "method": None,
            "sort": "top",
            "trust": "contextual_signal_not_semantics",
        },
    }


def normalize_comment_item(item: dict[str, Any]) -> dict[str, Any] | None:
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    return {
        "id": item.get("id"),
        "text": text,
        "like_count": item.get("like_count"),
        "reply_count": item.get("reply_count"),
        "published_at": item.get("timestamp") or item.get("time_text"),
        "is_pinned": item.get("is_pinned"),
        "provenance": {
            "source": "yt_dlp.comments",
            "trust": "contextual_signal_not_semantics",
        },
    }


def fetch_youtube_comments(
    url: str,
    *,
    requested_count: int,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    limit = max(0, int(requested_count))
    if limit <= 0:
        return default_comments_payload(limit)
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": constants.DEFAULT_YTDLP_METADATA_SOCKET_TIMEOUT_SECONDS,
        "getcomments": True,
        "extractor_args": {
            "youtube": {
                "comment_sort": ["top"],
                "max_comments": [str(limit)],
            }
        },
    }
    info = _run_yt_dlp_extract_info(
        opts,
        url,
        download=False,
        progress_callback=progress_callback,
        phase="comments",
        retry_message="YouTube comments fetch stalled",
    )
    items = [
        normalized
        for item in info.get("comments") or []
        if isinstance(item, dict)
        for normalized in [normalize_comment_item(item)]
        if normalized is not None
    ][:limit]
    return {
        "status": "extracted",
        "requested_count": limit,
        "returned_count": len(items),
        "source": "yt_dlp_python_api",
        "items": items,
        "interpretation_notes": [
            "top comments are contextual signals, not representative sampling",
            "comments are hints about visible audience context, not ground truth",
        ],
        "provenance": {
            "method": "yt_dlp_getcomments",
            "sort": "top",
            "requested_limit": limit,
            "trust": "contextual_signal_not_semantics",
        },
    }


def run_comments_stage(
    source: str,
    *,
    requested_count: int,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    count = max(0, int(requested_count))
    if count <= 0:
        return default_comments_payload(count)
    if not is_youtube_url(source):
        return default_comments_payload(count, status="not_applicable")
    try:
        return fetch_youtube_comments(
            source,
            requested_count=count,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        payload = default_comments_payload(count, status="failed")
        payload["error"] = str(exc)
        return payload


def download_youtube_subtitles(
    url: str,
    paths: AnalysisPaths,
    *,
    progress_callback: Callable[[str, str], None] | None = None,
) -> None:
    subtitle_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": constants.DEFAULT_YTDLP_MEDIA_SOCKET_TIMEOUT_SECONDS,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": preferred_subtitle_languages(),
        "subtitlesformat": "vtt",
        "outtmpl": {
            "subtitle": str(paths.subtitles_dir / "%(id)s.%(language)s.manual.%(ext)s"),
        },
    }
    _run_yt_dlp_extract_info(
        subtitle_opts,
        url,
        download=True,
        progress_callback=progress_callback,
        phase="transcript",
        retry_message="Subtitle download stalled",
    )


def youtube_format_selector(max_video_height: int | None) -> str | None:
    if max_video_height is None:
        return None
    height = int(max_video_height)
    return f"bestvideo[height<={height}]+bestaudio/best[height<={height}]/best"


def download_youtube_media(
    url: str,
    paths: AnalysisPaths,
    *,
    metadata_hint: dict[str, Any] | None = None,
    fetch_subtitles: bool = True,
    max_video_height: int | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], Path]:
    if fetch_subtitles:
        report_progress(progress_callback, "transcript", "Fetching subtitle tracks")
        try:
            download_youtube_subtitles(url, paths, progress_callback=progress_callback)
        except Exception:
            # Subtitle retrieval is best-effort. If YouTube rate limits or rejects
            # caption download, keep the pipeline alive and allow later fallback to
            # Whisper after the video is downloaded.
            if metadata_hint is not None:
                try:
                    report_progress(progress_callback, "transcript", "Trying metadata subtitle fallback")
                    download_subtitle_from_metadata(
                        metadata_hint,
                        paths,
                        source_url=url,
                        progress_callback=progress_callback,
                    )
                except Exception:
                    pass
        if choose_subtitle_file(paths.subtitles_dir) is None and metadata_hint is not None:
            try:
                report_progress(progress_callback, "transcript", "Trying metadata subtitle fallback")
                download_subtitle_from_metadata(
                    metadata_hint,
                    paths,
                    source_url=url,
                    progress_callback=progress_callback,
                )
            except Exception:
                pass
    video_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": constants.DEFAULT_YTDLP_MEDIA_SOCKET_TIMEOUT_SECONDS,
        "merge_output_format": "mp4",
        "progress_hooks": [make_download_progress_hook(progress_callback)],
        "outtmpl": {
            "default": str(paths.video_dir / "source.%(ext)s"),
        },
    }
    format_selector = youtube_format_selector(max_video_height)
    if format_selector is not None:
        video_opts["format"] = format_selector
    report_progress(progress_callback, "download", "Starting source media download")
    info = _run_yt_dlp_extract_info(
        video_opts,
        url,
        download=True,
        progress_callback=progress_callback,
        phase="download",
        retry_message="YouTube media download stalled",
    )
    if fetch_subtitles and choose_subtitle_file(paths.subtitles_dir) is None:
        try:
            report_progress(progress_callback, "transcript", "Trying metadata subtitle fallback")
            download_subtitle_from_metadata(
                info,
                paths,
                source_url=url,
                progress_callback=progress_callback,
            )
        except Exception:
            pass
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


def transcript_from_preferred_subtitles(paths: AnalysisPaths) -> dict[str, Any] | None:
    subtitle_file = choose_subtitle_file(paths.subtitles_dir)
    if subtitle_file is None:
        return None
    return transcript_from_subtitles(subtitle_file, paths)


def transcript_strategy_auto(
    audio_path: Path,
    paths: AnalysisPaths,
    *,
    local_asr_backend: str = constants.DEFAULT_LOCAL_ASR_BACKEND,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    subtitle_transcript = transcript_from_preferred_subtitles(paths)
    if subtitle_transcript is not None:
        return subtitle_transcript
    report_progress(
        progress_callback,
        "transcript",
        f"Running local Whisper transcription ({local_asr_backend})",
    )
    if local_asr_backend != constants.DEFAULT_LOCAL_ASR_BACKEND:
        try:
            return transcribe_with_whisper(audio_path, paths, backend=local_asr_backend)
        except Exception as exc:
            raise RuntimeError(
                f"Local ASR backend {local_asr_backend} failed: {exc}. "
                "Remote fallback is disabled for explicit backend selection."
            ) from exc

    local_failures: list[Exception] = []
    for attempt in range(2):
        try:
            return transcribe_with_whisper(audio_path, paths, backend=local_asr_backend)
        except Exception as exc:
            local_failures.append(exc)
            if attempt == 0:
                report_progress(
                    progress_callback,
                    "transcript",
                    "Local Whisper failed; retrying once",
                )

    failure_summary = " | ".join(str(exc) for exc in local_failures)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Local Whisper failed twice and OPENAI_API_KEY is not set; "
            f"remote fallback is unavailable. Failures: {failure_summary}"
        ) from local_failures[-1]

    report_progress(
        progress_callback,
        "transcript",
        "Local Whisper failed twice; falling back to OpenAI transcription",
    )
    try:
        return transcribe_with_openai_api(audio_path, paths)
    except Exception as exc:
        raise RuntimeError(
            f"Local Whisper failed twice ({failure_summary}); "
            f"OpenAI transcription also failed: {exc}"
        ) from exc


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


@dataclass(frozen=True)
class KeyframeSelection:
    rows: list[dict[str, Any]]
    candidate_count: int
    deduplicated_count: int
    selected_count: int
    max_frames: int | None
    method: str
    entries: list[dict[str, Any]]


def evenly_spaced_indices(total: int, limit: int) -> list[int]:
    if total <= 0 or limit <= 0:
        return []
    if limit >= total:
        return list(range(total))
    if limit == 1:
        return [0]
    return [round(index * (total - 1) / (limit - 1)) for index in range(limit)]


def select_keyframes_for_processing(
    keyframe_rows: list[dict[str, Any]],
    paths: AnalysisPaths,
    *,
    max_frames: int | None,
) -> KeyframeSelection:
    ordered_rows = sorted(keyframe_rows, key=lambda row: float(row["timestamp_seconds"]))
    if max_frames is None:
        entries = [
            {
                **row,
                "selected": True,
                "selection_reason": "uncapped",
                "duplicate_group": None,
                "representative_filename": row["filename"],
            }
            for row in ordered_rows
        ]
        selection = KeyframeSelection(
            rows=ordered_rows,
            candidate_count=len(ordered_rows),
            deduplicated_count=len(ordered_rows),
            selected_count=len(ordered_rows),
            max_frames=None,
            method="uncapped",
            entries=entries,
        )
    else:
        candidates: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        for row in ordered_rows:
            frame_path = paths.keyframes_dir / str(row["filename"])
            matrix = triage.read_grayscale_image(frame_path)
            candidate = {
                "row": row,
                "phash": triage.compute_phash(matrix, frame_path),
                "blur_score": triage.compute_blur_score(matrix),
            }
            selected_group: dict[str, Any] | None = None
            for group in groups:
                if (
                    triage.hamming_distance(candidate["phash"], group["reference_phash"])
                    <= constants.DEFAULT_PHASH_DUPLICATE_DISTANCE
                ):
                    selected_group = group
                    break
            if selected_group is None:
                selected_group = {
                    "group_id": f"pre-dup-{len(groups) + 1:04d}",
                    "reference_phash": candidate["phash"],
                    "members": [],
                }
                groups.append(selected_group)
            selected_group["members"].append(candidate)
            candidate["duplicate_group"] = selected_group["group_id"]
            candidates.append(candidate)

        representatives: list[dict[str, Any]] = []
        representative_by_group: dict[str, dict[str, Any]] = {}
        for group in groups:
            representative = max(
                group["members"],
                key=lambda item: (
                    float(item["blur_score"]),
                    -float(item["row"]["timestamp_seconds"]),
                ),
            )
            representatives.append(representative)
            representative_by_group[group["group_id"]] = representative
        representatives.sort(key=lambda item: float(item["row"]["timestamp_seconds"]))

        selected_indices = set(evenly_spaced_indices(len(representatives), max_frames))
        selected_representatives = [
            representative
            for index, representative in enumerate(representatives)
            if index in selected_indices
        ]
        selected_filenames = {
            str(representative["row"]["filename"])
            for representative in selected_representatives
        }
        entries = []
        for candidate in candidates:
            row = candidate["row"]
            group_id = str(candidate["duplicate_group"])
            representative = representative_by_group[group_id]
            filename = str(row["filename"])
            representative_filename = str(representative["row"]["filename"])
            if filename != representative_filename:
                reason = "duplicate"
            elif filename in selected_filenames:
                reason = "selected"
            else:
                reason = "over_budget"
            entries.append(
                {
                    **row,
                    "phash": candidate["phash"],
                    "blur_score": candidate["blur_score"],
                    "duplicate_group": group_id,
                    "representative_filename": representative_filename,
                    "selected": reason == "selected",
                    "selection_reason": reason,
                }
            )
        selection = KeyframeSelection(
            rows=[representative["row"] for representative in selected_representatives],
            candidate_count=len(ordered_rows),
            deduplicated_count=len(representatives),
            selected_count=len(selected_representatives),
            max_frames=max_frames,
            method="phash_sharpest_then_even_timeline",
            entries=entries,
        )

    write_json(
        paths.visual_selection_path,
        {
            "candidate_frame_count": selection.candidate_count,
            "deduplicated_frame_count": selection.deduplicated_count,
            "selected_frame_count": selection.selected_count,
            "max_frames": selection.max_frames,
            "selection_method": selection.method,
            "frames": selection.entries,
        },
    )
    return selection


def save_debug_selected_keyframes(
    paths: AnalysisPaths,
    selection: KeyframeSelection,
) -> None:
    candidates_dir = paths.visuals_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for entry in selection.entries:
        entry["debug_image_path"] = None
        if not entry.get("selected"):
            continue
        source_path = paths.keyframes_dir / str(entry["filename"])
        if not source_path.is_file():
            continue
        target_path = candidates_dir / source_path.name
        shutil.copy2(source_path, target_path)
        entry["debug_image_path"] = str(target_path.relative_to(paths.root))
    write_json(
        paths.visual_selection_path,
        {
            "candidate_frame_count": selection.candidate_count,
            "deduplicated_frame_count": selection.deduplicated_count,
            "selected_frame_count": selection.selected_count,
            "max_frames": selection.max_frames,
            "selection_method": selection.method,
            "frames": selection.entries,
        },
    )


def visual_sampling_payload(
    *,
    intake_profile: str,
    visual_density: str,
    visuals_mode: str,
    keyframe_mode: str,
    interval_seconds: int,
    scene_threshold: float,
    candidate_frame_count: int,
    deduplicated_frame_count: int,
    selected_frame_count: int,
    max_frames: int | None,
    selection_method: str,
    visuals_payload: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    retained_slide_count = len(visuals_payload.get("slides", []))
    retained_chart_count = len(visuals_payload.get("charts", []))
    status = "extracted" if visuals_mode == "on" else "skipped"
    return {
        "status": status,
        "profile": intake_profile,
        "density": visual_density,
        "keyframe_mode": keyframe_mode if visuals_mode == "on" else "off",
        "interval_seconds": interval_seconds if keyframe_mode in {"interval", "scene+interval"} else None,
        "scene_threshold": scene_threshold if keyframe_mode in {"scene", "scene+interval"} else None,
        "candidate_frame_count": candidate_frame_count,
        "deduplicated_frame_count": deduplicated_frame_count,
        "selected_frame_count": selected_frame_count,
        "max_frames": max_frames,
        "selection_method": selection_method,
        "retained_visual_count": retained_slide_count + retained_chart_count,
        "retained_slide_count": retained_slide_count,
        "retained_chart_count": retained_chart_count,
        "provenance": {
            "method": "local_keyframe_sampling_plus_heuristic_triage",
            "trust": "sampling_signal_not_semantics",
            "quality_notes": [
                "visual sampling exposes evidence candidates, not complete visual understanding",
                "retained visuals are heuristic promotions for downstream inspection",
            ],
        },
    }


def write_empty_stage_artifacts(paths: AnalysisPaths) -> None:
    paths.triage_frames_path.parent.mkdir(parents=True, exist_ok=True)
    paths.triage_segments_path.parent.mkdir(parents=True, exist_ok=True)
    paths.review_queue_path.parent.mkdir(parents=True, exist_ok=True)
    paths.review_decisions_path.parent.mkdir(parents=True, exist_ok=True)
    paths.routing_manifest_path.parent.mkdir(parents=True, exist_ok=True)
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


def cleanup_non_debug_artifacts(paths: AnalysisPaths) -> None:
    for directory in (
        paths.triage_dir,
        paths.review_dir,
        paths.routing_dir,
        paths.gpt_dir,
        paths.report_dir,
        paths.visuals_dir,
    ):
        shutil.rmtree(directory, ignore_errors=True)
    for path in (
        paths.metadata_path,
        paths.source_path,
        paths.transcript_json_path,
        paths.transcript_text_path,
        paths.error_path,
    ):
        path.unlink(missing_ok=True)


def non_negative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def analyze_source(
    source: str,
    *,
    intake_profile: str = constants.DEFAULT_INTAKE_PROFILE,
    transcript_mode: str = "auto",
    reuse_transcript: Path | None = None,
    keyframe_mode: str = "scene+interval",
    visuals_mode: str = constants.DEFAULT_VISUALS_MODE,
    visual_density: str = constants.DEFAULT_VISUAL_DENSITY,
    ocr_mode: str = constants.DEFAULT_OCR_MODE,
    burned_subtitles_mode: str = constants.DEFAULT_BURNED_SUBTITLES_MODE,
    audio_features_mode: str = constants.DEFAULT_AUDIO_FEATURES_MODE,
    comments_count: int = constants.DEFAULT_COMMENTS_COUNT,
    local_asr_backend: str = constants.DEFAULT_LOCAL_ASR_BACKEND,
    max_video_height: int | None = None,
    max_frames: int | None = None,
    out_dir: Path | None = None,
    output_root_base: Path | None = None,
    interval_seconds: int = constants.DEFAULT_INTERVAL_SECONDS,
    scene_threshold: float = constants.DEFAULT_SCENE_THRESHOLD,
    triage_mode: str = constants.DEFAULT_TRIAGE_MODE,
    gpt_mode: str = constants.DEFAULT_GPT_MODE,
    review_mode: str = constants.DEFAULT_REVIEW_MODE,
    gpt_model: str = constants.DEFAULT_GPT_MODEL,
    report_language: str = constants.DEFAULT_REPORT_LANGUAGE,
    artifacts_mode: str = constants.DEFAULT_ARTIFACTS_MODE,
    review_reset: bool = False,
    cleanup_intermediates: bool = True,
    review_input: Callable[[str], str] = input,
    progress_callback: Callable[[str, str], None] | None = None,
) -> Path:
    load_local_env()
    source_path = Path(source).expanduser()
    metadata_hint: dict[str, Any] | None = None
    if looks_like_url(source):
        report_progress(progress_callback, "metadata", "Fetching YouTube metadata")
        metadata_hint = fetch_youtube_metadata(source, progress_callback=progress_callback)
    output_root = out_dir or default_output_dir_for_source(
        source,
        (metadata_hint or {}).get("id"),
        (metadata_hint or {}).get("title"),
        output_root=output_root_base,
    )
    paths = analysis_paths(output_root)
    clear_stale_output_files(paths)
    ensure_dirs(paths)
    ensure_source_file(paths, source)
    metadata: dict[str, Any] = {}
    transcript: dict[str, Any] | None = None
    ocr_state = default_ocr_state(ocr_mode)
    burned_subtitles_state = default_burned_subtitles_state(burned_subtitles_mode)
    audio_features_payload = default_audio_features_payload(audio_features_mode)
    comments_payload = default_comments_payload(comments_count)
    frames: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    visuals_payload = visuals.empty_visuals_payload()
    visual_sampling = visual_sampling_payload(
        intake_profile=intake_profile,
        visual_density=visual_density,
        visuals_mode=visuals_mode,
        keyframe_mode="off",
        interval_seconds=interval_seconds,
        scene_threshold=scene_threshold,
        candidate_frame_count=0,
        deduplicated_frame_count=0,
        selected_frame_count=0,
        max_frames=max_frames,
        selection_method="not_run",
        visuals_payload=visuals_payload,
    )
    gpt_payload: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []
    run_status = "failed"
    pipeline_succeeded = False
    reuse_transcript_path = Path(reuse_transcript).expanduser() if reuse_transcript is not None else None
    should_fetch_subtitles = transcript_mode != "off" and reuse_transcript_path is None

    try:
        if looks_like_url(source):
            metadata = metadata_hint or fetch_youtube_metadata(source, progress_callback=progress_callback)
            write_json(paths.metadata_path, metadata)
            metadata, video_path = download_youtube_media(
                source,
                paths,
                metadata_hint=metadata,
                fetch_subtitles=should_fetch_subtitles,
                max_video_height=max_video_height,
                progress_callback=progress_callback,
            )
            audio_input = video_path
        else:
            if not source_path.exists():
                raise FileNotFoundError(f"Source file does not exist: {source_path}")
            report_progress(progress_callback, "source", "Inspecting local media")
            metadata, video_path = materialize_local_input(source_path.resolve(), paths)
            audio_input = video_path or source_path.resolve()

        if reuse_transcript_path is not None:
            report_progress(progress_callback, "transcript", "Reusing existing transcript artifact")
            transcript = load_reused_transcript(reuse_transcript_path, paths)
        elif transcript_mode == "off":
            report_progress(progress_callback, "transcript", "Skipping transcript extraction")
            transcript = skipped_transcript()
            write_transcript(paths, transcript)
        else:
            subtitle_transcript = None
            if transcript_mode in {"auto", "subtitles", "whisper", "api"}:
                report_progress(progress_callback, "transcript", "Checking text-track subtitles")
                subtitle_transcript = transcript_from_preferred_subtitles(paths)

            burned_subtitle_transcript = None
            if subtitle_transcript is None and transcript_mode in {"auto", "whisper", "api"}:
                if burned_subtitles_mode == "off":
                    report_progress(progress_callback, "transcript", "Skipping burned subtitle OCR (disabled)")
                else:
                    report_progress(
                        progress_callback,
                        "transcript",
                        f"Trying burned subtitle OCR ({burned_subtitles_mode})",
                    )
                burned_subtitle_transcript, burned_subtitles_state = run_burned_subtitles_stage(
                    video_path,
                    metadata,
                    paths,
                    mode=burned_subtitles_mode,
                )

            normalized_audio: Path | None = None
            if (
                subtitle_transcript is None
                and burned_subtitle_transcript is None
                and transcript_mode in {"auto", "whisper", "api"}
            ):
                report_progress(progress_callback, "audio", "Extracting normalized audio for transcription")
                normalized_audio = extract_audio(audio_input, paths.audio_dir)

            if transcript_mode == "auto":
                if subtitle_transcript is not None:
                    report_progress(progress_callback, "transcript", "Using text-track subtitles")
                    transcript = subtitle_transcript
                elif burned_subtitle_transcript is not None:
                    report_progress(progress_callback, "transcript", "Using burned subtitle OCR transcript")
                    transcript = burned_subtitle_transcript
                else:
                    if progress_callback is None:
                        transcript = transcript_strategy_auto(
                            normalized_audio,
                            paths,
                            local_asr_backend=local_asr_backend,
                        )
                    else:
                        transcript = transcript_strategy_auto(
                            normalized_audio,
                            paths,
                            local_asr_backend=local_asr_backend,
                            progress_callback=progress_callback,
                        )
            elif transcript_mode == "subtitles":
                if subtitle_transcript is None:
                    raise FileNotFoundError("No subtitle file is available for transcript_mode=subtitles.")
                report_progress(progress_callback, "transcript", "Using text-track subtitles")
                transcript = subtitle_transcript
            elif transcript_mode == "api":
                if subtitle_transcript is not None:
                    report_progress(progress_callback, "transcript", "Using text-track subtitles")
                    transcript = subtitle_transcript
                elif burned_subtitle_transcript is not None:
                    report_progress(progress_callback, "transcript", "Using burned subtitle OCR transcript")
                    transcript = burned_subtitle_transcript
                else:
                    report_progress(progress_callback, "transcript", "Running OpenAI transcription")
                    transcript = transcribe_with_openai_api(normalized_audio, paths)
            elif transcript_mode == "whisper":
                if subtitle_transcript is not None:
                    report_progress(progress_callback, "transcript", "Using text-track subtitles")
                    transcript = subtitle_transcript
                elif burned_subtitle_transcript is not None:
                    report_progress(progress_callback, "transcript", "Using burned subtitle OCR transcript")
                    transcript = burned_subtitle_transcript
                else:
                    report_progress(
                        progress_callback,
                        "transcript",
                        f"Running local Whisper transcription ({local_asr_backend})",
                    )
                    transcript = transcribe_with_whisper(
                        normalized_audio,
                        paths,
                        backend=local_asr_backend,
                    )
            else:
                raise ValueError(f"Unsupported transcript mode: {transcript_mode}")

        effective_keyframe_mode = keyframe_mode if visuals_mode == "on" else "off"
        if effective_keyframe_mode == "off":
            report_progress(progress_callback, "visuals", "Skipping visual pipeline")
        else:
            report_progress(progress_callback, "visuals", "Extracting keyframes")

        keyframe_rows = create_keyframes(
            video_path,
            metadata,
            paths,
            mode=effective_keyframe_mode,
            interval_seconds=interval_seconds,
            threshold=scene_threshold,
        )
        keyframe_selection = select_keyframes_for_processing(
            keyframe_rows,
            paths,
            max_frames=max_frames,
        )
        if artifacts_mode == "debug" and max_frames is not None:
            save_debug_selected_keyframes(paths, keyframe_selection)
        selected_keyframe_rows = keyframe_selection.rows

        try:
            if selected_keyframe_rows and ocr_mode != "off":
                report_progress(progress_callback, "visuals", "Running frame OCR")
            ocr_rows, ocr_state = run_ocr_stage(paths, selected_keyframe_rows, ocr_mode=ocr_mode)
        except Exception as exc:
            ocr_state = default_ocr_state(ocr_mode)
            ocr_state["attempted"] = bool(selected_keyframe_rows) and ocr_mode != "off"
            ocr_state["status"] = "failed"
            ocr_state["error"] = str(exc)
            raise

        if triage_mode == "on" and selected_keyframe_rows:
            report_progress(progress_callback, "visuals", "Running local triage")
            frames, segments = triage.run_local_triage(
                paths.root,
                selected_keyframe_rows,
                ocr_rows,
                transcript,
            )
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
            segments = []
            write_empty_stage_artifacts(paths)

        report_progress(progress_callback, "visuals", "Assembling retained visuals")
        visuals_payload = visuals.build_embedded_visuals(
            paths.root,
            frames=frames,
            segments=segments,
            manifest_entries=manifest_entries,
        )
        if artifacts_mode == "debug":
            visuals.save_durable_visuals(
                paths.root,
                frames=frames,
                segments=segments,
                manifest_entries=manifest_entries,
            )

        if gpt_mode == "on":
            approved_entries = [entry for entry in manifest_entries if entry.get("approved_for_gpt")]
            if approved_entries or (transcript or {}).get("text", "").strip():
                report_progress(progress_callback, "gpt", "Running GPT analysis")
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

        if audio_features_mode == "on":
            report_progress(progress_callback, "audio", "Extracting loudness and silence features")
        audio_features_payload = run_audio_features_stage(
            audio_input,
            metadata,
            mode=audio_features_mode,
        )

        if comments_count > 0:
            report_progress(progress_callback, "comments", f"Fetching top {comments_count} comments")
        comments_payload = run_comments_stage(
            source,
            requested_count=comments_count,
            progress_callback=progress_callback,
        )

        visual_sampling = visual_sampling_payload(
            intake_profile=intake_profile,
            visual_density=visual_density,
            visuals_mode=visuals_mode,
            keyframe_mode=effective_keyframe_mode,
            interval_seconds=interval_seconds,
            scene_threshold=scene_threshold,
            candidate_frame_count=keyframe_selection.candidate_count,
            deduplicated_frame_count=keyframe_selection.deduplicated_count,
            selected_frame_count=keyframe_selection.selected_count,
            max_frames=keyframe_selection.max_frames,
            selection_method=keyframe_selection.method,
            visuals_payload=visuals_payload,
        )

        pipeline_succeeded = True
        return paths.root
    except KeyboardInterrupt:
        run_status = "aborted"
        error = {
            "stage": "analyze",
            "kind": "interrupted",
            "message": "Analysis interrupted by user.",
        }
        errors.append(error)
        save_error(paths, error["stage"], error["message"])
        raise
    except Exception as exc:
        run_status = "failed"
        error = {"stage": "analyze", "message": str(exc)}
        errors.append(error)
        save_error(paths, error["stage"], error["message"])
        raise
    finally:
        cleanup_error: Exception | None = None
        try:
            if cleanup_intermediates:
                report_progress(progress_callback, "cleanup", "Removing intermediate media artifacts")
                cleanup_intermediate_artifacts(paths)
            if artifacts_mode == "minimal":
                report_progress(progress_callback, "cleanup", "Removing non-debug artifacts")
                cleanup_non_debug_artifacts(paths)
        except Exception as exc:
            cleanup_error = exc
            error = {"stage": "cleanup", "message": str(exc)}
            errors.append(error)
            save_error(paths, error["stage"], error["message"])
            if pipeline_succeeded:
                run_status = "failed"
        else:
            if pipeline_succeeded:
                run_status = "completed"

        report_progress(progress_callback, "output", "Writing output bundle")
        reporting.write_output_file(
            paths,
            source_input=source,
            is_url=looks_like_url(source),
            is_youtube_url=is_youtube_url(source),
            metadata=metadata,
            transcript=transcript,
            ocr=ocr_state,
            audio_features=audio_features_payload,
            comments=comments_payload,
            visual_sampling=visual_sampling,
            visuals_payload=visuals_payload,
            errors=errors,
            run_status=run_status,
            max_video_height=max_video_height,
            cleanup_intermediates=cleanup_intermediates,
            transcript_mode=transcript_mode,
            local_asr_backend=local_asr_backend,
            visuals_mode=visuals_mode,
            visual_density=visual_density,
            ocr_mode=ocr_mode,
            burned_subtitles=burned_subtitles_state,
            audio_features_mode=audio_features_mode,
            comments_count=comments_count,
            intake_profile=intake_profile,
            gpt_mode=gpt_mode,
            artifacts_mode=artifacts_mode,
            gpt_payload=gpt_payload,
        )
        if cleanup_error is not None and pipeline_succeeded:
            raise cleanup_error


def add_analysis_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_source: bool = True,
    include_out_dir: bool = True,
) -> argparse.ArgumentParser:
    if include_source:
        parser.add_argument("--source", required=True, help="YouTube URL or local media path")
    parser.add_argument(
        "--intake-profile",
        default=constants.DEFAULT_INTAKE_PROFILE,
        choices=[constants.DEFAULT_INTAKE_PROFILE, constants.RICH_INTAKE_PROFILE],
        help="Named intake profile (`rich` enables dense visual sampling, audio features, and top comments)",
    )
    parser.add_argument(
        "--transcript",
        default="auto",
        choices=["auto", "subtitles", "api", "whisper", "off"],
        help="Transcript strategy",
    )
    parser.add_argument(
        "--local-asr-backend",
        default=constants.DEFAULT_LOCAL_ASR_BACKEND,
        choices=list(constants.LOCAL_ASR_BACKENDS),
        help="Local Whisper backend used by auto/whisper transcript modes",
    )
    parser.add_argument(
        "--keyframes",
        default="scene+interval",
        choices=["off", "scene", "interval", "scene+interval"],
        help="Keyframe extraction strategy",
    )
    parser.add_argument(
        "--visuals",
        default=None,
        choices=["off", "on"],
        help="Enable or disable the visual pipeline (keyframes, OCR, triage, and visual extraction)",
    )
    parser.add_argument(
        "--visual-density",
        default=None,
        choices=list(constants.VISUAL_DENSITY_INTERVAL_SECONDS),
        help="Named visual sampling density when interval seconds are not set",
    )
    parser.add_argument(
        "--ocr",
        default=constants.DEFAULT_OCR_MODE,
        choices=["auto", "off", "on"],
        help="Run local OCR on keyframes (`auto` is best-effort and non-fatal)",
    )
    parser.add_argument(
        "--burned-subtitles",
        default=constants.DEFAULT_BURNED_SUBTITLES_MODE,
        choices=["auto", "off", "on"],
        help="Fallback OCR for fixed burned-in CJK subtitle bands when no text-track subtitles exist",
    )
    parser.add_argument(
        "--audio-features",
        default=None,
        choices=["off", "on"],
        help="Extract optional loudness and silence structure signals",
    )
    parser.add_argument(
        "--comments",
        type=non_negative_int,
        default=None,
        help="Fetch up to N top YouTube comments as contextual hints (default off)",
    )
    parser.add_argument(
        "--max-video-height",
        type=positive_int,
        help="Prefer downloaded YouTube video at or below this height (default: yt-dlp selection)",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        help="Cap keyframes entering OCR and triage after cheap duplicate removal",
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
        "--artifacts",
        default=constants.DEFAULT_ARTIFACTS_MODE,
        choices=["minimal", "debug"],
        help="Artifact output mode (`minimal` keeps only output.json, `debug` preserves stage artifacts)",
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
    if include_out_dir:
        parser.add_argument(
            "--out-dir",
            type=Path,
            help="Override output directory (default: output/youtube/<video-id-or-stem>)",
        )
    parser.add_argument(
        "--reuse-transcript",
        type=Path,
        help="Use an existing transcript JSON artifact instead of extracting a fresh transcript",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=None,
        help=f"Interval seconds for supplemental keyframes (default: {constants.DEFAULT_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=constants.DEFAULT_SCENE_THRESHOLD,
        help=f"Scene detection threshold (default: {constants.DEFAULT_SCENE_THRESHOLD})",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a YouTube URL or local media file into transcripts, keyframes, triage, and reports."
    )
    add_version_argument(parser)
    add_analysis_arguments(parser)
    return parser.parse_args(argv)


def analysis_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    intake_profile = args.intake_profile
    is_rich = intake_profile == constants.RICH_INTAKE_PROFILE
    visual_density = args.visual_density or (
        constants.RICH_VISUAL_DENSITY if is_rich else constants.DEFAULT_VISUAL_DENSITY
    )
    interval_seconds = args.interval_seconds
    if interval_seconds is None:
        interval_seconds = constants.VISUAL_DENSITY_INTERVAL_SECONDS[visual_density]
    visuals_mode = args.visuals or ("on" if is_rich else constants.DEFAULT_VISUALS_MODE)
    audio_features_mode = args.audio_features or ("on" if is_rich else constants.DEFAULT_AUDIO_FEATURES_MODE)
    comments_count = args.comments
    if comments_count is None:
        comments_count = constants.RICH_COMMENTS_COUNT if is_rich else constants.DEFAULT_COMMENTS_COUNT
    kwargs: dict[str, Any] = {
        "intake_profile": intake_profile,
        "transcript_mode": args.transcript,
        "local_asr_backend": args.local_asr_backend,
        "reuse_transcript": args.reuse_transcript,
        "keyframe_mode": args.keyframes,
        "visuals_mode": visuals_mode,
        "visual_density": visual_density,
        "ocr_mode": args.ocr,
        "burned_subtitles_mode": args.burned_subtitles,
        "audio_features_mode": audio_features_mode,
        "comments_count": comments_count,
        "max_video_height": args.max_video_height,
        "max_frames": args.max_frames,
        "interval_seconds": interval_seconds,
        "scene_threshold": args.scene_threshold,
        "triage_mode": args.triage,
        "gpt_mode": args.gpt,
        "review_mode": args.review,
        "gpt_model": args.gpt_model,
        "report_language": args.report_language,
        "artifacts_mode": args.artifacts,
        "review_reset": args.review_reset,
        "cleanup_intermediates": not args.keep_intermediates,
    }
    if hasattr(args, "out_dir"):
        kwargs["out_dir"] = args.out_dir
    return kwargs


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = parse_args(argv)
    output_root = analyze_source(
        args.source,
        **analysis_kwargs_from_args(args),
        progress_callback=StderrProgressReporter(),
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
