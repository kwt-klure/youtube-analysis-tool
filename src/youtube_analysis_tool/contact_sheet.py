from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from . import pipeline
from .artifacts import write_json
from .version import add_version_argument


@dataclass(frozen=True)
class ContactSheetGrid:
    columns: int
    rows: int


@dataclass(frozen=True)
class ContactSheetResult:
    root: Path
    sheet_path: Path
    manifest_path: Path
    video_path: Path | None


def auto_contact_sheet_grid(*, frame_count: int, columns: int | None = None) -> ContactSheetGrid:
    safe_frame_count = max(1, int(frame_count))
    if columns is not None:
        safe_columns = max(1, int(columns))
    elif safe_frame_count <= 6:
        safe_columns = safe_frame_count
    elif safe_frame_count <= 60:
        safe_columns = 6
    else:
        safe_columns = min(8, max(6, math.ceil(math.sqrt(safe_frame_count))))
    return ContactSheetGrid(
        columns=safe_columns,
        rows=math.ceil(safe_frame_count / safe_columns),
    )


def estimated_frame_count(*, duration_seconds: float, fps: float) -> int:
    safe_duration = max(0.0, float(duration_seconds))
    safe_fps = max(0.001, float(fps))
    return max(1, math.ceil(safe_duration * safe_fps))


def build_contact_sheet_command(
    *,
    video_path: Path,
    output_path: Path,
    fps: float,
    thumb_width: int,
    grid: ContactSheetGrid,
    padding: int,
    margin: int,
) -> list[str]:
    vf = ",".join(
        [
            f"fps={fps:g}",
            f"scale={int(thumb_width)}:-1",
            f"tile={grid.columns}x{grid.rows}:padding={int(padding)}:margin={int(margin)}",
        ]
    )
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        str(output_path),
    ]


def _default_output_root_for_contact_sheet(
    source: str,
    metadata_hint: dict[str, Any] | None,
    *,
    output_root_base: Path | None,
) -> Path:
    return pipeline.default_output_dir_for_source(
        source,
        (metadata_hint or {}).get("id"),
        (metadata_hint or {}).get("title"),
        output_root=output_root_base,
    )


def _source_media(
    source: str,
    paths: pipeline.AnalysisPaths,
    *,
    metadata_hint: dict[str, Any] | None,
    max_video_height: int | None,
    progress_callback=None,
) -> tuple[dict[str, Any], Path | None]:
    if pipeline.looks_like_url(source):
        return pipeline.download_youtube_media(
            source,
            paths,
            metadata_hint=metadata_hint,
            fetch_subtitles=False,
            max_video_height=max_video_height,
            progress_callback=progress_callback,
        )
    source_path = Path(source).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")
    return pipeline.materialize_local_input(source_path.resolve(), paths)


def create_contact_sheet(
    source: str,
    *,
    out_dir: Path | None = None,
    output_root_base: Path | None = None,
    fps: float = 1.0,
    columns: int | None = None,
    thumb_width: int = 240,
    padding: int = 6,
    margin: int = 8,
    max_video_height: int | None = None,
    progress_callback=None,
) -> ContactSheetResult:
    pipeline.load_local_env()
    metadata_hint: dict[str, Any] | None = None
    if pipeline.looks_like_url(source):
        pipeline.report_progress(progress_callback, "metadata", "Fetching YouTube metadata")
        metadata_hint = pipeline.fetch_youtube_metadata(source, progress_callback=progress_callback)

    output_root = out_dir or _default_output_root_for_contact_sheet(
        source,
        metadata_hint,
        output_root_base=output_root_base,
    )
    paths = pipeline.analysis_paths(output_root)
    pipeline.ensure_dirs(paths)
    pipeline.ensure_source_file(paths, source)

    pipeline.report_progress(progress_callback, "download", "Materializing source media")
    metadata, video_path = _source_media(
        source,
        paths,
        metadata_hint=metadata_hint,
        max_video_height=max_video_height,
        progress_callback=progress_callback,
    )
    if video_path is None:
        raise ValueError("Contact sheet requires a video stream.")

    frame_count = estimated_frame_count(
        duration_seconds=pipeline.duration_seconds(metadata),
        fps=fps,
    )
    grid = auto_contact_sheet_grid(frame_count=frame_count, columns=columns)
    sheet_path = paths.root / "contact-sheet.jpg"
    manifest_path = paths.root / "contact-sheet.json"
    command = build_contact_sheet_command(
        video_path=video_path,
        output_path=sheet_path,
        fps=fps,
        thumb_width=thumb_width,
        grid=grid,
        padding=padding,
        margin=margin,
    )

    pipeline.report_progress(progress_callback, "contact-sheet", "Rendering frame contact sheet")
    pipeline.run_command(command)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source": {
            "input": source,
            "is_url": pipeline.looks_like_url(source),
            "is_youtube_url": pipeline.is_youtube_url(source),
        },
        "metadata": {
            "id": metadata.get("id"),
            "title": metadata.get("title"),
            "channel": metadata.get("channel") or metadata.get("uploader"),
            "duration_seconds": pipeline.duration_seconds(metadata),
        },
        "media": {
            "video_path": str(video_path),
            "is_symlink": video_path.is_symlink(),
            "kept": True,
            "requested_max_video_height": max_video_height,
            "height_limit_applied": pipeline.looks_like_url(source) and max_video_height is not None,
        },
        "contact_sheet": {
            "path": str(sheet_path),
            "fps": float(fps),
            "estimated_frame_count": frame_count,
            "columns": grid.columns,
            "rows": grid.rows,
            "thumb_width": int(thumb_width),
            "padding": int(padding),
            "margin": int(margin),
            "command": command,
        },
    }
    write_json(manifest_path, manifest)
    return ContactSheetResult(
        root=paths.root,
        sheet_path=sheet_path,
        manifest_path=manifest_path,
        video_path=video_path,
    )


def positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local contact sheet for a YouTube URL or local video file."
    )
    add_version_argument(parser)
    parser.add_argument("--source", required=True, help="YouTube URL or local video path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Override output directory (default: output/youtube/<video-id-or-stem>)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Base output root when --out-dir is not provided",
    )
    parser.add_argument("--fps", type=positive_float, default=1.0, help="Frames per second sampled into the sheet")
    parser.add_argument("--columns", type=positive_int, help="Override contact-sheet column count")
    parser.add_argument("--thumb-width", type=positive_int, default=240, help="Width of each tile in pixels")
    parser.add_argument(
        "--max-video-height",
        type=positive_int,
        help="Prefer downloaded YouTube video at or below this height (default: yt-dlp selection)",
    )
    parser.add_argument("--padding", type=int, default=6, help="Padding between tiles")
    parser.add_argument("--margin", type=int, default=8, help="Outer margin around the sheet")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = create_contact_sheet(
        args.source,
        out_dir=args.out_dir,
        output_root_base=args.root,
        fps=args.fps,
        columns=args.columns,
        thumb_width=args.thumb_width,
        max_video_height=args.max_video_height,
        padding=args.padding,
        margin=args.margin,
        progress_callback=pipeline.StderrProgressReporter(),
    )
    print(result.sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
