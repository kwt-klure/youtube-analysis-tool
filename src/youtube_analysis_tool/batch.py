from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from . import constants
from .artifacts import write_json
from .pipeline import (
    StderrProgressReporter,
    add_analysis_arguments,
    analysis_kwargs_from_args,
    analyze_source,
    default_output_dir_for_source,
    load_local_env,
    slugify,
    youtube_video_id_from_source,
)


def read_source_list(path: Path) -> list[str]:
    entries: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def candidate_existing_outputs(source: str, *, root: Path) -> list[Path]:
    candidates: list[Path] = []
    video_id = youtube_video_id_from_source(source)
    if video_id:
        video_id_slug = slugify(video_id)
        direct = root / video_id_slug / "output.json"
        if direct.exists():
            candidates.append(direct)
        candidates.extend(sorted(root.glob(f"*-{video_id_slug}/output.json")))
        return candidates

    direct = default_output_dir_for_source(source, output_root=root) / "output.json"
    if direct.exists():
        candidates.append(direct)
    return candidates


def find_existing_output(source: str, *, root: Path) -> Path | None:
    candidates = candidate_existing_outputs(source, root=root)
    return candidates[0] if candidates else None


def build_batch_report(
    *,
    source_list: Path,
    root: Path,
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    totals = {
        "queued": len(items),
        "completed": sum(1 for item in items if item["status"] == "completed"),
        "skipped": sum(1 for item in items if item["status"] == "skipped"),
        "failed": sum(1 for item in items if item["status"] == "failed"),
    }
    return {
        "batch_version": constants.BATCH_REPORT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_list": str(source_list),
        "root": str(root),
        "totals": totals,
        "items": items,
    }


def batch_report_path(source_list: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    manifest_stem = slugify(source_list.stem)
    return constants.BATCH_REPORT_ROOT / f"{timestamp}-{manifest_stem}.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run youtube-analysis-tool over a newline-separated source list."
    )
    parser.add_argument(
        "--source-list",
        required=True,
        type=Path,
        help="Path to a newline-separated source list (YouTube URLs or local media paths).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=constants.DEFAULT_OUTPUT_ROOT,
        help="Output root for per-item analysis directories.",
    )
    add_analysis_arguments(parser, include_source=False, include_out_dir=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = parse_args(argv)
    sources = read_source_list(args.source_list)
    reporter = StderrProgressReporter()
    analysis_kwargs = analysis_kwargs_from_args(args)
    root = args.root
    items: list[dict[str, Any]] = []

    for index, source in enumerate(sources, start=1):
        reporter("batch", f"[{index}/{len(sources)}] Inspecting {source}")
        existing_output = find_existing_output(source, root=root)
        if existing_output is not None:
            reporter("batch", f"[{index}/{len(sources)}] Skipping existing {source}")
            items.append(
                {
                    "source": source,
                    "status": "skipped",
                    "output_path": str(existing_output.parent),
                    "error": None,
                }
            )
            continue

        reporter("batch", f"[{index}/{len(sources)}] Starting {source}")
        try:
            output_root = analyze_source(
                source,
                output_root_base=root,
                progress_callback=reporter,
                **analysis_kwargs,
            )
        except Exception as exc:
            reporter("batch", f"[{index}/{len(sources)}] Failed {source}: {exc}")
            items.append(
                {
                    "source": source,
                    "status": "failed",
                    "output_path": None,
                    "error": str(exc),
                }
            )
            continue

        reporter("batch", f"[{index}/{len(sources)}] Completed {source}")
        items.append(
            {
                "source": source,
                "status": "completed",
                "output_path": str(output_root),
                "error": None,
            }
        )

    report = build_batch_report(source_list=args.source_list, root=root, items=items)
    report_path = batch_report_path(args.source_list)
    write_json(report_path, report)
    print(report_path)
    return 1 if report["totals"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
