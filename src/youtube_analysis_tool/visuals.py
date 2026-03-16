from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .artifacts import write_json


TRANSCRIPT_EXCERPT_LIMIT = 1000
VISUAL_BUCKETS = {
    "slides": "slides",
    "chart_table": "charts",
}


def empty_visuals_payload() -> dict[str, list[dict[str, Any]]]:
    return {"slides": [], "charts": []}


def transcript_excerpt(text: str, *, limit: int = TRANSCRIPT_EXCERPT_LIMIT) -> str:
    return str(text or "")[:limit].strip()


def visual_bucket_for_label(label: str | None) -> str | None:
    return VISUAL_BUCKETS.get(str(label or ""))


def choose_primary_image_path(
    segment: dict[str, Any],
    frame_records: list[dict[str, Any]],
    saved_paths_by_frame_id: dict[str, str],
) -> str | None:
    if not saved_paths_by_frame_id:
        return None
    representative_paths = set(segment.get("representative_frame_paths", []))
    representative_records = [
        record
        for record in frame_records
        if record.get("frame_path") in representative_paths and record.get("frame_id") in saved_paths_by_frame_id
    ]
    if representative_records:
        primary_record = max(
            representative_records,
            key=lambda record: (
                float(record.get("blur_score", 0.0)),
                -float(record.get("timestamp_seconds", 0.0)),
            ),
        )
        return saved_paths_by_frame_id.get(primary_record["frame_id"])
    first_frame_id = frame_records[0]["frame_id"]
    return saved_paths_by_frame_id.get(first_frame_id)


def build_visual_entry(
    output_root: Path,
    *,
    bucket: str,
    segment: dict[str, Any],
    frame_records: list[dict[str, Any]],
    image_paths: list[str],
    primary_image_path: str | None,
) -> dict[str, Any]:
    del output_root
    return {
        "segment_id": segment["segment_id"],
        "effective_label": segment.get("effective_label", segment.get("heuristic_label")),
        "start_seconds": segment.get("start_seconds"),
        "end_seconds": segment.get("end_seconds"),
        "start_hms": segment.get("start_hms"),
        "end_hms": segment.get("end_hms"),
        "ocr_summary": segment.get("ocr_summary", ""),
        "ocr_char_count": segment.get("ocr_char_count", 0),
        "image_paths": image_paths,
        "primary_image_path": primary_image_path,
        "frame_count": len(image_paths),
        "transcript_excerpt": transcript_excerpt(segment.get("transcript_window", {}).get("text", "")),
        "source_segment_ref": {
            "segment_id": segment["segment_id"],
            "artifact_path": "triage/segments.json",
            "visual_bucket": bucket,
        },
    }


def save_durable_visuals(
    output_root: Path,
    *,
    frames: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    visuals_dir = output_root / "visuals"
    slides_dir = visuals_dir / "slides"
    charts_dir = visuals_dir / "charts"
    slides_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    payload = empty_visuals_payload()
    manifest_by_id = {entry["segment_id"]: entry for entry in manifest_entries}
    frames_by_id = {frame["frame_id"]: frame for frame in frames}

    for segment in segments:
        manifest_entry = manifest_by_id.get(segment["segment_id"], {})
        effective_label = manifest_entry.get("effective_label", segment.get("heuristic_label"))
        bucket = visual_bucket_for_label(effective_label)
        if bucket is None:
            continue

        frame_records = [
            frames_by_id[frame_id]
            for frame_id in segment.get("frame_ids", [])
            if frame_id in frames_by_id
        ]
        if not frame_records:
            continue

        target_dir = slides_dir if bucket == "slides" else charts_dir
        segment_dir = target_dir / segment["segment_id"]
        segment_dir.mkdir(parents=True, exist_ok=True)

        image_paths: list[str] = []
        saved_paths_by_frame_id: dict[str, str] = {}
        for frame_record in frame_records:
            source_path = output_root / frame_record["frame_path"]
            if not source_path.exists():
                continue
            destination_path = segment_dir / Path(frame_record["frame_path"]).name
            shutil.copy2(source_path, destination_path)
            relative_path = str(destination_path.relative_to(output_root))
            image_paths.append(relative_path)
            saved_paths_by_frame_id[frame_record["frame_id"]] = relative_path

        if not image_paths:
            continue

        primary_image_path = choose_primary_image_path(segment, frame_records, saved_paths_by_frame_id)
        payload[bucket].append(
            build_visual_entry(
                output_root,
                bucket=bucket,
                segment={
                    **segment,
                    "effective_label": effective_label,
                },
                frame_records=frame_records,
                image_paths=image_paths,
                primary_image_path=primary_image_path,
            )
        )

    write_json(visuals_dir / "manifest.json", payload)
    return payload
