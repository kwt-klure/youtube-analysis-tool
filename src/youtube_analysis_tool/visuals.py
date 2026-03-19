from __future__ import annotations

import base64
import mimetypes
import shutil
from pathlib import Path
from typing import Any

from .artifacts import write_json


TRANSCRIPT_EXCERPT_LIMIT = 1000
VISUAL_BUCKETS = {
    "slides": "slides",
    "chart_table": "charts",
}
CANONICAL_EFFECTIVE_LABELS = {
    "slides": "slide",
    "charts": "chart",
}


def empty_visuals_payload() -> dict[str, list[dict[str, Any]]]:
    return {"slides": [], "charts": []}


def transcript_excerpt(text: str, *, limit: int = TRANSCRIPT_EXCERPT_LIMIT) -> str:
    return str(text or "")[:limit].strip()


def visual_bucket_for_label(label: str | None) -> str | None:
    return VISUAL_BUCKETS.get(str(label or ""))


def canonical_effective_label(bucket: str) -> str:
    return CANONICAL_EFFECTIVE_LABELS[bucket]


def combine_ocr_text(frame_records: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for record in frame_records:
        text = str(record.get("ocr_text", "")).strip()
        if text and text not in chunks:
            chunks.append(text)
    return "\n".join(chunks).strip()


def choose_primary_frame_id(segment: dict[str, Any], frame_records: list[dict[str, Any]]) -> str | None:
    if not frame_records:
        return None
    representative_paths = set(segment.get("representative_frame_paths", []))
    representative_records = [
        record
        for record in frame_records
        if record.get("frame_path") in representative_paths
    ]
    if representative_records:
        primary_record = max(
            representative_records,
            key=lambda record: (
                float(record.get("blur_score", 0.0)),
                -float(record.get("timestamp_seconds", 0.0)),
            ),
        )
        return str(primary_record.get("frame_id"))
    return str(frame_records[0].get("frame_id"))


def collect_segment_frame_records(
    frames_by_id: dict[str, dict[str, Any]],
    segment: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        frames_by_id[frame_id]
        for frame_id in segment.get("frame_ids", [])
        if frame_id in frames_by_id
    ]


def encode_image_payload(image_path: Path) -> dict[str, Any]:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type:
        mime_type = "image/jpeg"
    return {
        "filename": image_path.name,
        "mime_type": mime_type,
        "encoding": "base64",
        "data": base64.b64encode(image_path.read_bytes()).decode("ascii"),
    }


def build_embedded_visual_entry(
    *,
    segment: dict[str, Any],
    bucket: str,
    frame_records: list[dict[str, Any]],
    image: dict[str, Any],
) -> dict[str, Any]:
    return {
        "segment_id": segment["segment_id"],
        "effective_label": canonical_effective_label(bucket),
        "heuristic_confidence": segment.get("heuristic_confidence"),
        "start_seconds": segment.get("start_seconds"),
        "end_seconds": segment.get("end_seconds"),
        "start_hms": segment.get("start_hms"),
        "end_hms": segment.get("end_hms"),
        "ocr_text": combine_ocr_text(frame_records),
        "ocr_summary": segment.get("ocr_summary", ""),
        "ocr_char_count": segment.get("ocr_char_count", 0),
        "transcript_excerpt": transcript_excerpt(segment.get("transcript_window", {}).get("text", "")),
        "images": [image],
        "primary_image_index": 0,
    }


def build_embedded_visuals(
    output_root: Path,
    *,
    frames: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    payload = empty_visuals_payload()
    manifest_by_id = {entry["segment_id"]: entry for entry in manifest_entries}
    frames_by_id = {frame["frame_id"]: frame for frame in frames}

    for segment in segments:
        manifest_entry = manifest_by_id.get(segment["segment_id"], {})
        effective_label = str(manifest_entry.get("effective_label", segment.get("heuristic_label")))
        bucket = visual_bucket_for_label(effective_label)
        if bucket is None:
            continue

        frame_records = collect_segment_frame_records(frames_by_id, segment)
        if not frame_records:
            continue

        primary_frame_id = choose_primary_frame_id(segment, frame_records)
        primary_image: dict[str, Any] | None = None
        for frame_record in frame_records:
            if str(frame_record.get("frame_id")) != primary_frame_id:
                continue
            image_path = output_root / str(frame_record["frame_path"])
            if not image_path.exists():
                continue
            primary_image = encode_image_payload(image_path)
            break
        if primary_image is None:
            for frame_record in frame_records:
                image_path = output_root / str(frame_record["frame_path"])
                if image_path.exists():
                    primary_image = encode_image_payload(image_path)
                    break

        if primary_image is None:
            continue

        payload[bucket].append(
            build_embedded_visual_entry(
                segment=segment,
                bucket=bucket,
                frame_records=frame_records,
                image=primary_image,
            )
        )

    return payload


def build_debug_visual_entry(
    *,
    bucket: str,
    segment: dict[str, Any],
    frame_records: list[dict[str, Any]],
    image_paths: list[str],
    primary_image_path: str | None,
) -> dict[str, Any]:
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
        effective_label = str(manifest_entry.get("effective_label", segment.get("heuristic_label")))
        bucket = visual_bucket_for_label(effective_label)
        if bucket is None:
            continue

        frame_records = collect_segment_frame_records(frames_by_id, segment)
        if not frame_records:
            continue

        target_dir = slides_dir if bucket == "slides" else charts_dir
        segment_dir = target_dir / segment["segment_id"]
        segment_dir.mkdir(parents=True, exist_ok=True)

        primary_frame_id = choose_primary_frame_id(segment, frame_records)
        image_paths: list[str] = []
        primary_image_path: str | None = None
        for frame_record in frame_records:
            source_path = output_root / str(frame_record["frame_path"])
            if not source_path.exists():
                continue
            destination_path = segment_dir / Path(str(frame_record["frame_path"])).name
            shutil.copy2(source_path, destination_path)
            relative_path = str(destination_path.relative_to(output_root))
            image_paths.append(relative_path)
            if str(frame_record.get("frame_id")) == primary_frame_id:
                primary_image_path = relative_path

        if not image_paths:
            continue
        if primary_image_path is None:
            primary_image_path = image_paths[0]

        payload[bucket].append(
            build_debug_visual_entry(
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
