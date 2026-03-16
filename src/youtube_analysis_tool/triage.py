from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from . import constants
from .artifacts import write_json, write_jsonl


Matrix = list[list[float]]


def read_grayscale_image(path: Path) -> Matrix | None:
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image.tolist()


def resize_matrix(matrix: Matrix, size: int) -> Matrix:
    if not matrix or not matrix[0]:
        return []
    height = len(matrix)
    width = len(matrix[0])
    if height == size and width == size:
        return [row[:] for row in matrix]
    resized: Matrix = []
    for row_index in range(size):
        source_row = min(height - 1, int(row_index * height / size))
        row: list[float] = []
        for column_index in range(size):
            source_column = min(width - 1, int(column_index * width / size))
            row.append(float(matrix[source_row][source_column]))
        resized.append(row)
    return resized


def compute_blur_score(matrix: Matrix | None) -> float:
    if matrix is None or len(matrix) < 3 or len(matrix[0]) < 3:
        return 0.0
    laplacians: list[float] = []
    for row_index in range(1, len(matrix) - 1):
        for column_index in range(1, len(matrix[row_index]) - 1):
            center = float(matrix[row_index][column_index])
            laplacian = (
                4 * center
                - float(matrix[row_index - 1][column_index])
                - float(matrix[row_index + 1][column_index])
                - float(matrix[row_index][column_index - 1])
                - float(matrix[row_index][column_index + 1])
            )
            laplacians.append(laplacian)
    if not laplacians:
        return 0.0
    mean = sum(laplacians) / len(laplacians)
    variance = sum((value - mean) ** 2 for value in laplacians) / len(laplacians)
    return round(variance, 3)


def compute_phash(matrix: Matrix | None, image_path: Path) -> str:
    if matrix is None:
        return hashlib.sha1(image_path.read_bytes()).hexdigest()[:16]
    resized = resize_matrix(matrix, 8)
    values = [value for row in resized for value in row]
    if not values:
        return hashlib.sha1(image_path.read_bytes()).hexdigest()[:16]
    mean = sum(values) / len(values)
    bits = "".join("1" if value >= mean else "0" for value in values)
    return f"{int(bits, 2):016x}"


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


def compute_motion_proxy(matrix: Matrix | None, previous_matrix: Matrix | None) -> float:
    if matrix is None or previous_matrix is None:
        return 0.0
    left = resize_matrix(matrix, 16)
    right = resize_matrix(previous_matrix, 16)
    diffs: list[float] = []
    for row_index in range(min(len(left), len(right))):
        for column_index in range(min(len(left[row_index]), len(right[row_index]))):
            diffs.append(abs(left[row_index][column_index] - right[row_index][column_index]) / 255.0)
    if not diffs:
        return 0.0
    return round(sum(diffs) / len(diffs), 3)


def ocr_char_count(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def numeric_token_ratio(text: str) -> float:
    tokens = re.findall(r"[A-Za-z]+|\d+(?:[.,]\d+)?%?", text)
    if not tokens:
        return 0.0
    numeric = sum(1 for token in tokens if any(character.isdigit() for character in token))
    return round(numeric / len(tokens), 3)


def chart_hint_score(text: str) -> float:
    keywords = (
        "chart",
        "table",
        "axis",
        "growth",
        "revenue",
        "quarter",
        "q1",
        "q2",
        "q3",
        "q4",
        "%",
        "yoy",
    )
    lowered = text.lower()
    hits = sum(1 for keyword in keywords if keyword in lowered)
    if hits == 0:
        return 0.0
    return round(min(1.0, hits / 4), 3)


def build_heuristic_scores(frame: dict[str, Any]) -> dict[str, float]:
    scores = {label: 0.0 for label in constants.ROUTING_LABELS if label != "uncertain"}
    chars = int(frame.get("ocr_char_count", 0))
    motion = float(frame.get("motion_proxy", 0.0))
    blur = float(frame.get("blur_score", 0.0))
    numeric_ratio = float(frame.get("numeric_token_ratio", 0.0))
    chart_score = float(frame.get("chart_hint_score", 0.0))

    if chars >= constants.DEFAULT_SLIDE_OCR_CHAR_COUNT:
        scores["slides"] += 0.55
    if motion <= constants.DEFAULT_MOTION_LOW_THRESHOLD:
        scores["slides"] += 0.20
        scores["chart_table"] += 0.10
        scores["talking_head"] += 0.20
    if blur >= constants.DEFAULT_BLUR_REJECTION_VARIANCE:
        scores["slides"] += 0.10
        scores["chart_table"] += 0.10
        scores["talking_head"] += 0.10
        scores["b_roll"] += 0.05

    if numeric_ratio >= constants.DEFAULT_CHART_NUMERIC_RATIO:
        scores["chart_table"] += 0.45
    if chart_score >= constants.DEFAULT_CHART_HINT_THRESHOLD:
        scores["chart_table"] += 0.35
    if chars >= max(10, constants.DEFAULT_SLIDE_OCR_CHAR_COUNT // 2):
        scores["chart_table"] += 0.10

    if chars <= constants.DEFAULT_TALKING_HEAD_OCR_CHAR_MAX:
        scores["talking_head"] += 0.45
        scores["b_roll"] += 0.25
    if motion >= constants.DEFAULT_MOTION_HIGH_THRESHOLD:
        scores["b_roll"] += 0.45
    if motion < constants.DEFAULT_MOTION_LOW_THRESHOLD / 2 and chars <= constants.DEFAULT_TALKING_HEAD_OCR_CHAR_MAX:
        scores["talking_head"] += 0.20
    if numeric_ratio < constants.DEFAULT_CHART_NUMERIC_RATIO:
        scores["b_roll"] += 0.05

    return {label: round(min(score, 1.0), 3) for label, score in scores.items()}


def choose_heuristic_label(frame: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    scores = build_heuristic_scores(frame)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, best_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = round(min(1.0, best_score * 0.75 + max(0.0, best_score - second_score) * 0.5), 3)
    if best_score < 0.55:
        return "uncertain", min(confidence, 0.55), scores
    return label, confidence, scores


def build_ocr_lookup(ocr_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in ocr_rows:
        lookup[str(row.get("filename", ""))] = row
    return lookup


def relative_frame_path(frame_filename: str) -> str:
    return str(Path("keyframes") / frame_filename)


def assign_duplicate_groups(frames: list[dict[str, Any]]) -> None:
    groups: list[dict[str, Any]] = []
    for frame in sorted(frames, key=lambda item: float(item["timestamp_seconds"])):
        selected_group: dict[str, Any] | None = None
        for group in groups:
            if hamming_distance(frame["phash"], group["reference_phash"]) <= constants.DEFAULT_PHASH_DUPLICATE_DISTANCE:
                selected_group = group
                break
        if selected_group is None:
            selected_group = {
                "group_id": f"dup-{len(groups) + 1:04d}",
                "reference_phash": frame["phash"],
                "members": [],
            }
            groups.append(selected_group)
        selected_group["members"].append(frame)
        frame["duplicate_group"] = selected_group["group_id"]

    for group in groups:
        members = sorted(
            group["members"],
            key=lambda item: (
                float(item.get("blur_score", 0.0)),
                int(item.get("ocr_char_count", 0)),
            ),
            reverse=True,
        )
        representative_frame_id = members[0]["frame_id"] if members else None
        for member in group["members"]:
            member["is_duplicate_representative"] = member["frame_id"] == representative_frame_id


def recommend_keep_drop(frame: dict[str, Any]) -> str:
    if float(frame.get("blur_score", 0.0)) < constants.DEFAULT_BLUR_REJECTION_VARIANCE:
        return "drop_blurry"
    if not frame.get("is_duplicate_representative", False):
        return "drop_duplicate"
    if frame.get("heuristic_label") in {"talking_head", "b_roll"}:
        return "drop_low_value"
    return "keep"


def summarize_ocr_text(records: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for record in records:
        text = str(record.get("ocr_text", "")).strip()
        if text and text not in chunks:
            chunks.append(text)
    if not chunks:
        return ""
    summary = "\n".join(chunks)
    return summary[:1000]


def collect_transcript_window(
    transcript: dict[str, Any] | None,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    if not transcript:
        return {
            "start_seconds": round(start_seconds, 3),
            "end_seconds": round(end_seconds, 3),
            "text": "",
            "segments": [],
        }
    padding = constants.DEFAULT_TRANSCRIPT_CONTEXT_PADDING_SECONDS
    window_start = max(0.0, start_seconds - padding)
    window_end = end_seconds + padding
    matched_segments = []
    for segment in transcript.get("segments", []):
        if float(segment.get("end", 0.0)) < window_start:
            continue
        if float(segment.get("start", 0.0)) > window_end:
            continue
        matched_segments.append(segment)
    text = "\n".join(segment.get("text", "").strip() for segment in matched_segments if segment.get("text"))
    return {
        "start_seconds": round(window_start, 3),
        "end_seconds": round(window_end, 3),
        "text": text.strip(),
        "segments": matched_segments,
    }


def representative_frame_paths(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    ordered = sorted(records, key=lambda item: float(item["timestamp_seconds"]))
    selected = [ordered[0]]
    sharpest = max(ordered, key=lambda item: float(item.get("blur_score", 0.0)))
    if sharpest["frame_id"] != ordered[0]["frame_id"]:
        selected.append(sharpest)
    if ordered[-1]["frame_id"] not in {frame["frame_id"] for frame in selected}:
        selected.append(ordered[-1])
    return [frame["frame_path"] for frame in selected[:3]]


def merge_frames_to_segments(
    frames: list[dict[str, Any]],
    transcript: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates = [
        frame
        for frame in sorted(frames, key=lambda item: float(item["timestamp_seconds"]))
        if frame.get("is_duplicate_representative", False)
        and frame.get("keep_drop_recommendation") != "drop_blurry"
    ]
    segments: list[list[dict[str, Any]]] = []
    for frame in candidates:
        if not segments:
            segments.append([frame])
            continue
        previous_segment = segments[-1]
        previous_frame = previous_segment[-1]
        gap = float(frame["timestamp_seconds"]) - float(previous_frame["timestamp_seconds"])
        if (
            frame["heuristic_label"] == previous_frame["heuristic_label"]
            and gap <= constants.DEFAULT_SEGMENT_MERGE_GAP_SECONDS
        ):
            previous_segment.append(frame)
            continue
        segments.append([frame])

    results: list[dict[str, Any]] = []
    for index, records in enumerate(segments, start=1):
        start_seconds = float(records[0]["timestamp_seconds"])
        end_seconds = float(records[-1]["timestamp_seconds"])
        confidence = round(
            sum(float(record.get("heuristic_confidence", 0.0)) for record in records) / len(records),
            3,
        )
        transcript_window = collect_transcript_window(transcript, start_seconds, end_seconds)
        label = str(records[0]["heuristic_label"])
        disposition = "candidate" if label in constants.GPT_PROMPT_LABELS else "suppressed"
        results.append(
            {
                "segment_id": f"segment-{index:04d}",
                "heuristic_label": label,
                "heuristic_confidence": confidence,
                "start_seconds": round(start_seconds, 3),
                "end_seconds": round(end_seconds, 3),
                "start_hms": records[0]["timestamp_hms"],
                "end_hms": records[-1]["timestamp_hms"],
                "frame_ids": [record["frame_id"] for record in records],
                "representative_frame_paths": representative_frame_paths(records),
                "ocr_summary": summarize_ocr_text(records),
                "ocr_char_count": max(int(record.get("ocr_char_count", 0)) for record in records),
                "numeric_token_ratio": round(
                    max(float(record.get("numeric_token_ratio", 0.0)) for record in records),
                    3,
                ),
                "chart_hint_score": round(
                    max(float(record.get("chart_hint_score", 0.0)) for record in records),
                    3,
                ),
                "transcript_window": transcript_window,
                "review_required": False,
                "review_status": "auto_approved" if disposition == "candidate" else "suppressed",
                "routing_disposition": disposition,
            }
        )
    return results


def build_frame_records(
    output_root: Path,
    keyframe_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ocr_lookup = build_ocr_lookup(ocr_rows)
    records: list[dict[str, Any]] = []
    previous_matrix: Matrix | None = None
    for index, row in enumerate(sorted(keyframe_rows, key=lambda item: float(item["timestamp_seconds"])), start=1):
        frame_path = output_root / "keyframes" / str(row["filename"])
        matrix = read_grayscale_image(frame_path)
        ocr_row = ocr_lookup.get(str(row["filename"]), {})
        ocr_text = str(ocr_row.get("text", "") or "").strip()
        record = {
            "frame_id": f"frame-{index:05d}",
            "timestamp_seconds": round(float(row["timestamp_seconds"]), 3),
            "timestamp_hms": row["timestamp_hms"],
            "frame_kind": row["kind"],
            "filename": row["filename"],
            "frame_path": relative_frame_path(str(row["filename"])),
            "ocr_text": ocr_text,
            "ocr_char_count": ocr_char_count(ocr_text),
            "blur_score": compute_blur_score(matrix),
            "phash": compute_phash(matrix, frame_path),
            "motion_proxy": compute_motion_proxy(matrix, previous_matrix),
            "numeric_token_ratio": numeric_token_ratio(ocr_text),
            "chart_hint_score": chart_hint_score(ocr_text),
        }
        label, confidence, label_scores = choose_heuristic_label(record)
        record["heuristic_label"] = label
        record["heuristic_confidence"] = confidence
        record["heuristic_scores"] = label_scores
        records.append(record)
        previous_matrix = matrix
    assign_duplicate_groups(records)
    for record in records:
        record["keep_drop_recommendation"] = recommend_keep_drop(record)
    return records


def run_local_triage(
    output_root: Path,
    keyframe_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
    transcript: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    triage_dir = output_root / "triage"
    triage_dir.mkdir(parents=True, exist_ok=True)
    frames = build_frame_records(output_root, keyframe_rows, ocr_rows)
    segments = merge_frames_to_segments(frames, transcript)
    write_jsonl(triage_dir / "frames.jsonl", frames)
    write_json(triage_dir / "segments.json", {"segments": segments})
    return frames, segments
