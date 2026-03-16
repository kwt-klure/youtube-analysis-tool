from __future__ import annotations

from pathlib import Path
from typing import Any

from . import constants
from .artifacts import write_json


def prompt_family_for_label(label: str) -> str | None:
    if label in constants.GPT_PROMPT_LABELS:
        return label
    return None


def model_supports_original(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in constants.MODEL_PREFIXES_SUPPORTING_ORIGINAL)


def resolve_detail_for_model(detail: str, model: str) -> str:
    if detail == "original" and not model_supports_original(model):
        return "high"
    return detail


def default_detail_for_segment(segment: dict[str, Any]) -> str:
    label = str(segment.get("heuristic_label", "uncertain"))
    if label == "chart_table":
        return "original"
    if int(segment.get("ocr_char_count", 0)) >= constants.DEFAULT_DENSE_SLIDE_OCR_CHAR_COUNT:
        return "original"
    return "low"


def manifest_entry_from_segment(segment: dict[str, Any], model: str) -> dict[str, Any]:
    label = str(segment.get("heuristic_label", "uncertain"))
    prompt_family = prompt_family_for_label(label)
    detail = resolve_detail_for_model(default_detail_for_segment(segment), model)
    disposition = "candidate" if prompt_family else "suppressed"
    reason = []
    if disposition == "suppressed":
        reason.append("label_not_routable")
    else:
        reason.append("high_value_visual_candidate")
    if detail != "low":
        reason.append(f"detail={detail}")
    return {
        "segment_id": segment["segment_id"],
        "heuristic_label": label,
        "effective_label": label,
        "heuristic_confidence": segment["heuristic_confidence"],
        "representative_frame_paths": segment["representative_frame_paths"],
        "ocr_summary": segment["ocr_summary"],
        "transcript_window": segment["transcript_window"],
        "review_required": False,
        "review_status": segment.get("review_status", "pending"),
        "routing_disposition": disposition,
        "approved_for_gpt": False,
        "prompt_family": prompt_family,
        "detail": detail,
        "reason": reason,
        "review_note": "",
    }


def finalize_manifest_entries(
    manifest_entries: list[dict[str, Any]],
    *,
    review_enabled: bool,
) -> list[dict[str, Any]]:
    for entry in manifest_entries:
        label = entry["effective_label"]
        entry["prompt_family"] = prompt_family_for_label(label)
        if entry["prompt_family"] is None:
            entry["routing_disposition"] = "suppressed"
            entry["approved_for_gpt"] = False
            continue
        if entry.get("review_required") and entry.get("review_status") not in {"approved"}:
            entry["routing_disposition"] = "pending_review"
            entry["approved_for_gpt"] = False
            continue
        if entry.get("review_status") == "skipped":
            entry["routing_disposition"] = "skipped"
            entry["approved_for_gpt"] = False
            continue
        entry["routing_disposition"] = "candidate"
        entry["approved_for_gpt"] = entry.get("review_status") in {"auto_approved", "approved"} or not entry.get(
            "review_required"
        )
    return manifest_entries


def build_routing_manifest(
    output_root: Path,
    segments: list[dict[str, Any]],
    *,
    model: str,
    review_enabled: bool,
) -> list[dict[str, Any]]:
    manifest_entries = [manifest_entry_from_segment(segment, model) for segment in segments]
    finalize_manifest_entries(manifest_entries, review_enabled=review_enabled)
    write_json(output_root / "routing" / "manifest.json", {"segments": manifest_entries})
    return manifest_entries
