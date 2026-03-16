from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from . import constants
from .artifacts import read_json, write_json


def queue_entry_from_manifest(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": entry["segment_id"],
        "heuristic_label": entry["heuristic_label"],
        "effective_label": entry["effective_label"],
        "heuristic_confidence": entry["heuristic_confidence"],
        "detail": entry["detail"],
        "representative_frame_paths": entry["representative_frame_paths"],
        "transcript_text": entry.get("transcript_window", {}).get("text", ""),
        "ocr_summary": entry.get("ocr_summary", ""),
        "routing_disposition": entry["routing_disposition"],
    }


def build_review_queue(output_root: Path, manifest_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for entry in manifest_entries:
        needs_review = (
            entry["effective_label"] == "uncertain"
            or float(entry["heuristic_confidence"]) < constants.DEFAULT_REVIEW_CONFIDENCE_THRESHOLD
            or entry["detail"] != "low"
        )
        entry["review_required"] = needs_review
        if needs_review:
            entry["review_status"] = "pending"
            queue.append(queue_entry_from_manifest(entry))
        elif entry["routing_disposition"] == "candidate":
            entry["review_status"] = "auto_approved"
        else:
            entry["review_status"] = "suppressed"
    write_json(output_root / "review" / "queue.json", {"queue": queue})
    return queue


def load_review_decisions(output_root: Path, *, reset: bool = False) -> dict[str, dict[str, Any]]:
    path = output_root / "review" / "decisions.json"
    if reset:
        write_json(path, {"decisions": {}})
        return {}
    payload = read_json(path, {"decisions": {}})
    return dict(payload.get("decisions", {}))


def save_review_decisions(output_root: Path, decisions: dict[str, dict[str, Any]]) -> None:
    write_json(output_root / "review" / "decisions.json", {"decisions": decisions})


def open_frame(output_root: Path, frame_path: str) -> None:
    absolute_path = output_root / frame_path
    if sys.platform == "darwin":
        subprocess.run(["open", str(absolute_path)], check=False, capture_output=True, text=True)


def interactive_review(
    output_root: Path,
    queue: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    *,
    input_func: Callable[[str], str] = input,
) -> dict[str, dict[str, Any]]:
    if input_func is input and not sys.stdin.isatty():
        raise RuntimeError("Interactive review requires a TTY.")
    allowed_labels = ", ".join(constants.ROUTING_LABELS)
    for entry in queue:
        segment_id = entry["segment_id"]
        decision = decisions.setdefault(
            segment_id,
            {
                "status": "pending",
                "label_override": None,
                "detail_override": None,
                "note": "",
            },
        )
        if decision.get("status") in {"approved", "skipped"}:
            continue
        while True:
            prompt = (
                f"{segment_id} label={entry['effective_label']} "
                f"confidence={entry['heuristic_confidence']} detail={entry['detail']} "
                "command [keep|skip|relabel <label>|note <text>|detail <low|original>|open]: "
            )
            raw = input_func(prompt).strip()
            if not raw:
                continue
            command, _, payload = raw.partition(" ")
            if command == "open":
                frame_paths = entry.get("representative_frame_paths", [])
                if frame_paths:
                    open_frame(output_root, frame_paths[0])
                continue
            if command == "note":
                decision["note"] = payload.strip()
                save_review_decisions(output_root, decisions)
                continue
            if command == "relabel":
                label = payload.strip()
                if label not in constants.ROUTING_LABELS:
                    raise ValueError(f"Unsupported label override: {label}. Allowed labels: {allowed_labels}")
                decision["label_override"] = label
                save_review_decisions(output_root, decisions)
                continue
            if command == "detail":
                detail = payload.strip()
                if detail not in {"low", "original"}:
                    raise ValueError("Detail override must be low or original.")
                decision["detail_override"] = detail
                save_review_decisions(output_root, decisions)
                continue
            if command == "keep":
                decision["status"] = "approved"
                save_review_decisions(output_root, decisions)
                break
            if command == "skip":
                decision["status"] = "skipped"
                save_review_decisions(output_root, decisions)
                break
            raise ValueError("Unsupported review command.")
    return decisions


def apply_review_decisions(
    manifest_entries: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    for entry in manifest_entries:
        decision = decisions.get(entry["segment_id"])
        if decision is None:
            if entry.get("review_required"):
                entry["review_status"] = "pending"
            continue
        if decision.get("label_override"):
            entry["effective_label"] = decision["label_override"]
        if decision.get("detail_override"):
            entry["detail"] = decision["detail_override"]
        if decision.get("note"):
            entry["review_note"] = decision["note"]
        status = decision.get("status")
        if status == "approved":
            entry["review_status"] = "approved"
        elif status == "skipped":
            entry["review_status"] = "skipped"
    return manifest_entries
