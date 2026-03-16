from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .artifacts import write_json


def create_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Install the optional youtube dependencies.") from exc
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set for GPT analysis.")
    return OpenAI()


def image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type or 'application/octet-stream'};base64,{payload}"


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    outputs = getattr(response, "output", None)
    if not outputs:
        raise RuntimeError("OpenAI response did not contain output text.")
    fragments: list[str] = []
    for item in outputs:
        content = getattr(item, "content", None) or item.get("content", [])
        for block in content:
            text = getattr(block, "text", None) or block.get("text", "")
            if text:
                fragments.append(text)
    if not fragments:
        raise RuntimeError("OpenAI response did not contain text content.")
    return "\n".join(fragments).strip()


def extract_json_payload(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def call_openai_json(
    client: Any,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_paths: list[Path] | None = None,
    detail: str = "low",
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
    for image_path in image_paths or []:
        content.append(
            {
                "type": "input_image",
                "image_url": image_to_data_url(image_path),
                "detail": detail,
            }
        )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ],
    )
    return extract_json_payload(extract_response_text(response))


def segment_system_prompt(label: str) -> str:
    return (
        "You analyze curated video segments and return strict JSON. "
        f"The current routing label is {label}. "
        "Return keys: segment_summary, key_evidence, inferred_visual_type, importance, "
        "confidence, uncertainties, transcript_linkage, source_segment_id."
    )


def segment_user_prompt(entry: dict[str, Any]) -> str:
    transcript_text = entry.get("transcript_window", {}).get("text", "")
    return (
        f"Segment ID: {entry['segment_id']}\n"
        f"Routing label: {entry['effective_label']}\n"
        f"OCR summary:\n{entry.get('ocr_summary', '')}\n\n"
        f"Transcript context:\n{transcript_text}\n\n"
        "Analyze only this segment and preserve uncertainty when evidence is thin."
    )


def final_report_system_prompt(language: str) -> str:
    return (
        "You synthesize a final video report and return strict JSON. "
        f"Write all human-facing strings in {language}. "
        "Return keys: title, executive_summary, main_sections, key_visuals, "
        "speaker_points, open_questions, source_segment_ids."
    )


def final_report_user_prompt(
    *,
    transcript: dict[str, Any] | None,
    segment_analyses: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    transcript_text = ""
    if transcript:
        transcript_text = transcript.get("text", "")
    compact_analyses = json.dumps(segment_analyses, ensure_ascii=False, indent=2)
    metadata_text = json.dumps(
        {
            "id": metadata.get("id"),
            "title": metadata.get("title"),
            "duration": metadata.get("duration"),
            "uploader": metadata.get("uploader"),
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "Build a final report from the curated evidence.\n\n"
        f"Metadata:\n{metadata_text}\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        f"Segment analyses:\n{compact_analyses}\n"
    )


def analyze_segments(
    output_root: Path,
    manifest_entries: list[dict[str, Any]],
    *,
    model: str,
    transcript: dict[str, Any] | None,
    metadata: dict[str, Any],
    report_language: str,
    client: Any | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    approved_entries = [entry for entry in manifest_entries if entry.get("approved_for_gpt")]
    if client is None:
        client = create_openai_client()
    segment_analyses: list[dict[str, Any]] = []
    for entry in approved_entries:
        image_paths = [output_root / frame_path for frame_path in entry.get("representative_frame_paths", [])]
        payload = call_openai_json(
            client,
            model=model,
            system_prompt=segment_system_prompt(entry["prompt_family"]),
            user_prompt=segment_user_prompt(entry),
            image_paths=image_paths,
            detail=entry["detail"],
        )
        payload["source_segment_id"] = entry["segment_id"]
        segment_analyses.append(payload)
    final_report = call_openai_json(
        client,
        model=model,
        system_prompt=final_report_system_prompt(report_language),
        user_prompt=final_report_user_prompt(
            transcript=transcript,
            segment_analyses=segment_analyses,
            metadata=metadata,
        ),
    )
    write_json(
        output_root / "gpt" / "analyses.json",
        {"segment_analyses": segment_analyses, "final_report": final_report},
    )
    return segment_analyses, final_report
