from __future__ import annotations

from pathlib import Path
from typing import Any

from . import constants
from .artifacts import write_json


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# {report.get('title', '影片分析報告')}",
        "",
        "## 摘要",
        "",
        str(report.get("executive_summary", "")).strip(),
        "",
    ]

    sections = report.get("main_sections", [])
    if sections:
        lines.extend(["## 主要段落", ""])
        for section in sections:
            heading = section.get("heading") or section.get("title") or "未命名段落"
            lines.append(f"### {heading}")
            lines.append("")
            lines.append(str(section.get("summary", "")).strip())
            source_segment_ids = section.get("source_segment_ids", [])
            if source_segment_ids:
                lines.append("")
                lines.append(f"來源片段：{', '.join(source_segment_ids)}")
            lines.append("")

    key_visuals = report.get("key_visuals", [])
    if key_visuals:
        lines.extend(["## 關鍵畫面", ""])
        for visual in key_visuals:
            lines.append(f"- {visual.get('summary', '')}".strip())
        lines.append("")

    speaker_points = report.get("speaker_points", [])
    if speaker_points:
        lines.extend(["## 講者重點", ""])
        for point in speaker_points:
            lines.append(f"- {point}".strip())
        lines.append("")

    open_questions = report.get("open_questions", [])
    if open_questions:
        lines.extend(["## 待確認問題", ""])
        for question in open_questions:
            lines.append(f"- {question}".strip())
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_report_files(output_root: Path, report: dict[str, Any]) -> None:
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "report.json", report)
    (report_dir / "report.md").write_text(render_markdown_report(report), encoding="utf-8")


def normalize_source(
    *,
    source_input: str,
    is_url: bool,
    is_youtube_url: bool,
) -> dict[str, Any]:
    return {
        "input": source_input,
        "kind": "url" if is_url else "local_path",
        "is_youtube_url": is_youtube_url,
    }


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "id": metadata.get("id"),
        "title": metadata.get("title"),
        "url": metadata.get("webpage_url") or metadata.get("original_url"),
        "channel": metadata.get("channel"),
        "uploader": metadata.get("uploader"),
        "duration": metadata.get("duration") or metadata.get("format", {}).get("duration"),
        "upload_date": metadata.get("upload_date"),
        "chapters": metadata.get("chapters") or [],
    }


def normalize_transcript_segments(segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments or []:
        normalized.append(
            {
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", ""),
            }
        )
    return normalized


def normalize_transcript(transcript: dict[str, Any] | None) -> dict[str, Any]:
    transcript = transcript or {}
    return {
        "source": transcript.get("source"),
        "language": transcript.get("language"),
        "full_text": transcript.get("text", ""),
        "segments": normalize_transcript_segments(transcript.get("segments")),
    }


def summarize_processing(
    *,
    transcript: dict[str, Any] | None,
    ocr: dict[str, Any],
    visuals_payload: dict[str, list[dict[str, Any]]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    ocr_mode: str,
    gpt_mode: str,
    artifacts_mode: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "transcript_mode": transcript_mode,
        "ocr_mode": ocr_mode,
        "gpt_enabled": gpt_mode == "on",
        "artifact_mode": artifacts_mode,
        "cleanup_applied": cleanup_intermediates,
        "ocr_status": ocr.get("status"),
        "counts": {
            "transcript_segments": len((transcript or {}).get("segments", [])),
            "slide_count": len(visuals_payload.get("slides", [])),
            "chart_count": len(visuals_payload.get("charts", [])),
            "error_count": len(errors),
        },
    }


def build_output_payload(
    *,
    source_input: str,
    is_url: bool,
    is_youtube_url: bool,
    metadata: dict[str, Any] | None,
    transcript: dict[str, Any] | None,
    ocr: dict[str, Any],
    visuals_payload: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    ocr_mode: str,
    gpt_mode: str,
    artifacts_mode: str,
    gpt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "output_version": constants.OUTPUT_VERSION,
        "source": normalize_source(
            source_input=source_input,
            is_url=is_url,
            is_youtube_url=is_youtube_url,
        ),
        "metadata": normalize_metadata(metadata),
        "transcript": normalize_transcript(transcript),
        "visuals": visuals_payload,
        "processing": summarize_processing(
            transcript=transcript,
            ocr=ocr,
            visuals_payload=visuals_payload,
            cleanup_intermediates=cleanup_intermediates,
            transcript_mode=transcript_mode,
            ocr_mode=ocr_mode,
            gpt_mode=gpt_mode,
            artifacts_mode=artifacts_mode,
            errors=errors,
        ),
        "errors": errors,
    }
    if gpt_payload is not None:
        payload["gpt"] = gpt_payload
    return payload


def write_output_file(
    paths: Any,
    *,
    source_input: str,
    is_url: bool,
    is_youtube_url: bool,
    metadata: dict[str, Any] | None,
    transcript: dict[str, Any] | None,
    ocr: dict[str, Any],
    visuals_payload: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    ocr_mode: str,
    gpt_mode: str,
    artifacts_mode: str,
    gpt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_output_payload(
        source_input=source_input,
        is_url=is_url,
        is_youtube_url=is_youtube_url,
        metadata=metadata,
        transcript=transcript,
        ocr=ocr,
        visuals_payload=visuals_payload,
        errors=errors,
        cleanup_intermediates=cleanup_intermediates,
        transcript_mode=transcript_mode,
        ocr_mode=ocr_mode,
        gpt_mode=gpt_mode,
        artifacts_mode=artifacts_mode,
        gpt_payload=gpt_payload,
    )
    write_json(paths.output_json_path, payload)
    return payload
