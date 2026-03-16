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
    if key_visuals:
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


def artifact_reference(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "exists": path.exists(),
    }


def normalize_source(
    *,
    source_input: str,
    output_root: Path,
    is_url: bool,
    is_youtube_url: bool,
    transcript_mode: str,
    keyframe_mode: str,
    ocr_mode: str,
    triage_mode: str,
    gpt_mode: str,
    review_mode: str,
    cleanup_intermediates: bool,
    interval_seconds: int,
    scene_threshold: float,
) -> dict[str, Any]:
    return {
        "input": source_input,
        "kind": "url" if is_url else "local_path",
        "is_youtube_url": is_youtube_url,
        "output_root": str(output_root),
        "modes": {
            "transcript": transcript_mode,
            "keyframes": keyframe_mode,
            "ocr": ocr_mode,
            "triage": triage_mode,
            "gpt": gpt_mode,
            "review": review_mode,
        },
        "cleanup_intermediates": cleanup_intermediates,
        "interval_seconds": interval_seconds,
        "scene_threshold": scene_threshold,
    }


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    metadata = metadata or {}
    return {
        "id": metadata.get("id"),
        "title": metadata.get("title"),
        "duration": metadata.get("duration") or metadata.get("format", {}).get("duration"),
        "uploader": metadata.get("uploader"),
        "channel": metadata.get("channel"),
        "webpage_url": metadata.get("webpage_url"),
        "original_url": metadata.get("original_url"),
        "source_type": "youtube" if metadata.get("id") else "local_or_unknown",
    }


def normalize_transcript(transcript: dict[str, Any] | None) -> dict[str, Any]:
    transcript = transcript or {}
    return {
        "source": transcript.get("source"),
        "language": transcript.get("language"),
        "source_path": transcript.get("source_path"),
        "segment_count": transcript.get("segment_count", 0),
        "text": transcript.get("text", ""),
        "segments": transcript.get("segments", []),
    }


def merge_segment_payloads(
    segments: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = {entry["segment_id"]: entry for entry in manifest_entries}
    merged: list[dict[str, Any]] = []
    for segment in segments:
        entry = manifest_by_id.get(segment["segment_id"], {})
        merged.append(
            {
                "segment_id": segment["segment_id"],
                "heuristic_label": segment.get("heuristic_label"),
                "effective_label": entry.get("effective_label", segment.get("heuristic_label")),
                "heuristic_confidence": segment.get("heuristic_confidence"),
                "start_seconds": segment.get("start_seconds"),
                "end_seconds": segment.get("end_seconds"),
                "start_hms": segment.get("start_hms"),
                "end_hms": segment.get("end_hms"),
                "frame_ids": segment.get("frame_ids", []),
                "representative_frame_paths": segment.get("representative_frame_paths", []),
                "ocr_summary": segment.get("ocr_summary", ""),
                "ocr_char_count": segment.get("ocr_char_count", 0),
                "numeric_token_ratio": segment.get("numeric_token_ratio", 0.0),
                "chart_hint_score": segment.get("chart_hint_score", 0.0),
                "transcript_window": segment.get("transcript_window", {}),
                "review_required": entry.get("review_required", segment.get("review_required", False)),
                "review_status": entry.get("review_status", segment.get("review_status")),
                "routing_disposition": entry.get("routing_disposition", segment.get("routing_disposition")),
                "approved_for_gpt": entry.get("approved_for_gpt", False),
                "detail": entry.get("detail"),
                "prompt_family": entry.get("prompt_family"),
                "review_note": entry.get("review_note", ""),
            }
        )
    return merged


def summarize_routing(manifest_entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "segment_count": len(manifest_entries),
        "approved_for_gpt_count": sum(1 for entry in manifest_entries if entry.get("approved_for_gpt")),
        "pending_review_count": sum(1 for entry in manifest_entries if entry.get("routing_disposition") == "pending_review"),
        "suppressed_count": sum(1 for entry in manifest_entries if entry.get("routing_disposition") == "suppressed"),
        "candidate_count": sum(1 for entry in manifest_entries if entry.get("routing_disposition") == "candidate"),
        "review_required_count": sum(1 for entry in manifest_entries if entry.get("review_required")),
    }


def collect_artifact_references(paths: Any) -> dict[str, Any]:
    root = paths.root
    return {
        "metadata_json": artifact_reference(root, paths.metadata_path),
        "source_txt": artifact_reference(root, paths.source_path),
        "transcript_json": artifact_reference(root, paths.transcript_json_path),
        "transcript_txt": artifact_reference(root, paths.transcript_text_path),
        "triage_frames_jsonl": artifact_reference(root, paths.triage_frames_path),
        "triage_segments_json": artifact_reference(root, paths.triage_segments_path),
        "review_queue_json": artifact_reference(root, paths.review_queue_path),
        "review_decisions_json": artifact_reference(root, paths.review_decisions_path),
        "routing_manifest_json": artifact_reference(root, paths.routing_manifest_path),
        "gpt_analyses_json": artifact_reference(root, paths.gpt_analyses_path),
        "report_json": artifact_reference(root, paths.report_json_path),
        "report_markdown": artifact_reference(root, paths.report_markdown_path),
        "error_json": artifact_reference(root, paths.error_path),
        "intermediates": {
            "audio_dir": artifact_reference(root, paths.audio_dir),
            "video_dir": artifact_reference(root, paths.video_dir),
            "subtitles_dir": artifact_reference(root, paths.subtitles_dir),
            "keyframes_dir": artifact_reference(root, paths.keyframes_dir),
            "ocr_dir": artifact_reference(root, paths.ocr_dir),
        },
    }


def build_analysis_payload(
    paths: Any,
    *,
    source_input: str,
    is_url: bool,
    is_youtube_url: bool,
    metadata: dict[str, Any] | None,
    transcript: dict[str, Any] | None,
    ocr: dict[str, Any],
    segments: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    keyframe_mode: str,
    ocr_mode: str,
    triage_mode: str,
    gpt_mode: str,
    review_mode: str,
    interval_seconds: int,
    scene_threshold: float,
    gpt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = paths.root
    payload = {
        "analysis_version": constants.ANALYSIS_VERSION,
        "source": normalize_source(
            source_input=source_input,
            output_root=root,
            is_url=is_url,
            is_youtube_url=is_youtube_url,
            transcript_mode=transcript_mode,
            keyframe_mode=keyframe_mode,
            ocr_mode=ocr_mode,
            triage_mode=triage_mode,
            gpt_mode=gpt_mode,
            review_mode=review_mode,
            cleanup_intermediates=cleanup_intermediates,
            interval_seconds=interval_seconds,
            scene_threshold=scene_threshold,
        ),
        "metadata": normalize_metadata(metadata),
        "transcript": normalize_transcript(transcript),
        "ocr": ocr,
        "segments": merge_segment_payloads(segments, manifest_entries),
        "routing": summarize_routing(manifest_entries),
        "artifacts": collect_artifact_references(paths),
        "errors": errors,
    }
    if gpt_payload is not None:
        payload["gpt"] = gpt_payload
    return payload


def write_analysis_file(
    paths: Any,
    *,
    source_input: str,
    is_url: bool,
    is_youtube_url: bool,
    metadata: dict[str, Any] | None,
    transcript: dict[str, Any] | None,
    ocr: dict[str, Any],
    segments: list[dict[str, Any]],
    manifest_entries: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    keyframe_mode: str,
    ocr_mode: str,
    triage_mode: str,
    gpt_mode: str,
    review_mode: str,
    interval_seconds: int,
    scene_threshold: float,
    gpt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = build_analysis_payload(
        paths,
        source_input=source_input,
        is_url=is_url,
        is_youtube_url=is_youtube_url,
        metadata=metadata,
        transcript=transcript,
        ocr=ocr,
        segments=segments,
        manifest_entries=manifest_entries,
        errors=errors,
        cleanup_intermediates=cleanup_intermediates,
        transcript_mode=transcript_mode,
        keyframe_mode=keyframe_mode,
        ocr_mode=ocr_mode,
        triage_mode=triage_mode,
        gpt_mode=gpt_mode,
        review_mode=review_mode,
        interval_seconds=interval_seconds,
        scene_threshold=scene_threshold,
        gpt_payload=gpt_payload,
    )
    payload["artifacts"]["analysis_json"] = {
        "path": str(paths.analysis_json_path.relative_to(paths.root)),
        "exists": True,
    }
    write_json(paths.analysis_json_path, payload)
    return payload
