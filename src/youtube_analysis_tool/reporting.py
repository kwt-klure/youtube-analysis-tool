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


def transcript_extraction_kind(source: str | None) -> str | None:
    value = str(source or "")
    if value in {"subtitle", "subtitle_manual", "subtitle_auto"}:
        return "text_track"
    if value == "burned_subtitle_ocr":
        return "frame_ocr"
    if value == "whisper":
        return "local_asr"
    if value == "openai":
        return "remote_asr"
    if not value:
        return None
    return "unknown"


def transcript_quality_notes(source: str | None) -> list[str]:
    value = str(source or "")
    if value in {"subtitle", "subtitle_manual"}:
        return []
    if value == "subtitle_auto":
        return ["machine_generated_text_track"]
    if value == "burned_subtitle_ocr":
        return ["frame_ocr_may_be_noisy"]
    if value == "whisper":
        return ["local_asr_may_mishear_proper_nouns"]
    if value == "openai":
        return ["remote_asr_may_mishear_proper_nouns"]
    if not value:
        return ["transcript_missing"]
    return ["transcript_source_unknown"]


def normalize_segment_text_for_quality(text: str) -> str:
    return "".join(str(text or "").split()).casefold()


def transcript_quality_signals(segments: list[dict[str, Any]] | None) -> list[str]:
    texts = [
        normalize_segment_text_for_quality(segment.get("text", ""))
        for segment in segments or []
    ]
    texts = [text for text in texts if text]
    if not texts:
        return []

    short_count = sum(1 for text in texts if len(text) <= 3)
    duplicate_hits = 0
    overlap_hits = 0
    adjacent_pairs = max(0, len(texts) - 1)

    for previous, current in zip(texts, texts[1:]):
        if previous == current and len(current) >= 6:
            duplicate_hits += 1
            continue
        shorter, longer = sorted((previous, current), key=len)
        if len(shorter) >= 6 and shorter in longer:
            overlap_hits += 1

    signals: list[str] = []
    if len(texts) >= 6 and short_count / len(texts) >= 0.45:
        signals.append("fragmented")
    if adjacent_pairs >= 3 and overlap_hits / adjacent_pairs >= 0.35:
        signals.append("overlap_heavy")
    if adjacent_pairs >= 3 and duplicate_hits / adjacent_pairs >= 0.2:
        signals.append("duplicate_heavy")
    return signals


def downgrade_trust_level(value: str) -> str:
    if value == "medium":
        return "medium_low"
    if value == "medium_low":
        return "low"
    return value


def transcript_read_mode(trust: str) -> str:
    if trust == "low":
        return "topic_only"
    return "verify_entities"


def transcript_interpretation(transcript: dict[str, Any] | None) -> dict[str, Any] | None:
    transcript = transcript or {}
    source = str(transcript.get("source") or "")
    if source in {"subtitle", "subtitle_manual"}:
        return None
    signals = transcript_quality_signals(transcript.get("segments"))

    base_trust: str | None
    if source == "subtitle_auto":
        base_trust = "medium_low"
    elif source in {"whisper", "openai"}:
        base_trust = "medium"
    elif source == "burned_subtitle_ocr":
        base_trust = "low"
    else:
        base_trust = None

    if base_trust is None and not signals:
        return None

    trust = base_trust or "medium_low"
    if base_trust is not None and signals:
        trust = downgrade_trust_level(trust)

    interpretation = {
        "trust": trust,
        "read_mode": transcript_read_mode(trust),
        "caution": ["names", "numbers", "exact_wording"],
    }
    if signals:
        interpretation["signals"] = signals
    return interpretation


def normalize_transcript_provenance(transcript: dict[str, Any] | None) -> dict[str, Any]:
    transcript = transcript or {}
    source = transcript.get("source")
    extraction_kind = transcript_extraction_kind(source)
    return {
        "source": source,
        "extraction_kind": extraction_kind,
        "is_direct_text_track": extraction_kind == "text_track",
        "quality_notes": transcript_quality_notes(source),
    }


def normalize_transcript(transcript: dict[str, Any] | None) -> dict[str, Any]:
    transcript = transcript or {}
    normalized = {
        "source": transcript.get("source"),
        "language": transcript.get("language"),
        "full_text": transcript.get("text", ""),
        "segments": normalize_transcript_segments(transcript.get("segments")),
        "segment_count": transcript.get("segment_count", len(transcript.get("segments") or [])),
        "provenance": normalize_transcript_provenance(transcript),
    }
    interpretation = transcript_interpretation(transcript)
    if interpretation is not None:
        normalized["interpretation"] = interpretation
    return normalized


def summarize_visual_bucket_provenance(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(items),
        "segment_ids": [item.get("segment_id") for item in items if item.get("segment_id")],
    }


def normalize_visual_item(item: dict[str, Any], *, bucket: str) -> dict[str, Any]:
    canonical_label = "slide" if bucket == "slides" else "chart"
    source_segment_ref = item.get("source_segment_ref") or {
        "segment_id": item.get("segment_id"),
        "artifact_path": "triage/segments.json",
        "visual_bucket": bucket,
    }
    provenance = item.get("provenance") or {
        "selection_kind": "heuristic_segment_promotion",
        "effective_label_source": "routing_label_after_review_or_heuristic_fallback",
        "timing_source": "triage_segment",
        "ocr_text_source": "frame_ocr_aggregate",
        "transcript_excerpt_source": "transcript_window_text",
        "image_source": "representative_or_first_available_frame",
        "quality_notes": [
            "effective label is a routing/debug label, not semantic understanding",
            f"visual is grouped into canonical {canonical_label} bucket for downstream AI consumption",
        ],
    }
    normalized = dict(item)
    normalized["source_segment_ref"] = source_segment_ref
    normalized["provenance"] = provenance
    return normalized


def normalize_visuals_payload(
    visuals_payload: dict[str, list[dict[str, Any]]] | None,
) -> dict[str, list[dict[str, Any]]]:
    visuals_payload = visuals_payload or {}
    return {
        "slides": [
            normalize_visual_item(item, bucket="slides")
            for item in visuals_payload.get("slides", [])
        ],
        "charts": [
            normalize_visual_item(item, bucket="charts")
            for item in visuals_payload.get("charts", [])
        ],
    }


def summarize_provenance(
    *,
    transcript: dict[str, Any] | None,
    visuals_payload: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "extraction_kind": "direct_metadata_extract",
            "quality_notes": [],
        },
        "transcript": normalize_transcript_provenance(transcript),
        "visuals": {
            "selection_kind": "heuristic_segment_promotion",
            "quality_notes": [
                "visual labels are heuristic routing labels",
                "visual timing is aligned to triage segments",
            ],
            "slides": summarize_visual_bucket_provenance(visuals_payload.get("slides", [])),
            "charts": summarize_visual_bucket_provenance(visuals_payload.get("charts", [])),
        },
    }


def summarize_processing(
    *,
    transcript: dict[str, Any] | None,
    ocr: dict[str, Any],
    burned_subtitles: dict[str, Any],
    visuals_payload: dict[str, list[dict[str, Any]]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    visuals_mode: str,
    ocr_mode: str,
    gpt_mode: str,
    artifacts_mode: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "transcript_mode": transcript_mode,
        "visuals_mode": visuals_mode,
        "ocr_mode": ocr_mode,
        "burned_subtitles_mode": burned_subtitles.get("mode"),
        "gpt_enabled": gpt_mode == "on",
        "artifact_mode": artifacts_mode,
        "cleanup_applied": cleanup_intermediates,
        "ocr_status": ocr.get("status"),
        "burned_subtitles_status": burned_subtitles.get("status"),
        "burned_subtitles_reason": burned_subtitles.get("reason"),
        "burned_subtitles_probe_hits": burned_subtitles.get("probe_hits", 0),
        "burned_subtitles_ocr_events": burned_subtitles.get("ocr_event_count", 0),
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
    burned_subtitles: dict[str, Any],
    visuals_payload: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    visuals_mode: str,
    ocr_mode: str,
    gpt_mode: str,
    artifacts_mode: str,
    gpt_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_visuals = normalize_visuals_payload(visuals_payload)
    payload = {
        "output_version": constants.OUTPUT_VERSION,
        "source": normalize_source(
            source_input=source_input,
            is_url=is_url,
            is_youtube_url=is_youtube_url,
        ),
        "metadata": normalize_metadata(metadata),
        "transcript": normalize_transcript(transcript),
        "visuals": normalized_visuals,
        "processing": summarize_processing(
            transcript=transcript,
            ocr=ocr,
            burned_subtitles=burned_subtitles,
            visuals_payload=normalized_visuals,
            cleanup_intermediates=cleanup_intermediates,
            transcript_mode=transcript_mode,
            visuals_mode=visuals_mode,
            ocr_mode=ocr_mode,
            gpt_mode=gpt_mode,
            artifacts_mode=artifacts_mode,
            errors=errors,
        ),
        "provenance": summarize_provenance(
            transcript=transcript,
            visuals_payload=normalized_visuals,
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
    burned_subtitles: dict[str, Any],
    visuals_payload: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    cleanup_intermediates: bool,
    transcript_mode: str,
    visuals_mode: str,
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
        burned_subtitles=burned_subtitles,
        visuals_payload=visuals_payload,
        errors=errors,
        cleanup_intermediates=cleanup_intermediates,
        transcript_mode=transcript_mode,
        visuals_mode=visuals_mode,
        ocr_mode=ocr_mode,
        gpt_mode=gpt_mode,
        artifacts_mode=artifacts_mode,
        gpt_payload=gpt_payload,
    )
    write_json(paths.output_json_path, payload)
    return payload
