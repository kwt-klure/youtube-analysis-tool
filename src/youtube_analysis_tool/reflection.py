from __future__ import annotations

from typing import Any

from . import constants


VISUAL_REFERENCE_KEYWORDS = (
    "這張",
    "這個畫面",
    "畫面",
    "截圖",
    "圖表",
    "投影片",
    "螢幕",
    "画面",
    "スクショ",
    "図",
    "screenshot",
    "screen",
    "slide",
    "image",
    "figure",
    "shown",
)

METHOD_QUESTION_KEYWORDS = (
    "怎麼判斷",
    "如何判斷",
    "依據",
    "根據",
    "抽幀",
    "取樣",
    "字幕",
    "逐字稿",
    "視覺",
    "畫面",
    "dB",
    "db",
    "loudness",
    "silence",
    "transcript",
    "sampling",
    "frame",
    "ocr",
)


def _text_contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _text_excerpt(text: Any, limit: int = 180) -> str:
    return str(text or "").strip()[:limit]


def _transcript_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        segment
        for segment in transcript.get("segments") or []
        if str(segment.get("text", "")).strip()
    ]


def _all_visual_items(visuals: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for bucket in ("slides", "charts"):
        for item in visuals.get(bucket) or []:
            items.append((bucket, item))
    return items


def _ranges_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
    *,
    padding_seconds: float = 10.0,
) -> bool:
    padded_left_start = left_start - padding_seconds
    padded_left_end = left_end + padding_seconds
    return max(padded_left_start, right_start) <= min(padded_left_end, right_end)


def _has_nearby_retained_visual(segment: dict[str, Any], visual_items: list[tuple[str, dict[str, Any]]]) -> bool:
    segment_start = _to_float(segment.get("start"), _to_float(segment.get("start_seconds")))
    segment_end = _to_float(segment.get("end"), _to_float(segment.get("end_seconds"), segment_start))
    for _, item in visual_items:
        visual_start = _to_float(item.get("start_seconds"))
        visual_end = _to_float(item.get("end_seconds"), visual_start)
        if _ranges_overlap(segment_start, segment_end, visual_start, visual_end):
            return True
    return False


def _add_unique(target: list[dict[str, Any]], item: dict[str, Any], *, key_fields: tuple[str, ...]) -> None:
    key = tuple(item.get(field) for field in key_fields)
    if any(tuple(existing.get(field) for field in key_fields) == key for existing in target):
        return
    target.append(item)


def _evidence_layers(
    *,
    transcript: dict[str, Any],
    visuals: dict[str, Any],
    visual_sampling: dict[str, Any],
    audio_features: dict[str, Any],
    comments: dict[str, Any],
) -> list[str]:
    layers: list[str] = []
    if transcript.get("source") not in {None, "skipped"} and (
        transcript.get("full_text") or transcript.get("segments")
    ):
        layers.append("transcript")
    if visual_sampling.get("status") == "extracted" and int(visual_sampling.get("candidate_frame_count") or 0) > 0:
        layers.append("visual_sampling")
    if _all_visual_items(visuals):
        layers.append("retained_visuals")
    if audio_features.get("status") == "extracted":
        layers.append("audio_features")
    if comments.get("status") == "extracted" and (comments.get("items") or []):
        layers.append("comments")
    return layers


def _limitations(
    *,
    transcript: dict[str, Any],
    visual_sampling: dict[str, Any],
    audio_features: dict[str, Any],
    comments: dict[str, Any],
) -> list[str]:
    limitations: list[str] = []
    if transcript.get("source") in {None, "skipped"} or not (
        transcript.get("full_text") or transcript.get("segments")
    ):
        limitations.append("transcript_unavailable")
    visual_status = visual_sampling.get("status", "skipped")
    if visual_status in {"disabled", "skipped"}:
        limitations.append("visuals_skipped")
    elif visual_status == "failed":
        limitations.append("visuals_failed")
    elif int(visual_sampling.get("retained_visual_count") or 0) == 0:
        limitations.append("no_retained_visuals")
    audio_status = audio_features.get("status", "disabled")
    if audio_status in {"disabled", "skipped"}:
        limitations.append("audio_features_disabled")
    elif audio_status == "failed":
        limitations.append("audio_features_failed")
    comment_status = comments.get("status", "disabled")
    if comment_status in {"disabled", "skipped"}:
        limitations.append("comments_disabled")
    elif comment_status == "failed":
        limitations.append("comments_failed")
    elif comment_status == "extracted" and not (comments.get("items") or []):
        limitations.append("no_comments_returned")
    return limitations


def _reflect_visual_references(
    *,
    transcript: dict[str, Any],
    visual_items: list[tuple[str, dict[str, Any]]],
    uncertainties: list[dict[str, Any]],
    adjustments: list[dict[str, Any]],
    skill_candidates: list[dict[str, Any]],
) -> None:
    for segment in _transcript_segments(transcript):
        text = str(segment.get("text") or "")
        if not _text_contains_any(text, VISUAL_REFERENCE_KEYWORDS):
            continue
        if _has_nearby_retained_visual(segment, visual_items):
            continue
        start = _to_float(segment.get("start"), _to_float(segment.get("start_seconds")))
        end = _to_float(segment.get("end"), _to_float(segment.get("end_seconds"), start))
        uncertainty = {
            "type": "visual_reference_without_retained_visual",
            "evidence_layers": ["transcript", "visual_sampling"],
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "text_excerpt": _text_excerpt(text),
            "why_it_matters": "transcript points at a visual object, but no retained visual covers the same time window",
        }
        _add_unique(uncertainties, uncertainty, key_fields=("type", "start_seconds", "text_excerpt"))
        _add_unique(
            adjustments,
            {
                "action": "increase_visual_density",
                "target": "visual_sampling",
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "reason": "visual reference was not backed by a retained visual",
            },
            key_fields=("action", "start_seconds", "reason"),
        )
        _add_unique(
            skill_candidates,
            {
                "area": "visual_sampling",
                "suggestion": "When transcript text points at screenshots, slides, or figures without a nearby retained visual, rerun that window with denser sampling.",
                "triggered_by": "visual_reference_without_retained_visual",
                "requires_human_review": True,
            },
            key_fields=("area", "triggered_by"),
        )


def _reflect_low_confidence_visuals(
    *,
    visual_items: list[tuple[str, dict[str, Any]]],
    uncertainties: list[dict[str, Any]],
    adjustments: list[dict[str, Any]],
) -> None:
    for bucket, item in visual_items:
        confidence = item.get("heuristic_confidence")
        if confidence is None or _to_float(confidence, 1.0) >= constants.DEFAULT_REFLECTION_LOW_VISUAL_CONFIDENCE:
            continue
        segment_id = item.get("segment_id")
        uncertainty = {
            "type": "low_confidence_retained_visual",
            "evidence_layers": ["retained_visuals"],
            "segment_id": segment_id,
            "visual_bucket": bucket,
            "heuristic_confidence": _to_float(confidence),
            "start_seconds": round(_to_float(item.get("start_seconds")), 3),
            "end_seconds": round(_to_float(item.get("end_seconds"), _to_float(item.get("start_seconds"))), 3),
            "why_it_matters": "retained visual was promoted with low heuristic confidence",
        }
        _add_unique(uncertainties, uncertainty, key_fields=("type", "segment_id"))
        _add_unique(
            adjustments,
            {
                "action": "inspect_debug_artifacts",
                "target": "visual_triage",
                "segment_id": segment_id,
                "reason": "low-confidence visual promotion should be inspected before relying on it",
            },
            key_fields=("action", "segment_id"),
        )


def _reflect_audio_anchors(
    *,
    audio_features: dict[str, Any],
    adjustments: list[dict[str, Any]],
) -> None:
    for change in audio_features.get("large_changes") or []:
        start = round(_to_float(change.get("start")), 3)
        end = round(_to_float(change.get("end"), start), 3)
        _add_unique(
            adjustments,
            {
                "action": "inspect_audio_change_anchor",
                "target": "cross_signal_alignment",
                "start_seconds": start,
                "end_seconds": end,
                "delta_db": change.get("delta_db"),
                "reason": "large within-video loudness change marks a useful inspection anchor",
                "caution": "audio features are routing signals, not semantic conclusions",
            },
            key_fields=("action", "start_seconds", "end_seconds"),
        )
    for kind, segments in (
        ("quiet", audio_features.get("quiet_segments") or []),
        ("loud", audio_features.get("loud_segments") or []),
    ):
        for segment in segments[:2]:
            start = round(_to_float(segment.get("start")), 3)
            end = round(_to_float(segment.get("end"), start), 3)
            _add_unique(
                adjustments,
                {
                    "action": f"inspect_{kind}_audio_anchor",
                    "target": "cross_signal_alignment",
                    "start_seconds": start,
                    "end_seconds": end,
                    "reason": f"{kind} audio window may help align transcript, frame, and scene boundaries",
                    "caution": "audio features are routing signals, not semantic conclusions",
                },
                key_fields=("action", "start_seconds", "end_seconds"),
            )


def _reflect_method_comments(
    *,
    comments: dict[str, Any],
    uncertainties: list[dict[str, Any]],
    adjustments: list[dict[str, Any]],
    skill_candidates: list[dict[str, Any]],
) -> None:
    for item in comments.get("items") or []:
        text = str(item.get("text") or "")
        if "?" not in text and "？" not in text:
            continue
        if not _text_contains_any(text, METHOD_QUESTION_KEYWORDS):
            continue
        comment_id = item.get("id")
        uncertainty = {
            "type": "method_question_from_comments",
            "evidence_layers": ["comments"],
            "comment_id": comment_id,
            "text_excerpt": _text_excerpt(text),
            "why_it_matters": "a visible comment challenges how the video or agent inferred something",
        }
        _add_unique(uncertainties, uncertainty, key_fields=("type", "comment_id", "text_excerpt"))
        _add_unique(
            adjustments,
            {
                "action": "review_comment_method_challenge",
                "target": "comments",
                "comment_id": comment_id,
                "reason": "comment asks for evidence or method behind an inference",
            },
            key_fields=("action", "comment_id", "reason"),
        )
        _add_unique(
            skill_candidates,
            {
                "area": "comment_triage",
                "suggestion": "Classify method-question comments as critique signals, not general audience sentiment.",
                "triggered_by": "method_question_from_comments",
                "requires_human_review": True,
            },
            key_fields=("area", "triggered_by"),
        )


def _overall_confidence(status: str, layers: list[str], uncertainty_count: int) -> str:
    if status == "skipped":
        return "none"
    if status == "limited":
        return "limited"
    if uncertainty_count:
        return "medium"
    if {"transcript", "visual_sampling", "retained_visuals", "audio_features", "comments"}.issubset(set(layers)):
        return "high"
    return "medium"


def build_run_reflection(
    *,
    transcript: dict[str, Any] | None,
    visuals: dict[str, Any] | None,
    visual_sampling: dict[str, Any] | None,
    audio_features: dict[str, Any] | None,
    comments: dict[str, Any] | None,
) -> dict[str, Any]:
    transcript = transcript or {}
    visuals = visuals or {"slides": [], "charts": []}
    visual_sampling = visual_sampling or {}
    audio_features = audio_features or {}
    comments = comments or {}

    visual_items = _all_visual_items(visuals)
    uncertainties: list[dict[str, Any]] = []
    adjustments: list[dict[str, Any]] = []
    skill_candidates: list[dict[str, Any]] = []

    _reflect_visual_references(
        transcript=transcript,
        visual_items=visual_items,
        uncertainties=uncertainties,
        adjustments=adjustments,
        skill_candidates=skill_candidates,
    )
    _reflect_low_confidence_visuals(
        visual_items=visual_items,
        uncertainties=uncertainties,
        adjustments=adjustments,
    )
    if audio_features.get("status") == "extracted":
        _reflect_audio_anchors(audio_features=audio_features, adjustments=adjustments)
    if comments.get("status") == "extracted":
        _reflect_method_comments(
            comments=comments,
            uncertainties=uncertainties,
            adjustments=adjustments,
            skill_candidates=skill_candidates,
        )

    layers = _evidence_layers(
        transcript=transcript,
        visuals=visuals,
        visual_sampling=visual_sampling,
        audio_features=audio_features,
        comments=comments,
    )
    limitations = _limitations(
        transcript=transcript,
        visual_sampling=visual_sampling,
        audio_features=audio_features,
        comments=comments,
    )
    if not layers:
        status = "skipped"
    elif limitations or uncertainties:
        status = "limited"
    else:
        status = "extracted"

    return {
        "status": status,
        "summary": {
            "overall_confidence": _overall_confidence(status, layers, len(uncertainties)),
            "available_evidence_layers": layers,
            "limitations": limitations,
            "uncertainty_count": len(uncertainties),
            "next_adjustment_count": len(adjustments),
        },
        "uncertainties": uncertainties,
        "next_run_adjustments": adjustments,
        "skill_patch_candidates": skill_candidates,
        "provenance": {
            "method": "local_heuristic_cross_signal_reflection",
            "trust": "routing_signal_not_semantics",
            "quality_notes": [
                "run reflection is deterministic heuristic evidence routing, not GPT interpretation",
                "skill patch candidates are suggestions and require human review",
            ],
        },
    }
