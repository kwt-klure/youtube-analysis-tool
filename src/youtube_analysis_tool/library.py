from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import constants


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index, filter, and grep local youtube-analysis-tool output bundles."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=constants.DEFAULT_OUTPUT_ROOT,
        help="Root directory to scan for output.json bundles.",
    )
    parser.add_argument("--transcript-source", help="Filter by transcript source.")
    parser.add_argument("--language", help="Filter by transcript language.")
    parser.add_argument("--trust", help="Filter by transcript trust.")
    parser.add_argument("--read-mode", help="Filter by transcript read_mode.")
    parser.add_argument("--channel-contains", help="Case-insensitive channel substring filter.")
    parser.add_argument(
        "--has-errors",
        action="store_true",
        help="Only include bundles with one or more recorded errors.",
    )
    parser.add_argument(
        "--grep",
        help="Case-insensitive substring search across title, channel, chapter titles, and transcript full_text.",
    )
    return parser.parse_args(argv)


def derive_record_trust(payload: dict[str, Any]) -> str | None:
    interpretation = (payload.get("transcript") or {}).get("interpretation") or {}
    if interpretation.get("trust"):
        return interpretation["trust"]
    source = (payload.get("transcript") or {}).get("source")
    if source in {"subtitle", "subtitle_manual"}:
        return "high"
    return None


def derive_record(payload_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") or {}
    transcript = payload.get("transcript") or {}
    interpretation = transcript.get("interpretation") or {}
    source = payload.get("source") or {}
    processing = payload.get("processing") or {}
    errors = payload.get("errors") or []
    return {
        "path": str(payload_path.resolve()),
        "url": metadata.get("url") or source.get("input"),
        "title": metadata.get("title"),
        "channel": metadata.get("channel") or metadata.get("uploader"),
        "upload_date": metadata.get("upload_date"),
        "transcript_source": transcript.get("source"),
        "language": transcript.get("language"),
        "trust": derive_record_trust(payload),
        "read_mode": interpretation.get("read_mode"),
        "visuals_mode": processing.get("visuals_mode"),
        "error_count": len(errors),
    }


def grep_match_fields(payload: dict[str, Any], query: str) -> list[str]:
    lowered = query.casefold()
    metadata = payload.get("metadata") or {}
    transcript = payload.get("transcript") or {}
    chapter_titles = [chapter.get("title", "") for chapter in metadata.get("chapters") or []]
    haystacks = {
        "title": metadata.get("title") or "",
        "channel": metadata.get("channel") or "",
        "chapters": "\n".join(chapter_titles),
        "full_text": transcript.get("full_text") or "",
    }
    return [
        name
        for name, value in haystacks.items()
        if lowered in str(value).casefold()
    ]


def matches_filters(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.transcript_source and record.get("transcript_source") != args.transcript_source:
        return False
    if args.language and record.get("language") != args.language:
        return False
    if args.trust and record.get("trust") != args.trust:
        return False
    if args.read_mode and record.get("read_mode") != args.read_mode:
        return False
    if args.channel_contains:
        channel = str(record.get("channel") or "")
        if args.channel_contains.casefold() not in channel.casefold():
            return False
    if args.has_errors and int(record.get("error_count") or 0) <= 0:
        return False
    return True


def iter_bundle_payloads(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for payload_path in sorted(root.rglob("output.json")):
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except Exception as exc:
            sys.stderr.write(f"[library] Skipping unreadable bundle {payload_path}: {exc}\n")
            sys.stderr.flush()
            continue
        rows.append((payload_path, payload))
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for payload_path, payload in iter_bundle_payloads(args.root):
        record = derive_record(payload_path, payload)
        if not matches_filters(record, args):
            continue
        if args.grep:
            match_fields = grep_match_fields(payload, args.grep)
            if not match_fields:
                continue
            record["match_fields"] = match_fields
        sys.stdout.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
