from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .version import add_version_argument


NON_FATAL_SEVERITIES = {"info", "warning", "nonfatal", "non_fatal"}


def error_is_fatal(error: Any) -> bool:
    if not isinstance(error, dict):
        return True
    if str(error.get("kind", "")).lower() == "interrupted":
        return True
    severity = str(error.get("severity", "fatal")).lower()
    return severity not in NON_FATAL_SEVERITIES


def check_bundle_payload(payload: Any, path: Path) -> dict[str, Any]:
    bundle_path = path.expanduser().resolve()
    result: dict[str, Any] = {
        "bundle_path": str(bundle_path),
        "valid": False,
        "parseable": True,
        "run_status": None,
        "error_count": 0,
        "fatal_error_count": 0,
        "transcript_source": None,
        "visual_status": None,
        "failure_reasons": [],
    }
    if not isinstance(payload, dict):
        result["failure_reasons"].append("bundle_not_object")
        return result

    processing = payload.get("processing")
    run_status = processing.get("run_status") if isinstance(processing, dict) else None
    result["run_status"] = run_status

    errors = payload.get("errors")
    if not isinstance(errors, list):
        errors = []
        result["failure_reasons"].append("errors_not_array")
    fatal_errors = [error for error in errors if error_is_fatal(error)]
    result["error_count"] = len(errors)
    result["fatal_error_count"] = len(fatal_errors)

    transcript = payload.get("transcript")
    if isinstance(transcript, dict):
        result["transcript_source"] = transcript.get("source")
    visual_sampling = payload.get("visual_sampling")
    if isinstance(visual_sampling, dict):
        result["visual_status"] = visual_sampling.get("status")

    if run_status is None:
        result["failure_reasons"].append("run_status_missing")
    elif run_status != "completed":
        result["failure_reasons"].append(f"run_status_{run_status}")
    if fatal_errors:
        result["failure_reasons"].append("fatal_errors_present")

    result["valid"] = not result["failure_reasons"]
    return result


def check_bundle(path: Path) -> dict[str, Any]:
    bundle_path = path.expanduser().resolve()
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        result = check_bundle_payload(None, bundle_path)
        result["parseable"] = False
        result["failure_reasons"] = [
            "bundle_not_found" if isinstance(exc, FileNotFoundError) else "bundle_not_parseable"
        ]
        return result
    return check_bundle_payload(payload, bundle_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a youtube-analysis-tool output bundle is complete."
    )
    add_version_argument(parser)
    parser.add_argument("bundle", type=Path, help="Path to output.json")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable result (the default output is a compact status line)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = check_bundle(args.bundle)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        status = "valid" if result["valid"] else "invalid"
        reasons = ",".join(result["failure_reasons"]) or "none"
        print(f"{status}: {result['bundle_path']} ({reasons})")
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
