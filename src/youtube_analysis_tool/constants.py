from __future__ import annotations

from pathlib import Path


DEFAULT_INTERVAL_SECONDS = 60
DEFAULT_SCENE_THRESHOLD = 0.35
DEFAULT_OUTPUT_ROOT = Path("output") / "youtube"
ANALYSIS_VERSION = "1.1"
SUBTITLE_LANGUAGE_PREFERENCE = (
    "zh-tw",
    "zh-hant",
    "zh-hk",
    "zh-cn",
    "zh-hans",
    "zh",
    "en",
    "en-us",
)
YOUTUBE_HOST_MARKERS = ("youtube.com", "youtu.be", "m.youtube.com")

DEFAULT_TRIAGE_MODE = "on"
DEFAULT_OCR_MODE = "auto"
DEFAULT_GPT_MODE = "off"
DEFAULT_REVIEW_MODE = "interactive"
DEFAULT_GPT_MODEL = "gpt-5.4"
DEFAULT_REPORT_LANGUAGE = "zh-TW"

# These defaults are intentionally provisional. They are centralized here so
# calibration can tune them later without rewriting the rule logic.
DEFAULT_BLUR_REJECTION_VARIANCE = 100.0
DEFAULT_PHASH_DUPLICATE_DISTANCE = 6
DEFAULT_SLIDE_OCR_CHAR_COUNT = 40
DEFAULT_DENSE_SLIDE_OCR_CHAR_COUNT = 120
DEFAULT_TALKING_HEAD_OCR_CHAR_MAX = 20
DEFAULT_CHART_NUMERIC_RATIO = 0.20
DEFAULT_CHART_HINT_THRESHOLD = 0.60
DEFAULT_MOTION_LOW_THRESHOLD = 0.08
DEFAULT_MOTION_HIGH_THRESHOLD = 0.18
DEFAULT_SEGMENT_MERGE_GAP_SECONDS = 12.0
DEFAULT_TRANSCRIPT_CONTEXT_PADDING_SECONDS = 5.0
DEFAULT_REVIEW_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_AUTO_APPROVE_CONFIDENCE = 0.82

MODEL_PREFIXES_SUPPORTING_ORIGINAL = ("gpt-5",)
GPT_PROMPT_LABELS = ("slides", "chart_table", "uncertain")
ROUTING_LABELS = ("slides", "chart_table", "talking_head", "b_roll", "uncertain")
