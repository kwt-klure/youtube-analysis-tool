# YouTube Analysis Tool

Local-first YouTube and video analysis that writes one canonical AI-facing
bundle: `output.json`.

This project is built for a specific workflow:

1. do the cheap extraction work locally
2. preserve transcript, timing, visuals, and provenance in one place
3. let another model decide what is worth deeper reasoning

The default output is intentionally optimized for downstream AI consumption, not
for a human skimming raw JSON in a text editor.

## What This Is

`youtube-analysis-tool` takes a YouTube URL or local media file and produces a
single structured bundle that another model can read directly.

The default run tries to answer:

- what was said
- when it was said
- which visuals were important enough to keep
- where those visuals came from
- how trustworthy the transcript source is

Instead of leaving you with a pile of loosely related artifacts, the tool folds
the useful parts into one `output.json`.

## What This Is Not

This is not a polished end-user summarizer.

It does not try to:

- replace human judgment
- produce perfect semantic understanding locally
- turn every video into a pretty human-readable report
- send the whole video to GPT by default

The project is deliberately narrower than that. It focuses on extraction,
alignment, provenance, and cost discipline.

## Why The Output Looks Dense

The default output is shaped for AI, not for comfortable human reading.

That means:

- one canonical `output.json` instead of many sibling files
- full transcript inline, because downstream AI benefits from direct context
- retained visuals embedded inline, so AI does not have to chase image paths
- explicit provenance fields, so downstream readers know whether a transcript
  came from manual subtitles, YouTube auto captions, burned subtitle OCR, or
  local Whisper

If you open the JSON yourself, it may feel dense or ugly. That is expected.

## What A Default Run Gives You

Running `youtube-analyze --source ...` with no extra flags:

- keeps `--gpt off`
- keeps `--visuals off`
- keeps `--artifacts minimal`
- writes `output/youtube/<title-id-or-stem>/output.json`
- preserves the full transcript inline
- skips the visual pipeline unless you explicitly opt in with `--visuals on`
- cleans up large intermediate files after the run

`output.json` is the canonical durable artifact. In minimal mode it is
self-contained and does not depend on sibling JSON files, image folders, or
report files.

## Installation

### Requirements

You will usually want these tools available on your system `PATH`:

- `ffmpeg`
- `ffprobe`
- `yt-dlp`
- `tesseract`
- `whisper`

The local Whisper CLI is only needed when the transcript path falls back to
local transcription. If a usable subtitle track exists, the tool will prefer
that and skip Whisper.

The CLI also prints simple phase progress to `stderr` while it runs. It is
intentionally stage-based, not a fake universal percentage bar.

OpenAI is optional:

- `--gpt on` needs a valid `OPENAI_API_KEY`
- API transcription fallback also needs `OPENAI_API_KEY`
- the tool will automatically load `.env` from the current working directory or
  one of its parent directories

The built-in GPT workflow is currently OpenAI-only. If you want to use another
LLM provider, the intended path today is to run with `--gpt off` and feed the
resulting `output.json` into your own downstream model workflow.

### API Key Setup

For GPT-backed features, copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENAI_API_KEY=your_real_key_here
```

You can still use a normal exported environment variable if you prefer. The
local `.env` file is only a convenience path for people using the repo
directly.

### Python Package

Install the package in editable mode with the optional YouTube dependencies:

```bash
python3 -m pip install -e '.[youtube]'
```

## Quick Start

Run on a YouTube URL:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Run on a local file:

```bash
youtube-analyze --source /path/to/video.mp4
```

Use GPT only when you explicitly want the extra semantic layer:

```bash
youtube-analyze --source /path/to/video.mp4 --gpt on
```

Keep debug artifacts instead of only the canonical bundle:

```bash
youtube-analyze --source /path/to/video.mp4 --artifacts debug
```

Skip the entire visual pipeline for transcript-only runs:

```bash
youtube-analyze --source /path/to/video.mp4 --visuals off
```

Opt in to retained visuals when you actually want keyframes, frame OCR, and
local triage:

```bash
youtube-analyze --source /path/to/video.mp4 --visuals on
```

Force a fully local transcript path when subtitles are available:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID' --transcript subtitles
```

Try local Whisper when there are no usable subtitles:

```bash
youtube-analyze --source /path/to/video.mp4 --transcript whisper
```

Opt in to burned subtitle OCR when you specifically want to try it:

```bash
youtube-analyze --source /path/to/video.mp4 --burned-subtitles on
```

Enable GPT after your API key is set:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID' --gpt on
```

## Common Run Patterns

### Fastest Useful Run For Subtitle-Rich Videos

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID' --visuals off
```

Use this when you mostly care about transcript and timing. If a usable subtitle
track exists, the tool should skip burned subtitle OCR, skip Whisper, and avoid
all visual work.

### Normal Local-First Run

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID' --visuals on
```

This keeps transcript plus retained visuals in a single AI-facing bundle.

### Inspect Internals

```bash
youtube-analyze --source /path/to/video.mp4 --artifacts debug
```

This keeps stage outputs such as `triage/`, `review/`, `routing/`, and
`visuals/` for inspection.

## What `--gpt on` Actually Sends

GPT is still opt-in and off by default.

The built-in GPT path is currently OpenAI-only. That does **not** mean the
project is OpenAI-only overall: the canonical `output.json` bundle is meant to
be provider-neutral. If you prefer another LLM, keep `--gpt off` and hand
`output.json` to your own Claude, Gemini, OpenRouter, local-model, or custom
agent workflow.

When you enable it, the tool does **not** upload the entire video and it does
**not** send the whole `output.json` bundle as-is.

Current behavior is:

1. segment pass
   - only segments marked `approved_for_gpt` are sent
   - each approved segment sends:
     - routing label
     - OCR summary
     - transcript window text
     - up to 3 representative frames
2. final synthesis pass
   - compact metadata
   - the full transcript text
   - the segment analyses returned by the first pass

This means the biggest GPT-side payload is usually the final synthesis step,
not the per-segment image pass.

## Transcript Resolution

Transcript resolution is subtitle-first:

1. manual subtitles
2. YouTube automatic captions
3. local Whisper
4. OpenAI transcription fallback

Optional burned subtitle OCR can be inserted before Whisper with
`--burned-subtitles on`. It is currently disabled by default because Whisper is
the more reliable general fallback.

Important details:

- if any usable subtitle track exists, it wins over Whisper
- language preference affects ordering, but any usable subtitle beats Whisper
- YouTube auto-translated subtitle tracks are treated as unusable by default;
  if only translated tracks exist, the tool falls back to original-audio ASR
- burned subtitle OCR is local, conservative, and opt-in by default
- if you enable it, it is allowed to fail fast and fall back
- if subtitles satisfy transcript resolution, the tool skips audio extraction
  for transcript work

This makes subtitle-rich videos much cheaper and faster than videos that must
go through local Whisper.

## Visual Pipeline

The visual pipeline is separate from transcript resolution.

The default is `--visuals off`. Opt in with `--visuals on` when the visual
layer is worth the extra local cost for that run.

With `--visuals on`, the tool may:

- extract keyframes
- run local OCR on frames
- run local triage
- retain promoted visuals as `slides` and `charts`
- embed primary images inline in `output.json`

With `--visuals off`, the tool hard-skips:

- keyframe extraction
- frame OCR
- local triage
- visual bundle assembly

This is useful when a video already has good subtitles and visuals are not
worth the extra local cost for that run.

## Output Contract

The canonical output is a single JSON bundle:

```text
output/youtube/<title-id-or-stem>/
└── output.json
```

Top-level shape:

```json
{
  "output_version": "1.0.7",
  "source": {},
  "metadata": {},
  "transcript": {},
  "visuals": {
    "slides": [],
    "charts": []
  },
  "processing": {},
  "provenance": {},
  "errors": [],
  "gpt": {}
}
```

Small truncated example:

```json
{
  "output_version": "1.0.7",
  "source": {
    "kind": "youtube",
    "input": "https://www.youtube.com/watch?v=VIDEO_ID"
  },
  "metadata": {
    "id": "VIDEO_ID",
    "title": "Example Video",
    "uploader": "Example Channel",
    "duration_seconds": 742.0
  },
  "transcript": {
    "source": "subtitle_auto",
    "language": "ja",
    "segment_count": 1573,
    "full_text": ".... full transcript text omitted ....",
    "segments": [
      {
        "start": 0.0,
        "end": 2.4,
        "text": "第一段字幕"
      }
    ],
    "provenance": {
      "kind": "subtitle_auto",
      "quality_notes": [
        "text_track_subtitles"
      ]
    },
    "interpretation": {
      "trust": "medium_low",
      "read_mode": "verify_entities",
      "caution": ["names", "numbers", "exact_wording"]
    }
  },
  "visuals": {
    "slides": [
      {
        "segment_id": "segment-0002",
        "effective_label": "slide",
        "start_hms": "00:04:00",
        "end_hms": "00:04:36",
        "ocr_summary": "投影片上的主要文字摘要",
        "transcript_excerpt": "這段畫面附近的 transcript 片段",
        "images": [
          {
            "filename": "interval-000240.jpg",
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "data": "<base64 omitted>"
          }
        ],
        "primary_image_index": 0,
        "source_segment_ref": "segments/segment-0002",
        "provenance": {
          "kind": "triage_promoted_visual"
        }
      }
    ],
    "charts": []
  },
  "processing": {
    "transcript_mode": "auto",
    "visuals_mode": "on",
    "artifacts_mode": "minimal",
    "gpt_mode": "off"
  },
  "provenance": {
    "transcript_source": "subtitle_auto"
  },
  "errors": []
}
```

The real bundle is usually much larger than this example because `full_text`,
timestamped `segments`, and embedded primary images are all kept inline on
purpose.

### `metadata`

The canonical metadata section is normalized for downstream use. It keeps the
useful fields such as title, uploader, duration, upload date, and chapters when
available. It is not meant to dump the entire raw `yt-dlp` metadata matrix in
minimal mode.

### `transcript`

`transcript` contains:

- `source`
- `language`
- `full_text`
- `segments`
- `segment_count`
- `provenance`
- `interpretation` when the source needs extra reading caution
  including a tiny `read_mode` such as `verify_entities` or `topic_only`

This is intentionally redundant from a human perspective. Another model usually
benefits from having both:

- the full semantic context in `full_text`
- the timestamped alignment in `segments`

### `visuals`

`visuals` contains only the canonical AI-facing buckets:

- `slides`
- `charts`

Each retained visual item includes:

- `segment_id`
- `effective_label`
- `heuristic_confidence`
- timing fields
- `ocr_text`
- `ocr_summary`
- `ocr_char_count`
- `transcript_excerpt`
- `images`
- `primary_image_index`
- `source_segment_ref`
- `provenance`

In minimal mode, `images` normally contains exactly one Base64-embedded primary
image. The full OCR text stays inline because it is usually more useful to AI
than extra near-duplicate frames.

### `processing`

`processing` is intentionally compact. It records what modes were selected and
what path the run took, without regrowing into a full trace dump.

Fields here include things like:

- transcript mode
- visuals mode
- artifact mode
- GPT mode
- burned subtitle OCR status
- compact counts

### `provenance`

`provenance` exists because not all sources are equally trustworthy.

Manual subtitles, YouTube auto captions, burned subtitle OCR, and Whisper do
not have the same confidence profile. The bundle tries to say that explicitly
instead of pretending every transcript source is equal.

When a lower-trust transcript source needs extra caution, the bundle may add a
small `transcript.interpretation` object. This is intentionally sparse and
AI-facing: it is not extra metadata, but a compact hint about how aggressively
another model should trust names, numbers, and exact wording. When needed, it
may also include a tiny `read_mode` and a couple of short quality signals such
as rolling-caption overlap or heavy fragmentation. Direct text-track subtitles
in the
`subtitle_manual` path stay unannotated by default.

When `visuals_mode` is `off`, `provenance.visuals.selection_kind` is marked as
`skipped` so the bundle does not imply that heuristic visual promotion ran.

## Artifact Modes

There are two output modes.

`minimal` is the default:

- writes only `output.json`
- keeps the canonical bundle self-contained
- removes trace artifacts after the bundle is assembled

`debug` is opt-in:

- keeps stage artifacts such as `triage/`, `review/`, `routing/`, and `visuals/`
- keeps `metadata.json`, `transcript.json`, `transcript.txt`, and `source.txt`
- keeps separate GPT/report artifacts when those stages run

Use debug mode when you want to inspect internals:

```bash
youtube-analyze --source /path/to/video.mp4 --artifacts debug
```

Large intermediate directories such as `audio/`, `video/`, `subtitles/`,
`keyframes/`, and `ocr/` are still cleaned by default in both modes unless you
pass `--keep-intermediates`.

## Folder Naming

For YouTube URLs, the default output folder uses the video title plus the video
id:

```text
output/youtube/<title-id>/
```

This keeps runs readable while still preserving a stable unique suffix. If a
title is unavailable, the tool falls back to the video id. Local files continue
to use the file stem.

## Why This Project Exists

Many video-analysis tools follow an expensive pattern:

1. extract lots of frames
2. send most or all of them to a vision model
3. pay token cost for repeated slides, talking heads, transitions, and other
   low-value footage

This project takes a different path:

- local transcript extraction first
- local OCR first
- local heuristic triage first
- selective promotion of `slides` and `charts`
- optional GPT only after the cheap filtering work is done

The goal is not just "analyze a YouTube video." The goal is to reduce
information overload without wasting tokens on frames that carry little value.

## Current Status

The current v1 core path is validated and intentionally stable.

- long YouTube runs complete on local hardware
- the default `minimal` single-file contract is holding up
- output size remains practical even on long inputs
- downstream AI can read and reason over the resulting bundle

Unless a real failure signal shows up, the project should prefer stability over
gratuitous rewrites.

## Known Limits

Current limitations are mostly about source quality, not the output contract.

- no-caption videos depend on local Whisper, so transcript quality can drift
- Japanese proper nouns, niche terminology, and low-frequency names can be wrong
- `visuals` currently promote `slides` and `charts` only; other valuable visual
  material may stay unpromoted
- burned subtitle OCR is intentionally conservative and should be treated as an
  opt-in lucky fallback, not a guaranteed transcript path
- the tool is better at preserving high-level structure than exact scene-by-scene
  reconstruction when the transcript is noisy

## Current Local-First Pipeline

The current implementation does this:

1. normalize media locally
2. resolve transcript from the cheapest trustworthy source available
3. optionally extract candidate keyframes locally
4. optionally run local OCR and heuristic frame triage
5. promote retained visuals into a single AI-friendly bundle
6. optionally run GPT on the filtered subset

The main local tools are:

- `ffmpeg` / `ffprobe`
- `yt-dlp`
- local `whisper`
- `tesseract`
- `opencv`

Implemented behavior includes:

- URL or local-file input
- canonical `output.json` artifact per run
- full transcript embedded in the canonical output
- `slides` / `charts` visual galleries embedded in the canonical output
- one inline primary image per visual item in minimal mode
- subtitle-first transcript strategy
- local Whisper fallback with OpenAI transcription as final fallback
- local OCR with `auto|off|on`
- burned subtitle OCR fallback with fast fail behavior (disabled by default)
- heuristic triage with dedupe, blur scoring, motion proxy, and routing labels
- optional GPT segment analysis plus final zh-TW report
- optional debug-mode trace artifacts

## Docs

- [Architecture](./docs/architecture.md)
- [Skill](./codex-skills/youtube-analysis/SKILL.md)
