# YouTube Analysis Tool

Local-first, token-efficient YouTube analysis that produces a single AI-ready `output.json`.

This project is built around one default: do the cheap work locally first, then
escalate to GPT only when that extra semantic layer is actually worth paying
for. A normal run is designed for downstream AI consumption, not for browsing a
pile of side artifacts by hand.

## AI-First, Not Human-First

The default output is intentionally shaped for another model to read, not for a
person to casually skim in a text editor.

That means:

- one canonical `output.json` instead of a pile of sibling artifacts
- full transcript inline, because downstream AI benefits from direct context
- retained visuals embedded inline, because AI should not have to chase image paths
- explicit provenance and source labeling, so downstream readers know what came
  from subtitles, Whisper, OCR, or heuristic routing

It also means `output.json` is not optimized to feel pleasant for direct human
reading. It is a machine-facing bundle first.

## Status

The current v1 core path is validated and intentionally stable.

- long YouTube runs complete on local hardware
- the default `minimal` single-file contract is holding up
- output size remains practical even on long inputs
- downstream AI can read and reason over the resulting bundle

Unless a real failure signal shows up, the project should prefer stability over
gratuitous rewrites.

## Validated Cases

The current implementation has already been exercised against:

- short and mid-length YouTube videos with existing subtitles
- YouTube videos without usable captions, using local Whisper fallback
- long-form inputs where a single `output.json` still remained manageable
- local-only runs with `--gpt off`

This does not mean every transcript is clean. It means the core path is
producing usable AI-ready bundles across the kinds of runs it was designed for.

## Known Limits

Current limitations are mostly about source quality, not the output contract.

- no-caption videos depend on local Whisper, so transcript quality can drift
- Japanese proper nouns, niche terminology, and low-frequency names can be wrong
- `visuals` currently promote `slides` and `charts` only; other valuable visual
  material may stay unpromoted
- the tool is better at preserving high-level structure than exact scene-by-scene
  reconstruction when the transcript is noisy

## What The Default Run Gives You

Running `youtube-analyze --source ...` with no extra flags:

- keeps `--gpt off`
- keeps `--artifacts minimal`
- writes `output/youtube/<title-id-or-stem>/output.json`
- preserves the full transcript inline
- embeds retained visual evidence inline under `visuals.slides` and `visuals.charts`
- keeps one inline primary image per retained visual item in minimal mode
- cleans up large intermediate files after the run

`output.json` is the canonical durable artifact. In minimal mode it is
self-contained and does not depend on sibling JSON files, image folders, or
report files.

If you open it yourself, it may feel dense or even ugly. That is expected. The
design target is "easy for downstream AI to ingest," not "pleasant for humans
to browse raw."

## Why This Exists

Many video-analysis tools follow a costly pattern:

1. extract lots of frames
2. send most or all of them to a vision model
3. pay token cost for repeated slides, talking heads, transitions, and other low-value footage

This tool takes a different path:

- local transcript extraction first
- local OCR first
- local heuristic triage first
- selective promotion of `slides` and `charts`
- optional GPT only after the cheap filtering work is done

The goal is not just "analyze a YouTube video." The goal is to reduce
information overload without wasting tokens on frames that carry little value.

## Output Contract

The canonical output is a single JSON bundle:

```text
output/youtube/<title-id-or-stem>/
└── output.json
```

Top-level shape:

```json
{
  "output_version": "1.0.1",
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

`transcript` contains:

- `source`
- `language`
- `full_text`
- `segments`
- `segment_count`
- `provenance`

`visuals` contains only the AI-facing buckets:

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

`provenance` exists because not all sources are equally trustworthy. Manual
subtitles, YouTube auto captions, burned subtitle OCR, and Whisper output should
not be treated as if they have the same confidence profile.

## Transcript And OCR Behavior

Transcript resolution is subtitle-first:

1. manual subtitles
2. YouTube automatic captions
3. burned subtitle OCR
4. local Whisper
5. OpenAI transcription fallback

If any usable subtitle track exists, the tool uses it directly even when the
language is not in the preferred list. Language preference still affects which
subtitle gets picked first, but any available subtitle beats Whisper.

When transcript resolution succeeds from subtitles, the tool skips audio
extraction for transcript work. This optimization does not reduce the visual
pipeline: video download, keyframe extraction, OCR, triage, and retained
visuals still run as usual.

If you want a fully local run with no OpenAI usage at all, use:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID' --transcript subtitles
```

or:

```bash
youtube-analyze --source /path/to/video.mp4 --transcript whisper
```

OCR is always local. The default is `--ocr auto`, which means:

- try OCR when keyframes exist
- keep going if OCR fails
- record the degraded state in `output.json`

Burned subtitle OCR is also local, but it is intentionally conservative. In
`auto` mode it is allowed to fail fast and fall back to Whisper. If it succeeds,
that is a bonus path, not the primary contract.

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

## Quick Start

Install the package in editable mode with the optional YouTube dependencies:

```bash
python3 -m pip install -e '.[youtube]'
```

Run on a YouTube URL:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Run on a local file:

```bash
youtube-analyze --source /path/to/video.mp4
```

Enable GPT only when you want the extra semantic layer:

```bash
youtube-analyze --source /path/to/video.mp4 --gpt on
```

Disable interactive review for a more batch-like GPT run:

```bash
youtube-analyze --source /path/to/video.mp4 --gpt on --review off
```

Keep debug artifacts:

```bash
youtube-analyze --source /path/to/video.mp4 --artifacts debug
```

## Current Local-First Pipeline

The current implementation does this:

1. normalize media locally
2. extract transcript locally first when possible
3. extract candidate keyframes locally
4. run local OCR and heuristic frame triage
5. promote retained visuals into a single AI-friendly bundle
6. optionally run GPT on the filtered subset

The main local tools are:

- `ffmpeg` / `ffprobe`
- `yt-dlp`
- local `whisper`
- `tesseract`
- `opencv`

Current implemented behavior includes:

- URL or local-file input
- canonical `output.json` artifact per run
- full transcript embedded in the canonical output
- `slides` / `charts` visual galleries embedded in the canonical output
- one inline primary image per visual item in minimal mode
- local OCR with `auto|off|on`
- subtitle-first transcript strategy
- local Whisper fallback with OpenAI transcription as final fallback
- heuristic triage with dedupe, blur scoring, motion proxy, and routing labels
- optional GPT segment analysis plus final zh-TW report
- optional debug-mode trace artifacts

## Docs

- [Architecture](./docs/architecture.md)
- [Skill](./codex-skills/youtube-analysis/SKILL.md)
