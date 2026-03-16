---
name: youtube-analysis
description: Use when the task is to turn a YouTube URL or local video/audio file into a transcript, keyframes, OCR artifacts, or a local-first visual triage workflow that minimizes GPT usage.
---

# YouTube Analysis

## Overview

Use this skill when a user wants to analyze a YouTube video or local media file without sending every frame to a remote model.

This skill is for:

- transcript extraction
- keyframe extraction
- OCR over candidate frames
- designing or running a local-first routing workflow
- deciding which frames should be escalated to GPT

## Workflow

### 1. Normalize the task

Determine:

- source type: YouTube URL or local media file
- desired outputs: transcript, keyframes, OCR, or a filtered GPT-ready frame set
- whether the user wants implementation, design, or one-off analysis

### 2. Prefer local work first

Before calling GPT, do as much as possible locally:

- metadata inspection with `ffprobe`
- subtitle retrieval and media download with `yt-dlp`
- transcript extraction from subtitles
- fallback transcription with local `whisper`
- scene-change and interval frame extraction with `ffmpeg`
- OCR with `tesseract` / `pytesseract`
- basic image heuristics with `opencv`

### 3. Triage frames before escalation

Default frame buckets:

- `slides`
- `chart_table`
- `talking_head`
- `b_roll`
- `uncertain`

Recommended routing:

- keep `slides`
- keep `chart_table`
- suppress `talking_head`
- suppress `b_roll`
- manually review or selectively escalate `uncertain`

### 4. Escalate only high-value frames

Use GPT only when local tools are not enough, for example:

- slide meaning and takeaway
- chart trend interpretation
- summary that combines transcript and visual evidence

Do not use GPT for full-video frame scanning or bulk OCR.

## Project files

- main package: `src/youtube_analysis_tool/pipeline.py`
- CLI wrapper: `scripts/youtube_analyze.py`
- architecture note: `docs/architecture.md`

## Quick commands

Install:

```bash
python3 -m pip install -e '.[youtube]'
```

Run:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Local file:

```bash
youtube-analyze --source /path/to/video.mp4 --transcript whisper --ocr on
```

