# Architecture

## Goal

Analyze YouTube videos without wasting GPT tokens on low-value frames.

## Core principle

Use a local-first funnel:

1. Download and normalize locally.
2. Extract transcript and candidate keyframes locally.
3. Score frames locally.
4. Keep only useful frames.
5. Send a very small subset to GPT for semantic interpretation.

## Processing layers

### 1. Ingest

- input: YouTube URL or local media file
- local tools: `yt-dlp`, `ffprobe`
- outputs: media metadata, local video/audio, subtitle files when available

### 2. Transcript

Priority order:

1. existing subtitles
2. local Whisper fallback
3. OpenAI transcription fallback

The transcript path should stay independent from visual selection.

### 3. Candidate frame generation

Generate candidates from:

- scene changes
- fixed intervals as backfill

This prevents missing important static slides when scene detection is sparse.

### 4. Local frame triage

Each frame should be scored with cheap local features:

- OCR text density
- motion history around timestamp
- blur/sharpness
- near-duplicate similarity
- chart-like text hints

v1 keeps these heuristics deterministic and centralized behind provisional defaults so they can be tuned without rewriting the pipeline.

Target labels:

- `slides`
- `chart_table`
- `talking_head`
- `b_roll`
- `uncertain`

### 5. Escalation policy

Default routing:

- `slides` -> keep
- `chart_table` -> keep
- `talking_head` -> suppress
- `b_roll` -> suppress
- `uncertain` -> limited review

Only kept frames should be considered for GPT.

Before GPT, v1 can stop at a terminal review gate for:

- `uncertain` segments
- low-confidence routed segments
- segments that need elevated image detail

### 6. GPT usage

GPT should answer questions that local tools cannot:

- "What is the point of this slide?"
- "What trend does this chart show?"
- "Summarize what the speaker says while this chart is shown."

GPT should not be used for:

- full-video frame scanning
- OCR-only extraction
- obvious talking-head rejection
- duplicate frame removal

## Artifacts

The default contract is a single AI-friendly bundle:

- `output.json`

`output.json` is the canonical long-term artifact for each video. It embeds:

- normalized source metadata
- the full transcript plus timestamped transcript segments
- grouped `slides` / `charts` visual galleries with one inline primary Base64 image per item in minimal mode
- compact processing metadata
- errors
- optional GPT output

Traceable stage outputs such as `triage/`, `review/`, `routing/`, `visuals/`,
`gpt/`, `report/`, `metadata.json`, and `transcript.json` still exist, but only
when debug artifacts are explicitly enabled.

## Apple Silicon fit

For a MacBook Air M3 with 24 GB RAM, the recommended split is:

- always local: `ffmpeg`, `ffprobe`, OCR, dedupe, blur, scene detection, transcript fallback
- maybe local: small CLIP/VLM classifier if it behaves well on MLX
- remote GPT: only semantic interpretation of a filtered frame set

This keeps heat, memory use, and token spend reasonable.
