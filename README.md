# YouTube Analysis Tool

Local-first, token-efficient YouTube analysis.

This project is built around a simple idea: the right default is not "send the
whole video to GPT."

Instead, it does most of the cheap work locally, keeps only the higher-value
visual evidence, and treats GPT as an optional escalation step. A default run
produces a durable per-video `analysis.json` without requiring GPT at all.

## Why This Is Different

Many video-summary tools follow the same broad pattern:

1. extract a large number of frames
2. send most or all of them to an LLM
3. pay the token cost for low-value footage such as talking heads, transitions,
   or repeated slides

This tool takes a more cost-optimized approach:

- local-first preprocessing before any GPT stage
- local OCR for candidate frames
- heuristic triage to separate `slides`, `chart_table`, `talking_head`,
  `b_roll`, and `uncertain`
- review and routing artifacts before GPT
- selective GPT escalation instead of bulk frame submission

The goal is not just "analyze a YouTube video." The goal is to do it without
wasting vision tokens on frames that do not carry much semantic value.

## What You Get From A Default Run

Running the tool with only `--source`:

- keeps `--gpt off`
- writes `output/youtube/<video-id-or-stem>/analysis.json`
- extracts a transcript locally first when possible
- runs local OCR in `auto` mode when keyframes exist
- writes triage, review, and routing artifacts
- cleans up large intermediate files after the run

`analysis.json` is the canonical output for each video. It survives cleanup and
bundles the normalized source metadata, transcript data, OCR status, segment
summaries, routing state, artifact references, and optional GPT output.

## When GPT Is Used

GPT is optional and off by default.

When enabled, GPT is intended for higher-semantic tasks such as:

- interpreting dense slides after OCR
- reading charts where OCR alone is not enough
- combining transcript context with selected visual evidence
- resolving a small `uncertain` bucket after local triage

GPT is not the default mechanism for:

- full-video frame scanning
- OCR extraction
- duplicate removal
- obvious talking-head suppression

`--gpt off` only disables the GPT vision and report stage. Transcript mode
`auto` currently means:

1. subtitles
2. local Whisper
3. OpenAI transcription fallback

If you want a fully local run with no OpenAI usage at all, use
`--transcript subtitles` or `--transcript whisper`.

## Quick Start

Install the package in editable mode and add the optional YouTube extras:

```bash
python3 -m pip install -e '.[youtube]'
```

Run it on a YouTube URL:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Run it on a local file:

```bash
youtube-analyze --source /path/to/video.mp4
```

Enable GPT only when you actually want the extra semantic layer:

```bash
youtube-analyze --source /path/to/video.mp4 --gpt on
```

Disable review for batch-style GPT runs:

```bash
youtube-analyze --source /path/to/video.mp4 --gpt on --review off
```

By default the tool cleans up downloaded media, extracted audio, subtitle files,
keyframe images, and OCR intermediates after each run. Use
`--keep-intermediates` if you want to retain them for debugging.

## Output Layout

```text
output/youtube/<video-id>/
├── analysis.json
├── audio/
├── video/
├── subtitles/
├── keyframes/
│   └── index.csv
├── ocr/
│   └── index.csv
├── triage/
│   ├── frames.jsonl
│   └── segments.json
├── review/
│   ├── queue.json
│   └── decisions.json
├── routing/
│   └── manifest.json
├── gpt/
│   └── analyses.json
├── report/
│   ├── report.json
│   └── report.md
├── metadata.json
├── source.txt
├── transcript.json
└── transcript.txt
```

After the default cleanup pass, large intermediate directories such as
`audio/`, `video/`, `subtitles/`, `keyframes/`, and `ocr/` are removed, but
`analysis.json` and the JSON/TXT stage artifacts remain.

## Architecture Notes

The current architecture is deliberately local-first:

1. inspect and normalize media locally
2. extract transcript and candidate keyframes locally
3. run local OCR and heuristic triage
4. keep representative segments and suppress obvious low-value footage
5. optionally escalate a small subset to GPT

These stages are good local defaults on a MacBook Air M3 with 24 GB RAM:

- `ffmpeg` / `ffprobe` for media inspection, audio extraction, scene changes,
  and frame extraction
- `yt-dlp` for YouTube download and subtitle retrieval
- local `whisper` for transcript fallback
- `tesseract` for OCR on candidate frames
- `opencv` for blur filtering, dedupe, and basic frame heuristics

Current implemented behavior includes:

- URL or local-file input
- output folders under `output/youtube/<video-id-or-stem>/`
- canonical `analysis.json` artifact per video
- subtitle-first transcript strategy
- local Whisper fallback with OpenAI transcription as final fallback
- scene-change plus interval keyframe extraction
- local OCR with `auto|off|on` modes
- heuristic frame triage with dedupe, blur scoring, motion proxy, and routing
  labels
- segment merge and transcript-window binding
- pre-GPT review queue with resumable decisions
- routing manifest for GPT candidates
- optional GPT segment analysis plus final zh-TW report

## Docs

- [Architecture](./docs/architecture.md)
- [Skill](./codex-skills/youtube-analysis/SKILL.md)
