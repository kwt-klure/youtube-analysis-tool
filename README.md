# YouTube Analysis Tool

Standalone tooling for turning a YouTube URL or local media file into:

- a durable per-video `analysis.json`
- transcript artifacts
- keyframe snapshots
- OCR artifacts
- local triage artifacts and review queues
- optional GPT-backed segment analysis and final report

## Current conclusion

The right architecture is not "send the whole video to GPT."

Instead:

1. Run cheap, high-volume preprocessing locally.
2. Keep only representative, high-value frames.
3. Send only `slides`, `charts/tables`, and `uncertain` frames to GPT.
4. Skip or heavily downsample `talking head` and `B-roll/demo` footage.

This keeps token usage under control and makes the system easier to scale.

## What should run locally

These stages are good local defaults on a MacBook Air M3 with 24 GB RAM:

- `ffmpeg` / `ffprobe` for media inspection, audio extraction, scene changes, and frame extraction
- `yt-dlp` for YouTube download and subtitle retrieval
- local `whisper` for fallback transcription
- `tesseract` for OCR on candidate frames
- `opencv` for blur filtering, dedupe, and basic frame heuristics
- optional face detection to suppress talking-head segments

## What should go to GPT

Use GPT only for high-semantic tasks:

- interpreting dense slides after OCR
- reading charts where OCR alone is not enough
- cross-linking transcript and visuals into summaries
- reviewing a small `uncertain` bucket that local rules cannot classify confidently

## Recommended routing policy

- `slides` -> keep, OCR locally, send selected frames to GPT when interpretation is needed
- `chart/table` -> keep, OCR locally, send selected frames to GPT when trend/explanation is needed
- `talking head` -> default skip
- `B-roll/demo clip` -> default skip
- `uncertain` -> review locally first, then escalate a few frames to GPT

## Current implementation status

Implemented now:

- URL or local-file input
- output folder layout under `output/youtube/<video-id-or-stem>/`
- canonical `analysis.json` artifact per video
- metadata capture
- subtitle-first transcript strategy
- local Whisper fallback
- OpenAI-transcribe final fallback hook
- scene-change plus interval keyframe extraction
- local OCR with `auto|off|on` modes (`auto` by default)
- heuristic frame triage with dedupe, blur scoring, motion proxy, and routing labels
- segment merge and transcript-window binding
- pre-GPT review queue with resumable decisions
- routing manifest for GPT candidates
- optional GPT segment analysis plus final zh-TW report

Planned next:

- richer visual heuristics such as face ratio
- calibration against more real-world video fixtures
- optional Apple-silicon-friendly local vision classification pass

## Quick start

Install the package in editable mode and add the optional YouTube extras:

```bash
python3 -m pip install -e '.[youtube]'
```

Then run:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

This default run keeps `--gpt off`, writes `output/youtube/<video-id>/analysis.json`,
and uses local OCR in `auto` mode when keyframes are available.

Or with a local file:

```bash
youtube-analyze --source /path/to/video.mp4 --transcript whisper --gpt on
```

Review can be disabled for batch runs:

```bash
youtube-analyze --source /path/to/video.mp4 --gpt on --review off
```

By default the tool cleans up downloaded media, extracted audio, subtitle files,
keyframe images, and OCR intermediates after each run. Use
`--keep-intermediates` if you want to retain them for debugging.

`--gpt off` only disables the GPT vision/report stage. If you want a fully local
run with no OpenAI usage at all, use `--transcript subtitles` or
`--transcript whisper`.

## Output layout

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

## Docs

- [Architecture](./docs/architecture.md)
- [Skill](./codex-skills/youtube-analysis/SKILL.md)
