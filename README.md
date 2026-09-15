# YouTube Analysis Tool

A local-first CLI for turning YouTube videos and local media into a structured
`output.json` bundle with transcript text, timing, visual evidence, and
provenance.

The default path is deliberately inexpensive: subtitles first, local Whisper
only when needed, visuals off, GPT off, and intermediate media cleaned after
the run. Additional evidence layers are opt-in.

## Highlights

- One canonical `output.json` per analysis run
- Manual subtitles, automatic captions, local Whisper, and optional API ASR
- Visual-only and transcript-reuse passes that avoid unnecessary ASR
- Scene and interval keyframes with local OCR and triage
- Optional audio structure and top-comment context
- A separate contact-sheet command for quick visual inspection
- Explicit `completed`, `failed`, and `aborted` run status
- Provenance and interpretation hints for downstream readers
- Optional GPT analysis over selected evidence rather than the full video
- Batch processing and local bundle search

## Requirements

Python 3.10 or later is required.

The local pipeline can use these system tools when their stages are enabled:

- `ffmpeg`
- `ffprobe`
- `yt-dlp`
- `tesseract`
- `whisper`

Install the Python package and YouTube-related dependencies in an isolated
environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[youtube]'
```

The OpenAI Whisper CLI is the default local ASR backend. Apple Silicon users
can optionally install MLX Whisper:

```bash
.venv/bin/python -m pip install -e '.[youtube,mlx-whisper]'
```

MLX is opt-in and uses the same Whisper `base` model family through
`mlx-community/whisper-base-mlx`. Its first run downloads the converted model
from Hugging Face.

Available commands:

- `youtube-analyze`
- `youtube-batch`
- `youtube-library`
- `youtube-contact-sheet`
- `youtube-bundle-check`

All commands support `--version`.

## Quick Start

Analyze a YouTube URL:

```bash
youtube-analyze --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Analyze a local media file:

```bash
youtube-analyze --source /path/to/video.mp4
```

The default run writes:

```text
output/youtube/<title-id-or-stem>/output.json
```

It keeps visuals, audio features, comments, and GPT disabled unless requested.

## Common Modes

### Transcript-First

Use the default mode when spoken content is the primary evidence:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --visuals off
```

Usable subtitle tracks take priority. If none are available, the automatic
strategy runs local Whisper, retries it once after a transient failure, and
only then uses optional OpenAI API transcription when `OPENAI_API_KEY` is set.
The API path uses the installed OpenAI SDK directly and does not depend on a
global Codex skill.

When captions satisfy the requested stages, no video or audio is downloaded.
ASR and audio-feature-only runs request audio; visual or burned-subtitle stages
request video. The frame cap does not bound source download or decoding work.

### MLX Whisper

Select MLX Whisper explicitly on Apple Silicon:

```bash
youtube-analyze \
  --source /path/to/video.mp4 \
  --transcript whisper \
  --local-asr-backend mlx-whisper
```

`openai-whisper` remains the default backend. Explicit MLX selection fails
clearly when unavailable and never silently uses remote ASR.

### Visual Evidence

Enable keyframes, OCR, triage, and retained visual output:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --visuals on \
  --max-frames 36
```

Available density presets are `default`, `medium`, and `dense`, corresponding
to 60, 30, and 15 second interval sampling. `--interval-seconds` overrides the
preset. `--max-frames` is optional; when present, byte-identical candidates are
grouped before an elapsed-time-spaced subset enters OCR and triage. See
[Visuals](#visuals) for the selection and provenance contract. Omitting the flag
preserves uncapped behavior.

### Visual-Only

Skip subtitles and ASR when only the image layer is needed:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --transcript off \
  --visuals on \
  --artifacts debug
```

The output records the transcript as intentionally skipped. Visual transcript
excerpts are `null` rather than silently fabricated.

### Reuse An Existing Transcript

Run a visual second pass without repeating transcript extraction:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --reuse-transcript /path/to/first-pass/output.json \
  --visuals on \
  --keyframes interval \
  --interval-seconds 60 \
  --max-frames 36 \
  --ocr off \
  --max-video-height 720 \
  --audio-features off \
  --comments 0 \
  --artifacts debug \
  --out-dir /path/to/fresh-second-pass
```

`--reuse-transcript` accepts either a raw transcript artifact or an existing
completed analysis bundle containing a top-level `transcript` object. Empty,
failed, skipped, malformed, and known source-mismatched inputs are rejected before
media acquisition. Raw transcripts without source identity remain supported, with
`reuse_source_identity: unverified` in provenance. This uncertainty is preserved
through subsequent reuse. Matching source identity is not verification of wording.
New bundles record `source.resolved_input`; legacy relative local paths whose
original working directory is unknown remain unverified, not falsely matched.

Always use a fresh second-pass output directory. Local source media and reusable
transcripts must be outside that directory; overlapping paths are rejected before
existing output is cleared, preserving the original input.

### Rich Intake

Enable dense visual sampling, audio structure, and up to five top comments:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --intake-profile rich
```

Each layer can still be overridden independently:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --intake-profile rich \
  --visual-density medium \
  --audio-features off \
  --comments 0
```

Rich mode collects more signals; it does not turn those signals into ground
truth.

### Contact Sheet

Create a visual overview without transcript, OCR, triage, or semantic output:

```bash
youtube-contact-sheet \
  --source 'https://youtube.com/shorts/VIDEO_ID' \
  --fps 1 \
  --columns 6 \
  --max-video-height 720
```

This command writes `contact-sheet.jpg` and `contact-sheet.json`. It keeps or
links the source media because local replay is part of this workflow. It does
not create or modify `output.json`.

## Bounded Downloads

Use `--max-video-height` when a high-resolution source would add cost without
improving the evidence:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --max-video-height 720
```

The option adds a height-bounded preferred format selector with a fallback for
sources that do not expose a matching format. When omitted, yt-dlp keeps its
normal selection behavior. Local files are never transcoded merely because
this option is present.

The requested limit and whether it applied are recorded in provenance. Contact
sheet manifests record the same information.

## Output Contract

The current output schema version is `1.0.13`.

Top-level shape:

```json
{
  "output_version": "1.0.13",
  "source": {},
  "metadata": {},
  "transcript": {},
  "visuals": {
    "slides": [],
    "charts": []
  },
  "visual_sampling": {},
  "audio_features": {},
  "comments": {},
  "run_reflection": {},
  "processing": {
    "run_status": "completed"
  },
  "provenance": {},
  "errors": []
}
```

### Run Status

An `output.json` file may be written after a failure or interruption so partial
evidence can be inspected. Consumers should require all of the following before
treating a bundle as complete:

1. The command exited with status 0.
2. `output.json` parses successfully.
3. `processing.run_status` is `completed`.
4. `errors` contains no fatal entry.

Possible run states are:

- `completed`: required stages and cleanup finished
- `failed`: a processing or cleanup stage raised an error
- `aborted`: the run was interrupted, such as with `Ctrl-C`

Verify an existing bundle with the same completion contract used by automated
workers and skills:

```bash
youtube-bundle-check /path/to/output.json --json
```

The checker exits nonzero for malformed bundles, missing run status, failed or
aborted runs, and fatal errors.

Batch processing uses this same checker. Older bundles without run status are
rerun rather than accepted as completed. Malformed cached bundles do not stop
other queued sources.

### Transcript

The transcript payload includes full text, timestamped segments, language,
source, provenance, and optional interpretation guidance. Extraction sources
include:

- manual subtitle tracks
- automatic subtitle tracks
- burned-subtitle OCR when explicitly enabled
- local Whisper
- optional remote ASR
- reused or intentionally skipped transcript state

Local Whisper provenance records the selected backend and model. Interpretation
hints identify text that should be checked carefully for names, numbers, or
exact wording.

### Visuals

Retained slides and charts include timing, OCR summary, nearby transcript text,
an embedded primary image, and selection provenance. Local labels are routing
heuristics rather than semantic guarantees.

`visual_sampling` reports raw candidate, pre-OCR deduplicated, selected, and
retained visual counts. Debug runs also preserve `visuals/selection.json` so a
frame can be traced as selected, duplicate, or over budget. Capped debug runs
keep selected candidate images under `visuals/candidates/` even when OCR is off
and local triage does not promote them as slides or charts.

Destructive duplicate filtering now requires byte-identical frame content, not
coarse pHash similarity alone. Similar-looking slides with changed text/numbers
remain separate. Capped selection uses elapsed-time targets across surviving
representatives; pHash is retained only as a diagnostic signal. This conservative
policy can retain more near-duplicate frames than previous versions.

### Run Reflection

`run_reflection` is deterministic, local evidence routing. It reports available
layers, limitations, uncertainties, and possible next-run adjustments. It does
not call GPT and does not modify the tool or its skills automatically.

### Audio And Comments

Audio output contains structural signals such as loudness windows, silence
segments, and within-video changes. Comments are optional top-comment context.
Neither layer is treated as semantic truth or representative audience sampling.

## GPT Mode

GPT is off by default. Enable it explicitly:

```bash
youtube-analyze \
  --source 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --visuals on \
  --gpt on
```

The GPT path sends approved segments and selected representative frames, then
uses the resulting segment analyses with transcript and metadata for final
synthesis. It does not upload the entire video or every extracted frame.

Set an API key through the environment or a local `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

The `.env` file is ignored by Git. OpenAI is optional; the core bundle can be
consumed by any downstream system.

## Artifact Lifecycle

Minimal mode is the default:

```bash
youtube-analyze --source /path/to/video.mp4 --artifacts minimal
```

It keeps the canonical bundle and removes stage-level artifacts and downloaded
media after the run.

Debug mode preserves inspectable stage output:

```bash
youtube-analyze --source /path/to/video.mp4 --artifacts debug
```

Use `--keep-intermediates` only when downloaded media or normalized audio must
remain on disk.

`processing.cleanup_requested` records the requested policy; `cleanup_applied`
is true only when requested cleanup completed without errors. Deletion failures
produce a failed bundle with a cleanup error rather than a false success. Inspect
and resolve any reported residual artifacts before treating the run as cleaned.

## Batch And Library Commands

Run a newline-separated source list:

```bash
youtube-batch --source-list /path/to/sources.txt
```

Batch mode skips completed bundles, reruns failed or aborted partial bundles,
continues past per-item failures, and writes a batch report.

Search local bundles:

```bash
youtube-library --grep 'prompt engineering'
```

Filters are available for transcript source, language, trust, read mode,
channel, and error presence.

## Privacy And Cost Defaults

The project defaults to local processing and minimal retention:

- GPT is off.
- Visual extraction is off.
- Audio features and comments are off.
- Subtitle tracks are preferred over ASR.
- Downloaded media and normalized audio are cleaned after analysis.
- `.env`, local output, private operation logs, and local agent workspace files
  are ignored by Git.

Review debug artifacts before sharing them. They may contain source frames,
OCR text, transcript excerpts, comments, or local paths.

## Known Limits

- OCR is a routing aid and may be noisy, especially for formulas and dense CJK
  text.
- Targeted timestamp or time-window extraction is not currently exposed as a
  CLI option.
- Visual labels are heuristic.
- Top comments are not representative sampling.
- Audio features are structural signals, not interpretation.
- GPT support is currently OpenAI-specific when enabled.
- The tool does not provide a polished end-user summary UI.

## Development

For a versioned macOS Apple Silicon / Python 3.11 recovery baseline, see
[Environment Recovery](docs/environment-recovery.md). Validate a new environment
before switching a working checkout to it; normal installation stays unchanged.

Run the unit test suite:

```bash
python -m unittest discover
```

Check patch whitespace before committing:

```bash
git diff --check
```

Check whether the repo-owned processing skill matches its installed runtime
copy without writing anything:

```bash
youtube-skill-sync --check --json
```

Install it explicitly with a verified runtime backup:

```bash
youtube-skill-sync --install --json
```

Installation serializes cooperating installers, checks the expected runtime
state immediately before replacement, and verifies the installed hash. Source,
target and backup paths must not overlap or contain symlinks (including ancestor
aliases); use physical paths. Backups are protected by owner-only directories.
The result includes `backup_path` and `preserved_previous_path`; the moved tree
is intentionally retained in case another process still holds it open. This is
not a lock against arbitrary external editors. Avoid editing the runtime during
deployment; archive old preimages only after checking for such writers.

The repository also includes a Codex-compatible processing skill under
[`codex-skills/youtube-analysis`](codex-skills/youtube-analysis/).

Additional implementation details are documented in
[`docs/architecture.md`](docs/architecture.md).
The [reliability repair plan](docs/reliability-plan.md) records this change set's
scope and acceptance criteria.
