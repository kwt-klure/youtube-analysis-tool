---
name: youtube-analysis
description: Turn a YouTube URL or local video/audio file into a transcript, bounded keyframes, OCR artifacts, reused-transcript visual evidence, or a contact sheet with local-first processing. Use for transcript extraction, visual-only inspection, transcript-aligned second passes, screen or slide evidence, share sheets, and debugging the youtube-analysis-tool pipeline while minimizing GPT use.
---

# YouTube Analysis

Use this skill for concrete media processing and evidence extraction. Keep
discussion intake, publishing, and archival decisions outside this processing
workflow.

## Workflow

1. Determine the source type and requested evidence.
2. Read [references/execution-modes.md](references/execution-modes.md) before
   running a concrete lane.
3. Select the narrowest lane that can answer the request.
4. Run local extraction before considering GPT.
5. Verify process exit and run the repo-owned bundle checker before using an
   analysis bundle as evidence.
6. Read the relevant provenance and evidence-sufficiency fields.
7. Clean temporary outputs according to the selected lane.

## Lane Selection

- Use transcript-first for ordinary semantic intake.
- Use visual-only when transcript extraction would be unnecessary or too slow.
- Reuse an existing transcript for a second visual pass instead of rerunning
  subtitles or ASR.
- Use contact-sheet mode for short visual/share requests that do not require
  semantic analysis.
- Use full rich intake only when multiple visual, audio, or comment layers are
  independently material.

Start bounded visual evidence passes with interval-only frames, a conservative
interval, a 36-frame pre-OCR cap, 720p media, and OCR off. Enable OCR only when
selected frames actually need text extraction.

## Success Gate

Treat `output.json` as completed evidence only when all are true:

- the wrapper exited with code 0;
- the JSON parses;
- `processing.run_status` is `completed`;
- no fatal entry exists in `errors`.

Treat `failed` and `aborted` bundles as diagnostic partial output. Never infer
success merely because `output.json` exists.

Use the machine-readable gate after each analysis run:

```bash
.venv/bin/python scripts/youtube_bundle_check.py \
  '<fresh-output-dir>/output.json' \
  --json
```

The checker must exit zero and report `"valid": true`.

## Local-First Boundary

Use local subtitles, Whisper, ffmpeg, OCR, triage, and provenance before remote
reasoning. Use GPT only for selected high-value frames or synthesis that local
evidence cannot supply. Do not send the full video or bulk frame set to GPT.

## Ownership Boundary

- `youtube-analysis-tool` is executable truth.
- This repo-owned skill is the canonical processing workflow.
- The installed runtime skill is a deployed copy and should match this folder.
- Use `youtube-skill-sync --check --json` to detect drift; installation is an
  explicit, backed-up action.
- `youtube-intake` owns conversational evidence escalation and optional
  archival routing.
