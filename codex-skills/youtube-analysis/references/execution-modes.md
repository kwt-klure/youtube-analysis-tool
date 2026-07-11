# Execution Modes

Read this reference when running a concrete `youtube-analysis-tool` lane.

## Resolve The Repo

Resolve the repo in this order:

1. `YOUTUBE_ANALYSIS_REPO`
2. The current Git checkout when it contains `scripts/youtube_analyze.py`
3. A parent directory containing this project's `pyproject.toml`

Stop and ask for the checkout path if no repo resolves. Run commands from the
resolved repo and prefer its `.venv` plus `scripts/youtube_analyze.py`.

Use a fresh temporary output directory unless the user requests durable output.
Do not reuse a previous run directory as a new success surface.

## Transcript-First

```bash
.venv/bin/python scripts/youtube_analyze.py \
  --source '<youtube-url-or-local-file>' \
  --visuals off \
  --artifacts minimal \
  --out-dir '<fresh-temp-dir>'
```

Use this for ordinary semantic intake. Subtitle-backed runs should remain
cheap; no-subtitle runs may fall back to Whisper and take longer.

## Visual-Only

```bash
.venv/bin/python scripts/youtube_analyze.py \
  --source '<youtube-url-or-local-file>' \
  --transcript off \
  --visuals on \
  --keyframes interval \
  --interval-seconds 60 \
  --ocr off \
  --max-video-height 720 \
  --artifacts debug \
  --out-dir '<fresh-temp-dir>'
```

Use this when the goal is frame inspection and transcript extraction would add
cost without helping.

## Reuse-Transcript Visual Pass

The reuse input may be a raw transcript artifact or the first pass
`output.json`.

```bash
.venv/bin/python scripts/youtube_analyze.py \
  --source '<same-source>' \
  --reuse-transcript '<first-pass-output.json>' \
  --visuals on \
  --keyframes interval \
  --interval-seconds 60 \
  --ocr off \
  --max-video-height 720 \
  --audio-features off \
  --comments 0 \
  --artifacts debug \
  --out-dir '<fresh-second-pass-dir>'
```

Inspect retained frames before enabling OCR. Rerun with OCR only when the
selected screens contain text that must be extracted.

## Contact Sheet

```bash
PYTHONPATH=src .venv/bin/python -m youtube_analysis_tool.contact_sheet \
  --source '<youtube-url-or-local-video>' \
  --max-video-height 720 \
  --out-dir '<fresh-temp-dir>'
```

Use this for visual sharing or quick short-video inspection. It writes
`contact-sheet.jpg` and `contact-sheet.json`, keeps or links media by design,
and does not create canonical semantic `output.json` evidence.

## Exceptional Rich Intake

```bash
.venv/bin/python scripts/youtube_analyze.py \
  --source '<youtube-url>' \
  --intake-profile rich \
  --max-video-height 720 \
  --artifacts minimal \
  --out-dir '<fresh-temp-dir>'
```

Use rich intake only when more than one rich evidence layer is material. Turn
off comments, audio features, or visuals that are not independently useful.

## Verify Analysis Output

Do not use an analysis bundle as evidence until all checks pass:

1. Capture and require wrapper exit code 0.
2. Parse `output.json` as JSON.
3. Require `processing.run_status = completed`.
4. Inspect `errors`; treat fatal or interruption entries as incomplete.
5. Read:
   - `metadata`
   - `transcript.provenance`
   - `transcript.interpretation`
   - `processing`
   - full `provenance`
   - `run_reflection`
   - the evidence layers used for the answer

An `output.json` written by a nonzero or interrupted run is diagnostic partial
output even when some fields look clean.

## Cleanup

- Let normal analysis cleanup remove downloaded media unless
  `--keep-intermediates` is explicitly required.
- Keep a successful temporary run only until its evidence has been read and the
  answer is complete, unless the user requests durable artifacts.
- Keep a failed or aborted run only long enough to inspect the failure. Then
  remove it intentionally or report the retained path and reason.
- Keep contact-sheet media until the share/inspection task is complete; remove
  the whole temporary directory afterward unless durability was requested.
