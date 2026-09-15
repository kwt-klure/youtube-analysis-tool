# Execution Modes

Read this reference when running a concrete `youtube-analysis-tool` lane.

## Resolve The Repo

Resolve once before running any lane or checker. Use this skill's
[`../scripts/resolve_repo.py`](../scripts/resolve_repo.py), not a helper inferred
from the caller's current directory. It uses only the Python standard library
and can run from an installed skill even when cwd is an old worktree.

Set `SKILL_DIR` to the absolute directory containing the `SKILL.md` being used.
Use an available Python 3.10+ interpreter for the resolver; do not assume the
caller's `.venv` is usable. With the usual installed location:

```bash
SKILL_DIR="${CODEX_HOME:-$HOME/.codex}/skills/youtube-analysis"
REPO="$(python3 "$SKILL_DIR/scripts/resolve_repo.py")" || exit 1
cd -- "$REPO" || exit 1
```

For a canonical skill checkout or a nonstandard installation, replace
`SKILL_DIR` with that skill directory's absolute path before the resolver call.

The resolver contract is:

1. Accept `YOUTUBE_ANALYSIS_REPO` or `MIRA_YOUTUBE_ANALYSIS_REPO`. If both are
   set, their expanded, absolute, symlink-resolved paths must match; otherwise
   fail with an explicit conflict error. Relative overrides are relative to the
   caller's cwd. Empty or invalid explicit overrides fail without discovery.
2. With neither override set, inspect cwd and then its parents, nearest first.
   A checkout candidate contains either named wrapper below or
   `codex-skills/youtube-analysis/SKILL.md`. Validate the first candidate only;
   do not skip an incomplete/stale worktree for another parent, search the
   installed skill's ancestors, or guess a house-local path.
3. Require a runnable `.venv/bin/python` (Python 3.10+) and both
   `scripts/youtube_analyze.py` and `scripts/youtube_bundle_check.py`. Run an
   interpreter probe and each wrapper's `--help` from the candidate repo,
   requiring exit zero and usable help output. Each probe has a 15-second
   timeout. These checks do not process media or request network access.
4. On success, stdout contains only the resolved absolute repo path plus a
   newline. On failure, stdout is empty, the exit status is nonzero, and stderr
   explains the failure and override remedy. Stop; do not run a lane after a
   failed resolver or fall back to a different interpreter.

Run **all** commands below from that one returned `REPO`, including the bundle
checker and contact-sheet lane. If tool calls use separate shells, set each
call's working directory to the captured absolute path; do not assume `cd`
persists or resolve again between calls. Intake must use this same resolver,
override contract, and returned path for processing and verification.

When a stale worktree fails, set either override to the intended runnable
checkout and rerun resolution. Ask for that path if it is unknown.

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
  --max-frames 36 \
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
  --max-frames 36 \
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

Run the repo-owned checker first:

```bash
.venv/bin/python scripts/youtube_bundle_check.py \
  '<fresh-output-dir>/output.json' \
  --json
```

Do not use an analysis bundle as evidence until the checker exits zero and all
checks pass:

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

For capped visual runs, also read `visual_sampling` and, in debug mode,
`visuals/selection.json` plus selected images under `visuals/candidates/`
before deciding whether the selected evidence is sufficient.

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
