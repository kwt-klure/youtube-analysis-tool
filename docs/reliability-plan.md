# Worker and skill reliability plan

Date: 2026-09-15

## Scope

Repair evidence-preservation, completion, bounded-selection and skill-deployment
failure cases while preserving the default transcript-first interface. Keep
visuals optional. No new remote inference, OCR quality project or automatic
scheduled work is included.

## Stages and acceptance

1. **Inputs and completion (A1, A2, A4, A11).** Reject overlapping input/output
   paths before clearing artifacts; validate reused transcript schema, bundle
   completion and known source identity before acquiring media. Raw transcripts
   without identity remain supported with explicit unverified provenance.
   Surface deletion failures and actual cleanup results. Use the common batch
   completion gate and isolate malformed cached bundles per item.
   Acceptance: original input survives rejected overlap; empty/failed/mismatched
   reuse cannot become successful evidence; injected unlink failures produce a
   failed bundle; malformed or legacy cache cannot abort or falsely skip a batch.
2. **Visual evidence (A3, A5).** Use conservative content equality for destructive
   duplicate elimination instead of coarse pHash alone. Select remaining frames
   by elapsed-time targets, preserving boundary coverage when the budget allows.
   Acceptance: same-layout changed text/numbers remain distinct, exact repeated
   frames deduplicate, skewed-density candidates cover the available timeline,
   and OCR still receives no more than the requested frame budget.
3. **Evidence-aware acquisition (A6).** Avoid full media downloads when usable
   subtitles satisfy the requested stages. Fetch audio for ASR/audio-only work,
   and video for requested visual/burned-subtitle work. Preserve standalone
   media-download behavior and no-fresh-ASR reuse/off semantics.
   Acceptance: usable captions finish without a media fetch; missing captions
   select the necessary media lane; requested visual stages still get video.
4. **Skills and deployment (A7-A10).** Serialize installation; verify expected
   target state immediately before replacement; enforce a consistent symlink
   policy and private backup permissions. Reject empty/missing trees in sync
   checks. Verify runnable interpreter, wrapper and checker during skill repo
   resolution and reconcile the supported override variables.
   Acceptance: concurrent modification is retained/rejected, unsupported links
   cannot copy outside content, empty/missing trees fail the check, and stale
   worktrees cannot be presented as runnable lanes.

## Verification and delivery

- Add focused regressions for each stage before its fix, then run the complete
  local suite. Use deterministic synthetic media for end-to-end smoke tests.
- Inspect public documentation, staged diff, ignored private notes, paths and
  secrets before publishing. Do not force-push or merge the default branch.
- Keep implementation commits grouped by behavior/rollback boundary. Record
  results and commit identifiers in the local operation note.
- Deploy the processing skill only after helper verification, preserving its
  private preimage and checking the final hash. Intake reference changes use the
  same expected-hash/preimage/readback discipline.

## Tracking

- [mira-loop:open] [owner:youtube-analysis-tool] Stage 1 inputs and completion - next: add regression cases and repair the boundaries.
- [mira-loop:open] [owner:youtube-analysis-tool] Stage 2 visual evidence - next: test content equality and time-based selection before changing the filter.
- [mira-loop:open] [owner:youtube-analysis-tool] Stage 3 evidence-aware acquisition - next: add caption-only and audio/video lane tests.
- [mira-loop:open] [owner:youtube-analysis-tool] Stage 4 skills and deployment - next: harden and verify sync plus repo selection, then deploy the verified skill.
- [mira-loop:open] [owner:youtube-analysis-tool] Verification and delivery - next: run integrated tests, inspect privacy, commit and push the feature branch.
