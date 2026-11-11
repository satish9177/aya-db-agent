# Amendment 7: A500 stage registrations and launcher sync fix (2026-11-07)

Written after G0-02 passed and before any A500 run. The user approved, as free work: fixing the
launcher's output-sync filter, building exact runnable 500-stage registrations, running the Base and
Pilot dev screens as a parallel second job, and leaving shop retention to the final evaluation.
**No upload, job or push is approved by this amendment.** Each registration needs the user's explicit
approval of its SHA256, uploads and storage before launch.

## Launcher sync fix

G0-02 lost `checkpoint-reload.json` because the snapshot ignore rule skipped every name starting with
`checkpoint-` that lacked a `_COMPLETE` child, files included. The rule now skips only *folders* named
`checkpoint-*` without `_COMPLETE`, plus `.tmp` files. A test syncs a plain `checkpoint-reload.json`
and still drops an incomplete checkpoint folder; it fails on the old rule. G0-01/G0-02 registrations
and results are unchanged.

## Two registrations, same hardware, engine and precision

1. **`a500-sql-agenttune-20261107-01`** (stage `train`): train A500 through the AgentTune audited factory
   on `data/sqale/data-build-v3/train-500.episodes.jsonl` with the run policy, seed-42 split 475/25,
   two passes, microbatch 1, accumulation 16, LR 2e-4, cosine, NF4/fp16 LoRA r16/a32/dropout .05,
   `trained-positions-v1`, checkpoints every 10 updates (keep 3), validation every 50 updates. HF Trainer
   gives 30 updates per pass, **60 in total**. Then screen the final adapter on the 95 effective dev tasks,
   first verifying that it reloads exactly the step-60 adapter tensors.
2. **`dev-baselines-sql-20261107-01`** (stage `evaluate`): untuned Base, then the existing Pilot adapter
   (`6f2827ae...`), on the same 95 dev tasks.

Both: t4-small, greedy, eight tool steps, 512 new tokens per turn, 8,192 context, `sql-submit-v1`,
table-read-v1 strict scorer with fallback and set scores, every-submission no-table counts, AuditKIT
agreement, one projected inference position, purpose `dev`. Runtime reference checks run on every
training row and dev task before any model call.

## Deviations from the preregistration, recorded before running

- **Retention** (shop dev 40 / mh 40) is not run with this screen. The preregistration says it is
  reported alongside each checkpoint screen; the user chose to run it once for Base, Pilot and the
  selected finalists at the final evaluation. A500 selection decisions here use SQaLe dev only.
- **Base one-example prompt.** The preregistration lists a fixed one-example prompt for Base on dev.
  No such prompt is implemented; this screen uses the zero-shot Base, as in G0 and the 500 draft. The
  fallback score already exposes format-learning effects for G1. A one-example Base can be added later
  as its own small registration if the user wants it; it is not silently dropped.

## Decisions after the runs (unchanged rules)

- G1: A500 beats Base by at least 10 percentage points on both strict and fallback dev accuracy.
- Proposed stage rule (separate approval): train 1,500 only if it beats 500 by at least five
  unrounded strict points on the same 95 tasks.
- Amendment 5's manual audit (20 failures, 10 successes per model, seed 42) follows this first full
  screen. No score changes.
- Report all results as they are, including negatives. No automatic retry, hardware change, Q2, BIRD,
  final evaluation, teacher model, push or publication.
