# Amendment 8: A1500 stage registration and query-shape diagnostic (2026-11-08)

Written after the approved 500-stage jobs finished (A500 28/95 strict, Base 2/95, Pilot 20/95; G1
passed; `experiments/sql-a500/RESULTS.md`) and before any 1,500-stage run. The user approved building
the exact 1,500 registration as free work. **No upload, job or push is approved by this amendment.**
The registration needs the user's explicit approval of its SHA256, uploads and storage before launch.

## What runs: `a1500-sql-agenttune-20261108-01` (stage `train`)

Everything matches A500 except the data size and the derived counts:

- Train A1500 **from the untuned base model** (not a continuation of A500, as in the nested-stage design)
  on `data/sqale/data-build-v3/train-1500.episodes.jsonl`. It contains all 500 A500 rows. Apply the
  same run policy and seed-42 5% split as the frozen runner input `data/sqale/runner-inputs-v1/a-1500.json`:
  1,425 train / 75 validation, 2,684,785 inputs and 330,742 targets per pass.
- Same settings: two passes, microbatch 1, accumulation 16, LR 2e-4 cosine, NF4/fp16 LoRA
  r16/a32/dropout .05, `trained-positions-v1`, checkpoints every 10 updates (keep 3), validation every
  50 updates. HF Trainer gives 90 updates per pass, **180 in total**.
- Then screen **only the final adapter** on the same 95 effective dev tasks, first verifying that it
  reloads exactly the step-180 tensors. There is no checkpoint selection on dev.
- Same t4-small, hf-user, greedy decoding, eight tool steps, 512 new tokens per turn, 8,192 context,
  `sql-submit-v1`, table-read-v1 strict plus fallback and set scores, AuditKIT agreement, and reference
  checks on every training row and dev task before any model call.
- Comparators are the already-measured Base, Pilot and A500 dev results (same tasks, hardware, pinned
  environment and protocol). They are **not rerun**.

## Decisions after the run (rules fixed now)

- **Stage rule** (preregistration, unchanged): stop data scaling at 1,500 unless A1500 beats A500 by at
  least five unrounded strict points on the same 95 tasks, i.e. **A1500 needs at least 33/95**. Passing
  only permits proposing the full stage as its own registration and approval.
- Choosing finalists for the final evaluation (and shop retention, deferred there by amendment 7) is not
  decided by this run. Finalists are proposed after this result and before any final evaluation.
- Amendment 5's manual audit is repeated for A1500 (seed 42, 20 failures / 10 successes, labels change
  no scores).

## New secondary diagnostic: results by query shape

`src/ayadb/queryshape.py` classifies each dev task by its reference SQL (sqlglot, SQLite dialect) into
five classes: 1-table lookup, 1-table aggregate, two tables, 3+ tables, multi-step (CTE or more than one
SELECT). "Complex" means 3+ tables or multi-step. On the 95 effective dev tasks the counts are frozen by
a test: 37 / 5 / 23 / 10 / 20, so **30 complex**.

For every model the report gives strict successes per class. It also gives "ran out of the eight tool
steps" counts per class, and, for A1500 against A500 on the 30 complex tasks, the discordant pairs.

**Disclosure:** these classes were defined after reading the A500 and baseline dev results, when the
user asked whether the gains came from simple lookups or complex queries. They are frozen here before
any A1500 result. The diagnostic is descriptive only: it is not a gate, score, reward or selection
rule. With 30 complex tasks, a difference of two or three questions can be chance.

For reference, the SQaLe training rows contain 104 complex examples at 500, 485 at 1,500 and 1,736 at
the full stage. The 235 converted shop rows are the same in every stage.

## Money and time

- One t4-small job, rate $0.006667/min, **9.5-hour hard timeout**: $3.80 compute ceiling plus the
  $0.04 first-month storage reserve, **$4.00 cap**. Step limits: training 7 h, dev screen 3.5 h.
- Forecast from A500's measured speed (364 input tokens/s; dev 62 s/episode): training ~4.1 h, dev
  screen ~1.6-2.5 h, so **about 6 h and $2.40**. At G0-02's slower 266 tokens/s, training is ~5.6 h.
- Outputs are like A500's: about 0.7 GB (three checkpoints with optimizer state, adapter, logs) within the 1 GiB cap.
  Storage is at most 30 days without renewed approval.

## Unchanged

Report all results as they are, including negatives. No automatic retry, hardware change, step-limit
or prompt change, Q2, A+/B, BIRD/final evaluation, teacher model, push or publication. Shop retention
stays deferred to the final evaluation. The sealed shop test stays closed.
