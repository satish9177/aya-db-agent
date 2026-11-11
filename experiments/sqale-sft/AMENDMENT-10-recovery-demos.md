# Amendment 10: recovery and step-by-step demos (data build v4) and one A1500-v4 run (2026-11-09)

Written after the A1500 hints re-screen (`experiments/sql-a1500-hints/RESULTS.md`) and before any run on
the new data. The user chose this option ("option B") and approved building the new training data
locally. **No upload, job or push is approved by this amendment.** The registration needs the user's
explicit approval of its SHA256, uploads and storage before launch.

## Why

A1500 never recovers from a failed query, and better error messages did not change that. With hints it
re-sent the identical query 20 of 25 times. After re-checking the table it wrote a new query only once
in 17. The 1,500 training demos explain it:

- **Repairs skip the correction step.** All 377 repairs jump from the error straight to the correct query,
  and none re-checks a table first.
- **The mistakes look nothing like the model's.** They are artificial quoted names (`"lane_number_code"`),
  wrong table names, ambiguity and aggregate misuse. A1500's real errors are plausible names: typos,
  naming-style slips, a name from another table, or a real column on the wrong alias.
- **Complex queries are written in one shot.** 483 of 485 complex demos go straight to the full query.

## What changes: data build v4 (same questions, different demonstrated paths)

`scripts/build_sqale_v4_recovery.py` writes `data/sqale/data-build-v4/train-1500.episodes.jsonl` from the
frozen v3 `train-1500` (SHA `0c920dab…`). It keeps the same 1,500 rows in the same order with the same
questions, databases and exploration calls, so the seed-42 split is identical (1,425 / 75, checked). There
are no model calls, and no dev or final data is used: the training schemas share nothing with dev.

| Group | v4 path | Rows |
|---|---|---:|
| Repair rows | realistic column mistake (`src/ayadb/sqlmutate_v4.py`) as **untrained context** → native `no such column` error → `describe_table` on the table that owns the real column → correct query → submit | 349 |
| Repair rows that would exceed 8 tool steps | the same without the re-check | 24 |
| Repair rows with no realistic mistake | v3 row kept | 4 |
| Complex gold rows (3+ tables or multi-step) | run the first CTE body or first uncorrelated subquery on its own (it must run and return rows) → full query → submit | 190 |
| Other complex gold rows | v3 row kept | 155 |
| Simple gold rows and the 235 shop rows | unchanged | 778 |

The realistic mistakes come in four kinds, chosen deterministically per episode and kept only if SQLite
answers `no such column`: 110 typos, 120 naming-style slips (snake/camel case, `_id`), 76 names taken
from another table, and 67 real columns on the wrong table alias. The mistaken query is never a training
target, as in v3. Every changed episode was executed, re-encoded without truncation (maximum 3,975
tokens, limit 4,096) and replayed through AgentTune's real parser and renderer. The build manifest
records the output SHA `569e8aed…`.

This is new training data derived from the same SQaLe training questions. Two changes are made at once
(recovery and step-by-step), so a difference cannot be attributed to one of them alone.

## What runs: `a1500v4-sql-agenttune-20261109-01` (stage `train`)

- Train from the untuned base model on the v4 file with A1500's settings unchanged: AgentTune audited
  factory, NF4/fp16 LoRA r16/a32/dropout .05, `trained-positions-v1`, LR 2e-4 cosine, accumulation 16,
  two passes, **180 updates**, checkpoints every 10 (keep 3), validation every 50.
- Then screen only the final adapter on the same 95 dev tasks, verifying it reloads the exact
  step-180 tensors first. The tools, prompts, step limit, decoding and scorer are the **original
  ones**: the error-feedback tools were not adopted (amendment 9).
- Comparators: A1500 (37/95, original tools) and A1500 with hints (38/95), pinned by metrics hash and
  not rerun.

## Decision rule (fixed now)

- **Finalist choice:** A1500-v4 replaces A1500 as the trained model carried to the final evaluation if
  its dev strict score is higher than A1500's 37/95. A tie keeps A1500. This is a selection between two
  models of the same size, data questions and cost, so no extra margin is required. The final evaluation
  on unseen data is the confirmation.
- This run does not decide the full stage, Q2, A+/B or the final evaluation; each needs its own approval.

## Reported diagnostics (descriptive)

- Strict, fallback and set; paired discordant tasks against A1500 (exact McNemar, unadjusted).
- Amendment-8 shapes and the 30 complex tasks.
- **Recoveries:** episodes with a failed `run_sql` that end strict (A1500: 0 of 24). Also what follows
  a failed query: the identical query (A1500: 76 of 80), `describe_table`, or a changed query.
- Ran out of steps; episodes that run a partial query before the full one.
- Training and validation loss, time, cost and GPU memory. Amendment 5's manual audit is repeated if
  A1500-v4 becomes the finalist.

**Disclosure:** the kinds of mistakes and the re-check step were chosen after reading A1500's dev
failures. The training data contains no dev or final questions or databases, but a dev gain may still be
optimistic. The final evaluation (SQaLe final on unseen databases, BIRD Mini-Dev from another source) is
the real test.

## Money and time

- One t4-small job, rate $0.006667/min, **9.5-hour hard timeout** (training step 7 h, dev 3.5 h): $3.80
  compute ceiling plus the $0.04 first-month storage reserve, **$4.00 cap**.
- Forecast: 2 x 2,798,156 input tokens at A1500's measured 368 tokens/s is about 4.2 h of training, plus
  about 1.8 h of dev screening: **about 6.1 h, ~$2.45**.
- Private upload: the v4 training file, dev tasks, the training and dev databases and the tokenizer
  (about 0.24 GB). Outputs are about 0.7 GB (three checkpoints with optimizer state, adapter, logs).
  Storage at most 30 days without renewed approval.

## Unchanged

Report all results as they are, including negatives. No automatic retry, hardware change, tool,
step-limit, prompt or decoding change. Q2, A+/B, the full stage, BIRD/final evaluation, teacher models,
push and publication remain unapproved. Shop retention stays deferred to the final evaluation. The
sealed shop test stays closed.
