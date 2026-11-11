# A1500-v4: train on recovery and step-by-step demos, then screen on the 95 dev tasks (awaiting approval)

Registration **`a1500v4-sql-agenttune-20261109-01`**, machine file `REGISTRATION.json` (SHA256 is printed by
`scripts/sql_job.py plan`; the approval must name it). Built by `scripts/prepare_sql_a1500v4.py` under
[amendment 10](../sqale-sft/AMENDMENT-10-recovery-demos.md). **No upload or launch is approved or performed.**

## What runs

| Step | What | Step limit |
|---|---|---:|
| train-a1500v4 | Train from the untuned base on `data-build-v4/train-1500` (A1500's 1,500 questions and split, 1,425 / 75), two passes, **180 updates** | 7 h |
| dev-a1500v4-nf4 | Reload exactly the step-180 adapter, then screen it on the same 95 dev tasks with the original tools | 3.5 h |

All settings match A1500. Only the training file differs:
- **Repairs:** 349 now show a realistic column mistake (untrained), the real `no such column` error, a
  re-check of the right table, then the fix. 24 more show the mistake and the fix without the re-check.
- **Complex questions:** 190 now run the first part of the query on its own before the full query.

Everything else is unchanged from v3. All 1,500 training and 95 dev references pass locally before any
model use, with zero model calls. All rows re-encode to the stored counts: 2,798,156 inputs and
347,808 targets per pass.

## Money, time and storage

- One t4-small job on hf-user, rate $0.006667/min, **9.5-hour hard timeout**: $3.80 compute ceiling
  plus a $0.04 first-month storage reserve, **$4.00 cap**.
- Expected from A1500's measured speed: training ~4.2 h, dev ~1.8 h, **about 6.0 h, ~$2.42**.
- Private upload: **1,211 files, 234,112,064 bytes** (v4 training file, dev tasks, 1,204 SQLite databases,
  tokenizer). Outputs about 0.7 GB. Storage at most 30 days without renewed approval.

## Pass, abort and decision

The pass and abort conditions are the same as A1500's. **No automatic retry or second job.**

**Decision:** A1500-v4 replaces A1500 as the trained finalist for the final evaluation if it scores
**at least 38/95** strict on dev (above A1500's 37); a tie keeps A1500. Reported regardless: recoveries
after a failed query (A1500: 0 of 24), what follows a failed query, complex tasks, running out of steps,
partial-query use, loss, time and cost.

Excluded: reruns of A1500/hints/Base/Pilot/A500, the error-feedback tools, intermediate checkpoint
screening, shop retention, the full stage, Q2, A+/B, BIRD/final, teacher/API, vLLM, hardware or protocol
changes, push and publication.
