# A500: train on 500 examples, then dev screen (awaiting approval)

Registration **`a500-sql-agenttune-20261107-01`**, machine file `REGISTRATION.json` (SHA256 is printed by
`scripts/sql_job.py plan`; the approval must name it). Built by `scripts/prepare_sql_a500.py` under
[amendment 7](../sqale-sft/AMENDMENT-7-a500-stage.md). **No upload or launch is approved or performed.**
It runs in parallel with [`dev-baselines-sql-20261107-01`](../sql-dev-baselines/REGISTRATION.md).

## What runs

| Step | What | Step limit |
|---|---|---:|
| train-a500 | AgentTune audited factory on the frozen 500 examples: policy, seed-42 split 475 train / 25 validation, 2 passes, microbatch 1 x accumulation 16 (**60 updates**), LR 2e-4 cosine, NF4/fp16 LoRA r16/a32/.05, `trained-positions-v1`, checkpoint every 10 updates (keep 3), validation every 50 | 4 h |
| dev-a500-nf4 | Reload the final adapter, check its tensors equal the saved step-60 adapter, then 95 effective dev tasks: greedy, 8 tool steps, 512 new tokens/turn, one projected position, table-read-v1 strict plus fallback/set scores, AuditKIT agreement | 3.5 h |

Same model, tokenizer, precision, hardware and evaluation settings as G0-02 and the baselines job.
Before any model call the job re-runs all 595 reference queries (500 training rows, 95 dev tasks);
they passed locally with zero model calls. All 500 rows re-encode locally to their frozen counts:
847,065 training inputs and 107,381 unique targets per pass; longest training input 3,849 tokens
(G0-02 trained a 4,072-token example at 8.91 GiB of 14.74 GiB).

## Money, time and storage

- One t4-small job on hf-user, rate $0.006667/min (rechecked by preflight), **7-hour hard timeout**.
  Compute ceiling $2.80 plus a $0.04 first-month storage reserve: **$3.00 cap**.
- Expected from G0-02's measurements: training ~1.8 h (~$0.71), dev screen 0.3-2.2 h ($0.13-0.88),
  so about **2.1-4.0 h, $0.84-1.59**. Rough, ±50%: speed comes from one long example, dev time from
  three tasks per model.
- Private upload: **367 files, 82,449,877 bytes** (500 training episodes, dev tasks, 360 SQLite
  databases, tokenizer). Outputs up to 1 GiB (expected ~0.7 GB: three checkpoints with optimizer state,
  final adapter, dev logs). Storage at most 30 days without renewed approval.

## Pass, abort and reporting

Pass means the run is valid, not that A500 is good: references pass before model use, fresh encoding
matches the frozen plan, 60 finite updates with none skipped, finite validation losses, exact final
adapter reload (record now synced after the launcher fix), 95 dev episodes with one finite projected
position, AuditKIT agrees with every score, memory events saved.

Abort on wrong account or higher price, CUDA failure/OOM/nonfinite values, any input/mask/plan/
checkpoint/reload mismatch, sync failure or byte cap, timeout or stall (30 minutes without output),
missing memory record, full-prompt inference projection or AuditKIT disagreement. **No automatic
retry or second job.** Accuracy results are reported whatever they are. G1 (A500 beats Base by 10
points on strict and fallback) and the 1,500 stage rule are computed only after both jobs finish,
followed by amendment 5's manual audit. Excluded: shop retention (deferred to the final evaluation),
a one-example Base prompt (not implemented), 1,500/full stages, Q2, A+/B, BIRD/final, teacher/API,
vLLM, hardware change, push and publication.
