# A1500 with error-feedback tools: one dev re-screen (awaiting approval)

Registration **`a1500-hints-sql-20261108-01`**, machine file `REGISTRATION.json` (SHA256 is printed by
`scripts/sql_job.py plan`; the approval must name it). Built by `scripts/prepare_sql_a1500_hints.py` under
[amendment 9](../sqale-sft/AMENDMENT-9-error-feedback-tools.md). **No upload or launch is approved or performed.**

## What runs

| Step | What | Step limit |
|---|---|---:|
| dev-a1500-hints-nf4 | Reload exactly A1500's step-180 adapter (pulled copy, SHA `69f0f125…`), then screen it on the same 95 dev tasks with the `column-hints-v1` tools | 4 h |

No training. Everything except the tool feedback matches A1500's dev step:
- greedy decoding, seed 42, 8 tool steps, 512 new tokens per turn, 8,192 context, one projected position;
- `sql-submit-v1`, table-read-v1 strict plus fallback and set, AuditKIT agreement.

All 95 dev references pass locally before any model use, with zero model calls. The comparator is A1500 with
the original tools (37/95 strict, 1/30 complex), pinned by its metrics hash and not rerun.

## The new tools

Same four tools with the same names, arguments and descriptions (identical AgentTune schemas, tested).
Two `run_sql` observations change:
1. `no such column` errors list the real columns of the tables in the query, plus `Column X is in: …`
   when the name exists elsewhere. `no such table` errors list the available tables.
2. A query identical to an earlier failed one is not run again; the tool says so and asks for a change.

Why: A1500 never recovered from a failed query (0 of 24 episodes) and repeated the identical query 76 of
80 times. Replaying its 104 failed calls offline, 77 would be refused and 26 of 27 distinct errors get a
hint. Hints are 233 characters median, 518 at most.

## Money, time and storage

- One t4-small job on hf-user, rate $0.006667/min (rechecked by preflight), **4.5-hour hard timeout**.
  Compute ceiling $1.80 plus a $0.04 first-month storage reserve: **$2.00 cap**.
- Expected from A1500's dev step (6,386 s): about **1.8 h, ~$0.73**.
- Private upload: **104 files, 157,283,029 bytes** (dev tasks, 94 dev databases, 5 tokenizer files and
  the 4 A1500 adapter files, 121 MB of the total). Output is a few MB (no checkpoints). Storage at most 30 days without renewed approval.

## Pass, abort and decision

Pass means the run is valid:
- account, rate, pins and private upload verified, and every dev reference passes before model use;
- an exact reload of the pinned step-180 adapter;
- 95 dev episodes with one finite projected position, and AuditKIT agreeing with every score;
- the new tools recorded in the summary and every trajectory.

It aborts on a wrong account or price, CUDA failure, OOM, reload mismatch, sync failure, timeout, stall,
or AuditKIT disagreement. **No automatic retry or second job.**

**Decision rule:** adopt the new tools for the final evaluation only if A1500 with hints reaches **at least
42/95 strict** (five unrounded points over 37). If adopted, every final-evaluation model uses the same
tools. Results are reported whatever they are, including recoveries after a failed query, the complex
subset, refusals and hints, and a determinism check.

Excluded: training, other models, step-limit, prompt or decoding changes, the full stage, Q2, A+/B, BIRD/final,
teacher/API, vLLM, hardware change, push and publication.
