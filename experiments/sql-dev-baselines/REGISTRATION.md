# Dev baselines: Base and Pilot on the 95 dev tasks (awaiting approval)

Registration **`dev-baselines-sql-20261107-01`**, machine file `REGISTRATION.json` (SHA256 is printed by
`scripts/sql_job.py plan`; the approval must name it). Built by `scripts/prepare_sql_a500.py` under
[amendment 7](../sqale-sft/AMENDMENT-7-a500-stage.md). **No upload or launch is approved or performed.**
It runs in parallel with [`a500-sql-agenttune-20261107-01`](../sql-a500/REGISTRATION.md), on the same
hardware type with the same model, precision and evaluation settings, so the three conditions compare.

## What runs

| Step | What | Step limit |
|---|---|---:|
| dev-base-nf4 | Untuned Tiny Aya Global (zero-shot) on the 95 effective dev tasks | 1.5 h |
| dev-pilot-nf4 | Existing Pilot adapter `6f2827ae...` on the same tasks | 4 h |

Both: greedy, 8 tool steps, 512 new tokens/turn, 8,192 context, one projected inference position,
`sql-submit-v1`, table-read-v1 strict plus fallback and set scores, every-submission no-table counts,
AuditKIT agreement. Base and Pilot never learned `submit_sql`; the fallback score (last successful
query) is reported so format learning is not counted as SQL skill. All 95 dev references pass
locally before any model use, with zero model calls.

## Money, time and storage

- One t4-small job on hf-user, rate $0.006667/min (rechecked by preflight), **4.5-hour hard timeout**.
  Compute ceiling $1.80 plus a $0.04 first-month storage reserve: **$2.00 cap**.
- Expected from G0-02's per-episode times: Base ~0.33 h (~$0.13), Pilot ~2.2 h (~$0.88), about
  **2.5 h, ~$1.02**. Rough, ±50%: based on three tasks per model.
- Private upload: **103 files, 157,283,004 bytes** (dev tasks, 94 SQLite databases, tokenizer, Pilot
  adapter). Outputs are small (trajectories and scores). Storage at most 30 days without renewed approval.

## Pass, abort and reporting

Pass means the run is valid: references pass before model use, 95 episodes per condition with one
finite projected position, AuditKIT agrees with every score, memory events saved. Abort on wrong
account or higher price, CUDA failure/OOM/nonfinite logits, input or database hash mismatch, sync
failure or byte cap, timeout or stall (30 minutes without output), missing memory record, full-prompt
projection or AuditKIT disagreement. **No automatic retry or second job.** Results are reported
whatever they are; G1 and the stage rule wait for both jobs. Excluded as in amendment 7: shop
retention, a one-example Base prompt, Q2, BIRD/final, teacher/API, vLLM, hardware change, push and
publication.
