# A1500: train on the 1,500 set, then screen on the 95 dev tasks (awaiting approval)

Registration **`a1500-sql-agenttune-20261108-01`**, machine file `REGISTRATION.json` (SHA256 is printed by
`scripts/sql_job.py plan`; the approval must name it). Built by `scripts/prepare_sql_a1500.py` under
[amendment 8](../sqale-sft/AMENDMENT-8-a1500-stage.md). **No upload or launch is approved or performed.**

## What runs

| Step | What | Step limit |
|---|---|---:|
| train-a1500 | Train from the untuned base on `train-1500.episodes.jsonl` (contains all 500 A500 rows): 1,425 train / 75 validation, two passes, **180 updates** | 7 h |
| dev-a1500-nf4 | Reload exactly the step-180 adapter, then screen it on the 95 effective dev tasks | 3.5 h |

Settings are identical to [A500](../sql-a500/REGISTRATION.md): AgentTune audited factory, NF4/fp16
LoRA r16/a32/dropout .05, `trained-positions-v1`, LR 2e-4 cosine, accumulation 16, checkpoints every 10
updates (keep 3), validation every 50. Dev: greedy, 8 tool steps, 512 new tokens/turn, 8,192 context,
`sql-submit-v1`, table-read-v1 strict plus fallback and set, AuditKIT agreement, one projected position.
All 1,500 training references and 95 dev references pass locally before any model use, with zero model calls.
All 1,500 rows re-encode to the frozen counts (2,684,785 inputs / 330,742 targets per pass).

Base 2/95, Pilot 20/95 and A500 28/95 are reused as comparators and are not rerun; their metrics files
are pinned by hash.

## Why this run

A500's gains came mostly from simple lookups; it solved 1 of the 30 complex dev questions (3+ tables or
multi-step). After the validation split, A1500 trains on 454 complex SQaLe rows; A500 trained on 100. This run measures whether more of that data helps. Amendment 8 adds a frozen query-shape breakdown
(`src/ayadb/queryshape.py`), reported for all four models. It is descriptive, not a gate.

## Money, time and storage

- One t4-small job on hf-user, rate $0.006667/min (rechecked by preflight), **9.5-hour hard timeout**.
  Compute ceiling $3.80 plus a $0.04 first-month storage reserve: **$4.00 cap**.
- Expected from A500's measured speed: training ~4.1 h (5.6 h at the slower G0-02 speed), dev screen
  ~1.6-2.5 h, about **5.8 h, ~$2.34**.
- Private upload: **1,211 files, 233,345,871 bytes** (training episodes, dev tasks, 1,204 SQLite databases,
  tokenizer). Outputs are about 0.7 GB (three checkpoints with optimizer state, adapter, logs), within the
  1 GiB cap. Storage at most 30 days without renewed approval.

## Pass, abort and decisions

Pass means the run is valid:
- account, rate, pins and private upload verified, and every reference passes before model use;
- fresh encoding matches the frozen plan;
- 180 finite optimizer updates, none skipped, with the ledger covering all of them;
- finite validation losses and memory events saved;
- an exact step-180 reload;
- 95 dev episodes with one finite projected position, and AuditKIT agreeing with every score.

The run aborts on any of the A500 abort conditions. **No automatic retry or second job.**

After the run:
- **Stage rule:** A1500 needs at least **33/95** strict (five unrounded points over A500's 28) before a
  full-stage registration may be proposed. Otherwise data scaling stops at 1,500.
- Finalists for the final evaluation are not chosen here.
- Amendment 5's manual audit is repeated for A1500.
- Results are reported whatever they are.

Excluded: reruns of Base/Pilot/A500, intermediate checkpoint screens, shop retention, step-limit or
prompt changes, the full stage, Q2, A+/B, BIRD/final, teacher/API, vLLM, hardware change, push and publication.
