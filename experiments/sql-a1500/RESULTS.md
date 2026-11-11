# A1500 result: PASS, 37/95 strict, and the stage rule is met (2026-11-08)

Job 6af0139be6320c66fad700ed,
hf-user, one t4-small (Tesla T4, CUDA 12.6). Launched once from clean local commit `67c530f` with the
exact approved registration SHA256 `92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363`.
The approved private input upload (1,211 files, 233,345,871 bytes) completed before launch. **No retry,
second job, extra upload, push or publication** followed. Machine-readable results: `RESULTS.json`
(generated from the pulled files; comparator metrics verified against their pinned hashes).

## Headline

| Model (95 effective dev tasks, greedy) | Strict | Fallback | Set |
|---|---:|---:|---:|
| Base, untuned zero-shot | 2 (2.1%) | 2 (2.1%) | 2 (2.1%) |
| Pilot (shop tasks only) | 20 (21.1%) | 21 (22.1%) | 22 (23.2%) |
| A500 (500 episodes: 265 SQaLe + 235 shop) | 28 (29.5%) | 29 (30.5%) | 30 (31.6%) |
| **A1500** (1,500 episodes: 1,265 SQaLe + the same 235 shop, fresh from base) | **37 (38.9%)** | **37 (38.9%)** | **40 (42.1%)** |

**Stage rule met.** A1500 beats A500 by 9.47 unrounded strict points (the rule needs 5, i.e. 33/95).
This only permits *proposing* a full-stage registration; it approves nothing.

Task by task on strict, A1500 solves 12 tasks A500 misses and misses 3 that A500 solves (exact McNemar
p = 0.035, run afterwards, unadjusted). Against the Pilot the split is 18 to 1, against Base 35 to 0.
Fallback against A500 is 12 to 4 (p = 0.077). With 95 tasks, a 38.9% score is consistent with roughly
30% to 49% (A500: 21% to 39%).

Strict successes by difficulty, Base / Pilot / A500 / A1500: simple (44) 2 / 16 / 22 / 28, moderate (32)
0 / 4 / 6 / 7, hard (19) 0 / 0 / 0 / **2**. The two hard wins are a window-function query and a two-table query.

## Complex queries did not improve

This was the reason for the run, so it comes first. Amendment 8's frozen query-shape diagnostic
(descriptive, not a gate; classes defined after reading the A500 results) gives:

| Shape (dev tasks) | Base | Pilot | A500 | **A1500** |
|---|---:|---:|---:|---:|
| 1-table lookup (37) | 2 | 16 | 21 | **25** |
| 1-table aggregate (5) | 0 | 0 | 1 | **2** |
| two tables (23) | 0 | 4 | 5 | **9** |
| 3+ tables (10) | 0 | 0 | 1 | **0** |
| multi-step (20) | 0 | 0 | 0 | **1** |
| **Complex total (30)** | 0 | 0 | 1 | **1** |

Ran out of the eight tool steps without submitting, per shape, Base / Pilot / A500 / A1500:
lookup 0 / 3 / 5 / 5, aggregate 1 / 1 / 0 / 0, two tables 0 / 6 / 6 / 5, 3+ tables 0 / 5 / 5 / 3,
multi-step 0 / 13 / 11 / 12. **Complex: 0 / 18 / 16 / 15 of 30.**

On the 30 complex tasks, A1500 and A500 each solve exactly one, and not the same one: A1500 gains a
window-function query, and loses A500's 3+-table win. That is 1 to 1, no change. Training on 454 complex
SQaLe rows instead of 100 did not move complex accuracy. Of A1500's 29 complex failures, 15 ran out of
steps and 14 submitted wrong logic.

All of A1500's net gain is on simpler shapes: of its 12 new wins, 6 are one-table lookups, 4 two-table
joins, 1 an aggregate and 1 multi-step. Two-table joins nearly doubled (5 → 9 of 23).

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, registration SHA, private upload | PASS | Launch/upload receipts; `environment.json` has the same SHA |
| Every training and dev reference passes before model use | PASS | `oracle-precheck.json`: 1,595 checked (1,500 train + 95 dev), 0 model calls |
| Fresh encoding matches the frozen 1,425/75 plan | PASS | 2,684,785 inputs and 330,742 targets per pass; `fresh_encoding_verified` |
| 180 finite optimizer updates via the AgentTune factory, none skipped | PASS | Ledger: 180 steps, 0 skipped, 2,850 example visits, 1,425 unique examples, 5,369,570 inputs / 661,484 targets, 0 padding |
| Validation losses finite | PASS | 0.0617 (update 50), 0.0529 (100), 0.0500 (150), 0.0497 (180) |
| Memory events after load, every update, completion | PASS | 180 update events; no OOM |
| Dev step reloads exactly the step-180 adapter | PASS | `checkpoint-reload.json`: 504 tensors, all match, SHA `69f0f125…` = final adapter = checkpoint 180 |
| 95 dev episodes, one finite projected position | PASS | 30,143 checked forwards = generated tokens; `logits_to_keep=1` |
| AuditKIT agrees with every saved score | PASS | 95/95; AuditKIT successes = 37 strict |

## Training

Same settings as A500 (AgentTune `create_distill_trainer → Project.distill → AuditedSFTFactory`, NF4/fp16
LoRA r16, `trained-positions-v1`, LR 2e-4 cosine, accumulation 16, two passes), trained fresh from the
base model on 1,425 examples with 75 held out. The logged loss fell from 0.597 at the first update to
0.017 at the last (mean 0.070). Validation loss fell steadily, 0.0617 → 0.0497. It is measured on a
different, larger held-out set than A500's (75 vs 25 examples), so it cannot be compared with A500's 0.038.
Trainer runtime 14,574 s (4.05 h), **368 input tokens/s** (A500: 364). Peak GPU memory 8.68 / 9.79 GiB
allocated/reserved of 14.74 GiB. Checkpoints 160, 170 and 180 are kept with full optimizer state.

## How A1500 behaves

| | Base | Pilot | A500 | **A1500** |
|---|---:|---:|---:|---:|
| Episodes that open with `list_tables` | 6 | 92 | 95 | **95** |
| Episodes that submitted | 54 | 65 | 67 | **70** |
| Submitted queries that failed to run | 49 | 14 | 1 | **0** |
| Strict successes among submitted | 2 / 54 | 20 / 65 | 28 / 67 | **37 / 70** |
| Submissions that read no table | 47 | 10 | 0 | **0** |
| Unsubmitted, all 8 tool steps used | 1 | 28 | 27 (26 repeated an identical call) | **25 (24 repeated)** |
| Malformed assistant turns | 32 | 3 | 1 | **0** |
| Invented tool results | 0 | 0 | 0 | 0 |
| SQL errors returned by `run_sql` | 98 | 129 | 127 | **112** |
| Tool calls per episode, mean | 1.8 | 5.6 | 5.8 | **5.5** |
| Seconds per episode, mean | 16.5 | 57.2 | 62.0 | **67.0** |
| Generated / prompt tokens | 11,669 / 197,710 | 21,627 / 819,865 | 24,727 / 857,007 | **30,143 / 814,197** |

A1500 is cleaner than A500 everywhere: every submitted query runs, no malformed turns, and 53% of its
submissions are fully correct (A500 42%). The biggest failure is unchanged: 25 of 95 episodes run out
of the eight tool steps, 24 of them after repeating an identical failing call, typically on a guessed or
misspelled column name.

## Manual audit (amendment 5, repeated)

The sample was fixed first by the registered seed-42 rule (20 of 58 failures, 10 of 37 successes), then
read. The reader is Claude, not a human; labels change no scores.

| Model | Failures: really wrong / scorer too strict | Successes: correct / lucky |
|---|---|---|
| A500 (earlier) | 19 / 1 (borderline) | 10 / 0 |
| **A1500** | **17 / 3** (1 borderline) | **10 / 0** |

- **Successes:** no lucky passes. Two used shortcuts that are safe only because a key is unique (a
  dropped filter on a unique backup ID, `budget` instead of `SUM(budget)` on a unique department name);
  both were checked against the database and are right.
- **"Scorer too strict":** two returned exactly the gold rows in an order the question did not ask for
  (set matches). One (borderline) had both correct totals but repeated them in 71 identical rows. Scaling
  2-3 of 20 to all 58 failures suggests roughly 6 to 9 failures are right answers in a different shape
  (A500: about 3 of 67). That is diagnostic only and does not change the stage rule; it would widen the
  gap, not close it.
- **Failures:** 9 of the 17 really-wrong cases ran out of the eight steps, 7 of them repeating a query
  on a non-existent or misspelled column. The other 8 submitted wrong logic: joins that multiply counts,
  a filter applied before an average, the wrong ID column, or a wrong join returning 0 rows.

Per-episode labels and reasons: `manual-audit/dev-a1500-nf4.json`.

## Time, cost and memory

- Created 06:15:55 UTC, started 06:16:08, finished 12:09:24 on 2026-11-08. Running **21,196 s (5.89 h)**;
  scheduling 13 s. Steps: training 14,644 s (4.07 h), dev screen 6,386 s (1.77 h), setup/sync 166 s.
  Forecast was ~5.8 h (training ~4.1 h, dev 1.6-2.5 h).
- Rate $0.006667/min: **$2.36** (prorated $2.3552). Rate-based, **not an invoice**. Cap was $4.00.
- All SQL runs on hf-user so far, by rate: G0-02 $0.08, A500 $1.19, baselines $0.80, A1500 $2.36.
- GPU peaks (14.74 GiB device): training 8.68 / 9.79 GiB; dev inference **10.14 / 10.42 GiB**, up from
  A500's 4.45 / 4.64 GiB. A1500's longest conversation is about twice A500's (15,193 vs 7,657 characters
  of task and tool text), which likely made one prefill much larger. This is not verified. 4.1 GiB stayed
  free and nothing ran out of memory, but a longer step limit would need its memory checked first.

Storage: inputs 233,345,871 bytes plus a 673 MB output snapshot (3 checkpoints with optimizer state,
final adapter, logs) in the private bucket `hf-user/aya-db-agent`. Approved retention is at most 30
days: **through 2026-12-08** unless the user renews it. Pulled copy:
`remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/`. Redacted provider log/metadata:
`runs/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/`.

## What this does and does not show

Tripling the training data (SQaLe episodes 265 → 1,265; the same 235 shop episodes), with everything else
fixed, lifts dev strict from 29.5% to 38.9% and
passes the preregistered stage rule. The gain comes from simpler shapes (lookups and two-table joins),
not from complex queries: complex stays at 1 of 30, and the eight-step loop is still the main failure.
More data of the same kind is therefore unlikely to fix complex queries by itself; the step-limit loop
and wrong multi-table logic are the targets. The full stage is now allowed to be proposed, but whether
it is worth it is the user's decision. Nothing was selected on a final split, BIRD, Q2 or the spent shop
test, and shop retention is still unmeasured (deferred to the final evaluation by amendment 7), so
the submission model is unchanged.
