# A500 result: PASS, and G1 passes (2026-11-07)

Job 6aef63bbe6320c66fad617fa,
hf-user, one t4-small (Tesla T4, CUDA 12.6). Launched once from clean local commit `15b12cc` with the
exact approved registration SHA256 `c3b006ce2377c24762a8d57bdddd5a3499e9e817c7bb43ef0d16a21af4e47e3d`,
in parallel with the [dev baselines job](../sql-dev-baselines/RESULTS.md). The approved private input
upload (367 files, 82,449,877 bytes) completed before launch. **No retry, second job, extra upload, push
or publication** followed. Machine-readable results: `RESULTS.json` (generated from the pulled files).

## Headline

| Model (95 effective dev tasks, greedy) | Strict | Fallback | Set |
|---|---:|---:|---:|
| Base, untuned zero-shot | 2 (2.1%) | 2 (2.1%) | 2 (2.1%) |
| Pilot (shop tasks only) | 20 (21.1%) | 21 (22.1%) | 22 (23.2%) |
| **A500** (500 episodes: 265 SQaLe + 235 shop, 2 passes) | **28 (29.5%)** | **29 (30.5%)** | **30 (31.6%)** |

**G1 passes.** A500 beats Base by 27.4 points on strict and 28.4 points on fallback; the rule needs 10 on both.

A500 also beats the Pilot by 8.4 points on strict, fallback and set. That comparison is not a registered
gate. Task by task, A500 solves 10 tasks the Pilot misses, and the Pilot solves 2 that A500 misses (an
exact McNemar test, run afterwards and unadjusted, gives p = 0.039). Against Base the split is 26 to 0.
With 95 tasks, a 29.5% score is consistent with roughly 21% to 39%.

No model solved any of the 19 "hard" tasks. Strict successes by difficulty, Base / Pilot / A500:
simple (44) 2 / 16 / 22, moderate (32) 0 / 4 / 6, hard (19) 0 / 0 / 0.

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, registration SHA, private upload | PASS | Launch/upload receipts; `environment.json` has the same SHA |
| Every training and dev reference passes before model use | PASS | `oracle-precheck.json`: 595 checked (500 train + 95 dev), 0 model calls |
| Fresh encoding matches the frozen 475/25 plan | PASS | 847,065 inputs and 107,381 targets per pass; `fresh_encoding_verified` |
| 60 finite optimizer updates via the AgentTune factory, none skipped | PASS | Ledger: 60 steps, 0 skipped, 950 example visits, 475 unique examples, 1,694,130 inputs / 214,762 targets |
| Validation losses finite | PASS | 0.03813 at update 50, 0.03824 at update 60 |
| Memory events after load, every update, completion | PASS | 60 update events; no OOM |
| Dev step reloads exactly the step-60 adapter | PASS | `checkpoint-reload.json`: 504 tensors, all match, SHA `8a1d3ac1…` = final adapter = checkpoint 60 |
| 95 dev episodes, one finite projected position | PASS | 24,727 checked forwards = generated tokens; `logits_to_keep=1` |
| AuditKIT agrees with every saved score | PASS | 95/95; AuditKIT successes = 28 strict |

The amendment-7 sync fix worked: `checkpoint-reload.json` is now in the synced output; G0-02 lost it.

## Training

Through AgentTune `create_distill_trainer → Project.distill → AuditedSFTFactory`, NF4/fp16 LoRA r16,
`trained-positions-v1`, LR 2e-4 cosine, accumulation 16, two passes over 475 examples (25 held out for
validation). The logged loss fell from 0.577 at the first update to 0.019 at the last (mean 0.073).
Trainer runtime 4,656s (1.29 h), about **364 input tokens/s**, faster than the 266 tokens/s forecast
measured on G0-02's single longest example. Peak GPU memory 8.48 / 9.69 GiB allocated/reserved of
14.74 GiB. Checkpoints 40, 50 and 60 are kept with full optimizer state. The adapter has 30.2M
trainable parameters.

## How A500 behaves

| | Base | Pilot | A500 |
|---|---:|---:|---:|
| Episodes that open with `list_tables` | 6 | 92 | **95** |
| Episodes that submitted | 54 | 65 | 67 |
| Submitted queries that failed to run | 49 | 14 | **1** |
| Strict successes among submitted | 2 / 54 | 20 / 65 | 28 / 67 |
| Submissions that read no table | 47 | 10 | **0** |
| Unsubmitted, all 8 tool steps used | 1 | 28 | 27 (26 repeated an identical call) |
| Malformed assistant turns | 32 | 3 | 1 |
| Invented tool results | 0 | 0 | 0 |
| Tool calls per episode, mean | 1.8 | 5.6 | 5.8 |
| Seconds per episode, mean / median | 16.5 / 14.1 | 57.2 / 43.3 | 62.0 / 50.2 |
| Generated / prompt tokens | 11,669 / 197,710 | 21,627 / 819,865 | 24,727 / 857,007 |

A500 learned the submit format and schema-first exploration. Almost everything it submits runs, and
42% of its submissions are fully correct. Its biggest loss is the same as the Pilot's: 27 of 95
episodes run out of the eight tool steps, almost always after repeating an identical failing call
(unquoted column names with spaces, a typo, a table name with a stray space).

## Manual audit (amendment 5)

Samples were fixed first by the registered seed-42 rule, then read. 20 failures and 10 successes per
model; Base had only 2 successes (shortfall 8). The reader is Claude, not a human; labels change no scores.

| Model | Failures: really wrong / scorer too strict | Successes: correct / lucky |
|---|---|---|
| Base | 20 / 0 | 2 / 0 |
| Pilot | 16 / 4 (2 borderline) | 10 / 0 |
| A500 | 19 / 1 (borderline) | 10 / 0 |

- **Successes:** no lucky passes. Every sampled success inspected the schema (or, once for Base, used
  the table list from an error), ran a correct query and submitted it.
- **"Scorer too strict" cases** are right answers in a different shape: an extra or missing column the
  question did not clearly ask for, an ORDER BY the question did not request, or duplicate rows. Scaling
  the A500 rate (1 of 20 sampled failures) to all 67 failures suggests only a few points are hidden by
  strictness, and the same kind of case appears for the Pilot. It does not change G1.
- **Notable failures:**
  - A500 failure [3] got the right two numbers only by coincidence. Its logic was wrong, and no loan in that database has both kinds of payment, so the wrong logic gave the same counts.
  - A500 failure [15] had the right query, but a malformed tool call ended the episode before it ran.
  - One Pilot failure divided money by 100, a habit carried over from the shop tasks.
- **Fallback quirk:** Base passed the fallback score once without submitting. Its query ignored a join
  that happened not to matter in that database.

Per-episode labels and reasons: `manual-audit/dev-a500-nf4.json` and
`../sql-dev-baselines/manual-audit/`.

## Stage rule (for a later, separately approved 1,500 run)

The preregistered spending rule is: stop at 1,500 unless it beats 500 by at least five unrounded strict
points on the same 95 tasks. It cannot be applied until a 1,500 model exists. For reference, a 1,500
model would need at least **33/95** strict (34.7%), i.e. five net additional successes over A500's 28.
(Amendment 7 phrases this as "train 1,500 only if it beats 500", which reads circularly; the
preregistration and run-input freeze wording above is the operative one.) **No 1,500 run is approved.**

## Time, cost and memory

- Created 17:45:32 UTC, started 17:46:54, finished 20:45:44 on 2026-11-07. Running **10,729s (2.98 h)**;
  scheduling 82s. Steps: training 4,724.8s (1.31 h), dev screen 5,910.1s (1.64 h), setup/sync 95s.
  Forecast was 1.77 h training plus about 2.2 h dev; training was faster and the dev screen shorter.
- Rate $0.006667/min: **$1.19** (prorated $1.1922). Rate-based, **not an invoice**. Cap was $3.00.
- Both 500-stage jobs together: about **$1.99** compute by rate (A500 $1.19 + baselines $0.80).
- GPU peaks (14.74 GiB device): training 8.48 / 9.69 GiB, dev inference 4.45 / 4.64 GiB. No OOM.

Storage: inputs 82,449,877 bytes plus a 672 MB output snapshot (3 checkpoints with optimizer state,
final adapter, logs) in the private bucket `hf-user/aya-db-agent`. Approved retention is at most 30
days: **through 2026-12-07** unless the user renews it. Pulled copy:
`remote/hf-sql/hf-user/a500-sql-agenttune-20261107-01/`. Redacted provider log/metadata:
`runs/hf-sql/hf-user/a500-sql-agenttune-20261107-01/`.

## What this does and does not show

Training on 500 verified episodes (265 SQaLe, 235 converted shop) makes Tiny Aya Global a working schema-exploring SQL agent:
2% → 29.5% strict on unseen dev databases, and clearly above the shop-only Pilot. It does not show how
far more data helps (that is the 1,500 stage), it does not touch hard multi-step questions (0/19), and
it was selected on nothing: no final split, BIRD, Q2 or spent shop test was used. Shop retention was
deferred to the final evaluation by the user's choice (amendment 7). The step-budget looping is the
clearest remaining failure. Fixing it would mean changing the protocol, and any such change needs its
own amendment and approval.
