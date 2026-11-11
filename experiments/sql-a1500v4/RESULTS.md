# A1500-v4 result: PASS (valid run), 35/95 strict, does not replace A1500 (2026-11-09)

Job 6af0d12ae6320c66fad77f73,
hf-user, one t4-small (Tesla T4). Launched once by Claude from clean local commit `2a5b81e`, with the
exact approved registration SHA256 `516dc07677efb90e1ed6c50188c4689c71e386881a6756999a377e5285aaad48`.
The approved private upload (1,211 files, 234,112,064 bytes) completed before launch. **No retry, second
job, extra upload, push or publication** followed. Machine-readable results: `RESULTS.json`.

## Headline

| 95 dev tasks, greedy, original tools | Strict | Fallback | Set | Complex (30) | Ran out of steps |
|---|---:|---:|---:|---:|---:|
| A1500 (v3 demos) | 37 | 37 | 40 | 1 | 25 |
| **A1500-v4** (same questions, recovery and step-by-step demos) | **35** | **36** | **39** | **2** | **20** |

**Decision rule not met:** A1500-v4 needed at least 38/95 to replace A1500. **A1500 stays the trained
finalist.**

Task by task, A1500-v4 solves 3 tasks A1500 misses and misses 5 that A1500 solves (exact McNemar p = 0.73).
That is no measurable difference either way. Four of the five lost tasks had no error at all: it submitted
a working but wrong query, which looks like ordinary run-to-run variation. Complex tasks went 1 → 2
(one more 3+-table query). Hard tasks were 2 / 19 for both.

## The behaviour changed as trained, the accuracy did not

| After a failed `run_sql` | A1500 | A1500-v4 |
|---|---:|---:|
| Episodes with a failed query | 24 | 21 |
| …of which ended correct | **0** | **0** |
| Failed `run_sql` calls | 104 | **45** |
| Next call: the identical query | **76** | **2** |
| Next call: `describe_table` | 0 | **35** |
| Next call: a changed query | 4 | 1 |
| Episodes that ran out of steps | 25 | **20** (multi-step 12 → 8) |

The new demos taught the habit they showed: after an error, A1500-v4 re-checks the table instead of
repeating itself, and it wastes far fewer steps. But **after the re-check it still sends the same failed
query 16 times out of 25**. It wrote a new query that ran only 5 times, and none of those episodes ended
correct. The step-by-step demos also took: 9 episodes ran a different, successful query before submitting.

This is the same wall the hints run hit (amendment 9). Twice now the right information was in front of
the model, as the column list in a hint or a fresh table description, and it could not turn it into a
corrected query. In the demos, the "fix" is the reference query, which the demonstrator already knew.
The model learned the moves around the fix, not how to produce one.

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, registration SHA, private upload | PASS | Launch/upload receipts; `environment.json` has the same SHA |
| Every training and dev reference passes before model use | PASS | `oracle-precheck.json`: 1,595 checked, 0 model calls |
| Fresh encoding matches the stored v4 counts (A1500's 1,425/75 split) | PASS | 2,798,156 inputs and 347,808 targets per pass |
| 180 finite updates via the AgentTune factory, none skipped | PASS | Ledger: 180 steps, 0 skipped, 2,850 visits, 5,596,312 inputs / 695,616 targets |
| Validation losses finite | PASS | 0.0629 (50), 0.0548 (100), 0.0506 (150), 0.0504 (180); A1500: 0.0497 at 180 (same questions, different paths) |
| Memory events after load, every update, completion | PASS | 180 update events; no OOM |
| Dev step reloads exactly the step-180 adapter | PASS | 504 tensors, all match, SHA `4fd882f4…` |
| 95 dev episodes, one finite projected position | PASS | 23,699 checked forwards = generated tokens |
| AuditKIT agrees with every saved score | PASS | 95/95; AuditKIT successes = 35 strict |

There was one malformed assistant turn (A1500: 0). Every submitted query ran, and no submission read no table.

## Training

Same settings as A1500, trained fresh from the base model. The loss fell from 0.600 to 0.021 (mean
0.072). Trainer runtime 15,714 s (4.37 h), 356 input tokens/s. Peak GPU memory 9.05 / 10.04 GiB in
training, and 6.14 / 6.34 GiB in dev (A1500 dev: 10.14 GiB, driven by one 7,018-token prompt that did
not recur). Checkpoints 160, 170 and 180 are kept with optimizer state.

## Time and cost

- Created 19:44:42 UTC on 2026-11-08, started 19:44:55, finished 01:44:38 on 2026-11-09. Running **21,583 s
  (6.0 h)**: training 15,786 s, dev 5,630 s, setup/sync 167 s. Forecast was ~6.0 h.
- Rate $0.006667/min: **$2.40** (prorated $2.3983). Rate-based, **not an invoice**. Cap was $4.00.
- Storage: inputs 234 MB plus a 673 MB output snapshot in the private bucket, approved **through
  2026-12-09**. Pulled copy: `remote/hf-sql/hf-user/a1500v4-sql-agenttune-20261109-01/`; redacted provider
  log/metadata: `runs/hf-sql/hf-user/a1500v4-sql-agenttune-20261109-01/`.

## What this does and does not show

Rebuilding the repair and complex demos changed how the model behaves after an error, but not how often
it is right: 35 vs 37 is within noise, and recoveries stay at zero. With A1500, the hints run and this
run, the evidence points to a limit in turning an error into a corrected query that neither better
messages nor imitation of an all-knowing demonstrator fixes at this model size and data scale. The
untested remaining idea is training on the model's *own* verified attempts (Q2 self-training), where
the fixes come from things the model can actually produce. That needs its own registration and
approval, and it competes with the final evaluation for the remaining days and credit. Nothing was
selected on final data, and the sealed test stayed closed.
