# Dev baselines result: PASS (2026-11-07)

Job 6aef6392e6320c66fad617ba,
hf-user, one t4-small (Tesla T4, CUDA 12.6). Launched once from clean local commit `15b12cc` with the
exact approved registration SHA256 `53049d18e68d5645e3a656e63a00232b6fe8ce86c49f2ee55979e02d89410ee9`.
The approved private input upload (103 files, 157,283,004 bytes) completed before launch. **No retry,
second job, extra upload, push or publication** followed. Machine-readable results: `RESULTS.json`
(generated from the pulled files). G1 is computed in `../sql-a500/RESULTS.md` once the A500 job finishes.

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, registration SHA, private upload | PASS | Launch/upload receipts; `environment.json` carries the same SHA |
| All 95 dev references pass before model use | PASS | `oracle-precheck.json`: 95 checked, 0 model calls |
| 95 episodes per condition, same tasks | PASS | Both summaries: 95 episodes; identical task sets |
| One finite projected inference position | PASS | Checked forwards equal generated tokens (11,669 and 21,627); `logits_to_keep=1` |
| AuditKIT agrees with every saved score | PASS | 95/95 per condition; AuditKIT successes equal strict successes (2 and 20) |
| Memory events saved, no OOM | PASS | After model load and at completion, per condition |
| Both steps exit 0 | PASS | `status.json`: complete |

## Dev accuracy (95 effective dev tasks, greedy, eight tool steps)

| Condition | Strict | Fallback | Set |
|---|---:|---:|---:|
| Base (untuned Tiny Aya Global, zero-shot) | **2/95 (2.1%)** | 2/95 (2.1%) | 2/95 (2.1%) |
| Pilot adapter (`6f2827ae…`, trained on the shop tasks only) | **20/95 (21.1%)** | 21/95 (22.1%) | 22/95 (23.2%) |

Strict is the primary score: the full submitted result must match, and the query must read a real
table (table-read-v1). Fallback is a separate diagnostic: it scores the last `run_sql` query that ran
without error, whether or not the model submitted, so it can pass when strict fails and the other way
round (Base: one of each). Set compares the submitted rows ignoring order and duplicates. Each figure is one greedy run on 95 tasks, so it is rough:
a 20/95 score is consistent with anything from about 14% to 30%.

## How the two models behave

| | Base | Pilot |
|---|---:|---:|
| Episodes that submitted a query | 54 | 65 |
| Episodes that never submitted | 41 | 30 (28 of them used all 8 tool steps) |
| Submitted queries that read no table | 47, all failed queries | 10, all failed queries |
| Constant-only answers that ran (valid no-table) | 0 | 0 |
| Tool calls per episode, mean / median | 1.8 / 1 | 5.6 / 5 |
| SQL errors, total | 98 | 129 |
| Malformed assistant turns (episodes) | 32 (23) | 3 (3) |
| Invented tool results | 0 | 0 |
| Repeated identical calls | 6 | 76 |
| Generated / prompt tokens | 11,669 / 197,710 | 21,627 / 819,865 |
| Seconds per episode, mean / median | 16.5 / 14.1 | 57.2 / 43.3 |

Base almost never lists the tables first (6 of 95 episodes). It guesses a table name and describes,
queries or submits against it right away. Of its 47 submissions that read no table, 33 failed with "no
such table", 8 with "no such column" and 6 were rejected as invalid. In the 41 episodes with no
submission, it stopped after a call or two with a plain-text reply (23) or a malformed call (18).

The Pilot opens with `list_tables` in 92 of 95 episodes, then describes tables before querying, as it
learned on the shop tasks. Some of that carries over to unseen SQaLe databases. Its main failure is
running out of the eight tool steps before submitting: 28 of its 30 unsubmitted episodes used all
eight, and 25 of those repeated an identical call.

## Time, cost and memory

- Created 17:44:50 UTC, started 17:47:35, finished 19:46:58 on 2026-11-07. Running **7,162.7s
  (1.99 h)**; provider scheduling 164s.
- Steps: Base 1,633.9s (0.45 h), Pilot 5,451.3s (1.51 h); setup and sync outside steps 77s.
  Forecast was 0.33 h + 2.21 h, about 2.5 h; actual 2.0 h, inside the 4.5-hour limit.
- Rate $0.006667/min: **$0.80** (prorated $0.7959). Rate-based, **not an invoice**. Cap was $2.00.
- GPU peaks (device 14.74 GiB): Base 3.19 / 3.33 GiB allocated/reserved, Pilot 3.95 / 4.13. No OOM.

Storage: inputs 157,283,004 bytes plus 2.7 MB of outputs in the private bucket `hf-user/aya-db-agent`.
Approved retention is at most 30 days: **through 2026-12-07** unless the user renews it. Pulled copy:
`remote/hf-sql/hf-user/dev-baselines-sql-20261107-01/`. Redacted provider log/metadata:
`runs/hf-sql/hf-user/dev-baselines-sql-20261107-01/`.

## What this does and does not show

These are the two reference points for the 500 stage. Base is close to zero on this format, and the
shop-trained Pilot already gets about a fifth of the SQaLe dev tasks right. A500 has to beat Base by
10 points on both strict and fallback (G1). Comparing it with the Pilot shows whether SQaLe training
adds anything beyond the shop data. Nothing here touches the final splits, BIRD or the spent shop
test.

The amendment-5 manual audit is done, with labels in `manual-audit/` and the summary in
`../sql-a500/RESULTS.md`. Samples were fixed by the seed-42 rule before reading, and the reader was
Claude, not a human. Base had 20 of 20 sampled failures really wrong and both successes correct. The Pilot had
16 failures really wrong and 4 "correct but scorer too strict" (2 borderline), and all 10 successes were correct. No
lucky passes. Scores are unchanged.

**G1 passes**: A500 scored 28/95 strict and 29/95 fallback against Base's 2 and 2 (see `../sql-a500/RESULTS.md`).
