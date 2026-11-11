# Final evaluation of B (amendment 14): both jobs PASS; B ≈ A1500 on every final set (2026-11-11)

Approved by the user ("approved", 2026-11-10) and launched once each by Claude from clean local commit
`eadd6e9`. **No retry, second job, extra upload, push or publication** followed. Machine-readable results:
`RESULTS.json`, written by a local read-only script.

| Job | Account | Registration SHA256 | Running | Cost (rate) |
|---|---|---|---|---:|
| 6af321f0ab34849891abec15, BIRD | hf-user | `f691c98b…aa6d3` | 12.30 h | $4.92 |
| 6af321efab34849891abec10, SQaLe final + retention | hf-user | `0b74eea1…5ba89` | 4.57 h | $1.83 |
| **Total** | | | 16.9 GPU-h | **$6.75** |

B was fixed as the submitted model from dev results only (amendment 14), before any of these numbers
existed. Base and A1500 numbers come from `experiments/sql-final-v1` (same tasks, protocol, GPU type and
SQLite 3.40.1). They were not re-run.

## Headline (strict, the primary score)

| Set | Base | A1500 | **B** | B only / A1500 only | p (exact McNemar) |
|---|---:|---:|---:|---:|---:|
| SQaLe final (144, unseen databases) | 4 (2.8%) | 45 (31.3%) | **44 (30.6%)** | 6 / 7 | 1.0 |
| BIRD Mini-Dev (493) | 0 (0.0%) | 8 (1.6%) | **7 (1.4%)** | 1 / 2 | 1.0 |
| Shop retention, dev (40) | 0 | 40 | **40** | 0 / 0 | 1.0 |
| Shop retention, multi-hop (40) | 0 | 1 | **0** | 0 / 1 | 1.0 |

The set metric (BIRD's usual execution metric) is a tie on both main sets: SQaLe final 51 vs 51 and BIRD
8 vs 8.

**Read plainly:**
- On every final set, B and A1500 are the same within noise. Pooled over all four sets, B solves 7 tasks
  A1500 misses and misses 10 that A1500 solves.
- B's dev edge (40 vs 37 of 95, p = 0.375) did not carry over to new data. That is what a small,
  non-significant dev difference usually does.
- **Before → after holds up for B as for A1500:** SQaLe final 2.8% → 30.6%, BIRD 0% → 1.4%, retention
  dev 0 → 40/40.

## Where the scores come from (strict)

SQaLe final by query shape:

| Shape | n | Base | A1500 | B |
|---|---:|---:|---:|---:|
| 1-table lookup | 36 | 4 | 27 | 29 |
| 1-table aggregate | 13 | 0 | 6 | 7 |
| two tables | 44 | 0 | 10 | 7 |
| 3+ tables | 19 | 0 | 1 | 0 |
| multi-step | 32 | 0 | 1 | 1 |

BIRD by the source's difficulty labels: challenging 6/453 (A1500 6), moderate 0/32 (A1500 0) and
simple 1/8 (A1500 2).

**Complex queries did not improve.** B solves 1 of 51 complex SQaLe final questions (A1500 2), and 0 of
40 multi-hop shop questions (A1500 1).

## Behaviour

| | SQaLe final: ran out of steps | BIRD: ran out of steps | Multi-hop: ran out of steps | Submitted query failed to run (all sets) |
|---|---:|---:|---:|---:|
| Base | 5 | 9 | 0 | 377 |
| A1500 | 39 | 235 | 20 | 2 |
| B | 46 | 252 | 32 | 2 |

B runs out of its eight tool steps somewhat more often than A1500, the opposite of what dev showed
(22 vs 25). That suggests the self-training data, which was mostly simple questions solved in few steps,
did not teach it to finish longer tasks. Malformed turns fell further: BIRD 7 vs 25 for A1500, SQaLe
final 0 vs 1.

## Checks (all PASS)

| Check | BIRD job | SQaLe + retention job |
|---|---|---|
| Completed, every step exit code 0 | PASS | PASS |
| Account and registration SHA in the job equal the launch receipt and the file | PASS | PASS |
| Every reference re-checked in the job before any model use (0 model calls) | PASS (493) | PASS (224) |
| Same GPU (Tesla T4) and SQLite (3.40.1) as A1500's final jobs | PASS | PASS |
| Every step reloads exactly B's step-40 adapter `90b04823…` (all tensors match) | PASS | PASS (3 steps) |
| Every task evaluated once; one finite projected position per generated token | PASS | PASS |
| AuditKIT agrees with every score | PASS | PASS |
| No out-of-memory event | PASS (peak 6.16 GiB) | PASS |

## What this means for the submission

- **Submitted model: B**, as fixed in advance from dev. Its final scores are effectively the same as A1500's.
- The self-training round is reported as it is:
  - it raised dev by 3 tasks (not significant);
  - it beat its matched control 4-0 on dev (not significant);
  - it gave no measurable gain on unseen final data.
