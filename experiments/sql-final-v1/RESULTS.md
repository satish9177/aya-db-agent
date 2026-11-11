# Final evaluation: Base vs A1500 (2026-11-09/10)

Frozen selection `SELECTION.json` (from dev only, amendment 11): untuned Tiny Aya Global (**Base**) and
**A1500** (adapter `69f0f125…`). Same greedy protocol as dev, original tools, eight tool steps, table-read-v1
strict scoring with the amendment-12 scoring limits. All checks pass: account and registration SHA, every
reference re-checked in the job before model use, frozen-selection gate, exact A1500 reload (504 tensors),
one finite projected position, AuditKIT agreeing with every score, no OOM. Machine-readable: `RESULTS.json`.

| Set (tasks) | Base strict | **A1500 strict** | Change | Base → A1500, set match |
|---|---:|---:|---|---:|
| **SQaLe final** (144, unseen databases) | 4 (2.8%) | **45 (31.3%)** | +28.5 points; 42 gained, 1 lost (p < 0.0001) | 5 → 51 (35.4%) |
| **BIRD Mini-Dev** (493, other source) | 0 (0.0%) | **8 (1.6%)** | +1.6 points; 8 gained, 0 lost (p = 0.008) | 0 → 8 |
| Shop retention, dev (40) | 0 | **40 (100%)** | +100 points | 0 → 40 |
| Shop retention, multi-hop (40) | 0 | **1 (2.5%)** | +2.5 points (not significant) | 0 → 1 |

95% intervals for A1500: SQaLe final 24-39%, BIRD 0.8-3.2%.

## What it shows

- **On new databases from the training source, the gain holds.** SQaLe final 31.3% is close to dev's
  38.9%, a little lower, as expected after choosing A1500 on dev. By query shape (strict, Base → A1500):
  - one-table lookups 4 → 27 of 36;
  - one-table aggregates 0 → 6 of 13;
  - two tables 0 → 10 of 44;
  - 3+ tables 0 → 1 of 19;
  - multi-step 0 → 1 of 32.

  Complex queries stay the weak spot (2 of 51), as on dev.
- **BIRD is far harder for both models.** The source labels 460 of its 500 questions "challenging" (453
  of the 493 kept). A1500 solves 8: 6 challenging and 2 of 8 simple. It runs out of the eight tool steps
  on **235 of 493** BIRD questions, against 39 of 144 on SQaLe final: BIRD schemas are larger and its
  questions need more exploration. Base rarely gets a working query (260 of its 282 submissions fail to run).
- **Shop skills are kept.** A1500 solves all 40 shop retention tasks in the new submission format. The
  three held-out multi-hop families stay essentially unsolved (1 of 40), matching the pilot's earlier 0 of 20.
- **Clean behaviour carries over.** A1500's submissions almost always run (1 failure on SQaLe final, 1 on
  BIRD) and never answer without reading a table (1 on SQaLe final). Base: 68 failing submissions on SQaLe
  final, 260 on BIRD.

## Time and cost (rate $0.006667/min; not invoices)

| Job | Account | Running | Cost |
|---|---|---|---|
| `final-bird-sql-20261109-02` (6af15ac9…) | hf-user | 54,184 s (15.1 h; A1500 step 12.3 h vs 9.2 h forecast) | **$6.02** |
| `final-sqale-sql-20261109-01` (6af179a6…) | hf-user | 20,234 s (5.6 h) | **$2.25** |

The first BIRD attempt (`-01`) stopped at its reference check after 126 s (~$0.01; amendment 12). GPU
peaks stayed at or below 7.4 GiB. Pulled copies: `remote/hf-sql/<account>/<run-id>/`; redacted provider logs:
`runs/hf-sql/<account>/<run-id>/`. Storage approved for at most 30 days.

## Caveats

- These are single greedy runs. The SQaLe final and BIRD gains are far outside chance, but the
  multi-hop retention change is not.
- BIRD questions include their "external knowledge" text when the source provides it (358 of 493).
- The job image runs SQLite 3.40.1 (amendment 12). Both models were scored in the same jobs under the same
  limits, so the comparison is fair.
- A self-training model B may still be added, by its own registration, only if its dev strict is at least 38/95.
