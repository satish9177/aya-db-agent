# SFT pilot: sealed test results

Preregistered in `experiments/sft-pilot/SEALED-TEST.md` before the split was generated. Same protocol as dev (`multistep-close`, greedy, 8 tool rounds, 512 tokens per turn), nf4 base, the pilot adapter unchanged. Column definitions as in `RESULTS-dev.md`.

## Validity

- `test-base-nf4` (20261104-200526-test-base-nf4): split test, 97 tasks, tasks sha256 `dbe99529da23222b3093552cdd099af9830043531d85ce83c8df5c4366745d14`, adapter `none`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 9}, AuditKit agreement 97/97
- `test-sft-nf4` (20261104-200531-test-sft-nf4): split test, 97 tasks, tasks sha256 `dbe99529da23222b3093552cdd099af9830043531d85ce83c8df5c4366745d14`, adapter `6f2827ae9b28`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 97/97
- both runs saw the same tasks: True; tasks are the pinned split: True

## Overall

| run, tasks | verified success | answer seen in SQL | recovered / errored | malformed episodes | fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |
|---|---|---|---|---|---|---|---|---|---|
| `test-base-nf4`, held-out families | **0/20** | 0 | 0/6 | 14 (8 final, 12 text) | 12 | 6 (6 max) | 3.2 / - | 4.2 | 302 |
| `test-sft-nf4`, held-out families | **0/20** | 10 | 0/11 | 0 (0 final, 0 text) | 0 | 11 (11 max) | 6.8 / - | 7.8 | 326 |
| `test-base-nf4`, shared families | **5/77** | 13 | 0/26 | 44 (28 final, 20 text) | 20 | 18 (18 max) | 3.8 / 3.0 | 4.8 | 224 |
| `test-sft-nf4`, shared families | **77/77** | 77 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.5 / 4.5 | 5.5 | 141 |

## Verified success by family

| family | held out | `test-base-nf4` | `test-sft-nf4` |
|---|---|---|---|
| cancel_leader_tickets | yes | 0/2 | 0/2 |
| largest_order | yes | 0/12 | 0/12 |
| segment_aov | yes | 0/6 | 0/6 |
| avg_resolution |  | 0/5 | 5/5 |
| customer_shipped_year |  | 0/12 | 12/12 |
| distinct_buyers |  | 0/12 | 12/12 |
| month_revenue |  | 0/12 | 12/12 |
| product_price |  | 0/12 | 12/12 |
| segment_city_count |  | 5/12 | 12/12 |
| ticket_top_customer |  | 0/4 | 4/4 |
| top_category_city |  | 0/8 | 8/8 |

## Per family: `test-sft-nf4`

| family | verified success | answer seen in SQL | recovered / errored | malformed episodes | fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |
|---|---|---|---|---|---|---|---|---|---|
| cancel_leader_tickets | **0/2** | 2 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.0 / - | 5.0 | 130 |
| largest_order | **0/12** | 4 | 0/10 | 0 (0 final, 0 text) | 0 | 10 (10 max) | 7.6 / - | 8.6 | 385 |
| segment_aov | **0/6** | 4 | 0/1 | 0 (0 final, 0 text) | 0 | 1 (1 max) | 6.3 / - | 7.3 | 274 |
| avg_resolution | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / 3.0 | 4.0 | 105 |
| customer_shipped_year | **12/12** | 12 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 5.0 / 5.0 | 6.0 | 139 |
| distinct_buyers | **12/12** | 12 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 6.0 / 6.0 | 7.0 | 185 |
| month_revenue | **12/12** | 12 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 5.0 / 5.0 | 6.0 | 194 |
| product_price | **12/12** | 12 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / 3.0 | 4.0 | 86 |
| segment_city_count | **12/12** | 12 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / 3.0 | 4.0 | 79 |
| ticket_top_customer | **4/4** | 4 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.0 / 4.0 | 5.0 | 138 |
| top_category_city | **8/8** | 8 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 6.0 / 6.0 | 7.0 | 197 |

## Paired comparisons (exact McNemar)

- **primary**, held-out families: `test-base-nf4` 0/20 -> `test-sft-nf4` 0/20; gained 0, lost 0, p = 1
- shared families: `test-base-nf4` 5/77 -> `test-sft-nf4` 77/77; gained 72, lost 0, p = 4.2e-22
- shared, question verbatim in training: `test-base-nf4` 5/52 -> `test-sft-nf4` 52/52; gained 47, lost 0, p = 1.4e-14
- shared, question not in training: `test-base-nf4` 0/25 -> `test-sft-nf4` 25/25; gained 25, lost 0, p = 6e-08

## Template replay (`test-sft-nf4`)

Episodes whose last query matches a training query once whitespace, case and literals are abstracted.

- held-out families: 0/20 episodes replay a training template (0 of them verified); 0 verified successes with a new query
- shared families: 77/77 episodes replay a training template (77 of them verified); 0 verified successes with a new query

## Interpretation (preregistered)

Held-out families: 0/20 -> 0/20, p = 1: **no evidence of transfer beyond the trained question types; the claim stays within them**.

## Reading (post-hoc, 2026-11-05; the preregistered verdict above stands)

- **In-family transfer holds on a fresh database and fresh questions**: 77/77 on the shared
  families, including 25/25 questions that never appear in training (base 5/77).
- **No transfer to new question types**: 0/20 on the held-out multi-hop families, the same as the
  base model. Only 20 held-out tasks were generated (2 + 12 + 6), so the test is small, but the
  effect is not: no held-out task succeeds in either run.
- **What does transfer is protocol discipline**: on the held-out families the trained model makes
  0 malformed episodes and 0 fabricated tool outputs (base: 14 and 12 of 20). It writes new
  queries there (0/20 replay a training template), assembled from trained fragments.
- **How it fails**: 11/20 held-out episodes hit an SQL error (mostly a wrong table alias,
  `r.order_no`) and none recovers; they re-describe tables, repeat the query and hit the step budget.
  The other 9 answer after one query that skips a hop or a filter (for example the first hop's
  customer id instead of that customer's ticket count, or total revenue instead of the average order
  value). The 66 recovery demonstrations in training (wrong table or column name only) did not
  produce recovery from other errors.
- **Claim supported by independent evidence**: SFT on 235 programmatic trajectories makes Tiny Aya
  Global a reliable tool user for the trained question types on unseen databases and unseen
  questions, and removes malformed calls and fabricated tool output even on unseen question types.
  It does not teach it to solve new multi-hop question types or to recover from new kinds of SQL
  errors.
