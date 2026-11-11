# SFT pilot: dev-set results

Frozen 40-task dev set, `multistep-close`, greedy, 8 tool rounds, 512 tokens per turn. Episode counts out of 40 unless stated. Malformed = an unparseable final turn, bad arguments, an unknown tool, or text around a call. Fabricated = invented tool output. Recovered = a successful query after the episode's last error, with a verified answer.

## Validity

- `base-nf4` (20261104-163504-base-nf4): dev frozen True, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 1}, AuditKit agreement 40/40
- `sft-nf4-r0` (20261104-181822-sft-nf4-r0): dev frozen True, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `sft-nf4-r1` (20261104-181825-sft-nf4-r1): dev frozen True, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `base-fp16` (20261104-163711-base-fp16): dev frozen True, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 3}, AuditKit agreement 40/40
- `sft-fp16` (20261104-182028-sft-fp16): dev frozen True, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- training: 56 steps, 28.6 min, train loss 0.065, final eval loss 0.008, all losses finite True, episodes sha256 `5cb48a813df0...`

## Overall

| run | verified success | answer seen in SQL | recovered / errored | malformed episodes | fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |
|---|---|---|---|---|---|---|---|---|---|
| `base-nf4` | **2/40** | 9 | 0/12 | 23 (19 final, 6 text) | 6 | 6 (6 max) | 3.5 / 3.0 | 4.5 | 176 |
| `sft-nf4-r0` | **40/40** | 40 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.4 / 4.4 | 5.4 | 140 |
| `sft-nf4-r1` | **40/40** | 40 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.4 / 4.4 | 5.4 | 140 |
| `base-fp16` | **1/40** | 5 | 0/24 | 20 (4 final, 16 text) | 21 | 14 (14 max) | 4.4 / 3.0 | 5.4 | 471 |
| `sft-fp16` | **37/40** | 39 | 0/1 | 0 (0 final, 0 text) | 0 | 1 (1 max) | 4.4 / 4.2 | 5.4 | 143 |

## Verified success by family

| family | `base-nf4` | `sft-nf4-r0` | `sft-nf4-r1` | `base-fp16` | `sft-fp16` |
|---|---|---|---|---|---|
| avg_resolution | 0/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| customer_shipped_year | 0/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| distinct_buyers | 0/5 | 5/5 | 5/5 | 0/5 | 2/5 |
| month_revenue | 0/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| product_price | 0/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| segment_city_count | 2/5 | 5/5 | 5/5 | 1/5 | 5/5 |
| ticket_top_customer | 0/5 | 5/5 | 5/5 | 0/5 | 5/5 |
| top_category_city | 0/5 | 5/5 | 5/5 | 0/5 | 5/5 |

## Per family: `base-nf4`

| family | verified success | answer seen in SQL | recovered / errored | malformed episodes | fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |
|---|---|---|---|---|---|---|---|---|---|
| avg_resolution | **0/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / - | 4.0 | 102 |
| customer_shipped_year | **0/5** | 0 | 0/0 | 5 (4 final, 1 text) | 1 | 0 (0 max) | 1.8 / - | 2.8 | 90 |
| distinct_buyers | **0/5** | 0 | 0/1 | 4 (4 final, 0 text) | 0 | 1 (1 max) | 3.2 / - | 4.2 | 118 |
| month_revenue | **0/5** | 0 | 0/0 | 5 (5 final, 2 text) | 2 | 0 (0 max) | 1.6 / - | 2.6 | 131 |
| product_price | **0/5** | 1 | 0/1 | 4 (1 final, 3 text) | 3 | 0 (0 max) | 1.6 / - | 2.6 | 196 |
| segment_city_count | **2/5** | 0 | 0/0 | 3 (3 final, 0 text) | 0 | 0 (0 max) | 2.4 / 3.0 | 3.4 | 78 |
| ticket_top_customer | **0/5** | 3 | 0/5 | 0 (0 final, 0 text) | 0 | 2 (2 max) | 7.0 / - | 8.0 | 289 |
| top_category_city | **0/5** | 0 | 0/5 | 2 (2 final, 0 text) | 0 | 3 (3 max) | 7.2 / - | 8.2 | 407 |

## Per family: `sft-nf4-r0`

| family | verified success | answer seen in SQL | recovered / errored | malformed episodes | fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |
|---|---|---|---|---|---|---|---|---|---|
| avg_resolution | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / 3.0 | 4.0 | 105 |
| customer_shipped_year | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 5.0 / 5.0 | 6.0 | 139 |
| distinct_buyers | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 6.0 / 6.0 | 7.0 | 184 |
| month_revenue | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 5.0 / 5.0 | 6.0 | 194 |
| product_price | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / 3.0 | 4.0 | 86 |
| segment_city_count | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 3.0 / 3.0 | 4.0 | 79 |
| ticket_top_customer | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.0 / 4.0 | 5.0 | 139 |
| top_category_city | **5/5** | 5 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 6.0 / 6.0 | 7.0 | 197 |

## Paired comparisons (exact McNemar)

- primary: `base-nf4` 2/40 -> `sft-nf4-r0` 40/40; gained 38, lost 0, p = 7.3e-12
- secondary: `base-fp16` 1/40 -> `sft-fp16` 37/40; gained 36, lost 0, p = 2.9e-11
- A/A (`sft-nf4-r0` vs `sft-nf4-r1`, identical runs): 0 success flips, 0 different final answers out of 40

## Decision rule (preregistered, on `sft-nf4-r0`)

40/40 verified successes: **proceed to broader evaluation** (>= 12 proceed, 4-11 investigate, <= 3 pivot).


## What 40/40 does and does not show (post-hoc analysis, 2026-11-05)

Not preregistered; computed from the saved trajectories and `data/sft_pilot/episodes.jsonl`.

- **Same question types, new databases.** All 8 dev families are training families. 25/40 dev
  questions appear verbatim in training (on other databases, with other answers); `sft-nf4-r0`
  also solves the other 15/15.
- **Learned per-family procedures with slot filling.** The final query of every success matches a
  training SQL template once literals are abstracted (40/40; 26/40 verbatim). Tool calls per task
  are constant within each family (3, 4, 5 or 6). The model replays the expert procedure on
  databases it has not seen; it is not composing new SQL.
- **Error recovery is untested.** `sft-nf4` hit no errors (0/0). `sft-fp16` hit one (dev-distinct_buyers-000)
  and looped to the budget. 66 of the 235 training episodes demonstrate a recovery (37 wrong table,
  29 wrong column; the mistake turn is masked from the loss).
- **A/A = 0 flips** shows greedy decoding is deterministic on this stack, not robustness.
- **fp16 misses** (3, all `distinct_buyers`): two answered 58/59 against gold 15/16 with the value
  taken from a query result (wrong query), one looped after 3 errors. The adapter was trained on
  the nf4 base.
- Supported now: on unseen databases with the trained question types, SFT on 235 programmatic
  trajectories takes Tiny Aya Global from 2/40 to 40/40 verified success (exact McNemar
  p = 7e-12) and removes malformed calls, fabricated tool output and loops. Not yet supported:
  new question types (the sealed test's 3 held-out families), error recovery, external SQL data.
