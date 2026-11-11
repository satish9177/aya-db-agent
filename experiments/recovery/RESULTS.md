# Recovery experiment: results

Preregistered in `experiments/recovery/PREREGISTRATION.md`. Arm A = the pilot adapter (discovery-phase mistakes), arm B = the same 235 episodes with query-phase mistakes at the same 66 positions, trained with the same settings. Protocol as on dev (`multistep-close`, greedy, 8 tool rounds, 512 tokens per turn, nf4). Column definitions as in `experiments/sft-pilot/RESULTS-dev.md`.

## Validity

- `drift-sft-a` (20261105-064805-drift-sft-a): split drift, 40 tasks, pinned tasks True, adapter `6f2827ae9b28`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `drift-sft-b` (20261105-071942-drift-sft-b): split drift, 40 tasks, pinned tasks True, adapter `93f9557e6991`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `sft-nf4-r0` (20261104-181822-sft-nf4-r0): split dev, 40 tasks, pinned tasks True, adapter `6f2827ae9b28`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `dev-sft-b` (20261105-071944-dev-sft-b): split dev, 40 tasks, pinned tasks True, adapter `93f9557e6991`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `mh-sft-a` (20261105-064815-mh-sft-a): split mh, 40 tasks, pinned tasks True, adapter `6f2827ae9b28`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- `mh-sft-b` (20261105-071956-mh-sft-b): split mh, 40 tasks, pinned tasks True, adapter `93f9557e6991`, orchestration {'orch_result_mismatch': 0, 'orch_parser_miss': 0, 'orch_boundary': 0, 'orch_truncated': 0}, AuditKit agreement 40/40
- arm B training: 56 steps, 27.8 min, train loss 0.063, final eval loss 0.006, all losses finite True, episodes sha256 `d0bf1bb35f55...` (pinned True)

## Overall

| run, split | verified success | answer seen in SQL | recovered / errored | malformed episodes | fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |
|---|---|---|---|---|---|---|---|---|---|
| `drift-sft-a`, drift | **40/40** | 40 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.4 / 4.4 | 5.4 | 141 |
| `drift-sft-b`, drift | **40/40** | 40 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.4 / 4.4 | 5.4 | 141 |
| `sft-nf4-r0`, dev | **40/40** | 40 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.4 / 4.4 | 5.4 | 140 |
| `dev-sft-b`, dev | **40/40** | 40 | 0/0 | 0 (0 final, 0 text) | 0 | 0 (0 max) | 4.4 / 4.4 | 5.4 | 140 |
| `mh-sft-a`, mh | **0/40** | 27 | 0/17 | 0 (0 final, 0 text) | 0 | 12 (12 max) | 6.5 / - | 7.5 | 296 |
| `mh-sft-b`, mh | **1/40** | 37 | 0/1 | 0 (0 final, 0 text) | 0 | 2 (2 max) | 6.3 / 7.0 | 7.3 | 233 |

## Verified success by family

drift:

| family | `drift-sft-a` | `drift-sft-b` |
|---|---|---|
| avg_resolution | 5/5 | 5/5 |
| customer_shipped_year | 5/5 | 5/5 |
| distinct_buyers | 5/5 | 5/5 |
| month_revenue | 5/5 | 5/5 |
| product_price | 5/5 | 5/5 |
| segment_city_count | 5/5 | 5/5 |
| ticket_top_customer | 5/5 | 5/5 |
| top_category_city | 5/5 | 5/5 |

mh:

| family | `mh-sft-a` | `mh-sft-b` |
|---|---|---|
| cancel_leader_tickets | 0/4 | 0/4 |
| largest_order | 0/18 | 1/18 |
| segment_aov | 0/18 | 0/18 |

## After an error

Every failed call, by what the agent did next. Edited = a different query on the next call.

| run | failed calls | edited, ran | edited, failed | repeated | schema lookup | answered | stopped | other | error kinds |
|---|---|---|---|---|---|---|---|---|---|
| `drift-sft-a` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `drift-sft-b` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `sft-nf4-r0` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `dev-sft-b` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |  |
| `mh-sft-a` | 26 | 0 | 5 | 1 | 18 | 0 | 2 | 0 | no such column 22, no such table 4 |
| `mh-sft-b` | 3 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | no such column 3 |

## Memory vs schema on drift

First query of each episode: does it replay a training query (training column names, literals abstracted)? On the drift database such a query fails if it names a renamed column. (Corrected 2026-11-05 after the runs: the preregistered wording said it always fails, but a two-hop family's first hop uses no renamed column.)

- `drift-sft-a`: 5/40 first queries replay a training query, 0 of them with a renamed column; 0 episodes run no query
- `drift-sft-b`: 5/40 first queries replay a training query, 0 of them with a renamed column; 0 episodes run no query

## Paired comparisons (exact McNemar)

- **primary**, drift: `drift-sft-a` 40/40 -> `drift-sft-b` 40/40; gained 0, lost 0, p = 1
- guard, frozen dev: `sft-nf4-r0` 40/40 -> `dev-sft-b` 40/40; gained 0, lost 0, p = 1
- secondary, new multi-hop: `mh-sft-a` 0/40 -> `mh-sft-b` 1/40; gained 1, lost 0, p = 1

## Interpretation (preregistered)

Drift, arm A 40/40 -> arm B 40/40, p = 1: **no evidence of a difference between the two kinds of recovery demonstration**.
How (mechanism): recovered / errored episodes, arm A 0/0, arm B 0/0; successes without any error, arm A 40, arm B 40.
Drift exercised recovery in fewer than 10 episodes per arm (0 and 0): the agent read the renamed schema instead of failing, so drift does not test recovery.
Frozen-dev guard (arm B >= 38/40, no significant loss): **passed** (40/40, lost 0, gained 0).
Arm B does not replace the pilot adapter as the submission model.
New multi-hop (secondary, descriptive): 0/40 -> 1/40, p = 1.

## Reading (post-hoc, 2026-11-05; the preregistered verdict above stands)

- **The primary question was not tested.** Drift produced no SQL error in either arm (0/40 and
  0/40), so the preregistered rule applies: drift measured schema grounding, not recovery. The design
  assumed the pilot model writes queries from memory, and it does not. In every drift episode both
  adapters describe the tables first, the descriptions show the renamed columns (40/40), and every
  episode's queries use the new names (40/40).
- **The two adapters behave identically where they were compared.** On drift and on the frozen dev
  set, arm A and arm B make the same tool calls and give the same final answer in 80/80 episodes.
  Different error context in 66 training episodes, with identical trained turns, changed nothing on
  paths without an error. The arms differ only on mh (24/40 episodes differ).
- **Independent evidence for the pilot adapter: schema grounding survives column renames.** Arm A
  scores 40/40 on drift, where five columns used by every family's training query have new names: it
  takes column names from `describe_table`, not from training. Limits: these are the frozen dev
  questions (training families, 25/40 verbatim in training), and only column names changed, not
  tables, types or wording.
- **New multi-hop question types still fail (secondary, descriptive).** 0/40 and 1/40 on three new
  databases, matching the sealed test (0/20). Most failures are wrong answers, not errors:
  - `segment_aov` (18/18 in both arms): one query that returns total revenue instead of the average
    order value, i.e. a training revenue query.
  - `cancel_leader_tickets` (4/4 in both arms): one query, then an answer from the wrong hop; neither
    arm queries the chosen customer's tickets.
  - `largest_order` (18): arm A hits an SQL error in 17/18 (15 an undefined table alias, as on the
    sealed test), recovers in none, and runs out of tool rounds in 12. Arm B errs in 1/18: it lists
    the customer's shipped orders, sums revenue over all of them and answers an order number. Neither
    arm ever compares revenue per order. Arm B's one success answers the first listed order, which
    happened to be the largest; we read it as chance.
- **Recovery is still not shown.** Across the 240 episodes, arm A recovered in 0 of 17 errored
  episodes and arm B in 0 of 1. Arm B's lower error count on `largest_order` comes from writing
  simpler queries, not from repairing failed ones; from one family in one run we do not claim it
  as an effect.
- **Decision (preregistered)**: the primary is not positive, so arm B is not adopted and the pilot
  adapter stays the submission model.
