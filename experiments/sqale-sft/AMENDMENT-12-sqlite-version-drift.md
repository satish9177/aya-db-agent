# Amendment 12: SQLite version drift between local preparation and the job image (2026-11-09)

Written after the first final-BIRD job failed and before any model saw final or Q2 data. Amendment 11 is
unchanged; this amendment adds to it. **No upload, job or push is approved by this amendment.**

## What happened

`final-bird-sql-20261109-01` (job 6af1536ce6320c66fad7d6ea, approved and launched once from `1da1bc3`)
stopped at its pre-model reference check after 126 s (about $0.01): `bird:219 interrupted`. No model
was loaded, and nothing was scored.

## Cause

The job image runs **SQLite 3.40.1**, while every dataset was prepared and verified locally on **3.53.1**.
A local Python 3.11.1 build (SQLite 3.40.0, the same 3.40 series) re-ran all 1,394 references (493
BIRD and 901 SQaLe dev, final, retention and Q2) under both versions:

- **Operation counts.** Under 3.40, four BIRD references need far more SQLite operations than under 3.53,
  because the older planner is weaker:

  | Reference | Under 3.40 | Under 3.53 |
  |---|---:|---:|
  | `bird:229` | 1,193M operations (15.5 s) | 4.8M |
  | `bird:219` | 627M (17.8 s) | 12.5M |
  | `bird:486` | 334M | 1.6M |
  | `bird:279` | 73M | 0.4M |

  The scorer allowed 50M, the same cap the agent's tools use, so the job stopped.
- **Results.** 27 references return slightly different floating-point values (for example 407.90000000000003
  vs 407.8999999999999). The scorer's tolerant comparison treats them as equal. Two Q2 references return
  genuinely different answers, because their result depends on how ties are broken:
  `schema_014507_s2_q7_f5426b14a9` (one of several tied people) and `schema_009445_s1_q4_ac28623f95`
  (`GROUP_CONCAT` order).

Earlier dev, training and Q2-free runs were unaffected. All their references are small: at most 0.61M
operations and 0.02 s under 3.40.

## Fix

- **Scoring limits** (`src/ayadb/scoringlimits.py`): scoring executions now allow **60 s and 2,000M
  operations** (amendment 11 set 30 s and left 50M). They still cover references, submitted and fallback
  queries, ordering probes and the pre-model reference check. The agent's tools keep 5 s and 50M operations.
- **Q2:** the two tie-dependent questions are excluded from sampling, leaving **577** questions in new
  slices `data/sqale/q2-sampling-v2/`, before any model use on Q2. `RUN-POLICY-v1.json` is unchanged.
- **New registrations:**
  - `final-bird-sql-20261109-02` on hf-user supersedes `-01`, whose approval is used up.
  - `final-sqale-sql-20261109-01` and the three sampling parts are rebuilt with these limits, for the second
    account once it is logged in.

## Verification (local, no model)

- The project's real pre-model reference check, run under SQLite 3.40.0 with the new limits, passes for
  all 493 BIRD (105 s), 144 SQaLe final and 80 retention references. It passes for Q2 once the two
  tie-dependent questions are excluded.
- Re-scoring all 570 earlier dev episodes with the new limits gives 0 differences.

## Notes

- The agent's own tools run SQLite 3.40.1 inside every job. That is the same for every model compared in
  the jobs, including all earlier dev screens.
- On the three heaviest BIRD questions, a query equivalent to the reference cannot run within the agent's
  50M-operation tool cap, so the agent cannot preview it. It can still submit it. This applies equally to
  Base and A1500.
- The `-01` upload (1.64 GB) stays in the private bucket (the launcher never deletes), so `-02` doubles BIRD
  storage to about 3.3 GB for at most 30 days.
