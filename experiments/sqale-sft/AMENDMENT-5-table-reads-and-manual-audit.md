# Amendment 5: table-read guard and diagnostic manual audit (2026-11-07)

Recorded after the approved G0 launch and before any broader dev screen. The sole approved job is
`g0-sql-agenttune-20261107-01`, registration SHA256
`84799ec8cbdd3a57602b09084e3bf8aec1f82f572b5baabeb71a5d51828c8c81`, job
`6aef3ba90e0338d1bb835c85`, launched from clean commit `413dff8`. Its uploaded source, registration,
scorer and outputs remain unchanged. This amendment authorizes no additional job, retry or upload.
Free implementation, tests, local commit and a 500-stage draft are explicitly requested by the user.

## Literal-answer guard: table-read-v1

For future SQL evaluations, strict success requires a valid explicit submission, a correct full query
result under the existing ordering/tolerance rules, **and a read of at least one existing user database
table**. SQLite's native authorizer must report SQLITE_READ on a persistent table from sqlite_master;
metadata tables named sqlite_* do not satisfy the guard. A constant SELECT, VALUES expression or
constant-only CTE cannot count as strict success even when its values happen to equal gold. COUNT(*)
and views/subqueries that read user tables remain eligible, including legitimate empty results.

Use the same database, executor, limits, prompt format and tools. Verify each reference query meets
the table-read requirement before relying on it; log every reference decision and fail on a missing
read rather than silently removing questions or rewriting gold. A native EXPLAIN preparation suffices
to inventory authorizer reads without using final-set model predictions. It proves a table reference,
not that every returned value causally depends on its contents; lucky passes remain possible.

For **every evaluation**, record counts of all submitted queries with zero user-table reads, including
rejected/repeated attempts, execution errors and successful no-table statements separately. Also
report the number of episodes whose accepted submission reads no table. Unsubmitted episodes are
not counted as literal submissions. Set-style and last-query fallback retain their existing result
semantics as diagnostics, with table-read status explicitly recorded; they cannot qualify self-training.

G0 keeps its registered scores. After pulling its logs, add a separate local table-read diagnostic,
including zero/not-run counts if inference never starts. Do not overwrite G0 outputs, registration,
the uploaded snapshot or its registered pass conditions. Future AuditKIT strict predicates must include
the new guard, and summaries must name the scorer policy so old and new results cannot be conflated.

## Self-training eligibility

When the Q2 filter is built, accept only guarded **strict successes that read a user table**, with
**zero malformed calls and zero invented tool output**. Recompute eligibility from recorded calls and
assistant turns, never trust an upstream success boolean alone. Missing/incomplete logs or mismatched
tool observations fail closed. Preserve the existing cap, deduplication, sampling-accounting and
post-drop size-gate rules. No Q2 sampling or new teacher is authorized by this amendment.

## Manual audit after the first dev screen

After the first full dev screen, manually read **20 failures and 10 successes per evaluated model**.
Choose separately within each model/outcome using a fixed seed42 shuffle of sorted trajectory IDs;
if fewer exist, inspect all and report the shortfall without replacement. Freeze the selected IDs before
reading their contents. Read the question, schema/tool observations, query and reference privately.
Label every selected episode exactly once as **really wrong**, **correct but scorer too strict**, or
**lucky pass**, with a short explanation. A correct grounded success belongs in the additional explicit
label **correct and scorer agrees**; the three error labels alone cannot describe honest successes.
Report all four counts by model and sampled outcome, plus sample availability and the selected IDs.
This descriptive audit changes **no scores, model choice thresholds or gold answers**. Any later scorer
change requires another prospective amendment, never retrospective score repair or final-set tuning.
