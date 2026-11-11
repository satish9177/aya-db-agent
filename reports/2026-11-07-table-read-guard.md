# Table-read guard and post-G0 local work (2026-11-07)

The approved G0 launched first from clean commit `413dff8`, with no edits before launch. It failed on
the longest example's first forward due to CUDA OOM; no retry or extra job was started. See
[G0 results](../experiments/sql-g0/RESULTS.md). Its registration and uploaded snapshot remain frozen.
Only the requested free scorer/filter/audit implementation, tests and blocked 500-stage draft followed.

Amendment 5 adds **table-read-v1** alongside the unchanged historical scorer/tools/data. Future strict
success requires explicit submission, full-result equality and a native SQLite authorizer read of an
existing user table. Constant-only SELECT/VALUES/CTEs and sqlite_* metadata-only statements cannot
pass strict. COUNT(*), views/subqueries reading base tables and legitimate empty results still work.
Set and fallback remain labelled diagnostics; only guarded strict success can qualify self-training.

`sqlguard.py` implements the scorer and an offline AuditKIT predicate using the same guard.
`sqlrunner.py` checks references before model loading, writes every submitted query's table-read/
execution status, and reports total/valid/error no-table submissions and affected accepted episodes.
`runinputs.verify_runtime_oracles` now also requires reference reads. G0 used the uploaded old versions;
these local changes cannot modify its frozen source or registered checks.

`selftrainfilter.candidate_eligibility` is a predicate for the future Q2 filter, not a sampling/build run.
It recomputes strict success from actual calls and the database, requires table reads, rejects malformed
calls/invented outputs, checks assistant calls against the execution log and observations, and rejects
missing/incomplete records. It does not trust an upstream strict=True field. No Q2 model samples exist.

## Reference and submission inventories

Native EXPLAIN/authorizer inventory checks every active reference without model predictions or changes
to frozen data. Private per-ID/source/database/reference-hash log:
`data/scorer-audit-v1/references/reference-reads.jsonl`; summary is alongside it.

| Active source | References | Missing table reads / other failures |
|---|---:|---:|
| Eligible full-stage training | 4,887 | 0 |
| Dev | 95 | 0 |
| Q2 | 579 | 0 |
| SQaLe final | 144 | 0 |
| Retention dev / mh | 40 / 40 | 0 / 0 |
| BIRD final | 493 | 0 |
| **Total** | **6,278** | **0** |

This checks reference eligibility, not model quality or causal dependence of answers on data. A query
can read a table while returning lucky constants; the guard does not prove SQL equivalence. The
manual audit and preregistered perturbation diagnostic address different remaining weaknesses.

G0's separate local diagnostic is `data/scorer-audit-v1/g0-diagnostics/summary.json`. All three planned
conditions have **0 episodes, 0 submissions, 0 no-table submissions**, explicitly **not run** after OOM.
No registered scores or pulled output files are overwritten. Future evaluation summaries include the
same counts even when all are zero, separating actual zero observations from missing inference.

## Manual audit and validation

After the first full dev screen, freeze sorted-ID/seed 42 samples of 20 failures and 10 successes per
model, or all available with shortfalls. `manualaudit.py` prevents resampling, leaves labels blank for
actual manual reading and requires explanations before counts are reported. The requested three error
labels remain: really wrong / correct but scorer too strict / lucky pass. To cover an ordinary correct
grounded success, the preregistration includes the additional label **correct and scorer agrees**.
No manual review was performed on nonexistent G0 inference, and this audit changes no scores.

Final suite: **103 passed**, 14 upstream torch JIT deprecation warnings, 31.99s:
`.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/pytest-table-read-final --tb=short`.
Tests cover constant/VALUES/CTE/metadata answers, base-table/view/count/empty queries, errors/repeated
submissions, scripted AgentTune-to-AuditKIT guarded scoring, strict filter rejection and manual sample
determinism/shortfalls. Scripted tests use no model training or paid service.

An initial full-suite attempt had 102 passes and one DNS failure: an older launcher test constructed
a real client for a read-only whoami before its fake upload client was injected. Both launch calls now
receive the fake explicitly, and the fixture forbids constructing a real HF client. The final 103 tests
pass without that dependency; no real job/write was performed by those tests.

The [500-stage draft](../experiments/sqale-500/REGISTRATION-DRAFT.md) is blocked/non-runnable: G0 has
zero successful training steps or inference episodes, so neither 500 nor 1,500 has a defensible measured
completed-cost forecast. No memory workaround, extra input upload, paid execution, push or publication
was performed. Commit the authorized local work, then stop.
