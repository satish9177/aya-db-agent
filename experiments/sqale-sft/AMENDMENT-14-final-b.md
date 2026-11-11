# Amendment 14: final evaluation of self-training model B (2026-11-10)

Written after the B and A+ dev screens (`experiments/sql-selftrain-v1/RESULTS.md`) and before any B
final-data use. Amendments 11-13 are unchanged. **No upload, job or push is approved by this amendment.**
Each registration needs the user's explicit approval of its SHA256, uploads and storage.

## Why B is evaluated

Amendment 11 fixed the rule in advance: B joins the final evaluation, by its own registration, only if
its dev strict score is at least 38/95. B scored **40/95** (A1500 37, A+ 36). A+ is a control and is
never a finalist.

**Frozen selection** (`experiments/sql-final-b/SELECTION.json`), made from dev results only: **B**, adapter
`90b04823…` (step 40, from A1500 plus two passes over its own 306 verified Q2 attempts), job
6af2d8baab34849891aae46b. No final, BIRD or retention task has been read by B or used for any decision.
Base and A1500 are not re-run: their final results already exist (`experiments/sql-final-v1/RESULTS.md`),
from the same tasks, protocol, image and GPU type.

## Submitted model, fixed now from dev only

The hackathon submission model is **B**, the best model on dev (40/95). This is decided before any B
final result exists, so the final sets are never used to choose a model. A1500 and B final results are
both reported as they are, whichever is higher. If the user prefers a different choice, this amendment
is revised before any launch.

## Protocol (identical to A1500's final)

Each set is evaluated once with greedy decoding: eight tool steps, 512 new tokens per turn, 8,192
context, the original tools, and one finite projected position. Scoring is table-read-v1 strict
(primary) plus set and fallback, with AuditKIT agreement. Every reference is checked in the job before
any model use, with scoring limits of 60 s and 2,000M operations (amendment 12). The B steps reload
exactly the step-40 adapter.

**Two jobs**, each within the remaining credit the user reported on 2026-11-10 ($6.69 and $3.30):
- `final-b-bird-sql-20261110-01`: B on the 493 BIRD Mini-Dev questions.
  - Expected ~12.4 h, ~$4.95 (A1500 took 12.3 h here, and B's dev screen was 5% faster than A1500's).
  - Hard timeout 15 h; budget $6.11 including a $0.10 storage reserve.
- `final-b-sqale-sql-20261110-01`: B on SQaLe final (144) and shop retention dev (40)
  and multi-hop (40).
  - Expected ~4.5 h, ~$1.80 (A1500 took 4.4 h).
  - Hard timeout 7 h; budget $2.91.

Each step's own timeout is twice A1500's measured time on that set, capped at the job timeout.

## Reporting

For every set: B strict (primary), set and fallback, next to Base and A1500 on the same tasks. B vs A1500
is shown task by task (exact McNemar), along with complex queries and running out of steps. Results are
reported whatever they are. Nothing is selected, tuned or re-run on final data. There are no automatic
retries or extra jobs.
