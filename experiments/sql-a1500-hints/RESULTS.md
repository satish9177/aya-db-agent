# A1500 with error-feedback tools: PASS (valid run), 38/95, not adopted (2026-11-08)

Job 6af0a9240e0338d1bb84c64a,
hf-user, one t4-small (Tesla T4). Launched once from clean local commit `e6b4269` with the exact
approved registration SHA256 `02984087902a41ba11d6c62e33743d5342e33788117fc0f455c9916be9154d6a`
(Claude's launch call was blocked by the auto-mode permission check; the launch ran from this machine).
The approved private upload (104 files, 157,283,029 bytes) completed before launch. **No retry, second
job, extra upload, push or publication** followed. Test only; no training. Machine-readable results:
`RESULTS.json`.

## Headline

| A1500, same adapter, 95 dev tasks, greedy | Strict | Fallback | Set | Complex (30) | Ran out of steps |
|---|---:|---:|---:|---:|---:|
| Original tools (job 6af0139b…, pinned comparator) | 37 | 37 | 40 | 1 | 25 |
| **`column-hints-v1` tools** | **38** | **38** | **41** | **1** | **23** |

**Decision rule not met:** the preregistered bar was 42/95 (five points over 37); the hints gave +1.05
points. **The final evaluation keeps the original tools.**

Task by task, the hints fixed exactly one task and broke none (1 to 0). The fixed task is a two-table
query: after `no such column: rs.tipoaccion` plus the hint listing `accion`, the model rewrote the
query with `accion` and submitted the right answer. Complex tasks are unchanged (the same 1 of 30).

## Why it barely helped: the model mostly ignores the feedback

| What the tools said | What the model did next |
|---|---|
| A hint listing the real columns (26 hints; 25 followed by another call) | sent the **identical broken query again 20 times**, changed it 5 times |
| "This exact query already failed" (38 refusals; 26 followed by another call) | `describe_table` 21 times, the identical query 4, a new query 1 |
| After re-checking the table following a refusal (17 times) | the same broken query again 7, another `describe_table` 9, a new query **1** |

So the refusal works as an instruction: the model stops and looks at the table. But it cannot turn
what it sees into a corrected query. Of the 24 episodes with a failed query, only 2 later ran a working
query and 1 ended correct (original tools: 0). The repeated calls just move into refusals and extra
`describe_table` calls until the eight steps run out: 23 episodes, against 25 before.

This matches the training data. In the 1,500 demos, all 377 repairs jump straight from the error to the
correct query, and none rewrites a query using what an error or a table description says. The model
learned to look and to retry, not to correct.

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, registration SHA, private upload | PASS | Launch/upload receipts; `environment.json` has the same SHA |
| Every dev reference passes before model use | PASS | `oracle-precheck.json`: 95 checked, 0 model calls |
| Exact reload of the pinned A1500 step-180 adapter | PASS | `checkpoint-reload.json`: 504 tensors, all match, SHA `69f0f125…` |
| 95 dev episodes, one finite projected position | PASS | 24,559 checked forwards = generated tokens; `logits_to_keep=1` |
| AuditKIT agrees with every saved score | PASS | 95/95; AuditKIT successes = 38 strict |
| New tools recorded in the summary and every trajectory | PASS | 26 hints, 38 refused repeats |
| Memory events, no OOM | PASS | peak 10.14 / 10.42 GiB, same as the original-tools run |

**Determinism check:** all 71 episodes with no failed query under the original tools replayed exactly
the same call log (0 differ). So the hints are the only thing that changed the other 24, and the +1 is
not noise from the GPU.

## Diagnostics

- **By shape**, strict original → hints: lookup 25 → 25, aggregate 2 → 2, two tables 9 → 10, 3+ tables
  0 → 0, multi-step 1 → 1. Ran out of steps: 5 → 4, 0 → 0, 5 → 4, 3 → 3, 12 → 12.
- **Failed tool calls** fell from 112 to 74: refused repeats are not executed, and several episodes spend
  their steps on `describe_table` instead.
- **Prompts:** the longest prompt was 7,018 tokens in both runs (the same episode, which has no errors), and
  the memory peak is identical.
- Many failing episodes spend 3-5 of their 8 steps describing tables before the first query, so only
  2-3 steps are left for any repair.

## Time and cost

- Created 16:53:56 UTC, started 16:54:01, finished 18:32:16 on 2026-11-08. Running **5,895 s (1.64 h)**;
  the dev step 5,781 s. Forecast was ~1.8 h.
- Rate $0.006667/min: **$0.66** (prorated $0.6551). Rate-based, **not an invoice**. Cap was $2.00.
- Storage: inputs 157 MB plus a 2 MB output snapshot in the private bucket, approved **through 2026-12-08**.
  Pulled copy: `remote/hf-sql/hf-user/a1500-hints-sql-20261108-01/`; redacted provider log/metadata:
  `runs/hf-sql/hf-user/a1500-hints-sql-20261108-01/`.

## What this does and does not show

Better error messages alone do not fix A1500's stuck loop: it can be made to stop and look again, but it
rarely uses what it sees. The one clean fix (a simple column rename) shows the information can help
when the model attends to it. The limit is in what the model was trained to do after an error, not in
what the tools tell it. So the next improvement has to come from training examples where the model
corrects its own query from the error and the table description, for example verified self-generated
attempts (Q2) or rebuilt repair demos. Each needs its own registration and approval. Hints were
designed on these same dev tasks; the +1 is not a final-split result.
