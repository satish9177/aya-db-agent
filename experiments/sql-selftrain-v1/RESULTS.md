# Self-training round (amendment 11): B 40/95, A+ 36/95, both runs PASS (2026-11-10)

Approved by the user ("yes launch both") and launched once each by Claude from clean local commit
`80732a3` on hf-user, one t4-small (Tesla T4) each, ~08:41 UTC 2026-11-10:
- **B (own verified attempts):** job 6af2d8baab34849891aae46b,
  registration SHA256 `154f26224c9c4424949ded201cca00d2f32d629ff58619c3619dffe8882f2a6c`.
- **A+ (matched teacher control):** job 6af2d8d2ab34849891aae4c0,
  registration SHA256 `33761609c9dff75cfef5391eb94670d3d1b381f6e6a99e3b73ea83731595aad1`.

Each uploaded 293 private files (B 200,919,277 bytes, A+ 201,043,090 bytes) before launch.
**No retry, second job, extra upload, push or publication** followed. Machine-readable results:
`RESULTS.json` (written by a local read-only script).

## Headline

| 95 dev tasks, greedy, original tools | Strict | Fallback | Set | Complex (30) | Hard (19) | Ran out of steps |
|---|---:|---:|---:|---:|---:|---:|
| A1500 (starting point) | 37 | 37 | 40 | 1 | 2 | 25 |
| **B** = A1500 + 2 passes over its own 306 verified Q2 attempts | **40** | **40** | **42** | 1 | 1 | 22 |
| A+ = A1500 + 2 passes over 306 matched teacher (gold-path) episodes | 36 | 36 | 39 | 1 | 1 | 21 |

**Decision rule met:** B needed at least 38/95. **B is a candidate for the final evaluation**, through
its own registration and the user's approval (not launched; nothing is approved yet). A+ is a control
and never a finalist.

## Task by task (exact McNemar, two-sided)

| Pair | First only | Second only | p |
|---|---:|---:|---:|
| B vs A+ (the scientific comparison) | 4 | 0 | 0.125 |
| B vs A1500 | 4 | 1 | 0.375 |
| A+ vs A1500 | 3 | 4 | 1.0 |
| B vs A+, complex tasks only | 0 | 0 | 1.0 |

Read plainly:
- Training on its own verified attempts did better than training on the same amount of teacher
  examples (4 tasks to 0). The direction is consistent, but at 95 tasks this is **not statistically
  clear** (p = 0.125).
- B over A1500 (+3 net) is also within noise (p = 0.375). The final evaluation is the real test.
- The same amount of extra teacher data (A+) did nothing (36 vs 37).
- B's gains: three one-table lookups and one 3+-table query. Its one loss: a hard multi-step task.

## What did not change

- **Complex queries:** 1/30 for A1500, B and A+.
- **Recovery after a failed query:** zero strict successes in every arm. B has 23 episodes with a
  failed query and solves none of them; A+ 22, also none; A1500 24, none. This is expected: only 3 of
  B's 306 training attempts contained a recovery.
- **Running out of the eight tool steps** fell a little (25 → 22 for B, 21 for A+), but stays the main failure.

By query shape (strict):

| Shape | A1500 | B | A+ |
|---|---:|---:|---:|
| 1-table lookup | 25 | 28 | 26 |
| 1-table aggregate | 2 | 2 | 2 |
| two tables | 9 | 9 | 7 |
| 3+ tables | 0 | 1 | 1 |
| multi-step | 1 | 0 | 0 |

## Checks (all PASS for both runs)

| Check | B | A+ |
|---|---|---|
| Completed, every step exit code 0 | PASS | PASS |
| Account hf-user, registration SHA in the job equals the launch receipt and the file | PASS | PASS |
| All 401 references re-checked in the job before any model call (0 model calls) | PASS | PASS |
| Fresh encoding in the job equals the frozen match: 49,717 targets per pass | PASS | PASS |
| 40 optimizer updates, none skipped; trainer global step 40 | PASS | PASS |
| GPU memory event after every update | PASS | PASS |
| Dev reloads exactly the final adapter (all tensors match) | PASS | PASS |
| 95 dev episodes, one finite projected position per generated token | PASS | PASS |
| AuditKIT agrees on all 95 and matches the strict count | PASS | PASS |
| No out-of-memory event | PASS | PASS |

Both started from the A1500 step-180 adapter: the training code refuses a continuation whose starting
adapter does not match the frozen hash `69f0f125…` (`src/ayadb/sqltraining.py`), and the registration pins
that file. Both arms saw the same number of trained target tokens (99,434 over two passes) in 612
microbatches. Only the data differed.

| | B | A+ |
|---|---|---|
| Final adapter SHA256 | `90b04823368f50d6ef5ce85c171fffff317b11a3bf43a591cf3952f01b076837` | `fb2542bd9b8812cafd1786aba5cf0a0b15dad13be2c77fd0dca409f100cdf8be` |
| Mean training loss | 0.021 | 0.036 |
| Training / dev step | 46.5 min / 100.6 min | 47.4 min / 93.5 min |
| Running time, cost by rate ($0.006667/min) | 2.48 h, **$0.99** | 2.38 h, **$0.95** |
| GPU peak, training / dev | 7.80 / 5.46 GiB | 7.80 / 5.62 GiB |

B's lower training loss is expected: its targets are the model's own outputs, which it already finds likely.

## Total cost of the self-training round

Q2 sampling $9.11, plus B $0.99, plus A+ $0.95: **$11.05 by rate**, all on hf-user.

## Not done

- B's final evaluation (SQaLe final 144, BIRD Mini-Dev 493, shop retention 40 + 40): needs its own
  registration and explicit approval. At A1500's measured final speeds it would take about 16.8 GPU-hours,
  **about $6.70**.
- No test data was read or used.
