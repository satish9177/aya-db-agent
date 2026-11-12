# Closed-model baseline v1: gpt-6-luna scores 95/144 on SQaLe final, far ahead of B (2026-11-12)

The user approved the run on 2026-11-12 ("yes none is better plan, start"). It ran once, locally, from
registration `7a1ae866…`, and was not retried. Machine-readable results: `RESULTS.json`, written by the
read-only script `scripts/closed_baseline_results.py`.

| | |
|---|---|
| Status | completed: 144/144 episodes, 740 API calls, 19.3 minutes |
| Spend | **$0.068** (cap $1.50) |
| AuditKIT | agrees with all 144 scores |
| Model reported by the API | `gpt-6-luna`, reasoning effort `none` |
| Parameters | temperature 0 accepted. `stop` was rejected and dropped under the preregistered rule; our own stop rules still applied. |

## Headline (strict, same 144 tasks)

| Model | Strict | 95% CI | Set |
|---|---:|---|---:|
| Base (untuned Tiny Aya) | 4 (2.8%) | 1.1–6.9% | 5 |
| A1500 | 45 (31.3%) | 24.2–39.2% | 51 |
| B (submitted) | 44 (30.6%) | 23.6–38.5% | 51 |
| **gpt-6-luna** | **95 (66.0%)** | 57.9–73.2% | 99 |

**Task by task against B:** gpt-6-luna solves 57 tasks that B misses, and B solves 6 that gpt-6-luna
misses (exact McNemar p = 1.6e-11).

## By query shape (strict)

| Shape | n | Base | A1500 | B | gpt-6-luna |
|---|---:|---:|---:|---:|---:|
| 1-table lookup | 36 | 4 | 27 | 29 | 31 |
| 1-table aggregate | 13 | 0 | 6 | 7 | 13 |
| two tables | 44 | 0 | 10 | 7 | 26 |
| 3+ tables | 19 | 0 | 1 | 0 | 13 |
| multi-step | 32 | 0 | 1 | 1 | 12 |

## Behaviour

| | Base | A1500 | B | gpt-6-luna |
|---|---:|---:|---:|---:|
| Submitted a query | 74 | 104 | 98 | 131 |
| Ran out of the eight tool steps | 5 | 39 | 46 | 10 |
| Submitted query failed to run | 68 | 1 | 0 | 0 |
| Malformed turns | 38 | 1 | 0 | 38 |
| Model calls per question (mean) | 2.0 | 6.0 | 6.2 | 5.1 |

**About gpt-6-luna's 38 malformed turns:**
- They are a format slip: the model wrote a tool call as a bare `{...}` object instead of the `[{...}]` list.
- In 16 of the 17 affected questions every call still ran, so the slip cost it at most one question.
- B never makes this slip.

## Cost and speed (ROI)

| | gpt-6-luna (API) | B (T4 GPU, by rate) |
|---|---:|---:|
| Total for the 144 questions | $0.068 | $1.095 |
| Per question | $0.00047 | $0.0076 |
| Per correct answer | $0.00071 | $0.025 |
| Seconds per question | 8.1 | 68.4 |
| One-time cost to build | none | $12.46 (A1500 $2.36 + Q2 sampling $9.11 + B $0.99) |

## Read plainly

- **Accuracy.** The cheapest current closed model scores about twice what B does. The gap is widest on
  complex questions: gpt-6-luna solves 25 of 51, and B solves 1.
- **Where fine-tuning closed the gap.** On one-table lookups, B is close to the closed model (29 vs 31 of
  36).
- **Cost.** In this setup, gpt-6-luna is also cheaper: about 16 times per question, 35 times per correct
  answer, and 8 times faster.
- **Protocol discipline.** B's formatting is cleaner: no malformed turns, and no failed submissions.
- **Not measured here:**
  - B's cost is the HF Jobs T4 list rate, with plain transformers generation, one question at a time.
    Batched or quantized serving on one's own hardware could cost far less.
  - Running locally keeps the data on one's own machine, and that is not priced.
- **Caveats:**
  - The closed model used Tiny Aya's text tool format, not its native function calling.
  - This is one sample, and the API may not be fully deterministic.
- **For the write-up:** a cheap closed model wins clearly on accuracy and cost per answer. B's case rests
  on being small, open and private, and on matching it for simple lookups, not on accuracy or measured
  cost.
