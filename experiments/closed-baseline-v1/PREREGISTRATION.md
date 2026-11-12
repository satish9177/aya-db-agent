# Closed-model baseline v1: gpt-6-luna on SQaLe final (preregistration, 2026-11-11)

Written before any API call. It covers **one run**. The run starts only after the user approves the exact
SHA256 of `REGISTRATION.json` (built by `scripts/prepare_closed_baseline.py`).

## Question

How does a cheap closed model compare with our fine-tuned Tiny Aya (B) on the same 144 SQaLe final
questions? We compare accuracy, and cost per question and per correct answer.

## Design

- **Model:** OpenAI `gpt-6-luna` with reasoning effort `none`, so it uses no hidden reasoning tokens.
  - It is OpenAI's cheapest current text model: $0.10 input and $0.50 output per 1M tokens on the Standard
    tier (official pricing page, checked 2026-11-11).
  - It was chosen before any result. It is a baseline only, never a submission candidate.
- **Everything else is the same as B's final run:**
  - the 144 questions;
  - the four original tools;
  - the system prompt, with the text tool list exactly as Tiny Aya sees it;
  - tool results returned as user turns, with the same wording;
  - eight tool steps and 512 output tokens per turn;
  - the 8,192-token context guard, counted with Tiny Aya's tokenizer;
  - the same stop rules, applied on our side;
  - `table-read-v1` strict scoring, and AuditKIT must agree with every score.
- **Decoding:** temperature 0, one sample.
  - If the API rejects `temperature` or `stop` for this model, that parameter is dropped and the run
    continues. This rule is decided in advance.
  - The stop rules on our side still apply, and any dropped parameter is reported.
- **Not done:**
  - no prompt tuning, few-shot examples or native function calling;
  - no other model;
  - no repeated episodes.

## Outcomes

- **Primary:** strict successes of the closed model vs B (44/144) on the same tasks, with an exact
  two-sided McNemar test.
- **Secondary:**
  - set and fallback scores;
  - running out of tool steps;
  - submitted queries that failed to run;
  - malformed turns;
  - strict results by query shape.
- **ROI:**
  - API dollars per question and per strict success;
  - against B's GPU cost by rate: $1.095 for the 144 questions ($0.40 per T4 hour), which is $0.0076 per
    question and $0.025 per strict success;
  - seconds per question for both.
  - B's one-time build cost (training and sampling) is reported separately.

## Budget and stopping

- **Cap: $1.50** of API spend. The user has about $2 of OpenAI credit.
- **Central estimate:** about $0.20, if usage looks like B's (1.31M prompt and 42k output Tiny Aya tokens,
  times 1.25 for tokenizer differences).
- **Worst case:** about $1.66, which assumes every one of 9 calls per question uses the full context.
- **Enforcement:** the engine refuses any call whose worst case could pass the cap. If that happens, or an
  error stops the run, the finished episodes are reported as they are.
- **Retries:** transient HTTP errors retry the same request (up to 5 times). An episode is never re-run.

## Limitations, stated in advance

- **Prompt format.** The closed model gets a prompt format designed for Tiny Aya: tool calls as JSON
  text. Closed models are usually run with native function calling, so this may understate them. We
  accept that to keep the protocol identical.
- **Determinism.** Closed APIs may not be fully deterministic even at temperature 0.
- **Data sent out.** The SQaLe questions and database samples are public data, and they are sent to OpenAI.

## Reporting

Results go in `experiments/closed-baseline-v1/RESULTS.md` and are reported as they are, whichever model
comes out ahead.
