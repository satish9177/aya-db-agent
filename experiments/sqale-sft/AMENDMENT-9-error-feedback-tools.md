# Amendment 9: error-feedback tools (`column-hints-v1`) and one A1500 re-screen (2026-11-08)

Written after the A1500 results (`experiments/sql-a1500/RESULTS.md`) and a free local analysis of
its errors, and before any run with the new tools. The user approved building this as free work.
**No upload, job or push is approved by this amendment.** The registration needs the user's explicit
approval of its SHA256, uploads and storage before launch.

## Why

A1500 scores 37/95 but solves only 1 of the 30 complex dev tasks. Half of its complex failures (15 of
29) run out of the eight tool steps, and the free analysis shows why:

- A1500 never recovers from a `run_sql` error. 24 dev episodes hit one and none passed.
- After an error, its next call repeats the identical query 76 times out of 80. 99 of its 104 errors
  are SQLite's bare `no such column: …`, which does not say what the right name is.
- Its training demos do the opposite. The 1,500 set contains 410 failed queries, each followed by a
  changed query or `describe_table`, never an identical repeat.

The remaining 14 complex failures submit wrong logic. This amendment does not target them.

## What changes: a new tool set beside the old one

`src/ayadb/sqlfeedback.py` adds `HintedSQLTools` (policy `column-hints-v1`). It has the same four tools
with the same names, arguments and descriptions. A test shows AgentTune builds identical tool schemas,
so the model sees the same prompt. Only two `run_sql` observations differ:

1. **Hints on name errors.** A `no such column` error also lists the real columns of every real table the
   query names. If the bad name is a real column of some table, it adds `Column X is in: …`. A
   `no such table` error, or a column error naming no real table, lists the available tables, as
   `describe_table` already does for unknown tables. Hints are capped at 800 characters.
2. **No identical failed repeats.** A query matching an earlier failed `run_sql` (ignoring whitespace and
   trailing semicolons) is not run again. The tool returns `Error: this exact query already failed with:
   <error>. It was not run again. Write a different query; check the column names first.`

It is opt-in per evaluation step (`"tool_feedback": "column-hints-v1"`, validated by `runconfig`).
Without it the runner uses the unchanged `SQLTools`, and its outputs carry no new fields. The system
prompt, tool prompt, step limit, token limits, context, decoding, scorer and AuditKIT check are
unchanged. Refused repeats are logged as failed `run_sql` calls, so they never feed the fallback score.

## Offline replay (no model)

Replaying A1500's 104 failed `run_sql` calls through the new tool code (`scripts/prepare_sql_a1500_hints.py`
recomputes these numbers):

- **Repeats:** 27 distinct failed queries; the other 77 calls are identical repeats that would be refused.
- **Hints:** 26 of the 27 get a hint (one syntax error does not), and 23 episodes would see at least one.
  Hints run 233 characters median, 518 at most.
- **What the hints cover:** the bad column name was invented or misspelled in 13 cases, belonged to
  another table already in the query in 8, and existed only in a table outside the query in 4. The
  `Column X is in` line covers the last 12.

Whether the model uses the hints is what the run measures.

**Disclosure:** the design was made after reading A1500's dev failures on the same 95 tasks this re-screen
uses. A gain here is selected on dev; the final evaluation on separate data is the confirmation.

## What runs: `a1500-hints-sql-20261108-01` (stage `evaluate`)

One step. It reloads exactly A1500's step-180 adapter (the pulled copy from job 6af0139be6320c66fad700ed,
pinned by hash) and screens it on the same 95 dev tasks with `column-hints-v1`. Everything else matches
A1500's dev step:
- greedy decoding, seed 42, eight tool steps, 512 new tokens per turn, 8,192 context;
- one projected position, table-read-v1 strict plus fallback and set, AuditKIT agreement;
- every dev reference checked before any model call.

The comparator is A1500's own result with the original tools (37/95), pinned by its metrics hash and not rerun.

## Decision rule (fixed now)

- **Adopt** `column-hints-v1` for the final evaluation only if A1500 with hints reaches **at least 42/95
  strict** on the same tasks (five unrounded points over 37, the project's usual margin). If adopted,
  every model in the final evaluation (Base, Pilot, finalists) uses the same tools, so before/after
  comparisons stay fair. Otherwise the final evaluation keeps the original tools.
- The result is reported as it is either way. Nothing here trains or changes a model.

## Reported diagnostics (descriptive, not gates)

- Strict, fallback and set; paired discordant tasks against A1500 with the original tools (exact
  McNemar, unadjusted).
- Amendment-8 query shapes: strict and "ran out of steps" per class, and the 30 complex tasks.
- **Recoveries:** episodes with at least one failed `run_sql` that end strict (original tools: 0 of 24).
- Hints shown, repeats refused, and the remaining unsubmitted episodes.
- **Determinism check:** episodes with no failed `run_sql` under the original tools should replay the
  same call log, since nothing differs until the first error. Report how many differ.
- Time, cost, longest prompt and GPU memory peak.

## Money and time

- One t4-small job, rate $0.006667/min, **4.5-hour hard timeout** (step limit 4 h): $1.80 compute
  ceiling plus the $0.04 first-month storage reserve, **$2.00 cap**.
- Forecast from A1500's dev step (6,386 s): about 1.8 h, **~$0.73**.
- Private upload: dev tasks, dev databases, tokenizer and the four A1500 adapter files (~121 MB). Output
  is a few MB (no checkpoints). Storage at most 30 days without renewed approval.
- Memory: A1500's dev step peaked at 10.14 GiB with a 7,018-token prompt. AgentTune trims prompts to
  about 7,936 tokens, and hints add at most ~800 characters. An OOM is journaled and fails the step,
  with no retry.

## Unchanged

Report all results as they are, including negatives. No automatic retry, training, hardware change,
step-limit, prompt or decoding change. Q2, A+/B, the full stage, BIRD/final evaluation, teacher
models, push and publication remain unapproved. Shop retention stays deferred to the final evaluation.
The sealed shop test stays closed.
