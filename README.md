# ayadb: a database agent from Tiny Aya

Lexsi AI Safety & Alignment Hackathon, **Track 3 (Agentic Fine-tuning)**, problem statement "data and database
agents". Solo entry. Stack: Cohere **Tiny Aya Global**, Lexsi **AgentTune** (training and rollouts) and
**AuditKIT** (evaluation).

We fine-tuned Tiny Aya Global with QLoRA to answer questions over SQLite databases it has never seen. The
model uses four tools (`list_tables`, `describe_table`, `run_sql`, `submit_sql`) within eight tool steps. Every
score is execution-based (no LLM judge), and AuditKIT re-checks it.

**Links:** HF model, dataset and write-up are added here at publication.

## Results (final sets, strict)

| Set | Base (untuned) | A1500 (SFT) | **B (submitted)** |
|---|---:|---:|---:|
| SQaLe final (144, unseen databases) | 2.8% | 31.3% | **30.6%** |
| BIRD Mini-Dev (493, never trained on) | 0.0% | 1.6% | **1.4%** |
| Shop retention, dev (40) | 0 | 40 | **40** |
| Shop retention, multi-hop (40) | 0 | 1 | **0** |

- **A1500:** QLoRA SFT on 1,500 verified multi-turn episodes (SQaLe plus shop).
- **B:** A1500 plus one self-training round on 306 of its own verified answers (rejection-sampling
  fine-tuning). B was fixed as the submitted model from dev results, before its final numbers existed.
- **Where it works:** fine-tuning makes the agent a clean tool user. Submitted queries that failed to run
  fell from 377 to 2 across the final sets, and one- and two-table questions improve a lot.
- **Where it does not:** questions over 3+ tables or needing several steps stay near zero (1 of 51 on
  SQaLe final). No model recovers after a failed query, and running out of the eight tool steps is the main
  failure.
- **Negative results:**
  - error-hint tools, repair demonstrations and self-training did not raise final accuracy;
  - self-training beat its matched control 4-0 on dev (p = 0.125), but gave no final gain.

Details: `experiments/sql-final-v1/RESULTS.md`, `experiments/sql-final-b/RESULTS.md` and
`experiments/sql-selftrain-v1/RESULTS.md`. All work ran on single T4 GPUs for about $37 of compute.

## What is in this repo

| Path | Contents |
|---|---|
| `src/ayadb/` | environment and tools, scoring (`sqlscore.py`, `sqlguard.py`), training and evaluation runner |
| `scripts/sql_job.py` | registered GPU runs: `plan`, `preflight`, `launch`, `status`, `pull` (see `docs/sql-runner.md`) |
| `scripts/prepare_*.py` | build each run's registration locally |
| `experiments/` | the plan written before each run (`REGISTRATION.json`), amendments, and results (`RESULTS.md`, `RESULTS.json`) |
| `docs/`, `reports/` | runner and data-build guides, source and licence notes, reviews |
| `tests/` | local tests |

## Reproduce

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
uv pip install --python .venv\Scripts\python.exe -r requirements-sql-runner.txt
.venv\Scripts\python.exe -m pytest -q
```

- **Data:** build the SQaLe and BIRD inputs with `docs/sqale-build.md`. The data is not in git. Tests that
  need it, or need pulled job outputs, report "skipped" with the missing path until it exists.
- **GPU runs:** every paid run is a registration with pinned model revision, data and code hashes. It is
  launched once with `scripts/sql_job.py`, and the job checks its inputs before any model use.

## About the run records

The files in `experiments/` are the records written before and after each run. For this public copy,
personal details were removed from them after the runs, and internal notes were left out. As a result:
- SHA256 values inside the records refer to the files as they were when each job ran. Some published copies
  no longer match them, and the tests that check those fingerprints report "skipped" here.
- Commit IDs in the records refer to the development history, which is not published.
- `hf-user` stands for the HF account a job ran on. To run a registration yourself, set its
  `expected_account` and `runconfig.ACCOUNTS` to your own HF username.

## Licences

- **Code:** Apache-2.0 (`LICENSE`).
- **Tiny Aya Global:** CC-BY-NC-4.0 (Cohere Labs). Adapters trained from it are for non-commercial use.
- **AgentTune and AuditKIT:** Lexsi source-available, non-commercial licences.
- **SQaLe** (Wolff, Gomm and Hulsebos, 2026): its card declares MIT; see
  `reports/2026-11-07-sqale-source-notices.md`.
- **BIRD Mini-Dev:** CC BY-SA 4.0, used for final evaluation only and not redistributed here; see
  `reports/2026-11-07-bird-source-notices.md`.

## Development log

The project log, in the order the work happened. Later entries supersede earlier ones.

Lexsi AI Safety & Alignment Hackathon, **Track 3 (Agentic Fine-tuning)**: reliable tool calling
and efficient trajectories. Stack: Lexsi **AgentTune** (rollouts, later training) and
**AuditKIT** (evaluation), Cohere **Tiny Aya Global**.

**Status (2026-11-05):** feasibility gate done; SFT pilot done (`experiments/sft-pilot/`: dev 2/40 -> 40/40,
sealed test: shared families 5/77 -> 77/77, held-out multi-hop families 0/20 -> 0/20); recovery experiment
done (`experiments/recovery/RESULTS.md`: no difference between the arms; the renamed-column split
produced no errors, so it measured schema grounding, 40/40 for both, not recovery; new multi-hop
types 0/40 vs 1/40; the pilot adapter stays). The gate results below are the untuned baseline.

**Local preparation (2026-11-07):** the active v3 build has 4,888 expert training episodes,
581 Q2 questions, 96 dev and 144 final questions, plus shop dev 40/mh 40 retention in the SQL-submission
format. All held-out primary prompts were individually read; exclusions and scalar-gold checks are logged.
The earlier v2 build is preserved. A preregistered run-input overlay gives **4,887 eligible full-stage
episodes, 579 Q2 and 95 dev**; SQaLe final and retention are unchanged. The 10% gate now uses the exact
post-validation A training set, and minimum-drop control matching is deterministic and coverage-reported.
See [the effective-input freeze](experiments/sqale-sft/RUN-INPUTS-FREEZE-v1.md),
[v3 report](reports/2026-11-07-sqale-v3.md) and [preparation guide](docs/sqale-build.md).

Approved BIRD Mini-Dev preparation is complete: **493/500** retained, seven logged execution-cap
exclusions, all 493 reference results independently verified. It stays **final-only**, with evidence
preserved, pinned source/databases and CC BY-SA 4.0 notices. See [the BIRD report](reports/2026-11-07-bird-final-build.md).
No BIRD model evaluation, training, tuning or selection has occurred.

The additive [SQL model runner](docs/sql-runner.md) now includes AgentTune-backed training, an expected
hf-user account check, complete checkpoints, token accounting and longer registered wall limits.
The approved [G0 check failed](experiments/sql-g0/RESULTS.md) with CUDA OOM on its first longest-example
forward, before any optimizer update or inference. It ran 136.66s (about $0.02 compute by rate; invoice
unavailable). No retry or other job was started. The pilot remains the submission model.

The prospective [table-read guard](reports/2026-11-07-table-read-guard.md) rejects literal-only strict
answers; all 6,278 active references pass its native read check. A clean-protocol self-training predicate
and diagnostic manual audit are prepared. The [selected-logits fix](reports/2026-11-07-selected-logits.md)
preserves native loss/LoRA gradients in tiny random CPU tests and adds peak/OOM logging plus explicit
last-position inference. **114 tests pass.**

The user approved [G0-02](experiments/sql-g0-02/REGISTRATION.md) and it **passed**
([results](experiments/sql-g0-02/RESULTS.md)): selected and full native loss agree (difference 6e-8),
three AgentTune updates on the longest 4,072-token example peaked at 8.91 GiB of 14.74 GiB on a T4,
checkpoint resume and adapter reload worked, and nine inference episodes ran with AuditKIT agreement.
Running time 670.9s, about $0.08 compute by rate (invoice unavailable). This is a compatibility check,
not an accuracy result.

**500 stage (2026-11-07, both runs approved and passed):** on the 95 effective SQaLe dev tasks (unseen
databases, greedy, strict table-read scoring), untuned Base scores **2/95**, the shop-trained Pilot
**20/95** and **A500** (500 verified episodes, 265 SQaLe plus the 235 converted shop episodes, two passes through AgentTune's audited factory)
**28/95**. Fallback is 2 / 21 / 29. **G1 passes** (A500 beats Base by 27.4 strict and 28.4 fallback
points). A500 lists tables first in every episode, submits only runnable queries (66 of 67) and never
answers without reading a table. Hard tasks remain 0/19, and its main failure is running out of the
eight tool steps. Compute was about $1.99 by rate. A manual audit found no lucky passes. See
[A500 results](experiments/sql-a500/RESULTS.md) and [baseline results](experiments/sql-dev-baselines/RESULTS.md).
Shop retention is not yet measured, so the submission model is unchanged.

Most of A500's gain is on simple one-table lookups (21 of its 28). It solves 1 of the 30 complex dev
questions (3+ tables or multi-step).

**1,500 stage (2026-11-08, approved and passed):** [A1500](experiments/sql-a1500/RESULTS.md) trains fresh
from the base model on the nested 1,500 set (454 complex SQaLe rows; A500 had 100), with A500's settings. It scores
**37/95 strict** (fallback 37, set 40), +9.5 points over A500, so the preregistered stage rule (needs
33/95) is met. That permits proposing the full stage; nothing is approved. Every submitted query runs,
and hard tasks go from 0/19 to 2/19. **Complex queries did not improve** (1/30, as for A500). The gain
is in lookups and two-table joins, and running out of the eight tool steps is still the main failure
(25 of 95). Compute was about $2.36 by rate. The manual audit found no lucky passes.

A free analysis found that A1500 never recovers from a failed query (0 of 24 episodes): it repeats the
identical query 76 times out of 80, usually after a bare `no such column` error. [A1500 with error-feedback tools](experiments/sql-a1500-hints/RESULTS.md) (amendment 9, approved and run
once, $0.66) re-tested the same model with tools that list the real column names on such errors and refuse
an identical failed repeat. It scored **38/95** (+1, not the 42 needed to adopt them), and complex stayed at
1/30. The model mostly ignores the feedback: it re-checks the table but rarely rewrites the query. So the
fix has to come from training examples that show correcting a query.

[A1500-v4](experiments/sql-a1500v4/RESULTS.md) (amendment 10, approved and run once, $2.40) retrained on
the same 1,500 questions with rebuilt examples. 349 repairs show a realistic column mistake, a re-check
of the right table, then the fix. 190 complex questions run the first part of the query first. It
learned the new habit: after an error it re-checks the table instead of repeating itself, and it runs
out of steps less often (20 vs 25). But it scored **35/95** (A1500: 37; 3 tasks gained, 5 lost, which
is within noise), and it still never recovers to a correct answer. **A1500 stays the trained finalist.**

**Final evaluation (2026-11-10, [results](experiments/sql-final-v1/RESULTS.md)), Base → A1500, strict:**

| Set | Base | A1500 |
|---|---:|---:|
| SQaLe final (144, unseen databases) | 2.8% | **31.3%** |
| BIRD Mini-Dev (493) | 0.0% | **1.6%** |
| Shop retention, dev (40) | 0 | **40** |
| Shop retention, multi-hop (40) | 0 | 1 |

All in-job checks pass. Complex SQaLe queries stay hard (2 of 51), and on BIRD's larger schemas A1500
often runs out of its eight tool steps.

**Self-training round (2026-11-10, [results](experiments/sql-selftrain-v1/RESULTS.md)):**
1. A1500 tried the 577 held-out Q2 questions twice each ($9.11) and got 380 attempts right.
2. Its 306 verified attempts trained **B**: two passes on top of A1500.
3. The control **A+** trained on the same questions with teacher examples, matched token for token.

On the 95 dev tasks, B scores **40/95** strict, A+ 36 and A1500 37. Own attempts beat matched teacher data
4 tasks to 0, but at this size that is not statistically clear (p = 0.125). Complex queries stay at 1/30,
and none of the arms recovers after a failed query. B meets the preregistered 38/95 bar for joining the
final evaluation, which needs its own registration and approval. Both runs cost $1.94 together, and all
checks pass.

**B's final evaluation (2026-11-11, [results](experiments/sql-final-b/RESULTS.md)), strict, approved and
run once, $6.75.** B was fixed in advance from dev as the submitted model.

| Set | Base | A1500 | B |
|---|---:|---:|---:|
| SQaLe final (144) | 2.8% | 31.3% | **30.6%** |
| BIRD Mini-Dev (493) | 0.0% | 1.6% | **1.4%** |
| Shop retention, dev (40) | 0 | 40 | **40** |
| Shop retention, multi-hop (40) | 0 | 1 | **0** |

B and A1500 are the same within noise on every set: B's dev edge did not carry over. Complex queries
stay unsolved. All checks pass.

Amendment 11 froze the final selection, **Base vs A1500**, from dev results only,
and prepares the final evaluation on SQaLe final (144), BIRD Mini-Dev (493) and shop retention (40 + 40). It
also prepares one self-training round: A1500 attempts the 579 held-out Q2 questions twice each, and a
model B is trained on its own verified attempts, with a matched control. Scoring-side SQL now allows
60 s and 2,000M operations (amendment 12; the agent's tools keep 5 s and 50M), because a BIRD reference
query can exceed 5 s and the job's older SQLite counts operations differently. Re-scoring all earlier
dev episodes changed nothing. **145 tests pass.**

Dev split, 40 tasks, verified success (correct answer backed by a successful query):

| Model / format | Decoding | Episodes | Success | Valid-call rate | Loops |
|---|---|---|---|---|---|
| Tiny Aya Global, AgentTune default format | greedy x2 | 80 | 0.0% | 92.6% | 5% |
| Tiny Aya Global, multistep + stop "Tool result" | greedy x2 | 80 | 0.0% | 94.8% | 12.5% |
| Tiny Aya Global, multistep, one call per turn | greedy | 40 | 0.0% | 95.5% | 15% |
| Tiny Aya Global, multistep, close on `}]` | greedy | 40 | 0.0% | 79.5% | 35% |
| Tiny Aya Global, multistep, close on `}]` | T=0.8 x4 | 160 | 2.5% (pass@4 on 4/40 tasks) | 93.2% | 24% |
| Aya Expanse 8B (Q4_K_M), multistep, close on `}]` | greedy | 40 | 12.5% | 100% | 17.5% |

Greedy A/A: 0 outcome flips between repeats. AuditKIT's verdicts agree with ours on all episodes.
Tiny Aya was run from Cohere's official ungated q8_0 GGUF because the gated HF repo was not yet
accessible; re-run on the HF weights before training.

### Environment

A deterministic, read-only SQLite database for a fictional supplier (`src/ayadb/env.py`):
`customers`, `products`, `orders`, `order_lines`, `tickets`. The schema has traps that can only be
passed by inspecting it: money in cents (`unit_price_cents`), keys named `cust_id` / `order_no` /
`sku`, products have a `title`, unresolved tickets have NULL `resolved_at`.

Tools (plain Python functions; AgentTune builds their JSON schemas from the docstrings):

| Tool | Does |
|---|---|
| `list_tables()` | table names and row counts |
| `describe_table(table)` | columns, foreign keys, 3 sample rows; unknown table -> error listing valid names |
| `run_sql(query)` | one read-only SELECT, max 20 rows; errors come back as text so the agent can recover |

Read-only is enforced three ways: `mode=ro` URI, `PRAGMA query_only`, and an authorizer that only
allows reads. Runaway queries are interrupted.

### Tasks and splits

`src/ayadb/tasks.py`: 11 task families. Each gold answer comes from a reference SQL query
on that split's own database; draws with ties or zero answers are rejected.

| Split | DB seed | Families | Tasks |
|---|---|---|---|
| train | 101 | 8 shared | 198 |
| dev | 202 | 8 shared | 40 |
| test (sealed, used once) | 303 | 8 shared + 3 held-out multi-hop | 97 |
| drift (recovery experiment) | dev's rows, 5 columns renamed | 8 shared | 40 |
| mh (recovery experiment) | 404, 405, 406 | 3 held-out multi-hop | 40 |

The three held-out families (`cancel_leader_tickets`, `largest_order`, `segment_aov`) never appear
in train or dev. The smoke runner refuses the test split without `--unseal`.

### Prompt format (the documented baseline)

System prompt (`SYSTEM_PROMPT` in `src/ayadb/smoke.py`) + AgentTune's
`DEFAULT_TOOLS_FALLBACK_PROMPT`. Tiny Aya's chat template ignores `tools`, so AgentTune appends the
JSON tool schemas to the system prompt, asks for calls as
`[{"tool_name": ..., "parameters": {...}}]`, and folds tool results into user turns
(`Tool result for <name>: ...`). The final answer must be one line: `ANSWER: <value>`.

Prompts are rendered by AgentTune's own `_render_chat_template`, so the smoke test sees the same
text a training rollout would. Generation runs in Ollama (raw mode) on the local GPU.

### Metrics (`src/ayadb/metrics.py`)

- **success** (primary): verified correct answer **and** at least one successful `run_sql`.
  A correct number with no query behind it is a guess and does not count.
- valid-call rate: executed calls with valid arguments / attempted calls (attempts include a
  call-shaped final turn that failed to parse).
- diagnostics: bad-argument calls, unknown tools, execution errors (bad SQL, wrong table),
  malformed final turn (invalid JSON ends the episode), repeated identical calls, loops
  (any identical call 3x, or step budget exhausted), turns, tool calls, tokens, wall time.
- AuditKIT re-scores the saved trajectories with the same verifier (`CustomPredicate`) as an
  integration check.

`tests/test_ayadb.py` includes adversarial trajectories: guesses without a query, invalid JSON,
wrong argument names, unparseable arguments, repeated calls, budget exhaustion, unknown tools,
hedged answers ("12 or 13").

### Run

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
ollama pull hf.co/CohereLabs/tiny-aya-global-GGUF:Q8_0
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ayadb.smoke --split dev --repeats 2 --temperature 0 --tag greedy-aa
```

Each run writes `runs/<id>/` with `config.json` (code rev, data hashes, model blob, versions,
sampling), `trajectories.jsonl` (AgentTune records), `metrics.jsonl`, `auditkit.json`,
`summary.json`.

### Licences

Code: Apache-2.0. Tiny Aya: CC-BY-NC-4.0 (Cohere Labs). AgentTune (LSAL v1.1) and AuditKIT
(LSAL-1.2) are source-available, non-commercial.
