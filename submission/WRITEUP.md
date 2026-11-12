# Teaching Tiny Aya to query databases: what fine-tuning fixes, and what it does not

*Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning), problem statement "data and database
agents". Solo entry. 2026-11-12.*

**Links:**
- [Collection](https://huggingface.co/collections/satish9177/tiny-aya-sql-agent-lexsi-hackathon-track-3-6af56be33215a3e1b8eb8280)
- [Model B (submitted)](https://huggingface.co/satish9177/tiny-aya-sql-agent)
- [Model A1500](https://huggingface.co/satish9177/tiny-aya-sql-agent-a1500)
- [Data, trajectories and logs](https://huggingface.co/datasets/satish9177/tiny-aya-sql-agent-data)
- [BIRD evaluation records](https://huggingface.co/datasets/satish9177/tiny-aya-sql-agent-bird-eval)
- [Code](https://github.com/satish9177/aya-db-agent)

## TL;DR

- **Before → after.** We fine-tuned Cohere's **Tiny Aya Global** with **QLoRA** through Lexsi
  **AgentTune** and checked every score with **AuditKIT**. On 144 questions over databases it never saw,
  strict accuracy rose from **2.8% to 30.6%**. On our shop database, the model went from **0 to 40 of 40**.
- **Clean tool use.** The fine-tuned agent reads the schema before querying, and almost every query it
  submits runs: 2 failures across all final sets, against 377 for the untuned model.
- **The limit is complex queries.** Questions over 3+ tables or needing several steps stay near zero
  (1 of 51). On BIRD Mini-Dev the model scores only 1.4%. It never recovers after a failed query, and
  running out of its eight tool steps is the main failure.
- **Three fixes did not help, and we report them:**
  - error-hint tools;
  - training examples that show how to fix a query;
  - one round of self-training on the model's own verified answers (rejection-sampling fine-tuning, RFT).

  Self-training beat a matched control on dev (4 tasks to 0, p = 0.125), but gave no gain on the final
  sets.
- **A cheap closed model is far ahead.** OpenAI's cheapest current model, `gpt-6-luna`, took the same test
  under the same rules and scored **66.0%** against B's 30.6%. It solved 25 of 51 complex questions, and
  it cost less per correct answer ($0.0007 vs $0.025). B comes close only on simple lookups (29 vs 31 of
  36).
- **One plan per run.** Every paid run had a registration with pinned hashes, written before launch, and
  ran once with no retries. The final sets were used only after the model was chosen. Total compute was
  about **$37** on single T4 GPUs.

## 1. Problem and setup

Small open models are attractive for database agents: cheap, private, local. Out of the box, Tiny Aya
Global cannot act as one. It invents tool output, writes malformed calls, and answers without querying.
We asked how far supervised fine-tuning on verified tool-use episodes can take it, and where it stops.

**Agent loop (`sql-submit-v1`).** The model gets a question and four tools over a read-only SQLite
database. It has **eight tool steps** per question.

| Tool | Does |
|---|---|
| `list_tables()` | lists the tables |
| `describe_table(table)` | shows columns, foreign keys and three sample rows |
| `run_sql(query)` | runs one read-only `SELECT` and previews up to 20 rows |
| `submit_sql(query)` | submits the final query and ends the episode |

Read-only access is enforced by SQLite itself: the database opens read-only, and an authorizer allows
reads only. The agent's queries are limited to 5 s each. Rollouts use AgentTune's loop.

**Base model:** `CohereLabs/tiny-aya-global`, pinned revision `af89d219…`, loaded in 4-bit NF4 with fp16
compute on a single Tesla T4 for both training and evaluation. Decoding is greedy.

## 2. Metrics

We report more than one metric, and the scoring uses no LLM judge.

- **Strict (primary), `table-read-v1`.** The submitted query's full result must equal the reference
  result. SQLite must also report that the query read a real table, so a query that only returns a
  constant (`SELECT 42`) cannot pass.
- **Set.** The result rows match the reference as a set. This is close to BIRD's usual execution
  accuracy.
- **Fallback.** If the submission fails, the last successful query is scored instead.
- **Behaviour:**
  - runs out of steps;
  - submitted queries that fail to run;
  - malformed turns and invented tool output;
  - recovery after an error;
  - tokens and runtime.
- **Retention.** Shop-database tasks check that SQL training did not break the earlier skill.

AuditKIT re-scores every saved episode offline with the same strict rule, and it agreed with every score
in every run.

## 3. Data

| Split | Source | Size | Used for |
|---|---|---:|---|
| Training | SQaLe (`trl-lab/SQaLe-2-text-to-SQL-*`, MIT per its card) plus the shop database | 1,500 episodes (1,265 SQaLe + 235 shop) | SFT |
| Q2 | SQaLe, held out from training, dev and final | 577 questions | self-training samples only |
| Dev | SQaLe, unseen databases | 95 | all model choices |
| **SQaLe final** | SQaLe, unseen databases | **144** | final only |
| **BIRD Mini-Dev** | BIRD, SQLite release, with its evidence notes | **493 of 500** | final only, never trained or tuned |
| Shop retention | our fictional-supplier database | 40 dev + 40 multi-hop | final only |

- **Splits.** All splits are separated by database and schema, so no final database appears in training.
- **Exclusions.** Seven BIRD questions were excluded before any model use because their reference query
  exceeded the execution cap. SQaLe questions whose answer depended on SQLite's row order, or that were
  not real database questions, were also excluded before model use. Each exclusion is logged.
- **BIRD evidence.** BIRD questions include the source's evidence note where one exists (358 of 493).
- **Training episodes come from the reference SQL, not from a teacher model.** A script replays a
  realistic path (list tables, describe the needed tables, run the query, submit) against the real
  database. Every tool observation is real output, and no paid API or LLM teacher was used.
- **Loss.** It covers only the model's own turns; tool outputs and error messages are never trained.

## 4. Method

**A500 → A1500 (supervised fine-tuning).** QLoRA through AgentTune's audited SFT factory:
- LoRA rank 16, alpha 32, dropout 0.05, on all attention and MLP projections;
- learning rate 2e-4, 2 epochs, effective batch 16 (180 updates for A1500);
- inputs up to 4,096 tokens.

We grew the data in preregistered stages (500, then 1,500 episodes). Each stage could continue only if
it beat the previous one by at least 5 strict points on dev.

**B, self-training (rejection-sampling fine-tuning).**
1. A1500 attempted each Q2 question twice (temperature 0.8) and got 380 of 1,154 attempts correct.
2. After strict verification and de-duplication, 306 attempts on 229 questions were kept.
3. B continues from A1500 for two passes over them (learning rate 1e-4, 40 updates).

**A+, the matched control.** It trains on reference-path episodes for the same 229 questions, matched to
exactly the same number of training tokens (49,717 per pass). Any difference between B and A+ therefore
comes from using the model's own attempts, not from more data.

**Model choice.** B was fixed as the submitted model from dev results, before any of its final numbers
existed.

## 5. Results

### Final evaluation (strict; all jobs passed every registered check)

| Set | Base (untuned) | A1500 | **B (submitted)** |
|---|---:|---:|---:|
| SQaLe final (144, unseen databases) | 4 (2.8%) | 45 (31.3%) | **44 (30.6%)** |
| BIRD Mini-Dev (493) | 0 (0.0%) | 8 (1.6%) | **7 (1.4%)** |
| Shop retention, dev (40) | 0 | 40 | **40** |
| Shop retention, multi-hop (40) | 0 | 1 | **0** |

- **Set metric:** A1500 and B tie on SQaLe final (51 vs 51) and on BIRD (8 vs 8).
- **B vs A1500:** the same within noise on every set (exact McNemar p = 1.0 on each).

### Dev progression (95 tasks, strict)

| Model | Strict | What changed |
|---|---:|---|
| Base | 2 | untuned |
| Pilot | 20 | 235 shop episodes only |
| A500 | 28 | 500 episodes |
| A1500 | 37 | 1,500 episodes (beat A500 12 to 3 task by task, p = 0.035) |
| A1500 + error-hint tools | 38 | same model, tools list the real columns after an error |
| A1500-v4 | 35 | retrained with 349 repair demos and step-by-step complex examples |
| A+ | 36 | A1500 + matched reference-path data |
| **B** | **40** | A1500 + its own 306 verified attempts |

### Where the gains are (SQaLe final, strict, by query shape)

| Shape | n | Base | A1500 | B |
|---|---:|---:|---:|---:|
| 1-table lookup | 36 | 4 | 27 | 29 |
| 1-table aggregate | 13 | 0 | 6 | 7 |
| two tables | 44 | 0 | 10 | 7 |
| 3+ tables | 19 | 0 | 1 | 0 |
| multi-step | 32 | 0 | 1 | 1 |

Fine-tuning taught simple and two-table questions well. Questions over 3+ tables or needing several steps
barely moved.

### Behaviour

| | Ran out of steps: SQaLe final | Ran out of steps: BIRD | Submitted query failed to run (all final sets) |
|---|---:|---:|---:|
| Base | 5 | 9 | 377 |
| A1500 | 39 | 235 | 2 |
| B | 46 | 252 | 2 |

The untuned model stops early because it submits broken queries. The fine-tuned models submit runnable
queries, then use up their eight steps on large schemas (BIRD) and on complex questions.

### Comparison with a closed model (ROI)

**Setup.** OpenAI's cheapest current text model, `gpt-6-luna`, took the same test as B:
- the same 144 SQaLe final questions;
- identical prompts, the same tools and eight steps;
- strict scoring, checked by AuditKIT;
- reasoning effort `none`.

It was preregistered and run once (`experiments/closed-baseline-v1/`).

| | B (submitted) | gpt-6-luna |
|---|---:|---:|
| Strict, 144 questions | 44 (30.6%) | **95 (66.0%)** |
| 1-table lookups (36) | 29 | 31 |
| Two tables (44) | 7 | 26 |
| Complex: 3+ tables or multi-step (51) | 1 | 25 |
| Ran out of steps | 46 | 10 |
| Malformed turns | **0** | 38 |
| Cost per question | $0.0076 (T4 GPU, list rate) | **$0.00047** (API) |
| Cost per correct answer | $0.025 | **$0.00071** |
| Seconds per question | 68 | 8 |

**What this shows:**
- **Accuracy.** Task by task, the closed model solves 57 questions that B misses, and B solves 6 that it
  misses (p = 1.6e-11).
- **Cost and speed.** The closed model is also cheaper and faster in our setup.
- **Where B holds up.**
  - It matches the closed model on simple one-table lookups.
  - It follows the tool format more cleanly. gpt-6-luna's 38 malformed turns were bare-object tool calls,
    which almost always still ran.
- **Not measured.**
  - B's cost assumes a rented T4 running plain transformers, one question at a time. Batched or local
    serving would cost much less.
  - Keeping data on one's own machine has no price here.

So B's case rests on being small, open and private, not on accuracy or measured cost.

## 6. What did not work (negative results)

1. **Error-hint tools** (38/95 vs 37; the adoption bar was 42). A1500 repeated the identical failed query
   in 76 of 80 retries. When the tools listed the real column names, it re-checked the table but rarely
   rewrote the query.
2. **Repair demonstrations** (A1500-v4: 35/95). The model learned the habit of re-checking the table after
   an error, and ran out of steps less often (20 vs 25). It still never reached a correct answer after an
   error.
3. **Self-training / RFT** (B: 40/95 on dev, but no final gain).
   - Its own verified attempts beat matched reference data 4 to 0 on dev, a consistent direction but not
     significant (p = 0.125).
   - The kept attempts were mostly simple: only 26 of 229 questions were complex, and only 3 of the 306
     episodes contained a recovery. So self-training reinforced what the model could already do.
4. **No model recovers.** In every arm, zero episodes with a failed query end in a correct answer.

**Why complex queries stay unsolved.**
- Two reasons are measured, in about equal shares. On dev, 15 of A1500's 29 complex failures ran out of
  the eight tool steps. The other 14 submitted a query that runs but answers the wrong question.
- One reason is inferred: the data offers too little signal on hard questions. Self-training only adds
  questions the model already solves, and adding more complex SFT rows (454 vs 100) did not move complex
  accuracy.

## 7. Cost

Rates are HF Jobs t4-small list rates ($0.40/h). These are not invoices.

| Run | Cost |
|---|---:|
| Compatibility checks (G0, G0-02) | $0.10 |
| Dev baselines + A500 | $1.99 |
| A1500 (train + dev) | $2.36 |
| Error-hint re-test | $0.66 |
| A1500-v4 | $2.40 |
| Final: Base + A1500 | $8.28 |
| Q2 sampling (1,154 attempts) | $9.11 |
| B + A+ training and dev | $1.94 |
| Final: B | $6.75 |
| **SQL work total** | **≈ $33.6** |
| Closed-model baseline (OpenAI API, 144 questions) | $0.07 |

Including the earlier shop-only pilot, the total is about $37. One run of the A1500 recipe, training plus
its dev check, cost $2.36.

## 8. Reproducibility

- **Runner:** `scripts/sql_job.py` (`plan`, `preflight`, `launch`, `status`, `pull`); see `docs/sql-runner.md`.
- **Pinned runs:**
  - each run's `REGISTRATION.json` pins the model revision, data and code hashes;
  - each job records its registration SHA256, which is checked against the launch receipt and the file;
  - each `RESULTS.md` records the job ID, checks and cost.
- **Amendments:** `experiments/sqale-sft/AMENDMENT-1…14` record every protocol change, each written
  before the run it affects.
- **Fixed inputs inside each job:**
  - every reference answer is re-checked before the model is used;
  - every evaluation reloads the exact adapter, with all tensors compared;
  - GPU type and SQLite version (3.40.1) are recorded.
- **Tests:** 147 local tests (`pytest`).
- **Logs.** The job logs and outputs are in the dataset repo. The adapter hash `90b04823…` matches the
  published file.

## 9. Limitations and next steps

- **Small sets.** Dev has 95 tasks and one seed, so differences of a few tasks are noise.
- **The eight-step limit.** It is part of the protocol, and it is the main failure. A fair next test is a
  preregistered larger step limit.
- **Signal for hard questions.** Complex questions need a training signal on attempts that fail. RL with
  execution rewards (for example GRPO) is the natural next step, but it was outside our budget and was
  not run. Related reports suggest small gains at this model size, so this is a hypothesis.
- **The closed-model test has limits.**
  - It used one cheap closed model and one sample.
  - That model got Tiny Aya's text tool format, not its native function calling.
  - B's cost was measured on rented GPU time, not optimised serving.
- **BIRD difficulty labels.** In the source file we used, 460 of 500 BIRD questions are labelled
  "challenging", unlike BIRD's published split, so we report BIRD only overall.

## 10. Licences and credits

- **Tiny Aya Global:** CC-BY-NC-4.0 (Cohere Labs).
- **AgentTune and AuditKIT:** Lexsi source-available, non-commercial licences.
- **SQaLe:** Wolff, Gomm and Hulsebos (2026). Its card declares MIT, but the repo has no standalone licence
  file.
- **BIRD Mini-Dev:** CC BY-SA 4.0. Its databases are not redistributed here; see the source.
- **Our code:** Apache-2.0. The adapter is released for non-commercial use, following the base model.
