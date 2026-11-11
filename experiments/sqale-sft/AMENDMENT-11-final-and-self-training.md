# Amendment 11: frozen final evaluation (Base, A1500) and one self-training round on Q2 (2026-11-09)

Written after A1500-v4 (`experiments/sql-a1500v4/RESULTS.md`) and before any final-data model use, Q2
sampling or self-training. The user approved preparing both ("yes") and said credits are not a
constraint. **No upload, job or push is approved by this amendment.** Each registration needs the user's
explicit approval of its SHA256, uploads and storage.

## Part 1: final evaluation

**Frozen selection** (`experiments/sql-final-v1/SELECTION.json`), made from dev results only:
- **Base:** untuned Tiny Aya Global, the "before" model (dev 2/95).
- **A1500:** the best trained model on dev (37/95), adapter `69f0f125…` from job 6af0139be6320c66fad700ed.
- **Not selected:** Pilot (optional, not run), A500 (28), A1500-v4 (35) and the error-feedback tools
  (not adopted). Finals use the original tools.

No final, BIRD or retention task has been read by a model or used for any decision. A self-training
model B may be added later by its own final registration, only under the dev rule in Part 2.

**Sets**, each evaluated once per model with greedy decoding and the A1500 dev protocol (eight tool
steps, 512 new tokens per turn, 8,192 context, table-read-v1 strict plus set and fallback, AuditKIT
agreement):
- SQaLe final: 144 questions on unseen databases.
- BIRD Mini-Dev: 493 questions, 11 databases, CC BY-SA 4.0, final-only.
- Shop retention: dev 40 and multi-hop 40.

**Two jobs**, so the 1.5 GB of BIRD databases are uploaded once:
- `final-bird-sql-20261109-01`: Base, then A1500, on BIRD.
- `final-sqale-sql-20261109-01`: Base and A1500 on SQaLe final, retention dev and retention multi-hop.

**Reporting:** strict (primary), set (BIRD's usual execution metric) and fallback for every set and model,
with the Base → A1500 change, whatever they are. Nothing is selected, tuned or rerun on final data.

**Scoring time limit (prospective fix).** BIRD's slowest reference queries take 2.7-2.95 s locally. A
second local check timed one out at the 5 s limit, so timing alone can cross it. Scoring re-runs each
reference several times per episode under that same limit. On a slower or busy machine that could crash
a run hours in. Worse, the ordering check could quietly mark a reference ineligible and score the
episode wrong. So from this amendment on:
- every **scoring** execution gets a **30 s** limit (`src/ayadb/scoringlimits.py`): the reference, the
  submitted and fallback queries, the ordering probes and the pre-model reference check;
- reference results are cached per frozen database file;
- the 50-million-operation cap is unchanged, so runaway queries stop at the same point on any machine;
- the agent's own tools keep their 5 s limit, so what the model sees is unchanged.

Re-scoring all 570 earlier dev episodes (Base, Pilot, A500, A1500, hints, A1500-v4) with the changed
scorer gives **0 differences**, so no earlier result changes. The frozen pre-amendment registrations are
tested by their SHA256 without today's source pins, as before.

## Part 2: one self-training round (B) with its control (A+)

**Sampling.** A1500 (exact step-180 adapter) attempts every effective Q2 question: **579 questions × 2
samples**. Q2 is held out of all training, dev and final. Sampling uses temperature 0.8 and top-p 0.95
(the runner's fixed value), seed 42, and the dev protocol and tools otherwise. It runs as three
registrations over contiguous thirds of Q2, `q2sample-a1500-sql-20261109-p1`, `-p2` and `-p3`, so they can
run in parallel. This is the plan's B with 2 samples instead of 4, a lighter version chosen for the deadline.

**Filter** (`scripts/build_q2_selftrain.py`, after sampling, no model):
- An attempt is kept only if `selftrainfilter.candidate_eligibility` passes. That means one well-formed
  call per turn, calls matching the tool log, no invented or mismatched tool output, a complete protocol,
  and a guarded strict success under table-read-v1.
- Exact duplicate call sequences per question are kept once, at most 2 per question.
- Each kept attempt is rebuilt as a training episode with `make_episode`, on the same database. Calls
  whose tool result was an error stay as **untrained context**; every other call is a target. The rebuilt
  observations must equal the sampled ones, the encoding must fit 4,096 tokens without truncation, and the
  AgentTune replay audit must pass.

**Control A+:** gold-path episodes (list tables, describe the gold tables, the reference query, submit) for
the same retained questions. Each question's existing alternative phrasings may be used, but no other
questions. The two arms are matched on **trained target tokens** by `continuation.match_control`, with
the frozen minimum-question-drop fallback if needed.

**Gates** (from the review, unchanged; both must pass or neither continuation trains):
- **G2:** at least 25% of the 579 Q2 questions get an eligible verified attempt.
- **Size:** unique verified B targets per pass must be at least 10% of A1500's unique training targets
  per pass (330,742, so at least 33,075).

**Continuations** (built after sampling, one registration each, B and A+ in parallel):
- Both start from the identical A1500 adapter and differ only in data.
- Common settings: `trained-positions-v1`, LR **1e-4** cosine (half of A's 2e-4, as a continuation from a
  trained adapter), accumulation 16, batch 1, two whole passes, no validation split, seed 42, same NF4/LoRA.
- Each run is then screened on the same 95 dev tasks with the original tools, after an exact reload.

**Decision rule (fixed now):**
- **B is added to the final evaluation** by its own registration only if its dev strict score is at
  least **38/95** (above A1500's 37).
- A+ is a control and never a finalist.
- The scientific comparison is B vs A+ on dev, paired and descriptive. A tie is reported as "no
  difference detectable at this size".
- If G2 or the size gate fails, nothing trains and that is reported.

**Diagnostics:**
- Q2 yield: questions with at least one success, by query shape.
- Kept attempts that contain a recovery (a failed call followed by success).
- Dev recoveries and complex tasks for B and A+.

## Money and time (rates $0.006667/min, t4-small; forecasts, not caps)

| Registration | Expected | Hard timeout / cap |
|---|---|---|
| final-bird (Base 2.3 h + A1500 9.2 h) | ~11.6 h, ~$4.6 | 16 h, $6.51 |
| final-sqale-retention (Base 1.0 h + A1500 4.2 h) | ~5.3 h, ~$2.1 | 8 h, $3.31 |
| Q2 sampling, each of 3 thirds (386 sampled episodes) | ~7.2 h, ~$2.9 | 12.5 h, $5.06 |
| B and A+ continuations (later) | ~3 h each, ~$1.2 each | set when built |

Uploads are private:
- final BIRD job: about 1.64 GB, mostly the BIRD databases;
- other final job: about 0.2 GB;
- each sampling third: about 0.2 GB, including the 121 MB A1500 adapter.

Storage is at most 30 days without renewed approval. Concurrent jobs on one account may queue; queued
time is not billed.

## Unchanged

No automatic retry. GRPO, teacher models, other datasets, push and publication remain unapproved. The
spent shop test stays closed. Results are reported as they are, including negatives.
