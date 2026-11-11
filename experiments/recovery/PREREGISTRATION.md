# Recovery experiment: preregistration (2026-11-05, before arm B is trained and before any drift or mh run)

> Ran once on 2026-11-05 at commit `7bb2389` (pushed before the run); all six jobs valid. Results:
> `RESULTS.md`. The plan below is unchanged.

The pilot's one matched-budget follow-up (pilot plan: "trajectory quality or error recovery").
Nothing below has been run. Running it needs the user's approval: `python scripts/hf_job.py recovery`.

## Why

On the sealed test, 11 of the 20 held-out episodes hit an SQL error and none recovered: the pilot
model re-described tables, repeated the same query and ran out of tool rounds (`RESULTS-test.md`).
All 66 recovery demonstrations in the pilot data show a *discovery-phase* mistake: a table or column
name guessed before the schema is read, followed by a genuine error. In every one of them the trained
next turn is `describe_table`. That is the behaviour the model shows: after an error, describe a
table. It never saw an error after the schema had been read, or a failing query being repaired.

**Question.** At a matched budget, do query-repair demonstrations (the query fails after the schema
was read; the trained next turn is the repaired query) make the agent succeed where the pilot's
discovery-phase demonstrations do not?

## Arms: one variable

| | Arm A (the pilot, unchanged) | Arm B |
|---|---|---|
| episodes | `data/sft_pilot/episodes.jsonl`, sha256 `5cb48a81...` | `data/sft_recovery/episodes.jsonl`, sha256 `d0bf1bb35f551e28db0a539a9480aab07ee030791fc8a4550bd7dae4a3de0180` |
| tasks, databases, order | 235 | the same 235 ids, databases and order |
| episodes with a mistake | 66 (the pilot's `mistake_schedule`) | the same 66 episodes |
| mistake | before the schema is read: 37 wrong table name in `describe_table`, 29 guessed column in `run_sql` | after every `describe_table` of the plan, in a `run_sql`: undefined alias 32, wrong column 30, syntax 21, ambiguous join key 7, missing join 5; 37 episodes fail once, 29 fail twice (the first repair has another mistake) |
| trained turn after the error | `describe_table` | the repaired query |
| trained turns | identical: every turn with `train: true` is the same string in both files, and the 169 mistake-free episodes are byte-identical | |
| target tokens | 38,920 | 38,920 |
| all tokens | 359,830 | 363,383 (+1.0%) |
| training | `python -m ayadb.sft` defaults, as in the pilot: nf4, LoRA r=16, 2 epochs = 56 steps, seed 42, the same 12 validation episodes | identical; only `--episodes` differs |

Mistaken turns are context in both arms, never a training target. So the two adapters differ only
in the context in front of the trained turns of 66 episodes: which error the model saw, and so what
its trained next turn is an answer to.

Built with `python -m ayadb.sftdata --out data/sft_recovery --recovery query`. Each episode was
replayed through AgentTune's loop on a fresh database, like the pilot's. None was rejected, and every
mistaken query returned an SQL error. The default path still rebuilds the pilot file byte for byte
(rebuilt 2026-11-05: `5cb48a81`). Arm B's wrong column names are disjoint from the drift renames
below (tested).

## Evaluation splits (generated and pinned now)

| split | what | tasks | sha256 (LF) |
|---|---|---|---|
| **drift** (primary) | the frozen dev questions and rows, with five columns renamed: `full_name`→`display_name`, `city`→`home_city`, `unit_price_cents`→`list_price_cents`, `order_date`→`placed_on`, `opened_at`→`opened_on` | 40 (5 per family) | `eaf5cbaf32120ca4f991629c379fd57d7fbd6edf0242e63254405cd9d9907ace` |
| **dev** (guard) | the frozen dev set; arm A's run exists (`sft-nf4-r0`, 40/40) | 40 | `7798e6fe...` (frozen 2026-11-04) |
| **mh** (secondary) | the three held-out multi-hop families on three new databases (seeds 404-406, 6 per family per database) | 40: cancel_leader_tickets 4, largest_order 18, segment_aov 18 | `d71da41d178eaecc64ccc96e4fbf13ad3bfc7b2056b308e7e96d41a559fcf2e5` |

- **drift**: the rows are unchanged, so the gold answers are too (the renamed reference queries
  are asserted to reproduce them). Every family's training query uses at least one renamed column,
  so every training query fails on this database (tested: 40/40 return `no such column`). An agent
  passes only by using the names it reads with `describe_table`, either from the start or after the
  error. Both arms' trained queries use the training names, so drift forces an error on any episode
  that writes its query from memory.
- **mh**: new databases and answers. The question wording repeats the sealed test's for these
  families, because there are only a few possible wordings (2 years for one family, 3 segments x 2
  years for another). The sealed test's database is not rebuilt, and the sealed test stays used up.
  The task file is not printed or read before both mh runs finish.
- **Disclosure**: arm B's design was motivated by the post-hoc reading of the sealed test (alias
  errors, no recovery). No test task, trajectory or query was used to build it: its mistakes are
  generated from the training families' own queries.

## Runs (one each)

| tag | adapter | split |
|---|---|---|
| `drift-sft-a` | A (`6f2827ae...`, the pilot adapter) | drift |
| `mh-sft-a` | A | mh |
| `drift-sft-b` | B | drift |
| `dev-sft-b` | B | dev |
| `mh-sft-b` | B | mh |

Protocol unchanged from the pilot: Hugging Face Jobs `t4-small`; `multistep-close`, greedy,
8 tool rounds, 512 new tokens per turn, nf4 base, the same system prompt, tools, verifier and
library versions. There are no repeats, because the dev A/A run showed 0 flips. `scripts/hf_job.py recovery` trains
arm B while arm A's two evaluations run, then starts arm B's three evaluations once training passes.

## Outcomes

- **Primary**: verified success on drift, `drift-sft-a` vs `drift-sft-b`, exact McNemar on paired
  task outcomes.
- **Mechanism** (every run): episodes with an error; recovered / errored; for each failed call,
  what came next (an edited query that ran, an edited query that failed again, an identical
  repeat, a schema lookup, an answer, or nothing because the budget ran out); error kinds; on drift,
  how many first queries replay a training query (memory) rather than the described schema.
- **Guard**: frozen dev, `sft-nf4-r0` (arm A) vs `dev-sft-b`.
- **Secondary**: mh, arm A vs arm B, verified success and the same mechanism measures. It is
  descriptive: 9 of the 20 held-out sealed-test failures were premature one-hop answers, which
  neither arm targets.

All of this is produced by
`python -m ayadb.compare remote/hf --split recovery --train-report remote/hf/recovery/sft-recovery/train_report.json`,
which was dry-run on renamed pilot dev runs only.

## Interpretation (fixed now)

- **Primary**:
  - Arm B better with p < 0.05: arm B is more robust to schema drift. Whether that comes from
    recovering after the error or from reading the schema first is read from the mechanism measures.
  - Arm A better with p < 0.05: query-repair demonstrations hurt.
  - Otherwise: no evidence of a difference.
- **Drift that does not test recovery**: if fewer than 10 drift episodes have an error in both
  arms, drift measured schema grounding, not recovery. Recovery is then answered only descriptively,
  by mh.
- **Guard**: arm B passes with at least 38/40 on the frozen dev set and no significant loss
  against arm A.
- **Adoption**: arm B replaces the pilot adapter as the submission model only if the primary is
  positive and the guard passes. Otherwise the pilot adapter stays, and the result is reported as
  it is.
- **Recovery on new question types**: claims come only from mh and are labelled secondary and
  descriptive.

## Validity and reruns

A run is invalid (reported, not interpreted) if:

- a job fails or reaches its time limit;
- a run's tasks sha256 is not the pinned LF hash;
- arm B's episode sha256 is not `d0bf1bb3...`;
- arm B's training loss is non-finite;
- arm A's adapter sha256 is not `6f2827ae...`;
- an orchestration check reports a tool-result turn the environment did not produce.

An invalid stage may be rerun once with identical settings, with the reason recorded here.
`AYADB_RECOVERY_RERUN=1` reruns the whole stage, and `python scripts/hf_job.py recovery_eval`
reruns only arm B's evaluations. A valid result is final: nothing is tuned or selected on drift or mh.

## Cost

| | expected | cap |
|---|---|---|
| arm B training (as the pilot: 56 steps, about 29 min) | ~$0.21 | 1.9 h, $0.77 |
| each of the five evaluations | $0.15-0.35 | 3.4 h, $1.37 |
| **total** | **about $1.2-1.8** | **$7.60** |

Each cap includes the GPU guard and setup. HF spend so far is about $1.85.
