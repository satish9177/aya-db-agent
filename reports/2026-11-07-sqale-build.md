# SQaLe local build completed (2026-11-07)

**Approved free preparation is complete. No model inference, sampling, optimizer step, paid job,
upload, publication, commit or push was performed. The submission model remains the original pilot.**

The independent review and handoff now contain the four agreed refinements: identical retained Q2
questions and starting A adapter/settings for A+/B, exact supervised-target matching with separate
token/cost accounting, preregistered secondary perturbation on logged queries after validation, and a
pre-sampling forecast followed by a frozen 10% verified-data gate. A tie means no difference detectable
at this size. AgentTune SFT feasibility is documented separately below.

## Prepared data

Private artifacts: `data/sqale/build-v2/`. The main manifest is
`712e8e5301f512c1822f567c8c151c62213a4bc7f62aa60436567b750ff923e8` (SHA256).

| Pool | Retained | Source schema groups | File |
|---|---:|---:|---|
| SQaLe training | 4,765 questions/episodes | 326 | `train.tasks.jsonl` |
| Converted shop training | 235 episodes | 9 database seeds, one shop schema | included in training episode files |
| Full training stage | **5,000 episodes** | SQaLe groups plus shop | `train-5000.episodes.jsonl` |
| Q2, held out of A | **600 questions** | 35 | `q2.tasks.jsonl` |
| Dev | **100 questions** | 95 | **`dev-100.tasks.jsonl`** |
| Final, still model-unseen | **150 questions** | 94 | `final.tasks.jsonl` |

Train, Q2, dev and final source schema groups are disjoint. Source phrasings stay with their question.
The schema screen found one near-duplicate link and grouped it before allocation. It uses exact DDL
or >=90% table-name containment with at least three shared tables; this is not proof of no semantic
duplicates. SQaLe difficulty labels in the selected training questions: 1,685 simple, 1,685 moderate,
1,395 hard. These are source labels, not independently verified difficulty judgments.

| Nested stage | Shop episodes included | Input tokens per pass | Supervised targets per pass |
|---|---:|---:|---:|
| 500 | 235 | 893,422 | 111,469 |
| 1,500 | 235 | 2,840,813 | 347,333 |
| 5,000 | 235 | 9,568,488 | 1,144,502 |

All three stages are nested and retain the same 235 shop conversions. Maximum input length is
**4,072 tokens**, under 4,096; maximum tool calls is eight. There are 1,409 new SQaLe alias-repair
demonstrations with executed SQL errors, plus 66 converted shop error demonstrations (1,475 total).
Injected failed calls and all tool observations are masked; only the trained assistant targets,
including end tokens, receive labels. These are expert demonstrations, not measured student recovery.

The token totals above are data counts per pass, not consumed optimizer-token totals. A's cumulative
training target count, excluding validation, must be recorded before Q2 sampling. The continuation
gate compares unique verified B targets before repetition with 10% of that actual denominator.
Neither the forecast nor the gate can be evaluated yet: A has not been trained or dev-scored.

## Verification and new format

`sql-submit-v1` is separate from the unchanged legacy environment. It adds terminal `submit_sql`, logs
all submissions, independently executes full result sets, preserves duplicates and relevant ordering,
and reports the last-successful-query fallback plus BIRD's exact set-result comparator separately.
Limits fail closed: 10,000 rows, 8 MiB result payload, 1 MB per SQLite value, five seconds/50M VM ops.
The agent preview is capped; the scorer never compares matching truncated previews.

All **8,005 main scripted expert replays** passed the real AgentTune loop and AuditKIT full-result
re-scoring. Five additional dev-preparation replays also passed: **8,010/8,010** altogether. The main
submission log contains 17,143 builder/replay submission records; the five dev additions have their
own submission log. These are scripted submissions. No model rollout has been generated.

The complete test suite passed **59 tests**. Checks include unrelated successful SQL, unsubmitted
answers, matching truncated previews, duplicate rows, ordering, numeric matching, write attempts,
timeouts/result overflow, terminal submission, injected-error/observation masks, AuditKIT negative
outcomes, exact target-budget masks and the AgentTune SFT factory. The default existing pytest temp
directory was locked after the permission-mode change; the successful command was:

```powershell
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/pytest-sqale-final --tb=short
.venv/Scripts/python.exe -m ayadb.sqaleverify data/sqale/build-v2 --amended-dev
```

Artifact verification passes with no remaining count shortfall. It checks hashes, database hashes,
split isolation, distinct questions, terminal submissions, token limits and nested stages. The 100 dev
questions also have 100 distinct (schema, normalized SQL) pairs and at most two questions/schema.
The old pilot episode hash, frozen dev hash and legacy environment/scorer/training files are unchanged.
The spent shop `data/test.jsonl` was not read, regenerated, evaluated or selected on by this build.

## Technical failure and preparation amendment

Build v1 stopped on a source integer larger than SQLite's 64-bit range. Its log and artifacts are
preserved; the failure was not a killed process or a model experiment. The corrected handler rejects
that schema without coercion. Build v2 reused 2,497 completed deterministic expert replays only after
matching complete prompt text, arguments and observations and rechecking full-result correctness.
New replays and cache provenance are saved. This is not reuse of model test trajectories.

Only 95 dev schemas materialized unchanged. Before any model run, amendment 1 added a second distinct
question from five existing dev schemas to reach 100. The original 95-task artifact and base manifest
are unchanged; use **`dev-100.tasks.jsonl`**, SHA256
`78e942ff20c0621907f6e14997e33f72348f839b03c17c24f798ebc73b8d68d8`.
The amendment and its manifest preserve the original hashes, selection rule, additional IDs and checks.
Final prefers one question/schema, then uses second questions to reach 150. Use schema-grouped
uncertainty; neither pool is 100/150 independent databases.

## Limits and approvals

This is a **filtered, sliced-schema SQaLe subset**, not the full published benchmark. Only the first
of sixteen training-schema shards was downloaded. The strict custom native-SQLite materializer rejected
425 of 976 downloaded schemas (43.5%): 164 train, 27 Q2, 117 dev, 117 final. This compatibility filtering
can bias the retained sample. The original source FK imperfections remain; no rows were repaired.
Execution verification establishes consistency with the reference SQL, not the correctness of every
natural-language annotation.

Other exclusion records: 5,042 placeholder phrasings and 208 question exclusions, including 46 gold
execution failures, 45 full/sliced result mismatches, 69 tool-budget cases and 11 contexts over 4,096.
The separate 25 repair-variant exclusions kept their eligible gold paths; do not sum all log scopes
as if they were unique rejected questions.

Both source cards declare MIT, but neither pinned repository contains a standalone licence file;
upstream SchemaPile terms remain unresolved. See the source-notices report. BIRD data, other teachers,
model training/evaluation, commits/pushes and public artifacts remain unapproved.

AgentTune already handles rendering/rollouts and AuditKIT already handles recorded evaluation. The
supported AgentTune custom SFT factory preserves a supplied model object and exact masks in a CPU-only
spy test and can accept an already prepared QLoRA model. Its stock factory is not a drop-in. Actual Tiny Aya gradients, memory, reload and inference
through this route still require a separately approved smoke run; no training benefit is established.
Perturbation is named in the preregistration and deferred until its own validation and logged model
queries exist. No perturbation result is claimed.

## Records

- `experiments/sqale-sft/PREREGISTRATION.md`: study draft; 10% gate frozen, no model run approved.
- `experiments/sqale-sft/DATA-FREEZE.md`: active artifact hashes, counts and preparation provenance.
- `experiments/sqale-sft/AMENDMENT-1-data-eligibility.md`: transparent 95-to-100 dev amendment.
- `experiments/sqale-sft/PAID-STAGES-DRAFT.md`: proposed caps and readiness gaps, no spending approval.
- `data/sqale/build-v2/verification-dev100.json`: final successful artifact checks.
- `data/sqale/build-v2/dev-100.manifest.json`: amended dev provenance and AuditKIT results.
- `data/sqale/build-v2/code-snapshot.zip` and `environment.json`: build code and environment records;
  amendment code has its own recorded hash.
- `reports/2026-11-07-agenttune-sft-feasibility.md`: supported SFT path and limits of the spy test.
- `reports/2026-11-07-sqale-source-notices.md`: pinned sources and licence inspection.

Next: prepare a concrete, individually registered throughput/SFT compatibility smoke, with billing
account and launcher/checkpoint requirements resolved, then obtain the user's explicit approval.
The roughly $3 estimate is not approval and nothing paid has been launched.
