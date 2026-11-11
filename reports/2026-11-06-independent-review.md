# Independent review and recommended plan (2026-11-06)

Status: review by Claude Opus 5.5. No paid job, download, training or publication was done for it. It
challenges `reports/2026-11-05-training-methods-and-data-plan.md`; where the two differ, this is the
current recommendation. **Only the scope listed in `HANDOFF.md` is approved.** Every cost below is an
estimate, not a measurement, unless it says "measured".

Amended 2026-11-07 with the user's agreed refinements: identical retained questions and exact target-token
matching for A+/B; a secondary, validated perturbation diagnostic; a continuation-data size gate; and an
AgentTune SFT-backend check. Subsequent explicit approval covers BIRD Mini-Dev final-only preparation,
the additive SQL runner/launcher, and two local commits. Every upload and paid/model run still needs
separate approval. See `HANDOFF.md` for the current authority, rather than historical scope below.

Active v3 preparation is recorded in `reports/2026-11-07-sqale-v3.md` and
`experiments/sqale-sft/AMENDMENT-2-v3-local-fixes.md`. It has 4,888 training episodes (4,653 SQaLe plus
235 shop), 581 Q2, 96 dev and 144 final questions, plus unchanged shop dev 40/mh 40 retention sets in
`sql-submit-v1`. All 850 SQaLe held-out primary prompts and 80 retention prompts were individually
read. Exclusions are logged and not replenished. Use `data/sqale/data-build-v3/` and its v3 freeze.
The older 5,000/600/100/150 build/report and amendment-1 dev top-up are preserved as historical v2.
These are filtered, sliced-schema preparation counts, not model results; no paid/model run occurred.
V3 passes 71 tests and 5,789/5,789 AgentTune/AuditKIT scripted checks, with 17 fresh changed-prompt
replays and exact validated reuse for the remainder. Full artifact verification and all 5,903 v2 byte
checks pass. Stage targets/gates and final hashes are in `experiments/sqale-sft/DATA-FREEZE-v3.md`.

**Current run-input correction (amendment 4):** preserve v3 bytes, but exclude one known training
fragment and three scan-order-dependent dev/Q2 references before any model use. Effective full corpus
is **4,887**, dev **95**, Q2 **579**, SQaLe final **144**, retention 40+40. Compute A's denominator from
its actual post-validation training split: unique targets/pass **107,381 / 330,742 / 1,067,919** for the
500/1,500/full source stages; B minima **10,739 / 33,075 / 106,792**. The runner freshly verifies factory
encoding and persists exact membership. The historical full-pool gate figures include validation and
are superseded. See `experiments/sqale-sft/RUN-INPUTS-FREEZE-v1.md`.

**BIRD final-only preparation complete (amendment 3):** pinned 2026 Mini-Dev, 11 full SQLite databases,
**493/500** retained with seven logged execution-cap exclusions, all retained references independently
recomputed. Official baseline includes evidence; we preserve it verbatim under a documented External
Knowledge header, using our own SQL tool protocol. No BIRD training, tuning, model selection or model
evaluation has occurred. See `reports/2026-11-07-bird-final-build.md` and its licence/source report.

The new runner includes an expected-hf-user account/price guard, longer registered wall limits,
complete optimizer checkpoints and account/run-specific outputs. One **$3 G0 protocol** supersedes
the old broad engine-smoke proposal: longest-example AgentTune factory training for two steps, resume
to three, then Base/Pilot/reloaded-checkpoint inference on three fixed dev questions. It tests HF NF4,
not vLLM. The user subsequently approved and launched that exact one job from clean commit 413dff8.
It **failed with CUDA OOM in the first forward, zero optimizer updates and no inference**. No retry or
second job was started. Runtime 136.66s; compute approximately $0.02 by rate, exact invoice and peak
counters unavailable. See `experiments/sql-g0/RESULTS.md`; the source registration remains frozen.

Amendment 5 prospectively adds the user-table-read guard, clean strict self-training filter predicate
and a descriptive manual audit after the first full dev screen. All 6,278 active reference queries read
user tables; 103 local tests pass. G0's separate no-table diagnostic reports all three inference
conditions **not run**, zero submissions, without altering old scores. The 500-stage draft is blocked:
zero successful G0 updates/inference gives no defensible completed-run cost forecast for 500/1500.

**Latest free fix, amendment 6:** native Cohere2 selected-position projection preserves logit_scale,
shifted labels and accumulation normalization. Tiny random CPU loss/all-LoRA-gradient equivalence,
checkpointing/multi-turn masks and base-class/PEFT last-position prefill checks pass; 114 tests total.
Memory peaks now persist after load/updates/OOM, and the launcher sets expandable_segments. A distinct
`experiments/sql-g0-02/REGISTRATION.md` proposes the same $3/T4/two-hour protocol plus one short GPU
loss comparison before training. The user later approved that exact run and it **passed** (job
6aef547de6320c66fad5ff75, ~$0.08 by rate): loss agreement 6e-8, three updates peaking at 8.91 GiB of
14.74 GiB, resume, reload and nine inference episodes. See `experiments/sql-g0-02/RESULTS.md`. The 500
draft now has forecasts from these measurements but is not a runnable or approved registration.

## Verdict

Keep the direction, cut the study. Broader training data and a stricter scorer are right: the project's
own results show a coverage problem (77/77 on trained question types, 0/20 on new ones). The five-branch
method comparison is not: it does not fit 8 days for one person, and GRPO and SDPO have the weakest
evidence at Tiny Aya's size.

## Model choice

The hackathon allows only Tiny Aya (Base, Global, Fire, Water, Earth) and Aya Expanse 8B/32B as the
submitted model. Other models may only be a teacher, validator or judge.

- **Smaller model first: no.** Tiny Aya is the smallest allowed, so a smaller proxy must be non-Cohere:
  different tokenizer and chat format, so the engineering is done twice. Method rankings also flip with
  size (SDPO beats GRPO on 8B models but loses on Qwen2.5-1.5B; SQL-Trail's RL gain is about zero at 3B and
  +5.5 BIRD points at 7B). Apply "start small" to experiment size on the same model instead.
- **Bigger model: last, as a comparison.** Aya Expanse 8B costs about 2.5x per token and has the same 8K
  context. Evaluate it untuned, and train it once with the winning recipe if money and time remain.
- **Main model: Tiny Aya Global.**

## Corrections to the 2026-11-05 note

| Note says | Source says | Consequence |
|---|---|---|
| [SQL-Trail](https://arxiv.org/html/2601.17699v1) is a strong precedent for the RL branch | Setup is right: Qwen2.5-Coder 3B/7B/14B, 1,000 SFT trajectories distilled from Claude-Sonnet-3.7, 1,027 RL questions, 6 rollouts per question, six weighted rewards. But Table 7 at 3B, SFT only vs SFT+RL: 83.1 vs 84.6 (Spider dev), 82.9 vs 84.3 (Spider test), **55.7 vs 55.2 (BIRD dev)**. At 7B BIRD dev is 58.7 vs 64.2; at 14B 64.8 vs 66.7. Compute is not reported. | Weak precedent at our size. |
| [SDPO](https://arxiv.org/html/2601.20802v2) is the best stretch method | "primarily applicable for RL-training stronger base models, while it can underperform GRPO on weaker models"; it underperforms GRPO on Qwen2.5-1.5B. The tool-use test had no rich feedback. Top-K=100 means memory is not the blocker. | Tiny Aya recovered 0/17 and 0/11 after real SQL errors here, so it does not use feedback in context, which the self-teacher needs. **Drop SDPO.** |
| SynSQL-2.5M is the first-choice data | [SQaLe model card](https://huggingface.co/trl-lab/qwen3.5-2b-grpo-sqale): the same 2B model and recipe on three corpora (length curriculum also differs). SynSQL is lowest on both: 50.7 vs 66.3 (SQaLe, SQaLe test), 44.3 vs 52.3 and 54.7 (SQaLe, BIRD; BIRD dev). SynSQL and BIRD schemas hold 5-10 tables. | **Prefer SQaLe as the training source.** SynSQL also needs the user's approval under the standing rules. |
| SQaLe agent recipe | Numbers are right: 1,800 x 18 x 8 = 259,200 episodes, RL only. Left out: up to 12,288 tokens per episode at 16K context, async vLLM rollouts, base model already at 19.3% on BIRD dev, schemas sampled at 4-64 tables. | Tiny Aya is documented for 8K. Use small schema slices. The budget covers about 1% of those rollouts. |
| Budget allocations | Measured: a 40-task dev eval takes about 1,410 s on a T4, about 35 s per episode on a 5-table database. AgentTune's GRPO runs episodes one at a time when a trainer is attached (`agenttune/agentic/rollout_engines/rollout_factory.py:1733`), and its vLLM path sends one prompt per call (`:1951`). | $12 of GRPO on a T4 is about 3,000 episodes. A null result at that size shows nothing. Evaluation and sampling were under-budgeted. |

Confirmed: SQaLe queries and schemas are MIT, SynSQL is Apache-2.0, BIRD Mini-Dev (2026 update, 500 aligned
IDs) and BIRD train-filtered (6,601 instances) are CC BY-SA 4.0.

Not re-checked: SDFT, SERL-SQL, Progress-SQL, SQL-Zero, ReViSQL's licence, TrustSQL. The plan does not depend on them.

## Limits of this review's own evidence

- **"Diverse SFT on SQaLe will help" is a hypothesis.** The SQaLe figures are the dataset authors' own, for
  Qwen3.5-2B trained with RL at huge scale, and its test set favours its own corpus. The gain over the
  untrained model was +27.6 (SQaLe), +15.3 (BIRD), +12.0 (SynSQL) on the SQaLe test; the gap between SQaLe
  and SynSQL is about 16 points. None of this is SFT on Tiny Aya. Gate G1 tests it.
- The gold-path trajectories below contain no exploration or reasoning, so they may teach little SQL. If G1
  fails, the fallback is the Apache-2.0 SQaLe-trained Qwen3.5-2B as a teacher (needs approval, and protocol conversion).
- BIRD Mini-Dev's 11 databases are now locally verified. The runner reports duplicate-preserving strict
  and set-style scores separately; our tool prompt/limits and seven exclusions are documented, so this
  is not a claim to reproduce the official leaderboard setup exactly.

## Experiment matrix

All models start from Tiny Aya Global and run on one GPU type, engine and precision.

| Arm | What | Why |
|---|---|---|
| Base | untuned, plus a 1-example prompt on dev only | the "before" number |
| Pilot | existing adapter `6f2827ae...` | shows the 8-template SFT does not transfer |
| A | SFT on SQaLe gold-path trajectories, repair demos from real execution errors, and the 235 shop episodes converted to the new format | broader data |
| A+ | the same A adapter continued on gold-path trajectories for exactly the questions B retains, with exactly B's target-token budget | control |
| B | A plus one self-training round on Q2: 4 samples per question, keep verified ones, at most 2 per question | learning from its own attempts |
| Reference | Aya Expanse 8B untuned; one closed model via API | headline and ROI |
| C | one GRPO run | only if G0 and G3 pass |

Q2 is held out of A's training. First verify B's samples and freeze the retained question IDs; A+ uses
exactly those IDs. Both continuations start from the identical A adapter with the same optimizer, learning
rate, seed, masking, precision and other training settings. Match **target tokens exactly**: count only
tokens with a training label (including trained end tokens), after masking and across all updates. Equal
steps or equal total input tokens are not a match. Extra existing phrasings or deterministic schema slices
of the same retained questions may supply enough control data; do not introduce other questions. Match
globally using whole examples and at most one partial mask overall, never one partial mask per question.
If full coverage is infeasible, amendment 4 freezes `minimum-question-drop-v1`: deterministically drop
the fewest control questions from **both** arms, report IDs and question/episode/target coverage,
then recheck the size gate. Its minimum cardinality is checked against an exhaustive small-set oracle.
Report unique and repeated data, input,
target and generated tokens, teacher/verification work and all sampling time/cost separately.

Before paying for Q2 sampling, use A's dev score and observed trajectory lengths to estimate the verified
episodes and target tokens likely to remain after verification and the two-per-question cap. Treat greedy
dev accuracy as a planning proxy, not an estimate of pass@4 under independent samples. Train neither A+
nor B unless unique verified B data **per pass** contains at least **10% of A's unique target tokens
per pass**, after deduplicating both sides and excluding validation. This threshold is frozen in the
amended preregistration. Neither side is multiplied by epochs/updates/repeated copies or padding.
Do not count extra control phrasings toward this gate. If a
comparison ties, report **no difference detectable at this size**, not equivalence between methods.

Upgrades in order: (1) data in stages of 500, 1,500, then the cleaned v3 full pool of 4,888 if the proposed
five-point stage rule passes (4,887 eligible full-stage examples); (2) the 144-question SQaLe final plus
493 prepared BIRD final questions for at most three finalists, plus retention; the data preparation is
approved, **model evaluation is not**; (3) up to 579 instead of 300
self-training questions; (4) a second seed for the winner
and its control; (5) training Aya Expanse 8B once; (6) the GRPO run, capped at $15 and 48 hours, starting by
about Nov 9-10. Skip GRPO on a T4 at any budget (48 hours is about 2,900 episodes).

Dropped: SDPO, paper-scale RL, SynSQL as main data, When2Call, broad retention suites.

**Effective evaluation sets.** Dev: 95 tasks from SQaLe test schemas, for selection only. Final, once
selection is frozen and evaluation is separately approved: 144 other-schema SQaLe tasks plus 493 BIRD
Mini-Dev tasks (**637**, reported by corpus, not silently pooled). Retention: shop dev 40 and mh 40,
reported separately. `data/test.jsonl` stays closed. Shared-database dependence means question counts
alone do not establish the detectable effect size; report grouped uncertainty.

**Train on SQaLe only,** so BIRD Mini-Dev is an out-of-distribution test. Results on sliced schemas are not
comparable with the SQaLe card's full-schema numbers; say so in the write-up.

## Task pool

- Pin dataset revisions, read their licence files, and record the SQaLe package version if it is used.
  Whether SQaLe's MIT covers redistribution of SchemaPile-derived DDL is unverified.
- Split by schema. Check SQaLe train and test for near-duplicate schemas (table-name overlap; the schemas come from GitHub).
- Keep all phrasings of a question together, and drop placeholder phrasings.
- Slice schemas like the SQaLe sampler: start from the gold tables, grow along foreign keys, shuffle. Start
  around 4-16 tables. `sft.py` already refuses episodes over `--max-len 4096`, and Tiny Aya's config has
  `sliding_window 4096`, so stay at or under 4096 tokens per example unless longer is tested.
- Recompute every gold result yourself (stored results are truncated). For SQaLe training preparation,
  drop empty, all-NULL and single-0 results. BIRD is a fixed final benchmark: keep legitimate empty/zero
  answers, use its full databases and log every exclusion under the frozen execution caps. The guard
  at `src/ayadb/env.py:180` can kill valid queries on bigger databases. Log all
  exclusions before any model runs.

## Scorer

The current check (`src/ayadb/metrics.py:80`) is a correct value plus any query that ran, so the query is not
tied to the answer. The one-value `ANSWER:` format (`src/ayadb/tasks.py:323`) cannot express multi-row results.

- Add a `submit_sql(query)` tool that ends the episode, as a new format. Keep the old format and tools
  unchanged so pinned results still reproduce.
- Score by running the submitted query on a separate connection with no 20-row cap and its own time and row
  limits, then comparing the full result to the gold query's.
- **Primary metric: strict, the query must be submitted. Also report a fallback score that uses the last
  query that ran.** Base and the pilot never learned `submit_sql`, so strict scoring alone would count
  learning the format as learning SQL, and the before/after gain would be inflated.
- Report BIRD's standard metric for comparability (check the official script) and a strict comparison that
  keeps duplicates.
- Log every submitted query, including failed and repeated submissions, with task ID, call order and
  execution/termination status. Database perturbation is a **secondary diagnostic**, named in the
  preregistration and run later on these logged queries; it does not gate or alter the primary metric.
  Validate the perturbation procedure on known-equivalent and known-wrong queries before applying it to
  model outputs. Preserve constraints, record perturbation seeds and failures, and choose any row-change
  rate through that validation rather than assuming 20-30% is established. Multiple instances follow the
  test-suite idea of Zhong et al. 2020; passing is stronger evidence, not proof of equivalence.
- Test repair separately: start from failing queries that actually ran (mined from approved dev rollouts
  or varied wrong-column/table, ambiguous-column and aggregate-misuse mutations), and score whether a
  correct result follows. Do not use the fixed bad-alias demos in v3. Report this apart from the
  natural error rate.
- Shop retention: check that each reference SQL's result equals the stored gold value before relying on
  result comparison. Rounding or extra columns may need the old tolerance rule.
  V3 preserves original dev 40/mh 40 IDs/seeds/pins, checks original and scalar-projected references against
  stored gold, and evaluates the scalar the question asks for. Log every projection/tolerance decision.
- Preserve correct tied-row permutations: mark tied/unresolved non-LIMIT orderings unordered, using
  native SQLite ordering probes, and exclude ambiguous LIMIT boundaries. Ordinary 'now' text,
  identifiers and comments are allowed; actual current-time functions remain blocked.
- Keep AuditKit re-scoring (`CustomPredicate`) on the new metric. The track asks for AuditKit evaluation.
- Keep reference SQL and results out of everything the agent sees.
- If GRPO runs, copy SQaLe's reward tiers (correct 3.0, runs but wrong at most 2.1) so partial credit never beats a correct answer.

## Budget

Assumptions: 35 s per episode measured on a T4, times 1.75 for bigger schemas (assumed), so about $0.0067 per
episode; 369 tokens/s training measured on a T4; 2,500 tokens per episode and 2 epochs assumed. The vLLM
column is a guess until the throughput check.

| Item | No vLLM (T4, HF, one at a time) | vLLM makes inference ~5x cheaper |
|---|---:|---:|
| Throughput check and 20-task smoke | $3 | $3 |
| Dev screens, 8 configurations x 100 tasks | $5 | $1 |
| Data stages 500, 1,500, 5,000 | $10 | $10 |
| A+ and B training | $5 | $5 |
| Self-training samples | $8 (300 x 4) | $3 (600 x 4) |
| Final, 650 tasks x 3 finalists, plus retention and repair | $15 | $4 |
| Second seed, winner and control | skip | $8 |
| Aya Expanse 8B, untuned and one trained, 250 tasks | $14 | $8 |
| One GRPO run | skip | $15 |
| Reserve | $5 | $10 |
| **Total** | **~$65** | **~$67** |

These historical planning rows are not registrations. The concrete G0 is now the single $3 factory/
resume/reload check above, with nine inference episodes and no vLLM test. Active dev/Q2/final counts
are 95/579/637 (plus 80 retention). Re-estimate later stages from measured G0 throughput before approval.

Set aside the final-evaluation money before spending on optional arms. Buy more only if the 5,000 stage and
GRPO both pass their gates.

Prices checked 2026-11-06, per hour. [HF Jobs](https://huggingface.co/docs/hub/jobs-pricing), billed per minute,
all-in: T4 small $0.40, L4 $0.80, A10G small $1.00, L40S $1.80, A100 $2.50. [Modal](https://modal.com/pricing),
per second: T4 $0.59, L4 $0.80, L40S $1.95, plus CPU ($0.0000131 per core-second) and memory ($0.00000222 per
GiB-second), so an L4 is about $1.00-1.25 and Modal's $20 is worth about $10-16 at HF rates. Vast.ai's page showed no figures.

**Where to run.** HF for everything that gets compared. Modal for spillover only (`scripts/modal_job.py` is
T4-only with the pilot stages, so it needs porting). Vast.ai only if money runs out, and only for self-training
samples: hardware varies by host, it needs a third launcher, and the gated-model token would sit on someone
else's machine (use a read-only token and revoke it afterwards).

## Gates

- **G0, current compatibility proposal:** does actual NF4 Tiny Aya training through the AgentTune
  custom factory preserve exact masks, resume its optimizer checkpoint, reload identical adapter
  tensors and run finite inference on a T4? Freeze the concrete longest-example/three-dev-task protocol
  in `experiments/sql-g0/REGISTRATION.json` before approval. vLLM is unresolved and outside this run.
  Train and serve in one precision: the pilot scored 37/40 on fp16 and 40/40 on nf4 (measured).
- **G1, broader data helps:** A beats Base by at least 10 points on dev, on the fallback score as well as the
  strict one. Otherwise fix the data before anything else.
- **Proposed stage rule, amended 2026-11-07:** stop at 1,500 unless it beats 500 by at least 5 percentage
  points on the same cleaned dev pool (primary strict accuracy, unrounded). This governs spending,
  not significance, and needs approval before a paid run uses it.
- **G2, self-training has material:** at least 25% of Q2 questions get a verified success within 4 samples.
- **G2-size, continuation is large enough:** estimate verified data from A's dev score before sampling;
  train A+/B only if unique verified B targets **per pass** before repetition contain at least 10% of
  A's unique targets **per pass**, with A deduplicated too. Exclude validation and never multiply either
  side by epochs/updates/repeated copies. The threshold is frozen in the preregistration. Report a tie as no difference detectable
  at this size.
- **G3, GRPO has signal:** at least 30% of Q2 questions get mixed results, and money remains after the final-eval reserve.

All gate numbers are proposals; freeze them in the preregistration.

## Order of work

1. Nov 6-7, free, approved: build the task pool, `submit_sql`, the scorer, tests and trajectories; preregister.
2. Nov 7, $3 cap, **needs the user's OK including uploads**: the one registered longest-example
   AgentTune factory/resume/reload G0 and nine planned dev inference episodes. This one run has now
   failed before update 1; no inference occurred. Do not retry it or launch a second job.
3. Nov 8: train A in stages; dev-screen Base, Pilot and A; G1.
4. Nov 9: estimate Q2 yield from A's dev score before sampling; after approved sampling, check G2,
   G2-size and G3. Freeze retained question IDs, build the exactly target-token-matched A+ control, then
   train A+ and B only if their gates pass; dev-screen them.
5. Nov 10-11: only what passed: GRPO, second seed, Aya Expanse 8B. Run evaluations as parallel jobs.
6. Nov 11-12: final evaluation once, plus retention, repair and reference rows.
7. Nov 12-14: write-up, collections, and the repo going public when the user decides. Do not let training use these days.

Each paid stage still needs the user's OK for that specific run.

## Risks

- **Checkpoints:** legacy `sft.py` remains checkpoint-free for reproducibility. The new SQL training
  path saves complete optimizer/scheduler/RNG/trainer/token-ledger checkpoints and syncs bounded
  snapshots. Real CUDA serialization/resume is an explicit G0 pass condition, still untested. What HF
  does when credit runs out remains unverified; no automatic restart is allowed.
- **Launcher:** the additive launcher guards account/prices/pins, scopes outputs by account/run and
  supports longer registered wall limits. Legacy limits are unchanged. The current registration is
  fixed to its account and hardware; changing either needs a new registration and approval.
- **Track wording.** The rules say to post-train "with AgentTune". `sft.py` uses the HF Trainer, with AgentTune
  only for rendering and rollouts. If GRPO is skipped, no AgentTune trainer runs at all. Check whether its SFT
  backend can run this and report the supported path, mask/QLoRA limitations and what remains untested;
  inspecting or testing construction is not permission to train. The official page lists AgentTune and
  AuditKIT for Track 3. Its general two-repo minimum explicitly says "except Track 3"; we already use
  both for rollouts/rendering and evaluation, but must describe the training integration accurately.
  Source checked 2026-11-07: https://hackathon.lexsi.ai/ (Track 3; Project Requirements; FAQ).
  Check completed: `reports/2026-11-07-agenttune-sft-feasibility.md`. The supported custom factory
  preserves the supplied model object and exact masks in a CPU-only spy test; an already prepared
  QLoRA model can use that path, while the stock factory is not
  a drop-in. Actual Tiny Aya training through this path still needs an approved smoke run.
- **Public artifacts.** All datasets, logs and adapters must be public and un-gated. Keep BIRD-derived files in a
  separate CC BY-SA dataset, and SQaLe-derived files with their notices. Tiny Aya is CC-BY-NC.

## Unresolved

- Whether vLLM runs Tiny Aya's `cohere2` architecture with a LoRA adapter, and whether it runs on a T4.
- Real episode lengths on sliced SQaLe schemas.
- Whether SQL-Trail's Table 7 is greedy or majority-vote (the paper does not say; some values match its majority
  column). Its Table 1 data column also conflicts with its section 4.2.
- SchemaPile's upstream licence terms.
- Neither pinned SQaLe repository contains a standalone licence file; both README metadata declarations
  say MIT. Source inspection and notices are recorded in `reports/2026-11-07-sqale-source-notices.md`.
- How much HF credit is available, and whether the account has Tiny Aya access.
