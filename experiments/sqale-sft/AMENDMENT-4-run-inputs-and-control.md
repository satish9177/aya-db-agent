# Amendment 4: effective training sets and minimum-drop control (2026-11-07)

Recorded before any model run or paid Q2 sampling. Free local runner/launcher construction, tests,
split accounting and a draft $3 registration are authorized. No training, inference, paid job,
database/bucket upload, push or publication is authorized. Local commits are approved.

## Effective inputs and the actual 10% denominator

Preserve v3 files and hashes. `RUN-POLICY-v1.json` excludes the one known entity-description training
fragment and three known scan-order-dependent dev/Q2 references reported in HANDOFF.md. This is a
pre-model eligibility overlay, not gold rewriting or model-outcome selection. No replacement occurs.
Keep BIRD strictly final-only under amendment 3.

Use exactly the rows passed to A's training factory **after** eligibility and the deterministic
validation split. Preserve file order, shuffle row indices with Python `random.Random(42)`, and hold
out `max(1, round(n * .05))` rows when validation is enabled; no holdout when explicitly zero for G0.
Deduplicate trajectories/context on both sides. Compute supervised labels freshly with the official
tokenizer and existing audited masks. Persist the train/validation IDs, encoding hashes and unique
target-token totals under the run, with source-file/policy identity. The runner and local planning
report use the same split function. Changing seed/share/input/mask requires a new run manifest.

The 10% threshold remains fixed. B's retained unique targets per pass must reach `ceil(A_train_unique
targets_per_pass / 10)`, excluding A validation. Prepared full-pool totals in the historical v3 freeze
are **not** A training-set denominators. Actual optimizer-consumed/repeated/input/generated/padded
tokens and sampling costs are separate ledger fields. Record the actual training-set manifest of the
adopted A checkpoint before forecasting or sampling Q2.

## Preregistered match_control fallback

First try all B-retained questions with whole control examples and at most one partial example overall.
Use only approved phrasings/slices of those same questions. If question coverage is infeasible under
that rule, drop the **fewest questions possible**, deterministically, from **both A+ and B**, then match
again. Never silently train B on a larger question set. Do not alter either adapter/settings or relax
the one-partial rule. Missing/invalid controls are errors, not a reason to select a new population.

Algorithm `minimum-question-drop-v1`: for each question q let m(q) be its smallest positive whole
control target count, b(q) B's target budget, d(q)=m(q)-b(q), D=sum d(q). For each possible retained
partial-mask anchor p, sort other questions with positive d descending, breaking ties by question ID.
Remove the shortest prefix whose summed d is at least D-(m(p)-1). A nonpositive requirement removes
nothing. Choose the candidate with smallest (number removed, sorted removed IDs, anchor ID).
At fixed p, no smaller removal set can remove more deficit than this sorted prefix; considering all
anchors gives a minimum question count. Positive targets allow at least one retained question.

Freeze/report original and retained question IDs/counts, dropped IDs/reasons, question coverage,
unique verified episode/target coverage, repeated control copies, per-question allocations and the
single partial mask's size. Recompute the 10% gate **after dropping questions**. If it fails, train
neither continuation. Report pre-drop Q2 success yield separately. A heavily reduced comparison
supports claims only about that retained subset. A tie remains no difference detectable at this size.

The new wrapper is `ayadb.continuation.match_control`; historical `tokenbudget.match_control` and the
v3 source snapshot remain unchanged. The wrapper returns the matched control **and filtered B data**.
No actual Q2/control data exist yet; local tests use synthetic labels/IDs only.
