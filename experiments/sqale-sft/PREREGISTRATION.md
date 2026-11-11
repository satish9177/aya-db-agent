# SQaLe broader SFT and verified self-training

Design recorded 2026-11-07 before any model run. **No training, sampling, GPU evaluation or paid stage is
approved.** The approved work is SQaLe download and local preparation. This draft must receive a frozen
data manifest and an individually approved, uniquely named run registration before each model run.
Never read or reuse `data/test.jsonl`. Keep all artifacts private; no commits or pushes are authorized.

Amendment 2 (`AMENDMENT-2-v3-local-fixes.md`) governs the active v3 preparation. v2 is historical and
immutable. All 850 primary SQaLe dev/final/Q2 prompts are individually reviewed, labels stripped and
exclusions logged; excluded questions are not replenished. Use the actual cleaned counts in the v3
freeze, not the original requested counts below. The free local preparation registration is
`sqale-v3-local-20261107-01`; its prepare and replay/audit phases are one preparation, with zero model runs.
The expanded prompt-completion pass is separately registered as `sqale-v3-local-20261107-02` in
`V3-PROMPT-COMPLETION-REGISTRATION.md`. It preserves the initial audit, rejects additional training
metadata fragments and freshly replays changed prompts; exact unchanged scripted records can be
reused only after complete conversation/call/database/result checks. Active counts and hashes are in
`DATA-FREEZE-v3.md`; the older v2 freeze remains historical. No model outcome informed either pass.

## Free preparation

Pin queries to `85fc905c3484f9d197ee8736d672a7d0e4903fc6` and schemas to
`091b1a9e946dec386470c7dc381af166d2b41c38`. Initially download the query train/test parquet files, the
first schema train shard and schema test shard, plus both README licence declarations. Record SHA256
and bytes for each. No other corpus or teacher is allowed. Record missing standalone licences and
unresolved SchemaPile upstream terms; downloading does not establish redistribution permission.

Split by source schema, grouping near-duplicate schemas before allocation; keep question phrasings
together. A and Q2 use disjoint training schema groups. Assign separate SQaLe test-schema groups to dev
(100 questions) and final (150); model-facing final evaluation is run once, after selection. This SQaLe
source test split is distinct from the prohibited shop `data/test.jsonl`. Store reference SQL/results
only in private scorer records, never model prompts. Use 4-16-table slices including all reference
dependencies, grow along foreign keys, deterministic shuffled discovery order. Recompute full results;
log every exclusion (empty/all-NULL/single zero, invalid SQL/data, duplicate group, oversized context,
execution/row limit, audit failure). Never retain matching truncated previews as verified results.
Preparation eligibility refinement before any model runs: dev has one question per eligible source
schema. Final prefers one question per eligible source schema; if fewer than 150 schemas materialize,
fill to 150 using at most a second question from those same final schemas. Use deterministic question
hash order and balance difficulty. Report original schema-group counts and use grouped uncertainty.
Source integers outside SQLite's 64-bit range reject the schema without coercion. A technical free-build
failure may reuse prior scripted replays only when complete prompt text, call arguments and observations
match, and full-result correctness is rechecked; preserve its provenance and failed-build log.

Gold-path episodes discover schema, execute reference SQL, then submit SQL. Repair examples contain
executed, genuinely failing mutations with error observations; mask the injected failed turn from the
loss. Replay every retained episode through AgentTune's real loop and the shared replay-audit checks.
Convert the 235 existing shop episodes alongside the old files. No old format, tool or pinned hash changes.
Use the official Tiny Aya tokenizer without its transformers pre-tokenizer substitution; no example may
exceed 4096 input tokens. The existing assistant-only target mask includes end-of-turn tokens and masks
all observations, system/task turns and injected errors.

## Arms and continuation control

Base: untuned Tiny Aya Global; fixed one-example prompt on dev only. Pilot: existing frozen adapter.
A: broader SQaLe SFT plus converted shop demonstrations, with proposed 500/1500/5000 data stages.
B: one continuation from A on verified self-generated attempts from Q2, four samples/question, at most
two retained successes/question. A+: gold-path continuation from the **identical A adapter** using
**exactly B's retained question IDs**. A+ and B share optimizer, seed, LR, precision, masks and other
settings. Match supervised target tokens exactly across actual updates, not steps or input length.
Extra source phrasings/schema slices must belong to those same IDs. Match globally over the entire
control set using whole examples, with at most one partially masked example overall. If all retained
questions cannot be represented under this rule, report infeasibility. Record the final target-mask
adjustment and per-question allocations. Report unique, repeated, input, target, generated, padded and masked tokens separately,
and the full sampling, verification and training time/cost. Never hide sampling cost in token matching.

Before sampling, forecast verified episodes and target tokens from A's dev correctness and observed
lengths. Greedy dev accuracy is a proxy; do not assume independence to turn it into pass@4. Report a
range and the assumptions. Freeze the continuation-size threshold at **10%**: verified B data after
deduplication and the per-question cap, before epoch repetition or padding, must contain at least 10%
of A's **unique supervised target tokens per pass**. Deduplicate A's trajectories/context too and
exclude validation. Neither side is multiplied by epochs, repeated copies, optimizer updates or
padding. Record both unique per-pass counts and A's exact denominator before Q2 sampling. Cumulative
training targets remain a separate cost/accounting measure. A+/B train only if this gate
and their material gate pass. A tie is **no difference detectable at this size**, not equivalence.

## Metrics and gates

New format: `sql-submit-v1`, alongside unchanged `multistep-close`. Primary: explicit final submitted
SQL matches the full reference result as a multiset, preserving duplicates; preserve row order when
the reference has a top-level ORDER BY with unique keys. SQLite-native rank probes detect tied keys;
mark those tasks unordered and record that policy. Mark unresolved no-LIMIT orderings unordered;
exclude unresolved LIMIT/offset selections and boundaries cutting tied groups. Ordinary 'now' literals,
comments and identifiers are permitted; actual clock-dependent date functions fail closed.
Numeric tolerance: abs 1e-6 + rel 1e-6, finite numbers only;
NULL and text compare exactly, column position preserved. Report exact BIRD set-result comparison
separately once checked against its official script, plus last-successfully-executed-query fallback
to expose protocol-learning effects. Queries fail closed on write attempts, execution failure, timeout,
overflow or unsubmitted SQL. Preview cap 20; scorer caps 10,000 rows, 8 MiB of result cell payload,
1 MB per SQLite value, and 5 seconds/50M VM ops per query. Result collection uses bounded batches.
AuditKIT re-scores the same saved records using CustomPredicate. Log every submitted query and outcome.

Other outcomes: calls/tokens/time/cost per success, malformed calls, fabricated observations, loops,
schema/phrasing slices, and recovery from real-error prefixes separately from natural error rate.
Report paired gains/losses and database-grouped uncertainty; final sample sizes do not guarantee power.

Retention: evaluate the unchanged 40 shop dev and 40 mh questions in `sql-submit-v1` for Base, Pilot
and each approved selected finalist. These sets never train A/A+/B. Regenerate only dev (seed 202) and
mh (seeds 404/405/406), preserve their original task IDs and LF hashes, and check every reference's
first-row/first-column result against stored gold under the original tolerance (int 0, money .011,
float1 .051, text casefold). Store the projection, full reference result and check. Report full-result
retention plus the legacy scalar projection separately; the gold check does not relax SQL scoring.
Retention is reported alongside each checkpoint screen/finalist and cannot justify selecting a
checkpoint using SQaLe final. No sealed shop test access is allowed.

G0: engine/LoRA/precision compatibility and measured cost. G1: A exceeds Base by 10 percentage points
on both strict and fallback dev accuracy. **Proposed stage spending rule (requires approval before
paid use): stop at 1,500 unless it beats 500 by at least 5 percentage points of primary strict accuracy
on the same cleaned dev pool.** With 96 dev tasks this requires at least five net additional successes.
Use unrounded scores; fallback and retention remain reported checks. This is a spending rule, not
evidence of significance. G2: at least 25% of sampled Q2 questions have a verified success among four attempts. G2-size:
the 10% target-token rule above. G3, optional GRPO only: at least 30% mixed-outcome Q2 questions and
funding left after final-evaluation reserve. Freeze run-specific settings and cost ceilings before launch.

## Secondary diagnostic: database perturbation

Save every submission, even failed/repeated calls, with task ID, call order and termination status.
After primary evaluation, apply a separately versioned, seeded, constraint-preserving procedure to
logged queries. First validate against a fixed suite of known-equivalent and known-wrong SQL; report
equivalent-query false negatives and wrong-query detection, excluded cases and perturbation failures.
Do not adopt an arbitrary row-change percentage without validation. This diagnostic never changes
the primary metric, selection, rewards or self-training acceptance and is not proof of SQL equivalence.

## Lexsi integration and spending

AgentTune renders and executes tool trajectories; AuditKIT re-scores saved trajectories. Inspect and
test AgentTune's supported SFT factory without downloading models or running an optimizer. Default
backend compatibility and any custom factory must preserve our tokenizer, QLoRA and exact masks.
The official hackathon rules list AgentTune/AuditKIT for Track 3 and exempt Track 3 from the general
two-repo minimum (https://hackathon.lexsi.ai/, checked 2026-11-07). Actual AgentTune-backed model
training remains to be demonstrated in an approved smoke run.

No throughput check, A/A+/B run, GRPO, second seed, Aya Expanse run, API reference or BIRD download is
authorized by this document. The review's $65-67 totals remain estimates. Report free build counts and
checks first; request approval for a concrete throughput smoke with a stated cap before spending.
