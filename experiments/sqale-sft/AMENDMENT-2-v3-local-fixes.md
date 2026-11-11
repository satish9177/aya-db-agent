# Preparation amendment 2: v3 quality, repair, retention and scoring (2026-11-07)

Authorized by the user before any model run. Only free local work is authorized. No paid run, commit,
push, BIRD/other-corpus download, teacher, model inference/training or publishing is approved.

Write a separate `data/sqale/data-build-v3/` and leave every v2 artifact unchanged. Snapshot v2 file
hashes before and after. Preserve v2 source code archives and manifests. The following supersedes
earlier design wording where it differs; it is a preparation/design change, not a model result.

1. Reject heading-only and generator-instruction prompts. Strip recognizable difficulty/number label
   prefixes only when the remainder is a usable task; otherwise reject. Review all 100 dev, 150 final
   and 600 Q2 source prompts individually, including flagged and unflagged prompts, and retain a
   per-prompt decision/change/exclusion log. Apply the same quality rules to training and alternative
   phrasings. Do not silently replace excluded held-out questions; report the reduced counts. No model
   outcomes inform curation. The supplied Claude scan is a flagging aid, not an automatic gold label.
2. Replace the fixed `__ayadb_bad_alias__` examples in v3 with deterministic, varied mutations of each
   gold query: wrong column, wrong table, ambiguous column and misused aggregate. Keep a mutation only
   if actual SQLite execution gives the intended error class. Preserve its observed error and mask the
   injected turn. Log failed/unsupported mutations and their distribution. Do not wrap every gold query
   in a fixed broken-alias query. New v3 trajectories use actual copied databases and are replay-audited.
3. Prepare shop dev (40) and mh (40) in `sql-submit-v1`, verifying each reference's scalar projection
   against the stored gold under the existing tolerance before relying on full-result scoring. Store
   any projection/tolerance decision. These are retention evaluations, never training data. Preserve
   existing task IDs, database seeds and pinned task hashes. Never open `data/test.jsonl`.
   The legacy top-category, top-customer and largest-order references return a ranked table while
   the question/stored gold asks for its first scalar. In v3 only, project that first column and
   take the unique top row, resolving ORDER BY aliases before dropping extra columns. Check the
   derived reference against the same stored gold and exclude any ambiguous top boundary. Apply
   the same conversion to retained shop training demos for consistent `sql-submit-v1` semantics;
   preserve the original reference/projection provenance. This changes no pinned legacy record.
4. The 10% gate is **per pass on both sides**: deduplicated, unique verified B target tokens per pass
   versus A's unique training target tokens per pass, excluding validation. Do not multiply either side
   by epochs, repeated copies, optimizer updates or padding. Retain question cap/dedup rules and
   forecast before paid sampling; record both denominators explicitly.
5. `match_control` matches the global target-token budget over the retained question set using whole
   examples, with at most one partially masked example overall. Keep the same question IDs, A adapter
   and training settings. Prefer existing phrasings/slices, disclose repeats, and never unmask an
   observation/error. If the global budget cannot represent every retained question with this rule,
   report infeasibility rather than applying one partial mask per question.
6. Ordering must not reject correct tied-row permutations. Detect reference ordering/ties and either
   compare within tie groups or mark tied/unresolved ordering tasks unordered, recording the policy.
   Do not rewrite a nondeterministic LIMIT boundary into an arbitrary deterministic gold answer; flag
   unsupported ambiguous selections for exclusion. Permit harmless 'now' text and quoted identifiers;
   distinguish actual current-time functions from ordinary literals/comments.
7. **Proposed spending stage rule:** stop at 1,500 unless it beats 500 by at least **5 percentage points**
   on the same cleaned dev pool. This is a proposed spending rule, not a significance claim or permission
   to train either stage. Any later full-pool stage uses the actual cleaned count, up to 5,000. Report
   ties as no difference detectable at this size, and report retention for each selected finalist.

Use the original approved SQaLe sources and existing shop generators only. New parser/dependency pins
may be added alongside existing pins; do not upgrade existing model/framework dependencies. Record
all new counts, hashes, exclusions, mutation errors, retention checks, tests and remaining limitations.
Ask before anything paid; every concrete run still requires its own registration and explicit approval.

Before freezing v3, the expanded training scan found additional SQL-logic/reasoning/domain fragments,
header instructions and title-only records. `V3-PROMPT-COMPLETION-REGISTRATION.md` records the second
free preparation phase (`sqale-v3-local-20261107-02`) before applying its extra exclusions/label edits.
The initial free audit is preserved; only exactly matching complete scripted conversations/calls and
unchanged database hashes may be reused, with a fresh full-result check and AuditKIT re-scoring.
This is a preparation correction before any model run, not an experiment outcome or paid approval.
