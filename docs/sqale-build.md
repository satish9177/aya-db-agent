# SQaLe preparation

This path is CPU-only preparation of the approved SQaLe corpus. It does not load a model, train,
sample, create HF Jobs, upload or publish anything. The legacy shop tools, `multistep-close` format,
training script and pinned data remain unchanged.

## Sources

`scripts/download_sqale.py` downloads two README files, query train/test parquet, the first train-schema
shard and source-test schemas at immutable revisions and checks SHA256. It needs network access.
`scripts/prepare_sqale_tokenizer.py` fetches only the existing Tiny Aya model's tokenizer/config files,
also pinned; it never fetches weights. Source-test here means SQaLe's source split, not the prohibited
shop `data/test.jsonl`. See `reports/2026-11-07-sqale-source-notices.md` for licence inspection.

## Build

The active preparation is v3, under `data/sqale/data-build-v3/`; the v2 commands/counts below are
historical and must not be rerun over their existing output. See `AMENDMENT-2-v3-local-fixes.md`,
`V3-PROMPT-COMPLETION-REGISTRATION.md` and `DATA-FREEZE-v3.md` in `experiments/sqale-sft/`.
V3 keeps 4,888 training episodes (4,653 SQaLe plus all 235 shop), 581 Q2, 96 dev and 144 final tasks.
It also prepares unchanged shop dev40/mh40 in `retention/`. Active dev is `dev.tasks.jsonl` within v3.

All 850 SQaLe dev/final/Q2 primary prompts and 80 retention prompts were read individually. Prompt,
alternative-phrasing and ordering exclusions are logged without replenishment. V3 repairs are actual
wrong-column, wrong-table, ambiguous-column and aggregate-misuse mutations, never fixed bad aliases.
Shop references are projected to the scalar the original question asks for and checked against stored
gold. Tied order keys are marked unordered; ambiguous LIMIT boundaries are excluded. All 5,903 v2
files are byte-checked against their before inventory.

The initial v3 check is preserved under `pre-prompt-completion/`. The final audit can reuse only exact
complete scripted conversations/calls with matching DB hashes and a fresh full-result check, then
AuditKIT re-scores every final record. These are preparation checks, not model results. Once complete,
the read-only artifact check is:

```powershell
.venv/Scripts/python.exe -m ayadb.sqaleverify_v3
```

The initial and prompt-completion builders refuse duplicate outputs; do not rerun them over the
completed preparation. Their code and registrations are snapshotted in the private artifact folder.

## Historical v2 build

Use the project venv and a new output directory; the builder refuses overwrites:

```powershell
.venv/Scripts/python.exe -u -m ayadb.sqalebuild --out data/sqale/build-v2 --audit-cache data/sqale/build-v1/replay-records.jsonl
.venv/Scripts/python.exe -m ayadb.sqaletopup data/sqale/build-v2
.venv/Scripts/python.exe -m ayadb.sqaleverify data/sqale/build-v2 --amended-dev
```

The cache argument is only for the preserved **scripted expert** replays of the failed first local
preparation. Omit it for a completely fresh build with another output name. Reuse requires identical
complete prompt text, call arguments and observations and a new full-result correctness check. The
old failure was a source integer beyond SQLite's range; it is now excluded without coercion. A killed
background command must not be restarted without the user's instruction.

The completed v2 base pool has 95 dev tasks. The separately preregistered, CPU-only `sqaletopup` step
adds five distinct questions from existing dev schemas without overwriting the base file or manifest.
It is already completed locally; do not rerun it over existing amended artifacts. Use the amended
`dev-100.tasks.jsonl` (100 questions, 95 groups) and its successful `verification-dev100.json`.

The first-shard limitation and every source/query/context/audit exclusion are recorded. Schemas are
grouped using exact DDL or >=90% table-name containment (at least three shared names), and groups stay
in one split. Q2 is held out of A's schemas. Dev/final use different source-test schema groups. Final
prefers one question/schema, with at most a second to reach 150 if materialization leaves too few
schemas. This identifier screen is not proof that all semantic duplicates have been removed.

Gold SQL is executed on a full materialized source database and on the slice. Only matching full
results are kept. The prepared schema has 4-16 tables (or the source's smaller size), gold dependencies
plus distractors, and shuffled discovery order. Unsupported original DDL/data is rejected, not silently
repaired. Source FK imperfections remain. SQL slices are not the full-schema SQaLe benchmark setting.

## Artifacts and verification

- `train.episodes.jsonl`: up to 5,000 retained episodes, including all 235 converted shop episodes.
- `train-{500,1500,5000}.episodes.jsonl`: nested stages with those same shop episodes.
- `train.tasks.jsonl`, `q2.tasks.jsonl`, `dev-100.tasks.jsonl`, `final.tasks.jsonl`: active private task/scorer data.
- `dev.tasks.jsonl`: preserved original 95-question pool; use the amended file for model dev runs.
- `schema-splits.json`, `exclusions.jsonl`, `submissions.jsonl`: membership, eligibility and every submission.
- `replay-records.jsonl`, `auditkit.json`: real AgentTune scripted replay and AuditKIT full-SQL re-scoring.
- `manifest.json`, `verification.json`: hashes, counts, target/input tokens, split isolation and limits.

Reference SQL/results live in private task records. The replay agent receives only the question and
ordinary tool observations. Each training target is an assistant turn; observations and deliberately
injected failed calls are masked. Max input length is 4096, with no truncation. No model result has
been measured by this preparation. Perturbation remains a named secondary diagnostic for later logged
model queries, after known-equivalent/known-wrong validation.

`sql-submit-v1` adds four tools in `sqlenv.py`: list, describe, run and terminal submit. The original
three-tool environment is separate. Scoring re-executes submitted SQL independently with full-result
limits, preserves duplicates and relevant row order, and also reports BIRD's set comparison and the
last-successful-query fallback. Submission stops further model generation through `SubmissionEngine`.

The continuation control utilities match retained question IDs and target tokens; final mask
adjustments can only remove audited labels. Actual training must count consumed targets across updates,
not infer them from steps or dataset size. AgentTune factory feasibility and remaining GPU checks are
documented in `reports/2026-11-07-agenttune-sft-feasibility.md`.

The v3 control matches target tokens globally using whole examples with at most one partial mask
overall. The 10% size gate compares unique B targets per pass with unique A targets per pass; epochs,
validation, copies and padding do not count on either side. Cumulative targets are still reported for
training-cost accounting. The proposed stage rule stops at 1,500 unless primary dev accuracy exceeds
500 by at least five percentage points on the same 96 questions; this is not paid approval.
