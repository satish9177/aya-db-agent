# Prepared data freeze (2026-11-07; before any model run)

Design: `PREREGISTRATION.md`, plus `AMENDMENT-1-data-eligibility.md`. This records prepared data, not
permission to train, sample, evaluate a model, spend money, commit or publish. A unique model-run
registration and explicit user approval are still required for each run.

Artifact root: `data/sqale/build-v2/` (private, gitignored).

| File | Count | SHA256 |
|---|---:|---|
| `manifest.json` | base preparation | `712e8e5301f512c1822f567c8c151c62213a4bc7f62aa60436567b750ff923e8` |
| `train-500.episodes.jsonl` | 500 | `fce67c0fe0a35cbaab655bb0d86793b953f944944059ab1e1f323460f2a18489` |
| `train-1500.episodes.jsonl` | 1,500 | `549f4104bc084235ba7f10004c24f6ce3331d9d00f1e6f882ec3bcb6c49976ba` |
| `train-5000.episodes.jsonl` | 5,000 | `5d11798e004f1b39ae86aa374b32e780c4cca57fc33ea2503255ddce317db563` |
| **`dev-100.tasks.jsonl`** | **100, across 95 groups** | `78e942ff20c0621907f6e14997e33f72348f839b03c17c24f798ebc73b8d68d8` |
| `q2.tasks.jsonl` | 600, across 35 groups | `cfc85f764b1961a7369ebfdbc5b1e90f837a739b9fc9c9ab9218295df846be00` |
| `final.tasks.jsonl` | 150, across 94 groups | `1916f136662b4b9381fce7a11687056a9077b10550b01ba6473222dffc6196c5` |

The stages contain the same 235 converted shop episodes and nested SQaLe subsets. Max input 4,072,
cap 4,096, with no episode truncation. The 5,000 stage has 9,568,488 input tokens and 1,144,502
supervised targets **per data pass**. Actual A targets consumed over optimizer updates, excluding
validation, must be measured and frozen before Q2 sampling; per-pass counts are not that denominator.

The original `dev.tasks.jsonl` has 95 questions and stays unchanged at
`00651c66a4c84f598f58cad1bbd1ec3a97c4157d6c714498e204fa3fadec001f`. It is not the active dev pool.
Amendment manifest: `dev-100.manifest.json`; successful combined check: `verification-dev100.json`.
All 100 dev (schema, normalized SQL) pairs are distinct; at most two questions share a schema.

Train/Q2/dev/final schema groups are disjoint. The one detected near-duplicate link was grouped before
allocation. Reference SQL/results remain private scorer fields, not agent prompt inputs. Final is
model-unseen; run it only once after selection. The spent shop `data/test.jsonl` remains closed.

Main replays: 8,005/8,005 pass AgentTune and AuditKIT; dev additions: 5/5. These are scripted-data checks,
not model performance. The filtered native-SQLite/sliced-schema setting excludes 425 of 976 downloaded
schemas; report that limitation and grouped uncertainty. Source and model-tokenizer revisions are in
the base manifest. Full source/code/environment records and logs remain under the private artifact root.

Build code snapshot SHA256: `92475c981b5aa63feb9c3a18abe68ea38f5804b77c3e5e1c3a9e41539f3fb37e`.
The later dev amendment's code and document hashes are recorded separately in its manifest.
