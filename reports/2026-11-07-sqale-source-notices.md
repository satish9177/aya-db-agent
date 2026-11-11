# SQaLe source and licence inspection (2026-11-07)

Downloaded only the approved SQaLe corpus and the tokenizer/config files for the existing Tiny Aya
model. No BIRD/SynSQL data, teacher model or Tiny Aya model weights were downloaded. Files are private
and gitignored in `data/sqale/`; nothing was uploaded or published.

| Source | Immutable revision | Files inspected/downloaded |
|---|---|---|
| `trl-lab/SQaLe-2-text-to-SQL-Queries` | `85fc905c3484f9d197ee8736d672a7d0e4903fc6` | README, train query parquet, source-test query parquet |
| `trl-lab/SQaLe-2-text-to-SQL-Schemas` | `091b1a9e946dec386470c7dc381af166d2b41c38` | README, first of sixteen train schema shards, source-test schema shard |
| `CohereLabs/tiny-aya-global` tokenizer only | `af89d219b53ed9b13b8a4645f9c8028973510324` | tokenizer.json, tokenizer_config.json, special_tokens_map.json, config.json |

Both SQaLe README files declare `license: mit` in their metadata. The full repository file inventories
at these revisions contain **no standalone LICENSE/COPYING file**, and the READMEs do not supply a full
MIT copyright/permission notice. Preserve the original README bytes, authorship, revision and source
links; do not describe the inspection as having found a standalone MIT licence text. Whether the
release's declaration covers all SchemaPile-derived upstream DDL remains unresolved. This is an
inspection record, not a redistribution clearance.

Authors named in the cards: Cornelius Wolff, Daniel Gomm and Madelon Hulsebos. Cite *SQaLe: A Large
Realistic Dataset to Empower Small Specialised Text-to-SQL Models* (2026). Original source links:
[queries](https://huggingface.co/datasets/trl-lab/SQaLe-2-text-to-SQL-Queries),
[schemas](https://huggingface.co/datasets/trl-lab/SQaLe-2-text-to-SQL-Schemas).

The six SQaLe files total **720,043,938 bytes**. `data/sqale/sources/downloads.json` records each SHA256,
size and revision; parquet hashes were verified against the Hub's LFS hashes. The first training schema
shard contains 553 schemas; the source-test shard contains 423. Query parquets contain 169,277 train and
8,100 source-test records. Source-test is the SQaLe split, and is unrelated to the prohibited shop
`data/test.jsonl`, which was not read.

The SQaLe Python package was not installed or used. A local materializer executes source CREATE TABLE/
INDEX statements and inserts source rows without repairing them. Unsupported DDL/data is excluded and
logged; successful construction and recomputed full query results determine eligibility. Existing source
foreign-key errors can remain because FK enforcement is off, consistent with the imperfect source rows.
Schema slicing preserves retained table rows; it is not the full-schema setting of the SQaLe model card.

The official tokenizer is loaded with the existing `official_tokenizer` helper (`TokenizersBackend`),
without `AutoTokenizer`'s known Tiny Aya pre-tokenizer replacement. Tokenizer file hashes and revision
are in `data/sqale/tokenizer/source.json`. The original tokenizer and experiment pins were not changed.

For scorer comparability, only the public BIRD evaluator source was inspected; **no BIRD dataset was
downloaded**. `evaluation/evaluation_ex.py` from
[BIRD Mini-Dev](https://github.com/bird-bench/mini_dev/blob/main/evaluation/evaluation_ex.py) compares
`set(predicted_res)` with `set(ground_truth_res)` (lines 18-22 at inspection). Its fetched SHA256 was
`da1bbcd4530be83692d7c650c814ea9704bb710d0c953eb75d02ccb38233cf89`.
The new scorer reports that exact set-result comparison separately from duplicate-preserving,
order-aware full-result correctness. Sharing a result comparator alone does not make our sliced-schema
scores comparable to BIRD leaderboard runs.
