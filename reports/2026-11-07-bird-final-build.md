# BIRD Mini-Dev final-only preparation (2026-11-07)

The approved local preparation is complete under gitignored `data/bird/final-v1/`. It is **final evaluation only**: no training, tuning, selection, model inference, GPU work, upload or paid service has used it. Amendment 3 and `experiments/bird-final/SOURCES.json` were recorded before gold recomputation. The source/licence report records provenance and attribution.

## Membership

Pinned source: `birdsql/bird_mini_dev@5b256b5a9c17478f06ed6912612e9a14d17065c8`. Of **500** SQLite questions across **11** full original databases, **493** are retained and **7** are excluded. Every reference was executed with the existing five-second/50M-operation, 10,000-row/8-MiB limits. All seven exclusions returned SQLite `interrupted` under those limits; the existing executor does not distinguish the wall-clock and VM-operation causes. No gold SQL was rewritten or replenished.

Excluded source IDs: 40, 129, 209, 253, 257, 271, 491. Details remain in `exclusions.jsonl` and `reference-checks.jsonl`.

| Database | Retained final tasks |
|---|---:|
| california_schools | 48 |
| card_games | 69 |
| codebase_community | 30 |
| debit_card_specializing | 14 |
| european_football_2 | 31 |
| financial | 45 |
| formula_1 | 52 |
| student_club | 37 |
| superhero | 50 |
| thrombosis_prediction | 86 |
| toxicology | 31 |

Source difficulty labels are preserved: {'moderate': 32, 'challenging': 453, 'simple': 8}. They are not the legacy release's advertised difficulty proportions. Native ordering policies retain 162 unique-key ordered tasks, mark one tied-key task unordered and five unresolved non-LIMIT orderings unordered; 325 have no top-level ordering.

## Evidence and scoring

The canonical baseline guide and validator show that nonempty evidence is model input (the official 2026 fixed template puts it before the question). Our existing tool protocol is preserved: source question, then `\n\nExternal Knowledge:\n` plus verbatim nonempty evidence. Amendment 3 explicitly records this header/order. There are **365** nonempty source evidence fields, **358** in retained tasks. Gold SQL and full results remain private scorer metadata and never enter this model prompt.

Legitimate empty/zero results are allowed. Full databases remain unchanged; no gold-dependent schema slice is made. The new SQL runner rejects BIRD for training, control construction, compatibility/dev checks or model selection, and requires a frozen selection record for a final run. Primary strict full-result scoring, auxiliary exact-set scoring and last-query fallback remain separate. This is a filtered tool-based evaluation, not an official 500-question fixed-prompt leaderboard reproduction.

## Identity and validation

Annotations SHA256: `926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606`.
Tasks SHA256: `c79978cbc0445ee5aeb9294909e51445a0e58afa8adbc9ff4e5301d38b1b7f6b`.
Manifest SHA256: `8955958e6e18fd9e501659b89b31a54151f13036039f6e833f4554527c5f7551`.

All eleven database digests match their published image-tag prefixes; the read-only preparation preserves **1,493,287,936** original SQLite bytes. Sources, layer digests, licence text, database hashes, all 500 reference decisions and retained full gold results are local. CC BY-SA 4.0 is declared by the pinned HF card; see `reports/2026-11-07-bird-source-notices.md` for the missing standalone licence and underlying-term caveat.

The read-only verifier checks source/database/output hashes, all 500 IDs, exact question/evidence preservation and independently reruns each retained reference against its saved full result. **Final verification passed all 493 retained references.** The initial checker used Windows default decoding on the raw UTF-8 source; that checker read was corrected, without changing any prepared data. Final status is recorded in `data/bird/final-v1/verification.json`.

Recheck: `.venv/Scripts/python.exe -m ayadb.birdverify`. Do not rerun a completed preparation over its output. No data are staged for Git, and no private-bucket upload is authorized.
