# BIRD Mini-Dev source and licence record (2026-11-07)

The user approved Mini-Dev download for final evaluation only, plus local commits without a push.
All annotations, databases, legal-code copies and scorer records remain under gitignored `data/bird/`.
No source data are committed or uploaded. No model training, tuning, selection or inference used BIRD.

## Immutable source identity

- Canonical annotations: [`birdsql/bird_mini_dev`](https://huggingface.co/datasets/birdsql/bird_mini_dev/tree/5b256b5a9c17478f06ed6912612e9a14d17065c8),
  revision `5b256b5a9c17478f06ed6912612e9a14d17065c8`, file `data/mini_dev_sqlite.jsonl`.
  SHA256: `926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606`.
- Database source: the card-linked official Docker repository
  [`3036197201/bird-mini-dev-sqlite`](https://hub.docker.com/r/3036197201/bird-mini-dev-sqlite).
  All eleven explicit `harbor-minidev1024-r1-*` tags are pinned to image-manifest and data-layer
  content digests in `experiments/bird-final/SOURCES.json`. Only regular SQLite files were extracted
  from verified data layers. No base-image software or container was executed. All database SHA256s
  match their published tag prefixes. SQLite bytes total **1,493,287,936**; compressed layers total
  **349,209,238**. The missing `latest` tag was a metadata lookup failure, not a data substitution.
- Official code/docs inspected at
  [`bird-bench/mini_dev@1c49a375e76c471e218564da3f4617513d33e371`](https://github.com/bird-bench/mini_dev/tree/1c49a375e76c471e218564da3f4617513d33e371).
  The canonical HF baseline guide and validator also establish evidence as model input. Our tool
  prompt preserves question/evidence, with the documented header/order in amendment 3; it does not
  reproduce the fixed-prompt chain-of-thought setup.

## Licence and attribution

The pinned dataset card declares **Creative Commons Attribution-ShareAlike 4.0 International
(CC BY-SA 4.0)**. There is no standalone LICENSE in that HF repository revision. The card is preserved
in `data/bird/source/README.md`; the licence text from
[Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/legalcode.txt) is preserved in
`data/bird/source/CC-BY-SA-4.0.txt`, SHA256
`28a9529c7d0bb4dc51f4bf5c116a3d16ef247a052f7591466768ddf563fd1cf5`.
Underlying database-specific terms have not been independently audited. This record is not approval
to redistribute; no publication or private-bucket upload is authorized.

Credit the BIRD team and cite Li, Jinyang et al., "Can LLM Already Serve as A Database Interface?
A Big Bench for Large-Scale Database Grounded Text-to-SQLs," NeurIPS 36 (2024), as requested by the
source. Keep BIRD-derived data/attribution separate from the SQaLe-derived material if later publication
is explicitly approved. Our changes are a tool-format/evidence-prompt conversion, full reference-result
recomputation and logged eligibility/order policies; original question/evidence/SQL/database contents
are not rewritten.

Machine records: `data/bird/downloads.json`, `data/bird/source/data/manifest.json` and the prepared
final manifest. Source-pin metadata SHA256:
`55182e3a0df612abe713e68018baec93e69d19a9ee77bba846aa4362a0055fb0`.
