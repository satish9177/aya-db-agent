# Amendment 3: approved BIRD Mini-Dev final preparation (2026-11-07)

The user explicitly approved BIRD Mini-Dev download/preparation and local commits, with no push.
Only free local work is authorized. No model run, training, tuning, selection, paid service, database
upload, private-bucket write or publication is approved. `data/test.jsonl` remains prohibited.

Preparation registration: `bird-final-local-20261107-01`, one local preparation before any model use.

- Pin `birdsql/bird_mini_dev` to `5b256b5a9c17478f06ed6912612e9a14d17065c8` (2026 release), SQLite
  annotations only. The source's SQLite annotation SHA256 is
  `926af2a0d1b47e5d0d15a7aa8a34d4721007f4550a881555b6a89496b54a2606`.
- Fetch only the eleven SQLite databases from the official card-linked
  `3036197201/bird-mini-dev-sqlite` images. Pin per-database image manifests and data-layer SHA256s
  in `experiments/bird-final/SOURCES.json`; verify before extraction, never run an image. Extract only
  regular SQLite files, preserve the original full databases, and record their byte hashes. The image
  has no `latest` tag; the explicit `harbor-minidev1024-r1-*` tags resolve to the pinned content digests.
- Store sources, manifests, licences, databases, private scorer records and exclusions under the
  gitignored `data/bird/`. No annotations, rows or databases enter Git. Keep the dataset separate from
  SQaLe. The HF card declares **CC BY-SA 4.0**; preserve its declaration, attribution/citation and a
  licence copy. Record any missing standalone licence and unverified underlying database terms.
- **Evidence is included.** The pinned 2026 baseline guide/template/validator and the official legacy
  `run_gpt.sh` (`use_knowledge=True`) establish that annotated evidence is model input. Our new
  `sql-submit-v1` task prompt is the source question followed, when nonempty, by
  `\n\nExternal Knowledge:\n` and the verbatim evidence. Never include source SQL/results in that
  prompt. Keep the existing tool protocol/system prompt; this is tool-based evaluation, not an exact
  reproduction of the fixed-prompt/chain-of-thought leaderboard setup.
- Use **final evaluation only**, once after model/checkpoint selection is frozen. Do not use these
  tasks for SFT, sampling, control construction, smoke/throughput checks, dev screens or tuning. The
  future runner must reject non-final purposes for BIRD and reject BIRD in training inputs.
- Recompute every reference result on the full pinned database with the existing read-only SQL
  scorer and its frozen caps (5 seconds/50M operations, 10,000 rows, 8 MiB payload). Preserve duplicates
  and numeric/text semantics. Keep legitimate zero/empty results rather than selecting on answer
  content. Log invalid/nonfinite results, resource failures and ambiguous/unsupported ordering as
  exclusions without rewriting gold or replenishment. Use the same tie-aware ordering policy as v3.
  Report the retained count and denominator by database and difficulty, plus every exclusion.
- Record schema, database and annotation identity; do not slice/rebuild the BIRD databases or use
  reference SQL to expose schema hints. Gold execution during preparation is not model evaluation.
  Report strict full-result and auxiliary set/fallback metrics separately in a later approved final
  run, with dataset filtering, resource limits and evidence policy explicit.

Primary provenance: [canonical dataset/card](https://huggingface.co/datasets/birdsql/bird_mini_dev/tree/5b256b5a9c17478f06ed6912612e9a14d17065c8),
[official Mini-Dev code](https://github.com/bird-bench/mini_dev/tree/1c49a375e76c471e218564da3f4617513d33e371).
No source-model baseline results influence preparation or selection.
