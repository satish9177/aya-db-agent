# SQL runner readiness and proposed G0 (2026-11-07)

Historical pre-G0 readiness record. The subsequently approved single G0 failed with CUDA OOM before
its first optimizer update; see [actual results](../experiments/sql-g0/RESULTS.md). No retry is approved.
Current scorer/tests and the blocked 500 draft are in [the follow-up report](2026-11-07-table-read-guard.md).

The requested free work is complete. The initial v3 work was committed locally as **`3fe7d82`** before
the BIRD download. The follow-up code, metadata and reports form the second requested local commit.
Data, model weights and credentials are excluded from both commits. Nothing was pushed or published.
No private-bucket upload, model training/inference or paid job has occurred during this work.

## Final-only BIRD and effective training inputs

BIRD Mini-Dev is pinned to `birdsql/bird_mini_dev@5b256b5a9c17478f06ed6912612e9a14d17065c8`.
Public database data layers and all eleven extracted SQLite files have recorded digests; no container
or downloaded code was executed. Local data live under gitignored `data/bird/`. The pinned card
declares CC BY-SA 4.0; a licence copy and attribution/provenance are recorded separately.

**493/500** final tasks are retained. Seven references exceeded the unchanged execution caps; all
exclusions are logged. Every retained full gold result was independently recomputed and checked.
Nonempty official evidence is preserved verbatim in the prompt under a documented header. Our own
tool protocol, limits and exclusions mean this is not an official fixed-prompt leaderboard reproduction.
Amendment 3, the [build report](2026-11-07-bird-final-build.md) and
[source notices](2026-11-07-bird-source-notices.md) contain details. BIRD is forbidden for training,
dev compatibility checks, tuning, sampling or selection. Its model evaluation needs a new approval and
frozen selection record; no BIRD data are included in G0.

Amendment 4 preserves v3 bytes while applying four known pre-model exclusions. Effective counts are
**4,887** full-stage training episodes, **95 dev**, **579 Q2**, **144 SQaLe final**, plus 40+40 retention.
Actual A split membership is shared with the factory; validation, repeats, padding and epochs are
excluded from the unique per-pass gate:

| Source stage | Actual A train / validation | Unique A targets/pass | Minimum B targets/pass |
|---|---:|---:|---:|
| 500 | 475 / 25 | 107,381 | 10,739 |
| 1,500 | 1,425 / 75 | 330,742 | 33,075 |
| 4,888 (4,887 eligible) | 4,643 / 244 | 1,067,919 | 106,792 |

The runner freshly verifies each factory encoding and saves its actual training-set manifest. Any
changed split/mask/input requires a new denominator. The control fallback drops the minimum number
of questions from both arms deterministically, reports IDs and coverage, and rechecks the 10% gate.
Exact global target matching uses whole examples with at most one partial mask overall. Its minimum
drop count is checked against an exhaustive small-set oracle. No real B/control package exists yet.
See [effective-input freeze](../experiments/sqale-sft/RUN-INPUTS-FREEZE-v1.md).

## Implemented runner and launch boundaries

The new `sqlrunner.py`, `sqltraining.py` and `scripts/sql_job.py` are additive; legacy tools, formats,
launcher, model/data pins and the immutable v2/v3 snapshots are preserved.

- Training uses AgentTune's supported custom-factory path into HF Trainer, retaining the prepared
  QLoRA model, official tokenizer and audited assistant/error masks. Token ledgers count actual
  committed optimizer input/target/padding/unique/repeated tokens and reject skipped updates.
- Complete checkpoints contain adapter, optimizer, scheduler, trainer/RNG state, data/settings identity
  and ledger. Explicit resume requires the exact checkpoint step and identities; inference verifies
  all saved adapter tensors after fresh reload. Only completed checkpoints enter bounded snapshots.
- Expected account and explicit billing namespace are **hf-user**. Account/price/model access,
  exact source/assets, clean commit, private bucket and explicit registration/upload/job approval are
  checked before writes or launch. Exclusive launch reservation blocks duplicate/ambiguous restarts.
- Registered wall limits support training 24h/evaluation 12h/sampling-final 24h within each run's smaller
  budget/provider timeout. G0 is two hours. Checkpoints sync every 60s and after steps, only under the
  new run prefix; status/pull paths are scoped by account/run. No automatic retry or account switch.
- CUDA, finite losses/logits, exact masks/reload, step/stall watchdogs and output caps fail closed.
  Runtime library/Python/SQLite/CUDA versions and reference-oracle checks precede model use. Every
  SQL submission is logged; strict/set/fallback remain separate, with offline AuditKIT agreement.
  BIRD/final selection and the spent shop-test guards apply before data use.

See [runner documentation](../docs/sql-runner.md). The image tag and exact package versions are recorded
in the launcher; the image is not digest-pinned. Runtime provenance and pre-model oracle checks cover
that remaining environment variability without silently changing the scorer.

## Free validation completed

- **89 tests passed**, 14 upstream torch JIT deprecation warnings, 59.45s:
  `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/pytest-sql-final-release --tb=short`.
  These cover deterministic minimum-drop matching against exhaustive search, actual split/gate
  accounting, source-oracle drift rejection, final-only guards, scripted AgentTune SQL/AuditKIT
  integration, account/approval/duplicate-launch guards, scoped complete-checkpoint sync, token-ledger
  resume, tensor reload/finite-logit checks and spy-only factory train/resume callbacks.
- The independent BIRD verifier passes all 493 references. V3 artifact verification also passes, including
  all 5,903 immutable v2 files and historical source snapshots. The spent `data/test.jsonl` was not read.
- Offline G0 plan validates all registered source hashes and **14 assets / 143,130,042 bytes**. Its four
  unique native reference inputs pass the local runtime-oracle check, with zero model calls.
- Actual **read-only** preflight confirms hf-user, access to the pinned gated model weights via
  metadata HEAD, and T4-small **$0.006667/min**. It starts zero jobs and writes zero bucket files. No
  base-model weights are downloaded. No G0 upload or launch receipt exists.

Scripted/spy tests use no model forward/backward or optimizer update. Real CUDA/QLoRA training,
optimizer checkpoint serialization/resume, fresh model reload, inference memory and throughput remain
untested. The proposed run below is designed to test them; the free checks are not evidence of SQL gains.

## One concrete proposal awaiting approval

Registration **`g0-sql-agenttune-20261107-01`**:
[human-readable draft](../experiments/sql-g0/REGISTRATION.md),
[machine registration](../experiments/sql-g0/REGISTRATION.json), SHA256
**`84799ec8cbdd3a57602b09084e3bf8aec1f82f572b5baabeb71a5d51828c8c81`**.

One T4-small job on hf-user, **$3 proposed total cap**, **7,200s hard provider timeout including
setup**. The longest retained example is 4,072 input/431 target tokens. Plan three optimizer steps
through AgentTune, stop/save at 2, resume optimizer/scheduler/RNG to 3, then run the same three fixed dev
tasks for Base, Pilot and freshly reloaded checkpoint 3 (**nine inference episodes**). Use NF4 double
quantization/fp16 throughout. Accuracy is reported as observed; it is not a compatibility pass criterion.
This disposable adapter cannot replace A or the submission model. vLLM and larger training are excluded.

Approval must include the **143MB pinned private input upload**, including four SQaLe databases,
tokenizer/configs and existing pilot adapter, plus source; inputs are bounded at 512 MiB and checkpoint
outputs at 1 GiB. The proposed envelope includes a first-month storage reserve; pull/review promptly and
remove or renew permission within 30 days. No additional storage package, job, automatic retry, push or
publication is included. **Stop here and ask the user before either uploading or starting the job.**
