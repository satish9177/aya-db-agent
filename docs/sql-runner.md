# Registered SQL model runs

The additive runner is `ayadb.sqlrunner`; the launcher is `scripts/sql_job.py`. Legacy `sft.py`,
`smoke.py`, `hf_job.py`, formats, tools, data and snapshots remain unchanged. **The single approved G0
has finished with CUDA OOM; no retry or further upload/job is approved.** Its registration/uploaded
snapshot stay frozen. See `experiments/sql-g0/RESULTS.md`; the 500-stage draft is blocked/non-runnable.

## Free local checks

```powershell
.venv/Scripts/python.exe scripts/sql_job.py plan --registration path/to/new-registration.json
.venv/Scripts/python.exe scripts/sql_job.py preflight --registration path/to/new-registration.json
```

`plan` validates source/data hashes and prints the bounded plan without a network call. `preflight`
only reads the HF account, a pinned gated-weight HEAD and hardware price. It must find `hf-user`,
verify model access and refuse an increased registered per-minute price. Neither writes a bucket or
starts a job. These are examples for a future complete registration. G0 now intentionally differs from
the locally improved scorer source; never refresh its registration or use it for another launch.

## Training and scoring

Training goes through AgentTune `create_distill_trainer -> Project.distill -> AuditedSFTFactory ->
HF Trainer`, using the exact prepared QLoRA model and the official tokenizer. The stock AgentTune SFT
backend is not used. The factory freshly verifies masks/counts, refuses truncation, retains supplied
partial masks and never unmasks a tool observation/error. The effective training/validation split is
shared with `runinputs.training_plan`; runtime encoding must match its actual-training-set gate.

Amendment 6 adds opt-in `loss_policy="trained-positions-v1"` beside the original full-logits path.
For each row, shift labels once, pass a boolean mask selecting trained next-token positions as
logits_to_keep, and matching selected shift_labels. Native Cohere2 still applies logit_scale and its
causal loss. Original labels and the Trainer accumulation-window num_items_in_batch remain intact;
there is no extra division by accumulation steps. The policy is part of checkpoint/settings identity.
Tiny random CPU native-loss/all-LoRA-gradient and real Trainer accumulation tests pass. Package,
tokenizer, whole-example lengths and masks are unchanged.

The default planned precision is NF4 double quantization with fp16 compute. Registered seeds, LoRA,
batch/accumulation, learning rate and token limits are explicit. `training-set.json` records actual
factory membership/fresh targets; `token-ledger.json` counts committed optimizer input/target/padding
tokens, visits, unique examples and skipped updates. A skipped optimizer step or nonfinite loss fails
the run. Continued training checks parent A identity and a frozen post-fallback matching/gate manifest;
without one it is blocked. Whole matched passes and no independent validation split are required for
A+/B so their token match survives training. Actual verified Q2 packages await separately approved sampling.

Checkpoints include model, optimizer, scheduler, RNG/trainer state and the token ledger. `_COMPLETE`
is written after saving. Explicit resume requires matching training data/settings identity and expected
step; there is no automatic restart. New training steps can have up to **24-hour** registered wall
limits (evaluations 12 hours, sampling/final 24), subject to each run's smaller job/budget timeout.
G0 remains limited to two job hours. These are runtime limits; the tool protocol still allows eight calls.

Inference uses the same four tools and AgentTune rollout loop, ending on `submit_sql`. Every submitted
query, including rejected/repeated calls, is logged. Full-result strict, exact-set and last-successful
query fallback scores stay separate. Native tool errors, malformed/fabricated turns, loops, generations,
tokens and runtime are recorded. AuditKIT re-scores the saved trajectories offline with no LLM scorer.
Generation forward passes check finite last-token logits. Reload checks compare all adapter tensors.

New registered inference can explicitly set `inference_logits_to_keep=1`, using LastPositionHFEngine.
The underlying GenerationMixin already defaults supported models to 1. The explicit route and native
base-class/PEFT prefill are CPU tested; runtime head pre-hooks reject projection over more than one
position before lm_head, and output hooks require a finite one-position result. The old HF engine is
unchanged. No inference protocol/generation-limit change is introduced.

Amendment 5 prospectively adds **table-read-v1** strict scoring through `sqlguard.py`: result equality
alone cannot pass a constant-only query; SQLite must report a read of an existing user table. Every
reference is checked before model use. Every submission gets a native read/execution diagnostic, and
summary counts include valid/error no-table queries and affected accepted episodes. Set/fallback
remain separate diagnostics. Offline AuditKIT uses the same guarded strict predicate. The old scorer
and G0's uploaded version are unchanged. A separate local inventory handles G0's historical logs.

The future Q2 eligibility predicate also requires zero malformed calls/invented observations and exact
assistant-call/log correspondence; it recomputes guarded strict success. After the first full dev screen,
the preregistered manual audit freezes 20 failures/10 successes per model, reports labels/shortfalls and
changes no scores. See `reports/2026-11-07-table-read-guard.md`.

Before the first model use, the remote launcher records Python, SQLite, library and CUDA/GPU versions,
then re-executes each unique registered reference query and compares prepared full gold where present.
Runtime-oracle drift fails the job before predictions or optimization, preserving the pinned scoring
contract across local Windows SQLite and the Linux job image.

BIRD is final-only. The runner rejects it in training and compatibility/dev/sample plans, and final
evaluation requires a frozen selection record matching the exact model revision/adapter. The spent
shop `data/test.jsonl` is explicitly rejected before reads.

## Upload and job boundaries

Every external write requires the exact reviewed registration SHA256 and explicit upload permission.
Launching additionally requires paid-job permission, a clean local commit, the expected account and
explicit personal billing namespace. There is no dirty-tree bypass or automatic extra job. A local
exclusive launch receipt prevents duplicate/ambiguous restarts. Failed steps stop the stage.

Only pinned, byte-bounded source/assets are bundled; G0 has four SQaLe databases and no BIRD/Q2.
All writes use private `hf-user/aya-db-agent`. Inputs are under a registration/hash prefix; outputs
are under `runs/sql/<run-id>`. Completed checkpoint snapshots sync every 60 seconds and after steps.
Only that new run prefix may be pruned to the registered checkpoint/byte limit; legacy outputs and
the bucket root are never deletion targets. Tokens and signed URL queries are redacted from logs.
CUDA guards prevent billed CPU fallback; step/stall watchdogs and the provider job timeout bound runs.

After a **separately approved and launched** run:

```powershell
.venv/Scripts/python.exe scripts/sql_job.py status --registration experiments/sql-g0/REGISTRATION.json
.venv/Scripts/python.exe scripts/sql_job.py pull --registration experiments/sql-g0/REGISTRATION.json
```

Results download into `remote/hf-sql/hf-user/<run-id>/`, avoiding cross-account/legacy collisions.
Status/pull do not require the current source to match an old snapshot and never launch a job. Review
and pull artifacts promptly; G0's private storage approval expires 2026-12-07. Its approved upload,
single launch and read-only pull have completed. No retry, additional input upload or cleanup occurred.

Real CUDA G0-01 failed in its first cross-entropy before any update; serialization/resume/reload and
inference throughput remain untested. Its peak counters were saved only on success, so that old OOM
log is a snapshot. New GpuMemoryRecorder wiring journals current/peak allocated/reserved bytes,
free/total and phase/step/time after model load, each update, completion and OOM. Peaks are cumulative
within each process. The OOM handler preserves/re-raises the error without clearing allocator state.
The launcher sets and records PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. G0-01 remains frozen.

The distinct `experiments/sql-g0-02/REGISTRATION.md` adds one GPU native full/selected loss
comparison on a whole 1,219-token/106-target example, no optimizer, before the original longest-example
train/resume/reload/inference steps. Global timeout remains two hours/$3. Local tests use tiny random
CPU models, scripted engines/tensors/spy trainers; **114 tests pass**.

G0-02 was approved by the user, launched once and **passed** (job 6aef547de6320c66fad5ff75; results in
`experiments/sql-g0-02/RESULTS.md`): real CUDA training, checkpoint resume, adapter reload and inference
now work on a T4. Its storage approval also expires 2026-12-07.

The output-sync ignore rule used to skip any name starting `checkpoint-` without a `_COMPLETE` child,
which lost G0-02's `checkpoint-reload.json` file. Since amendment 7 it skips only checkpoint *folders*
without `_COMPLETE` (and `.tmp` files); `snapshot_ignore` in `scripts/sql_job.py` has its own test.
The next registrations are `experiments/sql-a500/` and `experiments/sql-dev-baselines/`, built by
`scripts/prepare_sql_a500.py`. Both were approved, launched once from `15b12cc` and passed
(`RESULTS.md` in each folder). The A500 dev step's `checkpoint-reload.json` synced, which confirms the
fix on a real run. Any further registration needs its own explicit approval before upload or launch.
The next one, `experiments/sql-a1500/` (built by `scripts/prepare_sql_a1500.py`, amendment 8), was
approved, launched once from `67c530f` and passed (`RESULTS.md`): 180 updates, an exact step-180 reload
and 37/95 strict on dev. Its dev inference peaked at 10.1 GiB (A500: 4.5 GiB), probably from one longer
conversation; check memory before any registration with a longer step limit.

Amendment 9 adds an opt-in tool set beside the unchanged `SQLTools`: `src/ayadb/sqlfeedback.py`
(`column-hints-v1`), selected per evaluation step with `"tool_feedback": "column-hints-v1"` and
validated by `runconfig`. Tool names, arguments and descriptions are identical (tested AgentTune
schemas). A `no such column` error also lists the real columns of the tables in the query (and which
table has the named column). A query identical to an earlier failed one is refused instead of run
again. Without the field the runner writes the same outputs as before. Because this changed pinned
runner sources, the tests for the executed A500/baselines/A1500 registrations check their frozen SHA256
and load them without source pins. `experiments/sql-a1500-hints/` re-screened A1500 with these tools
once (approved, launched from `e6b4269`): 38/95 versus 37, below the 42 needed to adopt them, so later
registrations keep the original tools unless a new amendment says otherwise.

Amendment 10 adds training data build v4 (`scripts/build_sqale_v4_recovery.py`, `src/ayadb/sqlmutate_v4.py`)
beside the frozen v3 build. It uses the same rows and split as v3 `train-1500`, with re-check repairs and
step-by-step complex paths. Every changed episode is executed, re-encoded and replayed through AgentTune
before it is written. `experiments/sql-a1500v4/` trained on it with A1500's settings (approved, launched
once from `2a5b81e`). It passed every check but scored 35/95 against A1500's 37, so A1500 stays the trained
finalist.

Amendment 11 adds `src/ayadb/scoringlimits.py`. Every scoring-side execution gets a 30 s limit: the
reference, submitted and fallback queries, ordering probes and the pre-model reference check. Reference
results are cached per frozen database file. The agent tools keep 5 s, and the 50M-operation cap is
unchanged. Re-scoring all 570 earlier dev episodes gives identical results. Final registrations
(`experiments/sql-final-v1/`) use a frozen selection record (Base, A1500); a test now covers the runner's
selection gate. Q2 sampling registrations (`experiments/sql-q2-sampling-v1/`) use `stage: sample`, `purpose:
sampling`, two samples at temperature 0.8.
