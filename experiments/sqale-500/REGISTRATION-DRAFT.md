# A500 registration draft: superseded by two runnable registrations

Draft ID `a500-sql-20261107-draft-01`. **Superseded** under [amendment 7](../sqale-sft/AMENDMENT-7-a500-stage.md)
by [`a500-sql-agenttune-20261107-01`](../sql-a500/REGISTRATION.md) (train A500, then its dev screen) and
[`dev-baselines-sql-20261107-01`](../sql-dev-baselines/REGISTRATION.md) (Base and Pilot dev screens,
run in parallel). Neither is approved yet. The settings and forecasts below are the ones they use;
the "Before approval" items 1-3 are done (launcher fix, exact registrations, retention deferred).
Machine draft: `REGISTRATION-DRAFT.json`; the SQL launcher cannot consume it.

G0-01 failed with CUDA OOM before its first update. G0-02 (job `6aef547de6320c66fad5ff75`,
[results](../sql-g0-02/RESULTS.md)) **passed** on the same T4: native-loss check, three updates on the
longest 4,072-token example, checkpoint resume, adapter reload and nine inference episodes. That
removes the block on planning. A runnable registration still needs the items under "Before approval".

## Fixed study inputs and proposed settings (unchanged)

- Tiny Aya Global at `af89d219b53ed9b13b8a4645f9c8028973510324`, official tokenizer,
  `sql-submit-v1` tools and **table-read-v1** strict scorer. Expected account hf-user.
- Frozen source 500 examples: `data/sqale/data-build-v3/train-500.episodes.jsonl`, SHA256 in JSON.
  Apply the existing run policy, preserve file order and seed 42 index shuffle, split 475 train/25
  validation. Two complete passes, microbatch 1/accumulation 16, LR 2e-4, existing NF4/fp16 LoRA
  rank 16/alpha 32/dropout .05, `trained-positions-v1` loss as in G0-02.
- Training inputs per pass 847,065; unique supervised targets 107,381; 10% B minimum 10,739.
  Two-pass totals 1,694,130 input/214,762 target tokens. Runtime fresh encoding and exact train-set/
  committed-token manifests are mandatory. Preserve full examples and audited error/tool masks.
- Optimizer checkpoints every 10 updates, retain 3, sync only complete checkpoints.
- First dev screen: same 95 effective questions for Base, existing Pilot and A500; greedy, eight
  tool steps, 512 new tokens/turn, same engine/precision. Strict/table-read, set/fallback, protocol
  errors, every-submission no-table counts, token/time/cost and AuditKIT agreement reported separately.
  Shop dev 40/mh 40 retention is specified and budgeted separately.
- After that first full screen, freeze/read 20 failures and 10 successes per model under amendment 5's
  manual-audit protocol, reporting availability/shortfalls and descriptive labels; no scores change.

## Memory

The 500 set's longest training input is 3,849 tokens (validation 2,695), shorter than the 4,072-token
example G0-02 trained on with a peak of 8.91 GiB allocated / 10.21 GiB reserved of 14.74 GiB. Gradient
accumulation adds no activation memory. Validation runs loss-only: the trainer has no metrics function,
so Transformers 5.22.1 discards logits during evaluation (`trainer.py` evaluate: `prediction_loss_only=True
if self.compute_metrics is None`), and validation goes through the same selected-position loss.

## Forecast from G0-02 (estimates, not caps)

Basis: t4-small at $0.40/h; training **266.5 tokens/s** (two updates of 4,072 tokens in 30.56s,
including a checkpoint save after each); dev time per episode from G0-02's three tasks per model.

| Item | Hours | Cost |
|---|---:|---:|
| A500 training, 1,694,130 input tokens | 1.77 | $0.71 |
| A1500 training, 5,369,570 input tokens | 5.60 | $2.24 |
| Dev screen, Base, 95 tasks at 12.4s | 0.33 | $0.13 |
| Dev screen, existing Pilot, 95 tasks at 83.6s | 2.21 | $0.88 |
| Dev screen, a trained model (between Base and Pilot speed) | 0.33-2.21 | $0.13-0.88 |
| **A500 plus the three-condition dev screen** | **4.6-6.5** | **$1.85-2.60** |

Not included: setup and model loads (about 2-3 minutes per step), validation (about a minute for
500), retention, storage. Treat all figures as roughly ±50%: throughput comes from three updates on
one long example (the 500 set averages about 1,783 tokens per example), and dev times from three
tasks per model. Base and Pilot screens do not depend on A500, so they could run as a second job in
parallel, saving about 2.5 hours of wall time at the same cost.

## Before approval

1. Fix the launcher output-sync filter (prospective). It drops every name starting `checkpoint-`
   without a `_COMPLETE` child, which lost G0-02's `checkpoint-reload.json`.
2. Freeze exact runnable steps, assets (500 episodes and their databases, 95 dev tasks and databases,
   pilot adapter), upload bytes, source hashes, timeouts and a total cost cap including storage.
3. Decide retention scope and whether Base/Pilot screens run as a parallel job.
4. Ask the user to approve that exact registration, its uploads and storage. No hardware switch,
   training-memory change, upload, evaluation, teacher, Q2/BIRD access or paid job is authorized here.

The proposed later stage rule remains: stop at 1,500 unless it beats 500 by at least five unrounded
strict dev percentage points on the same 95 tasks. This draft neither runs that comparison nor
spends the final-evaluation reserve.
