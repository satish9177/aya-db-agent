# G0 SQL AgentTune compatibility check: draft, not approved

Registration ID: **`g0-sql-agenttune-20261107-01`**. Machine registration: `REGISTRATION.json`, SHA256 **`84799ec8cbdd3a57602b09084e3bf8aec1f82f572b5baabeb71a5d51828c8c81`**. This is one proposed paid job. The user must explicitly approve both its private uploads and job before either occurs. No upload, job, model inference or training has started.

## Fixed scope

- Expected HF account and billing namespace: **hf-user**. Hardware: **one t4-small**, NF4 double quantization/fp16 compute throughout. Refuse a different account, public bucket, changed source/data hash or higher registered hardware rate.
- Student: `CohereLabs/tiny-aya-global@af89d219b53ed9b13b8a4645f9c8028973510324`. Use the official, separately pinned tokenizer and audited assistant/error masks. Rank 16, alpha 32, dropout .05; LR 2e-4; seed 42; microbatch 1, accumulation 1 for this memory/optimizer smoke. No teacher or paid LLM API.
- Longest retained training example: `schema_011342_s2_q7_7b9a6dce95-gold`, **4,072 input / 431 target tokens**, freshly checked locally. One example only, no validation split for this disposable compatibility probe. It is not an A training run.
- Inference: the same three fixed SQaLe dev questions from three schema groups for Base, existing Pilot and the reloaded temporary checkpoint: **nine episodes total**, greedy, seed 42, eight tool steps, 512 new tokens/turn, 8,192 context. Correctness is reported as observed, not used to pass the compatibility check.
- No BIRD, Q2, final evaluation, full A/A+/B training, GRPO, other model, vLLM validation, publication, push or automatic retry. The three-step adapter is explicitly **not adoptable** as A or the submission model.

## Ordered job steps

Before any model use, record runtime/library/SQLite/CUDA versions, require CUDA, and recheck the four unique reference inputs (longest training example plus three dev golds). Abort on a runtime oracle mismatch rather than silently changing the scoring contract.

| Step | Registered action | Step ceiling |
|---|---|---:|
| train-to-2 | Plan three steps, stop/save after step 2 through AgentTune create_distill_trainer -> Project.distill -> AuditedSFTFactory -> HF Trainer | 1,800 s |
| resume-to-3 | Fresh process/model/factory; restore checkpoint 2 model, optimizer/scheduler and trainer/RNG state; complete step 3 with identical data/settings | 1,800 s |
| infer-base-nf4 | Untuned model on the three fixed dev tasks | 1,200 s |
| infer-pilot-nf4 | Existing pilot adapter, same model/precision/tasks | 1,200 s |
| infer-checkpoint3-nf4 | Freshly load checkpoint 3, compare every adapter tensor against saved weights, then infer the same tasks | 1,200 s |

Global job timeout is **7,200 seconds including setup**; per-step ceilings do not extend it. Stop at the first failed check. Never truncate the longest sequence, reduce precision/length, change examples, switch hardware or launch another job automatically after a failure.

## Upload and spending envelope

Only **14 pinned files / 143,130,042 bytes** of input assets are proposed: one training episode, three dev task records, **four SQaLe database files**, five tokenizer/config records and the existing pilot adapter files. Registered source files are bundled separately. Exact paths, hashes and sizes are in JSON; no raw BIRD/Q2 data or model base weights are uploaded. No generated data or credentials enter Git.

Private input prefix: `hf-user/aya-db-agent/sql-inputs/g0-sql-agenttune-20261107-01/<registration-sha>/`. Private output prefix: `hf-user/aya-db-agent/runs/sql/g0-sql-agenttune-20261107-01`. Existing runs and bucket root are outside deletion/sync scope. Input assets are bounded at 512 MiB; output snapshots at 1 GiB. Keep three completed optimizer checkpoints, sync every 60s and after each step; incomplete checkpoints are excluded. Pull into an account/run-specific local directory.

**Total proposed run budget: $3.** Current API price is $0.006667/minute; a two-hour T4 timeout implies about **$0.81 maximum scheduled compute**. Rate is rechecked before a write/job. Reserve **$0.04 for the first month of bounded private storage**, with remaining budget headroom. Review/pull artifacts promptly and remove/re-authorize the unique G0 storage prefixes within 30 days; the GPU timeout does not delete storage. No new storage package, larger quota, extra job or longer retention is automatically approved.
Sources checked 2026-11-07: [HF Jobs pricing/timeout](https://huggingface.co/docs/hub/jobs-pricing), [bucket pricing](https://huggingface.co/storage). These are ceilings/planning estimates, not a promise of observed runtime or account balance.

## Pass, fail and reporting

Pass requires three finite committed optimizer steps via the AgentTune factory, exact input/mask counts, checkpoint 2 data/settings/optimizer identity and resume to 3, exact checkpoint 3 adapter reload, finite inference logits, nonempty generated tokens in every condition, and AuditKIT agreement on all saved strict scores. Strict/set/fallback accuracy, malformed/fabricated calls, SQL errors/loops, input/generated/trained/padded/repeated tokens, timings, peak GPU memory and actual cost are reported separately. Compatibility success does not demonstrate broader SQL improvement, recovery, vLLM support or speedup.

Fail/stop on account/price/hash mismatch, no CUDA, OOM, nonfinite loss/logits, skipped optimizer update, wrong masks/steps/checkpoint/reload, source-oracle drift, sync failure, upload byte limit, stalled output, or step/job timeout. Save the failure and last complete checkpoint; no automatic restart. A new attempt needs a new registration and approval.

Free checks completed: CPU scripted/spy tests, actual longest-example tokenization, exact asset/source plan validation, runtime-oracle checks locally, and read-only hf-user gated-weight HEAD/price preflight. Actual CUDA training, real optimizer checkpoint serialization/resume/reload, inference and throughput remain untested until this proposal is approved.
