# G0-02: selected-logits compatibility check, awaiting approval

Registration **`g0-sql-agenttune-20261107-02`**, machine file `REGISTRATION.json`, SHA256
**`e86597148c4ccdeb645777fb85f4090ec0eb1156789172165b425242d70f2580`**.
One proposed job only; **no upload or launch is approved or performed**. G0-01's registration and
failed result remain frozen. This is a distinct prospective protocol under amendment 6.

## Fixed model, data and loss

Expected account/billing namespace **hf-user**, one **t4-small**, **$3 total cap**, **7,200s hard
provider timeout including setup**. Same Tiny Aya Global revision
`af89d219b53ed9b13b8a4645f9c8028973510324`, official tokenizer, NF4 double quantization/fp16 compute,
rank 16/alpha 32/dropout .05, LR 2e-4, seed 42, microbatch 1/accumulation 1, no validation split, same
three-step scheduler plan. Model/package pins, whole sequences, training labels, tools, prompt and
generation limits are unchanged.

Train through AgentTune's supported audited factory. Opt into **trained-positions-v1**: project only
the positions whose **next** label is trained, passing a boolean mask and matching shifted labels to
Cohere2's own scaled-logit/native causal loss. Preserve Trainer's accumulation-window denominator.
The longest example remains **4,072 inputs / 431 targets**, with no truncation or extra masking.

The first GPU check uses the shortest whole eligible existing example:
`schema_010335_s3_q2_518f1ca496-gold`, **1,219 inputs / 106 targets**. Fresh local official-tokenizer
encoding verifies both counts. Compare full and selected native losses under NF4/fp16 autocast, eval
mode to disable dropout, no gradients/optimizer update; rtol 1e-4/atol 1e-4. Actual lm_head input counts
must be 1,219 versus 106. Abort on mismatch/nonfinite/OOM before starting longest-example training.

## Ordered steps

| Step | Action | Step ceiling |
|---|---|---:|
| short-loss-equivalence | Native GPU full/selected loss and actual projection-count check on the whole short example | 600s |
| train-to-2 | Three-step plan through AgentTune on the same longest example; stop/save after 2 | 1,800s |
| resume-to-3 | Fresh process restores checkpoint 2 model/optimizer/scheduler/RNG/data identity and completes 3 | 1,800s |
| infer-base-nf4 | Base on the original three fixed dev questions | 1,200s |
| infer-pilot-nf4 | Existing Pilot on the same questions | 1,200s |
| infer-checkpoint3-nf4 | Fresh checkpoint 3 reload, exact adapter tensor comparison, same questions | 1,200s |

Step ceilings do **not** extend the global 7,200s timeout. Inference stays greedy, eight tool steps,
512 new tokens/turn and 8,192 context: **nine episodes total**. Force `generate(logits_to_keep=1)`;
pre-projection and output hooks require exactly one projected position on prefill/decode and finite
last-token logits. Use the prospective table-read-v1 strict scorer, separate set/fallback diagnostics,
every-submission no-table/protocol counts and offline AuditKIT agreement. Correctness is reported,
not a compatibility pass threshold. The temporary adapter remains non-adoptable.

## Memory and spending boundaries

Set **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** in the job. Persist current and peak
allocated/reserved bytes, device free/total, phase/step/time after model loading, each optimizer step,
completion and OOM. Peaks are cumulative within each fresh process; take the maximum across phase
logs for the run. Never empty the cache before capturing an OOM, truncate, reduce precision or retry.

Proposed private upload: **16 pinned assets / 143,206,958 bytes**, plus hashed source. This contains
the unchanged original 14 assets and one short training record plus its additional SQaLe database:
**five databases total**, tokenizer/configs and existing pilot adapter. No BIRD/Q2/base weights are
uploaded. Exact hashes and paths are in JSON; assets remain gitignored.

Input prefix: `hf-user/aya-db-agent/sql-inputs/g0-sql-agenttune-20261107-02/<registration-sha>/`.
Output prefix: `hf-user/aya-db-agent/runs/sql/g0-sql-agenttune-20261107-02/`. Private inputs capped
at 512 MiB and completed outputs at 1 GiB, three complete checkpoints, sync every 60s/after steps. Review/
pull promptly; storage approval is requested for at most 30 days, with no automatic extension or extra
storage package. Earlier G0-01 artifacts are outside this run's write/delete scope.

Maximum registered hardware rate is $0.006667/min, rechecked before a write/job. Two hours implies
approximately $0.81 scheduled compute, with a $0.04 first-month bounded-storage reserve within the
**$3 total envelope**. These are planning ceilings, not observed cost or a successful throughput estimate.

## Pass, abort and current evidence

Pass requires the short native-loss agreement/projection check, five runtime reference checks before
model use, exact longest masks, three finite committed optimizer updates via AgentTune, checkpoint 2
optimizer resume, checkpoint3 tensor reload, all three inference conditions generating tokens, finite
one-position logits, durable memory events and guarded AuditKIT agreement. Report actual tokens,
timings, GPU peaks and cost separately; no accuracy, GPU-fit or speedup claim precedes execution.

Abort on any account/rate/pin mismatch, CUDA/OOM/nonfinite/skipped update, missing measurement,
loss/mask/projection/checkpoint/reload/oracle mismatch, sync/byte-cap failure, stall or timeout. Save
the failure/last complete checkpoint. **No automatic retry or second job.** No full training, Q2,
BIRD/finals, teacher/API, vLLM, hardware switch, push or publication is included.

Free evidence: **114 local tests pass**, including tiny random CPU Cohere2 native loss/all-LoRA-gradient
equivalence, real Trainer accumulation and base/LoRA last-position prefill checks. The exact source/
asset plan is validated locally. Actual quantized GPU training, short comparison, checkpoint resume/
reload, memory fit and inference throughput remain untested. Ask the user to approve this exact
registration **and** its private uploads/storage before either upload or launch.
