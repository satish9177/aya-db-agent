# G0 result: failed on the first training forward (2026-11-07)

Job 6aef3ba90e0338d1bb835c85,
hf-user, one t4-small. Launched from clean local commit `413dff8` with the exact approved
registration SHA256 `84799ec8cbdd3a57602b09084e3bf8aec1f82f572b5baabeb71a5d51828c8c81`.
The approved private input upload completed before launch. No edits preceded launch. The job stopped
automatically; **no retry, second job, extra upload, push or publication** followed the failure.

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, source/data identity, private upload | PASS | Launcher guards and local upload/launch receipts |
| CUDA and pinned NF4 model loading | PASS | Tesla T4, CUDA 12.6; execution reached the first model loss |
| Runtime reference oracle | PASS | Four references checked before model use |
| AgentTune factory and exact longest-example encoding | PASS, construction only | Factory saved fresh training membership: 4,072 inputs/431 targets, exact audited encoding |
| Three finite committed optimizer steps | FAIL | CUDA OOM during step 0's forward cross-entropy; 0/3 updates completed |
| Checkpoint 2 optimizer/scheduler/RNG resume | NOT RUN | No checkpoint was reached |
| Checkpoint 3 exact tensor reload | NOT RUN | No checkpoint was reached |
| Finite inference and generated tokens: Base/Pilot/reloaded adapter | NOT RUN | All three inference conditions were skipped after training failed |
| AuditKIT agreement on saved inference scores | NOT RUN | No inference trajectories exist |

The traceback passes through AgentTune's training wrapper into the audited factory/HF Trainer, then
Cohere2's causal-LM loss and PyTorch cross-entropy. It failed while requesting another **3.98GiB**.
No optimizer update, trained adapter, model accuracy, recovery result or successful throughput was
produced. Construction success does not establish training compatibility. This run cannot pass G0.

## Time, cost and memory

- Created 14:54:33.496UTC; started 14:56:35.932UTC; finished 14:58:52.592UTC on 2026-11-07.
- Actual start-to-finish runtime **136.660s (2m16.66s)**. Provider integer durations: scheduling 122s,
  running 136s, total 259s. The failed train-to-2 subprocess lasted 70.567s, including model loading.
- Registered/live rate **$0.006667/min**. Runtime prorated at that rate is $0.01518; counting three whole
  running minutes gives **approximately $0.02000 compute**. These are rate-based calculations, **not an
  observed invoice charge**. The SDK exposes no job charge; the authenticated billing-page read reached
  a login page, so exact charged compute/storage is unavailable. No new billing action was taken.
  [Official pricing](https://huggingface.co/docs/hub/en/jobs-pricing) says Starting/Running are billable
  and failed jobs stop billing; queued/scheduling time is reported separately, not silently charged here.
- Exact **peak GPU memory is unavailable**: the registered runner saves peak counters after successful
  training, and this run failed first. At OOM, the actual diagnostic reports 14.74 GiB device total,
  2.31GiB free, **12.42GiB process usage**, 11.72 GiB allocated by PyTorch and 591.85 MiB reserved but
  unallocated. These are a failure-time snapshot, not peak counters. Do not substitute them for a peak.

Pulled private outputs: `remote/hf-sql/hf-user/g0-sql-agenttune-20261107-01/` (environment, oracle
check, failure status/log and factory training-set manifest). Redacted provider log/metadata remain
under `runs/hf-sql/hf-user/g0-sql-agenttune-20261107-01/`. Retention remains approved for at most
30 days from the upload, through 2026-12-07; no extension or further bucket writes are authorized.

## Literal-submission diagnostic and later-stage costs

Amendment 5 was recorded prospectively while G0's uploaded scorer stayed frozen. Separate local
diagnostics report Base/Pilot/checkpoint conditions **not run: 0 episodes, 0 submissions, 0 no-table
submissions each**. This is missing inference coverage, not evidence the model avoids literal answers.
G0's original scores/registration/output files were not overwritten.

**500 and 1,500 costs cannot be estimated from successful G0 throughput: there were zero completed
updates.** Their two-pass factory training workloads are 1,694,130 / 5,369,570 input tokens and
214,762 / 661,484 target tokens. Maximum actual training input lengths are 3,849 / 3,891, with validation
maxima 2,695 / 3,965. These shorter maxima do not establish that the current loss path fits on a T4.
The failed forward's five-second progress display is not a training-speed measurement. Do not carry
forward the historical 369 tokens/s or the previous $10 training cap as a measured forecast.

The [500-stage draft](../sqale-500/REGISTRATION-DRAFT.md) is explicitly **blocked and not runnable**.
Memory compatibility and valid training/inference timing must be resolved before it can have an
honest cost/time cap and be presented for approval. No memory/precision/hardware change or new job
has been attempted. Machine-readable results: `RESULTS.json`.
