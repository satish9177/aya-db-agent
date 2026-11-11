# G0-02 result: PASS (2026-11-07)

Job 6aef547de6320c66fad5ff75,
hf-user, one t4-small (Tesla T4, CUDA 12.6). Launched once from clean local commit `51270ad` with
the exact approved registration SHA256 `e86597148c4ccdeb645777fb85f4090ec0eb1156789172165b425242d70f2580`,
after the user's explicit approval in a new Claude Code session. The approved private input upload
(16 files, 143,206,958 bytes) completed before launch. **No retry, second job, extra upload, push or
publication** followed. Machine-readable results: `RESULTS.json` (generated from the pulled files).

## Checks

| Registered check | Result | Evidence |
|---|---|---|
| Expected account, pinned revision, price, private upload | PASS | Preflight: hf-user, `af89d219…`, $0.006667/min; launch/upload receipts |
| CUDA and pinned NF4 model loading | PASS | `environment.json`; `expandable_segments:True` recorded |
| Five runtime reference checks before model use | PASS | `oracle-precheck.json`: 5 checked, 0 model calls |
| Short native loss check, full vs selected positions | PASS | 0.6791874170 vs 0.6791874766, difference 6.0e-8 (tolerance 1e-4); 1,219 vs 106 projected positions; no update |
| AgentTune factory, exact longest-example masks | PASS | Fresh encoding verified: 4,072 inputs / 431 targets, `trained-positions-v1` |
| Three finite committed optimizer updates | PASS | Losses 0.5216 / 0.1681 / 0.1034, grad norms 1.337 / 0.463 / 0.207; 0 skipped; ledger 1,293 targets |
| Checkpoint 2 optimizer/scheduler resume | PASS | Fresh process resumed from checkpoint 2 and ran only step 3; step-3 learning rate 5e-5 matches the cosine schedule (a lost scheduler would restart at 2e-4) |
| Checkpoint 3 exact tensor reload | PASS, record file not synced | See "Known issue" below |
| Inference generates tokens: Base / Pilot / checkpoint 3 | PASS | 210 / 1,042 / 393 generated tokens |
| Finite one-position logits | PASS | Checked forwards equal generated tokens in all three; `logits_to_keep=1` enforced by hooks |
| Durable GPU memory events | PASS | After load, each update, completion, each inference phase; no OOM |
| AuditKIT agreement with the guarded scorer | PASS | 3/3 per condition, 9/9 total |

Adapter hashes match the reports: checkpoint 2 `bea188ac…`, checkpoint 3 and final adapter `11f9371e…`.
The temporary adapter was trained on one example for three steps. It is **not adoptable** and is not
a model result.

## Known issue: reload record not synced

The checkpoint-3 inference step writes `checkpoint-reload.json` (tensor count, all-match flag, adapter
SHA). It is missing from the synced output. Cause, reproduced locally: the output-sync ignore rule in
`scripts/sql_job.py` drops every name that starts with `checkpoint-` and has no `_COMPLETE` child.
The rule is meant for half-written checkpoint folders, but it also matches this file.

The check itself passed. `verify_reloaded_adapter` runs before inference and raises on any key or
tensor mismatch. Nothing catches that error (`record_oom` re-raises only OOM), and the step exited 0
and wrote its summary afterwards. Only the record of the check is lost. The launcher was **not
changed**; it needs a prospective fix before the next registration.

## Inference (compatibility only, not an accuracy result)

Three fixed dev questions, greedy, eight tool steps. Correctness is reported, not a pass condition,
and n=3 supports no accuracy claim.

| Condition | Strict / fallback / set | Submissions (no-table, all errors) | Tool calls per episode | Seconds per episode |
|---|---|---|---|---:|
| Base NF4 | 0/3, 0/3, 0/3 | 1 (1) | 1, 1, 1 | 12.4 |
| Pilot NF4 | 0/3, 0/3, 0/3 | 2 (1) | 7, 8, 7 | 83.6 |
| Checkpoint 3 NF4 | 0/3, 0/3, 0/3 | 2 (2) | 1, 2, 2 | 26.2 |

Every no-table submission was a query that failed (missing table/column or a syntax error), not a
constant answer. All valid-no-table counts are 0.

## Time, cost and memory

- Created 16:40:29.396 UTC; started 16:40:34.091; finished 16:51:44.986 on 2026-11-07. Running
  **670.895s**; provider scheduling 4s, total 675s.
- Step times: short check 68.5s (includes the 6.7GB base download), train-to-2 56.0s, resume-to-3 38.0s,
  Base 56.0s, Pilot 271.0s, checkpoint 3 99.3s. Setup and final sync outside steps: about 82s.
- Rate $0.006667/min: **$0.0745** prorated, **$0.0800** at 12 whole minutes. Rate-based, **not an
  invoice**; exact charged compute and storage are unavailable.
- Training speed: 2 updates of 4,072 tokens in a 30.56s Trainer runtime, about **266 tokens/s**,
  including a checkpoint save after every update. One example only.
- GPU peaks (device 14.74 GiB): short check 6.51 / 7.18 GiB allocated/reserved; train-to-2
  8.87 / 10.18; resume-to-3 **8.91 / 10.21** (run maximum); inference 2.67-3.54 / 2.80-3.72.
  Training now leaves about 4.5 GiB of headroom on the longest example, where G0-01 ran out.

Storage: inputs 143,206,958 bytes plus output snapshot 671,502,991 bytes (3 checkpoints with optimizer
state, final adapter, logs), both in the private bucket `hf-user/aya-db-agent`, within the
512 MiB / 1 GiB caps. Approved retention is at most 30 days: **through 2026-12-07** unless the user
renews it. Pulled copy: `remote/hf-sql/hf-user/g0-sql-agenttune-20261107-02/`. Redacted provider
log/metadata: `runs/hf-sql/hf-user/g0-sql-agenttune-20261107-02/`.

## What this does and does not show

It shows that Tiny Aya Global trains through AgentTune's audited factory on a T4 at the longest
training example without running out of memory, resumes from a full optimizer checkpoint, reloads
its adapter and runs finite inference. It does not show that training helps, and it gives no
accuracy number. The throughput figure comes from three updates on one example; the 500-stage
forecast in `../sqale-500/REGISTRATION-DRAFT.md` carries that uncertainty.
