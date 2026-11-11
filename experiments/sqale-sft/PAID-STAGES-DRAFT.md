# Proposed paid stages; G0 finished/failed, others unapproved (2026-11-07)

Update after the user's exact one-run approval: G0 job 6aef3ba90e0338d1bb835c85 failed with CUDA OOM
before its first optimizer update. No retry. See `../sql-g0/RESULTS.md`. The 500-stage draft is blocked
at `../sqale-500/REGISTRATION-DRAFT.md`; no successful measured throughput supports a stage cost yet.
All larger caps below remain historical planning estimates, not runnable registrations or approvals.

Latest free amendment 6 implements selected logits with native loss, CPU equivalence proof and durable
peak/OOM logs. The distinct `../sql-g0-02/REGISTRATION.md` is an unapproved $3 proposal, adding one short
GPU loss comparison to the original schedule. No second run/upload has occurred; 500/1500 costs stay TBD.

Update: the user approved G0-02 and it passed (job 6aef547de6320c66fad5ff75, ~$0.08 compute by rate;
`../sql-g0-02/RESULTS.md`). Measured forecasts for 500/1,500 training and the dev screen are now in
`../sqale-500/REGISTRATION-DRAFT.md` (A500 training ~1.8h/$0.71; A1500 ~5.6h/$2.24; estimates, not caps).
The caps below remain historical planning numbers; each paid stage still needs its own approval.

Companion to `PREREGISTRATION.md`. These are planning caps from the independent review, not measured
costs or permission to launch. Freeze one uniquely named registration and obtain explicit approval for
each run. Do not launch until the expected billing account, balance, model access and launcher guards
are established. No run may draw on another account's balance implicitly.

| Stage | Proposed cap | Inputs / stop rule |
|---|---:|---|
| G0 factory/resume/reload smoke | $3 total | One draft: `experiments/sql-g0/REGISTRATION.md`. Longest4072-token example through AgentTune for two steps, optimizer resume to 3; three fixed dev tasks each for Base/Pilot/reloaded checkpoint. One T4-small, two-hour hard job limit including setup, NF4/fp16 throughout. No vLLM test. Approval must include private input uploads and bounded storage. |
| A data stages | $10 total | Nested 500/1500/4888 source stages (4887 eligible full-stage examples after amendment4), including 235 shop conversions; train/screen only after G0 and individual approvals. Proposed rule: stop at 1500 unless it beats 500 by at least five unrounded dev percentage points on the same 95 tasks. |
| Dev screens | $5 total initially | At most eight configurations x 95 effective dev questions; keep final labels out of selection. Report shop dev 40/mh 40 retention separately and include its cost. |
| Q2 sampling | $8 maximum initially | Four attempts/question, initially 300 questions; consider the effective pool of 579 only after measured throughput and a pre-sampling verified-token forecast. No sampling without approval. |
| A+ and B continuation | $5 combined | Same A adapter/settings and retained question IDs, exact supervised-token match across actual updates, globally with at most one partial example. Use preregistered deterministic minimum-drop fallback on both arms and report coverage. Neither trains unless G2 and the post-drop 10% unique per-pass gate pass, using A's actual post-validation training set. |
| Final evaluation reserve | $15 initially | Reserve before optional arms. Three finalists at most; SQaLe 144 and BIRD 493 final tasks reported separately, plus shop dev 40/mh 40 retention. BIRD download/preparation is approved; model evaluation requires separate approval and frozen selection. |

This core ceiling is $46 before optional runs and reserve changes. It does not promise those stages
fit these caps; re-estimate from G0 and episode lengths. Faster-engine savings are unmeasured. The
independent review's broader $65-67 totals include options that are not authorized here.

Second seed, Aya Expanse, a closed-model reference and GRPO are separate decisions with separate
registrations/approvals. SDPO is outside the core study. None may consume the final-evaluation reserve.

Free G0 preparation is complete: dedicated SQL runner/launcher, expected-hf-user account/price
guard, pinned source/assets, complete checkpoints, long wall limits and exact local registration.
Read-only model-access/price preflight passed. The user requested both local commits; no push occurred.
Only the completed/failed G0 upload/job was approved. Local scripted/tensor/spy checks do not establish
real CUDA inference or optimizer serialization/resume; G0 failed before those checks. No future input
upload/job is approved. See `experiments/sql-g0/RESULTS.md` and amendment 5. No launch command is included here.
