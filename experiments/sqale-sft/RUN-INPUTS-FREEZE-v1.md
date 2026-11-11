# Effective run inputs and A gate (2026-11-07)

Amendment 4 supersedes the historical full-pool gate figures while preserving all v3 artifacts.
`RUN-POLICY-v1.json` SHA256 is
`7012d5357fd4518bdcd2dcca83165a7560a28ccd6806113e08111499cc31362e`.
The local/private manifests are `data/sqale/runner-inputs-v1/a-{500,1500,4888}.json` and `summary.json`.
They contain exact train/validation IDs, trajectory identities and input/source hashes. No model has run.

| Source stage | Eligible corpus | A factory train | Validation | Unique A train targets/pass | 10% minimum B targets/pass |
|---|---:|---:|---:|---:|---:|
| 500 | 500 | 475 | 25 | **107,381** | **10,739** |
| 1,500 | 1,500 | 1,425 | 75 | **330,742** | **33,075** |
| 4,888 | 4,887 | 4,643 | 244 | **1,067,919** | **106,792** |

Preserve file order, shuffle row indices with `random.Random(42)`, and hold out
`max(1, round(n*.05))`. The excluded fragment appears only in the full stage. The runner uses the
same function, freshly verifies factory encodings and persists its actual training-set manifest.
Validation target totals are respectively 5,162 / 19,439 / 55,541, excluded from A's denominator.
Training inputs per pass are 847,065 / 2,684,785 / 8,862,572. No epochs, copies, padding or validation
can inflate the unique gate. A later changed split/mask/input requires a new denominator.

Effective dev is **95**, Q2 **579**, SQaLe final **144**; shop retention remains 40+40. Four known
pre-model exclusions are explicit in the policy: one training entity-description fragment, one dev
scalar scan-order case and two Q2 scan-order cases. No gold rewrite/replenishment or model outcome
informed this overlay. The frozen v3 files/counts remain independently verifiable as before.

The proposed stage rule uses the same 95 dev tasks: stop at 1,500 unless it improves over 500 by at
least five unrounded percentage points (at least five net successes, 5/95=5.263 points). This is not
paid approval or significance evidence.

The active continuation wrapper is `ayadb.continuation.match_control`: retain the full question set
when feasible; otherwise apply preregistered `minimum-question-drop-v1` to both arms. Its minimum
cardinality is checked against exhaustive subset search in small synthetic tests. Report dropped IDs,
question/episode/target coverage, whole repeats and the single partial mask. Recompute the 10% gate
after the drop; neither continuation trains on a failed post-drop gate. Runtime continuation roles
also require a matching/gate manifest, common A-adapter hash, common settings and exact target budget.
No real B samples/control files exist yet; they await separately approved Q2 sampling.

Core stage SHA256s remain those in the v3 freeze. The first local commit is `3fe7d82`; the follow-up
implementation is committed separately after local validation. No dataset or tokens are committed,
and no push, private-bucket write or paid work is authorized by this freeze.
