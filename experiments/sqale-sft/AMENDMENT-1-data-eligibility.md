# Preparation amendment 1: fill the dev pool (2026-11-07)

Recorded after deterministic CPU preparation and **before any model inference, sampling or training**.
No paid run, new corpus, teacher, commit or publication is approved by this amendment.

Build v2 retained 95 eligible dev schemas and therefore 95 dev questions under the one-per-schema
preparation rule. Preserve `dev.tasks.jsonl` and the base manifest unchanged. To reach the planned 100
questions, add five distinct questions from five of those same dev schema groups, with at most two
questions per schema. Do not change train, Q2 or final membership.

Select existing dev schemas in SHA256(`dev-topup-v1:` + schema_id) order. Within each, examine source
questions in the existing SHA256(question_id) order, excluding already selected IDs. Apply the same
original-data materialization, full/sliced gold-result verification, placeholder filtering, context
limit and scripted replay checks. Take the first eligible additional question and continue to the next
schema until there are 100. Log submissions and exclusions; never consult a model outcome.

Write `dev-100.tasks.jsonl` and a separate amendment manifest with hashes of the base manifest, original
95 tasks, new 100 tasks, additional replay records and this amendment. The original artifact and its
95-task verification remain available. Report **100 questions across 95 schema groups** and use grouped
uncertainty; do not treat all 100 as independent databases. Final remains 150 questions across 94 groups.

This adjusts a data-preparation count after source eligibility checks. It does not alter the 10-point
G1 threshold, the proposed 3-point spending rule, the 10% continuation-size gate, any model-selection
result or the closed shop test. The model-run preregistrations must reference the amended dev hash.
