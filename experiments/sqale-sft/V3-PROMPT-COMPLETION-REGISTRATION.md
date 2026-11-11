# V3 local prompt completion (2026-11-07)

Free preparation registration: `sqale-v3-local-20261107-02`. No model sampling, inference, optimizer,
training, GPU, paid service, commit, publication or new dataset is authorized.

The first v3 automated filter missed marked training fragments (`SQL Logic`, `Reasoning`,
`Domain Grounding`, entity descriptions, title-only tasks, and "Generate basic/straightforward
questions"). A broader scan found them before any model run or data freeze. Read every flagged
primary prompt, reject these fragments, strip difficulty/title/formatting prefixes where a complete
database question remains, and apply the refined rules to alternatives. The 850 held-out primary
prompts already received individual full-text review; preserve those explicit decisions and report
any changed text/count separately. Do not replenish excluded questions.

Keep all intermediate files under `data-build-v3/pre-prompt-completion/`, including initial audit
records and source snapshots, before revising active v3 files. Keep v2 unchanged. Initial scripted
audits are preparation checks, never model results. Reuse an initial replay only when complete
conversation/prefix text, call names/arguments/results, database hash and final full-result score
match. Freshly replay every changed episode. Re-score every final record with AuditKIT. Log initial,
reused and new submissions with provenance; repeats are not additional unique data.

Recompute nested 500/1500/full counts and unique per-pass target gates, validate scalar shop golds,
record every new exclusion/change, verify all source snapshots and byte-check every v2 artifact.
This completes Amendment 2, including its proposed 5-point stage rule; it grants no paid approval.
