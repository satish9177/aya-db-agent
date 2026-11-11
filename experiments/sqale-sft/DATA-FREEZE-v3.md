# V3 private data freeze (2026-11-07)

Active root: `data/sqale/data-build-v3/`. Registration: `sqale-v3-local-20261107-02`, following the preserved first local preparation. This freezes preparation only; no model or paid run is approved. Never open `data/test.jsonl`.

## Membership and per-pass size gate

Training: 4,653 SQaLe + 235 shop = **4,888**. Dev **96** (91 schema groups), final **144** (92), Q2 **581** (35). Training retains 326 SQaLe schema groups. Shop retention: dev40 and mh40, separate from training.

| Stage | Unique episodes/pass | Input tokens/pass | Unique targets/pass | Masked input tokens/pass | Minimum unique verified B targets/pass (10%) |
|---|---:|---:|---:|---:|---:|
| 500 | 500 | 888,632 | 112,543 | 776,089 | 11,255 |
| 1500 | 1,500 | 2,833,034 | 350,181 | 2,482,853 | 35,019 |
| 4888 | 4,888 | 9,325,316 | 1,123,565 | 8,201,751 | 112,357 |

These are prepared per-pass counts, not cumulative optimizer totals. A later approved run must record its actually consumed unique training examples/labels and freeze that denominator before Q2 sampling. Deduplicate both A and verified B; exclude validation, epochs, copies and padding. Actual A and B model results do not exist. Training repetitions/padding and model-generated/sampling tokens in this preparation: zero; sampling and paid-service cost: $0.

Control matching is global on exactly B's retained questions, with whole examples and at most one partial mask overall. A+/B share A adapter and settings. Report every other token category and sampling cost separately. Proposed stage rule: stop at 1500 unless it improves over 500 by at least five percentage points of unrounded strict dev accuracy on the same 96 tasks (at least five net additional successes). No difference detectable at this size is the wording for a tie.

## Checks and provenance

All **5,789/5,789** final scripted episodes pass AgentTune replay checks and AuditKIT. Reused after exact complete-conversation/call/DB/result checks: **5,772**; fresh changed-prompt replays: **17**. These are data checks, not model evaluations. The first 5,819-record check remains under `pre-prompt-completion/`.

All 850 SQaLe dev/final/Q2 primary prompts and 80 shop retention prompts were individually read by the assistant. Removed: 96 prompt fragments plus 45 ambiguous/unresolved ordering tasks; no replenishment. Final SQaLe repair demos: wrong column230, wrong table437, ambiguous column371, aggregate misuse334 (1,372 total), plus 66 existing shop error episodes. Every injected failure is executed and masked.

Original and projected references for all 80 retention and 235 shop training tasks pass stored gold. Legacy IDs, seeds and LF task pins are preserved. Clock-dependent date functions remain blocked; ordinary now text is accepted. Tied top-level ordering is unordered; ambiguous LIMIT boundaries are excluded. Source/framework limitations remain as recorded in the report.

Every submitted query is logged with stage/provenance: **12,808** actual preparation submissions across both registrations, including intermediate/fresh builder and replay events. Cached final rows are labelled as cached and do not count as new submissions.

Full suite: **71 passed**. Maximum training input: **4,072**, below 4,096. New parser is separately hash-pinned in `requirements-sqale-v3.txt`; all previous pins remain. No sealed shop test access, model run, paid work, BIRD download, commit, push or publication occurred.

All **5,903 v2 files / 1,248,656,198 bytes** match the before inventory. The v2 inventory SHA256 is `ce2e82e09ed8c48fec2f780bf60d616aee96364d0bf6dd6d77d15b2a7d896aa4`.

## Artifact SHA256

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `48255585290122e602f15860856fc71b27702e55063fe8bddec8025ef6715f1e` |
| `train-500.episodes.jsonl` | `b34bbd824142b955f02375deb26bf1d09e75107c425feb16180e2b6d2818fe3c` |
| `train-1500.episodes.jsonl` | `0c920dab1c5b09256ef7f9c76b4864f7e075ce72db3ca14d4dc978849b46b62e` |
| `train-4888.episodes.jsonl` | `d735dc98b5090b60b0f1d0720fe3dcc809ef33213c4222898fd6a64801906f79` |
| `train.episodes.jsonl` | `db4ed958994c2d52510df782ba72148c5d68feedba424338a2446c9f826ce64e` |
| `train.tasks.jsonl` | `422420935aeac6074d1da241e57e0bdb26a78d4f7a4954f5c3a71bc0d1134890` |
| `shop.tasks.jsonl` | `7882aeb47ee5d17b35c48f17275bdb57b55c5c11690c44fa205127879c0c5423` |
| `dev.tasks.jsonl` | `db613ff79c2b08753d13f5624ea1950d830aa259b8e35a9f1434409001358716` |
| `final.tasks.jsonl` | `b3f54d429b96f074c69841392901b55fc9d6bd224653d4d7df802a7480c8f629` |
| `q2.tasks.jsonl` | `fec3673c7e5f8881d172cc5fe1edfc90d7861c4c6286e647a5be36447765b283` |
| `retention/dev.tasks.jsonl` | `9e2d74ca5c5f9fcc6b573fb48c342606ad8f6454c61563056f866bc46b677dee` |
| `retention/mh.tasks.jsonl` | `750c026cf1098ecb9bb1b9da7298b2ae6f7de6f854da741d33855d3317f2a6be` |
| `manual-review-sheet.jsonl` | `088ceb9fdc01a2767264955a4077d8f3135281a6efa88632b4b71188ab7287c6` |
| `manual-review-decisions.json` | `300e6a23c093153110e6062ceadaccedfe31928a8369f2c2afa5d15c9e845650` |
| `code-snapshot.json` | `9e78305db633be423a0db6e495e614ab585d8691e28edad5c3a3386d3fd20e7d` |

Retention original LF pins: dev `7798e6fe626c51fc4bacf39c8171120885fd78ea029b0627b89e9896e89cdaae`; mh `d71da41d178eaecc64ccc96e4fbf13ad3bfc7b2056b308e7e96d41a559fcf2e5`.

The private manifest hashes the recorded build outputs listed in `output_sha256`; its own hash is above. `code-snapshot.json` hashes the actual preparation, scorer, replay, mask/control and registration sources copied in `code/`. `verification.json` is a separate successful post-build check of those underlying artifacts. No commit was requested; the source snapshot is not a permission to bypass launcher requirements.

Read-only recheck: `.venv/Scripts/python.exe -m ayadb.sqaleverify_v3`. Do not rerun completed builders/audits over existing output. Each later model run needs a unique registration, an expected billing account and explicit user approval.
