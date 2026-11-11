# Amendment 6: selected training logits and G0-02 (2026-11-07)

The user authorizes a free additive loss/memory/inference fix, tiny random Cohere2 CPU equivalence
tests, a new G0-02 draft and local commit. **No upload, CUDA run, paid job or push is authorized.**
G0-01's registration/source/data pins/results remain frozen. No real pretrained model is loaded by
the free tests; they use a tiny random CPU model and synthetic masked tokens only.

## Native selected-position loss

Opt-in policy `trained-positions-v1` sits alongside the original `full-logits` path. For each batch
row, append one -100 to labels and shift left by one. Pass the resulting `shift_labels != -100`
**boolean** mask as `logits_to_keep`, and pass the matching selected shifted labels. Leave original
labels unchanged for Trainer's full accumulation-window target count. Do not shift between rows,
train the last hidden position, unmask tool/error observations, truncate examples or rescale targets.

Use Cohere2's own forward: it selects hidden states before lm_head, applies its native `logit_scale`,
then calls the same causal loss. Trainer forwards its original `num_items_in_batch`; retain native
sum-over-window reduction without a second gradient-accumulation division. Fail if native loss kwargs
are unavailable or a custom loss/label smoother would change the contract. Record loss policy in the
training/checkpoint identity, prohibiting resume with a different projection policy.

The G0-01 allocation of 4072*262144*4 bytes is 3.9765625 GiB, consistent with a dense fp32 vocabulary
buffer in its cross-entropy traceback. Projecting only 431 trained next-token positions reduces that
one fp32 tensor to 0.4208984375 GiB. This diagnoses that component, not total peak memory or GPU fit.

CPU proof compares native full and selected losses/logits and all LoRA gradients on a tiny random
Cohere2 with non-unit logit_scale and nonzero LoRA A/B gradients. Cover separated assistant spans,
masked observations, batched unequal masks, gradient checkpointing and real Trainer accumulation
against a single combined batch. Use loss/logit rtol 1e-6/atol 1e-7, gradients rtol 2e-5/atol 2e-7
(combined-batch accumulation rtol 3e-5/atol 3e-7). Position-zero labels must not inflate the denominator.

## Memory and inference

Persist allocated/reserved and peak allocated/reserved bytes after model load, after each optimizer
step, on completion and in an OOM handler, with free/total device memory and phase/step/time. Preserve
the allocator state: no empty-cache, truncation or retry. The launcher sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and records it in environment provenance.

The explicit inference path passes `generate(logits_to_keep=1)`. CPU base/LoRA generation tests check
the initial long-prompt prefill and subsequent vocabulary projections. Runtime head pre-hooks reject
more than one projected position before allocating a full-prompt vocabulary tensor; output hooks
also require one finite last-position logit vector. Existing tokenizer, prompt, tools, stop strings,
generation limits, pinned packages and model revision remain unchanged.

## New G0-02 proposal, not execution permission

One new ID: `g0-sql-agenttune-20261107-02`, expected hf-user, one t4-small, **$3 total cap**, the same
**7200s global timeout including setup**, no retry. Prepend one GPU native-loss comparison on the
shortest whole eligible existing training example (1,219 inputs/106 targets) using NF4/fp16 autocast,
dropout disabled and no gradients/optimizer update; compare losses at rtol 1e-4/atol 1e-4 and assert
the actual lm_head input counts. A mismatch/nonfinite/OOM aborts before longest-example training.

Then keep the original three-step schedule: stop/save after 2, resume to 3 through AgentTune on the
same 4,072-input/431-target example, now using selected logits. Checkpoint identity/reload rules and
all other hyperparameters remain. Infer Base/Pilot/reloaded checkpoint on the same three frozen dev
tasks (nine episodes), with last-position-only projection. Use amendment 5's guarded strict scorer and
report no-table/protocol/token/time/cost/memory diagnostics. Temporary adapters remain non-adoptable.

The short comparison has a 600s step ceiling; the original step ceilings stay unchanged. Their sum
does not extend the 7200s global deadline. Freeze exact assets/source hashes in the new registration,
with private inputs <=512 MiB, outputs <=1 GiB and retention <=30 days. BIRD, Q2, full A/A+/B, vLLM,
teachers, extra jobs, publication and pushes stay excluded. Stop after the local commit and ask for
explicit approval of the exact new registration before any private upload or launch.
