# Selected-logits fix and G0-02 readiness (2026-11-07)

The requested free implementation is complete. No upload, CUDA/model-weight run, paid job, push or
publication occurred. The random tiny-model proof is CPU-only. G0-01's registration/results and all
legacy data/model/dependency pins and formats remain unchanged.

## Diagnosis and implementation

4072*262144*4 bytes is **3.9765625GiB**, matching the size of the failed allocation. G0-01's traceback
ends inside cross-entropy; this identifies the size of a dense fp32 vocabulary buffer, not uniquely
whether it was an upcast or an internal loss buffer. Both are avoided for untrained positions by
selecting hidden states **before** the vocabulary projection. For431 trained targets, one equivalent
fp32 buffer is **0.4208984375GiB**. This is not a prediction of total model/optimizer peak or GPU fit.

Inspected primary implementation: installed/pinned transformers 5.22.1, Cohere2ForCausalLM.forward,
slice_logits_to_keep, ForCausalLMLoss/fixed_cross_entropy and Trainer's compute_loss/training_step/
_get_num_items_in_batch. Boolean(B,T) selection flattens only selected hidden rows; the model still
applies logit_scale and its native causal loss accepts explicit shifted labels. No custom lm_head or
replacement cross-entropy is introduced.

`positionloss.py` adds opt-in **trained-positions-v1**, preserving default full-logits behavior.
Shift labels per row with a terminal -100, select positions whose next label is trained and pass the
same selected shift_labels to the native loss. Original labels remain intact. Trainer counts targets
over the whole accumulation window, excluding position 0; native sum/N loss is not divided again.
Checkpoint/settings identity includes the opted-in policy. All tokenizer/end-token/error/tool masks
and unique-target counts are unchanged; no sequence is shortened or extra target removed.

`gpumemory.py` and runner wiring persist current/peak allocation/reservation after load, each update,
completion and on OOM, without clearing the allocator. OOM is re-raised after logging; a secondary
measurement error cannot swallow it. Each process resets peaks before loading, then retains cumulative
peaks, so all phase logs together support the overall maximum. The launcher sets and records
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Inspected GenerationMixin already defaults supported models to logits_to_keep=1, including prefill.
The additive `LastPositionHFEngine` makes it explicit for new registered inference. CPU hooks verify
the base-class and PEFT generation routes. Runtime head pre-hooks reject full-prompt projection before
its large allocation; output hooks check one finite final-position vector. The old engine is preserved.

## CPU proof and validation

**114 tests passed**, 14 upstream torch JIT deprecation warnings, 40.86s:
`.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --basetemp runs/pytest-g0-02-release --tb=short`.
No model download, real GPU operation or external training service is part of these tests.

- Tiny random Cohere2: vocab 67, hidden 32, two layers/full+sliding attention, logit_scale .37, every
  production LoRA target module, rank 3 and nonzero B weights to exercise both A/B gradients.
- Native full/selected loss and selected logits match at rtol 1e-6/atol 1e-7. Every LoRA gradient matches
  at rtol 2e-5/atol 2e-7 across separated assistant spans, masked observations, unequal batch rows,
  dropout with identical RNG and gradient checkpointing on/off. Head hooks prove projection shrinks.
- Real HF Trainer training_step accumulation on unequal 6/7-target microbatches uses the same 13-token
  denominator for full and selected paths. Both match a single combined batch's loss/all LoRA gradients
  (gradient rtol 3e-5/atol 3e-7). The synthetic position 0 labels are intentionally excluded from counts.
- An all-masked microbatch contributes finite zero when its accumulation window has positive targets;
  a wholly empty loss window fails. A too-small denominator or conflicting mask/shift is rejected.
- Base-class and PEFT generation on a 64-token prefill project one vocabulary row, with explicit
  logits_to_keep=1; memory/OOM tests use a fake allocator. AgentTune spy tests verify load and update
  memory event wiring, checkpoint resume and the allocator environment without real HF clients.

The CPU proof does not establish NF4/fp16 GPU equivalence or memory fit; the new short GPU check is
preregistered to test that separately before longest-example training. Package versions are unchanged.

## Concrete proposal

[G0-02 registration](../experiments/sql-g0-02/REGISTRATION.md), SHA256
**`e86597148c4ccdeb645777fb85f4090ec0eb1156789172165b425242d70f2580`**:
same hf-user/t4-small/$3/7200s schedule and longest 4072/431 example, plus a first short native-loss
check on a whole 1219/106 example under NF4/fp16 autocast. No optimizer update in that check; rtol1e-4/
atol1e-4 and actual full/selected projection-count assertions. Then train 2, resume to 3, and the same
nine inference episodes, with exact checkpoint identity/reload, memory events and guarded scoring.

The new assets are **16 files / 143,206,958 bytes** plus source: the original 14 files unchanged, one
short record and its database (five SQaLe databases total). No new dataset, BIRD, Q2 or base weights.
Short and long encodings are freshly checked, the source/assets plan is validated and five native
references pass before any model call. Inputs/outputs are bounded at 512 MiB/1GiB, storage at 30 days.

G0-02 has no upload/launch receipt and is **not approved**. The 500-stage draft remains blocked until
successful GPU compatibility/throughput exists. Commit this free work locally, then stop and ask before
uploading or launching the exact proposed registration. G0-01 must never be restarted or refreshed.
