# SFT pilot: amendment 1 (2026-11-04, before any official-weight run)

Written before the official `CohereLabs/tiny-aya-global` weights were accessible to this project,
so no output of the official model has been seen. Everything in `PREREGISTRATION.md` stands except
the points below.

## 1. Tokenizer loading (implementation fault, fixed)

On transformers 5.x, `AutoTokenizer` loads Tiny Aya as `CohereTokenizer`, whose `__init__` replaces
the pre-tokenizer stored in `tokenizer.json` (digits split in groups of three, plus a word regex)
with a hard-coded `Digits(individual_digits=True) + ByteLevel`. Text still decodes back to itself,
but the ids are not the ones the model was trained on. Upstream fix pending:
huggingface/transformers#49192; diagnosis in unslothai/unsloth#12130.

Measured with transformers 5.22.1 on the original `tokenizer.json` (public mirror
`1-800-LLMs/tiny-aya-global`, sha256 `2227ea9c52e8afb3f98bfed2679008b275f2664de69dfde174b374389eb0225d`):

| Texts (all 278 a run tokenizes: 3 probes, 235 SFT episodes, 40 dev prompts) | Tokens |
|---|---|
| `tokenizer.json` | 370,270 |
| transformers 5.22.1 `AutoTokenizer` | 415,901 (+12.3%; 278/278 texts differ) |
| `engine.official_tokenizer` (fix) | 370,270 (278/278 identical ids) |

llama.cpp (the Ollama baseline in the README) agrees with `tokenizer.json` on all three probes, so
the GGUF baseline was tokenized correctly and is unaffected.

Fix: `engine.official_tokenizer` loads `tokenizer.json` as-is (`TokenizersBackend`). Training,
evaluation, the sanity probe and kcheck all use it. Special tokens (BOS 2, `<|END_OF_TURN_TOKEN|>`
6 = eos, `<|END_RESPONSE|>` 261001, PAD 0), the rendered chat template and the assistant end
sequence are unchanged. kcheck now fails if the ids differ from the official `tokenizer.json` on
any probe, SFT episode or dev prompt, and records the official file's sha256.

## 2. Rollout protocol (clarified, not changed)

`multistep-close` as preregistered. The HF engine ends a turn on `<|END_RESPONSE|>`,
`<|END_OF_TURN_TOKEN|>` or the tokenizer eos, and on the `}]` call boundary. The end tokens are
passed explicitly because the model's `generation_config.eos_token_id` may not be the turn end
(it is `<EOS_TOKEN>` = 3 in the mirror); kcheck records the official value.

## 3. Compute host

Kaggle T4x2 is replaced by Hugging Face Jobs `t4-small` (one T4, same GPU, fp16 compute, nf4
base), one job per stage or evaluation instead of two GPUs in one session (`scripts/hf_job.py`).
`scripts/modal_job.py` runs the identical steps on Modal T4 and is kept as a fallback; each
stage's `status.json` records the host, GPU, job id and commit. Library versions are pinned to the
local environment: torch 2.11.0 and torchvision 0.26.0 (cu126), transformers 5.22.1, tokenizers 0.23.3,
huggingface_hub 1.33.0, peft 0.21.2, trl 1.7.1, accelerate 1.15.0, bitsandbytes 0.50.2, agenttune 1.1.3,
auditkit 1.1.3 (torchvision pinned after an unpinned one from PyPI broke the imports in T4 job `6aeb57c2`).
Order and gates are unchanged: check (kcheck on a CPU job, then fp16/nf4 sanity on the T4) -> baseline (base-nf4, base-fp16)
-> smoke (2 training steps, then 2 dev tasks with that adapter) -> pilot (training, then
sft-nf4-r0, sft-nf4-r1, sft-fp16). A stage refuses to start on a dirty git tree, and the launcher
refuses episodes whose sha256 differs from the preregistered one (`data/` is not in git).

## 4. Added diagnostics

These do not change the primary metric or the decision rule.

- `had_error` / `recovered`: genuine error recovery. The episode hit an error (bad SQL, unknown
  table or column, bad arguments), a successful `run_sql` came after its last error, and the answer
  is verified. Reported as recovered episodes / episodes with an error.
- Mean tool calls on successful episodes.

The report gives, overall and per task family: verified success; malformed calls (bad-argument
calls, unknown tools, malformed final turn, call-with-text turns); fabricated observations
(fake-output turns, episodes with hallucinated tool output); loops and repeated calls; tool calls,
turns and completion tokens per task; and recovery.

## 4b. First check on the official files (2026-11-04, CPU job `6aeb5422`, no model output)

- The official `tokenizer.json` has the mirror's sha256 (`2227ea9c...`), and `official_tokenizer` ids
  equal it on 278/278 texts. On the official repo, transformers 5.22.1 `AutoTokenizer` resolves to
  `TokenizersBackend` and also matches (0 texts differ): the v5 fault hit the mirror's tokenizer
  config, not the official one. The explicit loader stays. End tokens 6 and 261001;
  `generation_config.eos_token_id` is 3 (`<EOS_TOKEN>`), as anticipated in section 2.
- The dev-set gate failed on line endings only. `FROZEN_DEV_SHA256` (`242e89cc...`) was taken on
  Windows, where the task file is written with CRLF; the same 40 records written on Linux hash to
  `7798e6fe...`, and the two files are byte-identical once line endings are normalised. The gate
  (`tasks.is_frozen_dev`, used by kcheck and every dev evaluation) now compares with line endings
  normalised; the preregistered hash is unchanged and a test pins both serialisations.
- kcheck's template diagnostic compared a dict to a string; the official `default` template is
  identical to the GGUF template used by the earlier baselines.

## 5. Not changed

The data (`episodes.jsonl` sha256 `5cb48a81...`), the frozen dev set (`242e89cc...`), the sealed
test, the training configuration, the comparisons and the decision rule. No SynSQL, BIRD or other
external data, no GRPO and no dataset change before the first SFT result.
