# Competing baselines

## Tiny Aya Agent (`ferscm44/tiny-aya-agent`, code `github.com/fscm44xyz/aya-act`)

Reviewed 2026-11-04 at commit `c94a45a` (2026-07-23). Claim: Tiny Aya Global goes from 5/20 to
15/20 "verified tool-planning success" after QLoRA SFT on 80 trajectories distilled from Cohere's
North Mini Code and filtered by an automatic verifier.

**What it measures.** One assistant turn per task: the model writes a complete JSON plan
(a call list with `"$N.field"` references, an if/else object, or `{"error": "no tool available"}`).
No tool is executed and the model never sees a result; the verifier checks the plan's structure
(order, references, argument values and types, refusal, branches). Four types, 5 scenarios each:
`one_shot`, `data_chain`, `rejection`, `conditional`.

**What we checked** (`scenarios.py`, `train_80.jsonl`, the shipped baseline CSVs):

| Check | Finding |
|---|---|
| Train/eval overlap | All 20 evaluation scenarios (`one_shot_01-05`, `chain_01-05`, `reject_01-05`, `cond_01-05`) are among the 79 distinct scenarios in `train_80.jsonl`; the v1 20-example set is exactly those 20. There is no held-out split ("train on all, no eval split" in `train.py`). |
| Same pipeline before/after | Base: Ollama `tiny-aya-global-GGUF:Q4_K_M` at temperature 0 with Ollama's template. Adapter: transformers nf4 with the adapter repo's own chat template. Quantisation, engine and template all differ between the two numbers. |
| Tokenizer | Trained on transformers 5.x (the adapter's `tokenizer_config.json` says `CohereTokenizer`), so most likely on the non-canonical v5 ids described in `experiments/sft-pilot/AMENDMENT-1.md`. |
| Scale | 20 tasks; one seed; greedy; no confidence interval or paired test. |

**How to use it.** As a competing approach (verified distillation from a stronger teacher, with
dataset proportion as the main lever), not as evidence for what our model will reach. 15/20 is
in-sample plan-structure accuracy; our primary metric is out-of-sample, executed, multi-turn
verified success on a frozen dev set and a sealed test with three held-out families. The two are
not comparable on one scale. Its finding that refusal flips with the share of rejection examples
(25% over-refuses, 12.5% under-refuses) is a useful warning for any mixture we build later.

## Hinglish expense agent (`primce/tiny-aya-hinglish-lora`, Lexsi Track 3 entry)

Noted, not reviewed in depth: oracle-trajectory SFT on 184 tasks, 1/30 -> 29/30 on a pinned hard
split with paired McNemar, plus an injection safety split; GRPO (30 steps) gave no gain over SFT.
Same hackathon and track; useful as the reporting bar (pinned splits, paired tests, negative RL
result reported).
