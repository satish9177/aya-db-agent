# SFT pilot: preregistration (written 2026-11-04, before any training or official-model run)

**Question.** Does assistant-only QLoRA SFT on 235 audited expert episodes make Tiny Aya Global
complete multistep database tasks on the frozen dev set?

## Data (frozen)

- `data/sft_pilot/episodes.jsonl`, sha256 `5cb48a813df0238f0cc196037d7e609249bb30698c6fc511759fecea2790eb6b`.
  235 episodes, 8 families (30 each, `ticket_top_customer` 25), 66 with one mistake + recovery
  (29 column, 37 table). Train databases only (seeds 101, 1101-1108); 23 candidates dropped for
  matching a dev (prompt, answer) pair; 81 share prompt text with dev on a different database.
- Every episode replays through AgentTune's rollout loop on a fresh database with byte-identical
  observations, a verified answer, and AgentTune-rendered prompts equal to the SFT prefixes.
- Dev: 40 tasks, sha256 `242e89cc07b8f7a79c8dcb04ad629109362b2350d59148f6a2a90f5f5f017433`.
  The sealed test (97 tasks) is not generated, read or used.

## Training (fixed)

`python -m ayadb.sft` with defaults: `CohereLabs/tiny-aya-global`, nf4 + double quant, fp16
compute (T4), LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down, lr 2e-4 cosine, warmup
5%, 2 epochs, batch 1 x grad-accum 8, paged AdamW 8-bit, gradient checkpointing, max length 4096
(no truncation; longest episode ~1.9k tokens), seed 42. Loss on assistant turns only (call JSON or
`ANSWER:` line + `<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`); mistakes, tool results, system and task
are context. 5% of training episodes held out for a validation loss curve; no dev data during
training, no checkpoint selection.

## Evaluation (fixed)

`python -m ayadb.smoke --engine hf`: frozen dev, `multistep-close` protocol (AgentTune fallback
format with the multistep prompt, results folded as `Tool result for {name}: {content}`, stop at
the `}]` call boundary), greedy, 8 tool rounds, 512 new tokens per turn, same tools and verifier.

| Run | Base precision | Adapter |
|---|---|---|
| base-nf4 | nf4 | none |
| sft-nf4-r0 | nf4 | pilot |
| sft-nf4-r1 | nf4 | pilot (A/A repeat) |
| base-fp16 | fp16 | none |
| sft-fp16 | fp16 | pilot |

**Primary comparison:** base-nf4 vs sft-nf4-r0 (identical quantization, protocol and decoding).
Secondary: base-fp16 vs sft-fp16. Reported for each: verified success (overall and per family),
malformed calls, hallucinated tool output, call-with-text turns, repeated calls, loops, answer
grounding (successful query; answer seen in a query result), exact McNemar on paired task outcomes,
and A/A flips between sft-nf4-r0 and r1.

## Decision rule (on sft-nf4-r0 successes out of 40)

- **>= 12**: proceed to broader evaluation.
- **4-11**: investigate whether the gains include real multistep behaviour (dependent lookups,
  recovery after errors) before any scaling.
- **<= 3**: recommend a pivot unless a concrete implementation fault is identified.

GRPO is not considered until an SFT policy completes a useful fraction of tasks and produces
mixed success/failure rollouts under sampling.

## Invalid-run rules

The run is invalid (reported, not interpreted) if: the dev set does not regenerate to the frozen
hash; kcheck fails; the nf4 sanity probe fails; training loss is non-finite; or any orchestration
check reports a tool-result turn the environment did not produce.
