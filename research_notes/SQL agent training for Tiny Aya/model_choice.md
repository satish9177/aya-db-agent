# Model choice: which allowed Cohere model to fine-tune into a multi-turn SQL tool-calling agent (Lexsi hackathon, Track 3)

Scope: Tiny Aya Base / Global / Fire / Water / Earth, Aya Expanse 8B, Aya Expanse 32B. Research date 2026-11-04. All seven official repos are gated (CC-BY-NC-4.0), so configs and chat templates were read from public, ungated mirrors (`1-800-LLMs/tiny-aya-global`, `adamo1139/aya-expanse-8b-ungated`). These mirrors date from March 2026 and October 2024. The official Tiny Aya repos were last updated 2026-09-30, so the official chat template may have changed since the mirror was made (see Gaps).

---

## Q1. What exactly are Tiny Aya Base / Global / Fire / Water / Earth?

### Takeaway
All five are the same 3.35B dense Cohere2 decoder: 36 layers, 8K context, 262k vocabulary, tied embeddings, pretrained on 6T tokens including code. Base is the raw pretrained model. Global is SFT on all regions plus a deliberately minimal preference-tuning step. Fire, Water and Earth are regional SFT checkpoints merged with Global. The regional variants only help with translation into their regional languages, which is irrelevant to an English SQL agent, so the choice within the family is Global (or Base).

### Cited Findings
- Family: Tiny Aya Base (pretrained), Tiny Aya Global (instruction-tuned, "globally balanced"), and three region-specialised models. Earth is "strongest for languages across Africa and West Asia", Fire is "strongest for South Asian languages", and Water is "strongest for the Asia-Pacific and Europe regions" — [Tiny Aya tech report, arXiv 2603.11510](https://arxiv.org/abs/2603.11510)
- Release timing: the report was published on arXiv on 2026-03-12/13 — [arXiv 2603.11510](https://arxiv.org/html/2603.11510). The official GGUF repos were created about 2026-02-17 — [CohereLabs/tiny-aya-global-GGUF](https://huggingface.co/CohereLabs/tiny-aya-global-GGUF). All five safetensors repos were last updated 2026-09-30 — [Hub repo details, tiny-aya-global](https://hf.co/CohereLabs/tiny-aya-global)
- Architecture: dense decoder-only, following Command A. It uses parallel Transformer blocks and interleaves sliding-window and full attention 3:1 (RoPE on sliding layers, NoPE on full layers). Other features are SwiGLU, no biases, and GQA. Dimensions: embedding 2048, 36 layers, FFN 11008, 16 heads, 4 KV heads, sliding window 4096, vocab 262k. Embedding params are 0.5B, non-embedding 2.8B, 3.35B in total. Input context is 8192 tokens and output context 8192 tokens — [Tiny Aya report, Table 2](https://arxiv.org/html/2603.11510)
- config.json fields: `Cohere2ForCausalLM`, `model_type: cohere2`, `head_dim 128`, `use_embedding_sharing: true` (tied input/output embeddings), `use_parallel_block: true`, `logit_scale 1.0`, `rope_theta 50000`, `torch_dtype: bfloat16`, `vocab_size 262144`, `sliding_window 4096`. `max_position_embeddings` is 500000, even though the model is trained and documented for 8K — [config.json, ungated mirror](https://huggingface.co/1-800-LLMs/tiny-aya-global/blob/main/config.json)
- All five variants report 3349.2M parameters and the `cohere2` architecture. Global, Fire, Water and Earth carry the `conversational` tag and Base does not. All four derived variants are tagged `base_model:finetune:CohereLabs/tiny-aya-base` — [HF Hub repo details](https://hf.co/CohereLabs/tiny-aya-base)
- Pretraining: 6T tokens with a WSD schedule on 256 H100s using FP8/BF16 mixed precision. The data covers 70 languages plus "programming languages datasets". A cooldown (mid-training) phase upsamples high-quality data and adds instruction-style data — [Tiny Aya report §2.2, §3.1](https://arxiv.org/html/2603.11510)
- Post-training: languages are grouped into clusters, and "English and code are shared across all clusters". Every model is SFT'd for 3 epochs (peak LR 2.5e-5, global batch 32) on 16 H100s in under 24 hours. "We apply a minimal preference tuning phase on top of SFT for the Tiny Aya Global model ... We deliberately keep this phase minimal to prevent catastrophic forgetting and to facilitate users and researchers in adapting the model to downstream tasks" — [Tiny Aya report §2.3, §3.2](https://arxiv.org/html/2603.11510)
- Regional models are built by merging each region-SFT checkpoint with the Global checkpoint, with the merge operator selected by SimMerge. The goal is to keep Global's instruction-following and safety while retaining regional gains — [Tiny Aya report §3.3](https://arxiv.org/html/2603.11510)
- "Region-specificity matters most for translation". Regional variants beat Global on English-to-target translation by +1.7 chrF (Africa) up to +5.5 chrF (South Asia). Fire and Water do not support African languages — [Tiny Aya report §5.2](https://arxiv.org/html/2603.11510)
- The report attributes Fire's greater sensitivity to English prompting to "the high percentage of English data present in the training data composition for Tiny Aya Fire" — [Tiny Aya report §5.4.2](https://arxiv.org/html/2603.11510)
- Model card limitations for Global: "weaker on chain-of-thought tasks such as multilingual math (MGSM)", and it "works best with clear, simple instructions". No tool use is mentioned. The listed serving stacks are vLLM, SGLang, Docker Model Runner and Transformers — [CohereLabs/tiny-aya-global model card](https://huggingface.co/CohereLabs/tiny-aya-global)
- Chat template (mirror): one `default` template only, with no `tools` handling and no `tool` role. It enforces strict user/assistant alternation and raises an exception otherwise. Before any user system prompt it injects a long fixed preamble: contextual safety mode, "information cutoff date is June 2024", the full language list, and style rules. The user system prompt is placed under "# Developer Preamble". Assistant turns are wrapped as `<|CHATBOT_TOKEN|><|START_RESPONSE|>...<|END_RESPONSE|><|END_OF_TURN_TOKEN|>` — [tokenizer_config.json, ungated mirror](https://huggingface.co/1-800-LLMs/tiny-aya-global/blob/main/tokenizer_config.json)
- New, but not on the allowed list: Cohere Labs published `CohereLabs/tiny-aya-l2-thinker` and `CohereLabs/tiny-aya-en-thinker` (gated, updated 2026-10-30) — [HF Hub search](https://huggingface.co/CohereLabs/tiny-aya-l2-thinker)

### Inferences
- The variants differ only in which regional SFT data was merged in. Code and English are in every cluster, so nothing in the documentation suggests Fire, Water or Earth would be better at SQL or tool use than Global. Global is also the only variant described as receiving preference tuning, and that tuning was kept minimal specifically to ease downstream adaptation. That makes Global the natural starting point.
- Base would have to learn the chat format, turn-ending and instruction-following from scratch. With about 10 days and a small dataset, that is wasted capacity, so starting from Base only makes sense if the instruction-tuned template or preamble proves actively harmful.
- Fire's reportedly higher share of English data is a weak signal that it might do marginally better on English-only tasks. Testing it costs one zero-shot evaluation run, since the architecture is identical, but there is no published evidence that it helps.
- The template's strict alternation and lack of a `tool` role explain why tool results currently have to be sent as user turns. Each call also pays for the fixed default preamble, which eats into the 8K budget. Because the model is being fine-tuned anyway, a custom, shorter template (same special tokens, no default preamble, explicit tool-result turns) is a legitimate design choice.

### Gaps
- Could not read the official (gated) Tiny Aya repos. Whether the 2026-09-30 update changed the chat template (for example, added tool support) is unverified. Check `tokenizer_config.json` or `chat_template.jinja` in the local HF cache.
- The exact share of code in the pretraining and post-training mixtures is not published, and the report gives no code, SQL or tool-use evaluations.

---

## Q2. What are Aya Expanse 8B and 32B, and does their chat template support tool calling?

### Takeaway
Aya Expanse (released 2024-10-24) is the older Command-R-architecture (`CohereForCausalLM`) family. The 8B has 8K context and the 32B has 128K. Both were post-trained for multilingual chat through data arbitrage, iterative DPO and merging. Unlike Tiny Aya, the tokenizer ships a named Command-R-style `tool_use` template, but that template uses Cohere's legacy format ("Action: ```json [...]```" with a `directly-answer` tool and `parameter_definitions`) and has no tool-result role. No tool-use training or evaluation is documented.

### Cited Findings
- The Aya Expanse blog post was published 2024-10-24. Post-training used "data arbitrage" synthetic data from a pool of teacher models selected by reward model, offline DPO, then three iterations of online DPO, then weighted-average merging of language-cluster checkpoints. Weights are released "for the research community" — [HF blog: Aya Expanse](https://huggingface.co/blog/aya-expanse)
- Aya Expanse 8B card: 8B parameters, 8K context, 23 languages, an "optimized transformer architecture", CC-BY-NC plus Acceptable Use Policy, gated. Example sampling uses temperature 0.3. Tool use and function calling are not mentioned — [CohereLabs/aya-expanse-8b](https://huggingface.co/CohereLabs/aya-expanse-8b)
- Aya Expanse 32B card: "Context length: 128K", about 32B parameters, F16 tensors, CC-BY-NC-4.0. No mention of tool use — [CohereLabs/aya-expanse-32b](https://huggingface.co/CohereLabs/aya-expanse-32b)
- Hub metadata: 8B is 8028.0M params with architecture `cohere`. 32B is 32296.5M params with architecture `cohere`. Tiny Aya and Aya Expanse 32B have a live "cohere" inference provider on HF, and 8B has none — [HF Hub repo details](https://hf.co/CohereLabs/aya-expanse-8b)
- 8B config.json: `CohereForCausalLM`, hidden 4096, 32 layers, 32 heads, 8 KV heads, FFN 14336, vocab 256000, `logit_scale 0.125`, `max_position_embeddings 8192`, `rope_theta 10000`, `torch_dtype: float16` — [config.json, ungated mirror](https://huggingface.co/adamo1139/aya-expanse-8b-ungated/blob/main/config.json)
- 8B weights total about 16.06 GB in fp16 (4 shards: 4.92 + 4.92 + 5.00 + 1.22 GB) — [file listing, ungated mirror](https://huggingface.co/adamo1139/aya-expanse-8b-ungated/tree/main). Tiny Aya Global weights total about 6.70 GB (4.99 + 1.71 GB) — [file listing, ungated mirror](https://huggingface.co/1-800-LLMs/tiny-aya-global/tree/main)
- Aya Expanse chat templates (mirror): three named templates, `default`, `tool_use` and `rag`.
  - `default` enforces strict user/assistant alternation and has no tools.
  - `tool_use` renders each tool as a Python function stub from `tool.name`, `tool.description` and `tool.parameter_definitions` (Cohere v1 format, not OpenAI JSON schema). It then appends a system turn instructing: "Write 'Action:' followed by a json-formatted list of actions ... You should use the `directly-answer` tool if calling the other tools is unnecessary", with `[{"tool_name": ..., "parameters": {...}}]`. Its message loop renders only `user`, `system` and `assistant` roles, so tool outputs are not rendered unless re-sent under one of those roles.
  - [tokenizer_config.json, ungated mirror](https://huggingface.co/adamo1139/aya-expanse-8b-ungated/blob/main/tokenizer_config.json)
- The Aya Expanse report evaluates only multilingual win rates, discriminative tasks, Global-MMLU, MGSM and translation. There is no tool, code or agent evaluation — [Aya Expanse report, arXiv 2412.04261](https://arxiv.org/abs/2412.04261)

### Inferences
- The `tool_use` template appears to be inherited from the Command R tokenizer. Neither the card nor the paper claims tool-use training for Aya Expanse, so the template alone is not evidence of tool-use skill. The project's own measurement (Aya Expanse 8B Q4_K_M at 12.5% vs Tiny Aya Global at 0% greedy) is the most relevant evidence of a stronger prior.
- In practice both families need the same treatment: a custom tool protocol taught by SFT. Aya Expanse's tool template is not a decisive advantage.
- Aya Expanse 32B's 128K context would matter only for very large schemas, which a 3-tool SQLite agent does not need.

### Gaps
- No official Aya Expanse 8B/32B tool-use benchmark (BFCL or similar) was found.
- The 32B config was not read directly. The architecture is assumed identical in kind (`cohere`) based on Hub metadata.

---

## Q3. Published benchmarks relevant to this task (code, SQL, function calling, math, instruction following)

### Takeaway
No SQL, Spider/BIRD, BFCL, tau-bench or code benchmark has been published by Cohere for any of the seven models. The closest proxies are math (MGSM) and general knowledge. Tiny Aya Global trails same-size Qwen3-4B, Qwen3.5-4B and Gemma3-4B on math and MMLU by roughly 2-10 points. Aya Expanse 8B was best in its 2024 8B class on MGSM, and 32B is stronger again.

### Cited Findings
- Tiny Aya Global vs same-size peers (Table 5):

  | Model | mArenaHard (66 langs) | GlobalMGSM (35 langs) |
  |---|---|---|
  | Tiny Aya Global | 67.4 | 52.8 |
  | Gemma3-4B | 65.8 | 55.4 |
  | Qwen3-4B | 70.1 | 60.9 |
  | Qwen3.5-4B | 76.7 | 62.5 |
  | Ministral-3-3B | 63.7 | 49.6 |
  | SmolLM3-3B | 31.8 | 22.6 |

  The report notes that Global "lags behind the competitors on average by 2–10 points" on math — [Tiny Aya report §5.2](https://arxiv.org/html/2603.11510)
- Discriminative tasks (Table 4), Global MMLU (42 languages): Tiny Aya Global 44.9, Gemma3-4B 45.3, Qwen3-4B 49.3, Qwen3.5-4B 54.8, Ministral-3-3B 46.8, SmolLM3-3B 37.2 — [Tiny Aya report §5.1](https://arxiv.org/html/2603.11510)
- Tiny Aya "performs strongly in open-ended tasks, even more in non-technical domains (mDolly) than technical domains (mArenaHard)" — [Tiny Aya report §5.2](https://arxiv.org/html/2603.11510)
- Tiny Aya Global scores 28.28 on GPQA Diamond. This is the only numeric score in the card text — [tiny-aya-global card](https://huggingface.co/CohereLabs/tiny-aya-global)
- Aya Expanse 8B vs the 8B class (Table 2), with Global-MMLU 5-shot and MGSM 5-shot on 7 languages:

  | Model | Global-MMLU | MGSM |
  |---|---|---|
  | Aya Expanse 8B | 53.7 | 67.0 |
  | Llama-3.1-8B | 54.5 | 63.0 |
  | Ministral-8B | 52.3 | 55.1 |
  | Gemma-2-9B | 62.6 | 59.6 |
  | Qwen-2.5-7B | 62.8 | 55.1 |

  Aya-23-8B scored 48.1 and 36.5 — [Aya Expanse report, arXiv 2412.04261](https://arxiv.org/abs/2412.04261)
- Aya Expanse 32B (Table 3): Global-MMLU 66.9 and MGSM 73.4, against Qwen-2.5-32B (75.0 / 59.7), Gemma-2-27B (68.3 / 61.5) and Llama-3.1-70B (75.0 / 78.0) — [Aya Expanse report](https://arxiv.org/abs/2412.04261)
- Aya Expanse 8B scores 33.74 on MMLU-Pro (model card) — [aya-expanse-8b card](https://huggingface.co/CohereLabs/aya-expanse-8b)
- Open LLM Leaderboard v2 figures for aya-expanse-8b, as reported by an aggregator: IFEval 63.6, BBH 28.5, MATH Lvl5 8.6, GPQA 7.0, MuSR 4.4, MMLU-Pro 22.3 (normalised scores) — [benchmarklist.com aggregator](https://benchmarklist.com/models/cohereforai-aya-expanse-8b/). Raw details exist in [open-llm-leaderboard/CohereForAI__aya-expanse-8b-details](https://huggingface.co/datasets/open-llm-leaderboard/CohereForAI__aya-expanse-8b-details), but they were not cross-checked.

### Inferences
- The 4B-class peers (Qwen3-4B, Qwen3.5-4B) are stronger reasoners than Tiny Aya Global, and Tiny Aya's own report says it is weaker on technical tasks. That fits the 0% untuned agent baseline. Even so, the dominant failure mode reported in this project is protocol (not ending the turn, fabricating tool results), not reasoning, and SFT fixes that cheaply.
- Aya Expanse 8B has the better reasoning prior (MGSM 67.0 vs GlobalMGSM 52.8; the benchmarks differ in language set, so the comparison is only directional) and the better measured agent baseline (12.5% vs 0%). This is the main argument for it.
- MGSM-style math correlates only loosely with SQL aggregation and grounding. Treat all of these as weak proxies.

### Gaps
- No Spider, BIRD, BFCL, tau-bench, HumanEval or MBPP numbers were found for any of the seven models, from Cohere or third parties.
- The Open LLM Leaderboard numbers come from an aggregator and were not verified against the raw details dataset (gated).

---

## Q4. Licences and gating: can fine-tuned adapters or models be published on HF?

### Takeaway
All seven models are CC-BY-NC-4.0 plus Cohere Labs' Acceptable Use Policy, and gated behind a click-through. Non-commercial research publication of derivatives is common practice: several LoRA adapters, full fine-tunes and even re-hosted GGUFs of Tiny Aya are publicly listed on the Hub under the NC licence. No licence reason favours one model over another.

### Cited Findings
- Tiny Aya Global: CC-BY-NC 4.0 plus Acceptable Use Policy. Commercial use requires contacting Cohere sales, and the gate requires sharing HF contact info. The card's terms text oddly mentions a "111 billion parameter model", which looks like boilerplate inconsistency — [tiny-aya-global card](https://huggingface.co/CohereLabs/tiny-aya-global)
- All seven are tagged `license:cc-by-nc-4.0` and gated — [HF Hub repo details](https://hf.co/CohereLabs/aya-expanse-32b)
- Public derivatives exist:
  - LoRA adapter `Aditya0427/ToolBoundary-tiny-aya-lora` with `license_name: c4ai-cc-by-nc`. It states: "The adapter requires access to the original gated base model and its license/AUP. No base weights or tokenizer are redistributed here." — [ToolBoundary adapter card](https://huggingface.co/Aditya0427/ToolBoundary-tiny-aya-lora)
  - Full fine-tunes `qgallouedec/tiny-aya-global-tool-calling-SFT` and `RamAnanth1/tiny-aya-hermes-tool-calling` are public — [qgallouedec card](https://huggingface.co/qgallouedec/tiny-aya-global-tool-calling-SFT), [RamAnanth1 card](https://huggingface.co/RamAnanth1/tiny-aya-hermes-tool-calling)
  - `Bronsn/tiny-aya-global-tools-GGUF` redistributes quantised weights, labelled "CC-BY-NC-4.0 (inherited from Tiny Aya)" — [Bronsn GGUF card](https://huggingface.co/Bronsn/tiny-aya-global-tools-GGUF)

### Inferences
- The safest publication pattern is a LoRA adapter (or a merged model) labelled CC-BY-NC-4.0, with a link to the gated base and the AUP noted, as the ToolBoundary card does. This applies equally to all seven models.

### Gaps
- Did not check the hackathon's own rules on publishing requirements (out of scope by instruction). Cohere has not given explicit guidance on whether merged full-weight derivatives must themselves be gated.

---

## Q5. Fine-tuning practicality on Kaggle 2x T4 (16 GB, no bf16, no FA2): SFT and GRPO memory; framework support

### Takeaway
Tiny Aya (3.35B, 6.7 GB fp16) is the only allowed model where both LoRA SFT and on-policy RL (GRPO: generation plus training) fit comfortably on 2x T4 without quantisation. There is already public evidence of a Tiny Aya QLoRA run on a free Colab T4. Aya Expanse 8B (16.06 GB fp16) does not fit on a single T4 at fp16. It is workable for 4-bit QLoRA SFT, but GRPO is very tight. Aya Expanse 32B is not realistically trainable on this hardware or budget. Both architectures are supported by Transformers, TRL and vLLM (with LoRA). Unsloth support for `cohere2` could not be confirmed.

### Cited Findings
- vLLM lists `CohereForCausalLM` and `Cohere2ForCausalLM` (Command-R and Command-A family) with LoRA and pipeline-parallel support — [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html). The Tiny Aya card lists vLLM and SGLang as supported serving stacks — [tiny-aya-global card](https://huggingface.co/CohereLabs/tiny-aya-global)
- Older library versions fail on `cohere2`. Users report "model type cohere2" errors in vLLM, fixed by upgrading Transformers — [Command R7B discussion #5](https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024/discussions/5). MLX also had a "Model type cohere2 not supported" error — [mlx-community discussion](https://huggingface.co/mlx-community/c4ai-command-r7b-12-2024-4bit/discussions/2). Transformers documents `Cohere2` (at least from v4.51.3) — [Transformers Cohere2 docs](https://huggingface.co/docs/transformers/model_doc/cohere2)
- TRL works with Tiny Aya. TRL maintainer Quentin Gallouédec published an SFT of Tiny Aya Global on `bebechien/SimpleToolCalling` (TRL 0.28.0.dev0, Transformers 5.2.0.dev0, PyTorch 2.10.0) — [qgallouedec/tiny-aya-global-tool-calling-SFT](https://huggingface.co/qgallouedec/tiny-aya-global-tool-calling-SFT)
- T4 evidence for Tiny Aya: the ToolBoundary adapter was trained with "NF4 base, FP16 compute, FP32 trainable adapters; LoRA rank8 on q_proj/v_proj. Free Colab Tesla T4", using 240 synthetic demos for 60 completion-only steps in 348.9 s. Training went through **AgentTune (Lexsi Labs)** via its `trainer_factory` hook with TRL — [ToolBoundary card](https://huggingface.co/Aditya0427/ToolBoundary-tiny-aya-lora)
- Tiny Aya weights are stored in bfloat16 and the model was pretrained with FP8/BF16 compute, keeping softmax and layer norms in FP32 — [config.json mirror](https://huggingface.co/1-800-LLMs/tiny-aya-global/blob/main/config.json); [Tiny Aya report §3.1](https://arxiv.org/html/2603.11510). Aya Expanse 8B weights are stored in float16 (`torch_dtype: float16`) — [config.json mirror](https://huggingface.co/adamo1139/aya-expanse-8b-ungated/blob/main/config.json)
- 4-bit quantisations exist for the larger models: `Orion-zhen/aya-expanse-8b-AWQ`, `Orion-zhen/aya-expanse-32b-AWQ` and `kevinbazira/aya-expanse-32b-gptq-4bit` — [HF Hub search](https://huggingface.co/Orion-zhen/aya-expanse-8b-AWQ), [GPTQ 32B](https://huggingface.co/kevinbazira/aya-expanse-32b-gptq-4bit)
- Unsloth: searches turned up no Unsloth notebook, documentation entry or issue for Tiny Aya, Aya Expanse or `cohere2` — [web search results, e.g. docs.unsloth.ai](https://docs.unsloth.ai)

### Inferences (memory arithmetic from the cited configs; not measured)
- **Weights:**
  - Tiny Aya is 6.7 GB in fp16 and roughly 2.5-3 GB in NF4. That assumes 2.8B non-embedding parameters at about 0.5 bytes each, plus the 0.54B tied embedding (262144 x 2048) left in fp16 at about 1.07 GB.
  - Aya Expanse 8B is 16.06 GB in fp16, which exceeds one T4. In NF4 it is about 5.5-6 GB, assuming the 1.05B-parameter embedding (256000 x 4096) stays in fp16 at about 2.1 GB.
  - Aya Expanse 32B is about 64.6 GB in fp16 and about 18-20 GB in 4-bit. That is more than one T4 even quantised, so it would need both T4s with naive model parallelism, tiny batches and short sequences.
- **KV cache per token (fp16):**
  - Tiny Aya: 36 layers x 4 KV heads x 128 head_dim x 2 (K,V) x 2 bytes = about 72 KB, so an 8K-token trajectory needs about 0.6 GB.
  - Aya Expanse 8B: 32 x 8 x 128 x 2 x 2 = about 128 KB, so 8K tokens needs about 1.05 GB.
- **Logits are the hidden memory hog for both families** because of the huge vocabularies (262k and 256k). A 4,096-token sequence produces a 262144 x 4096 logit matrix: about 2.1 GB in fp16, or about 4.3 GB if upcast to fp32 for the loss, plus gradients. Mitigations are completion-only loss with modest max length (about 2-4K), batch size 1 with gradient accumulation, and gradient checkpointing. A chunked or fused cross-entropy (Liger, Unsloth-style) would help if it supports these architectures (unverified). Do not put LoRA on the tied embedding.
- **SFT:**
  - Tiny Aya fp16 LoRA fits on one T4 with about 9 GB to spare, or NF4 if longer sequences are needed. Both T4s can run DDP for throughput.
  - Aya Expanse 8B needs NF4 QLoRA (about 6 GB of weights plus activations and logits) on one T4. It will be slower per token because it has about 2.4x the parameters.
  - 32B is infeasible within the time budget.
- **GRPO (generation plus training):**
  - Tiny Aya can run a vLLM or HF-generate rollout worker on GPU1 (6.7 GB weights plus KV cache for about 8-16 parallel 8K rollouts) and the LoRA trainer on GPU0. This is the only allowed model where on-policy RL with verifiable SQL rewards is realistic on Kaggle.
  - For Aya Expanse 8B, fp16 generation alone needs both T4s (tensor parallel), leaving no GPU for training. It would need 4-bit generation plus a 4-bit trainer with adapter syncing, which is fragile on T4.
  - For 32B, GRPO is out of the question on T4s, and also on $20-30 of HF credits for any meaningful number of steps.
- **fp16 on T4:** Tiny Aya is a bf16-native model run in fp16 on T4, so activation overflow is a theoretical risk (Gemma and T5-style models are known for it), but no Tiny-Aya-specific report was found. The ToolBoundary T4 run (FP16 compute) completed successfully. Aya Expanse ships natively in fp16, which lowers this risk. Mitigations: monitor for NaN/inf losses, keep LoRA parameters in fp32, use fp16 AMP with loss scaling, and fall back to an HF Jobs GPU with bf16 if instability appears.
- **Sequence-length limit:** Tiny Aya's 4096 sliding window and 8K trained context, plus about 300+ tokens of default preamble per turn, set a hard budget for multi-turn trajectories that include schema dumps and SQL results. Truncating tool outputs (row caps) in the environment matters.

### Gaps
- Unsloth support for `Cohere2ForCausalLM` (Tiny Aya) and `CohereForCausalLM` (Aya Expanse) is unverified. If it matters, check Unsloth's GitHub `models/` directory or run a smoke test. Liger-kernel support for these architectures is also unverified.
- Whether current vLLM releases still run on T4 (compute capability 7.5) with Cohere2's sliding-window plus NoPE layers was not verified. A 5-minute smoke test on Kaggle is recommended before committing to vLLM-based GRPO.
- No measured tokens/sec or peak-memory numbers for Tiny Aya or Aya Expanse training on T4 were found. The figures above are arithmetic estimates.

---

## Q6. Existing community fine-tunes for tool use, SQL or agents

### Takeaway
Several Tiny Aya Global tool-calling fine-tunes exist, but only one, a tiny LoRA trained on a T4 with Lexsi's AgentTune, reports before/after numbers. It shows large gains in format validity from very little data. No SQL or database-agent fine-tune of any allowed model was found, and no tool-use fine-tunes of Aya Expanse were found.

### Cited Findings
- `Aditya0427/ToolBoundary-tiny-aya-lora` (Tiny Aya Global, LoRA r8, 240 demos, 60 steps, T4), on a strict-JSON synthetic benchmark of 24 cases per split:
  - Valid raw JSON actions: 7 to 24 (dev) and 6 to 24 (held-out).
  - Exact correct actions: 2 to 20 (dev) and 2 to 19 (held-out).
  - Unauthorized execute attempts rose from 0 to 3 on held-out.
  - The author cautions that the benchmark uses shared template families, single-step English only, so it is not independently blinded.
  - [ToolBoundary card](https://huggingface.co/Aditya0427/ToolBoundary-tiny-aya-lora)
- `qgallouedec/tiny-aya-global-tool-calling-SFT`: TRL SFT on `bebechien/SimpleToolCalling`. No evaluation reported — [card](https://huggingface.co/qgallouedec/tiny-aya-global-tool-calling-SFT)
- `RamAnanth1/tiny-aya-hermes-tool-calling`: TRL SFT on `NousResearch/hermes-function-calling-v1`. No evaluation reported — [card](https://huggingface.co/RamAnanth1/tiny-aya-hermes-tool-calling)
- `Bronsn/tiny-aya-{global,earth,fire,water}-tools-GGUF` ("Tiny Facade", Crane AI Labs / Cohere Labs Community): a template fix rather than a fine-tune. It reports that "The official Tiny Aya GGUFs on Ollama ship with the wrong chat template (Command-R's template instead of Tiny Aya's own)", which causes `<|END_OF_TURN_TOKEN|>`/`<|END_RESPONSE|>` leakage and broken turn termination. The fix registers both tokens as stop tokens and injects tools into the system prompt inside `<tools>` tags with `<tool_call>` outputs. No accuracy numbers are given — [Bronsn card](https://huggingface.co/Bronsn/tiny-aya-global-tools-GGUF)
- Hub search for "aya-expanse-8b sql" returned no models. Search for "aya tool" returned only Tiny Aya derivatives — [HF Hub search](https://huggingface.co/models?search=aya%20tool)

### Inferences
- The project's observed "doesn't end turn" failure may be partly a stop-token or template problem rather than purely a model-skill problem. The Bronsn finding that the end tokens are `<|END_RESPONSE|>` and `<|END_OF_TURN_TOKEN|>`, and that GGUF/Ollama builds used the wrong template, suggests checking that inference registers both as stop strings, especially in the GGUF path. The Aya Expanse 8B baseline was also run as a Q4_K_M GGUF, so the same check applies there.
- Tiny Aya has the most community and TRL-maintainer activity for tool calling, and there is Lexsi-tooling (AgentTune) precedent for it. Taken with the size argument, this lowers integration risk for the hackathon.
- A database agent for either family would be novel on the Hub. Novelty does not favour one model over the other.

### Gaps
- No public evaluation of Tiny Aya or Aya Expanse on BFCL, Spider/BIRD or multi-turn agent benchmarks after fine-tuning was found.

---

## Q7. Inference cost and speed (for an ROI comparison against a closed API)

### Takeaway
Tiny Aya is designed for on-device inference: Q4_K_M is 2.14 GB, runs at about 32 tok/s decode on an iPhone 17 Pro, and loses only 1.4 points on mDolly. Aya Expanse 8B is about 2.4x larger, so it needs roughly 2.4x more memory and is slower per token. No published per-token API prices were found.

### Cited Findings
- Tiny Aya quantisation: Q4_K_M is "optimal" at 2.14 GB, 32.4 tok/s decode, with a 1.4-point mDolly drop. Q4_0 loses 2.1 points and Q8_0 has negligible loss. Decode runs at about 10 tok/s on an iPhone 13 and 32 tok/s on an iPhone 17 Pro (100-in/100-out workload). Without quantisation the model "quickly run[s] out of memory even on newer generation devices" — [Tiny Aya report §6](https://arxiv.org/html/2603.11510)
- Tiny Aya (all four chat variants) and Aya Expanse 32B have a live "cohere" inference provider on HF. Aya Expanse 8B has none — [HF Hub repo details](https://hf.co/CohereLabs/tiny-aya-global)

### Inferences
- For an ROI slide: a fine-tuned 3.35B model at Q4_K_M (about 2 GB) runs locally on a laptop CPU or GPU at near-zero marginal cost per query, and fits easily alongside the database. That is the strongest deployment story among the allowed models. Aya Expanse 8B Q4_K_M is about 5 GB, still local-friendly but about 2.4x the compute per token. 32B needs a server GPU.
- Each agent episode is multi-turn: schema calls, SQL calls and an answer, often 2-6K tokens in total. Latency therefore scales with turns times tokens, which favours the smaller model further.

### Gaps
- No Cohere API or HF-provider prices per million tokens were found for Tiny Aya or Aya Expanse. Fetch the provider pricing page before building a quantitative ROI table. HF Jobs per-hour GPU prices were also not verified.

---

## Q8. Recommendation: best base model and fallback

### Takeaway
**Primary: Tiny Aya Global.** Do LoRA SFT on oracle multi-turn trajectories, then optionally a short GRPO stage with the verified-result reward. **Fallback: Aya Expanse 8B with 4-bit QLoRA SFT only.** Use it if Tiny Aya SFT plateaus clearly below it on the held-out set. **Rule out Aya Expanse 32B**, which cannot be trained on the hardware or budget, and **rule out Base, Fire, Water and Earth** as primaries, since they offer no documented advantage for an English SQL agent.

### Cited Findings (supporting the recommendation; all from sections above)
- Tiny Aya is 3.35B with 6.7 GB fp16 weights. Aya Expanse 8B is 16.06 GB fp16 and Aya Expanse 32B is about 32.3B parameters — [mirror file listings](https://huggingface.co/1-800-LLMs/tiny-aya-global/tree/main), [Aya Expanse 8B mirror](https://huggingface.co/adamo1139/aya-expanse-8b-ungated/tree/main), [Hub details](https://hf.co/CohereLabs/aya-expanse-32b)
- Global's preference tuning was "deliberately minimal ... to facilitate users and researchers in adapting the model to downstream tasks", and regional variants only help translation — [Tiny Aya report](https://arxiv.org/html/2603.11510)
- A Tiny Aya Global LoRA trained on a T4 with Lexsi's AgentTune raised exact-correct tool actions from 2/24 to 19/24 on held-out cases after 60 steps — [ToolBoundary card](https://huggingface.co/Aditya0427/ToolBoundary-tiny-aya-lora)
- Aya Expanse 8B has the stronger reasoning prior: MGSM 67.0 (best in its 2024 8B class) and Global-MMLU 53.7, against Tiny Aya Global's GlobalMGSM 52.8 and Global MMLU 44.9 — [Aya Expanse report](https://arxiv.org/abs/2412.04261), [Tiny Aya report](https://arxiv.org/html/2603.11510)

### Inferences
Comparison of the allowed models for this use case (synthesised from the cited facts above):

| | Tiny Aya Global | Tiny Aya Base | Tiny Aya Fire/Water/Earth | Aya Expanse 8B | Aya Expanse 32B |
|---|---|---|---|---|---|
| Params / arch | 3.35B, cohere2 | 3.35B, cohere2 | 3.35B, cohere2 | 8.03B, cohere (Command-R) | 32.3B, cohere |
| Context | 8K (4K sliding window, 3:1) | 8K | 8K | 8K | 128K |
| Instruction-tuned | SFT + minimal pref. tuning | No | SFT, merged with Global | SFT + DPO + merging | SFT + DPO + merging |
| Native tool template | No (strict alternation, no tool role) | No chat tag | Same as Global | Yes, legacy Command-R `tool_use` (no tool-result role) | Same as 8B |
| Measured agent baseline (this project) | 0% greedy / 2.5% T=0.8 | not tested | not tested | 12.5% (Q4_K_M GGUF) | not tested |
| Math proxy | GlobalMGSM 52.8 | n/a | not reported separately | MGSM 67.0 | MGSM 73.4 |
| fp16 weights vs one T4 | 6.7 GB, fits | fits | fits | 16.1 GB, does not fit | about 65 GB, no |
| SFT on 2xT4 | Easy (fp16 LoRA or QLoRA) | Easy | Easy | QLoRA only | Practically infeasible |
| GRPO on 2xT4 | Feasible (rollout GPU + train GPU) | Feasible | Feasible | Very tight / fragile | Infeasible |
| Community tool-use precedent | Several fine-tunes, incl. AgentTune T4 LoRA | none | GGUF template fixes only | none found | none found |

- **Why Tiny Aya Global first:**
  1. It is the only allowed model where the full SFT plus RL-on-verified-results loop, which the problem statement rewards, is realistic on Kaggle 2x T4 in about 10 days, with enough slack for several iterations.
  2. The observed failures (protocol, fabricated tool results, not ending the turn) are exactly what small-data SFT fixes fast; see the ToolBoundary result, and fix the stop tokens.
  3. The base model was explicitly kept "adaptable" by its authors.
  4. The improvement story from 0% to X% is strong and on-message for a hackathon about small, deployable, safe models. Tiny Aya also has the best documented multilingual safety numbers in its class, a nice add-on for an AI-safety hackathon.
- **Main risk with Tiny Aya:** its weaker reasoning ceiling (2-10 points behind Qwen3-4B-class models on math, worse on technical tasks) may cap accuracy on multi-join or aggregation questions even after SFT.
- **Mitigation and fallback:** build the SFT dataset once in a model-agnostic message format, and train Aya Expanse 8B with QLoRA on the same data on one T4, or on an HF Jobs bf16 GPU using the $20-30 credit. Compare both on the same held-out tasks. Switch the submission to Aya Expanse 8B only if its post-SFT accuracy beats Tiny Aya's post-SFT (and post-GRPO) accuracy by a clear margin. Even then, skip RL for 8B, or do only a tiny rejection-sampling / RFT round, which needs generation but not concurrent training.
- **Cheap check before committing:** run the identical zero-shot eval on Tiny Aya Fire. The report says it has a higher English share, it costs nothing to test, and it uses the same architecture and code path. Prefer Global unless Fire is clearly better.
- **Why not 32B:** about 18-20 GB even at 4-bit, which exceeds a single T4. Training would need naive splitting across both T4s at very low throughput, and GRPO is impossible on this budget. Its 128K context and stronger MGSM do not compensate within 10 days.
- **Why not Base:** it would spend scarce SFT data relearning chat formatting and turn-ending. Global already provides that, with only minimal preference tuning to "unlearn".

### Gaps
- No head-to-head evidence exists on SQL/agent tasks after fine-tuning for any pair of these models. The recommendation rests on compute feasibility, documented post-training design, proxy benchmarks and the project's own zero-shot baselines.
- Unsloth and vLLM-on-T4 support for `cohere2` should be smoke-tested before the GRPO plan is finalised. If vLLM fails on T4, TRL GRPO with HF `generate` still works for a 3.35B model, just slower.
