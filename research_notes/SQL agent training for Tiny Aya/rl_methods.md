# RL methods (GRPO & successors, multi-turn agentic RL) for small SQL / database / tool-calling agents

Researched 2026-11-04. Source-verification legend used throughout:
- **[fetched]** = page/abstract read in this session.
- **[search-confirmed]** = claim confirmed by search-result snippets in this session but full paper not read.
- **[not re-fetched]** = paper identity/claim from the researcher's background knowledge of the arXiv abstract; the arXiv ID is given so the writer can spot-check. Numbers marked this way should be treated as "reported, verify before quoting".
Dates after mid-2025 are stated explicitly.

---

## 1. GRPO and its 2025-2026 variants: what each fixes, and which matter for small models / multi-turn agents

### Takeaway
All the useful variants are small patches to GRPO's loss or batching. The ones that matter at 3B-8B on a sparse 0/1 reward are: (a) **dropping zero-variance groups / dynamic sampling** (DAPO, Magistral), so compute isn't wasted on all-fail or all-pass prompts; (b) **clip-higher plus token-level loss** (DAPO), against entropy collapse and length bias; (c) **Dr. GRPO's unbiased normalisation**; (d) **GiGPO's step-level credit assignment** for multi-turn agents. GSPO and CISPO mainly target big-model and MoE stability and add little for a 3B dense LoRA run. TRL and Unsloth already expose most of these as config flags, so picking one costs almost nothing in engineering.

### Cited Findings
- **GRPO** comes from DeepSeekMath (2402.03300). It is critic-free and uses the group-mean (and std) of rewards over G samples per prompt as the baseline. DeepSeek-R1 (2501.12948) trained with GRPO. — [HF blog, S. Paniego, "How frontier models train on outcomes in 2026" (Aug 2026)](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
- **DAPO** (2503.14476) has four components: Clip-Higher (a larger upper clip ε to keep exploration and fight entropy collapse), Dynamic Sampling (drop prompts whose group is all-correct or all-wrong, so no zero-advantage batches), a token-level policy-gradient loss (long sequences aren't down-weighted), and overlong reward shaping. — [HF blog (Aug 2026)](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]; [DAPO arXiv](https://arxiv.org/abs/2503.14476)
  - DAPO reports 50 points on AIME 2024 with Qwen2.5-32B, using about 50% of the training steps of DeepSeek-R1-Zero-Qwen-32B. — [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) [not re-fetched]
- **Dr. GRPO** (2503.20783, "Understanding R1-Zero-like training") fixes GRPO's length bias. Dividing by response length and by the group std biases the model toward long wrong answers and toward low-variance (too easy or too hard) questions, and Dr. GRPO removes both normalisations. — [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]; [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
  - The paper reports a minimalist R1-Zero recipe reaching 43.3% on AIME 2024 with a 7B base. — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783) [not re-fetched]
- **GSPO** (Qwen, 2507.18071, July 2025) uses a sequence-level importance ratio and sequence-level clipping instead of per-token ratios. Qwen3 work credits it. In TRL it is enabled with `importance_sampling_level="sequence"`. — [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
  - Its main motivation is stabilising large and MoE models. — [arXiv 2507.18071](https://arxiv.org/abs/2507.18071) [not re-fetched]
- **CISPO** (MiniMax-M1, 2506.13585, June 2025) clips the importance-sampling weights rather than zeroing the gradient of clipped tokens, so rare high-information tokens keep their gradient. — [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
  - MiniMax reports about a 2x speed-up over DAPO in a controlled Qwen2.5-32B comparison. — [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) [not re-fetched]
- **Magistral** (2506.10910) filters zero-advantage groups out before forming batches, the same idea as DAPO's dynamic sampling. — [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
- **GMPO** (Geometric-Mean Policy Optimization, 2507.20673, July 2025) maximises the geometric rather than arithmetic mean of token-level ratio-weighted rewards. This makes it less sensitive to outlier importance ratios and gives more stable updates. — [arXiv 2507.20673](https://arxiv.org/abs/2507.20673) [not re-fetched; no numbers verified]
- **RLOO** (Ahmadian et al. 2024, "Back to Basics") is REINFORCE with a leave-one-out baseline across the k samples of a prompt. It is the conceptual ancestor of GRPO's group baseline, without std normalisation or clipping. — [arXiv 2402.14740](https://arxiv.org/abs/2402.14740) [not re-fetched]
- **REINFORCE++** (2501.03262) is critic-free REINFORCE with PPO-style clipping and global (batch-level) advantage normalisation instead of per-prompt normalisation. It is pitched as more stable and less hack-prone than GRPO. — [arXiv 2501.03262](https://arxiv.org/abs/2501.03262) [not re-fetched]
- **GiGPO** (Group-in-Group PO, 2505.10978, NeurIPS 2025; v3 dated Oct 28, 2025) keeps GRPO's critic-free, low-memory design and adds step-level advantages. "Anchor state grouping" retroactively groups actions taken from identical environment states across trajectories and compares them. — [GiGPO arXiv HTML v3](https://arxiv.org/html/2505.10978v3) [search-confirmed]; [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118123)
  - With Qwen2.5-1.5B/3B/7B-Instruct it reports gains over GRPO of more than 12% on ALFWorld and more than 9% on WebShop, at the same GPU memory and rollout cost. Search-augmented QA reaches 42.1% at 3B and 47.2% at 7B. The code is in **verl-agent**. — [GiGPO arXiv](https://arxiv.org/abs/2505.10978v3) [search-confirmed]
- **Framework defaults as of 2026.** TRL's `GRPOTrainer` `loss_type` defaults to `"dapo"` as of Aug 2026. — [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
  - Unsloth's GRPOConfig exposes `loss_type='gspo'|'grpo'|'dr_grpo'`, `epsilon`, `epsilon_high` (clip-higher), `delta`, and `mask_truncated_completions=True`. — [Unsloth RL docs](https://unsloth.ai/docs/basics/reasoning-grpo-rl-ppo) [fetched]
- **Newer (2026) multi-turn stability work found but not read.** T²PO, "Uncertainty-Guided Exploration Control for Stable Multi-Turn Agentic RL" (arXiv 2605.02178, May 2026). — [arXiv PDF](https://arxiv.org/pdf/2605.02178) [title only]

### Inferences
- **What matters most for this project.** Base pass@1 is about 0-2.5%, so the dominant problem is that most GRPO groups are all-zero and contribute no gradient. **Dynamic sampling / zero-advantage filtering (DAPO, Magistral)** is the single most important variant feature. In practice it means pre-filtering the task pool to prompts where the SFT policy's success rate is strictly between 0 and 1.
- Use `loss_type="dapo"` (TRL default) or `dr_grpo` with `epsilon_high` around 0.28 (clip-higher). Neither costs anything extra. GSPO and CISPO are unlikely to matter for a 3B dense LoRA run.
- GiGPO is attractive conceptually for this domain, because every rollout of the same task starts from the same state and often repeats `list_tables()` / `describe_table()`. That gives natural anchor states. But it is only implemented in verl-agent (verl/Ray, which assumes bf16-capable multi-GPU setups), so porting it to TRL on T4s in 10 days is unrealistic.

### Gaps
- GMPO, REINFORCE++, RLOO and the Dr. GRPO / DAPO headline numbers were not re-read this session. No head-to-head comparison of these variants on a ~3B **multi-turn tool** task was found; nearly all variant papers benchmark on math.
- No 2026 variant was found that clearly supersedes DAPO/Dr. GRPO-style GRPO for small models. T²PO (May 2026) was found by title only.

---

## 2. Multi-turn and single-turn RL for SQL / database agents: base models, data, rewards, compute, gains

### Takeaway
RL for text-to-SQL works well at **7B-14B Qwen2.5-Coder** scale with tiny datasets (653-1,027 RL questions) and mostly binary execution rewards. Multi-turn (agentic) RL beats single-turn RL with the same data. Two qualifications apply. First, every strong result starts from a **code-specialised base that already scores 50-80% EX**. Second, the most recent and smallest-model work (SQL-Trail, which includes 3B) uses an **SFT cold start distilled from a strong teacher plus dense shaping rewards**. No published SQL-agent RL result starts from a ~0% base, and none uses a non-code 3B multilingual model like Tiny Aya.

### Cited Findings
- **SkyRL-SQL (NovaSky / Berkeley Sky, ~May 2025; presented at NeurIPS 2025)**
  - It is a multi-turn RL pipeline. SkyRL-SQL-7B (Qwen2.5-Coder-7B-Instruct) was trained on **653 samples** and is reported to outperform GPT-4o and o4-mini, improving base EX by up to 9.2% across 5 Spider benchmarks. — [SkyRL GitHub (mirror)](https://awesome.ecosyste.ms/projects/github.com%2FNovaSky-AI%2FSkyRL); [Notion blog](https://novasky-ai.notion.site/skyrl-sql) [search-confirmed; Notion page did not render on fetch]; [NeurIPS 2025 talk](https://neurips.cc/virtual/2025/128018)
  - The reward is **binary**: the SQL in the `<solution>` block must execute and its results must exactly match the gold query's results. — [search summary of blog/README](https://novasky-ai.notion.site/skyrl-sql) [search-confirmed]
  - Multi-turn vs single-turn on the same 653 samples and rewards: multi-turn reached 60% training reward **2.8x faster** (in steps), finished **+16% higher** after 35 steps, and was **+1.6% EX** on average at eval. — [Substack commentary](https://machinelearningatscale.substack.com/p/text-to-sql-just-got-a-lot-better) [search-confirmed]
  - Evaluation uses 5 turns. — [SkyRL GitHub](https://awesome.ecosyste.ms/projects/github.com%2FNovaSky-AI%2FSkyRL)
  - An **unofficial** community reproduction card lists up to 6 turns, batch 128 prompts x 5 samples, 16k context, and "10 optimizer steps over 2 epochs". — [community HF card](https://huggingface.co/lecunyin/Qwen2.5-Coder-7B-SkyRL-SQL) [search-confirmed; unofficial]
  - Conflict: "10 steps" vs "35 steps" in the blog comparison. Treat step counts as uncertain.
  - Official GPU-hours were **not found**. The only SkyRL hour figure found was for a SWE agent (SkyRL-Agent-14B-v0: 20 h on 8xH200), which is not SQL. — [SkyRL GitHub mirror](https://awesome.ecosyste.ms/projects/github.com%2FNovaSky-AI%2FSkyRL)
  - Progress-SQL (2026) reports SkyRL-SQL-7B at 83.9 EX on Spider Dev and 85.2 on Spider Test. — [Progress-SQL arXiv 2606.06825](https://arxiv.org/html/2606.06825) [fetched]
- **Arctic-Text2SQL-R1 (Snowflake, arXiv 2505.20315, May 2025)**
  - GRPO with a **simple reward**: execution correctness plus a syntax/executability term. The paper contrasts this with other GRPO text-to-SQL work that adds length, format or LLM-judge terms, and argues the simple reward is enough. — [arXiv 2505.20315](https://arxiv.org/abs/2505.20315) [search-confirmed]
  - BIRD scores (test / dev): 32B 71.8% / 70.5% (first on the BIRD single-model leaderboard at the time), 14B 70.0% / 70.1%, 7B 68.5% / 68.9%. Value retrieval and majority voting at inference add more. — [arXiv 2505.20315](https://arxiv.org/html/2505.20315v2); [HF model card](https://huggingface.co/Snowflake/Arctic-Text2SQL-R1-7B) [search-confirmed]
- **SQL-R1 (arXiv 2504.08600, Apr 2025)**
  - The 7B model reports 88.6% EX on Spider and 66.6% on BIRD. — [search snippet](https://huggingface.co/papers/2503.23157) [search-confirmed]
  - Progress-SQL's table lists SQL-R1-7B at **59.8** on BIRD Dev under its own protocol. This conflicts with 66.6%; the gap is likely self-consistency vs greedy decoding. — [Progress-SQL](https://arxiv.org/html/2606.06825) [fetched]
  - The rewards are format, execution, result and length. — [arXiv 2504.08600](https://arxiv.org/abs/2504.08600) [not re-fetched]
- **Reasoning-SQL (Google, arXiv 2503.23157, Mar 2025)**
  - GRPO with **partial rewards**: schema-linking, AI feedback (LLM judge), n-gram similarity and syntax check, designed to counter sparse execution rewards. — [arXiv 2503.23157](https://arxiv.org/abs/2503.23157v2) [search-confirmed]
  - The 14B model beats o3-mini by 4% and Gemini-1.5-Pro-002 by 3% on BIRD. A review notes the headline number is **with the CHASE-SQL pipeline**. — [arXiv](https://arxiv.org/abs/2503.23157v2); [Liner review](https://liner.com/review/reasoningsql-reinforcement-learning-with-sql-tailored-partial-rewards-for-reasoningenhanced) [search-confirmed]
  - Progress-SQL lists Reasoning-SQL-7B at 64.0 on BIRD Dev. — [Progress-SQL](https://arxiv.org/html/2606.06825) [fetched]
- **Think2SQL (arXiv 2504.15077, Apr 2025)** compares SFT, RL and SFT+RL for Qwen2.5-Coder on BIRD and Spider. It argues that small models gain more from RL with fine-grained, partial execution rewards. — [arXiv 2504.15077](https://arxiv.org/abs/2504.15077) [not re-fetched; ID confirmed to exist]
- **OmniSQL (arXiv 2503.02240, Mar 2025)** is an SFT-only family (7B/14B/32B) trained on the synthetic **SynSQL-2.5M** dataset. SynSQL is later reused as an RL question source by SQL-Trail and Progress-SQL. — [arXiv 2503.02240](https://arxiv.org/abs/2503.02240) [not re-fetched]; SynSQL usage in [SQL-Trail](https://arxiv.org/html/2601.17699v1) and [Progress-SQL](https://arxiv.org/html/2606.06825) [fetched]
- **SQL-Trail (arXiv 2601.17699, Jan 2026)**
  - A multi-turn reason → execute → observe agent with a 10-turn budget, built on **Qwen2.5-Coder 3B / 7B / 14B**. — [arXiv 2601.17699](https://arxiv.org/html/2601.17699v1) [fetched]
  - Stage 1 is **SFT on 1,000 correct multi-turn trajectories distilled from Claude-Sonnet-3.7**.
  - Stage 2 is RL on **1,027 questions**: 700 "hard-but-solvable" SynSQL questions plus 327 exploration items.
  - The RL algorithm is modified GRPO with **no KL** and **clip-higher**, 6 rollouts per query, temperature 1.0, LR 1e-6, batch 128.
  - The reward is a weighted sum: execution correctness (x5), turn-budget efficiency (x2), and schema grounding, bigram overlap, syntax validity and format (x1 each). The authors say binary execution alone is "overly sparse", and bigram overlap was the most useful partial reward.
  - Results: 7B scores 60.1% on BIRD-dev greedy and 64.2% with majority vote.
  - Under matched no-SFT conditions, multi-turn RL beats single-pass RL by about 3 points (59.3 vs 56.3, majority vote).
  - It claims up to **18x data efficiency** vs single-pass RL. Per 1k examples, the 7B gains 4.60 pp on BIRD-dev vs 1.6 for SQL-R1-7B.
  - Small-model lesson stated by the authors: smaller models struggle with long, structured agent prompts (formatting errors, premature tool calls, action loops), and **SFT cold start gives RL a stable starting point**. GPU hours are not reported in the visible text.
- **Progress-SQL (arXiv 2606.06825, Jun 5, 2026)**
  - Multi-turn RL with up to 4 revision turns, on Qwen2.5-Coder-7B and 14B, using GRPO in verl. — [arXiv 2606.06825](https://arxiv.org/html/2606.06825) [fetched]
  - Training data is 20k instances from BIRD, Spider and SynSQL. Compute was 8xH100 with G=8, LR 1e-6, batch 64.
  - The reward is a **trajectory-level progressive reward**: structural and lexical improvement from first to final SQL, a decayed early-correctness bonus (γ=0.5), execution-status keep/recover/deteriorate terms, and a format bonus. During training a gold-SQL-based diagnostic tree supplies clause-level error feedback.
  - Results: 7B scores 64.3 on BIRD Dev and 87.8 EX on Spider Test. It gains 8.5% average EX over base and beats SkyRL-SQL-7B on Spider.
  - Its critique of SkyRL-SQL: the reward is computed from the final turn only, so it is still effectively one-shot and sparse.
- **BIRD-RL (NeurIPS 2026 poster)** uses a four-stage agentic-RL **curriculum** for small models over stateful data environments. Its 7B and 14B models reach 44.40% and 48.00% success on BIRD-CRITIC-SQLite, on par with a Claude-Opus-4.6 agent at 48.20%. It also claims 12.8x RL efficiency from its execution infrastructure. — [NeurIPS 2026 poster page](https://neurips.cc/virtual/2026/poster/152242) [search-confirmed; abstract-level only]
- **MARFT (multi-agent RL fine-tuning, arXiv 2504.16129)** is aimed at LLM multi-agent systems. It is not relevant to a single-agent SQL loop. — [arXiv 2504.16129](https://arxiv.org/abs/2504.16129) [not re-fetched]

### Inferences
- **Data scale is not the bottleneck.** 235 expert trajectories for SFT plus a few hundred verifiable RL tasks is in the same order of magnitude as SkyRL-SQL (653) and SQL-Trail (1,000 SFT + 1,027 RL).
- **Base capability is the bottleneck.** Every SQL-RL success above starts from Qwen2.5-Coder models that already score 50-80% EX zero-shot. Tiny Aya at 0% is in a different regime.
- The closest analogue to this project's plan is **SQL-Trail's recipe**: SFT on about 1k correct multi-turn trajectories, then GRPO on "hard-but-solvable" questions with a turn-efficiency term. It is also the only one that reports a 3B model.
- On the "simple vs dense reward" disagreement: Arctic says a simple execution reward is enough at 7B-32B, while SQL-Trail, Think2SQL and Reasoning-SQL say partial rewards help, especially for smaller models. For a weak 3B policy, a small amount of shaping is the safer bet. Example terms: called `run_sql` successfully, final answer parses, turn budget respected. Keep the correct-answer term dominant, as SQL-Trail's x5 weight does.

### Gaps
- **BIRD-Interact** RL results: none found this session. Only BIRD-RL and BIRD-CRITIC numbers turned up.
- Official GPU-hours for SkyRL-SQL, SQL-Trail and Arctic were not found.
- Arctic's exact reward constants and whether it starts from OmniSQL checkpoints were not verified.
- No SQL-RL paper using Cohere or Aya models, or any non-code multilingual 3B base, was found.

---

## 3. General multi-turn tool-use RL frameworks and the tricks that matter

### Takeaway
The settled recipe for multi-turn tool RL has several parts. **Mask environment/tool tokens out of the loss.** Use a trajectory-level outcome reward, optionally with step-level credit (GiGPO), turn-efficiency shaping or a small format reward. **Filter zero-variance groups** and watch **reward-std and entropy as early-warning signals** of collapse (RAGEN's "Echo Trap"). A cold-start SFT that teaches the tool protocol is usually used. TRL now has a built-in multi-turn environment loop (`environment_factory`), which is the most practical option on Kaggle-class hardware. verl, verl-agent, SkyRL, rLLM and RAGEN are built for 8xA100/H100 clusters.

### Cited Findings
- **Masking tool outputs.**
  - Search-R1 masks retrieved tokens from the loss. It reports +41% (Qwen2.5-7B) and +20% (3B) over RAG baselines. — [arXiv 2503.09516](https://arxiv.org/abs/2503.09516) [not re-fetched]
  - A Sept 2026 small-search-agent paper also masks retrieved tokens. Its reproduced Search-R1 Qwen2.5-3B GRPO went from 0.199 to 0.341 EM. — [arXiv 2609.28765](https://arxiv.org/html/2609.28765v1) [fetched]
  - CacheRL (Jun 2026) uses token-level masking for tool results. — [arXiv 2606.14179](https://arxiv.org/abs/2606.14179) [fetched]
- **RAGEN / StarPO (arXiv 2504.20073, Apr 2025).**
  - Vanilla PPO/GRPO in multi-turn settings show an **"Echo Trap"**: reward-variance collapse, entropy drop and gradient spikes, after which performance collapses. — [arXiv 2504.20073](https://arxiv.org/abs/2504.20073) [search-confirmed]; [AI News summary](https://www.artificialintelligence-news.com/news/ragen-ai-framework-tackles-llm-agent-instability/)
  - Reward std and entropy fall *before* performance degrades. — [Northwestern summary](https://mccormick.northwestern.edu/computer-science/news-events/news/articles/2025/training-reasoning-agents-in-interactive-complex-environments.html) [secondary]
  - The fix, **StarPO-S**, adds variance-based trajectory filtering, critic incorporation and decoupled clipping.
  - The authors also say reasoning "hardly emerges" with outcome-only rewards unless the reward is reasoning-aware.
- **GiGPO / verl-agent** provides step-level advantages via anchor states, at no extra memory or rollout cost (see section 1). Results cover 1.5B to 7B. — [GiGPO](https://arxiv.org/abs/2505.10978v3) [search-confirmed]
- **ReTool (arXiv 2504.11536)** combines a code-interpreter tool with a cold-start SFT on code-integrated reasoning data, then RL. With a 32B model it reports 67% on AIME 2024 after 400 RL steps, vs 40% for text-only RL after 1,080 steps. — [arXiv 2504.11536](https://arxiv.org/abs/2504.11536) [not re-fetched]
- **ToolRL (arXiv 2504.13958)** is a systematic study of reward design for tool calling: fine-grained format and correctness components, with scale and dynamics varied. It reports about +17% over base models and +15% over SFT models with GRPO. — [arXiv 2504.13958](https://arxiv.org/abs/2504.13958) [not re-fetched]
- **ARPO (arXiv 2507.19849, Jul 2025)** adds extra branching rollouts at high-entropy steps right after tool calls, and reports needing about half the tool-call budget of prior methods. **AEPO (arXiv 2510.14545, Oct 2025)** is an entropy-balanced follow-up. — [ARPO](https://arxiv.org/abs/2507.19849); [AEPO](https://arxiv.org/abs/2510.14545) [both not re-fetched]
- **OpenPipe ART + RULER (July 2025).**
  - ART is an open-source GRPO trainer for agents. RULER (Relative Universal LLM-Elicited Rewards) uses an LLM judge to *rank* a group of trajectories relatively, replacing hand-written reward functions. — [ART GitHub/PyPI](https://pypi.org/project/openpipe-art/)
  - Founder claims (Show HN, **not independently verified**): small Qwen2.5 models trained with RULER+GRPO beat the best prompted frontier model on 4 tasks, and beat hand-crafted rewards on 3 of 4. The ART-E email agent (Qwen2.5-14B) beat o3 on email retrieval. — [Show HN mirror](https://hnews-reader.vercel.app/top/44535078); [ART summary](https://awesome.ecosyste.ms/projects/github.com%2FOpenPipe%2FART)
- **TRL multi-turn support (2026).**
  - `GRPOTrainer(environment_factory=...)` runs the tool loop automatically: generate, parse tool calls, execute against a stateful environment instance, feed results back. Public methods of the env class become tools (docstrings required). Rewards read env state. — [TRL OpenEnv docs](https://huggingface.co/docs/trl/en/openenv) [search-confirmed]
  - `max_completion_length` caps total tokens **across the whole multi-turn conversation**.
  - Examples include `grpo_wordle`, `grpo_sudoku` and `grpo_browsergym`, with OpenEnv, OpenReward and Harbor integrations. The docs exist for versions v1.5 to v1.9+.
  - This maps directly onto `list_tables()`, `describe_table()` and `run_sql()`.
- **Frontier and industry practice (2026).**
  - Liquid AI's **LFM2.5-2.6B** does agentic RL with one sandbox per rollout, GRPO, and an outcome reward mixing an LLM judge, programmatic checks and a safety gate. — [HF blog (Aug 2026)](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
  - Kimi K2 (2507.20534) uses a self-critic anchored on verifiable-reward prompts. — [HF blog (Aug 2026)](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
  - DeepSWE (rLLM / Agentica, on Qwen3-32B) used **no agent-specific SFT** before RL, but from a strong base. OpenThinkerAgent-8B-RL used about 9.4K cold-start SFT trajectories, then about 5K RL tasks. — [Modal roundup](https://modal.com/resources/best-open-source-models-agentic-rl) [search-confirmed]
- **CacheRL (arXiv 2606.14179, Jun 12, 2026)** works on Qwen3-4B-Thinking. It uses a three-tier fuzzy cache in place of live tool execution, token masking, a cache-aware reward, and iterative SFT then GRPO.
  - Validation reward rose from 0.43 to 0.78, with 92% process accuracy vs GPT-5's 94%.
  - The authors state plainly that "**RL improves training stability but yields limited gains beyond strong SFT**", and that data quality and reward design matter more than optimiser complexity. — [arXiv 2606.14179](https://arxiv.org/abs/2606.14179) [fetched, abstract]

### Inferences
- For this project's protocol failures (not ending the turn, fabricating tool results), the **rollout engine** prevents fabrication, not the RL objective. Stop generation at the tool-call boundary, inject real tool output, and mask it from the loss. TRL's `environment_factory` does this. Fabricated observations then become impossible during RL rollouts, and "not ending the turn" becomes a reward/format term.
- Monitor reward std, entropy and gradient norm every step. With G=4-8 and few prompts per step on T4s, variance collapse (Echo Trap) is likely. Stop early and keep the best checkpoint.
- RULER-style LLM-judge rewards are unnecessary here because the answer is verifiable. At most, use a judge for tie-breaking within groups where every rollout is wrong.

### Gaps
- No quantitative comparison found of turn-level vs trajectory-level advantages specifically on SQL agents; GiGPO tests ALFWorld and WebShop.
- Agent-R1 and rLLM papers were not read this session.
- Whether TRL's `environment_factory` path requires vLLM, and whether it supports Tiny Aya's chat and tool template, was not confirmed.

---

## 4. RL from a near-zero-success base: cold start, curriculum, hints, and the "RL sharpens vs expands" debate

### Takeaway
Evidence strongly indicates that **on-policy GRPO cannot bootstrap from ~0% success**. With all-zero groups the advantage is zero and there is no gradient. Even when it partially works, RLVR mostly **re-weights behaviours the base model already has** (raising pass@1 toward pass@k) rather than creating new ones, except under very long training (ProRL). The standard fix is an **SFT/distillation cold start** on interaction-dense expert trajectories. On top of that come a task pool filtered to "hard-but-solvable" for the *current* policy, a curriculum, or off-policy expert traces mixed into groups (LUFFY). The project's plan (SFT first, RL only once rollouts are mixed) matches the literature.

### Cited Findings
- **Yue et al. 2025, "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"** (arXiv 2504.13837, NeurIPS 2025; v5 Nov 24, 2025).
  - RLVR models beat base models at small k but **fall behind at large k** (pass@k up to 256).
  - Coverage and perplexity analysis shows reasoning abilities "originate from and are bounded by the base model". RL biases the distribution toward reward-yielding paths, narrowing the boundary.
  - **Distillation, by contrast, can introduce new patterns.** — [arXiv 2504.13837](https://arxiv.org/abs/2504.13837v4) [search-confirmed]
- **ProRL (NVIDIA, arXiv 2505.24864, May 2025)** is the counter-evidence.
  - Prolonged RL with KL control, reference-policy resets and a diverse task suite (136K problems) on DeepSeek-R1-Distill-Qwen-1.5B "uncovers novel reasoning strategies inaccessible to base models".
  - Gains: +15.7% math, +14.4% code, +54.8% logic puzzles (pass@1).
  - But there is a **negative correlation between base capability and gain**, and pass@128 was checked on only 18 of 96 Reasoning Gym tasks. — [arXiv 2505.24864](https://arxiv.org/pdf/2505.24864) [search-confirmed]
- **Small search agents (arXiv 2609.28765, Sep 23, 2026).**
  - Cites prior work (DGPO; Kotoge et al.) finding pure RL "too unstable to bootstrap search behaviour from a cold start at 0.5B to 1B", recoverable only by distillation from a larger teacher.
  - Its own pure-RL success on Qwen3.5-0.8B started from a **non-zero floor (EM 0.092)** and used the model's native tool-call format.
  - Dense F1 reward beat sparse EM at every seed, and the authors argue reward shape matters more at small scale.
  - Compute was 4x RTX PRO 6000 Blackwell, about $120-140 per ~312-step run, GRPO with G=5, β=1e-3 and LR 1e-6. — [arXiv 2609.28765](https://arxiv.org/html/2609.28765v1) [fetched]
- **ASTER (arXiv 2602.01204, Feb 1, 2026; 4B).**
  - Without cold start, "interaction collapse" occurs: models stop sustaining multi-turn tool use and degrade to internal reasoning with trivial tool calls. — [arXiv 2602.01204](https://arxiv.org/abs/2602.01204) [fetched, abstract]
  - A **small expert cold-start set of ~4K interaction-dense trajectories** gave the best downstream RL. — [arXiv 2602.01204](https://arxiv.org/abs/2602.01204) [fetched, abstract]
  - Cold-start data dominated by sparse, short tool interactions can limit RL gains. — [search snippet](https://arxiv.org/pdf/2602.01204) [search-confirmed]
- **"Plan Before Search" (arXiv 2605.28354, May 2026)** says SFT cold start via stronger-model distillation has become the default fallback when direct RL fails, applied inconsistently. It claims a small seed model suffices and a stronger teacher is not needed. — [arXiv 2605.28354](https://arxiv.org/pdf/2605.28354) [search-confirmed]
- **SQL-Trail** found SFT cold start necessary for small models, which otherwise produce formatting errors, premature tool calls and loops. It then ran RL on "hard-but-solvable" questions. — [arXiv 2601.17699](https://arxiv.org/html/2601.17699v1) [fetched]
  - **BIRD-RL** uses a four-stage curriculum. — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/poster/152242) [search-confirmed]
- **LUFFY (arXiv 2504.14945, NeurIPS 2025)** mixes off-policy expert traces into each GRPO group and normalises advantages over the combined rewards. Policy shaping of the off-policy importance weight (γ=0.1) avoids entropy collapse. — [NeurIPS 2025 abstract](https://papers.neurips.cc/paper_files/paper/2025/hash/a9d5c33e38442450f9781bd7bad3c81e-Abstract-Conference.html); [described in arXiv 2601.05787](https://arxiv.org/pdf/2601.05787) [search-confirmed]
  - Reports +6.4 average on six math benchmarks and "trains weak models where on-policy RLVR fails".
  - An unreviewed write-up notes that when pass@k is 0, only the off-policy traces carry signal (pure imitation), and that LUFFY can be unstable on hard problems. — [idrori.github.io (unreviewed)](https://idrori.github.io/open/problems/LG/stabilize-luffy-training-on-hard-problems/paper/main.pdf)
  - arXiv 2512.06337 proposes hybrid off-policy injection for hard tasks with uniformly poor rollouts. — [arXiv 2512.06337](https://arxiv.org/pdf/2512.06337v2) [search-confirmed]
- **DAPO dynamic sampling** and **Magistral zero-advantage filtering** drop all-wrong and all-right groups. Llama 4 online RL keeps only medium-to-hard prompts. — [HF blog (Aug 2026)](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
- **Unsloth's guidance**: wait at least 300 steps before expecting reward to rise, and use models of at least 1.5B parameters. — [Unsloth RL docs](https://unsloth.ai/docs/basics/reasoning-grpo-rl-ppo) [fetched]

### Inferences
- With base pass@4 at 4/40 and greedy at 0%, roughly 90% of tasks give all-zero groups at G=4. GRPO directly on Tiny Aya would waste nearly all compute and is likely to suffer interaction collapse (ASTER).
- **SFT first is mandatory.** The go/no-go gate for RL should be the measured success rate of the SFT checkpoint per task at G=4-8. RL is worthwhile only if a meaningful share of tasks (say 20% or more) have 0 < success < 1.
- Yue et al. imply that most of what RL would buy here is **pass@k → pass@1 sharpening**. That can be captured much more cheaply with **rejection-sampling fine-tuning / expert iteration (STaR-style)**: sample k rollouts per task from the SFT policy, keep execution-verified successes, add them to the SFT set, retrain. It needs only inference plus SFT, both of which already work on T4s.
- If RL is attempted, the most budget-friendly way to inject signal on all-fail tasks is LUFFY-like: include one of the 235 expert trajectories as an off-policy member of the group. With TRL this is non-trivial to implement. A simpler proxy is to mix an SFT loss on expert trajectories into the RL batch.

### Gaps
- No study found that runs multi-turn **SQL** agent RL from a ~0% base, or that quantifies the minimum SFT-policy success rate needed before GRPO helps.
- No direct study found of rejection-sampling FT vs GRPO for small tool agents. The comparison above is an inference from Yue et al. and CacheRL's "limited gains beyond strong SFT".

---

## 5. Smallest models where agentic RL works, compute used, and feasibility of GRPO + QLoRA on 2x T4 (incl. verdict)

### Takeaway
Agentic or tool RL has been shown at **0.8B-4B** with reasonable results: Search-R1 3B, GiGPO 1.5B/3B, SQL-Trail 3B, CacheRL 4B, ASTER 4B, Qwen3.5-0.8B search agent, LFM2.5-2.6B, and a 3B MLE-bench agent. But it ran on modern GPUs: 4x RTX PRO 6000 at about $120-140 per run for the cheapest, otherwise 8xA100/H100. GRPO with QLoRA **does run on a single T4** via Unsloth. However, multi-turn agentic RL for a 3.35B model on fp16-only T4s, with an unverified vLLM/Unsloth path for Tiny Aya's architecture, would likely yield only on the order of **100-300 steps in the available GPU time**. That is at or below Unsloth's own "≥300 steps before reward rises" rule of thumb. **Verdict:** treat GRPO as an optional stretch goal. The primary path should be SFT plus rejection-sampling self-improvement.

### Cited Findings
- **Small-model agentic RL evidence:**
  - **GiGPO** on Qwen2.5-1.5B/3B/7B: ALFWorld more than 12% and WebShop more than 9% over GRPO. — [arXiv 2505.10978](https://arxiv.org/abs/2505.10978v3) [search-confirmed]
  - **Search-R1** 3B: 0.199 → 0.341 EM in the 2026 reproduction. — [arXiv 2609.28765](https://arxiv.org/html/2609.28765v1) [fetched]
  - **Qwen3.5-0.8B** search agent, pure RL: 0.092 → 0.352 EM, about 312 steps on 4x RTX PRO 6000, about $120-140 per run. — [arXiv 2609.28765](https://arxiv.org/html/2609.28765v1) [fetched]
  - **SQL-Trail 3B** (Qwen2.5-Coder-3B), SFT then RL. — [arXiv 2601.17699](https://arxiv.org/html/2601.17699v1) [fetched]
  - **CacheRL 4B**, SFT then GRPO, claiming about 100x less compute than frontier approaches (no hardware figures). — [arXiv 2606.14179](https://arxiv.org/abs/2606.14179) [fetched]
  - **ASTER-4B**, 4K cold-start trajectories then RL. — [arXiv 2602.01204](https://arxiv.org/abs/2602.01204) [fetched]
  - **LFM2.5-2.6B**, agentic RL with GRPO in 2026. — [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
  - **DemyAgent-4B**, 3K SFT plus 30K RL samples. — [Open-AgentRL GitHub](https://github.com/Gen-Verse/Open-AgentRL) [search-confirmed]
  - **Qwen2.5-3B** RL on MLE-bench reportedly beat prompted Claude-3.5-Sonnet agent scaffolds by about 22% on average across 12 tasks. — [arXiv 2509.01684](https://arxiv.org/html/2509.01684v1) [search-confirmed]
  - RAGEN/StarPO studied very small Qwen models in toy multi-turn environments. — [arXiv 2504.20073](https://arxiv.org/abs/2504.20073) [search-confirmed]
- **Compute in SQL RL papers:**
  - Progress-SQL: 8xH100, G=8, batch 64. — [arXiv 2606.06825](https://arxiv.org/html/2606.06825) [fetched]
  - SkyRL ran on H100 nodes, with no SQL hours published. — [SkyRL mirror](https://awesome.ecosyste.ms/projects/github.com%2FNovaSky-AI%2FSkyRL)
  - SQL-Trail: batch 128, 6 rollouts, hardware not reported in the visible text. — [arXiv 2601.17699](https://arxiv.org/html/2601.17699v1)
- **Unsloth GRPO memory and speed:**
  - Up to 17B fits on 15GB with QLoRA, and models of 1.5B or less need about 5GB. — [Unsloth RL docs](https://unsloth.ai/docs/basics/reasoning-grpo-rl-ppo) [fetched]
  - The QLoRA VRAM rule of thumb is about the parameter count in GB, plus context.
  - vLLM `fast_inference=True` gives about **300 tok/s on a T4** for Llama-3.2-3B 4-bit, vs about 4,000 tok/s on an A100-40GB. Unsupported architectures need `fast_inference=False`.
  - Guidance: at least 300 steps; good results "may take at least 12 hours"; at least 500 rows of data; 8 generations, or 16.
- **T4 / fp16 caveat:** Unsloth reported that QLoRA GRPO runs past about 60 steps could diverge and error out **only on non-bf16 GPUs that fall back to fp16 (e.g. T4)**. Gemma could not use vLLM on T4 because vLLM restricted it to bf16. — [Unsloth vision-RL blog (Aug 2025)](https://unsloth.ai/blog/vision-rl) [search-confirmed]
  - Unsloth also says GRPO runs on a free Colab T4 for Qwen2.5-VL-7B. — [same](https://unsloth.ai/blog/vision-rl)
- **Counterpoint on precision:** "Defeating the Training-Inference Mismatch via FP16" (arXiv 2510.26788, Oct 30, 2025; Sea AI Lab/NUS) finds that **fp16 rather than bf16 reduces training/inference mismatch and stabilises RL**. Its setup was DeepSeek-R1-Distill-Qwen-1.5B on 8xA100. fp16 itself is therefore not disqualifying for RL. The T4 issues are more likely overflow and kernel support than fp16 per se. — [arXiv 2510.26788](https://arxiv.org/abs/2510.26788v1) [search-confirmed]
- **TRL** supports multi-turn environments natively (`environment_factory`), the DAPO loss (default) and GSPO. — [TRL OpenEnv docs](https://huggingface.co/docs/trl/en/openenv); [HF blog](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026)
  - No Unsloth or TRL documentation of Tiny Aya GRPO support was found. — [search, no hits](https://docs.unsloth.ai/basics/reasoning-grpo)

### Inferences
- **Rough throughput estimate (inference, not a measured number).**
  - Assumptions: Tiny Aya 3.35B 4-bit on one T4, about 300 tok/s aggregate generation *if* a vLLM path works (Unsloth's 3B figure). Step size of 4 tasks x G=8 = 32 rollouts, about 3-5 turns, about 600-900 generated tokens per rollout, giving about 20-30k generated tokens per step, or about 1-2 min of generation.
  - Add multi-turn prefill and QLoRA forward/backward on 32 sequences of 2-4k tokens with gradient checkpointing, likely about 2-4 min on a T4.
  - Result: about **3-6 min per step, ~10-20 steps/hour with vLLM**. Without vLLM (HF `generate`, plausibly several times slower for batched multi-turn sampling) it drops to **~3-6 steps/hour**.
  - 300 steps would then be about 15-30 GPU-hours with vLLM, or 50-100 hours without. A second T4 can serve generation (TRL vLLM server mode) while the first trains, maybe giving a 1.5-2x gain.
- **Verdict for a 10-day solo schedule on 2xT4 plus ~$20-30 HF credits:**
  1. Full-scale GRPO (hundreds of steps, G≥8) is **marginal to infeasible** given the step estimate, fp16 risks and unverified Tiny Aya support in Unsloth's or TRL's fast-inference path.
  2. **Feasible and high expected value:** QLoRA SFT on the expert trajectories, then 1-2 rounds of **rejection-sampling self-distillation** (sample G=4-8 per task with the SFT model, keep execution-verified correct trajectories, SFT again). This captures most of the "RL sharpening" effect described by Yue et al. using only inference plus SFT.
  3. **Optional stretch:** a short TRL `GRPOTrainer` + `environment_factory` run (about 50-150 steps) on the filtered "0<p<1" task subset. Use DAPO loss, clip-higher, KL off or tiny, LR around 1e-6 to 5e-6 for LoRA, and a mostly binary reward with small shaping. Keep it only if held-out success improves. The HF credits are best spent on a single short A10G/L4/A100 job for this, not on T4s. Exact HF per-hour pricing was not verified.
  4. Aya Expanse 8B at 12.5% base is a better RL substrate than Tiny Aya because more tasks start with mixed outcomes, but it costs about 2.4x more per step.

### Gaps
- No measured GRPO steps/hour for any ~3B model on T4 in a **multi-turn** setting was found. The estimate above is the researcher's own inference.
- Whether vLLM and Unsloth `fast_inference` support Tiny Aya's architecture in fp16 on T4 is not verified, and it is the single biggest feasibility unknown.
- Kaggle's current weekly GPU quota and session limits were not verified this session.
- HF Jobs and Spaces GPU pricing was not verified this session.

---

## 6. Process rewards, LLM-judge rewards and reward-hacking defences for verified SQL answers

### Takeaway
With a fully verifiable answer, the best evidence favours a **dominant binary execution/answer reward**. Arctic shows it is sufficient at 7B+. Add **small, hard-to-game shaping terms**: format, executability, turn efficiency, and partial overlap as in SQL-Trail and Reasoning-SQL. LLM-judge rewards (RULER, Reasoning-SQL's AI feedback) mainly help when outcomes can't be verified. Known hacking risks are answering without querying, fabricating tool outputs, and exploiting shaping terms. These are best blocked by environment design (stop sequences, real tool execution, masking) and **gating** (shaping counts only if the outcome condition holds), not by more reward terms.

### Cited Findings
- Arctic-Text2SQL-R1 argues a simple execution-correctness plus syntax reward beats elaborate reward suites, and reaches state of the art at 7B-32B. — [arXiv 2505.20315](https://arxiv.org/abs/2505.20315) [search-confirmed]
- SQL-Trail's six-term reward (execution x5, turn budget x2, schema grounding, bigram, syntax, format x1) has bigram overlap as the most valuable partial term. The turn reward produced adaptive budgeting: more turns on hard queries, fewer on easy ones. — [arXiv 2601.17699](https://arxiv.org/html/2601.17699v1) [fetched]
- Reasoning-SQL partial rewards are schema linking, AI feedback (LLM judge), n-gram similarity and syntax check. — [arXiv 2503.23157](https://arxiv.org/abs/2503.23157v2) [search-confirmed]
- Progress-SQL uses process-style trajectory rewards: improvement from first to last SQL, an early-correctness bonus, and execution keep/recover/deteriorate terms. It requires **gold SQL** for its diagnostic feedback. — [arXiv 2606.06825](https://arxiv.org/html/2606.06825) [fetched]
- SkyRL-SQL's reward is binary on the final `<solution>` SQL; results must execute and exactly match gold results. — [Notion blog via search](https://novasky-ai.notion.site/skyrl-sql) [search-confirmed]
- Small-search-agent study (2026): dense F1 beat sparse EM at all seeds. Adding a 0.1 format floor gave no clear further gain (0.307 vs 0.305 EM). — [arXiv 2609.28765](https://arxiv.org/html/2609.28765v1) [fetched]
- Unsloth advises using verifiable rewards plus a rubric of several smaller rewards rather than all-or-nothing, and warns that poorly designed rewards degrade the model. — [Unsloth RL docs](https://unsloth.ai/docs/basics/reasoning-grpo-rl-ppo) [fetched]
- RULER uses an LLM judge to rank trajectories within a group, removing hand-written rewards. Its claims are founder-reported. — [Show HN mirror](https://hnews-reader.vercel.app/top/44535078)
- CacheRL found cache-aware reward weighting added 17%, and concluded reward design and data quality matter more than optimiser complexity. — [arXiv 2606.14179](https://arxiv.org/abs/2606.14179) [fetched]
- Frontier practice combines rule-based rewards, outcome reward models and generative reward models (GLM-5). LFM2.5 uses a judge plus programmatic checks plus a **safety gate**. — [HF blog (Aug 2026)](https://huggingface.co/blog/sergiopaniego/agentic-rl-2026) [fetched]
- ASTER's "interaction collapse" is a hacking-adjacent failure: the policy learns to skip real tool use. — [arXiv 2602.01204](https://arxiv.org/abs/2602.01204) [fetched]

### Inferences
- Concrete reward for this project, synthesised from the above (not taken from any one paper):
  - +1.0 if the `ANSWER:` value matches the verified answer **and** at least one `run_sql` call succeeded in the episode. This gate blocks answering without querying.
  - +0.1 for a well-formed final `ANSWER:` line, only if the episode ended correctly (no text after `ANSWER`).
  - -0.05 per turn beyond a budget of about 6.
  - 0 if the episode hit max turns or max tokens. Use `mask_truncated_completions`.
- Fabricated tool results should be prevented structurally. Stop generation at the tool-call end token, execute for real, and mask tool-output tokens from the loss. Rewarding against fabrication alone is not enough.
- Exploit to watch for: guessing answers from schema or column names, or from tiny `LIMIT` probes. A guard is a held-out evaluation with paraphrased questions and different databases, and checking that the answer value actually appears in a returned result set.
- Reasoning-SQL-style LLM-judge or n-gram partial rewards need gold SQL. The programmatic tasks have it, so a bigram or result-set-overlap partial reward (as in SQL-Trail) is cheap. Its expected benefit is mainly on all-fail groups, where it restores some gradient.

### Gaps
- No paper found that reports reward-hacking incidents specifically in SQL agent RL (e.g. answer-without-query rates). The defences above are inferences from general agentic-RL findings.
- Process reward models (PRMs) for SQL agents: none found this session that were cheap enough or open for a 3B setting.
