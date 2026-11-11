# Non-RL and "cheaper than full RL" training for small LLM agents (SQL / DB / tool-calling), as of Nov 2026

Scope note: the evidence was gathered 2026-11-04. Most numbers below come from arXiv abstracts or HTML full text fetched in this session. Items marked **[recalled, not re-fetched]** are well-known abstract-level claims that I cite from memory of the paper, without re-opening it this session. The report writer should treat those as lower confidence. Items marked **[social/blog only]** come from a single practitioner source that has not been peer reviewed.

## 1. Agent SFT on trajectories: how much data, loss masking, error-recovery and negative examples

### Takeaway
A few hundred verified, task-matched trajectories is the normal data scale for agent SFT that works: FireAct used 500, AgentTuning's DB slice about 538, LIMI 78 at large scale. The two choices that matter most are (a) filtering to successful trajectories only, since unfiltered data can *lower* accuracy, and (b) computing loss on assistant turns only, including the end-of-turn token. Training on error-then-recovery trajectories helps in every paper that tried it, but the gains are modest (about +2 to +6 points). Format and protocol behaviour is learned quickly. Reasoning and grounding are learned more slowly.

### Cited Findings
**Data scale and what a small amount of data buys**
- FireAct (Oct 2023): fine-tuning Llama2-7B on **500 GPT-4 agent trajectories** gave a "77% HotpotQA performance increase". More diverse data, from multiple tasks and prompting methods, improved agents further. — [FireAct, arXiv 2310.05915](https://arxiv.org/abs/2310.05915)
- AgentTuning / AgentLM (Oct 2023): AgentInstruct has **1,866 trajectories filtered from 35,341** instructions across 6 tasks. **Only trajectories with reward r = 1 were kept** (Mind2Web used r ≥ 2/3). The Database task has 178 self-instruct and 360 BIRD-derived trajectories. AgentLM-7B scored **30.6 on the AgentBench Database task, against 0.3 for Llama-2-7B**, beating GPT-3.5 (15.0) and approaching GPT-4 (33.7). — [AgentTuning, arXiv 2310.12823](https://arxiv.org/html/2310.12823)
- AgentTuning also found that agent-only training hurts generalization. For 7B, agent-only scored held-in 1.34 / held-out 0.09 / general 0.22, while a mix with ShareGPT at η = 0.2 scored 1.96 / 0.67 / 0.63. The authors write: "Training solely on agent data, in fact, leads to a decline in generalization performance." — [AgentTuning](https://arxiv.org/html/2310.12823)
- AgentTuning hyperparameters: Llama-2-chat, LR 5e-5 (7B/13B), batch 64, sequence length 4,096, cosine schedule with 2% warm-up, **loss computed only on the model's output**. — [AgentTuning](https://arxiv.org/html/2310.12823)
- LIMI "Less is More for Agency" (Sep 2025): **78 curated samples** reached 73.5% on agency benchmarks, against GLM-4.5 at 45.1%. It reports a 53.7% improvement over models trained on 10k samples. Only very large models were tested. The abstract names no small models. — [LIMI, arXiv 2509.17567](https://arxiv.org/abs/2509.17567)
- APIGen-MT / xLAM-2 (Apr 2025): two phases. First, task blueprints with ground-truth actions, verified by an LLM-reviewer committee. Second, simulated human-agent rollouts. 5K trajectories were released. The models (xLAM-2-fc-r, 1B to 70B) are reported to beat GPT-4o and Claude 3.5 on τ-bench and BFCL, with the smaller models "particularly" strong in multi-turn settings. — [APIGen-MT, arXiv 2504.03601](https://arxiv.org/abs/2504.03601). xLAM-2-70b-fc-r scores 56.2% on τ-bench overall, against GPT-4o at 52.9%. The model card's text gives no 1B/3B/8B numbers. — [xLAM-2 model card](https://huggingface.co/Salesforce/Llama-xLAM-2-8b-fc-r)
- Toucan (Oct 2025): about **1.5M trajectories** synthesized from nearly 500 real MCP servers. Queries came from 5 models and trajectories from 3 teacher models, with rule-based and model-based validation. Toucan-fine-tuned models reportedly beat larger closed models on BFCL V3 and push the Pareto frontier on MCP-Universe. The abstract gives no numbers. — [Toucan, arXiv 2510.01179](https://arxiv.org/abs/2510.01179)
- Early Experience paper (Oct 2025, ICML 2026) imitation-learning baselines show what plain SFT reaches at 3B to 8B. On BFCLv3, IL gave 21.3 (Llama-3.2-3B), 26.7 (Qwen-2.5-7B) and 16.0 (Llama-3.1-8B). On τ-bench, IL gave 24.3 / 33.9 / 35.9. — [Agent Learning via Early Experience, arXiv 2510.08558](https://arxiv.org/html/2510.08558)

**Loss masking**
- TRL `SFTConfig(assistant_only_loss=True)` computes loss only on assistant turns. **It requires the chat template to contain `{% generation %}…{% endgeneration %}` markers.** TRL auto-patches the template only for known families such as Qwen3, so other models need a manual check. Pre-tokenized datasets can instead pass a `labels` column with -100 on excluded tokens, or an `assistant_masks` column. Tool-calling SFT is supported through a `tools` column plus `tool`-role messages. — [TRL SFTTrainer docs (v1.17.1)](https://huggingface.co/docs/trl/sft_trainer)
- TRL docs warn that the EOS token must match the chat template's end-of-turn token. Otherwise "responses [won't] terminate correctly". Set `eos_token` in SFTConfig. — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- In Tiny Aya's tokenizer config (read from the public mlx-community mirror), `eos_token` is `<|END_OF_TURN_TOKEN|>`, and the extra special tokens include `<|START_RESPONSE|>` / `<|END_RESPONSE|>`. — [mlx-community/tiny-aya-global-8bit-mlx](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx)
- Counterpoint, Instruction Modelling (NeurIPS 2024): putting loss on instruction or prompt tokens *as well* reduced overfitting and helped "especially … datasets with lengthy instructions paired with brief outputs" and small datasets. The best case was more than 100% better on AlpacaEval 1.0. — [Instruction Tuning With Loss Over Instructions, arXiv 2405.14394](https://arxiv.org/abs/2405.14394)

**Error-recovery, negative and failure data**
- Agent-FLAN (Mar 2024) found that agent corpora entangle format-following with reasoning, and that models learn different agent capabilities **at different speeds**. Agent tuning can *increase* hallucination. Adding constructed **negative samples** "greatly" reduced hallucination. Llama2-7B was +3.5% over the prior best across agent evals. — [Agent-FLAN, arXiv 2403.12881](https://arxiv.org/abs/2403.12881)
- AgentRefine (Jan 2025, ICLR 2025): "refinement tuning" teaches the model to fix its own erroneous actions from environment observations. It was built from synthesized environments in which a strong LLM corrects errors. It reportedly "significantly outperforms" prior agent-tuning on held-out generalization and is more robust to perturbation. The abstract gives no numbers. — [AgentRefine, arXiv 2501.01702](https://arxiv.org/abs/2501.01702)
- Agent-R (Jan 2025): MCTS finds the first error step in a failed trajectory and splices in a correct continuation from the same parent node, producing revision trajectories. It reports **+5.59%** over baselines and helps avoid loops. — [Agent-R, arXiv 2501.11425](https://arxiv.org/abs/2501.11425)
- EEF "Exploring Expert Failures" (Apr 2025): keeps useful sub-segments, such as plans and key actions, from *failed* GPT-4 expert trajectories and adds them to the RFT data. WebShop win rate was **62% vs 53.6% for RFT** (GPT-4 scored 35.6%). — [EEF, arXiv 2504.13145](https://arxiv.org/abs/2504.13145)
- Practitioner run on Qwen3-4B, τ2-bench: **SFT on unfiltered trajectories lowered greedy Pass@1 from 14.3% to 8.6%**. SFT on filtered/kept teacher trajectories (839 kept, including 266 partial-credit ones scoring ≥ 0.52) reached 27% on the 100-task split. **[social/blog only]** — [jbarnes850, "When sampling beats training" (Jan 2026)](https://jbarnes850.github.io/2026/01/15/when-sampling-beats-training)
- Hugging Face TRL maintainers published a tool-calling SFT of Tiny Aya Global (`bebechien/SimpleToolCalling`, TRL 0.28.dev, Feb 2026). This shows the TRL path works for this model family. No eval numbers were published. — [qgallouedec/tiny-aya-global-tool-calling-SFT](https://huggingface.co/qgallouedec/tiny-aya-global-tool-calling-SFT)

### Inferences
- The project's 235 trajectories (66 with recovery) are in the same range as FireAct (500) and AgentTuning's DB slice (~538), which both produced large jumps from near-zero. Fixing the protocol failures (not ending the turn, fabricating observations) is the kind of format-following that Agent-FLAN reports is learned *fast*. Grounding and error recovery are the slow part, and they are where iterative methods (§2, §3) add value.
- The fabricated-tool-result failure is mostly a **loss-masking and end-of-turn problem**. The model must be trained to emit `<|END_OF_TURN_TOKEN|>` immediately after each `run_sql(...)` call, and it must never be trained on observation text. Tiny Aya is a `cohere2` model, not a family TRL auto-patches, so the `{% generation %}` markers will probably have to be added by hand, or labels pre-tokenized. Print the token-level labels for one trajectory and check that (a) every tool-call turn including its END_OF_TURN token is unmasked, and (b) every observation turn is masked. A silent masking bug (all-masked, or observations included) is the single most likely way this SFT fails.
- The instruction-modelling counterpoint does *not* carry over to agent observations. Training on tool outputs would directly teach the "fabricate observation" behaviour. If prompt-loss regularisation is wanted, apply it only to the system and user question text, not to tool outputs.
- Agent-only data hurt held-out generalization in AgentTuning. If the hidden test uses different databases or question styles, mixing in a small general or tool-calling slice (10 to 20%) is cheap insurance. If dev and test are the same distribution, it is optional.

### Gaps
- No paper isolates "% of trajectories with injected mistake + recovery" as a variable at 3B. The best quantitative evidence for recovery data is Agent-R (+5.59%) and EEF (+8.4 pts vs RFT), neither at 3B on SQL.
- AgentBank (Oct 2024, 50k+ trajectories, 16 tasks) was not re-fetched. Its scaling results are not covered here.
- I could not verify whether Tiny Aya's official chat template supports a native `tool` role or tool-call format. The model card does not mention tool use. — [Tiny Aya Global model card](https://huggingface.co/CohereLabs/tiny-aya-global)

## 2. Rejection-sampling fine-tuning and self-improvement (RFT, STaR, ReST/ReST-EM, expert iteration, SWE-Gym/SWE-smith, Agent Q)

### Takeaway
RFT is the cheapest reliable step after SFT when the reward is verifiable. Sample K rollouts per training task, keep only the verified-correct ones (optionally including teacher-model rollouts), and re-SFT. It reliably beats plain SFT, though modestly: +1.4 to +3.4 points on agent benchmarks including InterCodeSQL, and a large gain on GSM8K. It saturates or overfits after about 2 to 4 rounds. Filtering is essential.

### Cited Findings
- RFT (Yuan et al., Aug 2023): rejection-sampling fine-tuning on GSM8K. **Combining rejection samples from multiple models pushed LLaMA-7B to 49.3%, against SFT at 35.9%.** More *distinct* reasoning paths help more, and RFT helps weaker models more. — [Scaling Relationship on Learning Mathematical Reasoning, arXiv 2308.01825](https://arxiv.org/abs/2308.01825)
- ReST-EM (Dec 2023, TMLR): generate, filter with binary feedback, fine-tune, and repeat "a few times". It significantly outperforms human-data-only SFT on MATH and APPS (PaLM-2) and scales favourably with model size. — [Beyond Human Data, arXiv 2312.06585](https://arxiv.org/abs/2312.06585)
- RFT against SFT on agent tasks (Llama-2-7B, from IPR's Table 2): **InterCodeSQL SFT 54.9 → RFT 56.3**; WebShop 60.2 → 63.6; ALFWorld seen 60.0 → 62.9. Average 60.6 → 62.3. — [IPR, arXiv 2406.11176](https://arxiv.org/html/2406.11176v2)
- SWE-Gym (Dec 2024, ICML 2025): 2,438 real Python tasks. Fine-tuning on sampled agent trajectories gave **up to +19% absolute** resolve rate on SWE-Bench Verified and Lite. Adding verifiers trained on agent trajectories for inference-time selection reached 32.0% on Verified and 26.0% on Lite. — [SWE-Gym, arXiv 2412.21139](https://arxiv.org/abs/2412.21139)
- SWE-smith (Apr 2025): 50k task instances from 128 repos. SWE-agent-LM-32B reached **40.2% pass@1 on SWE-bench Verified** (open-source SOTA at the time). — [SWE-smith, arXiv 2504.21798](https://arxiv.org/abs/2504.21798)
- Agent Q (Aug 2024): guided MCTS plus self-critique plus off-policy DPO on Llama-3-70B. Real booking success went from **18.6% zero-shot to 81.7%** after one day of data collection, and to 95.4% with online search. It outperforms behaviour cloning and RFT on WebShop. — [Agent Q, arXiv 2408.07199](https://arxiv.org/abs/2408.07199)
- EEF (Apr 2025): RFT that also mines good sub-actions from failed expert runs scored 62% against 53.6% for plain RFT on WebShop. — [EEF, arXiv 2504.13145](https://arxiv.org/abs/2504.13145)
- Practitioner pipeline (Qwen3-4B, τ2-bench): SFT on kept teacher trajectories, then GRPO. SFT1 sampled Pass@1 was 29% and Pass@4 60%. GRPO Pass@1 was 36% and Pass@4 59%. The author's reading is that the big gains came from *filtering*, and that RL mainly sharpens Pass@1. **[social/blog only]** — [jbarnes850 blog (Jan 2026)](https://jbarnes850.github.io/2026/01/15/when-sampling-beats-training)
- CacheRL (Jun 2026): Qwen3-4B-Thinking trained with iterative SFT and GRPO on multi-step tool calling reached 92% process accuracy, against GPT-5 at 94%. **Removing knowledge transfer from large models cost 41%**, and RL gave "limited gains beyond strong SFT". Conclusion: "data quality and reward design play a more important role than complex optimization methods". — [CacheRL, arXiv 2606.14179](https://arxiv.org/abs/2606.14179)
- STaR (2022): bootstrapping rationales by keeping only correct generations matched a 30x larger fine-tuned model on CommonsenseQA. **[recalled, not re-fetched]** — [STaR, arXiv 2203.14465](https://arxiv.org/abs/2203.14465)

### Inferences
- For this project, RFT is the natural "stage 2". Use the training-question pool, sample K = 4 to 8 rollouts per question from the SFT checkpoint at T ≈ 0.7 to 1.0 against the real SQLite env, keep verified-correct ones (dedupe by SQL or answer), add the 235 expert trajectories, and re-train **from the base model**. That is the ReST-EM convention; it avoids compounding drift. One to two rounds is the right budget. IPR saw overfitting by round 5, and ReST-EM uses "a few".
- RFT only helps if the SFT model's pass@K is clearly above its pass@1. The blog's 29% → 60% (pass@1 → pass@4) gap is the kind of headroom that makes RFT and RL worth it. Measure pass@4 on dev right after SFT; it is the cheapest go/no-go signal for stage 2.
- The "multiple-model rejection samples" result (RFT 49.3%) argues for also sampling Aya Expanse 8B/32B rollouts on the training questions and keeping only verified ones. This is off-policy distillation (§4), and it is tokenizer-agnostic.

### Gaps
- No SQL-agent RFT result at ~3B with numbers was found. The closest is InterCodeSQL at Llama-2-7B (+1.4 pts).
- The SWE-Gym number of trajectories used and the per-round gains were not visible in the abstract.

## 3. Preference optimisation on trajectories (DPO, KTO, step-level DPO, ETO, IPR, iterative DPO) and failure modes

### Takeaway
Trajectory-level DPO with success-vs-failure pairs from the agent's own exploration (ETO) adds about +2 to +8 points over SFT at 7B. Step-level variants (IPR) add about +6 on InterCodeSQL. **Both depend on keeping an SFT/NLL term on the successful trajectory.** Dropping it nearly halved InterCodeSQL accuracy. On multi-turn data, plain DPO is fragile: it has collapsed within tens of steps, and it shows length bias. It is a worthwhile but riskier third stage.

### Cited Findings
- ETO (Mar 2024, ACL 2024): explore after SFT, pair failures with expert trajectories, DPO, iterate. It "consistently surpasses baseline performance by a large margin". — [ETO, arXiv 2403.02502](https://arxiv.org/abs/2403.02502). On Llama-2-7B, ETO vs SFT: **InterCodeSQL 57.2 vs 54.9**, WebShop 67.4 vs 60.2, ALFWorld seen 68.6 vs 60.0. — [IPR Table 2, arXiv 2406.11176](https://arxiv.org/html/2406.11176v2)
- IPR (Jun 2024, EMNLP 2024): step rewards come from Monte Carlo (N = 5 continuations per step). The loss is outcome-DPO plus step-DPO plus **SFT loss**, with up to 4 iterations. **InterCodeSQL 61.3**, against 57.2 for ETO, 60.2 for Step-PPO, 56.3 for RFT, 54.9 for SFT and 52.4 for PPO. Average 69.4 against SFT's 60.6. — [IPR](https://arxiv.org/html/2406.11176v2)
- IPR ablations (WebShop / InterCodeSQL / ALFWorld-unseen): without step-DPO 66.4 / 58.0 / 70.2. **Without SFT loss 61.8 / 31.7 / 64.9**, the largest drop, with InterCodeSQL nearly halved. Performance dropped by iteration 5 (overfitting). — [IPR](https://arxiv.org/html/2406.11176v2)
- IPR cost on 8×A100: WebShop for 3 iterations took SFT ~1 h, ETO ~2.5 h, IPR ~5.3 h. — [IPR](https://arxiv.org/html/2406.11176v2)
- Early Experience compared DPO as a baseline on Llama-3.1-8B (WebShop / ALFWorld): IL 47.3 / 80.5, **DPO 53.1 / 82.8**, IWM 58.6 / 85.9, SR 58.2 / 85.2. The authors note that **"DPO collapsed within tens of steps."** STaR-style self-training *underperformed* IL (25.0 / 74.2), and long-CoT distillation collapsed (0.0 / 25.8). — [Early Experience, arXiv 2510.08558](https://arxiv.org/html/2510.08558)
- DMPO (Jun 2024): multi-turn DPO suffers a **length-dependent bias**, because won and lost trajectories differ in length. Without length normalisation, performance degrades as the length gap grows. DMPO adds length normalisation and a state-action occupancy constraint. — [DMPO, arXiv 2406.14868](https://arxiv.org/pdf/2406.14868)
- Iterative RPO (Apr 2024): iterative DPO on correct-vs-incorrect CoT **plus an NLL term** took Llama-2-70B-Chat on GSM8K from 55.6% to 81.6%. The NLL term was crucial. **[recalled, not re-fetched]** — [Iterative Reasoning Preference Optimization, arXiv 2404.19733](https://arxiv.org/abs/2404.19733)
- KTO (Feb 2024): uses only unpaired "desirable / undesirable" labels and matches or exceeds DPO from 1B to 30B. This fits naturally with binary verified rewards when pairs are unbalanced. **[recalled, not re-fetched]** — [KTO, arXiv 2402.01306](https://arxiv.org/abs/2402.01306)
- Likelihood displacement: DPO can *lower* the probability of the preferred response, shifting mass to unrelated outputs. **[recalled, not re-fetched]** — [Razin et al., arXiv 2410.08847](https://arxiv.org/abs/2410.08847)

### Inferences
- If stage 3 is attempted, use **ETO-style DPO with an NLL term on the chosen trajectory**. TRL `DPOTrainer` supports an RPO-style NLL weight. Build pairs from the same question (verified-success vs failed rollout) and use a low β (0.1). Train 1 epoch from the RFT checkpoint and keep the reference model equal to that checkpoint. Expect +2 to +6 points at best. Watch for collapse every few steps on dev; Early Experience saw collapse within tens of steps.
- KTO is attractive when most questions have only failures or only successes, which is likely at 3B. It avoids having to construct pairs.
- Given only ~10 days and T4 GPUs, preference optimisation has a worse gain-to-risk ratio than RFT. Rank it below RFT and above full online RL.

### Gaps
- No trajectory-DPO or KTO result at ~3B on an SQL agent was found. The closest is Llama-2-7B on InterCodeSQL.
- TRL's handling of multi-turn, masked-observation DPO (whether `DPOTrainer` masks tool outputs in conversational chosen/rejected) was not verified this session.

## 4. Distillation: off-policy (teacher trajectories) vs on-policy (GKD/MiniLLM/Thinking Machines OPD), and Aya Expanse → Tiny Aya tokenizer compatibility

### Takeaway
On-policy distillation is the most compute-efficient post-training method reported in 2025: about 9 to 30x cheaper than SFT and about 10x cheaper than RL in Thinking Machines' accounting. **It assumes a shared tokenizer, however, and Tiny Aya does not share Aya Expanse's tokenizer.** I measured a vocabulary Jaccard of 0.316, with only 2 shared strings at the same ID. Cross-tokenizer variants exist (HF GOLD; Alibaba Oct 2026), but they are experimental, need about 1k steps of warm-up, and are capped by teacher quality. Aya Expanse 8B scores only 12.5% on this task. For this project, **off-policy distillation (verified teacher trajectories → SFT/RFT) is the practical form.**

### Cited Findings
- Thinking Machines "On-Policy Distillation" (Oct 27, 2025, Kevin Lu et al.): sample from the student, score each token with teacher log-probs, and use per-token reverse KL as the advantage (discount 0). On AIME'24, starting from a 400K-prompt SFT checkpoint at 60%, it reached ~70% in ~150 steps. Reported cost: **~9x cheaper than SFT** when data exists and **~30x** when teacher sampling is counted. It cites Qwen3's report of RL at 67.6% using 17,920 GPU-h, against OPD at 74.4% using ~1,800 GPU-h. A single prompt for 20 steps roughly matched the teacher. Limitations: reverse KL cannot add mass where the student has none, so **SFT is needed first** when the teacher's behaviour is out of the student's support. Only sampled-token distillation was tested. In the personalization experiment, SFT on the model's own samples degraded IF-eval. — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- The blog itself does not discuss tokenizer requirements, but the method needs the teacher to score the student's exact token sequence. — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
- HF GOLD (Oct 29, 2025; byte-offset alignment update June 2026) is General On-Policy Logit Distillation. It extends ULD to on-policy training across **different tokenizers** by merging token spans and mapping shared vocab. Results on Countdown: Qwen3-4B → Llama-3.2-1B (token-content similarity 0.64) recovered **60% of teacher performance**, against 10% for ULD. Qwen3-4B → gemma-3-1b (similarity 0.063) scored 0.03, which is near failure. Same-family pairs recovered more than 80%. Cross-tokenizer GOLD beat GRPO by 20%. Limits: needs ~1,000 steps before gains appear, the student is capped by the teacher, it lives in `trl.experimental`, and the evidence comes mainly from one task. — [HF "Unlocking On-Policy Distillation for Any Model Family"](https://huggingfaceh4-on-policy-distillation.hf.space/)
- Alibaba (Oct 6, 2026), "Rethinking Cross-Tokenizer On-Policy Distillation": strict 1:1 aligned positions cover **85.6 to 97.0% of response tokens even when vocab Jaccard is only 0.39 to 0.65**. Reverse KL on strict positions restricted to the student's top-16 shared tokens keeps ≥96% of the gain. Adding loss on mismatched spans *hurt* in all 18 settings. Qwen2.5-7B → Llama-3.2-3B: base 26.96% → 32.64% (top-16) / 32.86% (full strict). Tested on math and code only, not agents. — [arXiv 2610.08448 (alphaXiv)](https://www.alphaxiv.org/abs/2610.08448)
- **Tokenizer check, my own measurement.** Configs and tokenizer.json files came from public mlx-community mirrors, because the official CohereLabs repos are gated:
  - Tiny Aya Global: `model_type: cohere2`, `vocab_size: 262144` (261,010 actual entries), BOS id 2, `<|END_OF_TURN_TOKEN|>` id 6. — [mlx-community/tiny-aya-global-8bit-mlx](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx)
  - Aya Expanse 8B: `model_type: cohere`, `vocab_size: 256000` (255,029 entries), BOS id 5, `<|END_OF_TURN_TOKEN|>` id 255001. — [mlx-community/aya-expanse-8b-4bit](https://huggingface.co/mlx-community/aya-expanse-8b-4bit)
  - Both are byte-level BPE with individual-digit splitting and use the same chat special-token *strings*. **123,916 token strings are shared (Jaccard 0.316; about 47 to 49% of each vocab), but only 2 shared strings have the same ID.** Same-ID GKD / Thinking Machines-style OPD is therefore impossible without a cross-tokenizer method.
  - The Tiny Aya report confirms a new 262k-vocab tokenizer trained on 50GB of Fineweb-2. Its post-training teachers were Gemma3-27B, Command A and DeepSeek-V3, not Aya Expanse. — [Tiny Aya report, arXiv 2603.11510](https://arxiv.org/html/2603.11510)
- GKD (generalized KD on student-generated sequences) and MiniLLM (reverse KL) are the academic precursors. TRL ships `GKDTrainer`, which assumes a shared tokenizer. **[recalled, not re-fetched]** — [GKD, arXiv 2306.13649](https://arxiv.org/abs/2306.13649); [MiniLLM, arXiv 2306.08543](https://arxiv.org/abs/2306.08543)
- Teacher quality matters. CacheRL found that removing large-model knowledge transfer cost 41%. — [CacheRL, arXiv 2606.14179](https://arxiv.org/abs/2606.14179)

### Inferences
- Aya Expanse 8B scores 12.5% on this task, so it is a *weak* teacher. Logit-level distillation from it would cap Tiny Aya below that level on most questions. Aya Expanse 32B's success rate is unknown and should be measured first; ~20 dev questions through HF Inference is enough.
- Highest value per GPU-hour: run the strongest allowed teacher (32B) on the *training* questions with K samples, keep **only verified-correct** trajectories, and add them to the SFT/RFT pool. This is off-policy, tokenizer-agnostic, and needs no teacher on the T4s at training time.
- On-policy cross-tokenizer distillation (GOLD) would need a 32B teacher resident during training. On 2×T4 (32GB total) that means a 4-bit 32B teacher (~18 to 20GB) plus a 3.35B student. It needs ~1k steps of warm-up and has unproven agent results. Expected gain per GPU-hour is low here; treat it as an explicit "not doing" item.
- Strict-position coverage for SQL-heavy text between these two tokenizers is unmeasured. Given the shared byte-level BPE and digit splitting, it is probably in the 85 to 97% range Alibaba saw, but that is unverified.

### Gaps
- I did not measure strict 1:1 token alignment on real trajectories, because the `tokenizers` Python package is not installed locally. A 10-line script on Kaggle would settle it.
- Aya Expanse 32B's tokenizer was not checked directly; its repo is gated. It is presumably the same as 8B's (same family), but unverified.
- No published on-policy distillation result for multi-turn tool or SQL agents at ~3B was found.

## 5. Newer 2025–2026 methods (Early Experience, SFT-then-RL debate, LoRA for RL, tiny-data RL, test-time scaling): which have real evidence for small models

### Takeaway
The best-evidenced newer method at 3B to 8B is **Early Experience** (Meta and collaborators, Oct 2025; ICML 2026). It needs no reward and gives about +5 to +6.5 points over imitation learning across 8 agent environments, and it makes a better RL starting point. For the SFT-vs-RL question, the most controlled 2026 study finds **LoRA-SFT wins in-distribution for 0.6B to 32B tool-calling (15 of 18 settings)**, while RL wins transfer only marginally (<1 point). Tiny-data RL (1-shot RLVR, SkyRL-SQL's 653 examples) is real, but it is full RL and needs rollout infrastructure.

### Cited Findings
- Early Experience (arXiv 2510.08558, Oct 9, 2025; v3 May 2026; ICML 2026). The policy samples K alternative actions at expert states and executes them, and the resulting next states supervise (a) **implicit world modelling (IWM)**, which predicts next states, and (b) **self-reflection (SR)**, a CoT contrasting the alternative with the expert action. Models were Llama-3.2-3B, Qwen-2.5-7B and Llama-3.1-8B. Gains over IL (3B / 7B / 8B): BFCLv3 IWM +4.0 / +2.6 / +4.0 and SR +8.0 / +5.3 / +4.0; τ-bench SR +4.4 / +5.6 / +5.8; WebShop IWM +18.4 / +4.6 / +11.3. Averaging across all 24 cells gives about +5.0 (IWM) and +6.5 (SR), my own calculation from the table. SR works best with small K (2 to 4). On WebShop, 1/8 of the demonstrations plus early experience beat full-data IL. As GRPO initialization, IWM and SR checkpoints ended above IL-initialized GRPO, and the gap never reversed. Compute was "at most 8 H100". — [Early Experience](https://arxiv.org/html/2510.08558)
- "SFT or RL for Tool-Calling LLM Agents? A Controlled Study" (NeurIPS 2026 SLM-Agents workshop): six Qwen3 models, 0.6B to 32B. **LoRA-SFT is best in-distribution in 15 of 18 settings.** GRPO wins 29 of 54 cross-dataset settings, but by <1 point on average. **SFT+GRPO is rarely the strongest.** LoRA beats full FT, which the authors attribute to better preservation of pretrained agentic behaviour. Dataset mixing gives strong transfer. — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/171173) / [OpenReview](https://openreview.net/forum?id=pKlPWwGqgM)
- "SFT Memorizes, RL Generalizes" (Jan 2025): RL generalizes better to rule and visual variants, but **SFT is still needed to stabilize output format** before RL. — [arXiv 2501.17161](https://arxiv.org/abs/2501.17161)
- LoRA Without Regret (Thinking Machines, Sep 29, 2025): LoRA matches full FT for small-to-medium SFT and for RL, **with rank 1 sufficient for RL** (~1 bit per episode). The **optimal LoRA LR is ~10x full-FT LR**, and ~15x for runs under ~100 steps (anecdotal). LoRA must go on **all layers, especially MLP**; attention-only is much worse. LoRA is **less tolerant of large batch sizes**, and it uses about 2/3 of full-FT FLOPs. — [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)
- 1-shot RLVR (Apr 2025): a single example took Qwen2.5-Math-1.5B MATH500 from 36.0% to **73.6%**, matching a 1.2k-example set. Similar results held on Llama-3.2-3B-Instruct and others. **An entropy bonus to keep exploration was critical.** — [arXiv 2504.20571](https://arxiv.org/abs/2504.20571)
- SkyRL-SQL (NovaSky, May 2025): multi-turn RL (the model queries the DB and refines over turns) on **653 training samples**, Qwen2.5-Coder-7B. It is reported to beat GPT-4o and o4-mini on Spider variants by up to 9.2% and to beat OmniSQL-7B, which was SFT on 2.5M samples. Multi-turn reached 60% reward 2.8x faster than single-turn and ended +16% higher after 35 steps. Built on VeRL and Search-R1. *(These numbers come from search snippets of the release notes and blog; the Notion blog did not render for me.)* — [SkyRL-SQL blog (Notion)](https://novasky-ai.notion.site/skyrl-sql); [SkyRL repo listing](https://awesome.ecosyste.ms/projects/github.com%2FNovaSky-AI%2FSkyRL); [NeurIPS 2025 talk "Train SQL Data Agent with Multi-Turn RL"](https://neurips.cc/virtual/2025/128018)
- Multi-turn SQL RL follow-ups in 2026. SQL-Trail (Jan 2026) uses an adaptive turn budget and composite reward; it reports up to 18x better data efficiency than single-pass RL, and its 7B/14B models beat larger proprietary systems by ~5% on average. — [arXiv 2601.17699](https://arxiv.org/abs/2601.17699). Progress-SQL (Jun 2026) uses progressive rewards plus an execution-status reward for recovering from invalid SQL, and criticises last-turn-only rewards. — [arXiv 2606.06825](https://arxiv.org/abs/2606.06825)
- Test-time scaling and sampling. In the practitioner run, Pass@4 of SFT alone (60%) equalled GRPO's Pass@4 (59%), but capturing that in deployment requires a verifier or selector. **[social/blog only]** — [jbarnes850 blog](https://jbarnes850.github.io/2026/01/15/when-sampling-beats-training). OmniSQL-7B majority voting adds +2.2 on BIRD dev (63.9 → 66.1). — [OmniSQL](https://arxiv.org/html/2503.02240). SWE-Gym verifier reranking reached 32.0% on Verified. — [SWE-Gym](https://arxiv.org/abs/2412.21139)
- RL instability for multi-turn agents (the "Echo Trap" in RAGEN, Apr 2025): reward-variance collapse and gradient spikes in multi-turn agent RL. **[recalled, not re-fetched]** — [RAGEN, arXiv 2504.20073](https://arxiv.org/abs/2504.20073)

### Inferences
- **Early Experience's self-reflection variant** is the best-supported new "no-reward" add-on for this project. The 66 injected-mistake trajectories are already a hand-built version of it. Generating more SR data programmatically (wrong SQL or column → real error message → reflection → correct call) is cheap because the environment is a local SQLite DB.
- **Caution on IWM for this project.** IWM trains the model to *predict observations*, and fabricating observations is one of Tiny Aya's main failure modes. If IWM is used at all, it must use a clearly different prompt format (e.g., "Predict the result of this query:") so it never appears inside an agent turn. Prefer SR.
- The 2026 controlled study and CacheRL both say a strong, filtered LoRA-SFT captures most of the in-distribution gain, and that RL mostly polishes Pass@1. Since the hackathon is scored on a fixed distribution with verified answers, SFT+RFT should capture most of the achievable gain. Treat RL as the last stage, only if the pipeline and decision rule pass early.
- If RL is attempted, LoRA-Without-Regret supports a tiny LoRA rank (even r = 1 to 8), which saves memory on T4. 1-shot RLVR and SkyRL-SQL support a small question set, provided the task pass@K is non-zero after SFT.

### Gaps
- No Early Experience result on SQL or database environments; the closest are BFCLv3 and τ-bench.
- SkyRL-SQL's GPU count and hours were not obtained; the Notion page did not render.
- I found no peer-reviewed evidence for 2026 X/Twitter claims about "agentic RL at 3B on a single consumer GPU". I did not find credible sources and so report none.

## 6. Text-to-SQL small-model evidence (OmniSQL, CodeS, Arctic-Text2SQL-R1, SLM-SQL, DIN-SQL-style distillation)

### Takeaway
Text-to-SQL is the best-studied verifiable-reward domain for small models. Large synthetic SFT (OmniSQL-7B, 2.5M samples) beats GPT-4o on average. Small models (0.5B to 1.5B) reach 57 to 67% BIRD-dev with SFT followed by RL. RL works best **from a strong SFT checkpoint with a simple execution reward**, and finer-grained reward shaping caused "lazy" behaviour. These are single-shot NL→SQL results, not multi-turn agents.

### Cited Findings
- OmniSQL (Mar 2025; v2 Jul 2025): SynSQL-2.5M covers 2.5M samples over 16k+ synthetic DBs with CoT. The models are Qwen2.5-Coder-7B/14B/32B-Instruct, with 7B and 14B fully fine-tuned and **32B with LoRA r = 256, α = 512**. Hyperparameters: LR 2e-5 (7B) / 4e-6 (14B) / 2e-4 (32B LoRA), batch 512, 2 epochs, context 8,192. **OmniSQL-7B greedy / majority:** Spider dev 81.2 / 81.6, Spider test 87.9 / 88.9, BIRD dev 63.9 / 66.1, 9-dataset average 61.2 / 63.1. Comparators: GPT-4o 58.9 / 59.9, DeepSeek-V3 59.9 / 60.7, Qwen2.5-Coder-7B base 52.8 / 58.4. Performance rises steadily with data fraction. — [OmniSQL, arXiv 2503.02240](https://arxiv.org/html/2503.02240)
- Arctic-Text2SQL-R1 (Snowflake, May 2025; v2 Jan 2026): GRPO with **reward 1 for an execution match, 0.1 for executable SQL, 0 otherwise**, using 16 rollouts, T = 0.8 and KL 0.001. Data: filtered BIRD (8,017), Spider (6,972 + 985) and 11,811 filtered synthetic samples; empty-result and >5s queries were removed. **Starting from OmniSQL SFT checkpoints beat starting from base.** The 7B scored BIRD dev 68.9 and Spider test 88.8. Things that did not help: PPO vs GRPO, unfiltered synthetic data (hurt), LLM augmentation, more rollouts, and finer-grained rewards, which caused "lazy" behaviour. — [Arctic-Text2SQL-R1, arXiv 2505.20315](https://arxiv.org/html/2505.20315)
- SLM-SQL (Jul 2025): **0.5B to 1.5B** models, SFT on SynSQL-Think-916K and then RL, with corrective self-consistency at inference. **BIRD dev 56.87% (0.5B) and 67.08% (1.5B)**, an average +31.4 pts over base across five models. — [SLM-SQL, arXiv 2507.22478](https://arxiv.org/abs/2507.22478)
- AgentTuning's Database slice includes 360 trajectories derived from BIRD, and AgentLM-7B reached 30.6% on AgentBench-DB. This is the closest *agentic* (multi-turn) DB SFT result at 7B. — [AgentTuning](https://arxiv.org/html/2310.12823)
- InterCodeSQL (interactive SQL with execution feedback) at Llama-2-7B: SFT 54.9 → RFT 56.3 → ETO 57.2 → Step-PPO 60.2 → IPR 61.3. — [IPR](https://arxiv.org/html/2406.11176v2)
- Multi-turn agentic SQL RL at 7B: SkyRL-SQL (653 samples) — see §5. MARS-SQL (Nov 2025) is a multi-agent RL framework for text-to-SQL; only the title was seen. — [MARS-SQL, arXiv 2511.01008](https://arxiv.org/pdf/2511.01008)

### Inferences
- The "lazy" result from fine-grained rewards matters for this project's RL stage, if one happens. Keep the verifiable answer reward simple (match = 1). If shaping is needed, give a small credit for valid protocol or executable SQL, not intermediate SQL similarity.
- SynSQL-2.5M and OmniSQL data are single-shot NL→SQL with CoT, not tool-calling trajectories. They could be converted cheaply: wrap (schema → SQL → result → ANSWER) into the project's 3-tool format for extra synthetic SFT data covering grounding. This is a data option, not evidence; whether it helps at 3.35B is untested.
- OmniSQL used 2 epochs, matching the project plan, and LoRA LR 2e-4 for its 32B. That is consistent with LoRA-Without-Regret's ~10x rule.

### Gaps
- CodeS (Feb 2024) and DIN-SQL-to-small-model distillation (e.g., DTS-SQL) were not re-fetched; they are older (2024) and superseded by OmniSQL and Arctic for small-model SQL.
- No multi-turn SQL-agent result specifically at ~3B was found. SLM-SQL is single-shot at 0.5B to 1.5B, and SkyRL-SQL and Arctic are 7B.

## 7. Hyperparameters for small-data agent SFT (epochs, LoRA rank, LR, packing) and overfitting / format-collapse risks

### Takeaway
For ~235 trajectories with QLoRA on Tiny Aya: put LoRA on **all linear layers including MLP**, use rank 16 to 64, LR ~2e-4 to 3e-4 (≈10 to 15x the full-FT LR; Tiny Aya's own SFT used 2.5e-5), a **small effective batch** (8 to 16), 2 to 3 epochs, and select the checkpoint by dev *task success*, not loss. The main risks are silent masking bugs, overfitting to the 235 templates (memorised SQL), loss of sampling diversity (which starves RFT/RL), and DPO collapse.

### Cited Findings
- LoRA Without Regret: LoRA LR ≈ 10x full-FT LR (≈15x for runs <100 steps, anecdotal). The rate is largely rank-independent. Use all layers including MLP; attention-only is clearly worse. LoRA is less tolerant of large batches, and rank does not fix this. Capacity only binds on large SFT datasets. — [Thinking Machines](https://thinkingmachines.ai/blog/lora/)
- Tiny Aya's own post-training: SFT for **3 epochs at peak LR 2.5e-5** (full FT), 36 layers, hidden 2048, 3.35B params, 8K context. — [Tiny Aya report](https://arxiv.org/html/2603.11510)
- TRL defaults: LoRA adapters "typically use a higher learning rate (≈1e-4)". SFTConfig defaults to 3 epochs, LR 2e-5 and max_length 1024. **Watch the max_length=1024 default**: multi-turn SQL trajectories with schemas easily exceed it and are truncated with `keep_start`, which cuts off the final ANSWER turn. `padding_free` is supported only with FlashAttention 2/3. Packing uses best-fit decreasing. — [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- OmniSQL used 2 epochs (7B full FT LR 2e-5; 32B LoRA r256 LR 2e-4). — [OmniSQL](https://arxiv.org/html/2503.02240)
- AgentTuning used LR 5e-5 (full FT), batch 64, seq 4,096 and warmup 2%. — [AgentTuning](https://arxiv.org/html/2310.12823)
- IPR used 3 SFT epochs for warm-up; iterative stages overfit by round 5. — [IPR](https://arxiv.org/html/2406.11176v2)
- 2026 controlled study: **LoRA > full FT** for small tool-calling agents, because it preserves pretrained agentic behaviour. — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/171173)
- Collapse modes: DPO collapsed within tens of steps, and long-CoT distillation and STaR underperformed IL. — [Early Experience](https://arxiv.org/html/2510.08558). 1-shot RLVR needed an entropy bonus to keep exploration. — [arXiv 2504.20571](https://arxiv.org/abs/2504.20571). SFT on narrow domain data degrades other capabilities: Qwen3-4B Codeforces SFT dropped IFEval from 83.4 to 79.5. — [HF GOLD post](https://huggingfaceh4-on-policy-distillation.hf.space/). Thinking Machines saw IF-eval drop from 85 to 45 under heavy domain mid-training. — [Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/)
- Instruction-modelling paper: for small datasets, overfitting is the dominant failure, and extra loss on prompt tokens reduced it. — [arXiv 2405.14394](https://arxiv.org/abs/2405.14394)

### Inferences
- **Concrete starting config.** QLoRA 4-bit with fp16 compute (T4 has no bf16). r = 16 to 32 with α = 2r, on all linear projections (q, k, v, o, gate, up, down). LR 2e-4, cosine, warmup 3 to 5%. Effective batch 8 to 16, 2 to 3 epochs (about 30 to 90 optimizer steps for 235 examples). max_length ≥ 4096, and check that no trajectory is truncated. No packing at first: correctness of masking matters more than throughput at this scale. Evaluate dev success at the end of each epoch and keep the best one.
- r = 16 is fine. With 235 examples, LoRA capacity is not the binding constraint according to LoRA-Without-Regret. Moving to all-MLP coverage and a higher LR matters more than raising rank.
- **Template-memorisation risk.** All 235 trajectories are programmatic, so the model may learn the *template* rather than grounding (e.g., always the same column names). Mitigations: diversify phrasing of reasoning lines, vary schema exploration order, hold out whole *databases* (not just questions) in dev, and track a "hallucinated column/table name" rate as a diagnostic.
- **Diversity for later stages.** Check that pass@4 on dev at T = 0.8 is clearly above greedy pass@1 after SFT. If SFT makes sampling near-deterministic, RFT and RL get no new successes. If so, lower the epochs or LR rather than adding RL.

### Gaps
- No published LR or epoch sweep exists for agent SFT at ~3B with ~200 trajectories. The settings above are extrapolations from LoRA-Without-Regret, Tiny Aya's own SFT, OmniSQL and AgentTuning.
- FlashAttention 2 does not support T4 (Turing) GPUs, so TRL `padding_free` is unavailable there. This comes from general knowledge, not a source fetched this session.

## 8. Verdict: best expected gain per GPU-hour in ~10 days, as a staged recipe

### Takeaway
Ranking by expected gain per GPU-hour on 2×T4 with a ~$20 to 30 HF budget and 10 days:
1. Correct assistant-only-loss SFT, with verified masking and end-of-turn.
2. Verified teacher trajectories from Aya Expanse 32B, as off-policy distillation.
3. 1 to 2 rounds of RFT on the student's own verified rollouts, with self-reflection/recovery data.
4. Optionally, a single DPO or KTO-with-NLL pass on success/failure pairs.
5. Online multi-turn GRPO only if time remains.

Logit-level on-policy distillation is **not recommended**. The tokenizers differ, the teacher is weak, and it needs heavy GPU memory.

### Cited Findings
- SFT with LoRA is the strongest in-distribution method for small tool-calling agents, and SFT+GRPO is rarely best. — [NeurIPS 2026 poster](https://neurips.cc/virtual/2026/171173)
- RL gave "limited gains beyond strong SFT", while teacher knowledge transfer was worth 41%. — [CacheRL](https://arxiv.org/abs/2606.14179)
- Filtering is decisive: unfiltered SFT went 14.3 → 8.6, while filtered SFT then GRPO reached 27 → 33 to 36. **[social/blog only]** — [jbarnes850 blog](https://jbarnes850.github.io/2026/01/15/when-sampling-beats-training)
- On interactive SQL (Llama-2-7B), the ladder runs SFT 54.9 → RFT 56.3 → ETO 57.2 → IPR 61.3, and IPR without its SFT loss falls to 31.7. — [IPR](https://arxiv.org/html/2406.11176v2)
- Early Experience self-reflection averages about +6.5 over IL at 3B to 8B, and makes a better RL initialisation. — [Early Experience](https://arxiv.org/html/2510.08558)
- On-policy distillation is about 10x cheaper than RL *when the tokenizer is shared and the teacher is strong*. — [Thinking Machines](https://thinkingmachines.ai/blog/on-policy-distillation/). Cross-tokenizer GOLD recovers about 60% of the teacher at token-similarity 0.64 and nearly fails at 0.06. — [HF GOLD](https://huggingfaceh4-on-policy-distillation.hf.space/). Tiny Aya and Aya Expanse vocab Jaccard is 0.316, with 2 same-ID tokens (own measurement from the [Tiny Aya mirror](https://huggingface.co/mlx-community/tiny-aya-global-8bit-mlx) and [Aya Expanse mirror](https://huggingface.co/mlx-community/aya-expanse-8b-4bit)).
- Multi-turn RL on ~650 SQL questions works at 7B (SkyRL-SQL). — [SkyRL-SQL](https://novasky-ai.notion.site/skyrl-sql). The reward must stay simple (execution match), because fine-grained shaping caused lazy behaviour. — [Arctic-Text2SQL-R1](https://arxiv.org/html/2505.20315)

### Inferences
Proposed staged recipe. GPU-hour figures are my rough estimates for a 3.35B QLoRA model on T4 and are not sourced.
- **Stage 0 (days 1–2, ~1–2 T4-h): SFT as planned, with fixes.**
  - Verify the label mask token by token. The END_OF_TURN token after each tool call must be in the loss, and observations must not be.
  - Raise max_length so nothing is truncated, and put LoRA on all linear layers.
  - Use LR 2e-4, a small batch, and 2 to 3 epochs. Keep the checkpoint by dev success.
  - Apply the pre-registered rule (≥12/40). Also record **pass@4 at T = 0.8** and the protocol-error rate.
- **Stage 0b (parallel, HF credits): teacher harvest.** Run Aya Expanse 32B, and 8B if cheaper, through the same env on the training questions with K = 4 samples each. Keep only verified-correct trajectories. This is off-policy distillation and the "multiple-model" RFT trick. Low cost, tokenizer-agnostic.
- **Stage 1 (days 3–5, ~3–6 T4-h, mostly generation): RFT round 1.**
  - Sample K = 4 to 8 rollouts per training question from the SFT model.
  - Keep verified successes, plus self-reflection/recovery segments mined from failures where a later step fixed an error (EEF / Agent-R style).
  - Retrain from base on expert, teacher and self data. Run round 2 only if dev improves by ≥3 successes out of 40.
- **Stage 2 (days 6–7, optional, ~2–4 T4-h): preference pass.** Use DPO with an NLL term (or KTO) on same-question success vs failure rollouts: β ≈ 0.1, 1 epoch, a small LR (≈5e-6 to 2e-5 for LoRA, i.e. well below the SFT LR), and evaluation every ~10 steps with rollback on collapse. Skip it if Stage 1 gives no headroom.
- **Stage 3 (days 7–9, stretch): multi-turn GRPO** with a tiny LoRA rank and a simple match reward, only if Stage 1 shows pass@K well above pass@1 and the rollout infrastructure is already stable.
- **Day 10:** freeze, run the final eval, and write up. Report every stage against the pre-registered rule; this also counts toward the safety/alignment framing.
- Expected size of gains, from the analogues above (low confidence at 3.35B): Stage 0 takes the model from ~0% to whatever fixing the protocol unlocks, which is the largest jump. RFT adds +1 to +5 points. DPO-with-NLL adds +1 to +4. RL adds +2 to +7 on Pass@1 but costs the most.
- Self-consistency or majority vote over N agent rollouts is the cheapest "test-time scaling" lever (OmniSQL +2.2; Pass@4 ≫ Pass@1). Use it only if the hackathon's evaluation protocol allows the submitted agent to run multiple internal attempts. Check the rules.

### Gaps
- No direct evidence exists for any of these stages on Tiny Aya or on a 3B multi-turn SQLite agent. All effect sizes are transferred from 7B to 8B analogues or non-SQL agents.
- Actual T4 throughput for Tiny Aya (cohere2) generation is unmeasured, including whether vLLM supports the `cohere2` architecture on T4 in fp16. That will decide whether RFT sampling takes hours or a day.
- Aya Expanse 32B's success rate on this task is unknown, and so is the HF-credit cost per trajectory. Both are needed before committing to the teacher-harvest stage.
