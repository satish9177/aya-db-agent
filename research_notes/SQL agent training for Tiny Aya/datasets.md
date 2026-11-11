# Public datasets for training and evaluating a small multi-turn SQL / database tool-calling agent (Tiny Aya / Aya Expanse), as of Nov 2026

Scope: (a) text-to-SQL with executable DBs, (b) multi-turn/agentic SQL trajectories, (c) general tool/function calling, (d) external evaluation benchmarks, plus licence/contamination analysis and a recommended shortlist. Research was done mainly through Hugging Face Hub dataset cards (via the HF MCP connector), file listings, arXiv papers and project pages. Repo IDs, sizes and licences below are taken from the HF cards and file listings as of 2026-11-04.

## (a) Text-to-SQL datasets with executable databases (Spider, BIRD, Spider 2.0, SynSQL/OmniSQL, SParC/CoSQL, WikiSQL, BIRD-Interact/Critic, LiveSQLBench)

### Takeaway
The useful executable sources are BIRD (train-filtered and dev/mini-dev, CC-BY-SA-4.0), Spider 1.0 (CC-BY-SA-4.0) and SynSQL-2.5M (Apache-2.0). SynSQL is the only large one with permissive licensing that keeps its SQLite databases on the HF repo itself, as a 54 MB `databases.zip`. Spider and BIRD put only question/SQL rows on HF; their SQLite files come from Google Drive, Aliyun or Docker. The BIRD team's 2025–2026 releases (bird23-train-filtered, a re-reviewed dev set, a 2026 mini-dev revision, mini-interact with SQLite DBs, LiveSQLBench-SQLite) are the most current. All of them are CC-BY-SA-4.0.

### Cited Findings
**Spider 1.0**
- `xlangai/spider`: CC-BY-SA-4.0. The repo holds only `spider/train-*.parquet` (~831 KB) and `validation-*.parquet` (~126 KB), with no SQLite files — [HF card](https://huggingface.co/datasets/xlangai/spider); [file listing via HF MCP ls](https://huggingface.co/datasets/xlangai/spider/tree/main)
- Spider has 10,181 questions, 5,693 unique SQL queries, 200 databases and 138 domains under CC BY-SA 4.0. The data, including SQLite DBs, is a single Google Drive download. The Spider 1.0 test set was released in Feb 2024, and the leaderboard no longer accepts submissions — [Spider site](https://yale-lily.github.io/spider)

**SParC / CoSQL (conversational, built on Spider DBs)**
- SParC is CC BY-SA 4.0, with 4,298 question sequences, 12k+ questions and 200 DBs across 138 domains. It is built on Spider and distributed via Google Drive. Its test set is unreleased — [SParC site](https://yale-lily.github.io/sparc)
- Neither `xlangai/sparc` nor `xlangai/cosql` exists on HF, and a hub search for "sparc cosql" returned nothing — [HF hub search, via MCP] (no URL; negative result)

**BIRD**
- `birdsql/bird23-train-filtered`: CC-BY-SA-4.0. It keeps **6,601 of 9,428** original BIRD train examples after a quality check and ships as a JSONL plus `train_column_meaning.json`. The databases are not on HF; they come from `https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip`. In an SFT check with **Qwen2.5-3B** (base), the base model scored 26.2 EX on Mini-Dev and 31.88 on Dev. After training on the original train set it scored 45.4 / 50.46, and on the filtered set 46.0 / 50.0 — [HF card](https://huggingface.co/datasets/birdsql/bird23-train-filtered)
- `birdsql/bird_sql_dev_20251106`: CC-BY-SA-4.0. This is a re-reviewed BIRD dev release from 2025-11-06 with refined questions and evidence and corrected SQL. Each row has `question_id, db_id, question, evidence, SQL, difficulty`. The databases come from a Google Drive "BIRD Dev Complete Package" — [HF card](https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106)
- `birdsql/bird_mini_dev` (2026 update, last modified 27 Oct 2026): CC-BY-SA-4.0, with 500 aligned question IDs each for SQLite, MySQL and PostgreSQL. The authors write that "Our aim is a benchmark that requires an agent to understand the request, explore the permitted database and construct the query", and they removed evidence "that supplied unnecessary schema links, query plans or answers". DB images are on Docker Hub (`3036197201/bird-mini-dev-sqlite`), and interactive tasks run on Harbor Hub — [HF card](https://huggingface.co/datasets/birdsql/bird_mini_dev)
- The original 2025-07-04 Mini-Dev was 500 pairs over 11 dev DBs: debit_card_specializing, student_club, thrombosis_prediction, european_football_2, formula_1, superhero, codebase_community, card_games, toxicology, california_schools, financial. The difficulty mix is 30% simple, 50% moderate and 20% challenging. DBs and data are in a Google Drive package — [archived card](https://huggingface.co/datasets/birdsql/bird_mini_dev/blob/main/archive/bird_mini_dev_2025-07-04/README.md)

**SynSQL-2.5M / OmniSQL**
- `seeklhy/SynSQL-2.5M`: Apache-2.0. It contains 2,544,390 `<database, question, SQL, CoT>` samples over **16,583 synthetic SQLite databases**, "generated entirely using open-source LLMs". The files are `data.json` (9.36 GB), `tables.json` (307 MB) and **`databases.zip` (54.5 MB)** — [HF card](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M); [arXiv 2503.02240](https://arxiv.org/abs/2503.02240)
- `seeklhy/OmniSQL-datasets`: Apache-2.0 tag. It is a single `data.zip` of **22.2 GB**. SkyRL's docs say these databases are "around 50GB in total" once unzipped — [HF repo](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets); [SkyRL docs](https://docs.skyrl.ai/docs/examples/multi_turn_text2sql)

**Spider 2.0**
- `xlangai/spider2-lite`: MIT tag. It is a single small `spider2-lite.jsonl` (~82 KB) mixing BigQuery (`bq*`), Snowflake and SQLite tasks, with `external_knowledge` markdown docs — [HF repo](https://huggingface.co/datasets/xlangai/spider2-lite)
- `xlangai/spider2-localdb` has no licence tag. It holds `sqlite.zip` (573 MB) and `duckdb.zip` (4.2 GB) — [HF repo](https://huggingface.co/datasets/xlangai/spider2-localdb)
- Even strong 7B SQL specialists do poorly on Spider2.0-SQLite: Arctic-Text2SQL-R1-7B gets 15.6, OmniSQL-7B 13.3 and Qwen2.5-Coder-7B-Instruct 4.4 — [Arctic-Text2SQL-R1 paper, Table 7](https://arxiv.org/html/2505.20315v2)

**BIRD-Interact / Mini-Interact / BIRD-Critic / LiveSQLBench (2025–2026)**
- `birdsql/mini-interact` (updated Sep 2026): CC-BY-SA-4.0. It has **300 ambiguous business-intelligence tasks** and uses **SQLite instead of PostgreSQL (no docker)**. The SQLite DBs sit in per-DB folders in the repo (27 DB folders such as `credit`, `crypto` and `reverse_logistics`). Ground-truth SQL and test cases are **withheld** and must be requested by email "to avoid data leakage". Evaluation can be conversational (user simulator) or agentic — [HF card](https://huggingface.co/datasets/birdsql/mini-interact)
- `birdsql/bird-interact-lite` / `bird-interact-full`: CC-BY-SA-4.0, n<1K. They come from the BIRD-INTERACT paper (ICLR 2026 Oral, arXiv 2510.05318), which pairs a function-driven user simulator with a hierarchical knowledge base — [HF card](https://huggingface.co/datasets/birdsql/bird-interact-lite)
- `birdsql/bird-critic-1.0-flash-exp` / `-open` / `-postgresql`: CC-BY-SA-4.0, n<1K. BIRD-Critic is a SQL debugging benchmark ("can LLMs fix user issues in real-world database applications"), arXiv 2506.18951 — [HF card](https://huggingface.co/datasets/birdsql/bird-critic-1.0-flash-exp)
- `birdsql/livesqlbench-base-lite-sqlite` (updated May 2026): CC-BY-SA-4.0, n<1K. It is "a dynamic, contamination-free benchmark", tagged multi-turn/interactive, and the SQLite variant of LiveSQLBench (BIRD-SQL Pro v0.5) — [HF card](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite)

**WikiSQL**
- `Salesforce/wikisql`: the HF licence is "unknown". It is crowd-sourced, single-table, 10K–100K rows — [HF card](https://huggingface.co/datasets/Salesforce/wikisql)

### Inferences
- **SynSQL-2.5M is the best permissive source of executable DBs.** It is Apache-2.0, its 54 MB `databases.zip` holds 16k+ SQLite DBs, and its creators say it was made only with open-source LLMs. You can sample a few hundred to a few thousand (DB, question, gold SQL) items. Executing the gold SQL produces verified answers, and running `list_tables → describe_table → run_sql → ANSWER:` gives oracle trajectories in the project's exact protocol. This matches the existing oracle-SFT plan.
- **BIRD train-filtered (6,601) is the best human-written executable train set.** BIRD train DBs are disjoint from the 11 dev DBs used by mini-dev, so training on BIRD train and evaluating on mini-dev is the standard, uncontaminated-by-DB setup. The cost is CC-BY-SA-4.0 share-alike on any derived trajectory dataset (see the licence section).
- BIRD's `evidence` field supplies domain hints. For the project's "explore the schema first" agent, either drop the evidence or feed it as a user hint. The 2026 mini-dev revision explicitly removed evidence that leaked schema links or answers.
- Spider 2.0 (lite/snow) and LiveSQLBench are too hard for a 3B model: 7B SQL specialists score about 15% on Spider2.0-SQLite. They are not worth the setup time for a 10-day solo project except as a "hard tail" mention.
- SParC/CoSQL are multi-turn in the *user-dialogue* sense (follow-up questions), not in the *tool-interaction* sense, so they matter less than single-question BIRD/Spider for this agent.
- WikiSQL is single-table and has no clear licence. Skip it.

### Gaps
- I did not verify the contents of `seeklhy/OmniSQL-datasets/data.zip`. It is presumably the Spider/BIRD/Spider2/Spider-DK/EHRSQL/ScienceBenchmark DBs plus SynSQL, but that is unconfirmed, so check the zip listing before downloading 22 GB.
- I did not find CoSQL's licence page. It is presumably CC BY-SA 4.0 like Spider/SParC, but unverified.
- I did not open the spider2-lite task count, the Spider 2.0-snow distribution terms, or the exact sizes of LiveSQLBench-SQLite and BIRD-Interact.
- WikiSQL's original GitHub licence was not checked.

## (b) Popular SFT-format SQL datasets without databases (b-mc2/sql-create-context, gretelai/synthetic_text_to_sql, Clinton/Text-to-sql-v1, etc.)

### Takeaway
These datasets are large and permissively licensed, but they put the full schema (CREATE TABLE) in the prompt and expect a single SQL string back. That trains exactly the "skip exploration, answer from context" behaviour the project wants to avoid, and none of them carry executed answers. Gretel is the exception worth considering: its `sql_context` includes INSERT statements, so its rows can be turned into tiny executable SQLite DBs.

### Cited Findings
- `b-mc2/sql-create-context`: CC-BY-4.0, **78,577** examples built from WikiSQL and Spider. Each has a natural-language question, the CREATE TABLE statements and the SQL. The stated goal is to prevent hallucinated table/column names — [HF card](https://huggingface.co/datasets/b-mc2/sql-create-context)
- `gretelai/synthetic_text_to_sql`: Apache-2.0, **105,851** records (100,000 train / 5,851 test), ~23M tokens across 100 domains, generated with Gretel Navigator. Fields include `sql_prompt`, `sql_context` (CREATE TABLE **plus INSERT INTO** statements), `sql`, `sql_explanation`, complexity and task-type tags — [HF card](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
- Arctic-Text2SQL-R1 used "Gretel-Synth-Filtered" (**11,811** examples). It kept only Gretel items where at least one of ten Qwen2.5-Coder-32B-Instruct generations was correct, which suggests raw Gretel rows have quality problems — [Arctic-Text2SQL-R1 paper, Table 3](https://arxiv.org/html/2505.20315v2). (The fetched summary described the Gretel data as "GPT-4o-generated". The Gretel card says only "Gretel Navigator", so the generator model is unconfirmed.)
- `Clinton/Text-to-sql-v1`: Apache-2.0 tag, 100K–1M rows. The card has **no description at all** (YAML only), so provenance and source datasets are undocumented — [HF card](https://huggingface.co/datasets/Clinton/Text-to-sql-v1)
- `lamini/bird_spider_train_text_to_sql`, `xu3kev/BIRD-SQL-data-train` and many `thanhdath/sft-gpt-4o-mini-...bird...` repos are community re-packagings of BIRD/Spider train, some with GPT-4o-mini-generated plans. `xu3kev/BIRD-SQL-data-train` has no licence tag — [HF hub search results](https://huggingface.co/datasets/xu3kev/BIRD-SQL-data-train)

### Inferences
- **Verdict: do not use b-mc2, Clinton or WikiSQL-derived data for this agent.** They teach single-shot SQL from in-context schemas. The base model's failures are protocol, grounding and recovery, not SQL syntax, so these add little and may hurt the "call describe_table first" behaviour. Clinton's undocumented provenance also makes it risky to republish.
- b-mc2 is built from Spider, which likely includes the Spider dev split (unverified). Training on it would contaminate any Spider-dev evaluation.
- **Optional use of Gretel (Apache-2.0):** run each `sql_context` into an in-memory SQLite DB, execute the gold SQL, keep only rows that execute and return non-empty results, and wrap them in the tool protocol. That yields cheap, diverse, permissively licensed mini-environments with 1–3 tables. It is low-value compared with SynSQL, which has real multi-table DBs, but it is an option if domain diversity is wanted. Expect dialect issues (MySQL/Postgres functions) and filter on execution success.

### Gaps
- I did not verify whether b-mc2 includes Spider dev rows, or how many Gretel rows execute cleanly in SQLite.
- Gretel Navigator's underlying generator model(s), and therefore any upstream model-terms caveats, were not confirmed.

## (c) Agentic / multi-turn SQL trajectory data (SkyRL-SQL-653, AgentBench DB, AgentInstruct DB, Arctic-Text2SQL-R1, SQL-R1, ReAct-style SQL trajectories)

### Takeaway
There are almost no public, permissively licensed ReAct-style SQL trajectories with execution feedback. Hub searches for "text2sql multi-turn" and "sql react trajectories" returned zero results. The best agentic asset is **SkyRL-SQL-653**, and it is an RL *task* set (prompt + gold SQL) rather than trajectories. AgentInstruct's 538-row DB split consists of GPT-4 trajectories with no stated licence. So the project's own generated trajectories, extended to SynSQL/BIRD DBs, are the main way to get multi-turn SQL data.

### Cited Findings
- `NovaSky-AI/SkyRL-SQL-653-data`: **653 train rows** plus a small validation parquet (~17 KB), **no licence tag** on HF. Columns are `data_source` (e.g. `synsql`), `prompt` (system + user messages), `db_id`, and `reward_model.ground_truth` (gold SQL). The system prompt defines a multi-turn protocol: `<think>`, `<sql>…</sql>` tool calls, `<observation>` with a dataframe truncated to 50 rows, and `<solution>` for the final SQL, "within limited turns". The user turn contains the full CREATE TABLE schema with example values — [HF dataset preview](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data)
- SkyRL's text2sql example uses `NovaSky-AI/SkyRL-SQL-653-data-newfmt` for training and `seeklhy/OmniSQL-datasets` `data.zip` for DBs (~50 GB). The base model is Qwen2.5-Coder-7B-Instruct with `max_turns=6`. The reward is negative for bad format, 0 for correct format with wrong SQL, and positive for correct SQL — [SkyRL docs](https://docs.skyrl.ai/docs/examples/multi_turn_text2sql)
- SkyRL README claim: SkyRL-SQL-7B, trained on "just 653 samples", outperforms GPT-4o and o4-mini — [NovaSky-AI/SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL). A secondary write-up reports up to +9.2% execution accuracy over the base model across 5 Spider variants. It also reports that multi-turn RL reached a 60% reward 2.8x faster than single-turn RL and finished 16% higher after 35 steps — [Substack summary (secondary)](https://machinelearningatscale.substack.com/p/text-to-sql-just-got-a-lot-better). No BIRD numbers were found for SkyRL-SQL.
- `zai-org/AgentInstruct` (formerly THUDM): 1,866 trajectories in total, of which the **`db` split has 538**, averaging **2.06 turns**. It uses ShareGPT-style `conversations` with a `loss` flag. The card says the trajectories are GPT-4 outputs filtered by reward ("Not all trajectories of GPT-4 are effective!"). **No licence is declared on the HF card** — [HF card](https://huggingface.co/datasets/zai-org/AgentInstruct)
- The AgentBench repo is Apache-2.0. Its DB task (dbbench) runs on **MySQL Docker** images, not SQLite, and the repo also has a function-calling version — [AgentBench GitHub](https://github.com/THUDM/AgentBench)
- Arctic-Text2SQL-R1 is single-turn GRPO with an execution reward: 1 for an exact result match, 0.1 for SQL that is syntactically valid and executable, 0 otherwise. Training data is BIRD-train 8,017 (filtered from 9,428), Spider-train 6,972, **Spider-dev 985** and Gretel-Synth-Filtered 11,811. SynSQL and BIRD-aug were tried and dropped. Model release: `Snowflake/Arctic-Text2SQL-R1-7B`. The paper does not say whether the training data is released, and `Snowflake/Arctic-Text2SQL-R1-data` does not exist on HF — [arXiv 2505.20315 HTML](https://arxiv.org/html/2505.20315v2)
- `Boakpe/pt-br-agentic-text-to-sql-distilled-trajectories` (Jun 2026): Apache-2.0, 1K–10K "message-only distilled trajectories for training tool-using Text-to-SQL agents in Brazilian Portuguese", LLM-judged correct — [HF card](https://huggingface.co/datasets/Boakpe/pt-br-agentic-text-to-sql-distilled-trajectories)
- Other hub hits were small or one-off: `asvs/o3_sql_agent_dataset` (o3-generated, 13 downloads) and `neyssemhajji16/text2sql-trajectories-stage3` (5 downloads) — [HF hub search](https://huggingface.co/datasets/asvs/o3_sql_agent_dataset)

### Inferences
- **SkyRL-SQL-653 verdict: the most directly relevant public asset, used as a task pool, not as trajectories.** Its rows carry gold SQL, so you can (1) generate oracle tool trajectories in the project's `list_tables/describe_table/run_sql/ANSWER:` protocol, or (2) use it as an RL prompt set with execution-verified rewards. For `synsql`-sourced rows, the needed DBs should be in SynSQL-2.5M's 54 MB `databases.zip` instead of the 22 GB OmniSQL bundle (inference, unverified per row). It has no licence tag, so cite it and inherit the underlying licences (Apache-2.0 for SynSQL; CC-BY-SA for any Spider/BIRD rows) when republishing derivatives.
- Its prompt puts the full schema in context. For this project, strip the schema so the agent must explore, which matches the hackathon's grounding goal.
- AgentInstruct-DB: **avoid**. It is GPT-4-generated (OpenAI terms caveat), has no explicit licence, averages 2 turns, and targets MySQL. AgentBench DB eval needs MySQL Docker, which is too heavy for the timeline.
- The Arctic-Text2SQL-R1 reward (1 / 0.1 / 0) is a good template for the project's verifier and is similar to the existing "correct answer backed by successful query" criterion. Its data mix includes Spider-dev, so a model trained on that mix should not be evaluated on Spider-dev.

### Gaps
- I did not research SQL-R1, Reasoning-SQL or MARFT training-set releases, or Spider-Agent trajectories. No HF repos turned up for SQL-R1 under the guessed IDs.
- The exact SkyRL-SQL Spider-variant numbers were not retrieved: the Notion blog would not render, so only the secondary summary is cited.
- I did not verify the source mix of the 653 rows beyond the first row (`synsql`).

## (d) General tool/function-calling datasets (xLAM-60k, APIGen-MT-5k, Hermes, ToolACE, Glaive, When2Call, BFCL, Toucan)

### Takeaway
Two cleanly licensed datasets fit this project best. **When2Call** (CC-BY-4.0, NVIDIA, "ready for commercial use") targets not calling or not hallucinating tools and has SFT plus preference splits. **Toucan-1.5M** (Apache-2.0, generated by open-weight models with *real* MCP tool executions) has a curated 119k SFT subset with multi-turn and "irrelevant tool" variants. xLAM-60k is CC-BY-4.0 but gated, research-framed and single-turn. **APIGen-MT-5k is CC-BY-NC-4.0, partly GPT-4-generated with an OpenAI non-compete clause, and covers tau-bench domains, so avoid it.**

### Cited Findings
- `nvidia/When2Call`: CC-BY-4.0, and "This dataset is ready for commercial use". It contains `train_sft` (15,000), `train_pref` (9,000, with chosen/rejected responses), test `mcq` (3,652) and `llm_judge` (300). It evaluates when to call a tool, when to ask a follow-up, and when to admit the tools cannot answer. Synthetic, NAACL 2025 — [HF card](https://huggingface.co/datasets/nvidia/When2Call)
- `Agent-Ark/Toucan-1.5M`: Apache-2.0, 1.5M+ trajectories from 495 real MCP servers and 2,000+ tools, "involving real tool executions across multi-round, multi-turn, sequential, and parallel tool calls". Configs are by generator: Kimi-K2 (518,516), OSS = GPT-OSS-120B (457,130) and Qwen3 = Qwen3-32B (551,613). There is also a curated **SFT** config of **119,287** rows (~1.35 GB) in Swift format, with tools in the system prompt in Hermes format. Subsets: single-turn-original, **irrelevant** (server shuffle), single-turn-diversify and **multi-turn**. arXiv 2510.01179 — [HF card](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)
- `Salesforce/xlam-function-calling-60k`: CC-BY-4.0, **gated**, 60,000 entries of `query/tools/answers` (single query per entry). Generated by DeepSeek-V2-Chat (IDs 0–33,658) and Mixtral-8x22B-Inst (the rest). The card notes "This release is for research purposes only in support of an academic paper" — [HF page](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- `Salesforce/APIGen-MT-5k`: **CC-BY-NC-4.0**, gated, 5,000 multi-turn trajectories (ShareGPT with `function_call`/`observation` turns) in the τ-bench **Retail and Airline** domains, made with GPT-4o and DeepSeek-V3. The card says: "A part of this dataset was generated using GPT-4 and should not be used to develop models that compete with OpenAI." — [HF card](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- `NousResearch/hermes-function-calling-v1`: Apache-2.0, five configs (`func_calling_singleturn`, `func_calling` multi-turn, `glaive_func_calling` 5k cleaned Glaive, `json_mode_agentic`, `json_mode_singleturn`). It uses `<tool_call>` / `<tool_response>` ChatML tags. The card does not name the generator model — [HF card](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- `Team-ACE/ToolACE`: Apache-2.0, en/zh, 10K–100K dialogs over a 26,507-API synthetic pool, multi-agent generation with rule-based and model-based checks. The card does not name the generator model. arXiv 2409.00920 — [HF card](https://huggingface.co/datasets/Team-ACE/ToolACE)
- `glaiveai/glaive-function-calling-v2`: Apache-2.0 tag, 100K–1M rows. The card has **no description** (YAML only) — [HF card](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- `gorilla-llm/Berkeley-Function-Calling-Leaderboard`: Apache-2.0, eval data for BFCL, updated Apr 2026 — [HF card](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

### Inferences
- **When2Call: use it.** It directly targets the observed failure "fabricates tool results / doesn't end its turn", and its preference split suits a small DPO stage after SFT. Convert it to the project's tool schema, or keep it in the native Aya tool format if one is used. The MCQ test (3,652) is a cheap, log-likelihood-style before/after metric for tool-use decisions.
- **Toucan SFT subset: optional, small dose (~1–3k rows), mainly the multi-turn and irrelevant subsets.** It can generalise the protocol (call, read the real result, then answer). The trajectories are long, so filter by token length to fit a 3B model's training context.
- **xLAM-60k:** licence-clean (CC-BY-4.0, open-model generated), but single-turn and gated. Low marginal value for a multi-turn SQL agent. Skip unless BFCL single-turn scores are targeted.
- **APIGen-MT-5k: avoid.** The NC licence is incompatible with mixing into CC-BY-SA BIRD-derived data, the OpenAI clause adds uncertainty, and training on τ-bench retail/airline contaminates τ-bench if it is reported.
- **Hermes / Glaive / ToolACE:** permissive tags but undocumented generators, mostly single-turn API calls unrelated to databases. Low priority. Glaive-v2's empty card makes provenance hard to defend in a public submission.
- Format mismatch is the main integration cost. Each dataset uses a different tool-call syntax (Hermes `<tool_call>`, ShareGPT `function_call`, Swift), and all must be normalised to the single chat template and tool format used for Tiny Aya / Aya Expanse.

### Gaps
- I did not confirm generator models for Hermes, Glaive-v2 or ToolACE. If any were GPT-4-generated, OpenAI-terms caveats would apply.
- I did not check for other 2025–2026 multi-turn tool-use datasets beyond Toucan (e.g. Nemotron tool-use mixes or the xLAM-2 follow-ups).

## (e) External evaluation benchmarks a small model can be cheaply reported on, with ~3B / ~8B reference scores

### Takeaway
The best external benchmark here is **BIRD mini-dev SQLite**: 500 questions over 11 dev DBs, CC-BY-SA, small, and the 2026 revision is explicitly framed as "explore the database" agent evaluation. Run it with the project's own tool protocol on the full 500 or a fixed 100–200 subset. **Spider dev** is the easier secondary check. **When2Call MCQ** is a cheap tool-decision metric. BFCL and τ-bench are more expensive and less aligned with the task. τ-bench also needs an LLM user simulator.

### Cited Findings
- **BIRD mini-dev 2026 fixed-prompt baselines** (EX %, SQLite / overall): Claude Opus 5.5 90.00 / 87.13, GPT-6.1 Sol 88.00 / 87.27, Claude Haiku 5.5 59.60 / 57.80, GLM-5.3 51.40 / 49.40, Qwen3.6-35B-A3B 21.20 / 21.13, empirical upper bound 99.60 SQLite — [HF card](https://huggingface.co/datasets/birdsql/bird_mini_dev)
- **BIRD mini-dev 2025-07 (older annotations), small models** (EX SQLite): llama3-8b-instruct **24.40**, mixtral-8x7b 21.60, phi-3-medium-128k 30.60, gpt-35-turbo 38.00, gpt-4 47.80 — [archived card](https://huggingface.co/datasets/birdsql/bird_mini_dev/blob/main/archive/bird_mini_dev_2025-07-04/README.md)
- **3B reference point:** Qwen2.5-3B base scores 26.2 Mini-Dev / 31.88 Dev EX, rising to **46.0 / 50.0** after SFT on bird23-train-filtered with Arctic's data processing — [bird23-train-filtered card](https://huggingface.co/datasets/birdsql/bird23-train-filtered)
- **7B references** (BIRD-dev / Spider-test EX): Qwen2.5-Coder-7B-Instruct 50.9 / 82.2; OmniSQL-7B 63.9 / 87.9; Arctic-Text2SQL-R1-7B 68.9 / 88.8 — [Arctic paper, Table 7](https://arxiv.org/html/2505.20315v2)
- **BFCL v3 overall** (FC mode, leaderboard of 04/03/2025): xLAM-2-1b-fc-r 58.90, **xLAM-2-3b-fc-r 65.11**, xLAM-2-8b-fc-r 72.83, ToolACE-8B 58.42. **τ-bench pass@1 overall:** xLAM-2-1b 21.8, **xLAM-2-3b 38.2** (retail 44.4, airline 32.0), xLAM-2-8b 46.7 — [APIGen-MT paper](https://arxiv.org/html/2504.03601)
- When2Call test: MCQ 3,652 items plus an LLM-judge subset of 300 — [HF card](https://huggingface.co/datasets/nvidia/When2Call)
- BIRD mini-dev also has a Harbor Hub interactive version in which solvers submit `/app/answer.sql`, with network disabled — [HF card](https://huggingface.co/datasets/birdsql/bird_mini_dev)

### Inferences
- The 2026 mini-dev numbers are **not comparable** to 2024–2025 papers, because annotations were revised and leaking evidence removed. That version is hard: a 35B-A3B MoE gets 21%. A Tiny Aya-based agent should probably expect single digits to the low 20s. State in the report which mini-dev version was used, or report both the archived 2025-07 version (comparable to llama3-8b 24.4) and the 2026 version.
- To make the external metric fit the project's success definition, use "ANSWER value or final query result matches the gold-SQL result set" rather than raw SQL. Many BIRD answers are multi-row or multi-column, so either restrict to single-value questions or score with result-set comparison of the last successful `run_sql`. Report the subset rule explicitly.
- Spider dev (1,034, 20 DBs, via Google Drive) is an easy-to-run sanity check, but contamination makes high scores less meaningful (next section).
- With small compute, a fixed **100–200 question subset of BIRD mini-dev SQLite** plus the **When2Call MCQ** test gives two before/after numbers in a few GPU-hours.

### Gaps
- I found no published BIRD/Spider/BFCL scores for Tiny Aya or Aya Expanse 8B/32B. The project must measure its own baseline.
- I did not retrieve exact Spider-dev EX for ~3B general models, or BFCL v4 numbers from 2026.

## (f) Contamination risk and licence compatibility with publishing derived datasets on HF

### Takeaway
Everything BIRD/Spider/SParC/BIRD-Interact/LiveSQLBench is **CC-BY-SA-4.0**. Trajectories derived from those DBs and questions can be published on HF but must carry CC-BY-SA-4.0 and attribution. SynSQL, Toucan, Gretel, Hermes, ToolACE and glaive are Apache-2.0, and When2Call and xLAM-60k are CC-BY-4.0. All of these can be mixed into a CC-BY-SA release. **CC-BY-NC sources (APIGen-MT-5k) and GPT-4 outputs with OpenAI clauses (APIGen-MT, AgentInstruct) should be excluded.** The cleanest eval-integrity story is to train on BIRD train / SynSQL and evaluate on BIRD mini-dev, which uses disjoint DBs.

### Cited Findings
- Spider is CC BY-SA 4.0 — [Spider site](https://yale-lily.github.io/spider); SParC is CC BY-SA 4.0 — [SParC site](https://yale-lily.github.io/sparc)
- BIRD train-filtered, dev and mini-dev, mini-interact, bird-interact, bird-critic and LiveSQLBench-SQLite are all tagged `license:cc-by-sa-4.0` — [bird23-train-filtered](https://huggingface.co/datasets/birdsql/bird23-train-filtered), [bird_mini_dev](https://huggingface.co/datasets/birdsql/bird_mini_dev), [mini-interact](https://huggingface.co/datasets/birdsql/mini-interact), [livesqlbench-base-lite-sqlite](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite)
- SynSQL-2.5M is Apache-2.0 and "generated entirely using open-source LLMs" — [HF card](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M)
- APIGen-MT-5k is CC-BY-NC-4.0, and "should not be used to develop models that compete with OpenAI" — [HF card](https://huggingface.co/datasets/Salesforce/APIGen-MT-5k)
- AgentInstruct consists of GPT-4 trajectories with no licence on the card — [HF card](https://huggingface.co/datasets/zai-org/AgentInstruct)
- xLAM-60k is CC-BY-4.0, generated by open models (DeepSeek-V2-Chat, Mixtral-8x22B), but framed "for research purposes only" — [HF page](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- The Spider 1.0 test set has been public since Feb 2024 — [Spider site](https://yale-lily.github.io/spider). Arctic-Text2SQL-R1 even trains on Spider-dev — [Arctic paper](https://arxiv.org/html/2505.20315v2)
- Mini-Interact withholds gold SQL and test cases "to avoid data leakage by auto-crawling", and LiveSQLBench describes itself as "contamination-free" — [mini-interact](https://huggingface.co/datasets/birdsql/mini-interact), [LiveSQLBench-SQLite](https://huggingface.co/datasets/birdsql/livesqlbench-base-lite-sqlite)
- The 2026 mini-dev revision corrected gold SQL and replaced selected instances — [HF card](https://huggingface.co/datasets/birdsql/bird_mini_dev)

### Inferences
- **Share-alike consequence.** Publish BIRD- or Spider-derived trajectories as a separate HF dataset licensed CC-BY-SA-4.0, crediting BIRD/Spider. Mixing in Apache-2.0 and CC-BY-4.0 material is generally fine, since both can be relicensed under BY-SA. CC-BY-NC material cannot be combined into a BY-SA adaptation, because the two licences' terms conflict. This is my reading of CC compatibility, not legal advice. Keep each source as a separately licensed config/repo, and list per-source licences in the dataset card.
- Publishing **Tiny Aya-generated** trajectories is a separate question (Tiny Aya is CC-BY-NC). Tag any model-generated trajectories with the generator and check Cohere's terms. This is outside this note's scope.
- **Contamination:** Spider dev/test are public and widely trained on (directly, and via b-mc2 / Clinton re-packagings). A Spider-dev number is weak evidence. BIRD mini-dev is better: train DBs are disjoint from dev DBs, and the 2026 revision changed items. Tiny Aya's pretraining exposure to BIRD dev questions is unknown. If time permits, a few LiveSQLBench-SQLite or Mini-Interact tasks would be the most contamination-resistant evidence, but gold SQL for Mini-Interact requires an email request.
- **Do not train on BIRD dev / mini-dev rows or their DBs** if mini-dev is the reported benchmark. Likewise, exclude Spider dev from training if Spider dev is reported.

### Gaps
- I did not verify the formal CC compatibility chart for this note, and had no legal review.
- Tiny Aya / Aya Expanse pretraining data composition, and therefore benchmark exposure, is unknown.

## (g) Recommended shortlist for ~10 days and small compute

### Takeaway
Train on the project's own synthetic environment as the core. Add **oracle tool trajectories generated over SynSQL-2.5M DBs (Apache-2.0) and BIRD train-filtered DBs (CC-BY-SA)**, including the SkyRL-SQL-653 task pool, all in the project's exact `list_tables/describe_table/run_sql/ANSWER:` protocol with injected-error recovery. Add a small slice of **When2Call** (anti-hallucination, with optional DPO). Evaluate on held-out own tasks plus a fixed subset of **BIRD mini-dev SQLite** and the **When2Call MCQ** test.

### Cited Findings
- SynSQL ships 16,583 SQLite DBs in a 54.5 MB zip under Apache-2.0 — [HF card](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M)
- SkyRL-SQL-653 is 653 RL tasks with gold SQL in a multi-turn `<sql>`/`<observation>` format. A 7B model trained on these 653 samples reportedly beats GPT-4o — [HF dataset](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data); [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL)
- BIRD train-filtered: 6,601 examples, and SFT lifts Qwen2.5-3B from 26.2 to 46.0 mini-dev EX — [HF card](https://huggingface.co/datasets/birdsql/bird23-train-filtered)
- When2Call: 15k SFT / 9k preference rows, CC-BY-4.0, commercial-OK — [HF card](https://huggingface.co/datasets/nvidia/When2Call)
- Toucan SFT subset: 119,287 rows, Apache-2.0, real tool executions, multi-turn and irrelevant-tool subsets — [HF card](https://huggingface.co/datasets/Agent-Ark/Toucan-1.5M)
- BIRD mini-dev: 500 SQLite questions over 11 dev DBs, with a 2026 revision aimed at agents that explore the DB — [HF card](https://huggingface.co/datasets/birdsql/bird_mini_dev)

### Inferences
**Training mix (SFT, then optional DPO/RL):**
1. **Own environment trajectories (235 existing, ~50–60% of the mix):** the core protocol, grounding and recovery behaviours.
2. **SynSQL oracle trajectories (~500–1,500):** sample DBs and questions of simple/moderate complexity, execute the gold SQL, and keep single-value or small results. Synthesise `list_tables → describe_table(s) → run_sql → ANSWER:` with some injected wrong-column or syntax errors followed by recovery, as in the existing generator. Apache-2.0, so it is easy to republish.
3. **SkyRL-SQL-653 (whole set) as an RL/rejection-sampling prompt pool**, and/or converted to oracle trajectories with the schema stripped from the prompt.
4. **BIRD train-filtered oracle trajectories (~300–1,000, optional):** human-written questions on realistic messy DBs. Publish these as a separate CC-BY-SA-4.0 config. Downloading train.zip (several GB) from Aliyun is the main time cost.
5. **When2Call (~1–2k SFT rows plus the 9k preference set for DPO if time allows):** targets fabricated tool results and calling when not needed. CC-BY-4.0.
6. *(Optional)* **Toucan SFT multi-turn/irrelevant subset (~1–2k length-filtered rows):** general multi-turn tool hygiene. Apache-2.0.
- **Skip:** APIGen-MT-5k (NC + OpenAI clause + τ-bench contamination), AgentInstruct (GPT-4, no licence, MySQL), b-mc2 / Clinton / WikiSQL / glaive (single-shot, schema-in-context, weak provenance), Spider 2.0 / LiveSQLBench for training (too hard, eval-only).

**Evaluation set:**
1. Held-out own tasks (primary, matches the hackathon's verified-answer criterion).
2. **BIRD mini-dev SQLite, fixed 100–200 question subset** (or all 500 if compute allows), run in the same agent loop and scored by result-set match. State the version (2026 revision vs archived 2025-07) and the subset rule. Context: llama3-8b 24.4 (2025-07 version); Qwen2.5-3B 26.2 base → 46.0 after SFT; Qwen3.6-35B-A3B 21.2 on the 2026 version.
3. **When2Call MCQ test (3,652)** for a cheap before/after on tool-decision behaviour.
4. *(Stretch)* Spider dev subset as an easy sanity check (flag contamination), or a handful of Mini-Interact / LiveSQLBench-SQLite tasks as contamination-resistant evidence.
- **Publishing:** separate HF dataset repos/configs per licence: own synthetic (author's choice), SynSQL-derived (Apache-2.0), BIRD-derived (CC-BY-SA-4.0), When2Call-derived (CC-BY-4.0). The dataset card should name sources, licences and generator models.

### Gaps
- The proportions above are judgement calls, not validated by an ablation in this project.
- Download and setup time for the BIRD train/dev Google Drive and Aliyun packages and the BIRD mini-dev Docker image was not measured.
