# Abstract

**Teaching Tiny Aya to query databases: what fine-tuning fixes, and what it does not.**
Track 3 (Agentic Fine-tuning), problem statement "data and database agents". Solo entry.

Untuned, Cohere's Tiny Aya Global is a poor database agent: it invents tool output, writes malformed
calls and answers without querying.

**Setup.** The agent gets four tools over a read-only SQLite database (list tables, describe a table, run
SQL, submit SQL) and eight tool steps per question.

**Method.** We fine-tuned Tiny Aya Global with QLoRA (4-bit, LoRA rank 16) through Lexsi AgentTune's
audited SFT factory. Training used 1,500 verified multi-turn episodes from SQaLe and a fictional shop
database. No teacher model was used: episodes replay the reference query path against the real
database. We then ran one round of self-training (rejection-sampling fine-tuning). The model answered 577
held-out questions and was trained on its 306 verified correct attempts. A token-matched control on the
same questions isolates the effect.

**Evaluation.** Scoring is execution-based, with no LLM judge. The primary metric is strict: the
submitted query's full result must equal the reference, and the query must read a real table. We also
report set and fallback accuracy, shop retention and behaviour. AuditKIT agreed with every
score. Every paid run was preregistered with pinned hashes and ran once. Final sets were used only after
choosing the model on a 95-task dev set.

**Results.** On 144 SQaLe questions over databases never seen in training, strict accuracy rose from
2.8% to 30.6%. Shop retention rose from 0 to 40 of 40. Submitted queries that failed to run fell from 377
to 2 across all final sets. On BIRD Mini-Dev (493 questions, never trained on), accuracy moved only from
0% to 1.4%. The gains are on one- and two-table questions. Complex questions (3+ tables or multi-step)
stay near zero (1 of 51).

**Negative results, reported as they are.** Three targeted fixes did not raise accuracy: error-hint
tools, repair demonstrations, and self-training. Self-training beat its matched control on dev, 4 tasks
to 0 (p = 0.125), but gave no final gain: the model's correct attempts were mostly simple. No model
recovered after a failed query. Running out of the eight tool steps is the main failure.

**Closed-model comparison (ROI).** OpenAI's cheapest current model, gpt-6-luna, took the same test with
identical prompts, tools and scoring. It scored 66.0% (25 of 51 complex), at $0.0007 per correct answer
against B's $0.025 on a rented T4. B matched it only on simple lookups (29 vs 31 of 36), with cleaner tool
formatting.

**Takeaway.** Supervised fine-tuning on verified episodes turns an unusable small agent into a clean tool
user for simple queries. It does not teach multi-step reasoning or error recovery, and a cheap closed
model remains far stronger. Execution-reward RL is the natural next test.

**Cost.** About $37 of T4 compute, plus $0.07 of API calls. We release the adapter, all training episodes,
evaluation trajectories and job logs.
