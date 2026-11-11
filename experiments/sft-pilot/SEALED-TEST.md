# SFT pilot: sealed test (preregistered 2026-11-05, before the test split is generated)

The pilot's dev result (`RESULTS-dev.md`: base-nf4 2/40, sft-nf4 40/40) is in-family: every dev
family is a training family, 25/40 dev questions appear verbatim in training (other databases),
and the trained model's final queries all match training query templates. The sealed test is the
independent check.

**Status of the split before this document.** It was generated once, on 2026-11-04 00:30 IST, while
the harness was built (the README's count of 97 tasks comes from it; `data/test.jsonl`, gitignored,
on the development laptop). No model has been run on it (no run config has `"split": "test"`), it
never fed training (`sftdata.py` asserts the dev and test seeds are excluded), and nothing in the
design used its contents beyond the count. Its sha256 is pinned here without displaying it:
`3d3bd3a39c406d73e9d7f070a32a7babaa1ced2561e2fe8b427f804e73aaec54` as written on Windows (CRLF),
`dbe99529da23222b3093552cdd099af9830043531d85ce83c8df5c4366745d14` with LF line endings, which is
what the Linux jobs must record. `ayadb.smoke` refuses the split without `--unseal`. Training
episodes were deduplicated against dev only, so shared-family test questions may appear verbatim
in training; the report separates them.

## What runs (once)

| Run | Base | Adapter |
|---|---|---|
| `test-base-nf4` | nf4 | none |
| `test-sft-nf4` | nf4 | the pilot adapter, unchanged (`runs/pilot/sft-pilot/adapter`, sha256 `6f2827ae9b288fb96de32ac3f0f8e258743acd61ea754739f3c67319c21e4794`, the one evaluated on dev) |

`python scripts/hf_job.py sealed_test`: Hugging Face Jobs `t4-small`, the dev protocol unchanged
(`multistep-close`, greedy, 8 tool rounds, 512 new tokens per turn, same system prompt, tools,
verifier and library versions). No fp16 runs and no repeats (A/A on dev: 0 flips). Each job
generates the split itself (`generate_split("test")`, database seed 303; 8 shared and 3 held-out
families, 97 tasks); both must record the LF sha256 above, which the report prints. The task file
is not printed or inspected before both runs finish.

## Outcomes

**Primary:** verified success on the three held-out families (`cancel_leader_tickets`,
`largest_order`, `segment_aov`; multi-hop, never in train or dev), `test-base-nf4` vs
`test-sft-nf4`, exact McNemar on paired task outcomes.

**Secondary:** the same comparison on the 8 shared families, and within them on questions that
appear verbatim in training vs those that do not; verified success per family; the dev diagnostics
(answer seen in SQL, recovery, malformed calls, fabricated tool output, loops, tool calls, turns,
tokens); template replay: how many of the trained model's final queries match a training query
once literals are abstracted, for held-out and shared families.

All of it is produced by `python -m ayadb.compare remote/hf --split test`, committed with this
document and dry-run on renamed dev runs only.

## Interpretation (fixed now)

On the held-out families, `test-sft-nf4` vs `test-base-nf4`:

- **Transfer**: sft better with p < 0.05 and at least half of the held-out tasks verified.
- **Partial transfer**: sft better with p < 0.05, fewer than half verified.
- **No evidence of transfer**: otherwise. The claim then stays within the trained question types.

On the shared families: a significant loss (more tasks lost than gained, p < 0.05) is reported as a
failure; sft below 80% means the dev result overestimates in-family performance. A held-out failure
that replays a training template is reported as the model applying a trained procedure to a new
question, not as a tool-use error.

## Validity and reruns

Invalid (reported, not interpreted): a job fails or reaches its time limit; a run's tasks sha256
is not the pinned LF hash; the adapter sha256 differs from the one above; an orchestration check reports a
tool-result turn the environment did not produce. An invalid run may be rerun once with identical
settings (`AYADB_SEALED_RERUN=1`), with the reason recorded here. A valid result is final: nothing is
tuned, trained or selected on the test set, and later experiments report on dev or on a newly
generated split.

## Cost

Two T4 jobs. Expected about $0.60-0.80 (dev: ~25 s per task); the cap is 4.4 h each (~$1.77, ~$3.54
for both).
