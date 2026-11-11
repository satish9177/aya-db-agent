# AgentTune SFT feasibility and Lexsi integration (2026-11-07)

## Finding

**Feasible through AgentTune's supported custom `trainer_factory`, not through its default factory
unchanged.** A CPU-only spy test exercises `create_distill_trainer -> Project.distill -> custom factory
-> trainer.train` and proves that the exact supplied model object, arguments and audited assistant-only
labels reach the trainer. No model weights, optimizer, teacher, model sampling or GPU training were used.
Actual Tiny Aya/QLoRA gradients, GPU memory and independently reloaded checkpoints remain untested in this
path and require an individually approved smoke run.

Installed version: AgentTune 1.1.3. Local source:
`.venv/Lib/site-packages/agenttune/agentic/distill_trainer.py` (`create_distill_trainer`,
`_DefaultSFTTrainerFactory`) and `agenttune/agentic/project.py` (`Project.distill`). The custom factory
is explicitly documented to return an object with `.train()`. It receives `model=student` and
`train_dataset=rows`; Project actually calls its `.train()` inside the distillation orchestration.

## Default path limitations

| Requirement | Installed default | Prepared custom path |
|---|---|---|
| Correct Tiny Aya tokenizer IDs | Loads `AutoTokenizer`; our known pre-tokenizer substitution would return | Uses the already loaded official `TokenizersBackend` |
| nf4/double-quant QLoRA | Model loader exposes dtype/device map but no explicit quantization config | Accepts the caller's already prepared PEFT/QLoRA model, without loading or wrapping it again |
| Mask folded tool observations and injected wrong calls | Generic EventLog export provides segment metadata; the default factory does not implement our exact per-turn masks | Explicit audited row adapter preserves flags; existing `sftdata.tokenize` creates labels |
| Existing JSON-list call syntax | Generic EventLog export rewrites calls into `<tool_call>` syntax | `AuditedTrajectory` preserves the original audited message bytes |
| Model/optimizer settings | Defaults include attention-only LoRA targets and automatic bf16 choice | Supplied training arguments and model keep our all-linear targets, optimizer and precision |

Additionally, `agenttune.backends.trl.sft.__all__` is empty in this installation: its initializer tries
to import `.sft.TRLSFTTrainer`, but that implementation file is absent. Do not rely on that registry
entry. The separate `create_distill_trainer` implementation is present and its custom-factory path
passes the spy test.

Implementation is additive: `src/ayadb/agenttune_sft.py`. The legacy `src/ayadb/sft.py` remains unchanged,
so old experiments retain their original training implementation. `AuditedTrajectory` is an explicit
EventLog extension, and `AuditedSFTFactory` builds an HF Trainer from the supplied model and pretokenized
examples. This is AgentTune-orchestrated custom SFT, not a claim that its stock SFT factory works as-is.
No command that starts real training was added to the approved build.

Validation: `test_agenttune_custom_factory_preserves_model_and_exact_masks` in `tests/test_sqlsubmit.py`.
It checks the full orchestration with a spy, identical model/argument identity and identical token labels,
including masking the executed injected error. This establishes construction/data-path feasibility,
not Tiny Aya training quality or organizer acceptance of a particular integration.

## Are we using two Lexsi tools?

**Yes.** AgentTune already renders prompts and executes the real tool-rollout loop in `smoke.py` and
`sftdata.py`. AuditKIT imports saved trajectories and evaluates outcomes in `smoke.py`; the new
`sqlsftdata.py` integration evaluates the full-result SQL metric through `CustomPredicate`.

The [official hackathon page](https://hackathon.lexsi.ai/) was checked on 2026-11-07. Its general minimum
is two track repos used together, explicitly **except Track 3** (both Project Requirements and FAQ).
Track 3 itself displays AgentTune for post-training and AuditKIT for evaluation. We already use both;
the outstanding training integration is why this supported factory path was prepared. There is no need
to add a third Lexsi library solely to satisfy a repo count.

## Next approval boundary

Before a real smoke run, freeze its preregistration, model/adapter hashes, precision, assistant masks,
checkpoint/reload checks, expected billing account and a dollar cap. Model loading/training and paid
compute remain unapproved. Do not switch to the default factory to save integration work.
