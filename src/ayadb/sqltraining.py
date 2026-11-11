"""Registered sql-submit-v1 SFT through the AgentTune custom factory; GPU use is opt-in."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .agenttune_sft import AuditedSFTFactory, prepare_trainer
from .engine import bnb_config, official_tokenizer
from .promptquality import fingerprint
from .runinputs import input_count, row_identity, supervised_count, training_plan
from .runledger import TokenLedger, atomic_json, object_sha
from .sft import Collator, LORA_TARGETS
from .sqlsftdata import encode_episode
from .positionloss import FULL, SELECTED, TrainedPositionLossMixin
from .gpumemory import GpuMemoryRecorder, record_oom


def validate_continuation(config, step, train, plan):
    if step.get("role") not in {"continuation_control", "continuation_self"}:
        return
    if not step.get("continuation_manifest"):
        raise ValueError("continuation requires a frozen post-fallback matching/gate manifest")
    matching = json.loads(Path(step["continuation_manifest"]).read_text(encoding="utf-8"))
    gate = matching["post_fallback_size_gate"]
    if not gate["passed"] or gate["verified_unique_targets_per_pass"] < (gate["a_unique_targets_per_pass"] + 9) // 10:
        raise ValueError("post-fallback continuation size gate failed")
    if len(matching["mask_adjustments"]) > 1 or {r["question_id"] for r in train} != set(matching["retained_question_ids"]):
        raise ValueError("continuation question coverage/mask rule differs from the frozen match")
    if plan["source_sha256"] != matching["datasets"][step["role"]]["sha256"] or plan["train_targets_per_pass"] != matching["matched_targets"]:
        raise ValueError("continuation dataset/target budget differs from the frozen match")
    if fingerprint(Path(step["base_adapter"]) / "adapter_model.safetensors") != matching["a_adapter_sha256"]:
        raise ValueError("continuation did not start from the frozen A adapter")
    settings = {k: step[k] for k in ("seed", "epochs", "batch_size", "gradient_accumulation", "learning_rate")}
    settings.update(dtype=config["dtype"], quantization=config["quantization"], rank=config["rank"],
                    alpha=config["alpha"], dropout=config["dropout"])
    if step.get("loss_policy", FULL) != FULL:
        settings["loss_policy"] = step["loss_policy"]
    if settings != matching["training_settings"]:
        raise ValueError("continuation settings differ from the common frozen settings")


def load_model(config, *, adapter=None, trainable=False):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["model_revision"], dtype=getattr(torch, config["dtype"]),
        device_map={"": 0}, quantization_config=bnb_config(config["dtype"]))
    if trainable:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = get_peft_model(model, LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
                               lora_dropout=config["dropout"], target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
    return model


def verify_reloaded_adapter(model, directory: Path):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(directory / "adapter_model.safetensors"), device="cpu")
    actual = get_peft_model_state_dict(model)
    if set(saved) != set(actual):
        raise ValueError("checkpoint reload adapter-key mismatch")
    for name, tensor in saved.items():
        if not torch.equal(actual[name].detach().cpu().to(tensor.dtype), tensor):
            raise ValueError("checkpoint reload tensor mismatch")
    return {"adapter_tensors": len(saved), "all_tensors_match": True,
            "adapter_sha256": fingerprint(directory / "adapter_model.safetensors")}


def run_training(config: dict, step: dict):
    memory = GpuMemoryRecorder(Path(step["out"]).with_name(Path(step["out"]).name + ".gpu-memory.jsonl"),
                               phase=step.get("name", "training"))
    with record_oom(memory):
        return _run_training(config, step, memory)


def _run_training(config: dict, step: dict, memory):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required before model loading")
    out = Path(step["out"])
    resume = Path(step["resume"]) if step.get("resume") else None
    if out.exists() and not resume:
        raise ValueError("training output exists; register a new run or an explicit checkpoint resume")
    if step.get("role") in {"continuation_control", "continuation_self"}:
        if step["validation_share"] or step["max_steps"] != -1 or not step.get("base_adapter"):
            raise ValueError("continuations need A adapter, no validation split, and complete matched passes")
        if int(step["epochs"]) != step["epochs"]:
            raise ValueError("continuations need whole matched passes")
    train, validation, plan = training_plan(Path(step["episodes"]), seed=step["seed"], val_share=step["validation_share"])
    validate_continuation(config, step, train, plan)
    out.mkdir(parents=True, exist_ok=True)
    train_set_sha = object_sha(plan)
    training_identity = {k: step.get(k) for k in (
        "seed", "epochs", "max_steps", "learning_rate", "batch_size", "gradient_accumulation",
        "validation_share", "base_adapter")}
    if "loss_policy" in step:
        training_identity["loss_policy"] = step["loss_policy"]
    training_config_sha = object_sha({"model": config, "training": training_identity})
    if resume:
        marker = json.loads((resume / "_COMPLETE").read_text(encoding="utf-8"))
        if marker["train_set_sha256"] != train_set_sha or marker["global_step"] != step["expected_resume_step"]:
            raise ValueError("checkpoint data identity/step mismatch")
        if marker.get("training_config_sha256") != training_config_sha:
            raise ValueError("checkpoint training settings changed")
        if not (resume / "optimizer.pt").exists() or not (resume / "scheduler.pt").exists():
            raise ValueError("checkpoint lacks optimizer/scheduler state")
        previous = json.loads((resume / "token-ledger.json").read_text(encoding="utf-8"))
    else:
        previous = None
    ledger = TokenLedger(out, previous=previous)
    set_seed(step["seed"])
    tokenizer = official_tokenizer(config["tokenizer"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    val_encoded = [encode_episode(ep, tokenizer, config["max_input_tokens"]) for ep in validation]
    if any(ex is None for ex in val_encoded):
        raise ValueError("validation exceeds context; refusing truncation")
    memory.reset()
    model = load_model(config, adapter=step.get("base_adapter"), trainable=True)
    memory.record("after_model_load")
    trainable, total = model.get_nb_trainable_parameters()
    if trainable <= 0:
        raise ValueError("no trainable adapter parameters")
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=step["epochs"], max_steps=step["max_steps"],
        per_device_train_batch_size=step["batch_size"], per_device_eval_batch_size=1,
        gradient_accumulation_steps=step["gradient_accumulation"], learning_rate=step["learning_rate"],
        lr_scheduler_type="cosine", warmup_steps=0, weight_decay=0., max_grad_norm=1.,
        logging_steps=1, logging_nan_inf_filter=False, eval_strategy="steps" if validation else "no",
        eval_steps=step.get("eval_steps", 50), save_strategy="steps", save_steps=step["save_steps"],
        save_total_limit=step["keep_checkpoints"], save_only_model=False,
        fp16=config["dtype"] == "float16", bf16=config["dtype"] == "bfloat16",
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, report_to="none",
        seed=step["seed"], data_seed=step["seed"], remove_unused_columns=False,
        dataloader_num_workers=0, dataloader_drop_last=False)

    class LedgerCollator(Collator):
        def __call__(self, batch):
            result = super().__call__(batch)
            if all("_run_key" in ex for ex in batch):
                result["_run_keys"] = [ex["_run_key"] for ex in batch]
            return result

    class Progress(TrainerCallback):
        trainer = None

        def on_step_end(self, args, state, control, **kwargs):
            skipped = bool(self.trainer.accelerator.optimizer_step_was_skipped)
            ledger.commit(state.global_step, skipped=skipped)
            memory.record("after_optimizer_step", step=state.global_step, skipped=skipped)
            if skipped:
                raise RuntimeError("optimizer step skipped; refusing a numerically invalid run")
            print(json.dumps({"optimizer_step": state.global_step, "target_tokens": ledger.state["target_tokens"]}), flush=True)
            if step.get("stop_after_steps") and state.global_step >= step["stop_after_steps"]:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            ledger.checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", train_set_sha, training_config_sha)

    progress = Progress()

    class LedgerTrainer(TrainedPositionLossMixin, Trainer):
        def __init__(self, **kwargs):
            self.loss_policy = step.get("loss_policy", FULL)
            encoded = kwargs["train_dataset"]
            if len(encoded) != len(train):
                raise ValueError("AgentTune changed training membership")
            actual = {}
            self.row_counts = {}
            for row, ex in zip(train, encoded):
                n = sum(label != -100 for label in ex["labels"])
                if n != supervised_count(row) or len(ex["input_ids"]) != input_count(row):
                    raise ValueError("fresh factory encoding differs from audited token plan")
                identity = object_sha([row["question_id"], ex["input_ids"], ex["labels"]])
                actual[identity] = n
                ex["_run_key"] = identity
                self.row_counts[identity] = (len(ex["input_ids"]), n)
            if sum(actual.values()) != plan["a_unique_train_targets_per_pass"]:
                raise ValueError("actual training-set denominator differs from planned unique targets")
            atomic_json(out / "training-set.json", {**plan, "train_set_sha256": train_set_sha,
                        "fresh_encoding_unique_targets_per_pass": sum(actual.values()), "fresh_encoding_verified": True})
            kwargs["data_collator"] = LedgerCollator(tokenizer.pad_token_id)
            kwargs["callbacks"] = [progress]
            super().__init__(**kwargs)
            progress.trainer = self

        def training_step(self, model, inputs, num_items_in_batch=None):
            identities = inputs.pop("_run_keys")
            counts = [self.row_counts[k] for k in identities]
            padded = int(inputs["input_ids"].numel()) - sum(n for n, _ in counts)
            ledger.batch(identities, [n for n, _ in counts], [n for _, n in counts], padded)
            loss = super().training_step(model, inputs, num_items_in_batch)
            if not math.isfinite(float(loss.detach().cpu())):
                raise RuntimeError("nonfinite training loss")
            return loss

        def train(self, *a, **kw):
            return super().train(*a, resume_from_checkpoint=str(resume) if resume else None, **kw)

    factory = AuditedSFTFactory(model, tokenizer, args, max_len=config["max_input_tokens"],
                                trainer_class=LedgerTrainer, eval_dataset=val_encoded or None)
    wrapper = prepare_trainer(config["model"], train, factory, str(out / "agenttune"))
    started = time.monotonic()
    result = wrapper.train()  # create_distill_trainer -> Project.distill -> our audited factory -> HF Trainer
    trainer = factory.trainer
    if ledger.state["last_global_step"] != trainer.state.global_step or ledger.state["optimizer_steps"] != trainer.state.global_step:
        raise RuntimeError("optimizer/token ledger did not track every committed step")
    if not math.isfinite(float(result.training_loss)):
        raise RuntimeError("nonfinite final loss")
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters() if p.requires_grad):
        raise RuntimeError("nonfinite adapter weights")
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    atomic_json(adapter / "_COMPLETE", {"global_step": trainer.state.global_step})
    expected_step = step.get("stop_after_steps") or (step["max_steps"] if step["max_steps"] > 0 else None)
    if expected_step and trainer.state.global_step != expected_step:
        raise RuntimeError("unexpected completed optimizer step")
    report = {"backend": "AgentTune.create_distill_trainer/Project.distill/AuditedSFTFactory", "model": config["model"],
              "model_revision": config["model_revision"], "train_set_sha256": train_set_sha,
              "training_set": plan, "global_step": trainer.state.global_step,
              "trainable_parameters": trainable, "total_parameters": total,
              "resumed_from": str(resume) if resume else None, "loss": float(result.training_loss),
              "runtime_seconds": time.monotonic() - started, "token_ledger": ledger.state,
              "adapter_sha256": fingerprint(adapter / "adapter_model.safetensors"),
              "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(), "checkpoint_optimizer_saved": True}
    report.update(loss_policy=step.get("loss_policy", FULL), gpu_memory_file=memory.path.as_posix())
    memory.record("training_complete", step=trainer.state.global_step)
    atomic_json(out / ("train-report-resumed.json" if resume else "train-report.json"), report)
    print(json.dumps({k: report[k] for k in ("backend", "global_step", "loss", "runtime_seconds", "adapter_sha256")}), flush=True)
    return report
