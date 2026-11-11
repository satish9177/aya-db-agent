"""Registered short-example GPU comparison; pure comparison also supports tiny CPU tests."""
from __future__ import annotations

from pathlib import Path

import torch

from .engine import official_tokenizer
from .gpumemory import GpuMemoryRecorder, record_oom
from .positionloss import selected_inputs
from .runinputs import eligible_training_rows, read_jsonl
from .runledger import atomic_json
from .sft import Collator
from .sqlsftdata import encode_episode
from .sqltraining import load_model


def compare_native_losses(model, inputs, *, rtol=1e-4, atol=1e-4):
    """No optimizer or gradient update; compare native scaled causal losses and actual head sizes."""
    was_training = model.training
    base = model.get_base_model() if hasattr(model, "peft_config") else model
    shapes = []
    hook = base.lm_head.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape)))
    denominator = inputs["labels"][..., 1:].ne(-100).sum()
    if not int(denominator):
        hook.remove()
        raise ValueError("short loss check has no trained next-token targets")
    try:
        model.eval()  # Dropout must not make the two comparisons different stochastic draws.
        with torch.no_grad():
            original = model(**inputs, num_items_in_batch=denominator, use_cache=False)
            full = original.loss.detach().float().cpu()
            del original  # Never retain the full-vocabulary tensor during the selected forward.
            reduced = model(**selected_inputs(inputs, num_items_in_batch=denominator),
                            num_items_in_batch=denominator, use_cache=False)
            selected = reduced.loss.detach().float().cpu()
            del reduced
        torch.testing.assert_close(selected, full, rtol=rtol, atol=atol)
        if not bool(torch.isfinite(full)) or not bool(torch.isfinite(selected)):
            raise ValueError("nonfinite native loss comparison")
        full_positions = inputs["labels"].numel()
        projected = [torch.tensor(shape[:-1]).prod().item() for shape in shapes]
        if projected != [full_positions, int(denominator)]:
            raise ValueError("loss check projected unexpected positions")
        return {"passed": True, "full_loss": float(full), "selected_loss": float(selected),
                "absolute_difference": float((selected - full).abs()), "rtol": rtol, "atol": atol,
                "full_projected_positions": full_positions, "selected_projected_positions": int(denominator),
                "optimizer_steps": 0, "native_cohere2_loss": True}
    finally:
        hook.remove()
        model.train(was_training)


def run_gpu_loss_check(config, step):
    if not torch.cuda.is_available():
        raise RuntimeError("registered loss check requires CUDA; refusing CPU fallback")
    out = Path(step["out"])
    if out.exists():
        raise ValueError("loss-check output exists; never repeat the registered check")
    out.mkdir(parents=True)
    memory = GpuMemoryRecorder(out / "gpu-memory.jsonl", phase=step["name"])
    with record_oom(memory):
        rows, _ = eligible_training_rows(read_jsonl(Path(step["episodes"])))
        if len(rows) != 1 or rows[0]["n_input_tokens"] > step["max_short_input_tokens"]:
            raise ValueError("loss check requires its one frozen, whole short training example")
        tokenizer = official_tokenizer(config["tokenizer"])
        encoded = encode_episode(rows[0], tokenizer, config["max_input_tokens"])
        if encoded is None or len(encoded["input_ids"]) != rows[0]["n_input_tokens"]:
            raise ValueError("short example encoding changed")
        if sum(v != -100 for v in encoded["labels"]) != rows[0]["n_target_tokens"]:
            raise ValueError("short example masks changed")
        from transformers import set_seed
        set_seed(step["seed"])
        memory.reset()
        model = load_model(config, trainable=True)
        memory.record("after_model_load")
        inputs = {k: v.to(model.device) for k, v in Collator(tokenizer.pad_token_id)([encoded]).items()}
        with torch.autocast(device_type="cuda", dtype=getattr(torch, config["dtype"])):
            report = compare_native_losses(model, inputs, rtol=step["rtol"], atol=step["atol"])
        report.update(example_id=rows[0]["id"], gpu_memory=memory.record("loss_comparison_complete"))
        atomic_json(out / "loss-comparison.json", report)
        return report
