import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import BatchEncoding, Cohere2Config, Cohere2ForCausalLM, Trainer, TrainingArguments

from ayadb.gpumemory import GpuMemoryRecorder, record_oom
from ayadb.lastposition import LastPositionHFEngine
from ayadb.positionloss import FULL, SELECTED, TrainedPositionLossMixin, selected_inputs
from ayadb.sft import LORA_TARGETS


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_model(*, dropout=0., checkpointing=False):
    torch.manual_seed(321)
    config = Cohere2Config(vocab_size=67, hidden_size=32, intermediate_size=48,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=8, max_position_embeddings=128, sliding_window=8,
                          layer_types=["sliding_attention", "full_attention"], logit_scale=.37,
                          pad_token_id=0, bos_token_id=1, eos_token_id=2, use_cache=False)
    config._attn_implementation = "eager"
    model = get_peft_model(Cohere2ForCausalLM(config), LoraConfig(r=3, lora_alpha=6,
        lora_dropout=dropout, target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM"))
    # Nonzero B exposes gradients for both A and B; a zero-initialized B hides A errors.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=.04)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.cpu()


def batch(length=23, *, second=False):
    ids = (torch.arange(length).reshape(1, -1) * 3 + 4) % 63 + 3
    labels = torch.full_like(ids, -100)
    # Separate assistant spans, masked intervening user/tool/error observations and padding.
    positions = [0, 2, 3, 8, 9, length - 2, length - 1] if not second else [0, 1, 5, 6, 7, 11, 12, 15]
    for position in positions:
        if position < length:
            labels[0, position] = ids[0, position]
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def gradients(model):
    values = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert values and all("lora_" in n for n in values)
    assert any(torch.count_nonzero(g) for n, g in values.items() if "lora_A" in n)
    return values


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_cpu_cohere2_native_loss_scaled_logits_and_all_lora_gradients_match(checkpointing, batched):
    model = tiny_model(dropout=.1, checkpointing=checkpointing).train()
    original = batch()
    if batched:
        other = batch(second=True)
        original = {k: torch.cat([v, other[k]]) for k, v in original.items()}
    selected = selected_inputs(original)
    keep = selected["logits_to_keep"]
    assert keep.dtype == torch.bool and keep.shape == original["labels"].shape
    assert int(keep.sum()) == int(original["labels"][..., 1:].ne(-100).sum())
    assert not bool(keep[0, -1]) and bool(keep[0, 1])
    assert "logits_to_keep" not in original
    head_shapes = []
    handle = model.get_base_model().lm_head.register_forward_pre_hook(lambda mod, args: head_shapes.append(tuple(args[0].shape)))
    torch.manual_seed(42)
    full = model(**original, use_cache=False)
    full_loss, full_logits = full.loss.detach().clone(), full.logits.detach().clone()
    full.loss.backward()
    full_gradients = gradients(model)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(42)
    reduced = model(**selected, use_cache=False)
    torch.testing.assert_close(reduced.loss, full_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(reduced.logits, full_logits[keep], rtol=1e-6, atol=1e-7)
    reduced.loss.backward()
    selected_gradients = gradients(model)
    assert selected_gradients.keys() == full_gradients.keys()
    for name, gradient in selected_gradients.items():
        torch.testing.assert_close(gradient, full_gradients[name], rtol=2e-5, atol=2e-7)
    handle.remove()
    assert (*original["labels"].shape, 32) in head_shapes
    assert (int(keep.sum()), 32) in head_shapes  # Projection really shrank, not sliced after logits.


def test_cpu_native_trainer_accumulation_counts_shifted_targets_across_unequal_masks(tmp_path):
    class PathTrainer(TrainedPositionLossMixin, Trainer):
        pass
    baseline = tiny_model()
    samples = [batch(17), batch(23, second=True)]
    expected_count = sum(int(s["labels"][..., 1:].ne(-100).sum()) for s in samples)
    results = []
    for policy in (FULL, SELECTED):
        model = copy.deepcopy(baseline).train()
        args = TrainingArguments(output_dir=str(tmp_path / policy), use_cpu=True,
                                 per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                 report_to="none", disable_tqdm=True)
        trainer = PathTrainer(model=model, args=args)
        trainer.loss_policy = policy
        trainer.current_gradient_accumulation_steps = 2
        count = trainer._get_num_items_in_batch(samples, torch.device("cpu"))
        assert count is not None and int(count) == expected_count
        losses = [trainer.training_step(model, dict(sample), num_items_in_batch=count) for sample in samples]
        results.append((sum(losses), gradients(model)))
    padded = {}
    for key in samples[0]:
        value = -100 if key == "labels" else 0
        padded[key] = torch.cat([torch.nn.functional.pad(s[key], (0, 23 - s[key].shape[1]), value=value) for s in samples])
    whole = copy.deepcopy(baseline).train()
    complete = whole(**padded, num_items_in_batch=expected_count, use_cache=False)
    complete.loss.backward()
    complete_gradients = gradients(whole)
    for loss, grad in results:
        torch.testing.assert_close(loss, complete.loss.detach(), rtol=1e-6, atol=1e-7)
        assert grad.keys() == complete_gradients.keys()
        for name in grad:
            torch.testing.assert_close(grad[name], complete_gradients[name], rtol=3e-5, atol=3e-7)


def test_cpu_validation_loss_matches_full_and_keeps_no_vocabulary_logits(tmp_path):
    # The 500 stage validates every 50 updates; G0-02 had no validation, so test that path on CPU.
    from ayadb.sft import Collator

    class PathTrainer(TrainedPositionLossMixin, Trainer):
        pass

    rows = []
    for length, second in ((17, False), (23, True), (20, False)):
        sample = batch(length, second=second)
        rows.append({"input_ids": sample["input_ids"][0].tolist(), "labels": sample["labels"][0].tolist()})
    baseline = tiny_model()
    results = {}
    for policy in (FULL, SELECTED):
        model = copy.deepcopy(baseline)
        args = TrainingArguments(output_dir=str(tmp_path / policy), use_cpu=True, per_device_eval_batch_size=1,
                                 report_to="none", disable_tqdm=True, remove_unused_columns=False)
        trainer = PathTrainer(model=model, args=args, eval_dataset=rows, data_collator=Collator(0))
        trainer.loss_policy = policy
        head_rows = []
        handle = model.get_base_model().lm_head.register_forward_pre_hook(
            lambda module, inputs: head_rows.append(inputs[0].shape[-2] if inputs[0].ndim == 3 else inputs[0].shape[0]))
        results[policy] = (trainer.evaluate()["eval_loss"], head_rows)
        handle.remove()
    full_loss, full_rows = results[FULL]
    selected_loss, selected_rows = results[SELECTED]
    assert torch.isfinite(torch.tensor(selected_loss))
    torch.testing.assert_close(torch.tensor(selected_loss), torch.tensor(full_loss), rtol=1e-6, atol=1e-7)
    assert full_rows == [17, 23, 20]
    assert selected_rows == [sum(v != -100 for v in row["labels"][1:]) for row in rows]


def test_boolean_selection_handles_unequal_rows_and_empty_accumulation_microbatch():
    inputs = batch()
    other = batch(second=True)
    inputs = {k: torch.cat([v, other[k]]) for k, v in inputs.items()}
    mask = selected_inputs(inputs)["logits_to_keep"]
    assert mask.shape == (2, 23) and int(mask[0].sum()) != int(mask[1].sum())
    empty = batch()
    empty["labels"].fill_(-100)
    with pytest.raises(ValueError, match="no trained"):
        selected_inputs(empty)
    model = tiny_model()
    out = model(**selected_inputs(empty, num_items_in_batch=5), num_items_in_batch=5)
    assert torch.isfinite(out.loss) and float(out.loss.detach()) == 0.
    with pytest.raises(ValueError, match="denominator"):
        selected_inputs(inputs, num_items_in_batch=1)


@pytest.mark.parametrize("peft", [False, True])
def test_generate_prefill_projects_last_position_for_base_and_lora(peft):
    adapted = tiny_model().eval()
    model = adapted if peft else adapted.get_base_model()
    shapes, prefill = [], []
    forward_model = adapted.get_base_model()
    head_hook = forward_model.lm_head.register_forward_pre_hook(lambda mod, args: shapes.append(tuple(args[0].shape)))
    forward_hook = forward_model.register_forward_pre_hook(lambda mod, args, kw: prefill.append(
        (kw["input_ids"].shape[-1], kw.get("logits_to_keep"))), with_kwargs=True)
    ids = batch(64)["input_ids"]
    class Tokenizer:
        eos_token_id = 2
        pad_token_id = 0
        def convert_tokens_to_ids(self, token): return None
        def __call__(self, *args, **kwargs): return BatchEncoding({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
        def decode(self, ids, **kwargs): return "generated"
    engine = LastPositionHFEngine(model, Tokenizer(), [], max_new_tokens=2)
    engine.render = lambda prompts, tools: "synthetic long prefill"
    engine.generate([], [], {"temperature": 0.})
    head_hook.remove(); forward_hook.remove()
    assert prefill[0] == (64, 1) and all(s[-2] == 1 for s in shapes)
    assert engine.calls[-1]["logits_to_keep"] == 1


def test_gpu_memory_snapshots_and_oom_handler_preserve_peaks_without_cuda(tmp_path):
    class Cuda:
        def reset_peak_memory_stats(self): pass
        def memory_allocated(self): return 11
        def memory_reserved(self): return 13
        def max_memory_allocated(self): return 17
        def max_memory_reserved(self): return 19
        def mem_get_info(self): return 21, 40
    recorder = GpuMemoryRecorder(tmp_path / "memory.jsonl", phase="fixture", cuda=Cuda())
    recorder.reset()
    recorder.record("after_model_load")
    recorder.record("after_optimizer_step", step=1)
    with pytest.raises(torch.OutOfMemoryError, match="synthetic"):
        with record_oom(recorder):
            raise torch.OutOfMemoryError("synthetic")
    rows = [json.loads(s) for s in recorder.path.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["after_model_load", "after_optimizer_step", "out_of_memory"]
    assert rows[-1]["peak_allocated_bytes"] == 17 and rows[-1]["device_free_bytes"] == 21
    class Broken(Cuda):
        def mem_get_info(self): raise RuntimeError("synthetic unavailable driver stats")
    broken = GpuMemoryRecorder(tmp_path / "broken-memory.jsonl", phase="fixture", cuda=Broken())
    with pytest.raises(torch.OutOfMemoryError, match="original"):
        with record_oom(broken):
            raise torch.OutOfMemoryError("original")
    assert json.loads(broken.path.read_text())["peak_allocated_bytes"] == 17


def test_short_native_loss_comparison_on_tiny_cpu_model_and_cuda_guard(monkeypatch, tmp_path):
    from ayadb.losscheck import compare_native_losses, run_gpu_loss_check
    model = tiny_model().train()
    inputs = batch()
    report = compare_native_losses(model, inputs, rtol=1e-6, atol=1e-7)
    assert report["passed"] and model.training
    assert report["selected_projected_positions"] == int(inputs["labels"][..., 1:].ne(-100).sum())
    assert report["full_projected_positions"] == 23
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        run_gpu_loss_check({}, {"out": str(tmp_path / "never-created")})
    assert not (tmp_path / "never-created").exists()


def test_g0_02_is_separate_same_pins_and_schedule_with_one_bounded_short_check():
    from ayadb.promptquality import fingerprint
    from ayadb.runconfig import load_registration
    original_path = Path("experiments/sql-g0/REGISTRATION.json")
    assert fingerprint(original_path) == "84799ec8cbdd3a57602b09084e3bf8aec1f82f572b5baabeb71a5d51828c8c81"
    old = json.loads(original_path.read_text(encoding="utf-8"))
    # G0-02 ran on 2026-11-07; its source pins were verified at launch. Later launcher fixes
    # (amendment 7) change pinned files, so check the frozen registration itself, not today's source.
    g0_02_path = Path("experiments/sql-g0-02/REGISTRATION.json")
    assert fingerprint(g0_02_path) == "e86597148c4ccdeb645777fb85f4090ec0eb1156789172165b425242d70f2580"
    new = load_registration(g0_02_path, check_files=False)
    assert new["model_config"] == old["model_config"]
    assert new["budget_usd"] == 3 and new["timeout_seconds"] == 7200 and new["hardware"] == "t4-small"
    assert new["expected_account"] == "hf-user" and new["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL"
    assert len(new["steps"]) == 6 and new["steps"][0]["action"] == "check_loss"
    assert new["steps"][0]["max_short_input_tokens"] == 1219
    for before, after in zip(old["steps"], new["steps"][1:]):
        expected = copy.deepcopy(before)
        for key in ("out", "resume", "adapter"):
            if expected.get(key):
                expected[key] = expected[key].replace(old["id"], new["id"])
        expected["loss_policy" if before["action"] == "train" else "inference_logits_to_keep"] = (
            SELECTED if before["action"] == "train" else 1)
        assert after == expected
    assert {a["path"] for a in old["assets"]}.issubset({a["path"] for a in new["assets"]})
    assert new["source_sha256"]["requirements-sql-runner.txt"] == old["source_sha256"]["requirements-sql-runner.txt"]
    # The user approved G0-02 and it was launched once on 2026-11-07. runs/ is local-only, so check
    # the receipt only where it exists: it must be that single approved launch.
    launch = Path("runs/hf-sql/hf-user") / new["id"] / "launch.json"
    if launch.exists():
        record = json.loads(launch.read_text(encoding="utf-8"))
        assert record["state"] == "launched" and record["job_id"] == "6aef547de6320c66fad5ff75"
        assert record["registration_sha256"] == fingerprint(Path("experiments/sql-g0-02/REGISTRATION.json"))
    invalid = copy.deepcopy(new)
    invalid["steps"][0]["rtol"] = .5
    with pytest.raises(ValueError, match="bounded whole input/tolerances"):
        temporary = Path("runs/invalid-g0-02-test-registration.json")
        temporary.write_text(json.dumps(invalid), encoding="utf-8")
        load_registration(temporary, check_files=False)
