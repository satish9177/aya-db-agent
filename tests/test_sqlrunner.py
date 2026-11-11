import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.runconfig import require_execution
from ayadb.runledger import TokenLedger
from ayadb.sftdata import ScriptedEngine
from ayadb.sqlrunner import evaluate_with_engine
from ayadb.sqlsftdata import call_text


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/sql_job.py"
    spec = importlib.util.spec_from_file_location("sql_job_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def no_real_client(*args, **kwargs):
        raise AssertionError("unit tests must inject a fake HF client")
    monkeypatch.setattr(module, "HfApi", no_real_client)
    return module


def test_wrong_account_fails_before_bucket_mutation(launcher):
    class Wrong:
        token = "fixture"
        def whoami(self): return {"name": "wrong-account"}
        def bucket_info(self, *a, **k): raise AssertionError("bucket accessed")
        def create_bucket(self, *a, **k): raise AssertionError("bucket created")
        def run_job(self, *a, **k): raise AssertionError("job started")
    with pytest.raises(PermissionError, match="expected HF account"):
        launcher.private_bucket(Wrong(), "hf-user")


def test_model_execution_and_uploads_need_exact_explicit_ack(launcher, tmp_path, monkeypatch):
    path = tmp_path / "registration.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("AYADB_APPROVED_REGISTRATION", raising=False)
    with pytest.raises(PermissionError, match="explicit approval"):
        require_execution(path)
    with pytest.raises(PermissionError, match="uploads"):
        launcher.approval(path, fingerprint(path), False)
    with pytest.raises(PermissionError, match="registration"):
        launcher.approval(path, "wrong", True)


def test_registration_rejects_unpinned_step_inputs_before_remote_activity(tmp_path):
    from ayadb.runconfig import load_registration
    source = Path("experiments/sql-g0/REGISTRATION.json")
    reg = json.loads(source.read_text(encoding="utf-8"))
    reg["steps"][0]["episodes"] = "data/unregistered.jsonl"
    path = tmp_path / "registration.json"; path.write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the pinned asset list"):
        load_registration(path, check_files=False)


def test_continuation_without_passed_post_fallback_gate_cannot_train(tmp_path):
    from ayadb.sqltraining import validate_continuation
    step = {"role": "continuation_control"}
    with pytest.raises(ValueError, match="post-fallback matching/gate"):
        validate_continuation({}, step, [], {})
    path = tmp_path / "matching.json"
    path.write_text(json.dumps({"post_fallback_size_gate": {"passed": True,
                      "verified_unique_targets_per_pass": 9, "a_unique_targets_per_pass": 100}}))
    step["continuation_manifest"] = str(path)
    with pytest.raises(ValueError, match="size gate failed"):
        validate_continuation({}, step, [], {})


def test_launch_reservation_prevents_a_second_job(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    path = tmp_path / "registration.json"; path.write_text("{}")
    calls = []
    class API:
        token = "fixture"
        def whoami(self): return {"name": "hf-user"}
        def run_job(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(id="fake-job", url="https://example.invalid/fake-job")
    api = API()
    reg = {"id": "test-one-job", "expected_account": "hf-user", "hardware": "t4-small",
           "timeout_seconds": 7200, "budget_usd": 3}
    receipt = {"registration_sha256": fingerprint(path), "git": "a" * 40,
               "bucket": "hf-user/aya-db-agent", "prefix": "sql-inputs/test-one-job/hash"}
    monkeypatch.setattr(launcher, "upload", lambda *a, **kw: (api, "fixture", receipt))
    monkeypatch.setattr(launcher, "clean_commit", lambda: "a" * 40)
    monkeypatch.setattr(launcher, "preflight", lambda *a, **kw: {"account": "hf-user"})
    launcher.launch(path, reg, fingerprint(path), allow_upload=True, allow_job=True, api=api)
    assert len(calls) == 1 and calls[0]["namespace"] == "hf-user"
    assert calls[0]["timeout"] == 7200
    assert calls[0]["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    with pytest.raises(ValueError, match="already has a launch record"):
        launcher.launch(path, reg, fingerprint(path), allow_upload=True, allow_job=True, api=api)
    assert len(calls) == 1


def test_only_completed_checkpoints_sync_to_this_run_prefix(launcher, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = Path("runs/sql/test-checkpoints")
    complete, partial = out / "training/checkpoints/checkpoint-1", out / "training/checkpoints/checkpoint-2"
    complete.mkdir(parents=True); partial.mkdir(parents=True)
    (complete / "_COMPLETE").write_text("{}")
    (complete / "optimizer.pt").write_bytes(b"optimizer fixture")
    (partial / "optimizer.pt").write_bytes(b"incomplete fixture")
    seen = {}
    class API:
        def bucket_info(self, bucket): return SimpleNamespace(private=True)
        def sync_bucket(self, source, dest, **kw):
            seen.update(dest=dest, delete=kw["delete"], files=[p.relative_to(source).as_posix() for p in Path(source).rglob("*") if p.is_file()])
    launcher.sync_checkpoint_snapshot(API(), {"id": "test-checkpoints", "expected_account": "hf-user", "max_output_upload_bytes": 1000}, out,
                                     "hf-user/aya-db-agent")
    assert seen["dest"].endswith("/runs/sql/test-checkpoints") and seen["delete"]
    assert any("checkpoint-1/optimizer.pt" in p for p in seen["files"])
    assert not any("checkpoint-2" in p for p in seen["files"])


def test_snapshot_keeps_plain_files_named_checkpoint(launcher, tmp_path, monkeypatch):
    # G0-02 lost infer-checkpoint3-nf4/checkpoint-reload.json because the old rule matched files too.
    monkeypatch.chdir(tmp_path)
    out = Path("runs/sql/test-reload-record")
    evaluation, partial = out / "infer-checkpoint3-nf4", out / "training/checkpoints/checkpoint-7"
    evaluation.mkdir(parents=True); partial.mkdir(parents=True)
    (evaluation / "checkpoint-reload.json").write_text('{"all_tensors_match": true}')
    (evaluation / "summary.json").write_text("{}")
    (evaluation / "partial.tmp").write_text("temporary")
    (partial / "optimizer.pt").write_bytes(b"incomplete fixture")
    seen = {}
    class API:
        def bucket_info(self, bucket): return SimpleNamespace(private=True)
        def sync_bucket(self, source, dest, **kw):
            seen.update(files=sorted(p.relative_to(source).as_posix() for p in Path(source).rglob("*") if p.is_file()))
    launcher.sync_checkpoint_snapshot(API(), {"id": "test-reload-record", "expected_account": "hf-user", "max_output_upload_bytes": 1000}, out,
                                     "hf-user/aya-db-agent")
    assert seen["files"] == ["infer-checkpoint3-nf4/checkpoint-reload.json", "infer-checkpoint3-nf4/summary.json"]


def test_status_and_pull_use_account_scoped_paths_without_bucket_writes(launcher, tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    reg = {"id": "test-read-results", "expected_account": "hf-user"}
    receipt = tmp_path / "runs/hf-sql/hf-user/test-read-results/launch.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"state": "launched", "job_id": "fixture-job", "url": "https://example.invalid/fixture"}))
    seen = []
    class API:
        def inspect_job(self, **kw):
            assert kw == {"job_id": "fixture-job", "namespace": "hf-user"}
            return SimpleNamespace(id="fixture-job", status=SimpleNamespace(stage="COMPLETED"))
        def sync_bucket(self, source, dest, **kw): seen.append((source, dest, kw))
    monkeypatch.setattr(launcher, "authorized_api", lambda *a, **k: (API(), "fixture"))
    assert launcher.status_or_pull(reg)["state"] == "COMPLETED"
    result = launcher.status_or_pull(reg, pull=True)
    assert "remote" in result["pulled_to"] and seen[0][0].endswith("/runs/sql/test-read-results")
    assert not seen[0][1].startswith("hf://") and seen[0][2]["delete"] is False


def test_token_ledger_counts_committed_batches_and_survives_checkpoint(tmp_path):
    ledger = TokenLedger(tmp_path)
    ledger.batch(["a"], [10], [3], 0)
    ledger.batch(["b"], [20], [5], 2)
    ledger.commit(1)
    ledger.checkpoint(tmp_path / "checkpoint-1", "set-sha", "config-sha")
    restored = TokenLedger(tmp_path, previous=json.loads((tmp_path / "checkpoint-1/token-ledger.json").read_text()))
    restored.batch(["a"], [10], [3], 0)
    restored.commit(2)
    assert restored.state["target_tokens"] == 11 and restored.state["input_tokens"] == 40
    assert restored.state["unique_examples_seen"] == ["a", "b"]
    restored.batch(["c"], [100], [99], 0)
    restored.commit(3, skipped=True)
    assert restored.state["target_tokens"] == 11 and restored.state["skipped_optimizer_steps"] == 1
    assert json.loads((tmp_path / "checkpoint-1/_COMPLETE").read_text())["training_config_sha256"] == "config-sha"


def test_new_runner_executes_real_tools_and_auditkit_with_scripted_engine(tmp_path):
    db = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(db); con.executescript("CREATE TABLE items(value INTEGER); INSERT INTO items VALUES(7);"); con.close()
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    class Engine(ScriptedEngine):
        def __init__(self):
            super().__init__(tokenizer, [call_text("run_sql", {"query": "SELECT value FROM items"}),
                                       call_text("submit_sql", {"query": "SELECT value FROM items"})])
            self.calls = []
        def generate(self, *a, **kw):
            self.calls.append({"prompt_tokens": 10, "completion_tokens": 5})
            return super().generate(*a, **kw)
    task = {"id": "fixture", "prompt": "Return the stored value.", "db_path": str(db),
            "db_sha256": fingerprint(db), "reference_sql": "SELECT value FROM items"}
    result = evaluate_with_engine([task], Engine(), tmp_path / "eval")
    assert result["strict_successes"] == result["auditkit_agreement"] == 1
    assert result["generated_tokens"] == 10
    submissions = [json.loads(s) for s in (tmp_path / "eval/submissions.jsonl").read_text().splitlines()]
    assert len(submissions) == 1 and submissions[0]["args"]["query"] == "SELECT value FROM items"


def test_checkpoint_reload_compares_actual_tensors(tmp_path, monkeypatch):
    import torch
    import peft
    from safetensors.torch import save_file
    from ayadb.sqltraining import verify_reloaded_adapter
    saved = {"fixture.weight": torch.tensor([[1., 2.], [3., 4.]])}
    save_file(saved, str(tmp_path / "adapter_model.safetensors"))
    monkeypatch.setattr(peft, "get_peft_model_state_dict", lambda model: saved)
    assert verify_reloaded_adapter(object(), tmp_path)["all_tensors_match"]
    monkeypatch.setattr(peft, "get_peft_model_state_dict", lambda model: {"fixture.weight": saved["fixture.weight"] + 1})
    with pytest.raises(ValueError, match="tensor mismatch"):
        verify_reloaded_adapter(object(), tmp_path)


def test_nonfinite_inference_logits_fail_without_running_a_model():
    import torch
    from ayadb.sqlrunner import check_last_logits
    check_last_logits(SimpleNamespace(logits=torch.zeros(1, 2, 4)))
    with pytest.raises(RuntimeError, match="nonfinite inference logits"):
        check_last_logits(SimpleNamespace(logits=torch.full((1, 2, 4), float("nan"))))


def test_factory_training_resume_and_masks_with_spy_only(tmp_path, monkeypatch):
    """Exercise the actual AgentTune path/callbacks; no forward/backward/optimizer exists."""
    import torch
    import transformers
    import ayadb.sqltraining as training
    from ayadb.sqlsftdata import encode_episode, make_episode
    db = tmp_path / "factory.sqlite"
    con = sqlite3.connect(db); con.executescript("CREATE TABLE items(value INTEGER); INSERT INTO items VALUES(7);"); con.close()
    task = {"id": "factory-fixture", "question_id": "factory-fixture", "schema_id": "fixture",
            "prompt": "Return the stored value.", "reference_sql": "SELECT value FROM items", "gold_tables": ["items"], "db_path": str(db)}
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    episode = make_episode(task)
    encoded = encode_episode(episode, tokenizer)
    episode.update(n_input_tokens=len(encoded["input_ids"]), n_target_tokens=sum(v != -100 for v in encoded["labels"]))
    source = tmp_path / "episodes.jsonl"; source.write_text(json.dumps(episode) + "\n", encoding="utf-8")
    captured = []
    class Model:
        def get_nb_trainable_parameters(self): return 1, 1
        def parameters(self): return [torch.nn.Parameter(torch.zeros(1))]
        def save_pretrained(self, directory):
            Path(directory).mkdir(parents=True, exist_ok=True)
            (Path(directory) / "adapter_model.safetensors").write_bytes(b"spy; not model weights")
    model = Model()
    class SpyTrainer:
        def __init__(self, **kw):
            self.args = kw["args"]; self.train_dataset = kw["train_dataset"]; self.data_collator = kw["data_collator"]
            self.callbacks = kw["callbacks"]; self.state = SimpleNamespace(global_step=0)
            self.accelerator = SimpleNamespace(optimizer_step_was_skipped=False)
            captured.append(kw)
        def training_step(self, model, inputs, num_items_in_batch=None):
            assert "_run_keys" not in inputs
            assert torch.equal(inputs["labels"][0], torch.tensor(encoded["labels"]))
            return torch.tensor(.5)  # No model or optimizer operation.
        def train(self, resume_from_checkpoint=None):
            if resume_from_checkpoint:
                self.state.global_step = json.loads((Path(resume_from_checkpoint) / "_COMPLETE").read_text())["global_step"]
            for _ in range(self.state.global_step, self.args.max_steps):
                self.training_step(model, self.data_collator([self.train_dataset[0]]))
                self.state.global_step += 1
                control = SimpleNamespace(should_training_stop=False, should_save=False)
                for cb in self.callbacks: cb.on_step_end(self.args, self.state, control)
                checkpoint = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
                checkpoint.mkdir(parents=True, exist_ok=True)
                for name in ("optimizer.pt", "scheduler.pt"): (checkpoint / name).write_bytes(b"spy state")
                for cb in self.callbacks: cb.on_save(self.args, self.state, control)
                if control.should_training_stop: break
            return SimpleNamespace(training_loss=.5)
    monkeypatch.setattr(transformers, "Trainer", SpyTrainer)
    monkeypatch.setattr(transformers, "TrainingArguments", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    monkeypatch.setattr(training, "load_model", lambda *a, **kw: model)
    memory_events = []
    monkeypatch.setattr(training, "GpuMemoryRecorder", lambda path, **kw:
                        SimpleNamespace(path=path, reset=lambda: None,
                                        record=lambda event, **details: memory_events.append((event, details))))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    config = {"model": "existing-Tiny-Aya", "model_revision": "fixture", "tokenizer": "data/sqale/tokenizer",
              "max_input_tokens": 4096, "dtype": "float16"}
    step = {"out": str(tmp_path / "training"), "episodes": str(source), "role": "compatibility", "seed": 42,
            "epochs": 2, "max_steps": 3, "stop_after_steps": 2, "validation_share": 0,
            "batch_size": 1, "gradient_accumulation": 1, "learning_rate": .0002, "save_steps": 1, "keep_checkpoints": 3}
    first = training.run_training(config, step)
    assert first["global_step"] == 2
    resumed = {**step, "stop_after_steps": None, "resume": str(tmp_path / "training/checkpoints/checkpoint-2"), "expected_resume_step": 2}
    second = training.run_training(config, resumed)
    assert second["global_step"] == second["token_ledger"]["optimizer_steps"] == 3
    assert second["token_ledger"]["target_tokens"] == 3 * episode["n_target_tokens"]
    assert all(k["model"] is model and k["args"].save_strategy == "steps" for k in captured)
    assert [d["step"] for event, d in memory_events if event == "after_optimizer_step"] == [1, 2, 3]
    assert sum(event == "after_model_load" for event, _ in memory_events) == 2
