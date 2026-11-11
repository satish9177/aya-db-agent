import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration
from ayadb.runinputs import read_jsonl

V4 = Path("experiments/sql-a1500v4/REGISTRATION.json")
A1500 = Path("experiments/sql-a1500/REGISTRATION.json")
V3_TRAIN = Path("data/sqale/data-build-v3/train-1500.episodes.jsonl")
V4_TRAIN = Path("data/sqale/data-build-v4/train-1500.episodes.jsonl")


def test_a1500v4_registration_changes_only_the_training_file():
    # Approved and launched once from 2a5b81e; amendment 11 later changed pinned scorer sources, so the
    # frozen registration is checked by its own SHA256 and loaded without today's source pins.
    assert fingerprint(V4) == "516dc07677efb90e1ed6c50188c4689c71e386881a6756999a377e5285aaad48"
    reg = load_registration(V4, check_files=False)
    assert fingerprint(A1500) == "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
    a1500 = json.loads(A1500.read_text(encoding="utf-8"))
    for key in ("model_config", "hardware", "expected_account", "scorer_policy", "inference_logits_to_keep",
                "required_environment", "dev_file", "dev_sha256", "max_input_upload_bytes",
                "max_output_upload_bytes", "budget_usd", "timeout_seconds", "format"):
        assert reg[key] == a1500[key], key
    assert reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL" and reg["stage"] == "train"
    assert not any(a["path"].startswith("data/bird") or "q2" in a["path"] for a in reg["assets"])
    assert reg["source_sha256"]["experiments/sqale-sft/AMENDMENT-10-recovery-demos.md"]
    assert reg["source_sha256"]["src/ayadb/sqlmutate_v4.py"]

    train, dev = reg["steps"]
    old_train, old_dev = a1500["steps"]
    assert train["episodes"] == V4_TRAIN.as_posix() and fingerprint(V4_TRAIN) == reg["training_plan"]["source_sha256"]
    for key in ("seed", "validation_share", "epochs", "max_steps", "batch_size", "gradient_accumulation",
                "learning_rate", "save_steps", "keep_checkpoints", "eval_steps", "loss_policy"):
        assert train[key] == old_train[key], key
    assert not train.get("base_adapter") and not train.get("resume") and "tool_feedback" not in dev
    for key in ("tasks", "purpose", "samples", "temperature", "seed", "inference_logits_to_keep",
                "verify_checkpoint_reload", "expected_checkpoint_step"):
        assert dev[key] == old_dev[key], key
    assert reg["training_plan"]["expected_optimizer_updates"] == 180 and reg["training_plan"]["same_split_as_a1500"]
    assert reg["comparators"]["dev-a1500-nf4"]["strict"] == 37
    assert reg["decision_rule"]["a1500v4_replaces_if_at_least"] == 38


def test_v4_file_keeps_v3_questions_and_never_trains_on_the_mistake():
    v3, v4 = read_jsonl(V3_TRAIN), read_jsonl(V4_TRAIN)
    assert [r["id"] for r in v3] == [r["id"] for r in v4]
    assert [r["question_id"] for r in v3] == [r["question_id"] for r in v4]
    assert [r["prompt"] for r in v3] == [r["prompt"] for r in v4]
    changed = [(a, b) for a, b in zip(v3, v4) if a != b]
    assert {b["v4_change"] for _, b in changed} == {"recheck_repair", "plain_repair", "stepwise"}
    for old, new in changed:
        assert new["reference_sql"] == old["reference_sql"] and new["db_sha256"] == old["db_sha256"]
        calls = [m for m in new["messages"] if m["role"] == "assistant"]
        assert calls[-1]["train"] and '"submit_sql"' in calls[-1]["content"]
        if new["v4_change"] == "recheck_repair":
            bad = next(i for i, m in enumerate(calls) if not m["train"])
            assert '"run_sql"' in calls[bad]["content"] and '"describe_table"' in calls[bad + 1]["content"]
            assert calls[bad + 1]["train"] and new["repair_error"].startswith("no such column")
        if new["v4_change"] == "stepwise":
            assert all(m["train"] for m in calls) and sum('"run_sql"' in m["content"] for m in calls) == 2
