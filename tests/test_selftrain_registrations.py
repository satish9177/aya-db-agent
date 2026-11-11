import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration
from ayadb.runinputs import training_plan

A1500 = Path("experiments/sql-a1500/REGISTRATION.json")
MATCHING = Path("data/sqale/selftrain-v1/matching.json")
REGS = {"b": Path("experiments/sql-selftrain-v1/b/REGISTRATION.json"),
        "aplus": Path("experiments/sql-selftrain-v1/aplus/REGISTRATION.json")}


def test_b_and_aplus_differ_only_in_matched_data():
    a1500 = json.loads(A1500.read_text(encoding="utf-8"))
    matching = json.loads(MATCHING.read_text(encoding="utf-8"))
    assert matching["continuations_may_train"] and matching["g2"]["passed"] and matching["post_fallback_size_gate"]["passed"]
    regs = {arm: load_registration(path) for arm, path in REGS.items()}
    train_steps = {arm: reg["steps"][0] for arm, reg in regs.items()}
    roles = {"b": "continuation_self", "aplus": "continuation_control"}
    for arm, reg in regs.items():
        train, dev = reg["steps"]
        assert reg["expected_account"] == "hf-user" and reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL"
        assert train["role"] == roles[arm] and train["validation_share"] == 0.0 and train["max_steps"] == -1
        assert train["continuation_manifest"] == MATCHING.as_posix()
        assert fingerprint(Path(train["episodes"])) == matching["datasets"][roles[arm]]["sha256"]
        weights = next(a for a in reg["assets"] if a["path"] == train["base_adapter"] + "/adapter_model.safetensors")
        assert weights["sha256"] == matching["a_adapter_sha256"] == "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
        _, validation, plan = training_plan(Path(train["episodes"]), seed=train["seed"], val_share=0.0)
        assert not validation and plan["train_targets_per_pass"] == matching["matched_targets"]
        for key in ("tasks", "purpose", "samples", "temperature", "seed", "inference_logits_to_keep"):
            assert dev[key] == a1500["steps"][1][key], key
        assert dev["verify_checkpoint_reload"] and dev["expected_checkpoint_step"] == reg["expected_optimizer_updates"]
        assert "tool_feedback" not in dev and "38/95" in reg["decision_rule"]
        assert reg["timeout_seconds"] / 3600 * .40 + reg["first_month_storage_reserve_usd"] <= reg["budget_usd"]
    common = ("seed", "epochs", "batch_size", "gradient_accumulation", "learning_rate", "loss_policy", "base_adapter",
              "save_steps", "keep_checkpoints")
    assert all(train_steps["b"][k] == train_steps["aplus"][k] for k in common)
    assert train_steps["b"]["learning_rate"] == 1e-4 and train_steps["b"]["epochs"] == 2
