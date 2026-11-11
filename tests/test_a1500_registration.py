import json
import math
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration

A1500 = Path("experiments/sql-a1500/REGISTRATION.json")
A500 = Path("experiments/sql-a500/REGISTRATION.json")
FROZEN_PLAN = Path("data/sqale/runner-inputs-v1/a-1500.json")


def test_a1500_registration_matches_a500_settings_and_amendment_8():
    # Approved and launched once from 67c530f; amendment 9 later changed pinned runner sources, so the
    # frozen registration is checked by its own SHA256 and loaded without today's source pins.
    assert fingerprint(A1500) == "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
    reg = load_registration(A1500, check_files=False)
    assert fingerprint(A500) == "c3b006ce2377c24762a8d57bdddd5a3499e9e817c7bb43ef0d16a21af4e47e3d"
    a500 = json.loads(A500.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))

    # Same model, account, hardware and evaluation protocol as A500.
    for key in ("model_config", "hardware", "expected_account", "scorer_policy", "inference_logits_to_keep",
                "required_environment", "dev_file", "dev_sha256", "max_input_upload_bytes", "max_output_upload_bytes"):
        assert reg[key] == a500[key], key
    assert reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL" and reg["stage"] == "train"
    assert reg["timeout_seconds"] / 3600 * .40 + reg["first_month_storage_reserve_usd"] <= reg["budget_usd"] == 4.0
    assert not any(a["path"].startswith("data/bird") or "q2" in a["path"] for a in reg["assets"])
    assert reg["source_sha256"]["experiments/sqale-sft/AMENDMENT-8-a1500-stage.md"]
    assert reg["source_sha256"]["src/ayadb/queryshape.py"]

    # Training: only the data differs from A500; fresh from base; 180 updates; exact final reload.
    train, dev = reg["steps"]
    old_train, old_dev = a500["steps"]
    assert train["episodes"] == "data/sqale/data-build-v3/train-1500.episodes.jsonl"
    assert fingerprint(Path(train["episodes"])) == frozen["source_sha256"] == reg["training_plan"]["source_sha256"]
    for key in ("seed", "validation_share", "epochs", "max_steps", "batch_size", "gradient_accumulation",
                "learning_rate", "save_steps", "keep_checkpoints", "eval_steps", "loss_policy"):
        assert train[key] == old_train[key], key
    assert not train.get("base_adapter") and not train.get("resume")
    updates = old_train["epochs"] * math.ceil(frozen["train_examples"] / old_train["gradient_accumulation"])
    assert updates == 180 == reg["training_plan"]["expected_optimizer_updates"]
    assert dev["adapter"] == train["out"] + "/adapter"
    assert dev["verify_checkpoint_reload"] is True and dev["expected_checkpoint_step"] == updates
    for key in ("tasks", "purpose", "samples", "temperature", "seed", "inference_logits_to_keep"):
        assert dev[key] == old_dev[key], key

    # Decisions and the descriptive shape diagnostic are fixed before the run.
    assert reg["stage_rule"]["a500_strict"] == 28 and reg["stage_rule"]["a1500_needs_at_least"] == 33
    assert reg["comparators"]["dev-a500-nf4"]["strict"] == 28
    shape = reg["query_shape_diagnostic"]
    assert list(shape["dev_counts"].values()) == [37, 5, 23, 10, 20] and len(shape["complex_dev_task_ids"]) == 30
