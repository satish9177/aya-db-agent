import json
import math
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration

A500 = Path("experiments/sql-a500/REGISTRATION.json")
BASELINES = Path("experiments/sql-dev-baselines/REGISTRATION.json")
DRAFT = Path("experiments/sqale-500/REGISTRATION-DRAFT.json")
G0_02 = Path("experiments/sql-g0-02/REGISTRATION.json")


def test_a500_and_baseline_registrations_follow_draft_and_amendment_7():
    # Both approved and launched once from 15b12cc; amendment 9 later changed pinned runner sources, so the
    # frozen registrations are checked by their own SHA256 and loaded without today's source pins.
    assert fingerprint(A500) == "c3b006ce2377c24762a8d57bdddd5a3499e9e817c7bb43ef0d16a21af4e47e3d"
    assert fingerprint(BASELINES) == "53049d18e68d5645e3a656e63a00232b6fe8ce86c49f2ee55979e02d89410ee9"
    a500, baselines = load_registration(A500, check_files=False), load_registration(BASELINES, check_files=False)
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    g0 = json.loads(G0_02.read_text(encoding="utf-8"))
    assert fingerprint(G0_02) == "e86597148c4ccdeb645777fb85f4090ec0eb1156789172165b425242d70f2580"

    # Same model, hardware, account and evaluation protocol for every compared condition.
    for reg in (a500, baselines):
        assert reg["model_config"] == g0["model_config"] == draft["model_config"]
        assert reg["hardware"] == "t4-small" and reg["expected_account"] == "hf-user"
        assert reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL"
        assert reg["timeout_seconds"] / 3600 * .40 + reg["first_month_storage_reserve_usd"] <= reg["budget_usd"]
        assert not any(a["path"].startswith("data/bird") or "q2" in a["path"] for a in reg["assets"])
        assert reg["source_sha256"]["experiments/sqale-sft/AMENDMENT-7-a500-stage.md"]
    dev_steps = [s for reg in (a500, baselines) for s in reg["steps"] if s["action"] == "evaluate"]
    assert [s["name"] for s in dev_steps] == ["dev-a500-nf4", "dev-base-nf4", "dev-pilot-nf4"]
    for step in dev_steps:
        assert step["tasks"] == draft["planned_dev_screen"]["file"] and step["purpose"] == "dev"
        assert (step["samples"], step["temperature"], step["seed"], step["inference_logits_to_keep"]) == (1, 0.0, 42, 1)
    assert fingerprint(Path(draft["planned_dev_screen"]["file"])) == draft["planned_dev_screen"]["sha256"]

    # Training follows the frozen 500 draft; the dev step reloads exactly the final update.
    train, dev = a500["steps"]
    planned = draft["planned_training"]
    assert train["episodes"] == planned["episodes"] and fingerprint(Path(train["episodes"])) == planned["episodes_sha256"]
    for key in ("seed", "validation_share", "epochs", "max_steps", "batch_size", "gradient_accumulation",
                "learning_rate", "save_steps", "keep_checkpoints"):
        assert train[key] == planned[key], key
    assert train["loss_policy"] == "trained-positions-v1"
    updates = 2 * math.ceil(planned["training_examples"] / planned["gradient_accumulation"])
    assert updates == 60 == a500["training_plan"]["expected_optimizer_updates"]
    assert dev["adapter"] == train["out"] + "/adapter"
    assert dev["verify_checkpoint_reload"] is True and dev["expected_checkpoint_step"] == updates
    assert a500["training_plan"]["a_unique_train_targets_per_pass"] == planned["unique_train_targets_per_pass"]

    # Baselines: untuned Base first, then the existing pinned Pilot adapter.
    base, pilot = baselines["steps"]
    assert base["adapter"] is None and pilot["adapter"] == "remote/hf/pilot/sft-pilot/adapter"
    weights = next(a for a in baselines["assets"] if a["path"].endswith("sft-pilot/adapter/adapter_model.safetensors"))
    assert weights["sha256"] == "6f2827ae9b288fb96de32ac3f0f8e258743acd61ea754739f3c67319c21e4794"
