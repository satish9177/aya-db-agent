import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration

HINTS = Path("experiments/sql-a1500-hints/REGISTRATION.json")
A1500 = Path("experiments/sql-a1500/REGISTRATION.json")


def test_hints_registration_changes_only_the_tool_feedback():
    # Approved and launched once from e6b4269; amendment 11 later changed pinned scorer sources, so the
    # frozen registration is checked by its own SHA256 and loaded without today's source pins.
    assert fingerprint(HINTS) == "02984087902a41ba11d6c62e33743d5342e33788117fc0f455c9916be9154d6a"
    reg = load_registration(HINTS, check_files=False)
    assert fingerprint(A1500) == "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
    a1500 = json.loads(A1500.read_text(encoding="utf-8"))
    for key in ("model_config", "hardware", "expected_account", "scorer_policy", "inference_logits_to_keep",
                "required_environment", "dev_file", "dev_sha256", "format"):
        assert reg[key] == a1500[key], key
    assert reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL" and reg["stage"] == "evaluate"
    assert reg["timeout_seconds"] / 3600 * .40 + reg["first_month_storage_reserve_usd"] <= reg["budget_usd"] == 2.0
    assert not any(a["path"].startswith("data/bird") or "q2" in a["path"] or "train-" in a["path"] for a in reg["assets"])
    assert reg["source_sha256"]["experiments/sqale-sft/AMENDMENT-9-error-feedback-tools.md"]
    assert reg["source_sha256"]["src/ayadb/sqlfeedback.py"]

    # One evaluation step: A1500's exact step-180 adapter, same dev protocol, only the new tool feedback.
    (step,) = reg["steps"]
    old = a1500["steps"][1]
    for key in ("tasks", "purpose", "samples", "temperature", "seed", "inference_logits_to_keep",
                "verify_checkpoint_reload", "expected_checkpoint_step"):
        assert step[key] == old[key], key
    assert step["tool_feedback"] == "column-hints-v1" and "tool_feedback" not in old
    weights = next(a for a in reg["assets"] if a["path"] == step["adapter"] + "/adapter_model.safetensors")
    assert weights["sha256"] == reg["a1500_adapter_sha256"] == \
        "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"

    # Comparator and decision fixed before the run.
    assert reg["comparators"]["dev-a1500-nf4"]["strict"] == 37 and reg["decision_rule"]["needs_at_least"] == 42
    assert reg["after_error_analysis"]["of_which_strict"] == 0
    assert reg["offline_replay"]["identical_repeats_refused"] == 77
    assert len(reg["query_shape_diagnostic"]["complex_dev_task_ids"]) == 30
