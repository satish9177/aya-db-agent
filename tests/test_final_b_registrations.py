import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration

B_ADAPTER_SHA = "90b04823368f50d6ef5ce85c171fffff317b11a3bf43a591cf3952f01b076837"
SELECTION = Path("experiments/sql-final-b/SELECTION.json")
A1500_FINALS = {"bird": Path("experiments/sql-final-v1/bird/REGISTRATION.json"),
                "sqale": Path("experiments/sql-final-v1/sqale-retention/REGISTRATION.json")}
FINALS_B = {"bird": (Path("experiments/sql-final-b/bird/REGISTRATION.json"), "hf-user", 6.69),
            "sqale": (Path("experiments/sql-final-b/sqale-retention/REGISTRATION.json"), "hf-user", 3.30)}


def test_b_final_selection_is_frozen_from_dev_and_names_only_b():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert selection["frozen"] and [m["name"] for m in selection["models"]] == ["b"]
    (b,) = selection["models"]
    assert b["adapter_sha256"] == B_ADAPTER_SHA and b["checkpoint_step"] == 40
    assert selection["dev_strict"]["b"] == 40 >= 38 > selection["dev_strict"]["a1500"]
    assert selection["submitted_model"].startswith("b ")


def test_b_final_runs_b_only_with_a1500_final_protocol_within_its_reported_credit():
    for key, (path, account, credit) in FINALS_B.items():
        reg = load_registration(path)
        template = load_registration(A1500_FINALS[key], check_files=False)
        assert reg["expected_account"] == account and reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL"
        assert reg["stage"] == "final" and reg["selection_sha256"] == fingerprint(SELECTION)
        assert reg["protocol_template"]["sha256"] == fingerprint(A1500_FINALS[key])
        assert reg["model_config"] == template["model_config"] and reg["hardware"] == template["hardware"] == "t4-small"
        assert reg["scoring_limits"] == {"seconds": 60.0, "max_ops": 2_000_000_000}
        assert reg["timeout_seconds"] / 3600 * .40 + reg["first_month_storage_reserve_usd"] <= reg["budget_usd"] <= credit
        data = [a["path"] for a in reg["assets"] if a["path"].startswith("data/")]
        assert data and not any(p == "data/test.jsonl" or "q2" in p or "train" in p or "episodes" in p
                                or "dev.tasks" in p and "retention" not in p for p in data)
        a1500_steps = {s["tasks"]: s for s in template["steps"] if s["name"].endswith("-a1500")}
        assert sorted(s["tasks"] for s in reg["steps"]) == sorted(a1500_steps)
        for step in reg["steps"]:
            assert step["name"].endswith("-b") and step["purpose"] == "final" and "tool_feedback" not in step
            for k in ("samples", "temperature", "seed", "inference_logits_to_keep"):
                assert step[k] == a1500_steps[step["tasks"]][k], k
            assert (step["samples"], step["temperature"]) == (1, 0.0)
            weights = next(a for a in reg["assets"] if a["path"] == step["adapter"] + "/adapter_model.safetensors")
            assert weights["sha256"] == B_ADAPTER_SHA
            assert step["verify_checkpoint_reload"] and step["expected_checkpoint_step"] == 40
