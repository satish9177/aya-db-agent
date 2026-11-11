import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration
from ayadb.runinputs import evaluation_tasks, read_jsonl

A1500_ADAPTER_SHA = "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
SELECTION = Path("experiments/sql-final-v1/SELECTION.json")
FINALS = [Path("experiments/sql-final-v1/bird/REGISTRATION.json"),
          Path("experiments/sql-final-v1/sqale-retention/REGISTRATION.json")]
SAMPLING = [Path(f"experiments/sql-q2-sampling-v1/part-{k}/REGISTRATION.json") for k in (1, 2, 3)]


def test_final_registrations_use_the_frozen_base_and_a1500_selection_only():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert selection["frozen"] and [m["name"] for m in selection["models"]] == ["base", "a1500"]
    assert [m["adapter_sha256"] for m in selection["models"]] == [None, A1500_ADAPTER_SHA]
    sets = {}
    for path in FINALS:
        launched = path == FINALS[0]  # final-bird -02 was approved and launched from add6a23 (amendment 12)
        if launched:
            assert fingerprint(path) == "2797857acd8e681dd2cb27830b980f3db21bbba88cfc19c6344b4adaba3304a8"
        reg = load_registration(path, check_files=not launched)
        assert reg["expected_account"] == "hf-user"
        assert reg["stage"] == "final" and reg["status"] == "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL"
        assert reg["selection_sha256"] == fingerprint(SELECTION)
        assert reg["timeout_seconds"] / 3600 * .40 + reg["first_month_storage_reserve_usd"] <= reg["budget_usd"]
        assert not any(a["path"] == "data/test.jsonl" or "q2" in a["path"] or "train-" in a["path"] for a in reg["assets"])
        for step in reg["steps"]:
            assert step["purpose"] == "final" and step["selection_sha256"] == fingerprint(SELECTION)
            assert (step["samples"], step["temperature"], step["inference_logits_to_keep"]) == (1, 0.0, 1)
            assert "tool_feedback" not in step
            if step["name"].endswith("-a1500"):
                weights = next(a for a in reg["assets"] if a["path"] == step["adapter"] + "/adapter_model.safetensors")
                assert weights["sha256"] == A1500_ADAPTER_SHA and step["verify_checkpoint_reload"]
            else:
                assert step["adapter"] is None
            sets.setdefault(step["tasks"], set()).add(step["name"].rsplit("-", 1)[1])
    assert all(models == {"base", "a1500"} for models in sets.values())
    assert sorted(sets) == ["data/bird/final-v1/tasks.jsonl", "data/sqale/data-build-v3/final.tasks.jsonl",
                            "data/sqale/data-build-v3/retention/dev.tasks.jsonl", "data/sqale/data-build-v3/retention/mh.tasks.jsonl"]


def test_sampling_covers_every_q2_question_once_with_a1500_only():
    covered = []
    for k, path in enumerate(SAMPLING, start=1):
        reg = load_registration(path)
        (step,) = reg["steps"]
        assert reg["stage"] == "sample" and step["purpose"] == "sampling" and reg["expected_account"] == "hf-user"
        assert (step["samples"], step["temperature"], step["seed"]) == (2, 0.8, 42)
        weights = next(a for a in reg["assets"] if a["path"] == step["adapter"] + "/adapter_model.safetensors")
        assert weights["sha256"] == A1500_ADAPTER_SHA
        assert not any(a["path"].startswith("data/bird") or "final" in a["path"] or "dev.tasks" in a["path"]
                       for a in reg["assets"])
        covered += [r["id"] for r in read_jsonl(Path(step["tasks"]))]
    tie_dependent = {"schema_014507_s2_q7_f5426b14a9", "schema_009445_s1_q4_ac28623f95"}  # amendment 12
    q2 = [t["id"] for t in evaluation_tasks(Path("data/sqale/data-build-v3/q2.tasks.jsonl"), purpose="sampling")]
    assert covered == [i for i in q2 if i not in tie_dependent] and len(set(covered)) == 577
