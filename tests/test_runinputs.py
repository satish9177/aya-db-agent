import copy
import itertools
import json
import random

import pytest

from ayadb.birdprepare import task_prompt
from ayadb.continuation import match_control
from ayadb.runinputs import eligible_training_rows, evaluation_tasks, read_jsonl, split_training_rows, training_plan
from ayadb.tokenbudget import match_control as strict_match
from ayadb.tokenbudget import targets


def ex(q, n):
    return {"id": q, "question_id": q, "input_ids": list(range(n + 1)), "labels": [-100] + [1] * n}


def test_fallback_drops_fewest_questions_from_both_arms_and_rechecks_gate():
    a = [ex(str(i), 100) for i in range(5)]
    b = [ex(str(i), 2) for i in range(5)]
    original = copy.deepcopy((a, b))
    matched, retained, report = match_control(a, b, a_training_examples=[ex("A", 100)])
    assert len(report["dropped_question_ids"]) == 4 and report["question_coverage"] == .2
    assert {e["question_id"] for e in matched} == {e["question_id"] for e in retained}
    assert sum(map(targets, matched)) == sum(map(targets, retained)) == 2
    assert len(report["mask_adjustments"]) == 1
    assert not report["continuations_may_train"]
    assert (a, b) == original


def test_minimum_removal_matches_exhaustive_subset_oracle_and_is_deterministic():
    rng = random.Random(20261107)
    for n in range(1, 7):
        for _ in range(15):
            a = [ex(str(i), rng.randint(1, 15)) for i in range(n)]
            b = [ex(str(i), rng.randint(1, 10)) for i in range(n)]
            _, kept, report = match_control(a, b)
            best = None
            for size in range(n, 0, -1):
                for subset in itertools.combinations(range(n), size):
                    q = set(map(str, subset))
                    try:
                        strict_match([e for e in a if e["question_id"] in q], [e for e in b if e["question_id"] in q])
                    except ValueError:
                        continue
                    best = size
                    break
                if best is not None:
                    break
            assert len({e["question_id"] for e in kept}) == best
            assert match_control(list(reversed(a)), list(reversed(b)))[2]["dropped_question_ids"] == report["dropped_question_ids"]


def test_fallback_does_not_hide_missing_control_questions():
    with pytest.raises(ValueError, match="same nonempty question set"):
        match_control([ex("a", 3)], [ex("a", 3), ex("b", 3)])


def test_actual_train_gate_excludes_validation_and_preserves_legacy_split_order(tmp_path):
    rows = [{"id": str(i), "task_id": str(i), "question_id": str(i), "format": "sql-submit-v1",
             "prompt": "question " + str(i), "messages": [], "db_sha256": "db", "n_input_tokens": 30,
             "n_target_tokens": i + 1} for i in range(20)]
    path = tmp_path / "episodes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    train, validation, plan = training_plan(path)
    assert len(train) == 19 and len(validation) == 1
    expected = list(range(20)); random.Random(42).shuffle(expected)
    assert plan["validation_episode_ids"] == [str(expected[0])]
    assert plan["a_unique_train_targets_per_pass"] == sum(r["n_target_tokens"] for r in train)
    assert plan["a_unique_train_targets_per_pass"] + plan["validation_targets_per_pass"] == 210
    assert plan["minimum_unique_verified_b_targets_per_pass"] == (plan["a_unique_train_targets_per_pass"] + 9) // 10


def test_final_bird_cannot_enter_train_or_dev_and_unicode_evidence_is_exact(tmp_path):
    question, evidence = "Which school is in S\u00e3o Paulo?", "\u2018Eligibility\u2019 means eligible / enrolled."
    row = {"id": "bird:1", "source": "bird-mini-dev", "dataset_usage": "final_only", "split": "final",
           "format": "sql-submit-v1", "question_id": "bird:1", "prompt": task_prompt(question, evidence)}
    path = tmp_path / "bird.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert read_jsonl(path)[0]["prompt"] == question + "\n\nExternal Knowledge:\n" + evidence
    assert task_prompt(question, "") == question
    with pytest.raises(ValueError, match="final-only"):
        eligible_training_rows([row])
    with pytest.raises(ValueError, match="final tasks"):
        evaluation_tasks(path, purpose="dev")
    assert evaluation_tasks(path, purpose="final", selection_sha256="a" * 64) == [row]
    with pytest.raises(ValueError, match="sealed"):
        read_jsonl("data/test.jsonl")


def test_runtime_oracle_drift_fails_before_any_model_call(tmp_path):
    import sqlite3
    from ayadb.runinputs import verify_runtime_oracles
    db = tmp_path / "oracle.sqlite"
    con = sqlite3.connect(db); con.executescript("CREATE TABLE t(x INTEGER); INSERT INTO t VALUES(7);"); con.close()
    row = {"id": "oracle-fixture", "db_path": str(db), "reference_sql": "SELECT x FROM t", "gold_rows": [[7]]}
    path = tmp_path / "tasks.jsonl"; path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    reg = {"steps": [{"tasks": str(path)}, {"tasks": str(path)}]}
    assert verify_runtime_oracles(reg) == {"references_checked": 1, "model_calls": 0}
    row["gold_rows"] = [[8]]; path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from prepared gold"):
        verify_runtime_oracles(reg)
