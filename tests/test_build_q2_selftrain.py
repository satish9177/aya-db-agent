"""Amendment 11 self-training build on scripted sampled attempts (no model): filter, rebuild, match, gates."""
import importlib.util
import json
import sqlite3
from pathlib import Path

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.sftdata import ScriptedEngine
from ayadb.sqlrunner import evaluate_with_engine
from ayadb.sqlsftdata import call_text


def load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_q2_selftrain.py"
    spec = importlib.util.spec_from_file_location("build_q2_selftrain_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selftrain_build_keeps_clean_verified_attempts_and_matches_control(tmp_path):
    db = tmp_path / "q2.sqlite"
    con = sqlite3.connect(db)
    con.executescript("CREATE TABLE items(id INTEGER, value INTEGER); INSERT INTO items VALUES (1, 7), (2, 9);")
    con.close()
    reference = "SELECT value FROM items ORDER BY id"
    task = {"id": "q2-fixture", "question_id": "q2-fixture", "split": "q2", "format": "sql-submit-v1",
            "schema_id": "fixture", "prompt": "List the item values in id order.", "phrasing": "verbose",
            "phrasings": {"verbose": "List the item values in id order.", "structured": "Values of items, ordered by id.",
                          "short_ambiguous": "values?"},
            "db_path": str(db), "db_sha256": fingerprint(db), "reference_sql": reference, "gold_tables": ["items"]}
    tasks_file = tmp_path / "part.tasks.jsonl"
    tasks_file.write_text(json.dumps(task) + "\n", encoding="utf-8")
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    attempts = [
        [call_text("list_tables", {}), call_text("run_sql", {"query": "SELECT val FROM items"}),
         call_text("run_sql", {"query": reference}), call_text("submit_sql", {"query": reference})],   # recovery
        [call_text("list_tables", {}), call_text("run_sql", {"query": reference}), call_text("submit_sql", {"query": reference})],
        [call_text("list_tables", {}), call_text("run_sql", {"query": reference}), call_text("submit_sql", {"query": reference})],  # duplicate
        [call_text("submit_sql", {"query": "SELECT 7"})],                                                # wrong
    ]

    class Engine(ScriptedEngine):
        def __init__(self, script):
            super().__init__(tokenizer, list(script))
            self.calls = []

        def generate(self, *a, **kw):
            self.calls.append({"prompt_tokens": 10, "completion_tokens": 5})
            return super().generate(*a, **kw)

    root = tmp_path / "remote"
    lines = []
    for k, script in enumerate(attempts):
        out = tmp_path / f"attempt-{k}"
        evaluate_with_engine([task], Engine(script), out, purpose="sampling")
        record = json.loads((out / "trajectories.jsonl").read_text(encoding="utf-8"))
        record["trajectory_id"] = record["metadata"]["task_id"] = f"q2-fixture#s{k}"
        lines.append(json.dumps(record))
    step_dir = root / "run" / "step"
    step_dir.mkdir(parents=True)
    (step_dir / "trajectories.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    builder = load_builder()
    b, aplus, report, eligibility = builder.build(parts=[("run", "step", tasks_file)], root=root, tokenizer=tokenizer)
    assert [e["eligible"] for e in eligibility] == [True, True, True, False]
    assert report["g2"]["passed"] and report["verified_episodes"] == 2 and report["verified_with_recovery"] == 1
    assert report["dropped_after_eligibility"] == {"duplicate_or_over_cap": 1}
    recovery = next(ep for ep in b if ep["contains_recovery"])
    calls = [m for m in recovery["messages"] if m["role"] == "assistant"]
    assert [m["train"] for m in calls] == [True, False, True, True]  # the failed query is untrained context
    # A+ is matched to B's trained target tokens, never using the vague phrasing.
    assert sum(sum(v != -100 for v in ex["labels"]) for ex in aplus) == sum(ep["n_target_tokens"] for ep in b)
    assert {ex["phrasing"] for ex in aplus} <= {"verbose", "structured"}
    assert report["matching"]["retained_question_ids"] == ["q2-fixture"]
    assert report["size_gate"]["passed"] is False and report["continuations_may_train"] is False  # tiny fixture

    # A second question whose eligible attempts are all dropped at rebuild must leave both arms entirely.
    task_b = {**task, "id": "q2-fixture-b", "question_id": "q2-fixture-b"}
    tasks_file.write_text(json.dumps(task) + "\n" + json.dumps(task_b) + "\n", encoding="utf-8")
    out_b = tmp_path / "attempt-b"
    evaluate_with_engine([task_b], Engine(attempts[1]), out_b, purpose="sampling")
    record_b = json.loads((out_b / "trajectories.jsonl").read_text(encoding="utf-8"))
    record_b["trajectory_id"] = record_b["metadata"]["task_id"] = "q2-fixture-b#s0"
    (step_dir / "trajectories.jsonl").write_text("\n".join(lines + [json.dumps(record_b)]) + "\n", encoding="utf-8")
    real = builder.rebuild_attempt
    builder.rebuild_attempt = lambda t, r, i, tok: ((None, "observation_differs_locally") if t["id"] == "q2-fixture-b"
                                                     else real(t, r, i, tok))
    b2, aplus2, report2, _ = builder.build(parts=[("run", "step", tasks_file)], root=root, tokenizer=tokenizer)
    assert report2["matching"]["retained_question_ids"] == ["q2-fixture"]
    assert {ex["question_id"] for ex in b2} == {ex["question_id"] for ex in aplus2} == {"q2-fixture"}
    assert report2["dropped_after_eligibility"]["observation_differs_locally"] == 1
