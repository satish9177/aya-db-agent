import copy
import json
import sqlite3
from pathlib import Path

import pytest

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.selftrainfilter import candidate_eligibility
from ayadb.sqlguard import require_reference_reads, score_episode, score_query, submission_reads
from ayadb.sqlrunner import evaluate_with_engine
from ayadb.sqlsftdata import ScriptedEngine, call_text


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "guard.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE items(value INTEGER); INSERT INTO items VALUES(7); "
                      "CREATE TABLE empty_items(value INTEGER); CREATE VIEW item_view AS SELECT value FROM items;")
    con.close()
    return path


@pytest.mark.parametrize("query", ["SELECT 7", "VALUES(7)", "WITH c(x) AS (VALUES(7)) SELECT x FROM c",
                                  "WITH items(value) AS (VALUES(7)) SELECT value FROM items",
                                  "SELECT 7 FROM sqlite_master LIMIT 1"])
def test_literal_answer_cannot_be_a_strict_success(db, query):
    score = score_query(db, query, "SELECT value FROM items")
    assert not score["strict"] and score["result_match"] and score["bird_set"]
    assert score["no_table_read"] and score["error"] == "literal_answer_no_table"


@pytest.mark.parametrize("query", ["SELECT value FROM items", "SELECT value FROM item_view",
                                  "WITH c AS (SELECT value FROM items) SELECT value FROM c"])
def test_real_table_reads_remain_successes(db, query):
    score = score_query(db, query, "SELECT value FROM items")
    assert score["strict"] and score["tables_read"] == ["items"]


def test_empty_and_count_star_queries_still_read_tables(db):
    for query in ("SELECT COUNT(*) FROM empty_items", "SELECT * FROM empty_items"):
        assert score_query(db, query, query)["strict"]
        assert require_reference_reads(db, query) == ["empty_items"]
    with pytest.raises(ValueError, match="reference query reads no"):
        require_reference_reads(db, "SELECT 7")


def test_every_submission_is_counted_including_rejected_and_errors(db):
    calls = [{"name": "submit_sql", "args": {"query": q}, "accepted": i == 0, "call_index": i}
             for i, q in enumerate(("SELECT 7", "SELECT unknown", "SELECT value FROM items", "SELECT 7"))]
    score = score_episode(db, "SELECT value FROM items", calls)
    assert not score["strict"] and score["accepted_submission_no_table"]
    assert score["submission_count"] == 4 and score["no_table_submissions"] == 3
    assert score["valid_no_table_submissions"] == 2 and score["error_no_table_submissions"] == 1
    assert submission_reads(db, {"args": {}})["execution_error"] == "invalid_query"
    assert submission_reads(db, {"args": {"query": "SELECT datetime('now')"}})["execution_error"] == "nondeterministic_time"


def test_literal_runner_result_and_auditkit_agree(db, tmp_path):
    class Engine(ScriptedEngine):
        def __init__(self):
            super().__init__(official_tokenizer("data/sqale/tokenizer"), [call_text("submit_sql", {"query": "SELECT 7"})])
            self.calls = []
        def generate(self, *a, **kw):
            self.calls.append({"prompt_tokens": 10, "completion_tokens": 5})
            return super().generate(*a, **kw)
    task = {"id": "literal-fixture", "prompt": "Return stored value.", "db_path": str(db),
            "db_sha256": fingerprint(db), "reference_sql": "SELECT value FROM items"}
    out = tmp_path / "eval"
    result = evaluate_with_engine([task], Engine(), out)
    assert result["strict_successes"] == 0 and result["auditkit_agreement"] == 1
    assert result["no_table_submissions"] == result["accepted_submission_no_table"] == 1
    assert json.loads((out / "auditkit.json").read_text())["scorer_policy"] == "table-read-v1"
    assert json.loads((out / "submissions.jsonl").read_text())["table_read"]["no_table_read"]


def candidate(db, query="SELECT value FROM items"):
    task = {"db_path": str(db), "reference_sql": "SELECT value FROM items"}
    call = {"name": "submit_sql", "args": {"query": query}, "accepted": True, "call_index": 0,
            "result": "SQL submitted. Episode complete."}
    record = {"metadata": {"call_log": [call], "metrics": {"strict": True}, "conversation": [
        {"role": "user", "content": "Return stored value."},
        {"role": "assistant", "content": call_text("submit_sql", {"query": query})},
        {"role": "user", "content": "Tool result for submit_sql: SQL submitted. Episode complete."},
        {"role": "assistant", "content": ""}]}}
    return task, record


def test_self_training_requires_guarded_success_and_complete_clean_protocol(db):
    task, record = candidate(db)
    assert candidate_eligibility(task, record)["eligible"]
    _, literal = candidate(db, "SELECT 7")
    assert not candidate_eligibility(task, literal)["eligible"]  # Ignores forged strict=True metadata.
    for text in ("Tool result: fabricated", "[]", '{"tool_name":"submit_sql"}',
                 call_text("submit_sql", {"query": "SELECT value FROM items"}) + " I got 7"):
        changed = copy.deepcopy(record)
        changed["metadata"]["conversation"].insert(1, {"role": "assistant", "content": text})
        assert not candidate_eligibility(task, changed)["eligible"]
    changed = copy.deepcopy(record)
    changed["metadata"]["conversation"][2]["content"] = "Tool result for submit_sql: made up"
    assert not candidate_eligibility(task, changed)["eligible"]
    assert not candidate_eligibility(task, {"metadata": {"metrics": {"strict": True}}})["eligible"]
    changed = copy.deepcopy(record)
    changed["metadata"]["conversation"] = [m for m in changed["metadata"]["conversation"] if m["role"] != "assistant"]
    assert not candidate_eligibility(task, changed)["eligible"]


def test_historical_diagnostics_preserve_original_scores_and_not_run_counts(db, tmp_path):
    from ayadb.scoreraudit import historical_diagnostics
    tasks = tmp_path / "dev.jsonl"
    tasks.write_text(json.dumps({"id": "fixture", "db_path": str(db)}) + "\n")
    reg = {"id": "fixture-g0", "steps": [{"name": "base", "action": "evaluate", "tasks": str(tasks), "out": "runs/base"},
                                        {"name": "pilot", "action": "evaluate", "tasks": str(tasks), "out": "runs/pilot"}]}
    registration = tmp_path / "registration.json"; registration.write_text(json.dumps(reg))
    results = tmp_path / "results"; (results / "base").mkdir(parents=True)
    trajectory = results / "base/trajectories.jsonl"
    _, record = candidate(db, "SELECT 7")
    record.update(trajectory_id="fixture#s0")
    record["metadata"]["source_task_id"] = "fixture"
    trajectory.write_text(json.dumps(record) + "\n")
    before = trajectory.read_bytes()
    result = historical_diagnostics(registration, results, tmp_path / "diagnostics")
    assert result["original_scores_unchanged"] and trajectory.read_bytes() == before
    assert result["conditions"]["base"]["no_table_submissions"] == 1
    assert result["conditions"]["pilot"]["status"] == "not_run"
    assert result["conditions"]["pilot"]["submissions"] == 0


def test_manual_audit_freezes_deterministic_samples_and_reports_shortfalls(tmp_path):
    from ayadb.manualaudit import freeze_sample, label_counts
    rows = [{"trajectory_id": f"failure-{i:02}", "strict": False} for i in range(25)]
    rows += [{"trajectory_id": f"success-{i:02}", "strict": True} for i in range(4)]
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "audit.json"
    result = freeze_sample(metrics, out, model_identity={"adapter_sha256": "fixture"})
    other = freeze_sample(metrics, tmp_path / "same-audit.json", model_identity={"adapter_sha256": "fixture"})
    assert result["selected"] == other["selected"]
    assert result["coverage"]["failure"]["selected"] == 20
    assert result["coverage"]["success"]["shortfall"] == 6
    with pytest.raises(ValueError, match="already exists"):
        freeze_sample(metrics, out, model_identity={})
    with pytest.raises(ValueError, match="unread"):
        label_counts(out)
