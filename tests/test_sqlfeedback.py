"""Amendment 9: column-hints-v1 tool feedback, beside the unchanged SQLTools."""
import json
import sqlite3

import pytest

import ayadb.sqlenv as sqlenv
from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.sftdata import ScriptedEngine
from ayadb.sqlenv import SQLTools
from ayadb.sqlfeedback import (MAX_HINT_CHARS, POLICY, HintedSQLTools, column_hint, named_tables,
                               normalize_query)
from ayadb.sqlguard import score_episode
from ayadb.sqlrunner import evaluate_with_engine
from ayadb.sqlsftdata import call_text


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE users(id INTEGER, email TEXT);"
        "CREATE TABLE orders(order_id INTEGER, user_id INTEGER, total REAL);"
        "CREATE TABLE audit_log(id INTEGER, action TEXT);"
        "INSERT INTO users VALUES (1, 'a@x'), (2, 'b@x');"
        "INSERT INTO orders VALUES (10, 1, 5.0), (11, 2, 7.5);")
    con.close()
    return path


def test_tool_prompt_schemas_are_identical_to_the_original_tools(db):
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema
    original = [_build_tool_schema(t) for t in SQLTools(db, "t")]
    hinted = [_build_tool_schema(t) for t in HintedSQLTools(db, "t")]
    assert original == hinted and [s["function"]["name"] for s in hinted] == [
        "list_tables", "describe_table", "run_sql", "submit_sql"]


def test_successful_queries_and_other_tools_are_unchanged(db):
    old, new = SQLTools(db, "t"), HintedSQLTools(db, "t")
    for i, (name, args) in enumerate([("list_tables", {}), ("describe_table", {"table": "orders"}),
                                      ("describe_table", {"table": "missing"}),
                                      ("run_sql", {"query": "SELECT email FROM users ORDER BY id"})]):
        assert old[[t.__name__ for t in old].index(name)](**args) == new[[t.__name__ for t in new].index(name)](**args)
    assert old.log == new.log


def test_missing_column_lists_real_columns_and_owner_table(db):
    tools = HintedSQLTools(db, "t")
    run_sql = tools[2]
    query = "SELECT u.mail FROM users u JOIN orders o ON o.user_id = u.id"
    result = run_sql(query=query)
    assert result.startswith("SQL error: no such column: u.mail. Columns of the tables in this query: ")
    assert "users(id, email)" in result and "orders(order_id, user_id, total)" in result
    assert "audit_log" not in result
    entry = tools.log[-1]
    assert entry["error"] and entry["feedback"] == "column_hint" and entry["result"] == result
    # The original tools return only SQLite's message.
    assert SQLTools(db, "t")[2](query=query) == "SQL error: no such column: u.mail"
    # A real column placed on the wrong table names its owner.
    owner = run_sql(query="SELECT action FROM users")
    assert owner.endswith("Columns of the tables in this query: users(id, email). Column action is in: audit_log")


def test_missing_table_lists_available_tables(db):
    result = HintedSQLTools(db, "t", ["users", "orders", "audit_log"])[2](query="SELECT * FROM customers")
    assert result == "SQL error: no such table: customers. Available tables: users, orders, audit_log"


def test_identical_failed_repeat_is_refused_without_running(db, monkeypatch):
    calls = []
    real = sqlenv.execute
    monkeypatch.setattr(sqlenv, "execute", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    tools = HintedSQLTools(db, "t")
    run_sql = tools[2]
    run_sql(query="SELECT nope FROM users")
    refused = run_sql(query="  SELECT nope   FROM users ;")
    assert len(calls) == 1
    assert refused == ("Error: this exact query already failed with: no such column: nope. It was not run again. "
                       "Write a different query; check the column names first.")
    entry = tools.log[-1]
    assert entry == {"task_id": "t", "call_index": 1, "name": "run_sql", "args": {"query": "  SELECT nope   FROM users ;"},
                     "accepted": False, "terminal": False, "result": refused, "error": True,
                     "terminal_after": False, "feedback": "refused_repeat"}
    # A changed query runs; a successful repeat is not refused.
    assert run_sql(query="SELECT email FROM users WHERE id = 1").startswith('{"columns"')
    assert run_sql(query="SELECT email FROM users WHERE id = 1").startswith('{"columns"')
    assert len(calls) == 3
    # Bad arguments and post-submission calls keep the original behaviour.
    assert run_sql(sql="SELECT 1") == "Error: invalid tool arguments."
    tools[3](query="SELECT email FROM users")
    assert run_sql(query="SELECT nope FROM users") == "Error: episode already ended by submit_sql."


def test_hints_are_bounded_and_matching_is_exact():
    names = ["t" + str(i) for i in range(400)]
    hint = column_hint("no such table: x", "SELECT 1", {}, names)
    assert len(hint) == MAX_HINT_CHARS + len(" [hint truncated]") and hint.endswith(" [hint truncated]")
    assert named_tables('SELECT * FROM "Course" c JOIN CourseId x', ["Course", "CourseId", "Cour"]) == ["Course", "CourseId"]
    assert normalize_query(" SELECT  1 ;; ") == "SELECT 1"
    assert column_hint("syntax error", "SELEC", {}, ["a"]) is None


def test_refused_repeats_never_feed_the_fallback_score(db):
    tools = HintedSQLTools(db, "t")
    tools[2](query="SELECT email FROM users ORDER BY id")
    tools[2](query="SELECT nope FROM users")
    tools[2](query="SELECT nope FROM users")
    scored = score_episode(db, "SELECT email FROM users ORDER BY id", tools.log)
    assert scored["fallback"] and not scored["submitted"]


def test_runner_uses_hinted_tools_only_when_registered(tmp_path, db):
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    script = [call_text("run_sql", {"query": "SELECT mail FROM users"}),
              call_text("run_sql", {"query": "SELECT mail FROM users"}),
              call_text("run_sql", {"query": "SELECT email FROM users"}),
              call_text("submit_sql", {"query": "SELECT email FROM users"})]

    class Engine(ScriptedEngine):
        def __init__(self):
            super().__init__(tokenizer, list(script))
            self.calls = []

        def generate(self, *a, **kw):
            self.calls.append({"prompt_tokens": 10, "completion_tokens": 5})
            return super().generate(*a, **kw)

    task = {"id": "fixture", "prompt": "List every user email.", "db_path": str(db),
            "db_sha256": fingerprint(db), "reference_sql": "SELECT email FROM users"}
    summary = evaluate_with_engine([task], Engine(), tmp_path / "hinted", tool_feedback=POLICY)
    assert summary["strict_successes"] == summary["auditkit_agreement"] == 1
    assert (summary["tool_feedback"], summary["column_hints"], summary["refused_repeats"]) == (POLICY, 1, 1)
    row = json.loads((tmp_path / "hinted/metrics.jsonl").read_text())
    assert (row["column_hints"], row["refused_repeats"], row["sql_errors"]) == (1, 1, 2)
    record = json.loads((tmp_path / "hinted/trajectories.jsonl").read_text())
    assert record["metadata"]["tool_feedback"] == POLICY
    assert "Columns of the tables in this query: users(id, email)" in record["metadata"]["call_log"][0]["result"]

    original = evaluate_with_engine([task], Engine(), tmp_path / "original")
    row = json.loads((tmp_path / "original/metrics.jsonl").read_text())
    assert "tool_feedback" not in original and "column_hints" not in row and row["sql_errors"] == 2
    with pytest.raises(ValueError, match="unknown tool feedback"):
        evaluate_with_engine([task], Engine(), tmp_path / "bad", tool_feedback="other")


def test_registration_accepts_only_known_feedback_on_evaluation_steps(tmp_path):
    from ayadb.runconfig import load_registration
    reg = json.loads(open("experiments/sql-a1500/REGISTRATION.json", encoding="utf-8").read())
    reg["steps"][1]["tool_feedback"] = POLICY
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(reg), encoding="utf-8")
    load_registration(path, check_files=False)
    for step, value in ((1, "other"), (0, POLICY)):
        bad = json.loads(json.dumps(reg))
        bad["steps"][1].pop("tool_feedback")
        bad["steps"][step]["tool_feedback"] = value
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match="tool feedback"):
            load_registration(path, check_files=False)
