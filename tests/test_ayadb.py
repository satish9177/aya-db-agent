"""Environment, verifier and metric tests, including adversarial trajectories a reward could be gamed by."""

import json
import sqlite3

import pytest

from ayadb.env import make_tools
from ayadb.metrics import logged, trajectory_metrics
from ayadb.tasks import (FROZEN_DEV_SHA256, SPLITS, check_answer, extract_answer, file_sha256, generate_split,
                         is_frozen_dev, load_tasks)


@pytest.fixture(scope="module")
def dev(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    db, tasks = generate_split("dev", d)
    return d, db, tasks


def test_generation_is_deterministic_and_splits_differ(dev, tmp_path):
    d, db, tasks = dev
    db2, tasks2 = generate_split("dev", tmp_path)
    assert file_sha256(tasks) == file_sha256(tasks2)
    other_db, _ = generate_split("train", tmp_path)
    rows = lambda p: sqlite3.connect(p).execute("SELECT * FROM orders").fetchall()  # noqa: E731
    assert rows(db) != rows(other_db)


def test_frozen_dev_check_ignores_line_endings_only(dev, tmp_path):
    d, db, tasks = dev
    raw = tasks.read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / "crlf.jsonl").write_bytes(raw.replace(b"\n", b"\r\n"))
    (tmp_path / "lf.jsonl").write_bytes(raw)
    (tmp_path / "edited.jsonl").write_bytes(raw.replace(b"dev-", b"dex-", 1))
    assert file_sha256(tmp_path / "crlf.jsonl") == FROZEN_DEV_SHA256  # the preregistered (Windows) bytes
    assert is_frozen_dev(tmp_path / "crlf.jsonl") and is_frozen_dev(tmp_path / "lf.jsonl")
    assert not is_frozen_dev(tmp_path / "edited.jsonl")


def test_gold_answers_match_reference_sql(dev):
    d, db, tasks = dev
    con = sqlite3.connect(db)
    for t in load_tasks(tasks):
        v = con.execute(t.reference_sql).fetchone()[0]
        assert check_answer(str(v), t.answer, t.kind), (t.id, v, t.answer)


def test_held_out_families_never_in_train_or_dev():
    held = set(SPLITS["test"]["families"]) - set(SPLITS["train"]["families"])
    assert held and not held & set(SPLITS["dev"]["families"])


def test_tools_are_read_only_and_errors_are_recoverable(dev):
    _, db, _ = dev
    list_tables, describe_table, run_sql = make_tools(db)
    for q in ["DELETE FROM orders", "DROP TABLE orders", "INSERT INTO products VALUES ('x','y','z',1)",
              "ATTACH 'evil.db' AS e", "PRAGMA writable_schema = 1"]:
        assert run_sql(q).startswith("SQL error"), q
    assert "Available tables" in describe_table("customer")
    assert run_sql("SELECT customer_id FROM orders").startswith("SQL error: no such column")
    assert "interrupted" in run_sql("SELECT COUNT(*) FROM order_lines a, order_lines b, order_lines c")
    assert sqlite3.connect(db).execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 640


@pytest.mark.parametrize("text,gold,kind,ok", [
    ("ANSWER: 42", "42", "int", True),
    ("Let me think.\nANSWER: 1,234.50", "1234.50", "money", True),
    ("ANSWER: $204.03", "204.03", "money", True),
    ("ANSWER: 9.0", "9.0", "float1", True),
    ("ANSWER: Priya Varma.", "Priya Varma", "text", True),
    ("ANSWER: 12 or 13", "12", "int", False),          # hedging across candidates
    ("ANSWER: 41", "42", "int", False),
    ("The answer is 42.", "42", "int", False),          # no marker: strict check fails
    ("ANSWER: 204", "204.03", "money", False),
    ("ANSWER: Priya", "Priya Varma", "text", False),
    ("ANSWER: 5\nANSWER: 42", "42", "int", True),       # last marker wins
])
def test_verifier(text, gold, kind, ok):
    assert check_answer(extract_answer(text), gold, kind) is ok


def _record(steps, final):
    conv = [{"role": "assistant", "content": ""} for _ in range(len(steps) + 1)]
    return {"steps": [{"action": {"tool_calls": s}} for s in steps] + [{"action": {}}],
            "final_response": final, "metadata": {"conversation": conv}}


def _call(tool, **args):
    return {"id": None, "name": tool, "arguments": args}


@pytest.fixture
def tools(dev):
    log = []
    return log, logged(make_tools(dev[1]), log)


def test_good_trajectory(tools):
    log, (lt, dt_, rs) = tools
    lt(); dt_(table="customers"); rs(query="SELECT COUNT(*) FROM customers")
    rec = _record([[_call("list_tables")], [_call("describe_table", table="customers")],
                   [_call("run_sql", query="SELECT COUNT(*) FROM customers")]], "ANSWER: 60")
    m = trajectory_metrics(rec, log, "60", "int", max_steps=8)
    assert m["success"] and m["valid_calls"] == 3 and m["exec_errors"] == 0 and not m["loop"]


def test_guess_without_query_is_not_success(tools):
    log, _ = tools
    m = trajectory_metrics(_record([], "ANSWER: 60"), log, "60", "int", max_steps=8)
    assert m["correct"] and not m["success"] and m["no_tool_answer"]


def test_invalid_json_call_is_premature_termination(tools):
    log, _ = tools
    rec = _record([], '[{"tool_name": "run_sql", "parameters": {"query": "SELECT 1"]')
    m = trajectory_metrics(rec, log, "60", "int", max_steps=8)
    assert m["malformed_final"] and not m["success"] and m["attempted_calls"] == 1
    assert m["valid_calls"] == 0


def test_wrong_argument_names_are_counted(tools):
    log, (_, dt_, rs) = tools
    dt_(name="customers")                # wrong kwarg -> logged as bad args
    rs(query="SELECT nope FROM customers")  # valid call, failing SQL -> exec error
    rec = _record([[_call("describe_table", name="customers")],
                   [_call("run_sql", query="SELECT nope FROM customers")]], "ANSWER: 1")
    m = trajectory_metrics(rec, log, "60", "int", max_steps=8)
    assert m["bad_arg_calls"] == 1 and m["exec_errors"] == 1 and m["valid_calls"] == 1
    assert not m["grounded"] and not m["success"]


def test_unparseable_arguments_never_executed_count_as_bad(tools):
    log, _ = tools  # AgentTune could not json-decode the args, so the tool never ran
    rec = _record([[{"id": None, "name": "run_sql", "arguments": "{query: SELECT"}]], "ANSWER: 1")
    m = trajectory_metrics(rec, log, "60", "int", max_steps=8)
    assert m["bad_arg_calls"] == 1 and m["valid_calls"] == 0


def test_repetitive_calls_flag_a_loop(tools):
    log, (lt, _, _) = tools
    for _ in range(3):
        lt()
    rec = _record([[_call("list_tables")]] * 3, "ANSWER: 60")
    m = trajectory_metrics(rec, log, "60", "int", max_steps=8)
    assert m["repeated_calls"] == 2 and m["loop"] and not m["success"]


def test_budget_exhaustion_is_a_loop_not_malformed_json(tools):
    log, (lt, _, _) = tools
    lt(); lt()
    rec = _record([[_call("list_tables")]] * 2, '[{"tool_name": "list_tables", "parameters": {}}]')
    m = trajectory_metrics(rec, log, "60", "int", max_steps=2)
    assert m["hit_max_steps"] and m["loop"] and not m["malformed_final"]


def test_unknown_tool_and_unnecessary_calls(tools):
    log, (lt, _, rs) = tools
    lt(); lt(); rs(query="SELECT COUNT(*) FROM customers")
    rec = _record([[_call("drop_everything")], [_call("list_tables")], [_call("list_tables")],
                   [_call("run_sql", query="SELECT COUNT(*) FROM customers")]], "ANSWER: 60")
    m = trajectory_metrics(rec, log, "60", "int", max_steps=8)
    assert m["unknown_tool_calls"] == 1 and m["repeated_calls"] == 1 and m["success"]
    assert json.dumps(m)  # serialisable


def test_mcnemar_exact_matches_binomial_tail():
    from ayadb.compare import mcnemar_exact
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(1, 1) == 1.0
    assert mcnemar_exact(5, 0) == mcnemar_exact(0, 5) == 2 / 32
    assert abs(mcnemar_exact(10, 2) - 2 * 79 / 4096) < 1e-12


# ---- recovery experiment ----------------------------------------------------------------------------

def test_query_mistakes_are_genuine_errors_and_change_only_context(dev):
    from ayadb.expert import _HOPS, QueryMistake, build_episode
    _, db, tasks = dev
    kinds = set()
    for t in load_tasks(tasks):
        clean = build_episode(t, db)
        for seed in range(6):
            m = QueryMistake(hop=seed % _HOPS.get(t.family, 1), twice=seed % 2 == 1, seed=seed)
            ep = build_episode(t, db, m)
            kinds |= set(m.kinds)
            bad = [c for c in ep.calls if not c["train"]]
            assert len(bad) == len(m.kinds) and all(c["result"].startswith("SQL error") for c in bad), (t.id, bad)
            assert all(c["name"] == "run_sql" for c in bad)
            trained = lambda e: [x["content"] for x in e.messages if x["role"] == "assistant" and x["train"]]  # noqa: E731
            assert trained(ep) == trained(clean) and ep.answer == clean.answer
    assert kinds == {"alias", "column", "ambiguous", "missing_join", "syntax"}


def test_drift_split_breaks_every_training_query_and_keeps_answers(tmp_path):
    from ayadb.expert import _WRONG_COLUMNS, QUERY_WRONG_COLUMNS
    from ayadb.tasks import DRIFT_RENAMES, PINNED_LF, generate_eval, lf_sha256
    path, dbs = generate_eval("drift", tmp_path)
    assert lf_sha256(path) == PINNED_LF["drift"]
    _, _, run_sql = make_tools(next(iter(dbs.values())))
    for t in load_tasks(tmp_path / "dev.jsonl"):  # training column names fail on the renamed schema
        assert run_sql(t.reference_sql).startswith("SQL error: no such column"), t.id
    new = set(DRIFT_RENAMES.values())
    assert not new & set(QUERY_WRONG_COLUMNS.values()) and not new & {b.split(".")[-1] for _, b in _WRONG_COLUMNS}


def test_mh_split_is_pinned_and_held_out_only(tmp_path):
    from ayadb.tasks import HELD_OUT, MH_SEEDS, PINNED_LF, TRAIN_POOL_SEEDS, generate_eval, lf_sha256
    path, dbs = generate_eval("mh", tmp_path)
    assert lf_sha256(path) == PINNED_LF["mh"]
    assert {t.family for t in load_tasks(path)} <= set(HELD_OUT) and len(set(dbs.values())) == len(MH_SEEDS)
    assert not set(MH_SEEDS) & ({c["seed"] for c in SPLITS.values()} | set(TRAIN_POOL_SEEDS))


def test_after_errors_classifies_the_next_call():
    from ayadb.compare import after_errors
    q = lambda s: {"name": "run_sql", "args": {"query": s}}  # noqa: E731
    log = [{**q("SELECT r.x FROM t"), "result": "SQL error: no such column: r.x", "error": True},
           {**q("SELECT r.x FROM t"), "result": "SQL error: no such column: r.x", "error": True},
           {"name": "describe_table", "args": {"table": "t"}, "result": "Table t: x", "error": False},
           {**q("SELECT y FROM t"), "result": "SQL error: no such column: y", "error": True},
           {**q("SELECT x FROM t"), "result": "x\n1", "error": False},
           {**q("SELECT x FROM t WHER x"), "result": 'SQL error: near "x": syntax error', "error": True}]
    ae = after_errors({"metadata": {"call_log": log}, "final_response": "ANSWER: 1"})
    assert ae["next"] == {"edited, ran": 1, "edited, failed": 0, "repeated": 1, "schema lookup": 1,
                          "answered": 1, "stopped": 0, "other": 0}
    assert ae["kinds"] == {"no such column": 3, "syntax": 1}
