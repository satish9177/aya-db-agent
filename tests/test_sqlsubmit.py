"""Adversarial checks for the new scorer, terminal tool and audited data path."""
import json
import sqlite3
from pathlib import Path

import pytest

from ayadb.sqlenv import SQLTools, SubmissionEngine
from ayadb.sqlscore import execute, rows_equal, score_episode, score_query, top_level_ordered


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "scorer.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE items(id INTEGER PRIMARY KEY, value REAL, label TEXT);"
                      "INSERT INTO items VALUES(1,10,'a'),(2,10,'a'),(3,20,'b');"
                      "CREATE TABLE spare(id INTEGER);")
    con.executemany("INSERT INTO spare VALUES(?)", [(i,) for i in range(25)])
    con.commit()
    con.close()
    return path


def test_full_results_and_duplicate_multiplicity(db):
    result = score_query(db, "SELECT DISTINCT label FROM items", "SELECT label FROM items")
    assert not result["strict"] and result["bird_set"]
    assert not score_query(db, "SELECT id FROM spare WHERE id<20", "SELECT id FROM spare")["strict"]
    assert score_query(db, "SELECT label FROM items ORDER BY id DESC", "SELECT label FROM items")["strict"]
    assert not score_query(db, "SELECT label FROM items ORDER BY id DESC", "SELECT label FROM items ORDER BY id")["strict"]


@pytest.mark.parametrize("query", ["DELETE FROM items", "UPDATE items SET value=0", "PRAGMA table_info(items)",
                                    "ATTACH DATABASE ':memory:' AS other", "SELECT load_extension('x')",
                                    "SELECT 1; SELECT 2", "SELECT random()", "SELECT CURRENT_TIMESTAMP"])
def test_write_and_nondeterministic_queries_fail_closed(db, query):
    assert execute(db, query).error
    assert len(execute(db, "SELECT * FROM items").rows) == 3


def test_overflow_and_operation_limits_fail_closed(db):
    assert execute(db, "SELECT id FROM spare", row_limit=20).error == "row_limit"
    assert execute(db, "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n", max_ops=3000).error
    assert execute(db, "SELECT hex(zeroblob(1000))", max_bytes=1000).error == "result_byte_limit"


def test_no_unrelated_query_or_unsubmitted_answer_can_pass(db):
    tools = SQLTools(db, "t")
    funcs = {fn.__name__: fn for fn in tools}
    funcs["run_sql"](query="SELECT 1")
    result = score_episode(db, "SELECT COUNT(*) FROM items", tools.log)
    assert not result["success"] and not result["fallback"]
    funcs["submit_sql"](query="SELECT 1")
    assert not score_episode(db, "SELECT COUNT(*) FROM items", tools.log)["success"]


def test_terminal_submission_logs_every_attempt_and_stops_model(db):
    tools = SQLTools(db, "t")
    funcs = {fn.__name__: fn for fn in tools}
    class FailEngine:
        def generate(self, *args):
            raise AssertionError("model called after submission")
    funcs["submit_sql"](query="SELECT COUNT(*) FROM items")
    assert SubmissionEngine(FailEngine(), tools).generate([], [], {})["completions"] == [""]
    assert funcs["submit_sql"](query="SELECT 999").startswith("Error")
    assert funcs["run_sql"](query="SELECT 999").startswith("Error")
    assert len(tools.log) == 3 and [c["accepted"] for c in tools.log] == [True, False, False]
    assert score_episode(db, "SELECT COUNT(*) FROM items", tools.log)["success"]


def test_numeric_bag_matching_handles_ambiguous_matches_and_nulls():
    assert rows_equal([(1, 2), (1.0000001, 1)], [(1, 1), (1.0000001, 2)])
    # Greedy exact cancellation would consume 0, leaving an impossible residual pair.
    assert rows_equal([(0,), (1.5e-6,)], [(0,), (-.5e-6,)]) is False
    assert rows_equal([(0,), (.8e-6,)], [(0,), (-.8e-6,)])
    assert not rows_equal([(None,)], [("NULL",)])
    assert not rows_equal([(float("inf"),)], [(float("inf"),)])
    assert not rows_equal([(1,), (1,)], [(1,), (2,)])


@pytest.mark.parametrize("sql,ordered", [
    ("SELECT 'ORDER BY' FROM items", False),
    ("SELECT * FROM (SELECT * FROM items ORDER BY id)", False),
    ("WITH q AS (SELECT * FROM items ORDER BY id) SELECT * FROM q", False),
    ("SELECT * FROM items ORDER /* x */ BY id", True),
    ("SELECT * FROM items -- ORDER BY id\n", False),
])
def test_top_level_order_rule(sql, ordered):
    assert top_level_ordered(sql) is ordered


@pytest.fixture(scope="module")
def tokenizer():
    from ayadb.engine import official_tokenizer
    path = Path("data/sqale/tokenizer")
    if not (path / "tokenizer.json").exists():
        pytest.skip("official local tokenizer required")
    return official_tokenizer(str(path))


def task_for(db):
    return {"id": "sample", "question_id": "sample", "schema_id": "schema-test", "prompt": "How many items are there?",
            "db_path": str(db), "reference_sql": "SELECT COUNT(*) FROM items", "gold_tables": ["items"]}


def test_gold_and_real_error_replay_preserve_assistant_only_masks(db, tokenizer):
    from ayadb.sftdata import replay_audit
    from ayadb.sqlenv import FALLBACK, SYSTEM_PROMPT
    from ayadb.sqlsftdata import encode_episode, make_episode
    ep = make_episode(task_for(db), repair=True)
    assert replay_audit(ep, None, ep["db_sha256"], tokenizer, SYSTEM_PROMPT, FALLBACK, sql_db=db) == []
    assert ep["_replay_record"]["metadata"]["call_log"][-1]["accepted"]
    example = encode_episode(ep, tokenizer)
    assert example is not None
    spans, part = [], []
    for token, label in zip(example["input_ids"], example["labels"]):
        if label == -100:
            if part:
                spans.append(part)
                part = []
        else:
            assert token == label
            part.append(label)
    if part:
        spans.append(part)
    decoded = [tokenizer.decode(span, skip_special_tokens=False) for span in spans]
    expected = [m["content"] + "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
                for m in ep["messages"] if m["role"] == "assistant" and m["train"]]
    assert decoded == expected
    assert not any("__ayadb_bad_alias__" in text for text in decoded)


def test_auditkit_uses_full_sql_metric_on_saved_records(db, tokenizer, tmp_path):
    from ayadb.sftdata import replay_audit
    from ayadb.sqlenv import FALLBACK, SYSTEM_PROMPT
    from ayadb.sqlsftdata import auditkit_check, make_episode
    task = task_for(db)
    ep = make_episode(task)
    assert not replay_audit(ep, None, ep["db_sha256"], tokenizer, SYSTEM_PROMPT, FALLBACK, sql_db=db)
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(ep["_replay_record"]) + "\n", encoding="utf-8")
    result = auditkit_check(path, {ep["id"]: task})
    assert result["n_rows"] == 1
    assert list(result["verdicts"].values()) == ["success"]
    # A valid preceding query cannot make an unrelated submitted query pass.
    record = json.loads(path.read_text(encoding="utf-8"))
    record["metadata"]["call_log"][-1]["args"]["query"] = "SELECT 999"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    negative = auditkit_check(path, {ep["id"]: task})
    assert list(negative["verdicts"].values()) != ["success"]


def test_agenttune_custom_factory_preserves_model_and_exact_masks(db, tokenizer, tmp_path):
    from ayadb.agenttune_sft import AuditedSFTFactory, prepare_trainer
    from ayadb.sqlsftdata import encode_episode, make_episode
    ep = make_episode(task_for(db), repair=True)
    captured = {}
    sentinel_model, sentinel_args = object(), object()
    class SpyTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def train(self):
            return {"dry_run": True, "optimizer_steps": 0}
    factory = AuditedSFTFactory(sentinel_model, tokenizer, sentinel_args, trainer_class=SpyTrainer)
    trainer = prepare_trainer("existing-Tiny-Aya", [ep], factory, str(tmp_path))
    assert trainer.sft_dataset()[0]["messages"] == ep["messages"]
    # Only the spy's train method runs; no model, optimizer or GPU is ever constructed.
    assert trainer.train() == {"dry_run": True, "optimizer_steps": 0}
    assert captured["model"] is sentinel_model and captured["args"] is sentinel_args
    assert captured["train_dataset"] == [encode_episode(ep, tokenizer)]


def test_custom_factory_keeps_exact_budget_masks_and_rejects_unmasking(db, tokenizer):
    import copy
    from ayadb.agenttune_sft import AuditedSFTFactory
    from ayadb.sqlsftdata import encode_episode, make_episode
    ep = make_episode(task_for(db), repair=True)
    original = encode_episode(ep, tokenizer)
    matched = {**ep, **copy.deepcopy(original)}
    indexes = [i for i, label in enumerate(matched["labels"]) if label != -100]
    matched["labels"][indexes[-1]] = -100
    captured = {}
    class SpyTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    factory = AuditedSFTFactory(object(), tokenizer, object(), trainer_class=SpyTrainer)
    factory(model="TinyAya", train_dataset=[matched])
    assert captured["train_dataset"][0]["labels"] == matched["labels"]
    forbidden = next(i for i, label in enumerate(original["labels"]) if label == -100)
    matched["labels"][forbidden] = matched["input_ids"][forbidden]
    with pytest.raises(ValueError, match="unaudited target"):
        factory(model="TinyAya", train_dataset=[matched])
