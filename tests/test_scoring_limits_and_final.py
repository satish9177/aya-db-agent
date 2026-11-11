"""Amendment 11: scoring-side time limit and the frozen final-selection gate (no model, no CUDA)."""
import json
import sqlite3
from pathlib import Path

import pytest

import ayadb.sqlenv as sqlenv
import ayadb.sqlguard as sqlguard
from ayadb.promptquality import fingerprint
from ayadb.scoringlimits import SCORING_MAX_OPS, SCORING_SECONDS


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "f.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE items(value INTEGER); INSERT INTO items VALUES (7), (9);")
    con.close()
    return path


def test_scoring_uses_its_own_limit_and_tools_keep_five_seconds(db, monkeypatch):
    seen = []
    real = sqlguard.execute
    monkeypatch.setattr(sqlguard, "execute", lambda *a, **k: seen.append((k.get("seconds"), k.get("max_ops"))) or real(*a, **k))
    sqlguard._reference.cache_clear()
    log = [{"name": "run_sql", "args": {"query": "SELECT value FROM items"}, "error": False},
           {"name": "submit_sql", "args": {"query": "SELECT value FROM items"}, "accepted": True, "call_index": 1}]
    result = sqlguard.score_episode(db, "SELECT value FROM items ORDER BY value", log)
    assert result["strict"] and result["fallback"]
    assert seen and set(seen) == {(SCORING_SECONDS, SCORING_MAX_OPS)}
    assert (SCORING_SECONDS, SCORING_MAX_OPS) == (60.0, 2_000_000_000)  # amendment 12
    # The reference ran once and was then cached for the strict, fallback and read checks.
    calls_before = len(seen)
    sqlguard.score_episode(db, "SELECT value FROM items ORDER BY value", log)
    assert len(seen) - calls_before == 3  # strict, fallback and submission-read executions only
    # Agent tools still use sqlscore's default 5 s limit.
    tool_seconds = []
    real_tool = sqlenv.execute
    monkeypatch.setattr(sqlenv, "execute", lambda *a, **k: tool_seconds.append(k.get("seconds", 5.0)) or real_tool(*a, **k))
    sqlenv.SQLTools(db, "t")[2](query="SELECT 1 FROM items")
    assert tool_seconds == [5.0]


def test_final_steps_refuse_unfrozen_or_unselected_models_before_loading(tmp_path, db, monkeypatch):
    import ayadb.sqlrunner as runner
    task = {"id": "final-1", "question_id": "final-1", "split": "final", "format": "sql-submit-v1",
            "prompt": "List the values.", "db_path": str(db), "db_sha256": fingerprint(db),
            "reference_sql": "SELECT value FROM items"}
    tasks = tmp_path / "final.tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n", encoding="utf-8")
    config = {"model": "CohereLabs/tiny-aya-global", "model_revision": "rev", "tokenizer": "data/sqale/tokenizer"}
    selection = tmp_path / "SELECTION.json"

    def step(sha):
        return {"name": "final-base", "tasks": str(tasks), "purpose": "final", "selection_file": str(selection),
                "selection_sha256": sha, "out": str(tmp_path / "out"), "adapter": None}

    loaded = []
    monkeypatch.setattr(runner, "load_model", lambda *a, **k: loaded.append(1) or (_ for _ in ()).throw(RuntimeError("stop: model load reached")))
    selection.write_text(json.dumps({"frozen": False, "models": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen model selection"):
        runner._run_evaluation(config, step(fingerprint(selection)), None)
    selection.write_text(json.dumps({"frozen": True, "models": [{"model": config["model"], "model_revision": "other",
                                                                  "adapter_sha256": None}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the frozen final selection"):
        runner._run_evaluation(config, step(fingerprint(selection)), None)
    with pytest.raises(ValueError, match="changed after registration"):
        runner._run_evaluation(config, step("0" * 64), None)
    selection.write_text(json.dumps({"frozen": True, "models": [{"model": config["model"], "model_revision": "rev",
                                                                  "adapter_sha256": None}]}), encoding="utf-8")

    class Memory:
        def reset(self): pass
    with pytest.raises(RuntimeError, match="model load reached"):
        runner._run_evaluation(config, step(fingerprint(selection)), Memory())
    assert loaded == [1]
