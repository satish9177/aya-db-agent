"""Prospective table-read-v1 scorer; frozen historical sqlscore remains unchanged."""
from __future__ import annotations

import json
import tempfile
import warnings
from functools import lru_cache
from itertools import islice
from pathlib import Path

from .scoringlimits import SCORING_MAX_OPS, SCORING_SECONDS
from .sqlscore import execute, readonly, rows_equal
from .sqlpolicy import ordering_policy

POLICY = "table-read-v1"


@lru_cache(maxsize=2048)
def _tables(path, size, mtime):
    con = readonly(Path(path))
    try:
        return frozenset(r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'") if not r[0].lower().startswith("sqlite_"))
    finally:
        con.close()


@lru_cache(maxsize=4096)
def _reference(path, size, mtime, sql, explain):
    return execute(Path(path), sql, seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS, explain=explain)


def reference_result(db, sql, *, explain=False):
    """Reference (gold) execution with the scoring time limit, cached per frozen database file."""
    path = Path(db).resolve()
    stat = path.stat()
    return _reference(str(path), stat.st_size, stat.st_mtime_ns, sql, explain)


def user_table_reads(db: Path, result):
    path = Path(db).resolve()
    stat = path.stat()
    return sorted(result.reads & _tables(str(path), stat.st_size, stat.st_mtime_ns))


def require_reference_reads(db, reference_sql, result=None):
    result = result if result is not None else reference_result(db, reference_sql, explain=True)
    if result.error:
        raise ValueError("reference table-read check failed: " + result.error)
    reads = user_table_reads(Path(db), result)
    if not reads:
        raise ValueError("reference query reads no user database table")
    return reads


def submission_reads(db: Path, call: dict):
    """Inventory every attempted submission without changing the original tool log."""
    query = call.get("args", {}).get("query")
    result = execute(Path(db), query, seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
    reads = user_table_reads(Path(db), result)
    return {"call_index": call.get("call_index"), "accepted": bool(call.get("accepted")),
            "tables_read": reads, "no_table_read": not bool(reads),
            "execution_error": result.error, "read_detection": "sqlite_authorizer_execution"}


def score_query(db: Path, query, reference_sql):
    gold = reference_result(db, reference_sql)
    reads_gold = require_reference_reads(db, reference_sql, gold)
    result = execute(db, query, seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS) if query is not None else None
    reads = user_table_reads(db, result) if result is not None else []
    base = {"strict": False, "result_match": False, "bird_set": False, "error": None,
            "scorer_policy": POLICY, "tables_read": reads, "reference_tables_read": reads_gold,
            "no_table_read": query is not None and not reads}
    if query is None:
        return {**base, "error": "unsubmitted"}
    if result.error:
        return {**base, "error": result.error}
    policy = ordering_policy(db, reference_sql)
    if not policy["eligible"]:
        return {**base, "error": "ineligible_reference: " + policy["reason"]}
    match = rows_equal(result.rows, gold.rows, ordered=policy["ordered"])
    return {**base, "strict": bool(reads) and match, "result_match": match,
            "bird_set": set(result.rows) == set(gold.rows), "ordering_policy": policy,
            "error": None if reads else "literal_answer_no_table"}


def score_episode(db: Path, reference_sql, log):
    accepted = next((c for c in log if c["name"] == "submit_sql" and c.get("accepted")), None)
    last = next((c["args"].get("query") for c in reversed(log)
                 if c["name"] == "run_sql" and not c.get("error")), None)
    submitted = accepted["args"].get("query") if accepted else None
    strict = score_query(db, submitted, reference_sql)
    fallback = score_query(db, last, reference_sql)
    diagnostics = [submission_reads(db, c) for c in log if c["name"] == "submit_sql"]
    return {"success": strict["strict"], "strict": strict["strict"], "bird_set": strict["bird_set"],
            "fallback": fallback["result_match"], "submitted": accepted is not None,
            "error": strict["error"], "scorer_policy": POLICY, "submitted_tables_read": strict["tables_read"],
            "accepted_submission_no_table": bool(accepted) and strict["no_table_read"],
            "fallback_tables_read": fallback["tables_read"], "submission_read_diagnostics": diagnostics,
            "submission_count": len(diagnostics),
            "no_table_submissions": sum(d["no_table_read"] for d in diagnostics),
            "valid_no_table_submissions": sum(d["no_table_read"] and d["execution_error"] is None for d in diagnostics),
            "error_no_table_submissions": sum(d["no_table_read"] and d["execution_error"] is not None for d in diagnostics)}


def auditkit_check(path: Path, tasks):
    from auditkit.agent_eval import importers
    from auditkit.agent_eval.outcome import CustomPredicate
    from auditkit.agent_eval.runner import AgentEvalRunner, AgentEvalSpec
    from auditkit.agent_eval.types import AgentCase

    totals = {"n_episodes": 0, "n_rows": 0, "verdicts": {}, "scorer_policy": POLICY}
    with path.open(encoding="utf-8") as source, tempfile.TemporaryDirectory() as temp:
        batch_path = Path(temp) / "records.jsonl"
        while lines := list(islice(source, 200)):
            records = [json.loads(line) for line in lines if line.strip()]
            by_id = {str(r["metadata"]["task_id"]): r for r in records}

            def verified(episode, case):
                task = tasks[case.id]
                return score_episode(Path(task["db_path"]), task["reference_sql"],
                                     by_id[case.id]["metadata"]["call_log"])["success"]

            predicate = CustomPredicate(verified, version=POLICY, name="ayadb.sql_table_read")
            cases = [AgentCase(id=tid, task=tasks[tid]["prompt"], category="sql", outcome=predicate) for tid in by_id]
            batch_path.write_text("".join(lines), encoding="utf-8")
            episodes = importers.episodes_from_agenttune(str(batch_path))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes,
                                                           name=POLICY, scorers=[]))
            totals["n_episodes"] += len(episodes)
            totals["n_rows"] += len(result.rows)
            totals["verdicts"].update({r.case_id: r.outcome.get("verdict") for r in result.rows})
    return totals
