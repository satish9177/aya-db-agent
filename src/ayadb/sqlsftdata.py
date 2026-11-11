"""Gold paths, real-error repair and AgentTune replay checks for sql-submit-v1."""
from __future__ import annotations

import json
import re
import warnings
import tempfile
from itertools import islice
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema, _run_trajectories

from .sftdata import ScriptedEngine, conversation, render, system_message, tokenize
from .sqlenv import FALLBACK, RESULT_FORMAT, SYSTEM_PROMPT, SQLTools, SubmissionEngine
from .sqlscore import FORMAT, execute, score_episode
from .tasks import file_sha256

MAX_STEPS = 8


def call_text(name: str, args: dict) -> str:
    return json.dumps([{"tool_name": name, "parameters": args}], ensure_ascii=False)


def make_episode(task: dict, *, repair: bool = False, calls: list[dict] | None = None,
                 repair_query: str | None = None) -> dict:
    """All observations are executed; a failed injected query is context, never a target."""
    tools = SQLTools(Path(task["db_path"]), task["id"], task.get("discovery_order"))
    by_name = {fn.__name__: fn for fn in tools}
    planned = calls
    if planned is None:
        planned = [{"name": "list_tables", "args": {}, "train": True}]
        planned += [{"name": "describe_table", "args": {"table": t}, "train": True}
                    for t in task["gold_tables"]]
        if repair:
            # Historical v2 path remains available. v3 supplies a gold-derived,
            # independently error-class-checked mutation explicitly.
            bad = repair_query if repair_query is not None else re.sub(
                r"\bSELECT\s+(DISTINCT\s+)?", r"SELECT \1__ayadb_bad_alias__.x, ",
                task["reference_sql"], count=1, flags=re.I)
            failure = execute(Path(task["db_path"]), bad)
            if not failure.error:
                raise ValueError("injected repair query did not actually fail")
            planned.append({"name": "run_sql", "args": {"query": bad}, "train": False})
        planned += [{"name": "run_sql", "args": {"query": task["reference_sql"]}, "train": True},
                    {"name": "submit_sql", "args": {"query": task["reference_sql"]}, "train": True}]
    if len(planned) > MAX_STEPS:
        raise ValueError("tool_step_limit")
    messages = []
    for c in planned:
        result = by_name[c["name"]](**c["args"])
        if result.startswith(("Error", "SQL error")) and c.get("train", True):
            raise ValueError("expert_call_failed: " + result)
        messages.append({"role": "assistant", "content": call_text(c["name"], c["args"]),
                         "train": c.get("train", True)})
        if c["name"] != "submit_sql":
            messages.append({"role": "user", "content": RESULT_FORMAT.format(name=c["name"], content=result),
                             "train": False})
    return {"id": task["id"] + ("-repair" if repair else "-gold"), "format": FORMAT,
            "task_id": task["id"], "prompt": task["prompt"], "schema_id": task["schema_id"],
            "question_id": task["question_id"], "phrasing": task.get("phrasing", "verbose"),
            "reference_sql": task["reference_sql"],
            "db_path": task["db_path"], "db_sha256": file_sha256(Path(task["db_path"])),
            "discovery_order": task.get("discovery_order"), "messages": messages,
            "calls": tools.log, "repair": repair}


def encode_episode(ep: dict, tokenizer, max_len: int = 4096) -> dict | None:
    tools = SQLTools(Path(ep["db_path"]), ep["task_id"], ep.get("discovery_order"))
    schemas = [_build_tool_schema(fn) for fn in tools]
    conv = conversation(ep["prompt"], ep["messages"], system_message(SYSTEM_PROMPT, FALLBACK, schemas))
    flags = [False, False] + [bool(m.get("train")) for m in ep["messages"]]
    return tokenize(tokenizer, conv, flags, schemas, max_len)


def replay_sql_audit(ep: dict, db: Path, db_sha: str, tokenizer) -> list[str]:
    """Called via sftdata.replay_audit; real AgentTune parser, tools, rendering and scoring.

    Replays on the immutable prepared slice through independent read-only connections; the
    before/after hash verifies that no replay can mutate it. No model is loaded or sampled.
    """
    problems = []
    if file_sha256(db) != db_sha:
        return ["prepared database hash mismatch"]
    tools = SQLTools(db, ep["task_id"], ep.get("discovery_order"))
    schemas = [_build_tool_schema(fn) for fn in tools]
    scripted = ScriptedEngine(tokenizer, [m["content"] for m in ep["messages"] if m["role"] == "assistant"])
    engine = SubmissionEngine(scripted, tools)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = _run_trajectories([{"prompt": ep["prompt"], "answer": ""}], tools, True, engine,
                                 MAX_STEPS, None, None, {}, system_prompt=SYSTEM_PROMPT,
                                 context_length=8192, tools_fallback_prompt=FALLBACK,
                                 tool_result_format=RESULT_FORMAT)[0]
    traj.trajectory_id = ep["id"]
    record = traj.to_dict()
    record.setdefault("metadata", {}).update({"task_id": ep["id"], "source_task_id": ep["task_id"],
                                               "call_log": tools.log, "format": FORMAT})
    ep["_replay_record"] = record
    if scripted.outputs:
        problems.append("scripted turns not consumed")
    keys = lambda log: [(c["name"], c["args"], c["result"]) for c in log]
    if keys(tools.log) != keys(ep["calls"]):
        problems.append("replayed observations differ")
    if not tools.finished or not tools.log[-1].get("accepted"):
        problems.append("no terminal submission")
    conv = conversation(ep["prompt"], ep["messages"], system_message(SYSTEM_PROMPT, FALLBACK, schemas))
    prefixes = [render(tokenizer, conv[:i], schemas, True) for i, m in enumerate(conv) if m["role"] == "assistant"]
    if scripted.prompts != prefixes:
        problems.append("AgentTune prompts differ from training prefixes")
    if file_sha256(db) != db_sha:
        problems.append("database mutated")
    submitted = [c for c in tools.log if c["name"] == "submit_sql" and c.get("accepted")]
    if len(submitted) != 1:
        problems.append("expected exactly one accepted submission")
    if not score_episode(db, ep["reference_sql"], tools.log)["success"]:
        problems.append("submitted query fails full-result verification")
    return problems


def auditkit_check(path: Path, tasks: dict[str, dict]) -> dict:
    """AuditKIT independently re-scores saved AgentTune records using the full SQL metric."""
    from auditkit.agent_eval import importers
    from auditkit.agent_eval.outcome import CustomPredicate
    from auditkit.agent_eval.runner import AgentEvalRunner, AgentEvalSpec
    from auditkit.agent_eval.types import AgentCase

    totals = {"n_episodes": 0, "n_rows": 0, "verdicts": {}}
    # Bounded batches keep the complete trajectory corpus out of memory.
    with path.open(encoding="utf-8") as source, tempfile.TemporaryDirectory() as temporary:
        batch_path = Path(temporary) / "records.jsonl"
        while lines := list(islice(source, 200)):
            records = [json.loads(line) for line in lines if line.strip()]
            by_id = {str(r["metadata"]["task_id"]): r for r in records}

            def verified(episode, case):
                task = tasks[case.id]
                log = by_id[case.id]["metadata"]["call_log"]
                return score_episode(Path(task["db_path"]), task["reference_sql"], log)["success"]

            predicate = CustomPredicate(verified, version=FORMAT, name="ayadb.sql_full_result")
            cases = [AgentCase(id=tid, task=tasks[tid]["prompt"], category="sql", outcome=predicate) for tid in by_id]
            batch_path.write_text("".join(lines), encoding="utf-8")
            episodes = importers.episodes_from_agenttune(str(batch_path))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes,
                                                           name=FORMAT, scorers=[]))
            totals["n_episodes"] += len(episodes)
            totals["n_rows"] += len(result.rows)
            totals["verdicts"].update({r.case_id: r.outcome.get("verdict") for r in result.rows})
    return totals
