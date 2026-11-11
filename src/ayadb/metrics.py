"""Per-trajectory metrics, computed from the executed-call log and the AgentTune trajectory record.

Primary metric: `success` = verified correct answer AND at least one successful `run_sql`
(an answer with no query behind it is a guess, not an executed task). Everything else is
diagnostic. Works on saved JSONL records, so runs can be re-scored without the model.
"""

from __future__ import annotations

import functools
import json
import re
from collections import Counter

from .tasks import check_answer, extract_answer

TOOL_NAMES = ("list_tables", "describe_table", "run_sql")
ERROR_PREFIXES = ("Error", "SQL error")
_CALL_ATTEMPT = re.compile(r'tool_name|"parameters"|^\s*[\[{]', re.IGNORECASE)


def logged(tools: list, log: list) -> list:
    """Wrap tools so every execution lands in `log` as {name, args, result, error, bad_args}."""
    def wrap(fn):
        @functools.wraps(fn)
        def w(*args, **kwargs):
            bad = False
            try:
                out = fn(*args, **kwargs)
            except TypeError as e:  # wrong/missing argument names
                out, bad = f"Error: invalid arguments for {fn.__name__}: {e}", True
            log.append({"name": fn.__name__, "args": kwargs, "result": out, "bad_args": bad,
                        "error": bad or str(out).startswith(ERROR_PREFIXES)})
            return out
        return w
    return [wrap(t) for t in tools]


def _parsed_calls(record: dict) -> list[dict]:
    calls = []
    for step in record.get("steps") or []:
        action = step.get("action") or {}
        calls.extend(c for c in action.get("tool_calls") or [] if isinstance(c, dict))
    return calls


def _call_name(c: dict) -> str | None:
    fn = c["function"] if isinstance(c.get("function"), dict) else c
    return fn.get("name") or fn.get("tool_name")


def trajectory_metrics(record: dict, call_log: list[dict], gold: str, kind: str,
                       max_steps: int) -> dict:
    """`record` is Trajectory.to_dict(); `call_log` is what the logged tools executed."""
    parsed = _parsed_calls(record)
    names = [_call_name(c) for c in parsed]
    unknown = sum(1 for n in names if n not in TOOL_NAMES)
    known = len(parsed) - unknown
    executed = len(call_log)
    bad_args = sum(1 for c in call_log if c["bad_args"]) + max(known - executed, 0)
    valid = sum(1 for c in call_log if not c["bad_args"])
    exec_errors = sum(1 for c in call_log if c["error"] and not c["bad_args"])

    # AgentTune appends a terminal step with an empty action; only tool steps use the budget.
    tool_steps = [s for s in record.get("steps") or [] if (s.get("action") or {}).get("tool_calls")]
    hit_max = len(tool_steps) >= max_steps
    final = record.get("final_response") or ""
    # A call-shaped final turn that did not parse ended the episode early (invalid JSON).
    # After the budget runs out, the last turn is unexecuted by design: counted as hit_max.
    malformed_final = (not hit_max and bool(_CALL_ATTEMPT.search(final))
                       and extract_answer(final) is None)
    attempted = len(parsed) + int(malformed_final)

    keys = Counter((c["name"], json.dumps(c["args"], sort_keys=True)) for c in call_log)
    repeated = sum(v - 1 for v in keys.values() if v > 1)
    loop = any(v >= 3 for v in keys.values()) or hit_max

    answer = extract_answer(final)
    correct = check_answer(answer, gold, kind)
    grounded = any(c["name"] == "run_sql" and not c["error"] for c in call_log)
    # Diagnostic: the answered value literally appears in a successful query result.
    seen = answer is not None and any(
        c["name"] == "run_sql" and not c["error"] and re.search(
            rf"(?<![\w.]){re.escape(answer.strip())}(?![\w.])", str(c["result"]), re.IGNORECASE)
        for c in call_log)
    # Genuine recovery: the episode hit an error (bad SQL, unknown table/column, bad arguments),
    # then a successful run_sql came after its last error, and the answer is verified.
    errs = [i for i, c in enumerate(call_log) if c["error"]]
    recovered = bool(errs) and correct and grounded and any(
        c["name"] == "run_sql" and not c["error"] for c in call_log[errs[-1] + 1:])
    conv = (record.get("metadata") or {}).get("conversation") or []
    turns = sum(1 for m in conv if isinstance(m, dict) and m.get("role") == "assistant")
    return {
        "success": correct and grounded,
        "had_error": bool(errs),
        "recovered": recovered,
        "correct": correct,
        "grounded": grounded,
        "answer_seen_in_sql": bool(seen),
        "answered": answer is not None,
        "lenient_correct": bool(re.search(rf"(?<![\w.]){re.escape(gold)}(?![\w])", final,
                                          re.IGNORECASE)),
        "no_tool_answer": answer is not None and executed == 0,
        "turns": turns,
        "tool_calls": len(parsed),
        "valid_calls": valid,
        "attempted_calls": attempted,
        "unknown_tool_calls": unknown,
        "bad_arg_calls": bad_args,
        "exec_errors": exec_errors,
        "malformed_final": malformed_final,
        "repeated_calls": repeated,
        "loop": loop,
        "hit_max_steps": hit_max,
        "answer": answer,
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-trajectory metric rows."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    rate = lambda k: sum(bool(r[k]) for r in rows) / n  # noqa: E731
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    attempted = sum(r["attempted_calls"] for r in rows)
    with_error = sum(bool(r.get("had_error")) for r in rows)
    successes = [r for r in rows if r["success"]]
    return {
        "n": n,
        "success_rate": rate("success"),
        "correct_rate": rate("correct"),
        "answered_rate": rate("answered"),
        "lenient_correct_rate": rate("lenient_correct"),
        "grounded_rate": rate("grounded"),
        "answer_seen_in_sql_rate": rate("answer_seen_in_sql"),
        "no_tool_answer_rate": rate("no_tool_answer"),
        "valid_call_rate": (sum(r["valid_calls"] for r in rows) / attempted) if attempted else None,
        "malformed_final_rate": rate("malformed_final"),
        "bad_arg_calls_per_task": mean("bad_arg_calls"),
        "unknown_tool_calls_per_task": mean("unknown_tool_calls"),
        "exec_errors_per_task": mean("exec_errors"),
        "repeated_calls_per_task": mean("repeated_calls"),
        "loop_rate": rate("loop"),
        "mean_tool_calls": mean("tool_calls"),
        "mean_turns": mean("turns"),
        "zero_call_rate": sum(r["tool_calls"] == 0 for r in rows) / n,
        "error_episode_rate": with_error / n,
        "recovered_episodes": sum(bool(r.get("recovered")) for r in rows),
        "recovery_rate": (sum(bool(r.get("recovered")) for r in rows) / with_error) if with_error else None,
        "mean_tool_calls_when_successful": (sum(r["tool_calls"] for r in successes) / len(successes))
        if successes else None,
    }
