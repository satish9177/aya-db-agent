"""Eligibility predicate for a future, separately approved Q2 filter; never samples."""
from __future__ import annotations

import json
from pathlib import Path

from .protocol import audit_record
from .sqlguard import POLICY, score_episode

PARAMETERS = {"list_tables": set(), "describe_table": {"table"},
              "run_sql": {"query"}, "submit_sql": {"query"}}


def candidate_eligibility(task, record):
    meta = record.get("metadata") or {}
    conversation, log = meta.get("conversation"), meta.get("call_log")
    reasons = []
    if not isinstance(conversation, list) or not conversation or not isinstance(log, list) or not log:
        return {"eligible": False, "reasons": ["missing_conversation_or_call_log"], "scorer_policy": POLICY}
    parsed_calls = []
    for msg in conversation:
        if msg.get("role") != "assistant" or not msg.get("content"):
            continue  # Empty local terminal turn is not a generated malformed call.
        try:
            calls = json.loads(msg["content"])
            valid = (isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict)
                     and set(calls[0]) == {"tool_name", "parameters"}
                     and calls[0]["tool_name"] in PARAMETERS
                     and isinstance(calls[0]["parameters"], dict)
                     and set(calls[0]["parameters"]) == PARAMETERS[calls[0]["tool_name"]]
                     and all(isinstance(v, str) for v in calls[0]["parameters"].values()))
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid:
            reasons.append("malformed_or_invented_assistant_turn")
        else:
            parsed_calls.append([calls[0]["tool_name"], calls[0]["parameters"]])
    if parsed_calls != [[c.get("name"), c.get("args")] for c in log]:
        reasons.append("assistant_calls_do_not_match_log")
    audit = audit_record(record)
    if audit["hallucinated_tool_output"] or audit["orch_result_mismatch"]:
        reasons.append("invented_or_mismatched_tool_output")
    if audit["orch_parser_miss"] or audit["orch_boundary"] or audit["orch_truncated"]:
        reasons.append("incomplete_protocol")
    try:
        metric = score_episode(Path(task["db_path"]), task["reference_sql"], log)
    except (ValueError, KeyError, TypeError):
        return {"eligible": False, "reasons": sorted(set(reasons + ["invalid_score_inputs"])), "scorer_policy": POLICY}
    if not metric["strict"] or not metric["submitted_tables_read"]:
        reasons.append("not_guarded_strict_success")
    return {"eligible": not reasons, "reasons": sorted(set(reasons)), "scorer_policy": POLICY,
            "strict": metric["strict"], "tables_read": metric["submitted_tables_read"]}
