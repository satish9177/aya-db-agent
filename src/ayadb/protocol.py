"""Rollout protocol audit: separates model failures from orchestration failures.

Model failures (per assistant turn): text around a call, invented tool output, prose instead of
a call or answer. Orchestration failures (per episode): a tool-result turn whose content is not
exactly what the environment returned, a well-formed call the parser missed, text generated past
a call boundary, or a generation cut by the token limit.
"""

from __future__ import annotations

import json
import re

from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

from .tasks import extract_answer

TOOL_NAMES = {"list_tables", "describe_table", "run_sql"}
# Text a model only writes when it imitates a tool's output instead of waiting for it.
_FAKE_OUTPUT = re.compile(r"Tool result|Sample rows|^Tables?:|SQL error|Columns?:|^\|.*\|$",
                          re.IGNORECASE | re.MULTILINE)
_CALL_LIST = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def _pure_call(text: str) -> bool:
    try:
        calls = json.loads(text.strip())
    except ValueError:
        return False
    return (isinstance(calls, list) and bool(calls)
            and all(isinstance(c, dict) and set(c) == {"tool_name", "parameters"} for c in calls))


def classify_turn(content: str) -> str:
    """pure_call | call_with_text | answer | fake_output | other_text (one assistant turn)."""
    text = content or ""
    if _extract_tool_calls({"role": "assistant", "content": text}, text):
        return "pure_call" if _pure_call(text) else "call_with_text"
    if _FAKE_OUTPUT.search(text):
        return "fake_output"
    if extract_answer(text) is not None:
        return "answer"
    return "other_text"


def _segments(content: str, result_format: str) -> list[tuple[str, str]]:
    """Split a folded tool-result user turn into (name, content) pairs using its format."""
    prefix, _, rest = result_format.partition("{name}")
    sep, _, suffix = rest.partition("{content}")
    parts = re.split(r"(?:^|\n\n)" + re.escape(prefix), content)
    out = []
    for p in parts[1:]:
        name, _, body = p.partition(sep)
        if suffix:
            if not body.endswith(suffix):
                return []
            body = body[: -len(suffix)]
        out.append((name, body))
    return out if parts[0] == "" else []


def audit_record(record: dict, result_format: str = "Tool result for {name}: {content}") -> dict:
    """Protocol audit of one saved trajectory record (Trajectory.to_dict() + our metadata)."""
    meta = record.get("metadata") or {}
    conv = meta.get("conversation") or []
    log = list(meta.get("call_log") or [])
    turns, result_mismatch, parser_miss, boundary = [], 0, 0, 0
    seen_task = False
    for msg in conv:
        role, content = msg.get("role"), msg.get("content") or ""
        if role == "assistant":
            kind = classify_turn(content)
            turns.append(kind)
            if kind in ("pure_call", "call_with_text") and not content.rstrip().endswith("]"):
                boundary += 1
            if kind not in ("pure_call", "call_with_text"):
                m = _CALL_LIST.search(content)
                if m and _pure_call(m.group(0)) and all(
                        c["tool_name"] in TOOL_NAMES for c in json.loads(m.group(0))):
                    parser_miss += 1
        elif role == "user":
            if not seen_task:
                seen_task = True
                continue
            segs = _segments(content, result_format)
            if not segs:
                result_mismatch += 1
                continue
            for name, body in segs:
                if log and log[0]["name"] == name and log[0]["result"] == body:
                    log.pop(0)  # exactly what the environment returned
                elif not body.startswith("{'error'"):
                    # AgentTune's own error dict (unknown tool, undecodable args) is allowed;
                    # anything else is an observation the environment did not produce.
                    result_mismatch += 1
    unconsumed = len(log)  # executed calls whose results never reached the model
    truncated = sum(1 for u in meta.get("usage") or [] if u.get("done_reason") == "length")
    return {
        "turn_kinds": turns,
        "n_turns": len(turns),
        "pure_call_turns": turns.count("pure_call"),
        "call_with_text_turns": turns.count("call_with_text"),
        "fake_output_turns": turns.count("fake_output"),
        "other_text_turns": turns.count("other_text"),
        "answer_turns": turns.count("answer"),
        "hallucinated_tool_output": turns.count("fake_output") + sum(
            1 for msg in conv if msg.get("role") == "assistant"
            and classify_turn(msg.get("content") or "") == "call_with_text"
            and "tool result" in (msg.get("content") or "").lower()),
        "orch_result_mismatch": result_mismatch + unconsumed,
        "orch_parser_miss": parser_miss,
        "orch_boundary": boundary,
        "orch_truncated": truncated,
    }


def summarize_audit(rows: list[dict]) -> dict:
    n = len(rows)
    t = sum(r["n_turns"] for r in rows) or 1
    tot = lambda k: sum(r[k] for r in rows)  # noqa: E731
    return {
        "episodes": n, "assistant_turns": t,
        "pure_call_turn_share": tot("pure_call_turns") / t,
        "call_with_text_turn_share": tot("call_with_text_turns") / t,
        "fake_output_turn_share": tot("fake_output_turns") / t,
        "other_text_turn_share": tot("other_text_turns") / t,
        "episodes_with_hallucinated_tool_output": sum(r["hallucinated_tool_output"] > 0 for r in rows) / n,
        "orchestration": {k: tot(k) for k in ("orch_result_mismatch", "orch_parser_miss",
                                              "orch_boundary", "orch_truncated")},
    }
