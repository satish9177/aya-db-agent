"""Closed-model baseline engine: fake HTTP only, no network and no spend."""
import json
import sqlite3

import pytest

from ayadb.apiengine import BudgetExceeded, ChatAPIEngine, call_cost, read_api_key
from ayadb.promptquality import fingerprint
from ayadb.sqlsftdata import call_text

KEY = "fake-test-key-not-real-0123456789"
PRICES = {"input": 0.10, "cached_input": 0.01, "output": 0.50}


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class Reply:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def ok(content, prompt=100, completion=20, cached=0):
    return Reply(200, {"model": "fake-model", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                                 "prompt_tokens_details": {"cached_tokens": cached}}})


class FakeSession:
    def __init__(self, replies):
        self.replies, self.bodies, self.headers = list(replies), [], []

    def post(self, url, headers, json, timeout):
        self.bodies.append(dict(json))
        self.headers.append(headers)
        return self.replies.pop(0)


def engine(session, tokenizer=None, **kw):
    options = dict(base_url="https://api.example/v1", model="fake-model", api_key=KEY,
                   tokenizer=tokenizer or FakeTokenizer(), prices=PRICES, max_cost_usd=1.0,
                   session=session, sleep=lambda s: None)
    options.update(kw)
    return ChatAPIEngine(**options)


def test_messages_match_the_tiny_aya_text_turns():
    conversation = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "[{\"tool_name\": \"list_tables\", \"parameters\": {}}]",
                     "tool_calls": [{"type": "function", "function": {"name": "list_tables", "arguments": {}}}]},
                    {"role": "tool", "name": "list_tables", "content": "Tables: a"}]
    msgs = engine(FakeSession([])).messages(conversation)
    assert msgs == [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "[{\"tool_name\": \"list_tables\", \"parameters\": {}}]"},
                    {"role": "user", "content": "Tool result for list_tables: Tables: a"}]


def test_reply_uses_the_shared_stop_rules_and_counts_cost():
    session = FakeSession([ok('[{"tool_name": "list_tables", "parameters": {}}]\nTool result for list_tables: made up',
                              prompt=1000, completion=40, cached=600)])
    e = engine(session, request_options={"reasoning_effort": "none"})
    out = e.generate([{"role": "user", "content": "Q"}], [], {"temperature": 0.0})
    assert out["completions"] == ['[{"tool_name": "list_tables", "parameters": {}}]']
    body = session.bodies[0]
    assert body["max_completion_tokens"] == 512 and body["reasoning_effort"] == "none" and body["model"] == "fake-model"
    assert e.spent_usd == pytest.approx((400 * .10 + 600 * .01 + 40 * .50) / 1e6)
    assert e.calls[0]["cached_tokens"] == 600 and KEY not in json.dumps(e.calls)


def test_call_is_refused_before_it_can_pass_the_cap():
    session = FakeSession([ok("x")])
    e = engine(session, max_cost_usd=0.0001)
    with pytest.raises(BudgetExceeded):
        e.generate([{"role": "user", "content": "word " * 50}], [], {"temperature": 0.0})
    assert session.bodies == []


def test_rejected_optional_parameter_is_dropped_once_then_retried():
    session = FakeSession([Reply(400, text="Unsupported parameter: 'temperature' is not supported with this model."),
                           ok("ANSWER"), ok("ANSWER")])
    e = engine(session, request_options={"temperature": 0, "stop": ["}]"]}, drop_if_rejected=["temperature", "stop"])
    e.generate([{"role": "user", "content": "Q"}], [], {"temperature": 0.0})
    e.generate([{"role": "user", "content": "Q"}], [], {"temperature": 0.0})
    assert e.dropped_options == ["temperature"]
    assert "temperature" in session.bodies[0] and all("temperature" not in b for b in session.bodies[1:])
    assert all(b["stop"] == ["}]"] for b in session.bodies)


def test_transient_errors_retry_and_auth_errors_do_not():
    session = FakeSession([Reply(429, text="slow down"), ok("ANSWER")])
    e = engine(session)
    e.generate([{"role": "user", "content": "Q"}], [], {"temperature": 0.0})
    assert e.calls[0]["retries"] == 1
    bad = engine(FakeSession([Reply(401, text="invalid key")]))
    with pytest.raises(RuntimeError) as err:
        bad.generate([{"role": "user", "content": "Q"}], [], {"temperature": 0.0})
    assert KEY not in str(err.value)


def test_only_greedy_decoding_is_allowed():
    with pytest.raises(ValueError, match="greedy"):
        engine(FakeSession([])).generate([{"role": "user", "content": "Q"}], [], {"temperature": 0.8})


def test_key_comes_from_env_or_file_and_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("AYADB_TEST_KEY", raising=False)
    with pytest.raises(PermissionError):
        read_api_key("AYADB_TEST_KEY", str(tmp_path / "missing.key"))
    (tmp_path / "k.key").write_text(" file-key \n", encoding="utf-8")
    assert read_api_key("AYADB_TEST_KEY", str(tmp_path / "k.key")) == "file-key"
    monkeypatch.setenv("AYADB_TEST_KEY", "env-key")
    assert read_api_key("AYADB_TEST_KEY", str(tmp_path / "k.key")) == "env-key"


def test_cost_uses_provider_usage():
    assert call_cost({"prompt_tokens": 2_000_000, "completion_tokens": 1_000_000}, PRICES) == pytest.approx(0.70)


def test_full_episode_through_the_real_runner_sees_the_same_prompt_as_tiny_aya(tmp_path):
    from ayadb.engine import official_tokenizer
    from ayadb.sqlrunner import evaluate_with_engine
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    db = tmp_path / "fixture.sqlite"
    con = sqlite3.connect(db); con.executescript("CREATE TABLE items(value INTEGER); INSERT INTO items VALUES(7);"); con.close()
    query = "SELECT value FROM items"
    session = FakeSession([ok(call_text("run_sql", {"query": query})), ok(call_text("submit_sql", {"query": query}))])
    e = engine(session, tokenizer=tokenizer)
    task = {"id": "fixture", "prompt": "Return the stored value.", "db_path": str(db),
            "db_sha256": fingerprint(db), "reference_sql": query}
    result = evaluate_with_engine([task], e, tmp_path / "eval")
    assert result["strict_successes"] == result["auditkit_agreement"] == 1
    first, second = session.bodies
    system = first["messages"][0]
    assert system["role"] == "system" and '"submit_sql"' in system["content"] and '"run_sql"' in system["content"]
    assert second["messages"][-1]["role"] == "user" and second["messages"][-1]["content"].startswith("Tool result for run_sql:")
    assert result["prompt_tokens"] == 200 and result["generated_tokens"] == 40
    for path in (tmp_path / "eval").iterdir():
        assert KEY not in path.read_text(encoding="utf-8")
