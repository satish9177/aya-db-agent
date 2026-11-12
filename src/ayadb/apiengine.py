"""Closed-model baseline: the same text protocol as Tiny Aya, sent to an OpenAI-compatible chat API.

AgentTune gets the official Tiny Aya tokenizer from `_get_tokenizer()`, so it builds exactly the same
conversation as for Tiny Aya: the text tool prompt in the system message, and tool results folded into
user turns. This engine sends that conversation as plain chat turns and applies the same stop handling
(`finish_completion`). Nothing about the tools, step limit or scoring changes.

The API key comes from an environment variable or a key file outside the repo. It is never stored,
printed or logged. Spending is capped: a call is refused if its worst case could pass the cap.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests
from agenttune.agentic.rollout_engines.base import RolloutEngine
from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user

from .engine import finish_completion

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    pass


def read_api_key(key_env: str, key_file: str | None = None) -> str:
    key = os.environ.get(key_env, "").strip()
    if not key and key_file:
        path = Path(os.path.expandvars(os.path.expanduser(key_file)))
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise PermissionError(f"no API key: set {key_env} or create the registered key file")
    return key


def call_cost(usage: dict, prices: dict) -> float:
    """USD for one call from the provider's own usage numbers (output includes any reasoning tokens)."""
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0)
    return ((prompt - cached) * prices["input"] + cached * prices.get("cached_input", prices["input"])
            + output * prices["output"]) / 1e6


class ChatAPIEngine(RolloutEngine):
    def __init__(self, *, base_url: str, model: str, api_key: str, tokenizer, prices: dict,
                 max_cost_usd: float, stops=("}]", "Tool result"), max_new_tokens: int = 512,
                 max_tokens_field: str = "max_completion_tokens", request_options: dict | None = None,
                 drop_if_rejected=(), tool_result_format: str = "Tool result for {name}: {content}",
                 timeout: float = 120, max_retries: int = 5, session=None, sleep=time.sleep):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.tokenizer, self.prices = model, tokenizer, dict(prices)
        self.max_cost_usd, self.stops, self.max_new_tokens = float(max_cost_usd), list(stops), int(max_new_tokens)
        self.max_tokens_field = max_tokens_field
        self.options = dict(request_options or {})
        self.drop_if_rejected = list(drop_if_rejected)
        self.dropped_options: list[str] = []
        self.tool_result_format, self.timeout, self.max_retries = tool_result_format, timeout, max_retries
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.session, self.sleep = session or requests.Session(), sleep
        self.calls: list[dict] = []
        self.spent_usd = 0.0
        self.first_messages: list[dict] | None = None

    def _get_tokenizer(self):
        return self.tokenizer

    def messages(self, conversation: list[dict]) -> list[dict]:
        out = []
        for m in fold_tool_messages_into_user(conversation, self.tool_result_format):
            if m["role"] not in ("system", "user", "assistant"):
                raise ValueError("unexpected conversation role: " + m["role"])
            out.append({"role": m["role"], "content": m.get("content") or ""})
        return out

    def _worst_case_usd(self, messages: list[dict]) -> float:
        # Generous: 1.5x the Tiny Aya token count, nothing cached, a full-length reply.
        text = "\n".join(m["content"] for m in messages)
        prompt = 1.5 * len(self.tokenizer(text, add_special_tokens=False)["input_ids"]) + 16 * len(messages)
        return (prompt * self.prices["input"] + self.max_new_tokens * self.prices["output"]) / 1e6

    def _post(self, body: dict) -> tuple[dict, int]:
        retries = 0
        while True:
            try:
                r = self.session.post(self.url, headers=self._headers, json=body, timeout=self.timeout)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                return r.json(), retries
            status = None if r is None else r.status_code
            if status == 400 and self.drop_if_rejected:
                # Preregistered fallback: drop an optional parameter the model rejects, then retry.
                message = r.text.lower()
                name = next((p for p in self.drop_if_rejected if p in message and p in body), None)
                if name:
                    body.pop(name)
                    self.options.pop(name, None)
                    self.dropped_options.append(name)
                    continue
            if (status is None or status in RETRY_STATUS) and retries < self.max_retries:
                retries += 1
                self.sleep(min(60, 2 ** retries))
                continue
            detail = "" if r is None else f" {r.status_code}: {r.text[:300]}"
            raise RuntimeError("chat API request failed" + detail)

    def generate(self, prompts, tools, gen_cfg):
        if float(gen_cfg.get("temperature", 0.0)) != 0.0:
            raise ValueError("the closed-model baseline is registered with greedy decoding only")
        messages = self.messages(prompts)
        if self.first_messages is None:
            self.first_messages = messages
        if self.spent_usd + self._worst_case_usd(messages) > self.max_cost_usd:
            raise BudgetExceeded(f"next call could pass the ${self.max_cost_usd:.2f} cap "
                                 f"(spent ${self.spent_usd:.4f})")
        body = {"model": self.model, "messages": messages, self.max_tokens_field: self.max_new_tokens,
                **self.options}
        t0 = time.time()
        out, retries = self._post(body)
        choice = (out.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = out.get("usage") or {}
        cost = call_cost(usage, self.prices)
        self.spent_usd += cost
        self.calls.append({
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "seconds": round(time.time() - t0, 3), "done_reason": choice.get("finish_reason"),
            "retries": retries, "cost_usd": round(cost, 8), "model_reported": out.get("model")})
        completion = finish_completion(text, self.stops)
        return {"completions": [completion], "completions_raw": [completion], "logprobs": [None], "metadata": {}}
