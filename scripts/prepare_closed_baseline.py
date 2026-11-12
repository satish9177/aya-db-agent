"""Build the closed-model baseline registration locally (no network, no spend).

One closed model answers the 144 SQaLe final questions with the same tools, eight tool steps, prompt
text and strict scoring as Tiny Aya. Running it needs the user's explicit approval of this file's SHA256.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ayadb.promptquality import fingerprint

RUN_ID = "closed-gpt6luna-sqale-final-20261111-01"
OUT = Path("experiments/closed-baseline-v1/REGISTRATION.json")
TASKS = Path("data/sqale/data-build-v3/final.tasks.jsonl")
SELECTION = Path("experiments/sql-final-b/SELECTION.json")
TOKENIZER = Path("data/sqale/tokenizer")
B_RUN = Path("remote/hf-sql/hf-user/final-b-sqale-sql-20261110-01/sqale-final-b")
A1500_RUN = Path("remote/hf-sql/hf-user/final-sqale-sql-20261109-01/sqale-final-a1500")
BASE_RUN = Path("remote/hf-sql/hf-user/final-sqale-sql-20261109-01/sqale-final-base")
T4_USD_PER_HOUR = 0.40
SOURCES = ["src/ayadb/apiengine.py", "src/ayadb/engine.py", "src/ayadb/sqlrunner.py", "src/ayadb/sqlenv.py",
           "src/ayadb/sqlguard.py", "src/ayadb/sqlscore.py", "src/ayadb/runinputs.py", "src/ayadb/protocol.py",
           "scripts/closed_baseline.py", "scripts/prepare_closed_baseline.py",
           "experiments/closed-baseline-v1/PREREGISTRATION.md"]


def comparator(run: Path, name: str) -> dict:
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    return {"name": name, "metrics": (run / "metrics.jsonl").as_posix(), "metrics_sha256": fingerprint(run / "metrics.jsonl"),
            "summary_sha256": fingerprint(run / "summary.json"), "strict": summary["strict_successes"],
            "episodes": summary["episodes"], "prompt_tokens": summary["prompt_tokens"],
            "generated_tokens": summary["generated_tokens"], "seconds": round(summary["seconds"], 1),
            "gpu_usd_by_rate": round(summary["seconds"] / 3600 * T4_USD_PER_HOUR, 4)}


def build() -> dict:
    return {
        "id": RUN_ID,
        "kind": "closed-model-baseline",
        "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
        "authorization": "Draft only: the user's explicit approval must name this file's SHA256 and covers one run, "
                         "at most max_cost_usd of API spend",
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-6-luna",
        "key_env": "OPENAI_API_KEY",
        "key_file": "~/.aya-keys/openai.key",
        "max_tokens_field": "max_completion_tokens",
        "max_new_tokens": 512,
        "request_options": {"reasoning_effort": "none", "temperature": 0, "stop": ["}]", "Tool result"]},
        "drop_if_rejected": ["temperature", "stop"],
        "prices_usd_per_million": {"input": 0.10, "cached_input": 0.01, "output": 0.50,
                                   "source": "https://developers.openai.com/api/docs/pricing",
                                   "checked": "2026-11-11", "tier": "Standard"},
        "max_cost_usd": 1.50,
        "retry_policy": "Transient HTTP errors (408/409/429/5xx, connection) retry the same request up to 5 times. "
                        "No episode is ever re-run. If the cap or an error stops the run, report what finished.",
        "tasks": {"path": TASKS.as_posix(), "sha256": fingerprint(TASKS), "episodes": 144},
        "selection": {"path": SELECTION.as_posix(), "sha256": fingerprint(SELECTION)},
        "tokenizer": {"path": TOKENIZER.as_posix(), "tokenizer_json_sha256": fingerprint(TOKENIZER / "tokenizer.json"),
                      "role": "prompt construction and the 8,192-token length guard, exactly as for Tiny Aya"},
        "protocol": {"format": "sql-submit-v1", "tools": "original (no error hints)", "tool_steps": 8,
                     "context_tokens": 8192, "temperature": 0.0, "seed": 42, "samples": 1,
                     "scorer_policy": "table-read-v1", "auditkit_agreement_required": True},
        "out": f"runs/closed/{RUN_ID}/sqale-final",
        "comparators": [comparator(BASE_RUN, "Base"), comparator(A1500_RUN, "A1500"), comparator(B_RUN, "B")],
        "analysis": {
            "primary": "strict successes of the closed model vs B on the same 144 tasks; exact two-sided McNemar",
            "secondary": ["set and fallback scores", "ran out of tool steps", "submitted queries that failed to run",
                          "malformed turns", "strict by query shape"],
            "roi": "API USD per question and per strict success vs B's T4 USD per question and per strict success "
                   "(rate $0.40/h); seconds per question; B's one-time training cost reported separately",
        },
        "source_sha256": {s: fingerprint(Path(s)) for s in SOURCES},
    }


def main():
    if OUT.exists() and "--force" not in sys.argv:
        sys.exit(f"{OUT} exists; registrations are not rewritten")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(OUT, fingerprint(OUT))


if __name__ == "__main__":
    main()
