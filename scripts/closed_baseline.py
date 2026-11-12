"""Closed-model baseline on SQaLe final: free `plan`, approved `run` (one run per registration).

  .venv/Scripts/python.exe scripts/closed_baseline.py plan experiments/closed-baseline-v1/REGISTRATION.json
  .venv/Scripts/python.exe scripts/closed_baseline.py run  experiments/closed-baseline-v1/REGISTRATION.json

`run` needs AYADB_APPROVED_REGISTRATION set to the registration's SHA256 (the user's approval) and an API
key in the registered environment variable or key file. The key is never printed or written.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from ayadb.apiengine import BudgetExceeded, ChatAPIEngine, read_api_key
from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.runinputs import evaluation_tasks
from ayadb.runledger import atomic_json
from ayadb.sqlenv import RESULT_FORMAT
from ayadb.sqlguard import require_reference_reads
from ayadb.sqlrunner import evaluate_with_engine

TOKEN_FACTOR = 1.25  # provider tokens per Tiny Aya token, a cautious guess for the estimate only


def load(path: Path) -> dict:
    reg = json.loads(path.read_text(encoding="utf-8"))
    if reg.get("kind") != "closed-model-baseline":
        raise ValueError("not a closed-model baseline registration")
    for item in (reg["tasks"], reg["selection"]):
        if Path(item["path"]).resolve() == Path("data/test.jsonl").resolve():
            raise ValueError("sealed test is prohibited")
        if fingerprint(Path(item["path"])) != item["sha256"]:
            raise ValueError("registered input changed: " + item["path"])
    tok = Path(reg["tokenizer"]["path"]) / "tokenizer.json"
    if fingerprint(tok) != reg["tokenizer"]["tokenizer_json_sha256"]:
        raise ValueError("tokenizer changed")
    for name, digest in reg["source_sha256"].items():
        if fingerprint(Path(name)) != digest:
            raise ValueError("registered source changed: " + name)
    for c in reg["comparators"]:
        if fingerprint(Path(c["metrics"])) != c["metrics_sha256"]:
            raise ValueError("comparator metrics changed: " + c["name"])
    if not 0 < reg["max_cost_usd"] <= 5:
        raise ValueError("spending cap missing or above $5")
    return reg


def estimate(reg: dict) -> dict:
    p, steps = reg["prices_usd_per_million"], reg["protocol"]["tool_steps"]
    b = next(c for c in reg["comparators"] if c["name"] == "B")
    central = (b["prompt_tokens"] * TOKEN_FACTOR * p["input"] + b["generated_tokens"] * TOKEN_FACTOR * p["output"]) / 1e6
    calls = reg["tasks"]["episodes"] * (steps + 1)
    worst = (calls * reg["protocol"]["context_tokens"] * TOKEN_FACTOR * p["input"]
             + calls * reg["max_new_tokens"] * p["output"]) / 1e6
    return {"central_usd_like_B": round(central, 3), "worst_case_usd_every_call_full": round(worst, 3),
            "cap_usd": reg["max_cost_usd"], "b_gpu_usd_by_rate": b["gpu_usd_by_rate"],
            "b_usd_per_question": round(b["gpu_usd_by_rate"] / b["episodes"], 5),
            "b_usd_per_strict_success": round(b["gpu_usd_by_rate"] / b["strict"], 5)}


def run(path: Path, reg: dict) -> dict:
    digest = fingerprint(path)
    if os.environ.get("AYADB_APPROVED_REGISTRATION") != digest:
        raise PermissionError("running needs the user's approval: AYADB_APPROVED_REGISTRATION must equal " + digest)
    out = Path(reg["out"])
    if out.exists():
        raise FileExistsError(f"{out} exists: one run per registration")
    tasks = evaluation_tasks(Path(reg["tasks"]["path"]), purpose="final", selection_sha256=reg["selection"]["sha256"])
    if len(tasks) != reg["tasks"]["episodes"]:
        raise ValueError(f"expected {reg['tasks']['episodes']} tasks, found {len(tasks)}")
    for task in tasks:
        require_reference_reads(Path(task["db_path"]), task["reference_sql"])
    engine = ChatAPIEngine(
        base_url=reg["base_url"], model=reg["model"], api_key=read_api_key(reg["key_env"], reg.get("key_file")),
        tokenizer=official_tokenizer(reg["tokenizer"]["path"]),
        prices={k: reg["prices_usd_per_million"][k] for k in ("input", "cached_input", "output")},
        max_cost_usd=reg["max_cost_usd"], max_new_tokens=reg["max_new_tokens"],
        max_tokens_field=reg["max_tokens_field"], request_options=reg["request_options"],
        drop_if_rejected=reg["drop_if_rejected"], tool_result_format=RESULT_FORMAT)
    record_dir = out.parent
    record_dir.mkdir(parents=True, exist_ok=True)
    started, status, error = time.time(), "completed", None
    proto = reg["protocol"]
    try:
        summary = evaluate_with_engine(tasks, engine, out, samples=proto["samples"], temperature=proto["temperature"],
                                       seed=proto["seed"], context_tokens=proto["context_tokens"],
                                       tool_steps=proto["tool_steps"], purpose="closed-baseline-final")
    except BudgetExceeded as exc:
        summary, status, error = None, "stopped_at_spending_cap", str(exc)
    except Exception as exc:  # report what finished; never retried automatically
        summary, status, error = None, "stopped_by_error", f"{type(exc).__name__}: {exc}"[:500]
    with (record_dir / "api-calls.jsonl").open("w", encoding="utf-8") as f:
        for call in engine.calls:
            f.write(json.dumps(call) + "\n")
    done = sum(1 for _ in (out / "metrics.jsonl").open(encoding="utf-8")) if (out / "metrics.jsonl").exists() else 0
    report = {"run_id": reg["id"], "registration_sha256": digest, "status": status, "error": error,
              "episodes_finished": done, "api_calls": len(engine.calls), "spent_usd": round(engine.spent_usd, 6),
              "dropped_options": engine.dropped_options,
              "models_reported": sorted({c["model_reported"] for c in engine.calls if c.get("model_reported")}),
              "first_request_messages_sha256": None, "summary": summary,
              "started_unix": round(started), "seconds": round(time.time() - started, 1),
              "python": platform.python_version()}
    if engine.first_messages is not None:
        first = record_dir / "first-request-messages.json"
        first.write_text(json.dumps(engine.first_messages, indent=1), encoding="utf-8")
        report["first_request_messages_sha256"] = fingerprint(first)
    atomic_json(record_dir / "run.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "run"])
    parser.add_argument("registration", type=Path)
    args = parser.parse_args(argv)
    reg = load(args.registration)
    info = {"id": reg["id"], "registration_sha256": fingerprint(args.registration), "model": reg["model"],
            "episodes": reg["tasks"]["episodes"], "status": reg["status"], **estimate(reg)}
    if args.action == "plan":
        print(json.dumps(info, indent=2))
        return 0
    report = run(args.registration, reg)
    print(json.dumps({k: report[k] for k in ("status", "episodes_finished", "api_calls", "spent_usd", "error")}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
