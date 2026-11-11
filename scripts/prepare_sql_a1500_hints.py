"""Build the column-hints-v1 re-screen registration locally (amendment 9); never loads weights, uploads or launches.

a1500-hints-sql-20261108-01: reload A1500's pinned step-180 adapter and screen it on the same 95 dev tasks
with the new error-feedback tools. A1500's original-tools result (37/95) is the pinned comparator.
"""
from __future__ import annotations

import copy
import json
import statistics
from collections import Counter
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.queryshape import CLASSES, classify, is_complex
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import evaluation_tasks, read_jsonl, verify_runtime_oracles
from ayadb.runledger import atomic_json
from ayadb.sqlenv import SQLTools
from ayadb.sqlfeedback import (MAX_HINT_CHARS, POLICY, bad_column, column_hint, named_tables, normalize_query,
                               schema_columns)

RUN_ID = "a1500-hints-sql-20261108-01"
REG = Path("experiments/sql-a1500-hints/REGISTRATION.json")
A1500_REG = Path("experiments/sql-a1500/REGISTRATION.json")
A1500_SHA = "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
A1500_JOB = "6af0139be6320c66fad700ed"
A1500_OUT = Path("remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01")
A1500_METRICS = A1500_OUT / "dev-a1500-nf4/metrics.jsonl"
A1500_METRICS_SHA = "202aef13a5e7488cc4f41a294545a7879ae4d2f97189853121986caef7b54724"
A1500_TRAJECTORIES = A1500_OUT / "dev-a1500-nf4/trajectories.jsonl"
ADAPTER = A1500_OUT / "training/adapter"
ADAPTER_SHA = "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
TRAIN_1500 = Path("data/sqale/data-build-v3/train-1500.episodes.jsonl")
DEV = Path("data/sqale/data-build-v3/dev.tasks.jsonl")
DEV_SHA = "db613ff79c2b08753d13f5624ea1950d830aa259b8e35a9f1434409001358716"
SEED, PER_HOUR = 42, .40
A1500_DEV_SECONDS = 6385.886
JOB_TIMEOUT, DEV_TIMEOUT, BUDGET = 16200, 14400, 2.0
STEP = "dev-a1500-hints-nf4"


def asset(path: Path, kind: str):
    return {"path": path.as_posix(), "kind": kind, "sha256": fingerprint(path), "bytes": path.stat().st_size}


def database_assets(rows):
    assets = {}
    for row in rows:
        db = Path(row["db_path"])
        if fingerprint(db) != row["db_sha256"]:
            raise ValueError("database differs from its frozen row hash: " + row["db_path"])
        assets[db.as_posix()] = asset(db, "sqale_database")
    return list(assets.values())


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_a1500_hints.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json",
        "experiments/sqale-sft/AMENDMENT-4-run-inputs-and-control.md",
        "experiments/sqale-sft/AMENDMENT-5-table-reads-and-manual-audit.md",
        "experiments/sqale-sft/AMENDMENT-6-selected-logits-g0-02.md",
        "experiments/sqale-sft/AMENDMENT-7-a500-stage.md",
        "experiments/sqale-sft/AMENDMENT-8-a1500-stage.md",
        "experiments/sqale-sft/AMENDMENT-9-error-feedback-tools.md")]
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def is_sql_error(call):
    return call["name"] == "run_sql" and str(call.get("result", "")).startswith("SQL error")


def after_error_analysis(tasks):
    """Original-tools behaviour after a failed run_sql (A1500 dev) and in the 1,500 training demos."""
    next_kind, kinds, with_error, recovered = Counter(), Counter(), 0, 0
    metrics = {r["task_id"]: r for r in read_jsonl(A1500_METRICS)}
    for record in read_jsonl(A1500_TRAJECTORIES):
        log = record["metadata"]["call_log"]
        failed = [i for i, c in enumerate(log) if is_sql_error(c)]
        if failed:
            with_error += 1
            recovered += bool(metrics[record["metadata"]["source_task_id"]]["strict"])
        for i in failed:
            kinds["no such column" if "no such column" in log[i]["result"] else "other"] += 1
            if i + 1 < len(log):
                same = log[i + 1]["name"] == "run_sql" and log[i + 1]["args"] == log[i]["args"]
                next_kind["identical repeat" if same else "different call"] += 1
    demo_next = Counter()
    for row in read_jsonl(TRAIN_1500):
        calls = row["calls"]
        for i, c in enumerate(calls):
            if c["name"] == "run_sql" and "error" in str(c.get("result", "")).lower()[:40] and i + 1 < len(calls):
                same = calls[i + 1]["name"] == "run_sql" and calls[i + 1]["args"] == c["args"]
                demo_next["identical repeat" if same else "different call"] += 1
    return {"dev_episodes_with_failed_run_sql": with_error, "of_which_strict": recovered,
            "failed_run_sql_calls": sum(kinds.values()), "no_such_column_errors": kinds["no such column"],
            "next_call_after_failure": dict(next_kind), "training_demo_failed_queries": sum(demo_next.values()),
            "training_demo_next_call": dict(demo_next)}


def offline_replay(tasks):
    """What column-hints-v1 would have returned for A1500's real failed queries (no model)."""
    stats, owner, lengths, episodes = Counter(), Counter(), [], set()
    for record in read_jsonl(A1500_TRAJECTORIES):
        tid = record["metadata"]["source_task_id"]
        task = tasks[tid]
        names = SQLTools(Path(task["db_path"]), tid, task.get("discovery_order")).names
        cols = schema_columns(Path(task["db_path"]), names)
        seen = set()
        for call in record["metadata"]["call_log"]:
            if not is_sql_error(call):
                continue
            query = call["args"]["query"]
            if normalize_query(query) in seen:
                stats["identical_repeats_refused"] += 1
                continue
            seen.add(normalize_query(query))
            stats["distinct_failed_queries"] += 1
            error = call["result"][len("SQL error: "):]
            hint = column_hint(error, query, cols, names)
            if not hint:
                continue
            stats["hinted"] += 1
            episodes.add(tid)
            lengths.append(len(hint))
            if error.startswith("no such column"):
                name = bad_column(error).lower()
                listed = {c.lower() for t in named_tables(query, names) for c in cols[t]}
                anywhere = {c.lower() for t in names for c in cols[t]}
                owner["in a table already in the query" if name in listed else
                      "only in a table outside the query" if name in anywhere else "invented or misspelled"] += 1
    return {**stats, "episodes_with_a_hint": len(episodes), "hint_chars_median": statistics.median(lengths),
            "hint_chars_max": max(lengths), "hints_truncated": sum(n > MAX_HINT_CHARS for n in lengths),
            "bad_column_name": dict(owner)}


def main():
    receipt = Path("runs/hf-sql/hf-user") / RUN_ID
    if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
        raise ValueError(RUN_ID + " has external activity; never revise or retry")
    if fingerprint(A1500_REG) != A1500_SHA or fingerprint(DEV) != DEV_SHA:
        raise ValueError("A1500 registration or dev tasks changed")
    if fingerprint(A1500_METRICS) != A1500_METRICS_SHA:
        raise ValueError("A1500 dev metrics differ from the reported result")
    if fingerprint(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("pulled A1500 adapter differs from the trained step-180 adapter")
    if json.loads((ADAPTER / "_COMPLETE").read_text(encoding="utf-8"))["global_step"] != 180:
        raise ValueError("A1500 adapter marker is not step 180")
    a1500 = json.loads(A1500_REG.read_text(encoding="utf-8"))
    model_config = a1500["model_config"]
    old_dev = a1500["steps"][1]

    dev_rows = read_jsonl(DEV)
    effective_dev = evaluation_tasks(DEV, purpose="dev")
    tasks = {t["id"]: t for t in effective_dev}
    if len(effective_dev) != 95:
        raise ValueError("expected 95 effective dev tasks")
    shape_counts = Counter(classify(t["reference_sql"]) for t in effective_dev)
    if [shape_counts[c] for c in CLASSES] != [37, 5, 23, 10, 20]:
        raise ValueError("dev query-shape counts differ from amendment 8")
    rows = read_jsonl(A1500_METRICS)
    comparator = {"registration_sha256": A1500_SHA, "job_id": A1500_JOB, "metrics_sha256": A1500_METRICS_SHA,
                  "tool_feedback": None, "strict": sum(r["strict"] for r in rows),
                  "fallback": sum(r["fallback"] for r in rows), "set": sum(r["bird_set"] for r in rows),
                  "complex_strict": sum(r["strict"] for r in rows if is_complex(tasks[r["task_id"]]["reference_sql"]))}
    if (comparator["strict"], comparator["fallback"], comparator["set"], comparator["complex_strict"]) != (37, 37, 40, 1):
        raise ValueError("A1500 comparator differs from its reported result")

    analysis = after_error_analysis(tasks)
    replay = offline_replay(tasks)
    # The numbers quoted in amendment 9 must be exactly what the pinned files give.
    expected_analysis = {"dev_episodes_with_failed_run_sql": 24, "of_which_strict": 0, "failed_run_sql_calls": 104,
                         "no_such_column_errors": 99, "next_call_after_failure": {"identical repeat": 76, "different call": 4},
                         "training_demo_failed_queries": 410, "training_demo_next_call": {"different call": 410}}
    if analysis != expected_analysis:
        raise ValueError("after-error analysis differs from amendment 9: " + json.dumps(analysis))
    if (replay["distinct_failed_queries"], replay["identical_repeats_refused"], replay["hinted"],
            replay["episodes_with_a_hint"], replay["hint_chars_max"], replay["hints_truncated"]) != (27, 77, 26, 23, 518, 0) \
            or replay["bad_column_name"] != {"invented or misspelled": 13, "in a table already in the query": 8,
                                             "only in a table outside the query": 4}:
        raise ValueError("offline replay differs from amendment 9: " + json.dumps(replay))

    reg = {"id": RUN_ID, "stage": "evaluate"}
    for key in ("format", "expected_account", "hardware", "max_price_per_minute_usd", "stall_seconds",
                "checkpoint_sync_seconds", "max_input_upload_bytes", "max_output_upload_bytes", "model_config",
                "scorer_policy", "inference_logits_to_keep", "required_environment", "dev_file", "dev_sha256",
                "dev_effective_questions", "no_bird_or_q2", "no_teacher_or_paid_api",
                "first_month_storage_reserve_usd", "artifact_retention"):
        reg[key] = copy.deepcopy(a1500[key])
    reg.update({
        "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
        "amendment": "experiments/sqale-sft/AMENDMENT-9-error-feedback-tools.md",
        "tool_feedback": POLICY,
        "a1500_registration_sha256": A1500_SHA, "a1500_job_id": A1500_JOB, "a1500_adapter_sha256": ADAPTER_SHA,
        "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                         "uploads, one job and bounded storage",
        "scope_exclusions": ["A1500 original-tools rerun (reused comparator)", "Base/Pilot/A500 with the new tools "
                             "(only in a final-evaluation registration, if adopted)", "any training",
                             "step-limit, prompt or decoding changes", "shop retention (deferred to final evaluation)",
                             "full stage, Q2 sampling, A+/B", "BIRD/final evaluation", "teacher/paid API", "vLLM",
                             "automatic retries or extra jobs", "hardware change", "push/publication"],
        "budget_usd": BUDGET, "timeout_seconds": JOB_TIMEOUT,
    })
    step = {"name": STEP, "action": "evaluate", "tasks": DEV.as_posix(), "purpose": "dev",
            "out": f"runs/sql/{RUN_ID}/{STEP}", "adapter": ADAPTER.as_posix(),
            **{k: old_dev[k] for k in ("samples", "temperature", "seed", "inference_logits_to_keep")},
            "timeout_seconds": DEV_TIMEOUT, "verify_checkpoint_reload": True, "expected_checkpoint_step": 180,
            "tool_feedback": POLICY}
    reg["steps"] = [step]
    tokenizer_folder = Path(model_config["tokenizer"])
    reg["assets"] = sorted([asset(DEV, "dev_tasks"),
                            *[asset(p, "official_tokenizer") for p in sorted(tokenizer_folder.iterdir()) if p.is_file()],
                            *database_assets(dev_rows),
                            *[asset(p, "a1500_step_180_adapter") for p in sorted(ADAPTER.iterdir()) if p.is_file()]],
                           key=lambda a: a["path"])
    reg["source_sha256"] = source_hashes()
    reg["comparators"] = {"dev-a1500-nf4": comparator}
    reg["decision_rule"] = {
        "rule": "adopt column-hints-v1 for the final evaluation only if A1500 with hints reaches at least five "
                "unrounded strict points over A1500 with the original tools on the same 95 tasks; if adopted, every "
                "final-evaluation model uses the same tools",
        "a1500_original_strict": 37, "needs_at_least": 42}
    reg["after_error_analysis"] = analysis
    reg["offline_replay"] = replay
    reg["query_shape_diagnostic"] = {"module": "src/ayadb/queryshape.py", "dev_counts": {c: shape_counts[c] for c in CLASSES},
                                     "complex_dev_task_ids": sorted(t for t in tasks if is_complex(tasks[t]["reference_sql"])),
                                     "role": "descriptive only (amendment 8)"}
    reg["forecast"] = {"dev_screen_hours": round(A1500_DEV_SECONDS / 3600 + .05, 2), "range_hours": [1.5, 2.5],
                       "basis": "A1500's original-tools dev step took 6,386 s on the same t4-small; t4-small $0.40/h. "
                                "Planning estimate, not a cap."}
    reg["forecast"]["usd"] = round(reg["forecast"]["dev_screen_hours"] * PER_HOUR, 2)
    reg["pass_criteria"] = [
        "account, rate, pins and private upload verified; every dev reference passes before model use",
        f"the dev step reloads the pinned A1500 adapter and its tensors exactly equal the step-180 files ({ADAPTER_SHA[:8]}...)",
        "95 dev episodes with one finite projected inference position; AuditKIT agrees with every saved score",
        "column-hints-v1 recorded in the summary and every trajectory, with hint/refusal counts"]
    reg["abort_criteria"] = ["wrong account or higher hardware price", "CUDA failure/OOM/nonfinite logits",
                             "adapter reload mismatch", "sync failure or byte cap", "step/job timeout or stalled output",
                             "missing memory measurement, full-prompt inference projection or AuditKIT disagreement"]
    reg["reporting"] = ("Strict, fallback and set dev accuracy against A1500's original-tools 37/95, paired discordant "
                        "tasks, amendment-8 shapes, recoveries after a failed query, hints and refusals, the "
                        "determinism check, time, cost and memory are reported whatever they are.")

    REG.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REG, reg)
    load_registration(REG)
    oracle = verify_runtime_oracles(reg)
    print(json.dumps({**registration_plan(REG), "local_reference_checks": oracle,
                      "compute_ceiling_usd": round(reg["timeout_seconds"] / 3600 * PER_HOUR, 2),
                      "forecast": reg["forecast"], "comparator": comparator, "after_error_analysis": analysis,
                      "offline_replay": replay}, indent=2))


if __name__ == "__main__":
    main()
