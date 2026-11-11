"""Freeze the final-evaluation selection and build its two registrations locally (amendment 11).

Never loads weights, uploads or launches. Finalists are frozen from dev results only: untuned Base (the
"before" model) and A1500 (the best trained model on dev). Two jobs so the 1.5 GB of BIRD databases are
uploaded once:
- final-bird-sql-20261109-01: Base, then A1500, on the 493 BIRD Mini-Dev final questions;
- final-sqale-sql-20261109-01: Base and A1500 on the 144 SQaLe final questions and the shop retention
  sets (dev 40, multi-hop 40).
"""
from __future__ import annotations

import copy
import json
import time
from collections import Counter
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import evaluation_tasks, read_jsonl, verify_runtime_oracles
from ayadb.runledger import atomic_json
from ayadb.scoringlimits import SCORING_MAX_OPS, SCORING_SECONDS
from ayadb.sqlscore import execute

A1500_REG = Path("experiments/sql-a1500/REGISTRATION.json")
A1500_SHA = "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
ADAPTER = Path("remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/training/adapter")
ADAPTER_SHA = "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
SELECTION = Path("experiments/sql-final-v1/SELECTION.json")
BIRD = Path("data/bird/final-v1/tasks.jsonl")
SQALE_FINAL = Path("data/sqale/data-build-v3/final.tasks.jsonl")
RET_DEV = Path("data/sqale/data-build-v3/retention/dev.tasks.jsonl")
RET_MH = Path("data/sqale/data-build-v3/retention/mh.tasks.jsonl")
DEV_RESULTS = {  # dev strict (95 tasks) used for the frozen selection; nothing from final data
    "base": ("dev-baselines-sql-20261107-01", "bc78d3ffab86c90314fac5b4b3182297ef80f2d430cf77bb4b5ce7062054636c", 2),
    "pilot": ("dev-baselines-sql-20261107-01", "501f18f6d449bc8d36d06aa4cb3693fccb7654920ce1021c120e7d5f5f252f7f", 20),
    "a500": ("a500-sql-agenttune-20261107-01", "1b574d5919545db0f29563ff4028cceac8aaa075f81525bbcac362bb2f6f4a46", 28),
    "a1500": ("a1500-sql-agenttune-20261108-01", "202aef13a5e7488cc4f41a294545a7879ae4d2f97189853121986caef7b54724", 37),
    "a1500-hints": ("a1500-hints-sql-20261108-01", "89cb257e7d3e344d344a61e465af36214c647027947ab0a03e950bd98661b749", 38),
    "a1500v4": ("a1500v4-sql-agenttune-20261109-01", None, 35),
}
# Seconds per dev episode measured on the same t4-small (Base 16.5, A1500 67.0); planning only.
SECONDS = {"base": 16.5, "a1500": 67.0}
PER_HOUR = .40
REGS = {
    "bird": ("final-bird-sql-20261109-02", Path("experiments/sql-final-v1/bird/REGISTRATION.json"),
             [("bird", BIRD)], 57600, 2.0, "hf-user"),
    "sqale": ("final-sqale-sql-20261109-01", Path("experiments/sql-final-v1/sqale-retention/REGISTRATION.json"),
              [("sqale-final", SQALE_FINAL), ("retention-dev", RET_DEV), ("retention-mh", RET_MH)], 28800, 2.0,
              "hf-user"),  # amendment 13
}


def asset(path: Path, kind: str):
    return {"path": path.as_posix(), "kind": kind, "sha256": fingerprint(path), "bytes": path.stat().st_size}


def database_assets(rows, kind):
    out = {}
    for row in rows:
        db = Path(row["db_path"])
        if fingerprint(db) != row["db_sha256"]:
            raise ValueError("database differs from its frozen row hash: " + row["db_path"])
        out[db.as_posix()] = asset(db, kind)
    return list(out.values())


def freeze_selection(model_config):
    if SELECTION.exists():
        frozen = json.loads(SELECTION.read_text(encoding="utf-8"))
        if not frozen.get("frozen"):
            raise ValueError("selection record exists but is not frozen")
        return frozen
    a1500 = read_jsonl(Path("remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/dev-a1500-nf4/metrics.jsonl"))
    if sum(r["strict"] for r in a1500) != 37:
        raise ValueError("A1500 dev result differs")
    record = {
        "frozen": True, "frozen_on": "2026-11-09", "amendment": "experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
        "basis": "dev strict on the same 95 effective dev tasks only; no final, BIRD or retention result was seen",
        "dev_strict": {k: v[2] for k, v in DEV_RESULTS.items()},
        "models": [
            {"name": "base", "role": "before (untuned)", "model": model_config["model"],
             "model_revision": model_config["model_revision"], "adapter_sha256": None},
            {"name": "a1500", "role": "best trained model on dev (37/95)", "model": model_config["model"],
             "model_revision": model_config["model_revision"], "adapter_sha256": ADAPTER_SHA,
             "registration_sha256": A1500_SHA, "job_id": "6af0139be6320c66fad700ed"}],
        "not_selected": {"pilot": "shop-only model; optional, not run (cost/time)", "a500": "below A1500 on dev",
                         "a1500v4": "35/95, below A1500 (amendment 10)",
                         "column-hints-v1 tools": "not adopted (amendment 9); finals use the original tools"},
        "conditional_addition": "self-training model B (amendment 11) is added by a separate final registration only "
                                "if its dev strict is at least 38/95; decided on dev only",
        "tools": "original SQLTools (no tool_feedback)",
    }
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(SELECTION, record)
    return record


def bird_timing():
    """Local execution time of every BIRD reference query (the job re-runs them before any model use)."""
    times = []
    for row in read_jsonl(BIRD):
        t0 = time.monotonic()
        result = execute(Path(row["db_path"]), row["reference_sql"], seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
        times.append(time.monotonic() - t0)
        if result.error:
            raise ValueError("BIRD reference failed locally: " + row["id"] + " " + result.error)
    times.sort()
    return {"queries": len(times), "max_seconds": round(times[-1], 2), "p99_seconds": round(times[int(.99 * len(times))], 2),
            "over_2_seconds": sum(t > 2 for t in times), "over_3_seconds": sum(t > 3 for t in times), "over_5_seconds": sum(t > 5 for t in times),
            "scoring_cap_seconds": SCORING_SECONDS, "agent_tool_cap_seconds": 5.0}


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_final_v1.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json", "experiments/bird-final/SOURCES.json",
        *[f"experiments/sqale-sft/{n}" for n in (
            "AMENDMENT-3-bird-final.md", "AMENDMENT-4-run-inputs-and-control.md",
            "AMENDMENT-5-table-reads-and-manual-audit.md", "AMENDMENT-6-selected-logits-g0-02.md",
            "AMENDMENT-7-a500-stage.md", "AMENDMENT-8-a1500-stage.md", "AMENDMENT-9-error-feedback-tools.md",
            "AMENDMENT-10-recovery-demos.md", "AMENDMENT-11-final-and-self-training.md",
            "AMENDMENT-12-sqlite-version-drift.md", "AMENDMENT-13-account-scoped-jobs.md")])]
    missing = [p.as_posix() for p in files if not p.exists()]
    if missing:
        raise ValueError("missing pinned source: " + ", ".join(missing))
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def main():
    if fingerprint(A1500_REG) != A1500_SHA or fingerprint(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("A1500 registration or adapter changed")
    a1500 = json.loads(A1500_REG.read_text(encoding="utf-8"))
    model_config = a1500["model_config"]
    selection = freeze_selection(model_config)
    selection_sha = fingerprint(SELECTION)
    sources = source_hashes()
    tokenizer = [asset(p, "official_tokenizer") for p in sorted(Path(model_config["tokenizer"]).iterdir()) if p.is_file()]
    adapter = [asset(p, "a1500_step_180_adapter") for p in sorted(ADAPTER.iterdir()) if p.is_file()]
    summary = {"selection_sha256": selection_sha, "bird_reference_timing": bird_timing()}
    for key, (run_id, path, sets, job_timeout, reserve, account) in REGS.items():
        receipt = Path("runs/hf-sql") / account / run_id
        if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
            launched = json.loads((receipt / "upload.json").read_text(encoding="utf-8"))
            if fingerprint(path) != launched["registration_sha256"]:
                raise ValueError(run_id + " was launched from a different registration file")
            summary[key] = {"id": run_id, "already_launched": True, "registration_sha256": fingerprint(path)}
            continue  # never revise or retry a launched registration
        steps, assets, counts, hours = [], [asset(SELECTION, "frozen_final_selection")], {}, 0.0
        for name, tasks_path in sets:
            effective = evaluation_tasks(tasks_path, purpose="final", selection_sha256=selection_sha)
            counts[name] = len(effective)
            kind = "bird_final_database" if tasks_path == BIRD else "sqale_or_shop_database"
            assets += [asset(tasks_path, "final_tasks" if name != "bird" else "bird_final_tasks"),
                       *database_assets(read_jsonl(tasks_path), kind)]
            for model in ("base", "a1500"):
                seconds = SECONDS[model] * len(effective)
                hours += seconds / 3600
                step = {"name": f"{name}-{model}", "action": "evaluate", "tasks": tasks_path.as_posix(),
                        "purpose": "final", "selection_file": SELECTION.as_posix(), "selection_sha256": selection_sha,
                        "out": f"runs/sql/{run_id}/{name}-{model}", "adapter": ADAPTER.as_posix() if model == "a1500" else None,
                        "samples": 1, "temperature": 0.0, "seed": 42, "inference_logits_to_keep": 1,
                        "timeout_seconds": int(min(job_timeout, max(3600, seconds * 2)))}
                if model == "a1500":
                    step.update(verify_checkpoint_reload=True, expected_checkpoint_step=180)
                steps.append(step)
        assets = sorted({a["path"]: a for a in assets + tokenizer + adapter}.values(), key=lambda a: a["path"])
        reg = {"id": run_id, "stage": "final"}
        for k in ("format", "expected_account", "hardware", "max_price_per_minute_usd", "stall_seconds",
                  "checkpoint_sync_seconds", "max_output_upload_bytes", "model_config", "scorer_policy",
                  "inference_logits_to_keep", "required_environment", "no_teacher_or_paid_api", "artifact_retention"):
            reg[k] = copy.deepcopy(a1500[k])
        reg["expected_account"] = account
        total_bytes = sum(a["bytes"] for a in assets)
        reg.update({
            "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
            "amendment": "experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
            "amendments": ["experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
                           "experiments/sqale-sft/AMENDMENT-12-sqlite-version-drift.md",
                           "experiments/sqale-sft/AMENDMENT-13-account-scoped-jobs.md"],
            "supersedes": {"bird": "final-bird-sql-20261109-01 (job 6af1536ce6320c66fad7d6ea stopped at the reference check: SQLite 3.40.1 operation counts; amendment 12)"}.get(key),
            "job_sqlite_version": "3.40.1", "scoring_limits": {"seconds": SCORING_SECONDS, "max_ops": SCORING_MAX_OPS},
            "selection_sha256": selection_sha, "finalists": [m["name"] for m in selection["models"]],
            "max_input_upload_bytes": max(536870912, ((total_bytes // 2 ** 28) + 1) * 2 ** 28),
            "first_month_storage_reserve_usd": reserve / 20,
            "timeout_seconds": job_timeout,
            "budget_usd": round(job_timeout / 3600 * PER_HOUR + reserve / 20 + .01, 2),
            "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                             "uploads, one job and bounded storage",
            "scope_exclusions": ["any training", "Pilot or other models", "error-feedback tools",
                                 "the spent shop test (data/test.jsonl)", "dev or Q2 tasks", "automatic retries or extra jobs",
                                 "hardware change", "push/publication"],
            "steps": steps, "assets": assets, "source_sha256": sources,
            "task_counts": counts,
            "forecast": {"hours": round(hours + .1, 2), "usd": round((hours + .1) * PER_HOUR, 2),
                         "basis": "dev seconds per episode on the same t4-small (Base 16.5 s, A1500 67.0 s); planning only"},
            "pass_criteria": [
                "account, rate, pins and private upload verified; every final reference passes before model use",
                "the frozen selection record matches each evaluated model and adapter",
                "A1500 steps reload exactly the pinned step-180 adapter",
                "every task evaluated once per model with one finite projected position; AuditKIT agrees with every score"],
            "abort_criteria": ["wrong account or higher hardware price", "reference/oracle drift before model use",
                               "CUDA failure/OOM/nonfinite logits", "selection or adapter mismatch", "sync failure or byte cap",
                               "step/job timeout or stalled output", "AuditKIT disagreement"],
            "reporting": "Strict (primary), set (BIRD's usual metric) and fallback for every set and model, with the "
                         "Base -> A1500 change, are reported whatever they are; no selection, tuning or rerun on final data.",
        })
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, reg)
        load_registration(path)
        summary[key] = {**registration_plan(path), "task_counts": counts, "forecast": reg["forecast"],
                        "asset_bytes": total_bytes, "budget_usd": reg["budget_usd"],
                        "local_reference_checks": verify_runtime_oracles(reg)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
