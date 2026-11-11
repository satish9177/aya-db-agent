"""Freeze B's final selection and build its two final registrations locally (amendment 14).

Never loads weights, uploads or launches. B qualified on dev (40/95, rule >= 38/95 from amendment 11).
Same sets and protocol as A1500's final (scripts/prepare_sql_final_v1.py); Base and A1500 are not re-run:
- final-b-bird-sql-20261110-01: B on the 493 BIRD Mini-Dev final questions;
- final-b-sqale-sql-20261110-01: B on SQaLe final (144) and shop retention (dev 40, multi-hop 40).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import evaluation_tasks, read_jsonl, verify_runtime_oracles
from ayadb.runledger import atomic_json
from ayadb.scoringlimits import SCORING_MAX_OPS, SCORING_SECONDS
from prepare_sql_final_v1 import BIRD, RET_DEV, RET_MH, SQALE_FINAL, asset, database_assets

A1500_FINALS = {"bird": Path("experiments/sql-final-v1/bird/REGISTRATION.json"),
                "sqale": Path("experiments/sql-final-v1/sqale-retention/REGISTRATION.json")}
A1500_FINAL_SHAS = {"bird": "2797857acd8e681dd2cb27830b980f3db21bbba88cfc19c6344b4adaba3304a8",
                    "sqale": "fe167f3e030dd9cf07a16b8b3638cd8b502fd37cd0795995852daf495fa550c2"}  # as launched
B_REG = Path("experiments/sql-selftrain-v1/b/REGISTRATION.json")
B_REG_SHA = "154f26224c9c4424949ded201cca00d2f32d629ff58619c3619dffe8882f2a6c"
B_RUN = Path("remote/hf-sql/hf-user/b-selftrain-sql-20261109-01")
ADAPTER = B_RUN / "training/adapter"
ADAPTER_SHA = "90b04823368f50d6ef5ce85c171fffff317b11a3bf43a591cf3952f01b076837"
ADAPTER_STEP = 40
SELECTION = Path("experiments/sql-final-b/SELECTION.json")
AMENDMENT = "experiments/sqale-sft/AMENDMENT-14-final-b.md"
# A1500's measured final step seconds on the same t4-small (experiments/sql-final-v1/RESULTS.json); planning only.
A1500_SECONDS = {"bird": 44365.3, "sqale-final": 10021.9, "retention-dev": 2113.3, "retention-mh": 3812.2}
PER_HOUR = .40
RESERVE = .10
REGS = {
    "bird": ("final-b-bird-sql-20261110-01", Path("experiments/sql-final-b/bird/REGISTRATION.json"),
             [("bird", BIRD)], 54000, "hf-user"),
    "sqale": ("final-b-sqale-sql-20261110-01", Path("experiments/sql-final-b/sqale-retention/REGISTRATION.json"),
              [("sqale-final", SQALE_FINAL), ("retention-dev", RET_DEV), ("retention-mh", RET_MH)], 25200, "hf-user"),
}
CREDIT = {"final-b-bird-sql-20261110-01": 6.69, "final-b-sqale-sql-20261110-01": 3.30}  # per job: remaining credit the user reported on 2026-11-10


def freeze_selection(model_config):
    if SELECTION.exists():
        frozen = json.loads(SELECTION.read_text(encoding="utf-8"))
        if not frozen.get("frozen"):
            raise ValueError("selection record exists but is not frozen")
        return frozen
    dev = read_jsonl(B_RUN / "dev-b-nf4/metrics.jsonl")
    if len(dev) != 95 or sum(r["strict"] for r in dev) != 40:
        raise ValueError("B dev result differs")
    record = {
        "frozen": True, "frozen_on": "2026-11-10", "amendment": AMENDMENT,
        "basis": "dev strict on the same 95 effective dev tasks only (amendment 11 rule: B joins if >= 38/95); "
                 "no final, BIRD or retention result for B was seen",
        "dev_strict": {"base": 2, "a1500": 37, "aplus": 36, "b": 40},
        "models": [{"name": "b", "role": "self-training model B, qualified on dev (40/95); submitted model",
                    "model": model_config["model"], "model_revision": model_config["model_revision"],
                    "adapter_sha256": ADAPTER_SHA, "checkpoint_step": ADAPTER_STEP,
                    "registration_sha256": B_REG_SHA, "job_id": "6af2d8baab34849891aae46b"}],
        "already_evaluated": {"base": "experiments/sql-final-v1 (not re-run)", "a1500": "experiments/sql-final-v1 (not re-run)"},
        "not_selected": {"aplus": "control, never a finalist (amendment 11)"},
        "submitted_model": "b (fixed from dev before any B final result; amendment 14)",
        "tools": "original SQLTools (no tool_feedback)",
    }
    SELECTION.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(SELECTION, record)
    return record


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_final_v1.py", "scripts/prepare_sql_final_b.py",
        "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json", "experiments/bird-final/SOURCES.json",
        *[f"experiments/sqale-sft/{n}" for n in (
            "AMENDMENT-3-bird-final.md", "AMENDMENT-4-run-inputs-and-control.md",
            "AMENDMENT-5-table-reads-and-manual-audit.md", "AMENDMENT-6-selected-logits-g0-02.md",
            "AMENDMENT-7-a500-stage.md", "AMENDMENT-8-a1500-stage.md", "AMENDMENT-9-error-feedback-tools.md",
            "AMENDMENT-10-recovery-demos.md", "AMENDMENT-11-final-and-self-training.md",
            "AMENDMENT-12-sqlite-version-drift.md", "AMENDMENT-13-account-scoped-jobs.md",
            "AMENDMENT-14-final-b.md")])]
    missing = [p.as_posix() for p in files if not p.exists()]
    if missing:
        raise ValueError("missing pinned source: " + ", ".join(missing))
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def main():
    if fingerprint(B_REG) != B_REG_SHA or fingerprint(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("B registration or adapter changed")
    if json.loads((ADAPTER / "_COMPLETE").read_text(encoding="utf-8"))["global_step"] != ADAPTER_STEP:
        raise ValueError("B adapter is not the step-40 final adapter")
    if json.loads((B_RUN / "training/train-report.json").read_text(encoding="utf-8"))["adapter_sha256"] != ADAPTER_SHA:
        raise ValueError("B adapter differs from the one its job trained")
    if any(fingerprint(p) != A1500_FINAL_SHAS[k] for k, p in A1500_FINALS.items()):
        raise ValueError("an A1500 final registration changed since launch")
    templates = {k: load_registration(p, check_files=False) for k, p in A1500_FINALS.items()}
    model_config = templates["bird"]["model_config"]
    if templates["sqale"]["model_config"] != model_config:
        raise ValueError("A1500 final registrations differ in model config")
    selection = freeze_selection(model_config)
    selection_sha = fingerprint(SELECTION)
    sources = source_hashes()
    tokenizer = [asset(p, "official_tokenizer") for p in sorted(Path(model_config["tokenizer"]).iterdir()) if p.is_file()]
    adapter = [asset(p, "b_step_40_adapter") for p in sorted(ADAPTER.iterdir()) if p.is_file()]
    summary = {"selection_sha256": selection_sha}
    for key, (run_id, path, sets, job_timeout, account) in REGS.items():
        receipt = Path("runs/hf-sql") / account / run_id
        if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
            launched = json.loads((receipt / "upload.json").read_text(encoding="utf-8"))
            if fingerprint(path) != launched["registration_sha256"]:
                raise ValueError(run_id + " was launched from a different registration file")
            summary[key] = {"id": run_id, "already_launched": True, "registration_sha256": fingerprint(path)}
            continue  # never revise or retry a launched registration
        template = templates[key]
        steps, assets, counts, seconds_total = [], [asset(SELECTION, "frozen_final_selection")], {}, 0.0
        for name, tasks_path in sets:
            effective = evaluation_tasks(tasks_path, purpose="final", selection_sha256=selection_sha)
            counts[name] = len(effective)
            kind = "bird_final_database" if tasks_path == BIRD else "sqale_or_shop_database"
            assets += [asset(tasks_path, "final_tasks" if name != "bird" else "bird_final_tasks"),
                       *database_assets(read_jsonl(tasks_path), kind)]
            a1500_step = next(s for s in template["steps"] if s["name"] == f"{name}-a1500")
            seconds_total += A1500_SECONDS[name]
            step = {"name": f"{name}-b", "action": "evaluate", "tasks": tasks_path.as_posix(),
                    "purpose": "final", "selection_file": SELECTION.as_posix(), "selection_sha256": selection_sha,
                    "out": f"runs/sql/{run_id}/{name}-b", "adapter": ADAPTER.as_posix(),
                    "timeout_seconds": int(min(job_timeout, max(3600, A1500_SECONDS[name] * 2))),
                    "verify_checkpoint_reload": True, "expected_checkpoint_step": ADAPTER_STEP}
            for k in ("samples", "temperature", "seed", "inference_logits_to_keep"):
                step[k] = a1500_step[k]
            steps.append(step)
        assets = sorted({a["path"]: a for a in assets + tokenizer + adapter}.values(), key=lambda a: a["path"])
        reg = {"id": run_id, "stage": "final"}
        for k in ("format", "hardware", "max_price_per_minute_usd", "stall_seconds", "checkpoint_sync_seconds",
                  "max_output_upload_bytes", "model_config", "scorer_policy", "inference_logits_to_keep",
                  "required_environment", "no_teacher_or_paid_api", "artifact_retention", "job_sqlite_version"):
            reg[k] = copy.deepcopy(template[k])
        reg["expected_account"] = account
        total_bytes = sum(a["bytes"] for a in assets)
        budget = round(job_timeout / 3600 * PER_HOUR + RESERVE + .01, 2)
        if budget > CREDIT[run_id]:
            raise ValueError(f"{run_id} budget ${budget} exceeds its reported credit")
        hours = seconds_total / 3600 + .1
        reg.update({
            "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
            "amendment": AMENDMENT,
            "amendments": ["experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
                           "experiments/sqale-sft/AMENDMENT-12-sqlite-version-drift.md",
                           "experiments/sqale-sft/AMENDMENT-13-account-scoped-jobs.md", AMENDMENT],
            "supersedes": None,
            "scoring_limits": {"seconds": SCORING_SECONDS, "max_ops": SCORING_MAX_OPS},
            "selection_sha256": selection_sha, "finalists": [m["name"] for m in selection["models"]],
            "protocol_template": {"path": A1500_FINALS[key].as_posix(), "sha256": fingerprint(A1500_FINALS[key])},
            "max_input_upload_bytes": max(536870912, ((total_bytes // 2 ** 28) + 1) * 2 ** 28),
            "first_month_storage_reserve_usd": RESERVE,
            "timeout_seconds": job_timeout,
            "budget_usd": budget,
            "account_credit_reported_usd": CREDIT[run_id],
            "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                             "uploads, one job and bounded storage",
            "scope_exclusions": ["any training", "Base, A1500, A+ or other models", "error-feedback tools",
                                 "the spent shop test (data/test.jsonl)", "dev or Q2 tasks", "automatic retries or extra jobs",
                                 "hardware change", "push/publication"],
            "steps": steps, "assets": assets, "source_sha256": sources,
            "task_counts": counts,
            "forecast": {"hours": round(hours, 2), "usd": round(hours * PER_HOUR, 2),
                         "basis": "A1500's measured final seconds on the same sets and t4-small; B's dev screen was 5% "
                                  "faster than A1500's; planning only"},
            "pass_criteria": [
                "account, rate, pins and private upload verified; every final reference passes before model use",
                "the frozen selection record matches the evaluated model and adapter",
                "B steps reload exactly the pinned step-40 adapter",
                "every task evaluated once with one finite projected position; AuditKIT agrees with every score"],
            "abort_criteria": ["wrong account or higher hardware price", "reference/oracle drift before model use",
                               "CUDA failure/OOM/nonfinite logits", "selection or adapter mismatch", "sync failure or byte cap",
                               "step/job timeout or stalled output", "AuditKIT disagreement"],
            "reporting": "Strict (primary), set and fallback for every set, next to Base and A1500 on the same tasks, "
                         "with B vs A1500 task by task, are reported whatever they are; no selection, tuning or rerun "
                         "on final data. Submitted model fixed in advance: B (amendment 14).",
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
