"""Build the B and A+ continuation registrations (amendment 11) after build_q2_selftrain.py; no model, upload or job.

Both start from the identical A1500 step-180 adapter, use the same settings and differ only in data:
- B: A1500's own verified Q2 attempts (role continuation_self);
- A+: gold-path control on the same questions, matched on trained target tokens (role continuation_control).
Each is then screened on the 95 dev tasks with the original tools after an exact reload.
B joins the final evaluation (its own registration) only if its dev strict is at least 38/95.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import read_jsonl, training_plan, verify_runtime_oracles
from ayadb.runledger import atomic_json

ACCOUNT = "hf-user"  # amendment 13
A1500_REG = Path("experiments/sql-a1500/REGISTRATION.json")
A1500_SHA = "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
ADAPTER = Path("remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/training/adapter")
ADAPTER_SHA = "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
DATA = Path("data/sqale/selftrain-v1")
DEV = Path("data/sqale/data-build-v3/dev.tasks.jsonl")
TOKENS_PER_SECOND, DEV_HOURS, PER_HOUR = 368.4, 1.8, .40
ARMS = {"b": ("b-selftrain-sql-20261109-01", "b.episodes.jsonl", "continuation_self"),
        "aplus": ("aplus-control-sql-20261109-01", "aplus.episodes.jsonl", "continuation_control")}


def asset(path: Path, kind: str):
    return {"path": path.as_posix(), "kind": kind, "sha256": fingerprint(path), "bytes": path.stat().st_size}


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/build_q2_selftrain.py", "scripts/prepare_sql_selftrain.py",
        "requirements-sql-runner.txt", "pyproject.toml", "experiments/sqale-sft/RUN-POLICY-v1.json",
        *[f"experiments/sqale-sft/{n}" for n in (
            "AMENDMENT-4-run-inputs-and-control.md", "AMENDMENT-5-table-reads-and-manual-audit.md",
            "AMENDMENT-6-selected-logits-g0-02.md", "AMENDMENT-7-a500-stage.md", "AMENDMENT-8-a1500-stage.md",
            "AMENDMENT-9-error-feedback-tools.md", "AMENDMENT-10-recovery-demos.md",
            "AMENDMENT-11-final-and-self-training.md", "AMENDMENT-12-sqlite-version-drift.md",
            "AMENDMENT-13-account-scoped-jobs.md")])]
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def main():
    matching_path = DATA / "matching.json"
    matching = json.loads(matching_path.read_text(encoding="utf-8"))
    if not matching.get("continuations_may_train"):
        raise ValueError("G2 or the size gate failed; amendment 11 forbids training either continuation")
    if fingerprint(A1500_REG) != A1500_SHA or fingerprint(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("A1500 registration or adapter changed")
    a1500 = json.loads(A1500_REG.read_text(encoding="utf-8"))
    settings = matching["training_settings"]
    sources = source_hashes()
    common_assets = [asset(matching_path, "continuation_manifest"), asset(DEV, "dev_tasks"),
                     *[asset(p, "official_tokenizer") for p in sorted(Path(a1500["model_config"]["tokenizer"]).iterdir()) if p.is_file()],
                     *[asset(p, "a1500_step_180_adapter") for p in sorted(ADAPTER.iterdir()) if p.is_file()]]
    summary = {}
    for arm, (run_id, filename, role) in ARMS.items():
        receipt = Path("runs/hf-sql") / ACCOUNT / run_id
        if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
            raise ValueError(run_id + " has external activity; never revise or retry")
        episodes = DATA / filename
        if fingerprint(episodes) != matching["datasets"][role]["sha256"]:
            raise ValueError("continuation data differs from the frozen match: " + filename)
        train, validation, plan = training_plan(episodes, seed=settings["seed"], val_share=0.0)
        if validation or plan["train_targets_per_pass"] != matching["matched_targets"]:
            raise ValueError("continuation plan differs from the frozen match")
        updates = settings["epochs"] * math.ceil(len(train) / settings["gradient_accumulation"])
        train_hours = settings["epochs"] * plan["train_input_tokens_per_pass"] / TOKENS_PER_SECOND / 3600
        job_hours = math.ceil((train_hours + DEV_HOURS) * 1.6 + .5)
        databases = {}
        for row in read_jsonl(episodes) + read_jsonl(DEV):
            db = Path(row["db_path"])
            if fingerprint(db) != row["db_sha256"]:
                raise ValueError("database differs from its frozen row hash: " + row["db_path"])
            databases[db.as_posix()] = asset(db, "sqale_database")
        reg = {"id": run_id, "stage": "train"}
        for key in ("format", "hardware", "max_price_per_minute_usd", "stall_seconds", "checkpoint_sync_seconds",
                    "max_input_upload_bytes", "max_output_upload_bytes", "model_config", "scorer_policy",
                    "inference_logits_to_keep", "required_environment", "dev_file", "dev_sha256",
                    "dev_effective_questions", "no_bird_or_q2", "no_teacher_or_paid_api",
                    "first_month_storage_reserve_usd", "artifact_retention"):
            reg[key] = copy.deepcopy(a1500[key])
        reg["no_bird_or_q2"] = False  # trains on verified Q2-derived episodes by design (amendment 11)
        out = f"runs/sql/{run_id}"
        reg.update({
            "expected_account": ACCOUNT, "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
            "amendments": ["experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
                           "experiments/sqale-sft/AMENDMENT-12-sqlite-version-drift.md",
                           "experiments/sqale-sft/AMENDMENT-13-account-scoped-jobs.md"],
            "arm": arm, "a1500_registration_sha256": A1500_SHA, "a1500_adapter_sha256": ADAPTER_SHA,
            "continuation_manifest_sha256": fingerprint(matching_path),
            "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                             "uploads, one job and bounded storage",
            "scope_exclusions": ["final/BIRD evaluation (B only, by its own registration, if dev >= 38/95)",
                                 "other data or teacher", "GRPO", "automatic retries or extra jobs", "push/publication"],
            "timeout_seconds": job_hours * 3600,
            "budget_usd": round(job_hours * PER_HOUR + reg["first_month_storage_reserve_usd"] + .01, 2),
            "steps": [
                {"name": f"train-{arm}", "action": "train", "episodes": episodes.as_posix(), "out": f"{out}/training",
                 "role": role, "base_adapter": ADAPTER.as_posix(), "continuation_manifest": matching_path.as_posix(),
                 "seed": settings["seed"], "validation_share": 0.0, "epochs": settings["epochs"], "max_steps": -1,
                 "batch_size": settings["batch_size"], "gradient_accumulation": settings["gradient_accumulation"],
                 "learning_rate": settings["learning_rate"], "save_steps": 10, "keep_checkpoints": 2,
                 "loss_policy": settings["loss_policy"],
                 "timeout_seconds": int(max(3600, train_hours * 3600 * 2))},
                {"name": f"dev-{arm}-nf4", "action": "evaluate", "tasks": DEV.as_posix(), "purpose": "dev",
                 "out": f"{out}/dev-{arm}-nf4", "adapter": f"{out}/training/adapter", "samples": 1, "temperature": 0.0,
                 "seed": 42, "inference_logits_to_keep": 1, "timeout_seconds": 12600,
                 "verify_checkpoint_reload": True, "expected_checkpoint_step": updates}],
            "assets": sorted({a["path"]: a for a in [asset(episodes, f"{role}_episodes"), *common_assets,
                                                     *databases.values()]}.values(), key=lambda a: a["path"]),
            "source_sha256": sources,
            "training_plan": {k: plan[k] for k in ("source_sha256", "train_examples", "train_input_tokens_per_pass",
                                                   "train_targets_per_pass", "a_unique_train_targets_per_pass")},
            "expected_optimizer_updates": updates,
            "forecast": {"hours": round(train_hours + DEV_HOURS, 2), "usd": round((train_hours + DEV_HOURS) * PER_HOUR, 2),
                         "basis": "A1500's 368.4 input tokens/s and ~1.8 h dev screen on the same t4-small"},
            "decision_rule": "B joins the final evaluation by its own registration only if its dev strict is at least "
                             "38/95; A+ is a control and never a finalist; B vs A+ is reported paired and descriptive",
            "pass_criteria": ["account, rate, pins and private upload verified; every training and dev reference passes",
                              "continuation starts from the pinned A1500 adapter with the frozen matched data and settings",
                              f"{updates} finite updates, none skipped; exact reload of the final adapter",
                              "95 dev episodes with one finite projected position; AuditKIT agrees with every score"],
            "abort_criteria": copy.deepcopy(a1500["abort_criteria"]),
        })
        path = Path(f"experiments/sql-selftrain-v1/{arm}/REGISTRATION.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, reg)
        load_registration(path)
        summary[run_id] = {**{k: v for k, v in registration_plan(path).items() if k != "steps"},
                           "updates": updates, "forecast": reg["forecast"], "local_reference_checks": verify_runtime_oracles(reg)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
