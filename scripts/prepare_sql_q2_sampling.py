"""Build the three Q2 sampling registrations locally (amendment 11); never loads weights, uploads or launches.

A1500 (exact step-180 adapter) attempts every effective Q2 question twice at temperature 0.8. Q2 is split
into contiguous thirds so the three jobs can run in parallel. Q2 is held out of all training, dev and final.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import evaluation_tasks, verify_runtime_oracles
from ayadb.runledger import atomic_json

A1500_REG = Path("experiments/sql-a1500/REGISTRATION.json")
A1500_SHA = "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
ADAPTER = Path("remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/training/adapter")
ADAPTER_SHA = "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
Q2 = Path("data/sqale/data-build-v3/q2.tasks.jsonl")
SLICES = Path("data/sqale/q2-sampling-v2")
# Amendment 12: results depend on SQLite tie-breaking (3.40 vs 3.53); excluded before any model use on Q2.
TIE_DEPENDENT = {"schema_014507_s2_q7_f5426b14a9": "tie among people with the same count",
                 "schema_009445_s1_q4_ac28623f95": "GROUP_CONCAT order"}
PARTS, SAMPLES, TEMPERATURE, SEED = 3, 2, 0.8, 42
ACCOUNT = "hf-user"  # amendment 13
SECONDS_PER_EPISODE, PER_HOUR = 67.0, .40
JOB_TIMEOUT, STEP_TIMEOUT, BUDGET = 45000, 44400, 5.06


def asset(path: Path, kind: str):
    return {"path": path.as_posix(), "kind": kind, "sha256": fingerprint(path), "bytes": path.stat().st_size}


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_q2_sampling.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json",
        *[f"experiments/sqale-sft/{n}" for n in (
            "AMENDMENT-4-run-inputs-and-control.md", "AMENDMENT-5-table-reads-and-manual-audit.md",
            "AMENDMENT-6-selected-logits-g0-02.md", "AMENDMENT-7-a500-stage.md", "AMENDMENT-8-a1500-stage.md",
            "AMENDMENT-9-error-feedback-tools.md", "AMENDMENT-10-recovery-demos.md",
            "AMENDMENT-11-final-and-self-training.md", "AMENDMENT-12-sqlite-version-drift.md",
            "AMENDMENT-13-account-scoped-jobs.md")])]
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def write_slices(tasks):
    size = -(-len(tasks) // PARTS)
    paths = []
    SLICES.mkdir(parents=True, exist_ok=True)
    for k in range(PARTS):
        rows = tasks[k * size:(k + 1) * size]
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        path = SLICES / f"part-{k + 1}.tasks.jsonl"
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ValueError("existing Q2 slice differs; never rewrite: " + path.as_posix())
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="\n") as target:
                target.write(text)
        paths.append((path, rows))
    covered = [r["id"] for _, rows in paths for r in rows]
    if covered != [t["id"] for t in tasks] or len(set(covered)) != len(tasks):
        raise ValueError("slices must cover every effective Q2 question exactly once")
    return paths


def main():
    if fingerprint(A1500_REG) != A1500_SHA or fingerprint(ADAPTER / "adapter_model.safetensors") != ADAPTER_SHA:
        raise ValueError("A1500 registration or adapter changed")
    a1500 = json.loads(A1500_REG.read_text(encoding="utf-8"))
    model_config = a1500["model_config"]
    effective = evaluation_tasks(Q2, purpose="sampling")
    if len(effective) != 579 or not TIE_DEPENDENT.keys() <= {t["id"] for t in effective}:
        raise ValueError("expected 579 effective Q2 questions including the two tie-dependent ones")
    tasks = [t for t in effective if t["id"] not in TIE_DEPENDENT]
    sources = source_hashes()
    tokenizer = [asset(p, "official_tokenizer") for p in sorted(Path(model_config["tokenizer"]).iterdir()) if p.is_file()]
    adapter = [asset(p, "a1500_step_180_adapter") for p in sorted(ADAPTER.iterdir()) if p.is_file()]
    summary = {}
    for k, (path, rows) in enumerate(write_slices(tasks), start=1):
        run_id = f"q2sample-a1500-sql-20261109-p{k}"
        reg_path = Path(f"experiments/sql-q2-sampling-v1/part-{k}/REGISTRATION.json")
        receipt = Path("runs/hf-sql") / ACCOUNT / run_id
        if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
            raise ValueError(run_id + " has external activity; never revise or retry")
        databases = {}
        for row in rows:
            db = Path(row["db_path"])
            if fingerprint(db) != row["db_sha256"]:
                raise ValueError("database differs from its frozen row hash: " + row["db_path"])
            databases[db.as_posix()] = asset(db, "sqale_database")
        reg = {"id": run_id, "stage": "sample"}
        for key in ("format", "expected_account", "hardware", "max_price_per_minute_usd", "stall_seconds",
                    "checkpoint_sync_seconds", "max_input_upload_bytes", "max_output_upload_bytes", "model_config",
                    "scorer_policy", "inference_logits_to_keep", "required_environment", "no_teacher_or_paid_api",
                    "first_month_storage_reserve_usd", "artifact_retention"):
            reg[key] = copy.deepcopy(a1500[key])
        reg["expected_account"] = ACCOUNT
        hours = len(rows) * SAMPLES * SECONDS_PER_EPISODE / 3600
        reg.update({
            "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
            "amendment": "experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
            "amendments": ["experiments/sqale-sft/AMENDMENT-11-final-and-self-training.md",
                           "experiments/sqale-sft/AMENDMENT-12-sqlite-version-drift.md",
                           "experiments/sqale-sft/AMENDMENT-13-account-scoped-jobs.md"],
            "q2_excluded_tie_dependent": TIE_DEPENDENT,
            "a1500_registration_sha256": A1500_SHA, "a1500_adapter_sha256": ADAPTER_SHA,
            "q2_part": k, "q2_parts": PARTS, "q2_questions": len(rows), "q2_total_effective": len(tasks),
            "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                             "uploads, one job and bounded storage",
            "scope_exclusions": ["any training", "dev, final or BIRD tasks", "other models", "error-feedback tools",
                                 "automatic retries or extra jobs", "hardware change", "push/publication"],
            "budget_usd": BUDGET, "timeout_seconds": JOB_TIMEOUT,
            "steps": [{"name": f"q2-sample-p{k}", "action": "evaluate", "tasks": path.as_posix(), "purpose": "sampling",
                       "out": f"runs/sql/{run_id}/q2-sample-p{k}", "adapter": ADAPTER.as_posix(),
                       "samples": SAMPLES, "temperature": TEMPERATURE, "seed": SEED, "inference_logits_to_keep": 1,
                       "timeout_seconds": STEP_TIMEOUT, "verify_checkpoint_reload": True, "expected_checkpoint_step": 180}],
            "assets": sorted([asset(path, "q2_sampling_slice"), *databases.values(), *tokenizer, *adapter],
                             key=lambda a: a["path"]),
            "source_sha256": sources,
            "forecast": {"hours": round(hours + .1, 2), "usd": round((hours + .1) * PER_HOUR, 2),
                         "basis": "A1500 dev 67 s per greedy episode on the same t4-small; sampled episodes may run "
                                  "longer. Planning estimate, not a cap."},
            "pass_criteria": [
                "account, rate, pins and private upload verified; every Q2 reference in this part passes before model use",
                "the step reloads exactly the pinned A1500 step-180 adapter",
                f"{len(rows)} questions x {SAMPLES} samples with one finite projected position; AuditKIT agrees with every score"],
            "abort_criteria": ["wrong account or higher hardware price", "reference/oracle drift before model use",
                               "CUDA failure/OOM/nonfinite logits", "adapter reload mismatch", "sync failure or byte cap",
                               "step/job timeout or stalled output", "AuditKIT disagreement"],
            "reporting": "Sampled strict/fallback per question and sample, eligibility under amendment 11's filter, and "
                         "the G2 and size gates are reported after all three parts finish, whatever they are.",
        })
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(reg_path, reg)
        load_registration(reg_path)
        summary[run_id] = {**{k2: v for k2, v in registration_plan(reg_path).items() if k2 != "steps"},
                           "questions": len(rows), "forecast": reg["forecast"],
                           "local_reference_checks": verify_runtime_oracles(reg)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
