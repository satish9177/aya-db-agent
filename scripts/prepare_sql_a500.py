"""Build the two A500-stage registrations locally (amendment 7); never loads weights, uploads or launches.

1. a500-sql-agenttune-20261107-01: train A500, then screen its final adapter on the 95 dev tasks.
2. dev-baselines-sql-20261107-01: screen untuned Base and the existing Pilot on the same 95 tasks.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import evaluation_tasks, read_jsonl, training_plan, verify_runtime_oracles
from ayadb.runledger import atomic_json
from ayadb.sqlsftdata import encode_episode

A500_ID = "a500-sql-agenttune-20261107-01"
BASELINES_ID = "dev-baselines-sql-20261107-01"
A500_REG = Path("experiments/sql-a500/REGISTRATION.json")
BASELINES_REG = Path("experiments/sql-dev-baselines/REGISTRATION.json")
G0_02 = Path("experiments/sql-g0-02/REGISTRATION.json")
G0_02_SHA = "e86597148c4ccdeb645777fb85f4090ec0eb1156789172165b425242d70f2580"
TRAIN = Path("data/sqale/data-build-v3/train-500.episodes.jsonl")
TRAIN_SHA = "b34bbd824142b955f02375deb26bf1d09e75107c425feb16180e2b6d2818fe3c"
DEV = Path("data/sqale/data-build-v3/dev.tasks.jsonl")
DEV_SHA = "db613ff79c2b08753d13f5624ea1950d830aa259b8e35a9f1434409001358716"
PILOT = Path("remote/hf/pilot/sft-pilot/adapter")
PILOT_SHA = "6f2827ae9b288fb96de32ac3f0f8e258743acd61ea754739f3c67319c21e4794"
FROZEN_PLAN = Path("data/sqale/runner-inputs-v1/a-500.json")
DRAFT = Path("experiments/sqale-500/REGISTRATION-DRAFT.json")
SEED, VAL_SHARE, EPOCHS, ACCUMULATION = 42, .05, 2, 16
PER_HOUR = .40


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


def tokenizer_assets(model_config):
    folder = Path(model_config["tokenizer"])
    return [asset(p, "official_tokenizer") for p in sorted(folder.iterdir()) if p.is_file()]


def check_training_inputs(model_config):
    if fingerprint(TRAIN) != TRAIN_SHA or fingerprint(DEV) != DEV_SHA:
        raise ValueError("frozen 500/dev inputs changed")
    train, validation, plan = training_plan(TRAIN, seed=SEED, val_share=VAL_SHARE)
    frozen = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    for key in ("source_sha256", "train_examples", "validation_examples", "train_episode_ids",
                "validation_episode_ids", "train_input_tokens_per_pass", "train_targets_per_pass",
                "a_unique_train_targets_per_pass", "minimum_unique_verified_b_targets_per_pass"):
        if plan[key] != frozen[key]:
            raise ValueError("training plan differs from the frozen runner input: " + key)
    tokenizer = official_tokenizer(model_config["tokenizer"])
    for row in train + validation:
        encoded = encode_episode(row, tokenizer, model_config["max_input_tokens"])
        if encoded is None or len(encoded["input_ids"]) != row["n_input_tokens"] or \
                sum(v != -100 for v in encoded["labels"]) != row["n_target_tokens"]:
            raise ValueError("fresh encoding differs from frozen counts: " + row["id"])
    updates_per_pass = len(train) // ACCUMULATION + int(len(train) % ACCUMULATION > 0)
    return train, validation, plan, updates_per_pass * EPOCHS


def common_fields(g0):
    return {
        "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL", "format": "sql-submit-v1",
        "expected_account": "hf-user", "hardware": "t4-small", "max_price_per_minute_usd": 0.006667,
        "stall_seconds": 1800, "checkpoint_sync_seconds": 60,
        "max_input_upload_bytes": 536870912, "max_output_upload_bytes": 1073741824,
        "model_config": copy.deepcopy(g0["model_config"]),
        "scorer_policy": "table-read-v1", "inference_logits_to_keep": 1,
        "required_environment": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        "amendment": "experiments/sqale-sft/AMENDMENT-7-a500-stage.md",
        "g0_02_registration_sha256": G0_02_SHA, "g0_02_job_id": "6aef547de6320c66fad5ff75",
        "dev_file": DEV.as_posix(), "dev_sha256": DEV_SHA, "dev_effective_questions": 95,
        "no_bird_or_q2": True, "no_teacher_or_paid_api": True,
        "first_month_storage_reserve_usd": 0.04,
        "artifact_retention": "Pull and review promptly; keep this run's private input/output prefixes at most 30 days "
                              "without renewed storage approval.",
        "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                         "uploads, one job and bounded storage",
        "scope_exclusions": ["shop retention (deferred to final evaluation, amendment 7)",
                             "one-example Base prompt (not implemented, amendment 7)",
                             "1,500/full stages, Q2 sampling, A+/B", "BIRD/final evaluation", "teacher/paid API",
                             "vLLM", "automatic retries or extra jobs", "hardware change", "push/publication"],
    }


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_a500.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json",
        "experiments/sqale-sft/AMENDMENT-4-run-inputs-and-control.md",
        "experiments/sqale-sft/AMENDMENT-5-table-reads-and-manual-audit.md",
        "experiments/sqale-sft/AMENDMENT-6-selected-logits-g0-02.md",
        "experiments/sqale-sft/AMENDMENT-7-a500-stage.md")]
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def dev_step(name, out_id, adapter, timeout, **extra):
    return {"name": name, "action": "evaluate", "tasks": DEV.as_posix(), "purpose": "dev",
            "out": f"runs/sql/{out_id}/{name}", "adapter": adapter, "samples": 1, "temperature": 0.0,
            "seed": SEED, "inference_logits_to_keep": 1, "timeout_seconds": timeout, **extra}


def main():
    for run_id in (A500_ID, BASELINES_ID):
        receipt = Path("runs/hf-sql/hf-user") / run_id
        if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
            raise ValueError(run_id + " has external activity; never revise or retry")
    if fingerprint(G0_02) != G0_02_SHA or fingerprint(PILOT / "adapter_model.safetensors") != PILOT_SHA:
        raise ValueError("G0-02 registration or pilot adapter changed")
    g0 = json.loads(G0_02.read_text(encoding="utf-8"))
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    model_config = g0["model_config"]
    if draft["model_config"] != model_config:
        raise ValueError("draft and G0-02 model settings differ")
    train, validation, plan, total_updates = check_training_inputs(model_config)
    dev_rows = read_jsonl(DEV)
    effective_dev = evaluation_tasks(DEV, purpose="dev")
    if len(effective_dev) != 95:
        raise ValueError("expected 95 effective dev tasks")
    tokenizer = tokenizer_assets(model_config)
    forecast = draft["cost_estimates_usd"]
    sources = source_hashes()

    a500 = {"id": A500_ID, "stage": "train", **common_fields(g0), "budget_usd": 3.0, "timeout_seconds": 25200,
            "loss_policy": "trained-positions-v1"}
    planned = draft["planned_training"]
    a500["steps"] = [
        {"name": "train-a500", "action": "train", "episodes": TRAIN.as_posix(), "out": f"runs/sql/{A500_ID}/training",
         "role": "a500", "seed": SEED, "validation_share": VAL_SHARE, "epochs": EPOCHS, "max_steps": -1,
         "batch_size": planned["batch_size"], "gradient_accumulation": planned["gradient_accumulation"],
         "learning_rate": planned["learning_rate"], "save_steps": planned["save_steps"],
         "keep_checkpoints": planned["keep_checkpoints"], "eval_steps": 50,
         "loss_policy": "trained-positions-v1", "timeout_seconds": 14400},
        dev_step("dev-a500-nf4", A500_ID, f"runs/sql/{A500_ID}/training/adapter", 12600,
                 verify_checkpoint_reload=True, expected_checkpoint_step=total_updates),
    ]
    a500["assets"] = sorted([asset(TRAIN, "training_episodes"), asset(DEV, "dev_tasks"), *tokenizer,
                             *database_assets(train + validation + dev_rows)], key=lambda a: a["path"])
    a500["source_sha256"] = sources
    a500["training_plan"] = {k: plan[k] for k in (
        "source_sha256", "policy_sha256", "train_examples", "validation_examples", "train_input_tokens_per_pass",
        "train_targets_per_pass", "validation_targets_per_pass", "a_unique_train_targets_per_pass",
        "minimum_unique_verified_b_targets_per_pass")}
    a500["training_plan"].update(expected_optimizer_updates=total_updates,
                                 max_train_input_tokens=max(r["n_input_tokens"] for r in train),
                                 max_validation_input_tokens=max(r["n_input_tokens"] for r in validation),
                                 fresh_local_encoding_verified=True)
    a500["forecast"] = {"training_hours": forecast["a500_training"]["hours"],
                        "training_usd": forecast["a500_training"]["usd"],
                        "dev_screen_hours": forecast["dev_screen_95"]["trained_model_range"]["hours"],
                        "dev_screen_usd": forecast["dev_screen_95"]["trained_model_range"]["usd"],
                        "basis": forecast["basis"], "uncertainty": forecast["uncertainty"]}
    a500["pass_criteria"] = [
        "account, rate, pins and private upload verified; every training and dev reference passes before model use",
        "fresh factory encoding matches the frozen 475/25 plan (847,065 inputs, 107,381 unique targets per pass)",
        f"{total_updates} finite committed optimizer updates via the AgentTune factory, none skipped; ledger covers all",
        "validation losses finite; GPU memory events after load, every update, completion and any OOM",
        f"the dev step reloads the final adapter and its tensors exactly equal the saved step-{total_updates} adapter",
        "95 dev episodes with one finite projected inference position; AuditKIT agrees with every saved score"]
    a500["abort_criteria"] = [
        "wrong account or higher hardware price", "CUDA failure/OOM/nonfinite loss/skipped optimizer step",
        "input/mask/plan/checkpoint/reload mismatch", "sync failure or byte cap", "step/job timeout or stalled output",
        "missing memory measurement, full-prompt inference projection or AuditKIT disagreement"]
    a500["reporting"] = ("Strict, fallback and set dev accuracy, submissions and no-table counts, tokens, time and cost "
                         "are reported whatever they are. G1 and the 1,500 stage rule are computed only after both "
                         "this run and the baselines run finish.")

    baselines = {"id": BASELINES_ID, "stage": "evaluate", **common_fields(g0), "budget_usd": 2.0,
                 "timeout_seconds": 16200}
    baselines["steps"] = [dev_step("dev-base-nf4", BASELINES_ID, None, 5400),
                          dev_step("dev-pilot-nf4", BASELINES_ID, PILOT.as_posix(), 14400)]
    baselines["assets"] = sorted([asset(DEV, "dev_tasks"), *tokenizer, *database_assets(dev_rows),
                                  *[asset(p, "existing_pilot_adapter") for p in sorted(PILOT.iterdir()) if p.is_file()]],
                                 key=lambda a: a["path"])
    baselines["source_sha256"] = sources
    baselines["forecast"] = {"base_hours": forecast["dev_screen_95"]["base_nf4"]["hours"],
                             "base_usd": forecast["dev_screen_95"]["base_nf4"]["usd"],
                             "pilot_hours": forecast["dev_screen_95"]["existing_pilot_nf4"]["hours"],
                             "pilot_usd": forecast["dev_screen_95"]["existing_pilot_nf4"]["usd"],
                             "basis": forecast["basis"], "uncertainty": forecast["uncertainty"]}
    baselines["pass_criteria"] = [
        "account, rate, pins and private upload verified; every dev reference passes before model use",
        "Pilot adapter pinned at 6f2827ae... (the launcher re-hashes every asset when building the upload bundle)",
        "95 dev episodes per condition with one finite projected inference position; AuditKIT agrees with every "
        "saved score; memory events persisted"]
    baselines["abort_criteria"] = [
        "wrong account or higher hardware price", "CUDA failure/OOM/nonfinite inference logits",
        "input or database hash mismatch", "sync failure or byte cap", "step/job timeout or stalled output",
        "missing memory measurement, full-prompt inference projection or AuditKIT disagreement"]
    baselines["reporting"] = a500["reporting"]

    results = {}
    for path, reg in ((A500_REG, a500), (BASELINES_REG, baselines)):
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, reg)
        load_registration(path)
        oracle = verify_runtime_oracles(reg)
        results[reg["id"]] = {**registration_plan(path), "local_reference_checks": oracle,
                              "compute_ceiling_usd": round(reg["timeout_seconds"] / 3600 * PER_HOUR, 2)}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
