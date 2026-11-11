"""Build the A1500-stage registration locally (amendment 8); never loads weights, uploads or launches.

a1500-sql-agenttune-20261108-01: train A1500 from the base model on the nested 1,500 set with A500's
settings, then screen its final adapter on the same 95 dev tasks. Base/Pilot/A500 are not rerun.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.queryshape import CLASSES, classify, is_complex
from ayadb.runconfig import load_registration, registration_plan
from ayadb.runinputs import evaluation_tasks, read_jsonl, training_plan, verify_runtime_oracles
from ayadb.runledger import atomic_json
from ayadb.sqlsftdata import encode_episode

RUN_ID = "a1500-sql-agenttune-20261108-01"
REG = Path("experiments/sql-a1500/REGISTRATION.json")
A500_REG = Path("experiments/sql-a500/REGISTRATION.json")
A500_SHA = "c3b006ce2377c24762a8d57bdddd5a3499e9e817c7bb43ef0d16a21af4e47e3d"
BASELINES_SHA = "53049d18e68d5645e3a656e63a00232b6fe8ce86c49f2ee55979e02d89410ee9"
A500_METRICS = Path("remote/hf-sql/hf-user/a500-sql-agenttune-20261107-01/dev-a500-nf4/metrics.jsonl")
BASE_METRICS = Path("remote/hf-sql/hf-user/dev-baselines-sql-20261107-01/dev-base-nf4/metrics.jsonl")
PILOT_METRICS = Path("remote/hf-sql/hf-user/dev-baselines-sql-20261107-01/dev-pilot-nf4/metrics.jsonl")
TRAIN = Path("data/sqale/data-build-v3/train-1500.episodes.jsonl")
TRAIN_SHA = "0c920dab1c5b09256ef7f9c76b4864f7e075ce72db3ca14d4dc978849b46b62e"
TRAIN_500 = Path("data/sqale/data-build-v3/train-500.episodes.jsonl")
DEV = Path("data/sqale/data-build-v3/dev.tasks.jsonl")
DEV_SHA = "db613ff79c2b08753d13f5624ea1950d830aa259b8e35a9f1434409001358716"
FROZEN_PLAN = Path("data/sqale/runner-inputs-v1/a-1500.json")
SEED, VAL_SHARE, EPOCHS, ACCUMULATION = 42, .05, 2, 16
PER_HOUR = .40
A500_TOKENS_PER_SECOND, G0_02_TOKENS_PER_SECOND = 363.8, 266.5
A500_DEV_SECONDS = 5910.1
JOB_TIMEOUT, TRAIN_TIMEOUT, DEV_TIMEOUT, BUDGET = 34200, 25200, 12600, 4.0


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


def check_training_inputs(model_config):
    if fingerprint(TRAIN) != TRAIN_SHA or fingerprint(DEV) != DEV_SHA:
        raise ValueError("frozen 1,500/dev inputs changed")
    ids_500 = {r["id"] for r in read_jsonl(TRAIN_500)}
    if not ids_500 <= {r["id"] for r in read_jsonl(TRAIN)}:
        raise ValueError("the 1,500 set must contain every 500 row (nested stages)")
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
    updates_per_pass = math.ceil(len(train) / ACCUMULATION)
    return train, validation, plan, updates_per_pass * EPOCHS


def comparators():
    out = {}
    for name, path, registration in (("dev-base-nf4", BASE_METRICS, BASELINES_SHA),
                                     ("dev-pilot-nf4", PILOT_METRICS, BASELINES_SHA),
                                     ("dev-a500-nf4", A500_METRICS, A500_SHA)):
        rows = read_jsonl(path)
        if len(rows) != 95:
            raise ValueError("comparator must have 95 dev episodes: " + name)
        out[name] = {"registration_sha256": registration, "metrics_sha256": fingerprint(path),
                     "strict": sum(r["strict"] for r in rows), "fallback": sum(r["fallback"] for r in rows),
                     "set": sum(r["bird_set"] for r in rows)}
    return out


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_a1500.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json",
        "experiments/sqale-sft/AMENDMENT-4-run-inputs-and-control.md",
        "experiments/sqale-sft/AMENDMENT-5-table-reads-and-manual-audit.md",
        "experiments/sqale-sft/AMENDMENT-6-selected-logits-g0-02.md",
        "experiments/sqale-sft/AMENDMENT-7-a500-stage.md",
        "experiments/sqale-sft/AMENDMENT-8-a1500-stage.md")]
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def main():
    receipt = Path("runs/hf-sql/hf-user") / RUN_ID
    if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
        raise ValueError(RUN_ID + " has external activity; never revise or retry")
    if fingerprint(A500_REG) != A500_SHA:
        raise ValueError("A500 registration changed")
    a500 = json.loads(A500_REG.read_text(encoding="utf-8"))
    model_config = a500["model_config"]
    train, validation, plan, total_updates = check_training_inputs(model_config)
    dev_rows = read_jsonl(DEV)
    effective_dev = evaluation_tasks(DEV, purpose="dev")
    if len(effective_dev) != 95:
        raise ValueError("expected 95 effective dev tasks")
    shapes = {t["id"]: classify(t["reference_sql"]) for t in effective_dev}
    shape_counts = Counter(shapes.values())
    if [shape_counts[c] for c in CLASSES] != [37, 5, 23, 10, 20]:
        raise ValueError("dev query-shape counts differ from amendment 8")
    compared = comparators()
    if compared["dev-a500-nf4"]["strict"] != 28:
        raise ValueError("A500 comparator differs from its reported result")

    reg = {"id": RUN_ID, "stage": "train"}
    for key in ("format", "expected_account", "hardware", "max_price_per_minute_usd", "stall_seconds",
                "checkpoint_sync_seconds", "max_input_upload_bytes", "max_output_upload_bytes", "model_config",
                "scorer_policy", "inference_logits_to_keep", "required_environment", "g0_02_registration_sha256",
                "g0_02_job_id", "dev_file", "dev_sha256", "dev_effective_questions", "no_bird_or_q2",
                "no_teacher_or_paid_api", "first_month_storage_reserve_usd", "artifact_retention", "loss_policy"):
        reg[key] = copy.deepcopy(a500[key])
    reg.update({
        "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
        "amendment": "experiments/sqale-sft/AMENDMENT-8-a1500-stage.md",
        "a500_registration_sha256": A500_SHA, "a500_job_id": "6aef63bbe6320c66fad617fa",
        "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                         "uploads, one job and bounded storage",
        "scope_exclusions": ["Base/Pilot/A500 reruns (reused comparators)", "intermediate checkpoint screening",
                             "shop retention (deferred to final evaluation, amendment 7)",
                             "full stage, Q2 sampling, A+/B", "BIRD/final evaluation", "teacher/paid API",
                             "step-limit or prompt changes", "vLLM", "automatic retries or extra jobs",
                             "hardware change", "push/publication"],
        "budget_usd": BUDGET, "timeout_seconds": JOB_TIMEOUT,
    })
    planned = a500["steps"][0]
    reg["steps"] = [
        {"name": "train-a1500", "action": "train", "episodes": TRAIN.as_posix(), "out": f"runs/sql/{RUN_ID}/training",
         "role": "a1500", **{k: planned[k] for k in (
             "seed", "validation_share", "epochs", "max_steps", "batch_size", "gradient_accumulation",
             "learning_rate", "save_steps", "keep_checkpoints", "eval_steps", "loss_policy")},
         "timeout_seconds": TRAIN_TIMEOUT},
        {"name": "dev-a1500-nf4", "action": "evaluate", "tasks": DEV.as_posix(), "purpose": "dev",
         "out": f"runs/sql/{RUN_ID}/dev-a1500-nf4", "adapter": f"runs/sql/{RUN_ID}/training/adapter",
         "samples": 1, "temperature": 0.0, "seed": SEED, "inference_logits_to_keep": 1,
         "timeout_seconds": DEV_TIMEOUT, "verify_checkpoint_reload": True,
         "expected_checkpoint_step": total_updates},
    ]
    tokenizer_folder = Path(model_config["tokenizer"])
    reg["assets"] = sorted([asset(TRAIN, "training_episodes"), asset(DEV, "dev_tasks"),
                            *[asset(p, "official_tokenizer") for p in sorted(tokenizer_folder.iterdir()) if p.is_file()],
                            *database_assets(train + validation + dev_rows)], key=lambda a: a["path"])
    reg["source_sha256"] = source_hashes()
    reg["training_plan"] = {k: plan[k] for k in (
        "source_sha256", "policy_sha256", "train_examples", "validation_examples", "train_input_tokens_per_pass",
        "train_targets_per_pass", "validation_targets_per_pass", "a_unique_train_targets_per_pass",
        "minimum_unique_verified_b_targets_per_pass")}
    sqale_train = [r for r in train if r.get("source") != "shop-converted"]
    reg["training_plan"].update(
        expected_optimizer_updates=total_updates,
        max_train_input_tokens=max(r["n_input_tokens"] for r in train),
        max_validation_input_tokens=max(r["n_input_tokens"] for r in validation),
        contains_all_a500_rows=True, starts_from="untuned base model (no parent adapter)",
        sqale_train_rows_by_shape=dict(Counter(classify(r["reference_sql"]) for r in sqale_train)),
        fresh_local_encoding_verified=True)
    two_pass_tokens = 2 * plan["train_input_tokens_per_pass"]
    reg["forecast"] = {
        "training_hours": round(two_pass_tokens / A500_TOKENS_PER_SECOND / 3600, 2),
        "training_hours_at_g0_02_speed": round(two_pass_tokens / G0_02_TOKENS_PER_SECOND / 3600, 2),
        "dev_screen_hours": [round(A500_DEV_SECONDS / 3600, 2), 2.5],
        "total_hours_central": round(two_pass_tokens / A500_TOKENS_PER_SECOND / 3600 + A500_DEV_SECONDS / 3600 + .1, 2),
        "basis": "A500 measured 363.8 input tokens/s (including saves and validation) and 5,910s for the 95-task dev "
                 "screen on the same t4-small; t4-small $0.40/h. Planning estimates, not caps.",
        "uncertainty": "The 1,500 rows average 1,884 input tokens vs 1,783 for 500; a model that explores more takes "
                       "longer per dev episode. The job timeout bounds the cost either way."}
    reg["forecast"]["total_usd_central"] = round(reg["forecast"]["total_hours_central"] * PER_HOUR, 2)
    reg["comparators"] = compared
    reg["stage_rule"] = {"rule": "stop data scaling at 1,500 unless A1500 beats A500 by at least five unrounded "
                                 "strict points on the same 95 tasks",
                         "a500_strict": 28, "a1500_needs_at_least": 33}
    reg["query_shape_diagnostic"] = {
        "module": "src/ayadb/queryshape.py", "classes": list(CLASSES), "complex": ["3+ tables", "multi-step"],
        "dev_counts": {c: shape_counts[c] for c in CLASSES},
        "complex_dev_task_ids": sorted(t["id"] for t in effective_dev if is_complex(t["reference_sql"])),
        "role": "descriptive only; defined after reading A500/baseline dev results, frozen before A1500; "
                "never a gate, score, reward or selection rule"}
    reg["pass_criteria"] = [
        "account, rate, pins and private upload verified; every training and dev reference passes before model use",
        f"fresh factory encoding matches the frozen 1,425/75 plan ({plan['train_input_tokens_per_pass']:,} inputs, "
        f"{plan['train_targets_per_pass']:,} unique targets per pass)",
        f"{total_updates} finite committed optimizer updates via the AgentTune factory, none skipped; ledger covers all",
        "validation losses finite; GPU memory events after load, every update, completion and any OOM",
        f"the dev step reloads the final adapter and its tensors exactly equal the saved step-{total_updates} adapter",
        "95 dev episodes with one finite projected inference position; AuditKIT agrees with every saved score"]
    reg["abort_criteria"] = copy.deepcopy(a500["abort_criteria"])
    reg["reporting"] = ("Strict, fallback and set dev accuracy, submissions and no-table counts, tokens, time and cost "
                        "are reported whatever they are, with the amendment-8 query-shape breakdown for Base, Pilot, "
                        "A500 and A1500 and the stage rule against A500's 28/95. Amendment 5's audit is repeated.")

    REG.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REG, reg)
    load_registration(REG)
    oracle = verify_runtime_oracles(reg)
    result = {**registration_plan(REG), "local_reference_checks": oracle,
              "compute_ceiling_usd": round(reg["timeout_seconds"] / 3600 * PER_HOUR, 2),
              "expected_optimizer_updates": total_updates, "forecast": reg["forecast"],
              "comparators": compared, "sqale_train_rows_by_shape": reg["training_plan"]["sqale_train_rows_by_shape"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
