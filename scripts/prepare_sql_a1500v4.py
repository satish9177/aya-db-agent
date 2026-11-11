"""Build the A1500-v4 registration locally (amendment 10); never loads weights, uploads or launches.

a1500v4-sql-agenttune-20261109-01: train from the base model on data-build-v4/train-1500 (same 1,500
questions and split as A1500, recovery and step-by-step demo paths) with A1500's settings, then screen
the final adapter on the same 95 dev tasks with the original tools.
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

RUN_ID = "a1500v4-sql-agenttune-20261109-01"
REG = Path("experiments/sql-a1500v4/REGISTRATION.json")
A1500_REG = Path("experiments/sql-a1500/REGISTRATION.json")
A1500_SHA = "92a8fc109e26f6de9a8e2196ecfdfb4bc62d85f23b20468a4d65ccd974aa8363"
HINTS_SHA = "02984087902a41ba11d6c62e33743d5342e33788117fc0f455c9916be9154d6a"
A1500_METRICS = Path("remote/hf-sql/hf-user/a1500-sql-agenttune-20261108-01/dev-a1500-nf4/metrics.jsonl")
A1500_METRICS_SHA = "202aef13a5e7488cc4f41a294545a7879ae4d2f97189853121986caef7b54724"
HINTS_METRICS = Path("remote/hf-sql/hf-user/a1500-hints-sql-20261108-01/dev-a1500-hints-nf4/metrics.jsonl")
V3_TRAIN_SHA = "0c920dab1c5b09256ef7f9c76b4864f7e075ce72db3ca14d4dc978849b46b62e"
TRAIN = Path("data/sqale/data-build-v4/train-1500.episodes.jsonl")
TRAIN_SHA = "569e8aedd46dcbb0552012cd141bd039d388b98fef682d3dcd442a183122f1a0"
MANIFEST = Path("data/sqale/data-build-v4/manifest.json")
DEV = Path("data/sqale/data-build-v3/dev.tasks.jsonl")
DEV_SHA = "db613ff79c2b08753d13f5624ea1950d830aa259b8e35a9f1434409001358716"
FROZEN_1500 = Path("data/sqale/runner-inputs-v1/a-1500.json")
SEED, VAL_SHARE, EPOCHS, ACCUMULATION = 42, .05, 2, 16
PER_HOUR, A1500_TOKENS_PER_SECOND, A1500_DEV_SECONDS = .40, 368.4, 6385.9
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
        raise ValueError("v4 training file or dev tasks changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["source_sha256"] != V3_TRAIN_SHA or manifest["output_sha256"] != TRAIN_SHA or manifest["model_calls"]:
        raise ValueError("v4 build manifest does not match")
    train, validation, plan = training_plan(TRAIN, seed=SEED, val_share=VAL_SHARE)
    frozen = json.loads(FROZEN_1500.read_text(encoding="utf-8"))
    for key in ("train_examples", "validation_examples", "train_episode_ids", "validation_episode_ids"):
        if plan[key] != frozen[key]:
            raise ValueError("v4 split differs from A1500's frozen split: " + key)
    tokenizer = official_tokenizer(model_config["tokenizer"])
    for row in train + validation:
        encoded = encode_episode(row, tokenizer, model_config["max_input_tokens"])
        if encoded is None or len(encoded["input_ids"]) != row["n_input_tokens"] or \
                sum(v != -100 for v in encoded["labels"]) != row["n_target_tokens"]:
            raise ValueError("fresh encoding differs from stored counts: " + row["id"])
    return train, validation, plan, math.ceil(len(train) / ACCUMULATION) * EPOCHS, manifest


def comparator(path, registration, expected):
    rows = read_jsonl(path)
    got = (sum(r["strict"] for r in rows), sum(r["fallback"] for r in rows), sum(r["bird_set"] for r in rows))
    if len(rows) != 95 or got != expected:
        raise ValueError("comparator differs from its reported result: " + path.as_posix())
    return {"registration_sha256": registration, "metrics_sha256": fingerprint(path),
            "strict": got[0], "fallback": got[1], "set": got[2]}


def source_hashes():
    files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/build_sqale_v4_recovery.py", "scripts/prepare_sql_a1500v4.py",
        "requirements-sql-runner.txt", "pyproject.toml", "experiments/sqale-sft/RUN-POLICY-v1.json",
        *[f"experiments/sqale-sft/{n}" for n in (
            "AMENDMENT-4-run-inputs-and-control.md", "AMENDMENT-5-table-reads-and-manual-audit.md",
            "AMENDMENT-6-selected-logits-g0-02.md", "AMENDMENT-7-a500-stage.md", "AMENDMENT-8-a1500-stage.md",
            "AMENDMENT-9-error-feedback-tools.md", "AMENDMENT-10-recovery-demos.md")])]
    return {p.as_posix(): fingerprint(p) for p in sorted(files)}


def main():
    receipt = Path("runs/hf-sql/hf-user") / RUN_ID
    if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
        raise ValueError(RUN_ID + " has external activity; never revise or retry")
    if fingerprint(A1500_REG) != A1500_SHA or fingerprint(A1500_METRICS) != A1500_METRICS_SHA:
        raise ValueError("A1500 registration or result changed")
    a1500 = json.loads(A1500_REG.read_text(encoding="utf-8"))
    model_config = a1500["model_config"]
    train, validation, plan, total_updates, manifest = check_training_inputs(model_config)
    dev_rows = read_jsonl(DEV)
    effective_dev = evaluation_tasks(DEV, purpose="dev")
    shape_counts = Counter(classify(t["reference_sql"]) for t in effective_dev)
    if len(effective_dev) != 95 or [shape_counts[c] for c in CLASSES] != [37, 5, 23, 10, 20]:
        raise ValueError("dev tasks or shape counts differ from amendment 8")
    compared = {"dev-a1500-nf4": comparator(A1500_METRICS, A1500_SHA, (37, 37, 40)),
                "dev-a1500-hints-nf4": {**comparator(HINTS_METRICS, HINTS_SHA, (38, 38, 41)),
                                        "tool_feedback": "column-hints-v1"}}

    reg = {"id": RUN_ID, "stage": "train"}
    for key in ("format", "expected_account", "hardware", "max_price_per_minute_usd", "stall_seconds",
                "checkpoint_sync_seconds", "max_input_upload_bytes", "max_output_upload_bytes", "model_config",
                "scorer_policy", "inference_logits_to_keep", "required_environment", "g0_02_registration_sha256",
                "g0_02_job_id", "dev_file", "dev_sha256", "dev_effective_questions", "no_bird_or_q2",
                "no_teacher_or_paid_api", "first_month_storage_reserve_usd", "artifact_retention", "loss_policy"):
        reg[key] = copy.deepcopy(a1500[key])
    reg.update({
        "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL",
        "amendment": "experiments/sqale-sft/AMENDMENT-10-recovery-demos.md",
        "a1500_registration_sha256": A1500_SHA, "a1500_job_id": "6af0139be6320c66fad700ed",
        "authorization": "Draft only: the user's explicit approval must name this SHA256 and cover its private "
                         "uploads, one job and bounded storage",
        "scope_exclusions": ["A1500/hints/Base/Pilot/A500 reruns (reused comparators)", "intermediate checkpoint screening",
                             "error-feedback tools (not adopted, amendment 9)", "shop retention (deferred to final evaluation)",
                             "full stage, Q2 sampling, A+/B", "BIRD/final evaluation", "teacher/paid API",
                             "step-limit, prompt or decoding changes", "vLLM", "automatic retries or extra jobs",
                             "hardware change", "push/publication"],
        "budget_usd": BUDGET, "timeout_seconds": JOB_TIMEOUT,
    })
    old_train, old_dev = a1500["steps"]
    reg["steps"] = [
        {"name": "train-a1500v4", "action": "train", "episodes": TRAIN.as_posix(), "out": f"runs/sql/{RUN_ID}/training",
         "role": "a1500v4", **{k: old_train[k] for k in (
             "seed", "validation_share", "epochs", "max_steps", "batch_size", "gradient_accumulation",
             "learning_rate", "save_steps", "keep_checkpoints", "eval_steps", "loss_policy")},
         "timeout_seconds": TRAIN_TIMEOUT},
        {"name": "dev-a1500v4-nf4", "action": "evaluate", "tasks": DEV.as_posix(), "purpose": "dev",
         "out": f"runs/sql/{RUN_ID}/dev-a1500v4-nf4", "adapter": f"runs/sql/{RUN_ID}/training/adapter",
         **{k: old_dev[k] for k in ("samples", "temperature", "seed", "inference_logits_to_keep")},
         "timeout_seconds": DEV_TIMEOUT, "verify_checkpoint_reload": True, "expected_checkpoint_step": total_updates},
    ]
    tokenizer_folder = Path(model_config["tokenizer"])
    reg["assets"] = sorted([asset(TRAIN, "training_episodes_v4"), asset(DEV, "dev_tasks"),
                            *[asset(p, "official_tokenizer") for p in sorted(tokenizer_folder.iterdir()) if p.is_file()],
                            *database_assets(train + validation + dev_rows)], key=lambda a: a["path"])
    reg["source_sha256"] = source_hashes()
    reg["training_plan"] = {k: plan[k] for k in (
        "source_sha256", "policy_sha256", "train_examples", "validation_examples", "train_input_tokens_per_pass",
        "train_targets_per_pass", "validation_targets_per_pass", "a_unique_train_targets_per_pass")}
    reg["training_plan"].update(
        expected_optimizer_updates=total_updates, same_split_as_a1500=True,
        max_train_input_tokens=max(r["n_input_tokens"] for r in train),
        max_validation_input_tokens=max(r["n_input_tokens"] for r in validation),
        starts_from="untuned base model (no parent adapter)", fresh_local_encoding_verified=True,
        v4_changes_in_train_split=dict(Counter(r.get("v4_change") for r in train if r.get("v4_change"))),
        v4_build_counts=manifest["counts"])
    hours = 2 * plan["train_input_tokens_per_pass"] / A1500_TOKENS_PER_SECOND / 3600
    reg["forecast"] = {"training_hours": round(hours, 2), "dev_screen_hours": round(A1500_DEV_SECONDS / 3600, 2),
                       "total_hours_central": round(hours + A1500_DEV_SECONDS / 3600 + .05, 2),
                       "basis": "A1500 measured 368.4 input tokens/s and 6,386 s for the dev screen on the same "
                                "t4-small; $0.40/h. Planning estimates, not caps."}
    reg["forecast"]["total_usd_central"] = round(reg["forecast"]["total_hours_central"] * PER_HOUR, 2)
    reg["comparators"] = compared
    reg["decision_rule"] = {"rule": "A1500-v4 replaces A1500 as the trained finalist for the final evaluation if its "
                                    "dev strict score is higher than A1500's on the same 95 tasks; a tie keeps A1500",
                            "a1500_strict": 37, "a1500v4_replaces_if_at_least": 38}
    reg["query_shape_diagnostic"] = {
        "module": "src/ayadb/queryshape.py", "dev_counts": {c: shape_counts[c] for c in CLASSES},
        "complex_dev_task_ids": sorted(t["id"] for t in effective_dev if is_complex(t["reference_sql"])),
        "role": "descriptive only (amendment 8)"}
    reg["pass_criteria"] = [
        "account, rate, pins and private upload verified; every training and dev reference passes before model use",
        f"fresh factory encoding matches the stored v4 counts ({plan['train_input_tokens_per_pass']:,} inputs, "
        f"{plan['train_targets_per_pass']:,} targets per pass; 1,425/75, A1500's split)",
        f"{total_updates} finite committed optimizer updates via the AgentTune factory, none skipped; ledger covers all",
        "validation losses finite; GPU memory events after load, every update, completion and any OOM",
        f"the dev step reloads the final adapter and its tensors exactly equal the saved step-{total_updates} adapter",
        "95 dev episodes with one finite projected inference position; AuditKIT agrees with every saved score"]
    reg["abort_criteria"] = copy.deepcopy(a1500["abort_criteria"])
    reg["reporting"] = ("Strict, fallback and set dev accuracy against A1500's 37/95, paired discordant tasks, "
                        "amendment-8 shapes, recoveries after a failed query, what follows a failed query, partial-query "
                        "use, loss, time, cost and memory are reported whatever they are.")

    REG.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REG, reg)
    load_registration(REG)
    oracle = verify_runtime_oracles(reg)
    print(json.dumps({**registration_plan(REG), "local_reference_checks": oracle,
                      "compute_ceiling_usd": round(reg["timeout_seconds"] / 3600 * PER_HOUR, 2),
                      "expected_optimizer_updates": total_updates, "forecast": reg["forecast"],
                      "training_plan": reg["training_plan"], "comparators": compared}, indent=2))


if __name__ == "__main__":
    main()
