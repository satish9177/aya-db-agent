"""Prepare one reviewable, unapproved G0 registration and local assets; never uploads/runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.runinputs import eligible_training_rows, evaluation_tasks, input_count, read_jsonl
from ayadb.runledger import atomic_json
from ayadb.sqlsftdata import encode_episode

RUN_ID = "g0-sql-agenttune-20261107-01"
REG_PATH = Path("experiments/sql-g0/REGISTRATION.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-draft", action="store_true")
    args = parser.parse_args()
    receipt = Path("runs/hf-sql/hf-user") / RUN_ID
    if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
        raise ValueError("registration has external activity; never revise or restart it")
    out = Path("data/sql-g0-v1")
    out.mkdir(parents=True, exist_ok=args.refresh_draft)
    rows, _ = eligible_training_rows(read_jsonl(Path("data/sqale/data-build-v3/train-4888.episodes.jsonl")))
    longest = sorted(rows, key=lambda e: (-input_count(e), e["id"]))[0]
    actual = encode_episode(longest, official_tokenizer("data/sqale/tokenizer"))
    if len(actual["input_ids"]) != longest["n_input_tokens"] or sum(v != -100 for v in actual["labels"]) != longest["n_target_tokens"]:
        raise ValueError("longest example's fresh encoding differs from audited counts")
    dev, seen = [], set()
    for task in sorted(evaluation_tasks(Path("data/sqale/data-build-v3/dev.tasks.jsonl"), purpose="compatibility"), key=lambda t: t["id"]):
        if task["schema_group"] not in seen:
            dev.append(task); seen.add(task["schema_group"])
        if len(dev) == 3:
            break
    episodes, tasks = out / "longest.episodes.jsonl", out / "dev-3.tasks.jsonl"
    episodes.write_text(json.dumps(longest, ensure_ascii=True) + "\n", encoding="utf-8")
    tasks.write_text("".join(json.dumps(t, ensure_ascii=True) + "\n" for t in dev), encoding="utf-8")
    asset_kinds = {episodes.as_posix(): "training_episode", tasks.as_posix(): "dev_tasks"}
    for filename in {longest["db_path"]} | {t["db_path"] for t in dev}:
        asset_kinds[filename] = "sqale_database"
    for name in ("config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "source.json"):
        asset_kinds["data/sqale/tokenizer/" + name] = "official_tokenizer"
    pilot = Path("remote/hf/pilot/sft-pilot/adapter")
    for name in ("adapter_model.safetensors", "adapter_config.json", "README.md"):
        asset_kinds[(pilot / name).as_posix()] = "existing_pilot_adapter"
    assets = [{"path": name, "kind": kind, "sha256": fingerprint(Path(name)), "bytes": Path(name).stat().st_size}
              for name, kind in sorted(asset_kinds.items())]
    source_paths = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_g0.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sqale-sft/RUN-POLICY-v1.json", "experiments/sqale-sft/AMENDMENT-4-run-inputs-and-control.md")]
    sources = {p.as_posix(): fingerprint(p) for p in sorted(source_paths)}
    cfg = {"model": "CohereLabs/tiny-aya-global", "model_revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
           "tokenizer": "data/sqale/tokenizer", "dtype": "float16", "quantization": "nf4-double-quant",
           "rank": 16, "alpha": 32, "dropout": .05, "max_input_tokens": 4096,
           "context_tokens": 8192, "max_new_tokens": 512, "tool_steps": 8}
    train_out = f"runs/sql/{RUN_ID}/training"
    common = {"action": "train", "episodes": episodes.as_posix(), "out": train_out, "role": "compatibility",
              "seed": 42, "validation_share": 0, "epochs": 2, "max_steps": 3, "batch_size": 1,
              "gradient_accumulation": 1, "learning_rate": .0002, "save_steps": 1, "keep_checkpoints": 3,
              "timeout_seconds": 1800}
    steps = [{**common, "name": "train-to-2", "stop_after_steps": 2},
             {**common, "name": "resume-to-3", "resume": train_out + "/checkpoints/checkpoint-2", "expected_resume_step": 2}]
    for tag, adapter in (("base-nf4", None), ("pilot-nf4", pilot.as_posix()),
                         ("checkpoint3-nf4", train_out + "/checkpoints/checkpoint-3")):
        step = {"name": "infer-" + tag, "action": "evaluate", "tasks": tasks.as_posix(), "purpose": "compatibility",
                "out": f"runs/sql/{RUN_ID}/infer-{tag}", "adapter": adapter, "samples": 1,
                "temperature": 0., "seed": 42, "timeout_seconds": 1200}
        if tag == "checkpoint3-nf4":
            step.update(verify_checkpoint_reload=True, expected_checkpoint_step=3)
        steps.append(step)
    reg = {"id": RUN_ID, "status": "DRAFT_AWAITING_EXPLICIT_USER_APPROVAL", "format": "sql-submit-v1",
           "stage": "g0", "expected_account": "hf-user", "hardware": "t4-small", "budget_usd": 3,
           "max_price_per_minute_usd": .006667, "timeout_seconds": 7200, "stall_seconds": 1200,
           "checkpoint_sync_seconds": 60, "max_input_upload_bytes": 512 * 1024 ** 2,
           "max_output_upload_bytes": 1024 ** 3, "model_config": cfg, "steps": steps, "assets": assets,
           "source_sha256": sources, "longest_example_id": longest["id"],
           "longest_input_tokens": longest["n_input_tokens"], "longest_target_tokens": longest["n_target_tokens"],
           "dev_task_ids": [t["id"] for t in dev], "temporary_adapter_not_adoptable": True,
           "no_bird_or_q2": True, "no_teacher_or_paid_api": True,
           "first_month_storage_reserve_usd": .04,
           "artifact_retention": "Review/pull the unique G0 input/output prefixes after completion; retain at most 30 days without renewed storage approval.",
           "pass_criteria": ["three finite optimizer steps via AgentTune factory, split as two then resume one",
                             "runtime reference SQL succeeds and prepared dev gold agrees before any model use",
                             "fresh encoding matches 4072-token longest example and assistant/error masks",
                             "checkpoint-2 includes optimizer/scheduler and identical data/settings identity",
                             "fresh process loads checkpoint-3 and every saved adapter tensor matches",
                             "last-token inference logits are finite on every checked generation forward",
                             "all three conditions generate model tokens for the same three dev tasks",
                             "AuditKIT agrees with all saved strict scores; correctness is reported, not a compatibility gate"],
           "abort_criteria": ["wrong account or higher hardware price", "CUDA failure/OOM/nonfinite loss/skipped optimizer step",
                              "mask/input/checkpoint/reload mismatch", "sync failure or upload byte cap",
                              "step/job timeout or stalled output"],
           "scope_exclusions": ["vLLM validation and speedup claims", "full A/A+/B training", "Q2 sampling",
                                "BIRD/final evaluation", "automatic retries or extra jobs", "push/publication"]}
    REG_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REG_PATH, reg)
    atomic_json(out / "preparation.json", {"registration": RUN_ID, "longest_example_id": longest["id"],
                "fresh_longest_encoding_verified": True, "input_tokens": longest["n_input_tokens"],
                "dev_task_ids": reg["dev_task_ids"], "asset_bytes": sum(a["bytes"] for a in assets),
                "database_files": len([a for a in assets if a["kind"] == "sqale_database"]),
                "uploads": 0, "model_runs": 0, "paid_runs": 0})
    print(json.dumps({"registration": str(REG_PATH), "sha256": fingerprint(REG_PATH),
                      "asset_files": len(assets), "asset_bytes": sum(a["bytes"] for a in assets),
                      "longest_input_tokens": longest["n_input_tokens"], "dev_task_ids": reg["dev_task_ids"],
                      "uploads": 0, "model_runs": 0, "paid_runs": 0}, indent=2))


if __name__ == "__main__":
    main()
