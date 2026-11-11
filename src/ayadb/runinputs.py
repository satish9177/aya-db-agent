"""One effective-input/split implementation for local plans and the future model runner."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .promptquality import fingerprint

POLICY_PATH = Path("experiments/sqale-sft/RUN-POLICY-v1.json")


def read_jsonl(path: Path):
    if Path(path).resolve() == Path("data/test.jsonl").resolve():
        raise ValueError("the spent sealed shop test is prohibited")
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def row_identity(row: dict) -> str:
    fields = [row.get("question_id"), row.get("prompt"), row.get("messages"), row.get("db_sha256"),
              row.get("discovery_order"), row.get("labels")]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def supervised_count(row: dict) -> int:
    if "labels" in row:
        return sum(v != -100 for v in row["labels"])
    return int(row["n_target_tokens"])


def input_count(row: dict) -> int:
    return len(row["input_ids"]) if "input_ids" in row else int(row["n_input_tokens"])


def eligible_training_rows(rows: list[dict], policy=None):
    policy = policy or load_policy()
    kept, excluded = [], []
    for row in rows:
        if row.get("source") == "bird-mini-dev" or row.get("dataset_usage") == "final_only" or str(row.get("id", "")).startswith("bird:"):
            raise ValueError("BIRD is final-only and cannot be a training input")
        if row.get("format") != "sql-submit-v1" or not isinstance(row.get("messages"), list):
            raise ValueError("expected audited sql-submit-v1 training episodes")
        identifier = row.get("task_id", row["question_id"])
        if identifier in policy["excluded_ids"]:
            excluded.append({"id": row["id"], "task_id": identifier, "reason": policy["excluded_ids"][identifier]})
        else:
            kept.append(row)
    if not kept:
        raise ValueError("no eligible training rows")
    return kept, excluded


def split_training_rows(rows: list[dict], *, seed=42, val_share=.05):
    if not 0 <= val_share < 1:
        raise ValueError("invalid validation share")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, round(len(rows) * val_share)) if val_share else 0
    if n_val >= len(rows):
        raise ValueError("validation would consume every training example")
    return [rows[i] for i in indices[n_val:]], [rows[i] for i in indices[:n_val]]


def training_plan(path: Path, *, seed=42, val_share=.05, policy=None):
    policy = policy or load_policy()
    rows, excluded = eligible_training_rows(read_jsonl(path), policy)
    train, validation = split_training_rows(rows, seed=seed, val_share=val_share)
    unique = {row_identity(r): r for r in train}
    denominator = sum(supervised_count(r) for r in unique.values())
    plan = {"source_file": Path(path).as_posix(), "source_sha256": fingerprint(path),
            "policy_sha256": fingerprint(POLICY_PATH), "policy_version": policy["version"],
            "seed": seed, "validation_share": val_share, "eligible_rows": len(rows), "excluded": excluded,
            "train_examples": len(train), "validation_examples": len(validation),
            "train_episode_ids": [r["id"] for r in train], "validation_episode_ids": [r["id"] for r in validation],
            "train_row_identities": [row_identity(r) for r in train],
            "train_input_tokens_per_pass": sum(input_count(r) for r in train),
            "train_targets_per_pass": sum(supervised_count(r) for r in train),
            "validation_targets_per_pass": sum(supervised_count(r) for r in validation),
            "unique_train_examples_per_pass": len(unique), "a_unique_train_targets_per_pass": denominator,
            "minimum_unique_verified_b_targets_per_pass": (denominator + 9) // 10,
            "basis": "actual factory training rows after policy/split, unique per pass; validation/epochs/copies excluded"}
    return train, validation, plan


def evaluation_tasks(path: Path, *, purpose: str, selection_sha256: str | None = None):
    rows, policy = read_jsonl(path), load_policy()
    for row in rows:
        bird = row.get("source") == "bird-mini-dev" or row.get("dataset_usage") == "final_only"
        final = bird or row.get("split") == "final"
        if final and (purpose != "final" or not selection_sha256):
            raise ValueError("final tasks require final purpose and a frozen selection identity")
        if row.get("split") == "q2" and purpose != "sampling":
            raise ValueError("Q2 is for registered sampling, not dev/checkpoint selection")
        if row.get("format", "sql-submit-v1") != "sql-submit-v1":
            raise ValueError("unexpected evaluation format")
    return [r for r in rows if r["id"] not in policy["excluded_ids"]]


def verify_runtime_oracles(reg):
    """Before model use, catch SQL/runtime drift against prepared final/dev gold results."""
    from .birdverify import decode_gold
    from .scoringlimits import SCORING_MAX_OPS, SCORING_SECONDS
    from .sqlscore import execute, rows_equal
    from .sqlguard import require_reference_reads
    checked, seen = 0, set()
    for step in reg["steps"]:
        filename = step.get("tasks") or step.get("episodes")
        if not filename or filename in seen:
            continue
        seen.add(filename)
        for row in read_jsonl(Path(filename)):
            identifier = row.get("task_id", row["id"])
            if identifier in load_policy()["excluded_ids"]:
                continue
            result = execute(Path(row["db_path"]), row["reference_sql"], seconds=SCORING_SECONDS, max_ops=SCORING_MAX_OPS)
            if result.error:
                raise ValueError("runtime reference failed before model use: " + identifier + " " + result.error)
            require_reference_reads(Path(row["db_path"]), row["reference_sql"], result)
            if "gold_rows" in row:
                gold = [[decode_gold(v) for v in r] for r in row["gold_rows"]]
                if not rows_equal(result.rows, gold, ordered=row.get("ordered", False)):
                    raise ValueError("runtime reference differs from prepared gold: " + identifier)
            checked += 1
    return {"references_checked": checked, "model_calls": 0}


def write_stage_plans(out=Path("data/sqale/runner-inputs-v1")):
    out.mkdir(parents=True, exist_ok=False)
    plans = {}
    for stage in (500, 1500, 4888):
        _, _, plan = training_plan(Path(f"data/sqale/data-build-v3/train-{stage}.episodes.jsonl"))
        (out / f"a-{stage}.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        plans[str(stage)] = {k: v for k, v in plan.items() if not k.endswith("_ids") and k != "train_row_identities"}
    summary = {"stages": plans, "dev": len(evaluation_tasks(Path("data/sqale/data-build-v3/dev.tasks.jsonl"), purpose="dev")),
               "q2": len(evaluation_tasks(Path("data/sqale/data-build-v3/q2.tasks.jsonl"), purpose="sampling")),
               "model_runs": 0, "paid_runs": 0, "uploads": 0}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    write_stage_plans()
