"""Read-only: compare the closed-model baseline with Base, A1500 and B on the 144 SQaLe final tasks.

Writes experiments/closed-baseline-v1/RESULTS.json. Uses only local files pinned in the registration.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from ayadb.promptquality import fingerprint
from ayadb.queryshape import CLASSES, classify

REG = Path("experiments/closed-baseline-v1/REGISTRATION.json")
OUT = Path("experiments/closed-baseline-v1/RESULTS.json")
# One-time cost to build B, by rate (experiments/sql-a1500, sql-q2-sampling-v1, sql-selftrain-v1 RESULTS.md).
B_BUILD_USD = {"A1500 training + dev screen": 2.36, "Q2 sampling": 9.11, "B training + dev screen": 0.99}


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def wilson(k, n):
    p, z = k / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def behaviour(rows):
    diags = [d for r in rows for d in r["submission_read_diagnostics"]]
    return {"submitted": sum(r["submitted"] for r in rows),
            "out_of_steps": sum(1 for r in rows if not r["submitted"] and r["tool_calls"] >= 8),
            "submitted_failed_to_run": sum(1 for d in diags if d["accepted"] and d["execution_error"]),
            "malformed_turns": sum(r["malformed_turns"] for r in rows),
            "fabricated_output_turns": sum(r["fabricated_output_turns"] for r in rows),
            "mean_generations": round(sum(r["generations"] for r in rows) / len(rows), 2)}


def main():
    reg = json.loads(REG.read_text(encoding="utf-8"))
    run_dir = Path(reg["out"])
    run = json.loads((run_dir.parent / "run.json").read_text(encoding="utf-8"))
    if run["registration_sha256"] != fingerprint(REG) or run["status"] != "completed":
        raise SystemExit("run record does not match this registration, or the run did not complete")
    tasks = {t["id"]: t for t in jsonl(reg["tasks"]["path"])}
    arms = {c["name"]: {r["task_id"]: r for r in jsonl(c["metrics"])} for c in reg["comparators"]}
    for c in reg["comparators"]:
        if fingerprint(Path(c["metrics"])) != c["metrics_sha256"]:
            raise SystemExit("comparator changed: " + c["name"])
    arms["gpt-6-luna"] = {r["task_id"]: r for r in jsonl(run_dir / "metrics.jsonl")}
    ids = sorted(arms["gpt-6-luna"])
    if any(sorted(a) != ids for a in arms.values()) or len(ids) != 144:
        raise SystemExit("arms do not cover the same 144 tasks")
    shape = {i: classify(tasks[i]["reference_sql"]) for i in ids}
    n = len(ids)
    result = {"registration_sha256": run["registration_sha256"], "model_reported": run["models_reported"],
              "dropped_options": run["dropped_options"], "api_calls": run["api_calls"], "episodes": n,
              "auditkit_agreement": json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["auditkit_agreement"],
              "arms": {}, "paired_vs_B": {}, "by_shape": {}, "roi": {}}
    for name, rows in arms.items():
        rs = [rows[i] for i in ids]
        strict = sum(r["strict"] for r in rs)
        result["arms"][name] = {"strict": strict, "strict_pct": round(100 * strict / n, 1), "strict_95ci_pct": wilson(strict, n),
                                "set": sum(r["bird_set"] for r in rs), "fallback": sum(r["fallback"] for r in rs),
                                **behaviour(rs)}
        result["by_shape"][name] = {c: sum(rows[i]["strict"] for i in ids if shape[i] == c) for c in CLASSES}
    result["by_shape"]["n"] = dict(Counter(shape[i] for i in ids))
    luna = arms["gpt-6-luna"]
    for name in ("Base", "A1500", "B"):
        only_luna = sum(1 for i in ids if luna[i]["strict"] and not arms[name][i]["strict"])
        only_other = sum(1 for i in ids if arms[name][i]["strict"] and not luna[i]["strict"])
        result["paired_vs_B" if name == "B" else f"paired_vs_{name}"] = {
            "luna_only": only_luna, f"{name}_only": only_other, "p_exact_mcnemar": mcnemar(only_luna, only_other)}
    b = next(c for c in reg["comparators"] if c["name"] == "B")
    luna_seconds = run["seconds"]
    result["roi"] = {
        "luna_api_usd": round(run["spent_usd"], 4),
        "luna_usd_per_question": round(run["spent_usd"] / n, 6),
        "luna_usd_per_strict_success": round(run["spent_usd"] / result["arms"]["gpt-6-luna"]["strict"], 6),
        "luna_seconds_per_question": round(luna_seconds / n, 1),
        "b_gpu_usd_by_rate": b["gpu_usd_by_rate"],
        "b_usd_per_question": round(b["gpu_usd_by_rate"] / n, 6),
        "b_usd_per_strict_success": round(b["gpu_usd_by_rate"] / result["arms"]["B"]["strict"], 6),
        "b_seconds_per_question": round(b["seconds"] / n, 1),
        "b_one_time_build_usd": B_BUILD_USD, "b_one_time_build_total_usd": round(sum(B_BUILD_USD.values()), 2),
        "luna_tokens": {"prompt": sum(luna[i]["prompt_tokens"] for i in ids),
                        "generated": sum(luna[i]["generated_tokens"] for i in ids)},
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
