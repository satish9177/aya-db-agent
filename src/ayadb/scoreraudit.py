"""Free reference inventory and separate historical-log table-read diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .promptquality import fingerprint
from .runinputs import eligible_training_rows, load_policy, read_jsonl
from .runledger import atomic_json
from .sqlguard import POLICY, require_reference_reads, submission_reads


def reference_inventory(out: Path):
    out.mkdir(parents=True, exist_ok=False)
    root = Path("data/sqale/data-build-v3")
    paths = [root / "train-4888.episodes.jsonl", root / "dev.tasks.jsonl", root / "q2.tasks.jsonl",
             root / "final.tasks.jsonl", root / "retention/dev.tasks.jsonl", root / "retention/mh.tasks.jsonl",
             Path("data/bird/final-v1/tasks.jsonl")]
    excluded = load_policy()["excluded_ids"]
    seen, summary = {}, {"scorer_policy": POLICY, "sources": {}, "model_calls": 0}
    with (out / "reference-reads.jsonl").open("w", encoding="utf-8") as records:
        for path in paths:
            rows = read_jsonl(path)
            if path.name.endswith("episodes.jsonl"):
                rows, _ = eligible_training_rows(rows)
            else:
                rows = [r for r in rows if r["id"] not in excluded]
            count, errors = 0, 0
            for row in rows:
                key = (row["db_sha256"], row["reference_sql"])
                if key not in seen:
                    db = Path(row["db_path"])
                    if fingerprint(db) != row["db_sha256"]:
                        raise ValueError("reference database hash changed")
                    try:
                        seen[key] = {"tables_read": require_reference_reads(db, row["reference_sql"]), "error": None}
                    except ValueError as error:
                        seen[key] = {"tables_read": [], "error": str(error)}
                result = seen[key]
                records.write(json.dumps({"source": path.as_posix(), "id": row["id"],
                    "db_sha256": row["db_sha256"], "reference_sha256": hashlib.sha256(row["reference_sql"].encode()).hexdigest(),
                    **result}) + "\n")
                count += 1
                errors += result["error"] is not None
            summary["sources"][path.as_posix()] = {"references": count, "failures": errors, "source_sha256": fingerprint(path)}
    summary.update(unique_references=len(seen), references=sum(s["references"] for s in summary["sources"].values()),
                   failures=sum(s["failures"] for s in summary["sources"].values()),
                   passed=all(s["error"] is None for s in seen.values()))
    atomic_json(out / "summary.json", summary)
    if not summary["passed"]:
        raise ValueError("reference table-read inventory has failures; inspect log, do not drop tasks")
    return summary


def historical_diagnostics(registration: Path, results: Path, out: Path):
    """Never rewrite registered outputs or substitute new scores for historical ones."""
    reg = json.loads(registration.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=False)
    summary = {"registration_id": reg["id"], "registration_sha256": fingerprint(registration),
               "diagnostic_policy": POLICY, "original_scores_unchanged": True, "conditions": {}}
    for step in reg["steps"]:
        if step["action"] != "evaluate":
            continue
        tasks = {r["id"]: r for r in read_jsonl(Path(step["tasks"]))}
        trajectory_path = results / Path(step["out"]).name / "trajectories.jsonl"
        totals = {"episodes": 0, "submissions": 0, "no_table_submissions": 0,
                  "valid_no_table_submissions": 0, "error_no_table_submissions": 0,
                  "accepted_no_table_episodes": 0, "status": "not_run"}
        with (out / (step["name"] + ".jsonl")).open("w", encoding="utf-8") as records:
            for record in read_jsonl(trajectory_path) if trajectory_path.exists() else []:
                meta = record["metadata"]
                task = tasks[meta["source_task_id"]]
                totals["episodes"] += 1
                totals["status"] = "logged"
                accepted_no_table = False
                for call in meta["call_log"]:
                    if call["name"] != "submit_sql":
                        continue
                    diag = submission_reads(Path(task["db_path"]), call)
                    records.write(json.dumps({"trajectory_id": record["trajectory_id"], **diag}) + "\n")
                    totals["submissions"] += 1
                    totals["no_table_submissions"] += diag["no_table_read"]
                    totals["valid_no_table_submissions"] += diag["no_table_read"] and diag["execution_error"] is None
                    totals["error_no_table_submissions"] += diag["no_table_read"] and diag["execution_error"] is not None
                    accepted_no_table |= diag["accepted"] and diag["no_table_read"]
                totals["accepted_no_table_episodes"] += accepted_no_table
        summary["conditions"][step["name"]] = totals
    atomic_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("references", "historical"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    result = reference_inventory(args.out) if args.action == "references" else historical_diagnostics(args.registration, args.results, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
