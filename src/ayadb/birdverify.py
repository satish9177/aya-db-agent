"""Read-only final-only source/result check for the approved BIRD preparation."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from .birdprepare import task_prompt, write
from .promptquality import fingerprint
from .sqlscore import execute, rows_equal


def decode_gold(value):
    if isinstance(value, dict) and set(value) == {"sqlite_blob_base64"}:
        return base64.b64decode(value["sqlite_blob_base64"])
    return value


def verify(root=Path("data/bird")):
    out = root / "final-v1"
    meta = json.loads((out / "manifest.json").read_text())
    downloads = json.loads((root / "downloads.json").read_text())
    source = root / "source" / downloads["pins"]["annotations_file"]
    assert fingerprint(source) == meta["source_sha256"] == downloads["pins"]["annotations_sha256"]
    raw = {r["question_id"]: r for r in map(json.loads, source.read_text(encoding="utf-8").splitlines())}
    for db in downloads["databases"]:
        assert fingerprint(Path(db["db_path"])) == db["db_sha256"]
        assert db["db_sha256"].startswith(db["tag"].rsplit("-", 2)[1])
    for name, sha in meta["output_sha256"].items():
        assert fingerprint(out / name) == sha
    tasks = [json.loads(s) for s in (out / "tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    excluded = [json.loads(s) for s in (out / "exclusions.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [t["source_question_id"] for t in tasks + excluded]
    assert len(ids) == len(set(ids)) == 500 and set(ids) == set(raw)
    for i, task in enumerate(tasks):
        original = raw[task["source_question_id"]]
        assert task["source"] == "bird-mini-dev" and task["dataset_usage"] == "final_only" and task["split"] == "final"
        assert task["reference_sql"] == original["SQL"]
        assert task["prompt"] == task_prompt(original["question"], original["evidence"])
        assert task["question"] == original["question"] and task["evidence"] == original["evidence"]
        result = execute(Path(task["db_path"]), task["reference_sql"])
        assert not result.error, (task["id"], result.error)
        saved = [[decode_gold(v) for v in r] for r in task["gold_rows"]]
        assert rows_equal(result.rows, saved, ordered=task["ordered"]), task["id"]
        if (i + 1) % 50 == 0:
            print(json.dumps({"reference_results_rechecked": i + 1}), flush=True)
    result = {"passed": True, "source_questions": 500, "retained": len(tasks), "excluded": len(excluded),
              "database_hashes": 11, "gold_results_recomputed_and_rechecked": len(tasks),
              "evidence_policy_checked": True, "final_only": True, "model_runs": 0, "paid_runs": 0, "uploads": 0}
    write(out / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    verify(parser.parse_args().root)
