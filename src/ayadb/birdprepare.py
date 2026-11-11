"""Approved local BIRD Mini-Dev final-only preparation. No trajectories or model calls."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from collections import Counter
from contextlib import closing
from pathlib import Path

from .promptquality import fingerprint
from .sqlenv import SQLTools
from .sqlpolicy import ordering_policy
from .sqlscore import ABS_TOL, MAX_RESULT_BYTES, MAX_ROWS, REL_TOL, execute, readonly

REGISTRATION = "bird-final-local-20261107-01"


def task_prompt(question: str, evidence: str) -> str:
    if not isinstance(question, str) or not isinstance(evidence, str):
        raise ValueError("question and evidence must be strings")
    return question + ("\n\nExternal Knowledge:\n" + evidence if evidence else "")


def encode_blob(value):
    if isinstance(value, bytes):
        return {"sqlite_blob_base64": base64.b64encode(value).decode("ascii")}
    raise TypeError(type(value).__name__)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=encode_blob, allow_nan=False) + "\n", encoding="utf-8")


def prepare(root=Path("data/bird")):
    out = root / "final-v1"
    out.mkdir(exist_ok=False)
    downloads = json.loads((root / "downloads.json").read_text())
    pins = downloads["pins"]
    source = root / "source" / pins["annotations_file"]
    if fingerprint(source) != pins["annotations_sha256"]:
        raise ValueError("BIRD annotation hash mismatch")
    databases = {r["db_id"]: r for r in downloads["databases"]}
    for db in databases.values():
        if fingerprint(Path(db["db_path"])) != db["db_sha256"]:
            raise ValueError("BIRD database hash mismatch")
        with closing(readonly(Path(db["db_path"]))) as con:
            if con.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 500 or {r["question_id"] for r in rows} != set(range(500)):
        raise ValueError("expected the pinned 500 aligned Mini-Dev IDs")
    kept, excluded, decisions = [], [], []
    for row in rows:
        identifier = "bird:" + str(row["question_id"])
        meta = {"id": identifier, "source_question_id": row["question_id"], "db_id": row["db_id"],
                "difficulty": row["difficulty"]}
        db = databases[row["db_id"]]
        path = Path(db["db_path"])
        gold = execute(path, row["SQL"])
        reason = gold.error
        if not reason and any(isinstance(v, float) and not math.isfinite(v) for r in gold.rows for v in r):
            reason = "nonfinite_reference_result"
        policy = ordering_policy(path, row["SQL"]) if not reason else None
        if policy and not policy["eligible"]:
            reason = "ordering: " + policy["reason"]
        if reason:
            excluded.append({**meta, "reason": reason, "ordering_policy": policy})
        else:
            task = {**meta, "question_id": identifier, "source": "bird-mini-dev", "dataset_usage": "final_only",
                    "format": "sql-submit-v1", "split": "final", "schema_id": "bird:" + row["db_id"],
                    "schema_group": "bird:" + row["db_id"], "question": row["question"], "evidence": row["evidence"],
                    "prompt": task_prompt(row["question"], row["evidence"]),
                    "evidence_included": bool(row["evidence"]), "reference_sql": row["SQL"],
                    "gold_rows": gold.rows, "gold_columns": gold.columns,
                    "gold_tables": sorted(t for t in gold.reads if t and not t.startswith("sqlite_")),
                    "db_path": path.as_posix(), "db_sha256": db["db_sha256"],
                    "discovery_order": SQLTools(path, identifier).names,
                    "ordered": policy["ordered"], "ordering_policy": policy}
            kept.append(task)
        decisions.append({**meta, "decision": "exclude" if reason else "keep", "reason": reason,
                          "reference_result_rows": len(gold.rows), "ordering_policy": policy})
        if len(decisions) % 25 == 0:
            print(json.dumps({"references_checked": len(decisions), "retained": len(kept), "excluded": len(excluded)}), flush=True)
    for name, values in (("tasks.jsonl", kept), ("exclusions.jsonl", excluded), ("reference-checks.jsonl", decisions)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as target:
            for value in values:
                target.write(json.dumps(value, ensure_ascii=True, default=encode_blob, allow_nan=False) + "\n")
    if any(fingerprint(Path(db["db_path"])) != db["db_sha256"] for db in databases.values()):
        raise ValueError("BIRD database changed during preparation")
    summary = {"registration": REGISTRATION, "format": "sql-submit-v1", "usage": "final_only",
               "source_repo": pins["annotations_repo"], "source_revision": pins["annotations_revision"],
               "source_sha256": fingerprint(source), "license": pins["license"],
               "source_questions": len(rows), "retained": len(kept), "excluded": len(excluded),
               "retained_by_database": dict(Counter(t["db_id"] for t in kept)),
               "retained_by_difficulty": dict(Counter(t["difficulty"] for t in kept)),
               "exclusions_by_reason": dict(Counter(t["reason"] for t in excluded)),
               "exclusions_by_database": dict(Counter(t["db_id"] for t in excluded)),
               "ordering": dict(Counter(t["ordering_policy"]["reason"] for t in kept)),
               "evidence": {"source_nonempty": sum(bool(r["evidence"]) for r in rows),
                            "retained_nonempty": sum(t["evidence_included"] for t in kept),
                            "policy": "verbatim source question; nonempty evidence appended after External Knowledge header"},
               "scorer_caps": {"seconds": 5, "vm_ops": 50_000_000, "rows": MAX_ROWS,
                               "result_bytes": MAX_RESULT_BYTES, "abs_tol": ABS_TOL, "rel_tol": REL_TOL},
               "databases": databases, "databases_unchanged": True, "zero_empty_results_kept": True,
               "model_runs": 0, "training_runs": 0, "paid_runs": 0, "uploads": 0,
               "amendment_sha256": fingerprint(Path("experiments/sqale-sft/AMENDMENT-3-bird-final.md")),
               "output_sha256": {p.name: fingerprint(p) for p in out.glob("*.jsonl")}}
    write(out / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "databases"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/bird"))
    prepare(parser.parse_args().root)
