"""CPU-only amendment: add five distinct dev questions without changing base build artifacts."""
from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from .engine import official_tokenizer
from .sftdata import replay_audit
from .sqalebuild import ROOT, questions_by_schema, write_json, write_lines
from .sqalepool import digest, materialize, prepare_task
from .sqlenv import FALLBACK, SYSTEM_PROMPT
from .sqlsftdata import auditkit_check, encode_episode, make_episode
from .tasks import file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args(argv)
    out = args.build.resolve()
    if (out / "dev-100.tasks.jsonl").exists():
        raise SystemExit("amended dev pool already exists; do not overwrite it")
    original = [json.loads(line) for line in (out / "dev.tasks.jsonl").read_text(encoding="utf-8").splitlines()]
    base_manifest_sha = file_sha256(out / "manifest.json")
    existing = {task["question_id"] for task in original}
    schemas = {task["schema_id"]: task for task in original}
    questions = questions_by_schema(ROOT / "data/sqale/sources", set(schemas))
    ranks = {sid: i for i, sid in enumerate(sorted(schemas, key=lambda sid: digest("dev-topup-v1:" + sid)))}
    source = ROOT / "data/sqale/sources/schemas/data/test-00000-of-00001.parquet"
    # Only the 95 already eligible source schemas can supply additions.
    rows = {}
    for batch in pq.ParquetFile(source).iter_batches(batch_size=1):
        row = batch.to_pylist()[0]
        if row["schema_id"] in schemas:
            rows[row["schema_id"]] = row
    tokenizer = official_tokenizer(str(ROOT / "data/sqale/tokenizer"))
    additions, replays, submissions, exclusions, replay_tasks = [], [], [], [], {}
    with tempfile.TemporaryDirectory(dir=out, prefix="dev-topup-") as temporary:
        for sid in sorted(schemas, key=lambda sid: ranks[sid]):
            if len(original) + len(additions) >= 100:
                break
            db = Path(temporary) / (sid + ".sqlite")
            materialize(rows[sid], db)
            for q in sorted(questions[sid], key=lambda q: digest(q["question_id"])):
                if q["question_id"] in existing:
                    continue
                try:
                    task = prepare_task(q, db, out / "db", "dev", schemas[sid]["schema_group"])
                    task["db_path"] = Path(task["db_path"]).relative_to(ROOT).as_posix()
                    task["db_sha256"] = file_sha256(Path(task["db_path"]))
                    task["schema_question_rank"] = 1
                    ep = make_episode(task)
                    submissions += [{"stage": "dev-topup-builder", "episode_id": ep["id"], **c}
                                    for c in ep["calls"] if c["name"] == "submit_sql"]
                    encoded = encode_episode(ep, tokenizer)
                    if encoded is None:
                        raise ValueError("context_over_4096")
                    issues = replay_audit(ep, None, ep["db_sha256"], tokenizer, SYSTEM_PROMPT,
                                          FALLBACK, sql_db=Path(ep["db_path"]))
                    record = ep.pop("_replay_record")
                    submissions += [{"stage": "dev-topup-replay", "episode_id": ep["id"], **c}
                                    for c in record["metadata"]["call_log"] if c["name"] == "submit_sql"]
                    if issues:
                        raise ValueError("replay_failure: " + "; ".join(issues))
                    additions.append(task)
                    replays.append(record)
                    replay_tasks[ep["id"]] = task
                    existing.add(q["question_id"])
                    break
                except (ValueError, sqlite3.Error, AssertionError) as error:
                    exclusions.append({"question_id": q["question_id"], "reason": str(error)})
    amended = original + additions
    if len(amended) != 100:
        raise ValueError("not enough eligible second questions in existing dev schemas")
    write_lines(out / "dev-100.tasks.jsonl", amended)
    write_lines(out / "dev-topup-replays.jsonl", replays)
    write_lines(out / "dev-topup-submissions.jsonl", submissions)
    write_lines(out / "dev-topup-exclusions.jsonl", exclusions)
    audit = auditkit_check(out / "dev-topup-replays.jsonl", replay_tasks)
    write_json(out / "dev-topup-auditkit.json", audit)
    manifest = {"base_manifest_sha256": base_manifest_sha,
                "original_dev_sha256": file_sha256(out / "dev.tasks.jsonl"),
                "file": "dev-100.tasks.jsonl", "sha256": file_sha256(out / "dev-100.tasks.jsonl"),
                "questions": len(amended), "schema_groups": len({t["schema_group"] for t in amended}),
                "additional_question_ids": [t["id"] for t in additions],
                "auditkit_pass": sum(v == "success" for v in audit["verdicts"].values()),
                "replay_records_sha256": file_sha256(out / "dev-topup-replays.jsonl"),
                "amendment_sha256": file_sha256(ROOT / "experiments/sqale-sft/AMENDMENT-1-data-eligibility.md"),
                "code_sha256": file_sha256(Path(__file__)), "model_runs": 0, "paid_runs": 0}
    write_json(out / "dev-100.manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return int(manifest["auditkit_pass"] != len(additions))


if __name__ == "__main__":
    raise SystemExit(main())
