"""Approved, CPU-only SQaLe preparation. Never loads, samples or trains a model.

    python -m ayadb.sqalebuild --out data/sqale/build-v1

Only explicit SQaLe source paths and the existing shop training episodes are read. The spent
shop test is never opened. All eligibility decisions are made before any model runs.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import tempfile
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import pyarrow.parquet as pq

from .engine import official_tokenizer
from .env import build_db
from .sftdata import load_episodes, replay_audit
from .sqalepool import (assign_splits, digest, materialize, prepare_task, schema_groups,
                        usable_question)
from .sqlenv import FALLBACK, SYSTEM_PROMPT
from .sqlscore import execute, score_episode
from .sqlsftdata import auditkit_check, encode_episode, make_episode
from .tasks import TRAIN_POOL_SEEDS, check_answer, file_sha256, load_tasks

ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def write_lines(path: Path, values):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=True, default=str) + "\n")


def schema_inventory(sources: Path) -> list[dict]:
    inventory = []
    for split in ("train", "test"):
        for file in sorted((sources / "schemas" / "data").glob(split + "-*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=128, columns=["schema_id", "full_schema"]):
                inventory += [{**row, "source_split": split} for row in batch.to_pylist()]
    return inventory


def questions_by_schema(sources: Path, schema_ids: set[str]) -> dict[str, list[dict]]:
    questions = defaultdict(list)
    columns = ["question_id", "schema_id", "sql", "difficulty", "questions"]
    for split in ("train", "test"):
        for file in sorted((sources / "queries" / "data").glob(split + "-*.parquet")):
            for batch in pq.ParquetFile(file).iter_batches(batch_size=1024, columns=columns):
                for q in batch.to_pylist():
                    if q["schema_id"] in schema_ids:
                        questions[q["schema_id"]].append(q)
    return questions


def converted_shop(out: Path) -> list[tuple[dict, dict]]:
    """All 235 original episodes; replace their final query and scalar answer with gold SQL.

    Earlier executed calls and injected-error flags remain. Last-query replacement handles
    legacy scalar tasks whose original query returned extra rows/columns.
    """
    rows = load_episodes(ROOT / "data" / "sft_pilot" / "episodes.jsonl")
    tasks = {}
    for name in ["train"] + [f"pool{seed}" for seed in TRAIN_POOL_SEEDS]:
        for task in load_tasks(ROOT / "data" / (name + ".jsonl")):
            tasks[task.id] = task
    converted, dbs = [], {}
    shop_dir = out / "shop-db"
    shop_dir.mkdir()
    for row in rows:
        seed = row["db_seed"]
        if seed not in {101, *TRAIN_POOL_SEEDS}:
            raise ValueError("non-training shop database refused")
        if seed not in dbs:
            dbs[seed] = build_db(shop_dir / f"shop-{seed}.sqlite", seed)
        db = dbs[seed]
        if file_sha256(db) != row["db_sha256"]:
            raise ValueError("shop training database hash mismatch")
        old = tasks[row["id"]]
        gold = execute(db, old.reference_sql)
        if gold.error or not gold.rows or not check_answer(str(gold.rows[0][0]), row["gold"], row["kind"]):
            raise ValueError("shop reference result does not agree with stored scalar gold")
        calls = []
        for message in row["messages"]:
            if message["role"] != "assistant" or not message["content"].lstrip().startswith("["):
                continue
            for call in json.loads(message["content"]):
                calls.append({"name": call["tool_name"], "args": call["parameters"],
                              "train": bool(message.get("train"))})
        final_query = max(i for i, c in enumerate(calls) if c["name"] == "run_sql")
        calls[final_query] = {"name": "run_sql", "args": {"query": old.reference_sql}, "train": True}
        calls.append({"name": "submit_sql", "args": {"query": old.reference_sql}, "train": True})
        tid = "shop:" + row["id"]
        task = {"id": tid, "question_id": tid, "schema_id": f"shop-{seed}", "schema_group": "shop",
                "split": "train", "difficulty": row["family"], "prompt": row["prompt"],
                "reference_sql": old.reference_sql, "db_path": db.relative_to(ROOT).as_posix(),
                "gold_rows": gold.rows, "gold_tables": [], "source": "shop-converted"}
        episode = make_episode(task, calls=calls)
        episode["source"] = "shop-converted"
        episode["original_id"] = row["id"]
        converted.append((task, episode))
    return converted


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/sqale/sources")
    parser.add_argument("--tokenizer", default="data/sqale/tokenizer")
    parser.add_argument("--out", default="data/sqale/build-v1")
    parser.add_argument("--train", type=int, default=5000, help="total episodes including the 235 shop conversions")
    parser.add_argument("--q2", type=int, default=600)
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--final", type=int, default=150)
    parser.add_argument("--audit-cache", default=None,
                        help="reuse matching deterministic expert replays from a prior preparation, never model trajectories")
    args = parser.parse_args(argv)
    out, sources = ROOT / args.out, ROOT / args.sources
    if out.exists():
        raise SystemExit("output exists: use a new build name; never silently overwrite a preparation")
    out.mkdir(parents=True)
    # All downloaded bytes, tokenizer and code must be identifiable before model work.
    source_manifest = json.loads((sources / "downloads.json").read_text(encoding="utf-8"))
    for item in source_manifest:
        path = sources / item["source"] / item["file"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("source file checksum mismatch")
    tokenizer = official_tokenizer(str(ROOT / args.tokenizer))
    inventory = schema_inventory(sources)
    groups, links = schema_groups(inventory)
    assigned = assign_splits(inventory, groups)
    questions = questions_by_schema(sources, {r["schema_id"] for r in inventory})
    write_json(out / "schema-splits.json", {"assigned": assigned, "groups": groups, "near_duplicate_links": links})
    print(json.dumps({"inventory_schemas": len(inventory), "schema_splits": dict(Counter(assigned.values()))}), flush=True)
    candidates, episodes, excluded = defaultdict(list), {}, Counter()
    records, submissions, failures = (out / "replay-records.jsonl").open("w", encoding="utf-8"), (out / "submissions.jsonl").open("w", encoding="utf-8"), (out / "exclusions.jsonl").open("w", encoding="utf-8")
    replay_tasks = {}
    token_counts = {}
    cached_records, reused = {}, 0
    cache_sha = None
    if args.audit_cache:
        cache_path = ROOT / args.audit_cache
        cache_sha = file_sha256(cache_path)
        with cache_path.open(encoding="utf-8") as cache_file:
            for line in cache_file:
                record = json.loads(line)
                if record.get("metadata", {}).get("format") == "sql-submit-v1":
                    cached_records[record["trajectory_id"]] = record

    def reject(scope, identifier, reason):
        excluded[reason.split(":", 1)[0]] += 1
        failures.write(json.dumps({"scope": scope, "id": identifier, "reason": reason[:400]}) + "\n")

    def log_submissions(ep, stage):
        for call in ep["calls"]:
            if call["name"] == "submit_sql":
                submissions.write(json.dumps({"stage": stage, "episode_id": ep["id"], **call}) + "\n")

    def check_episode(task, ep, keep_record):
        nonlocal reused
        log_submissions(ep, "builder")
        example = encode_episode(ep, tokenizer)
        if example is None:
            raise ValueError("context_over_4096")
        if keep_record:
            from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema
            from .sftdata import conversation, system_message
            from .sqlenv import SQLTools
            cached = cached_records.pop(ep["id"], None)
            record, issues = None, []
            if cached:
                schemas = [_build_tool_schema(fn) for fn in SQLTools(Path(ep["db_path"]), ep["task_id"], ep.get("discovery_order"))]
                expected = conversation(ep["prompt"], ep["messages"], system_message(SYSTEM_PROMPT, FALLBACK, schemas))
                prior = cached["metadata"].get("conversation", [])
                normal = lambda conv: [(m["role"], m.get("content")) for m in conv]
                calls = lambda log: [(c["name"], c["args"], c["result"]) for c in log]
                if (normal(prior[:len(expected)]) == normal(expected)
                        and calls(cached["metadata"]["call_log"]) == calls(ep["calls"])
                        and score_episode(Path(ep["db_path"]), ep["reference_sql"], ep["calls"])["success"]):
                    record = cached
                    record["metadata"]["replay_cache"] = {"file": args.audit_cache, "sha256": cache_sha}
                    reused += 1
            if record is None:
                issues = replay_audit(ep, None, ep["db_sha256"], tokenizer, SYSTEM_PROMPT, FALLBACK,
                                      sql_db=Path(ep["db_path"]))
                record = ep.pop("_replay_record", None)
            if record:
                records.write(json.dumps(record) + "\n")
                for c in record["metadata"]["call_log"]:
                    if c["name"] == "submit_sql":
                        submissions.write(json.dumps({"stage": "AgentTune-replay", "episode_id": ep["id"], **c}) + "\n")
            if issues:
                raise ValueError("replay_failure: " + "; ".join(issues))
            replay_tasks[ep["id"]] = task
        ep["n_input_tokens"] = len(example["input_ids"])
        ep["n_target_tokens"] = sum(t != -100 for t in example["labels"])
        token_counts[ep["id"]] = (ep["n_input_tokens"], ep["n_target_tokens"])
        return ep

    for task, ep in converted_shop(out):
        check_episode(task, ep, True)
        candidates["shop"].append(task)
        episodes[task["id"]] = ep
    print("converted and replay-audited 235 shop episodes", flush=True)
    schema_count = 0
    seen_queries = set()
    try:
        with tempfile.TemporaryDirectory(dir=out, prefix="materialize-") as temporary:
            for source_split in ("train", "test"):
                for file in sorted((sources / "schemas" / "data").glob(source_split + "-*.parquet")):
                    for batch in pq.ParquetFile(file).iter_batches(batch_size=1):
                        row = batch.to_pylist()[0]
                        sid, split = row["schema_id"], assigned[row["schema_id"]]
                        schema_count += 1
                        if split == "excluded_overlap":
                            reject("schema", sid, "near_duplicate_eval_overlap")
                            continue
                        full_db = Path(temporary) / (sid + ".sqlite")
                        try:
                            materialize(row, full_db)
                        except (sqlite3.Error, ValueError, TypeError, OverflowError) as error:
                            reject("schema", sid, "materialization: " + str(error))
                            continue
                        eligible_for_schema = 0
                        for q in sorted(questions.get(sid, []), key=lambda q: digest(q["question_id"])):
                            key = (sid, " ".join(q["sql"].split()).lower())
                            if key in seen_queries:
                                reject("question", q["question_id"], "duplicate_sql_within_schema")
                                continue
                            seen_queries.add(key)
                            for phrasing, text in q["questions"].items():
                                if not usable_question(text):
                                    reject("phrasing", q["question_id"] + ":" + phrasing, "placeholder_phrasing")
                            try:
                                task = prepare_task(q, full_db, out / "db", split, groups[sid])
                                task["db_path"] = Path(task["db_path"]).relative_to(ROOT).as_posix()
                                task["db_sha256"] = file_sha256(Path(task["db_path"]))
                                repair = split == "train" and int(digest(q["question_id"])[-8:], 16) % 10 < 3
                                if repair and len(task["gold_tables"]) > 4:
                                    repair = False
                                    reject("repair_variant", q["question_id"], "repair_tool_budget_use_gold_path")
                                ep = make_episode(task, repair=repair)
                                check_episode(task, ep, split == "train")
                                task["schema_question_rank"] = eligible_for_schema
                                candidates[split].append(task)
                                eligible_for_schema += 1
                                if split == "train":
                                    episodes[task["id"]] = ep
                                # Prefer one final question/schema; a second is available
                                # only if source materialization leaves fewer than 150 schemas.
                                if split == "dev" or (split == "final" and eligible_for_schema >= 2):
                                    break
                            except (ValueError, sqlite3.Error, AssertionError) as error:
                                reject("question", q["question_id"], str(error))
                        full_db.unlink(missing_ok=True)
                        if schema_count % 25 == 0:
                            print(json.dumps({"schemas_processed": schema_count,
                                              "eligible": {k: len(v) for k, v in candidates.items()},
                                              "exclusions": dict(excluded)}), flush=True)
        selected = {}
        for split, count in {"train": max(0, args.train - len(candidates["shop"])), "q2": args.q2,
                             "dev": args.dev, "final": args.final}.items():
            # Stable structure/difficulty balance before any model outcome is available.
            chosen = []
            phases = ([t for t in candidates[split] if t["schema_question_rank"] == 0],
                      [t for t in candidates[split] if t["schema_question_rank"] > 0]) if split == "final" else (candidates[split],)
            for phase in phases:
                by_difficulty = defaultdict(list)
                for task in sorted(phase, key=lambda t: digest(t["id"])):
                    by_difficulty[task["difficulty"]].append(task)
                queues = [iter(v) for _, v in sorted(by_difficulty.items())]
                while queues and len(chosen) < count:
                    alive = []
                    for queue in queues:
                        item = next(queue, None)
                        if item is not None:
                            chosen.append(item)
                            alive.append(queue)
                            if len(chosen) == count:
                                break
                    queues = alive
            selected[split] = chosen
        training = [episodes[t["id"]] for t in candidates["shop"] + selected["train"]]
        rng = random.Random(20261107)
        rng.shuffle(training)
        for split, tasks in selected.items():
            write_lines(out / (split + ".tasks.jsonl"), tasks)
        write_lines(out / "shop.tasks.jsonl", candidates["shop"])
        write_lines(out / "train.episodes.jsonl", training)
        # Every stage contains the same 235 shop conversions; SQaLe membership is nested.
        shop_episodes = [episodes[t["id"]] for t in candidates["shop"]]
        sqale_episodes = [episodes[t["id"]] for t in selected["train"]]
        stage_counts = {}
        for n in (500, 1500, 5000):
            if len(training) < n:
                continue
            stage = shop_episodes + sqale_episodes[:max(0, n - len(shop_episodes))]
            random.Random(20261107).shuffle(stage)
            write_lines(out / f"train-{n}.episodes.jsonl", stage)
            stage_counts[str(n)] = {"episodes": len(stage), "input_tokens": sum(e["n_input_tokens"] for e in stage),
                                    "target_tokens_per_pass": sum(e["n_target_tokens"] for e in stage)}
    finally:
        records.close()
        submissions.close()
        failures.close()
    audit = auditkit_check(out / "replay-records.jsonl", replay_tasks)
    write_json(out / "auditkit.json", audit)
    manifest = {"format": "sql-submit-v1", "model_runs": 0, "paid_runs": 0,
                "sources": source_manifest, "tokenizer": json.loads((ROOT / args.tokenizer / "source.json").read_text()),
                "versions": {name: version(name) for name in ("agenttune", "auditkit", "transformers", "pyarrow")},
                "eligible_counts": {k: len(v) for k, v in candidates.items()},
                "selected_counts": {k: len(v) for k, v in selected.items()}, "shop_converted": len(shop_episodes),
                "selected_schema_groups": {k: len({t["schema_group"] for t in v}) for k, v in selected.items()},
                "requested": {"train_episodes": args.train, "q2": args.q2, "dev": args.dev, "final": args.final},
                "train_episodes": len(training), "train_repair_episodes": sum(e["repair"] for e in training),
                "stages": stage_counts, "exclusions": dict(excluded), "near_duplicate_links": len(links),
                "replay_audited": len(replay_tasks), "reused_deterministic_replays": reused,
                "auditkit_pass": sum(v == "success" for v in audit["verdicts"].values()),
                "max_input_tokens": max(e["n_input_tokens"] for e in training),
                "sealed_shop_test": "not read, generated, evaluated or selected on",
                "perturbation": "preregistered secondary diagnostic; not run",
                "limitations": ["Only the first of sixteen training-schema shards was downloaded.",
                                "Custom SQLite materializer rejects unsupported source DDL/data without repairs.",
                                "Near-duplicate screening uses >=90% table-name containment, >=3 shared tables, plus exact DDL.",
                                "Source foreign-key errors are inherited; no data repair or perturbation was done."]}
    manifest["output_sha256"] = {p.name: file_sha256(p) for p in sorted(out.glob("*.jsonl"))}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"sources", "tokenizer", "output_sha256"}}, indent=2), flush=True)
    return 0 if manifest["auditkit_pass"] == manifest["replay_audited"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
