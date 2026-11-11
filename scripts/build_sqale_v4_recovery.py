"""Build data-build-v4/train-1500 (amendment 10) from the frozen v3 file; free and local, no model calls.

Same 1,500 rows, same order and questions as v3 train-1500 (so the seed-42 split is the same). Only the
demonstrated tool path changes for two groups; everything else is copied byte-for-byte as JSON rows:

- repair rows: a realistic column mistake (sqlmutate_v4) as untrained context, then describe_table on
  the table that owns the real column, then the correct query and submission ("recheck" repair). If
  that does not fit the eight tool steps, the re-check is left out ("plain" repair). If no realistic
  mistake exists, the v3 row is kept.
- complex gold rows (3+ tables or multi-step): run the first CTE body or first uncorrelated subquery on
  its own, then the full query, then submit ("stepwise"), when that part runs and returns rows.

Every changed episode is executed, re-encoded (no truncation, at most 4,096 tokens, else the v3 row is
kept) and replayed through AgentTune's real parser/renderer before it is written.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.queryshape import is_complex
from ayadb.runinputs import read_jsonl
from ayadb.runledger import atomic_json
from ayadb.sqlmutate_v4 import first_step_query, make_column_mutation
from ayadb.sqlsftdata import MAX_STEPS, encode_episode, make_episode, replay_sql_audit

V3 = Path("data/sqale/data-build-v3/train-1500.episodes.jsonl")
V3_SHA = "0c920dab1c5b09256ef7f9c76b4864f7e075ce72db3ca14d4dc978849b46b62e"
DEV = Path("data/sqale/data-build-v3/dev.tasks.jsonl")
OUT = Path("data/sqale/data-build-v4")
TOKENIZER = "data/sqale/tokenizer"
MAX_TOKENS = 4096


def task_of(row):
    return {"id": row["task_id"], "db_path": row["db_path"], "discovery_order": row.get("discovery_order"),
            "schema_id": row["schema_id"], "question_id": row["question_id"], "phrasing": row.get("phrasing", "verbose"),
            "prompt": row["prompt"], "reference_sql": row["reference_sql"]}


def exploration(row):
    """The v3 row's list_tables/describe_table calls, in order."""
    return [{"name": c["name"], "args": c["args"], "train": True} for c in row["calls"]
            if c["name"] in ("list_tables", "describe_table") and not c["error"]]


def finish(row):
    ref = row["reference_sql"]
    return [{"name": "run_sql", "args": {"query": ref}, "train": True},
            {"name": "submit_sql", "args": {"query": ref}, "train": True}]


def build(row, tokenizer, plan, extra):
    ep = make_episode(task_of(row), repair=row["repair"], calls=plan)
    if ep["id"] != row["id"]:
        raise ValueError("episode identity changed: " + row["id"])
    encoded = encode_episode(ep, tokenizer, MAX_TOKENS)
    if encoded is None:
        return None
    ep.update(n_input_tokens=len(encoded["input_ids"]), n_target_tokens=sum(v != -100 for v in encoded["labels"]), **extra)
    problems = replay_sql_audit(ep, Path(row["db_path"]), row["db_sha256"], tokenizer)
    ep.pop("_replay_record", None)
    if problems:
        raise ValueError(row["id"] + ": " + "; ".join(problems))
    return ep


def main():
    if fingerprint(V3) != V3_SHA:
        raise ValueError("v3 train-1500 changed")
    if (OUT / "train-1500.episodes.jsonl").exists():
        raise ValueError("v4 output exists; never rebuild over it")
    rows = read_jsonl(V3)
    dev_schemas = {t["schema_id"] for t in read_jsonl(DEV)}
    if any(r["schema_id"] in dev_schemas for r in rows):
        raise ValueError("training row shares a schema with dev")
    tokenizer = official_tokenizer(TOKENIZER)
    out, log, counts = [], [], Counter()
    for row in rows:
        new = None
        if row.get("source") == "shop-converted":
            counts["shop unchanged"] += 1
        elif row["repair"]:
            try:
                m = make_column_mutation(Path(row["db_path"]), row["reference_sql"], row["id"])
            except ValueError as error:
                counts["repair kept v3 (no realistic mistake)"] += 1
                log.append({"id": row["id"], "change": None, "reason": str(error)})
            else:
                bad = {"name": "run_sql", "args": {"query": m["query"]}, "train": False}
                recheck = {"name": "describe_table", "args": {"table": m["owner_table"]}, "train": True}
                base = exploration(row)
                for style, plan in (("recheck", base + [bad, recheck] + finish(row)), ("plain", base + [bad] + finish(row))):
                    if len(plan) > MAX_STEPS:
                        continue
                    new = build(row, tokenizer, plan, {"repair_kind": "v4_" + m["kind"], "repair_error": m["error"],
                                                       "v4_change": style + "_repair", "v4_owner_table": m["owner_table"]})
                    if new is not None:
                        break
                counts[f"repair -> {new['v4_change']}" if new else "repair kept v3 (too long)"] += 1
                log.append({"id": row["id"], "change": new["v4_change"] if new else None, **m})
        elif is_complex(row["reference_sql"]):
            part = first_step_query(Path(row["db_path"]), row["reference_sql"])
            plan = exploration(row) + ([{"name": "run_sql", "args": {"query": part}, "train": True}] if part else []) + finish(row)
            if part and len(plan) <= MAX_STEPS:
                new = build(row, tokenizer, plan, {"v4_change": "stepwise"})
            counts["complex -> stepwise" if new else "complex kept v3"] += 1
            log.append({"id": row["id"], "change": "stepwise" if new else None, "first_step": part})
        else:
            counts["gold unchanged"] += 1
        out.append(new if new is not None else row)
    if [r["id"] for r in out] != [r["id"] for r in rows]:
        raise ValueError("row order or identity changed")
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "train-1500.episodes.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for r in out:
            target.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "changes.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for entry in log:
            target.write(json.dumps(entry, ensure_ascii=False) + "\n")
    manifest = {"source": V3.as_posix(), "source_sha256": V3_SHA, "output": path.as_posix(),
                "output_sha256": fingerprint(path), "rows": len(out), "counts": dict(counts),
                "repair_kinds": dict(Counter(r.get("repair_kind") for r in out if r.get("repair"))),
                "input_tokens": sum(r["n_input_tokens"] for r in out), "target_tokens": sum(r["n_target_tokens"] for r in out),
                "max_input_tokens": max(r["n_input_tokens"] for r in out), "model_calls": 0}
    atomic_json(OUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
