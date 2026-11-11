"""Deterministic SQaLe schema grouping, materialization, slicing and eligibility.

No dataset/network/model access occurs on import. Only the caller-supplied SQaLe paths are read.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from .sqlscore import execute, quote, readonly, rows_equal, top_level_ordered


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def statements(script: str):
    """Split at SQLite-complete statement boundaries, including quoted semicolons."""
    start = 0
    for i, char in enumerate(script):
        if char == ";" and sqlite3.complete_statement(script[start:i + 1]):
            yield script[start:i + 1].strip()
            start = i + 1
    if script[start:].strip():
        yield script[start:].strip()


def table_identifiers(ddl: str) -> set[str]:
    pattern = r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?:\"([^\"]+)\"|`([^`]+)`|\[([^\]]+)\]|([\w.]+))"
    return {next(v for v in parts if v).lower().split(".")[-1] for parts in re.findall(pattern, ddl, re.I)}


def schema_groups(inventory: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """Conservative near-duplicate screen: >=90% table-name containment, >=3 shared.

    Connected components stay together. This detects identifier overlap, not arbitrary semantic
    duplicates; disclose it and the fact that only downloaded training schemas are eligible.
    """
    parent = {r["schema_id"]: r["schema_id"] for r in inventory}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tables = {r["schema_id"]: table_identifiers(r["full_schema"]) for r in inventory}
    exact, inverted, links = {}, defaultdict(list), []
    for row in sorted(inventory, key=lambda r: r["schema_id"]):
        sid = row["schema_id"]
        ddl_hash = digest(re.sub(r"\s+", " ", row["full_schema"]).strip().lower())
        candidates = Counter(other for name in tables[sid] for other in inverted[name])
        if ddl_hash in exact:
            candidates[exact[ddl_hash]] = max(len(tables[sid]), 3)
        for other, overlap in sorted(candidates.items()):
            denominator = min(len(tables[sid]), len(tables[other]))
            if (overlap >= 3 and denominator and overlap / denominator >= .9) or exact.get(ddl_hash) == other:
                a, b = root(sid), root(other)
                parent[max(a, b)] = min(a, b)
                links.append({"a": sid, "b": other, "shared_tables": overlap,
                              "containment": overlap / denominator if denominator else 1.0})
        exact[ddl_hash] = sid
        for name in tables[sid]:
            inverted[name].append(sid)
    return {sid: root(sid) for sid in parent}, links


def assign_splits(inventory: list[dict], groups: dict[str, str]) -> dict[str, str]:
    eval_groups = {groups[r["schema_id"]] for r in inventory if r["source_split"] == "test"}
    assigned = {}
    for r in inventory:
        group = groups[r["schema_id"]]
        slot = int(digest("sqale-split-v1:" + group)[:8], 16)
        if r["source_split"] == "train":
            assigned[r["schema_id"]] = "excluded_overlap" if group in eval_groups else ("q2" if slot % 10 == 0 else "train")
        else:
            assigned[r["schema_id"]] = "dev" if slot % 2 == 0 else "final"
    return assigned


def usable_question(value) -> bool:
    if not isinstance(value, str) or len(value.strip()) < 15:
        return False
    clean = value.strip().lower()
    return not (clean in {"...", "<string>", "simple", "moderate", "hard"}
                or re.fullmatch(r"[.\s<>_\-]+", clean)
                or clean.startswith(("<string>", "[insert", "placeholder")))


def materialize(row: dict, path: Path) -> dict:
    """Execute original CREATE statements and insert exact source values, without repairing data.

    If source DDL/data cannot be materialized exactly, reject this schema and log the error.
    Foreign-key enforcement is off (source validity is itself imperfect); no rows are repaired.
    No generated Python or arbitrary source code is evaluated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    con = sqlite3.connect(path)
    try:
        con.execute("PRAGMA trusted_schema=OFF")
        for statement in statements(row["full_schema"]):
            # Ignore comments before CREATE, but do not run any non-DDL statement.
            clean = re.sub(r"^(?:\s|--[^\n]*(?:\n|$)|/\*.*?\*/)*", "", statement, flags=re.S)
            if not re.match(r"CREATE\s+(?:UNIQUE\s+)?(?:TABLE|INDEX)\b", clean, re.I):
                raise ValueError("unsupported_source_ddl")
            con.execute(clean)
        contents = json.loads(row["schema_content"])
        counts = {}
        for table, rows in contents.items():
            counts[table] = len(rows)
            if not rows:
                continue
            cols = [c[1] for c in con.execute(f"PRAGMA table_info({quote(table)})")]
            if not cols:
                raise ValueError("missing_populated_table")
            if any(set(r) - set(cols) for r in rows):
                raise ValueError("unknown_source_columns")
            sql = f"INSERT INTO {quote(table)} ({','.join(quote(c) for c in cols)}) VALUES ({','.join('?' for _ in cols)})"
            values = [[r.get(c) if not isinstance(r.get(c), (list, dict)) else json.dumps(r[c]) for c in cols] for r in rows]
            con.executemany(sql, values)
        con.commit()
        return {"tables": len(table_identifiers(row["full_schema"])), "rows": sum(counts.values())}
    except Exception:
        con.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        con.close()


def slice_tables(db: Path, required: set[str], key: str) -> tuple[list[str], list[str]]:
    with closing(readonly(db)) as con:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        if not required <= set(names):
            raise ValueError("missing_gold_tables")
        graph = {name: set() for name in names}
        for name in names:
            for fk in con.execute(f"PRAGMA foreign_key_list({quote(name)})"):
                if fk[2] in graph:
                    graph[name].add(fk[2])
                    graph[fk[2]].add(name)
    rng = random.Random(int(digest(key)[:16], 16))
    target = min(len(names), rng.randint(max(4, len(required)), 16))
    kept = set(required)
    while len(kept) < target:
        neighbors = sorted(set().union(*(graph[t] for t in kept)) - kept) if kept else []
        options = neighbors or sorted(set(names) - kept)
        kept.add(rng.choice(options))
    ordered = sorted(kept)
    rng.shuffle(ordered)
    return sorted(kept), ordered


def copy_slice(source: Path, target: Path, tables: list[str]) -> None:
    if target.exists():
        return
    with closing(readonly(source)) as src, closing(sqlite3.connect(target)) as dst:
        for table in tables:
            ddl = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
            dst.execute(ddl)
            cols = [r[1] for r in src.execute(f"PRAGMA table_info({quote(table)})")]
            rows = src.execute(f"SELECT {','.join(quote(c) for c in cols)} FROM {quote(table)}").fetchall()
            dst.executemany(f"INSERT INTO {quote(table)} ({','.join(quote(c) for c in cols)}) VALUES ({','.join('?' for _ in cols)})", rows)
        dst.commit()


def prepare_task(question: dict, full_db: Path, out_dir: Path, split: str, group: str) -> dict:
    sql = question["sql"]
    gold = execute(full_db, sql)
    if gold.error:
        raise ValueError("gold_execution: " + gold.error)
    if not gold.rows:
        raise ValueError("empty_result")
    if all(v is None for row in gold.rows for v in row):
        raise ValueError("all_null_result")
    if gold.rows == [(0,)] or gold.rows == [(0.0,)]:
        raise ValueError("single_zero_result")
    reads = {t for t in gold.reads if not t.startswith("sqlite_")}
    if not reads:
        raise ValueError("no_gold_tables")
    if len(reads) > 5:
        raise ValueError("tool_step_limit")
    sid = question["schema_id"]
    kept, order = slice_tables(full_db, reads, sid + ":" + ",".join(sorted(reads)))
    out_dir.mkdir(parents=True, exist_ok=True)
    sliced = out_dir / f"{sid}-{digest(','.join(kept))[:12]}.sqlite"
    copy_slice(full_db, sliced, kept)
    rechecked = execute(sliced, sql)
    if rechecked.error or not rows_equal(rechecked.rows, gold.rows, ordered=top_level_ordered(sql)):
        raise ValueError("sliced_gold_mismatch")
    phrasings = {k: v for k, v in question["questions"].items() if usable_question(v)}
    if not phrasings:
        raise ValueError("no_usable_phrasing")
    # Prefer verbose; all variants remain grouped and available for approved controls.
    phrasing = "verbose" if "verbose" in phrasings else next(iter(phrasings))
    qid = question["question_id"]
    return {"id": qid, "question_id": qid, "schema_id": sid, "schema_group": group,
            "split": split, "difficulty": question["difficulty"], "phrasing": phrasing,
            "prompt": phrasings[phrasing], "phrasings": phrasings, "db_path": str(sliced.as_posix()),
            "reference_sql": sql, "gold_rows": gold.rows, "gold_tables": sorted(reads),
            "slice_tables": kept, "discovery_order": order, "ordered": top_level_ordered(sql)}
