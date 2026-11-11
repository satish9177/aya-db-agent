"""Amendment 10: realistic column mistakes for repair demos, beside the unchanged v3 sqlmutate.

The v3 mutations used artificial quoted names ("lane_number_code"). A1500's real dev errors were
plausible names instead: typos, naming-style slips, a name from another table, or a real column on the
wrong table alias. Each mutation here edits one column of the gold query, keeps the rest unchanged,
and is kept only if SQLite really answers "no such column". It also names the table that owns the
real column, so the demo can re-check that table before the fix.
"""
from __future__ import annotations

import hashlib
import re
from contextlib import closing
from pathlib import Path

import sqlglot
from sqlglot import exp

from .sqlscore import execute, quote, readonly

KINDS = ("typo", "naming_style", "other_table_name", "wrong_alias")


def table_columns(db: Path) -> dict[str, list[str]]:
    with closing(readonly(Path(db))) as con:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {n: [c[1] for c in con.execute(f"PRAGMA table_info({quote(n)})")] for n in names}


def _snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(p[:1].upper() + p[1:] for p in rest)


def name_variants(name: str, kind: str, others: list[str]):
    if kind == "typo" and len(name) >= 5:
        for i in range(2, len(name) - 1):
            if name[i].isalpha() and name[i] != name[i - 1]:
                yield name[:i] + name[i + 1:]
    elif kind == "naming_style":
        for candidate in (_snake(name), _camel(name), name.replace("_", ""),
                          name[:-3] if name.lower().endswith("_id") else name + "_id",
                          name.split("_", 1)[1] if "_" in name else "", ):
            if candidate:
                yield candidate
    elif kind == "other_table_name":
        tokens = {t for t in re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", name.lower()) if len(t) >= 3}
        shared = [o for o in others if tokens & set(re.split(r"[_\W]+|(?<=[a-z])(?=[A-Z])", o.lower()))]
        yield from shared or others


def _real_aliases(tree, columns):
    """alias (lower) -> real table name, for real tables in the query."""
    lookup = {n.lower(): n for n in columns}
    aliases = {}
    for table in tree.find_all(exp.Table):
        real = lookup.get(table.name.lower())
        if real:
            aliases[table.alias_or_name.lower()] = real
    return aliases


def _owner(column, aliases, columns):
    if column.table:
        real = aliases.get(column.table.lower())
        return real if real and column.name.lower() in {c.lower() for c in columns[real]} else None
    owners = {t for t in aliases.values() if column.name.lower() in {c.lower() for c in columns[t]}}
    return owners.pop() if len(owners) == 1 else None


def make_column_mutation(db: Path, gold_sql: str, identifier: str) -> dict:
    """Deterministic (by identifier) realistic column mistake verified by a native SQLite error."""
    columns = table_columns(db)
    tree = sqlglot.parse_one(gold_sql, read="sqlite")
    aliases = _real_aliases(tree, columns)
    every = {c.lower() for cols in columns.values() for c in cols}
    digest = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
    order = [KINDS[(digest + k) % len(KINDS)] for k in range(len(KINDS))]
    targets = [(i, c) for i, c in enumerate(tree.find_all(exp.Column)) if _owner(c, aliases, columns)]
    if not targets:
        raise ValueError("no_owned_column: " + identifier)
    start = digest % len(targets)
    targets = targets[start:] + targets[:start]
    for kind in order:
        for index, column in targets:
            owner = _owner(column, aliases, columns)
            own = {c.lower() for c in columns[owner]}
            if kind == "wrong_alias":
                if not column.table:
                    continue
                choices = [a for a, t in aliases.items() if t != owner and column.name.lower() not in
                           {c.lower() for c in columns[t]}]
                edits = [("table", a) for a in choices]
            else:
                others = sorted(c for c in every if c not in own)
                edits = [("this", v) for v in name_variants(column.name, kind, others)
                         if v and v.lower() not in own and v.lower() != column.name.lower()]
            for field, value in edits:
                out = tree.copy()
                target = list(out.find_all(exp.Column))[index]
                target.set(field, exp.to_identifier(value))
                query = out.sql(dialect="sqlite")
                result = execute(Path(db), query)
                if result.error and result.error.lower().startswith("no such column"):
                    return {"kind": kind, "query": query, "error": result.error, "owner_table": owner,
                            "original_column": column.name, "edited": f"{field}={value}"}
    raise ValueError("no_verified_column_mutation: " + identifier)


def first_step_query(db: Path, gold_sql: str) -> str | None:
    """For a multi-step query: the first CTE body, or the first uncorrelated subquery, if it runs on its own."""
    tree = sqlglot.parse_one(gold_sql, read="sqlite")
    ctes = list(tree.find_all(exp.CTE))
    candidates = [ctes[0].this] if ctes else [s for s in tree.find_all(exp.Select) if s is not tree]
    for select in candidates:
        query = select.sql(dialect="sqlite")
        result = execute(Path(db), query)
        if not result.error and result.rows and query.strip() != gold_sql.strip():
            return query
    return None
