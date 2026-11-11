"""Gold-derived SQL mistakes, retained only after the intended native SQLite error."""
from __future__ import annotations

import hashlib
from pathlib import Path

import sqlglot
from sqlglot import exp

from .sqlscore import execute

KINDS = ("wrong_column", "wrong_table", "ambiguous_column", "misused_aggregate")
ERRORS = {"wrong_column": "no such column:", "wrong_table": "no such table:",
          "ambiguous_column": "ambiguous column name:", "misused_aggregate": "misuse of aggregate"}


def candidates(tree, kind):
    if kind == "wrong_column":
        for i, column in enumerate(tree.find_all(exp.Column)):
            for name in (column.name + "_code", column.name + "s", column.name[:-1] + "_value"):
                out = tree.copy()
                target = list(out.find_all(exp.Column))[i]
                target.set("this", exp.to_identifier(name, quoted=True))
                yield out
    elif kind == "wrong_table":
        for i, table in enumerate(tree.find_all(exp.Table)):
            for name in (table.name + "s", table.name + "_archive", table.name + "_history"):
                out = tree.copy()
                list(out.find_all(exp.Table))[i].set("this", exp.to_identifier(name, quoted=True))
                yield out
    elif kind == "ambiguous_column":
        # First try a natural join ambiguity by dropping a qualifier.
        for i, column in enumerate(tree.find_all(exp.Column)):
            if column.table:
                out = tree.copy()
                list(out.find_all(exp.Column))[i].set("table", None)
                yield out
        # A duplicated FROM relation is another common query-editing error.
        # Preserve the query structure and make an actual gold column ambiguous.
        for s, select in enumerate(tree.find_all(exp.Select)):
            source = select.args.get("from_")
            if source is None or not isinstance(source.this, exp.Table):
                continue
            table = source.this
            for column in select.find_all(exp.Column):
                if column.table.lower() in {"", table.alias_or_name.lower()}:
                    out = tree.copy()
                    edited = list(out.find_all(exp.Select))[s]
                    duplicate = table.copy()
                    duplicate.set("alias", exp.TableAlias(this=exp.to_identifier(table.alias_or_name + "_copy")))
                    edited.join(duplicate, join_type="CROSS", copy=False)
                    for col in edited.find_all(exp.Column):
                        if col.name == column.name and col.table.lower() in {"", table.alias_or_name.lower()}:
                            col.set("table", None)
                            break
                    yield out
    elif kind == "misused_aggregate":
        for i, aggregate in enumerate(tree.find_all(exp.AggFunc)):
            out = tree.copy()
            target = list(out.find_all(exp.AggFunc))[i]
            target.replace(exp.Sum(this=target.copy()))
            yield out
        for s, select in enumerate(tree.find_all(exp.Select)):
            column = next(select.find_all(exp.Column), None)
            if column is not None:
                out = tree.copy()
                edited = list(out.find_all(exp.Select))[s]
                edited.where(exp.GT(this=exp.Sum(this=column.copy()), expression=exp.Literal.number(0)),
                             append=True, copy=False)
                yield out


def make_mutation(db: Path, gold_sql: str, identifier: str, *, requested: str | None = None) -> dict:
    tree = sqlglot.parse_one(gold_sql, read="sqlite")
    start = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) % len(KINDS)
    desired = requested or KINDS[start]
    order = [desired] + [k for k in KINDS if k != desired]
    attempts = []
    for kind in order:
        seen = set()
        for candidate in candidates(tree, kind):
            query = candidate.sql(dialect="sqlite")
            if query in seen:
                continue
            seen.add(query)
            result = execute(db, query)
            attempts.append({"kind": kind, "query": query, "error": result.error})
            if result.error and ERRORS[kind] in result.error.lower():
                return {"requested_kind": desired, "kind": kind, "query": query,
                        "error": result.error, "attempts": attempts}
            if len(seen) >= 32:
                break
    raise ValueError("no_verified_mutation: " + identifier)
