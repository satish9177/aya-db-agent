"""Query-shape classes for reporting dev results by reference-SQL complexity (amendment 8).

Descriptive only: never a score, gate, reward or selection rule. Classes come from the reference SQL,
parsed with sqlglot (SQLite dialect):

- multi-step: a CTE, or more than one SELECT (subquery, UNION, scalar subquery)
- 3+ tables: one SELECT reading three or more table references
- two tables: one SELECT reading two table references (one join, including self-joins)
- 1-table aggregate: one table with an aggregate, GROUP BY or window function
- 1-table lookup: one table, plain filter/sort

"Complex" means 3+ tables or multi-step.
"""
from __future__ import annotations

import sqlglot
from sqlglot import exp

CLASSES = ("1-table lookup", "1-table aggregate", "two tables", "3+ tables", "multi-step")
COMPLEX = frozenset({"3+ tables", "multi-step"})


def classify(sql: str) -> str:
    tree = sqlglot.parse_one(sql, read="sqlite")
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    selects = list(tree.find_all(exp.Select))
    if ctes or len(selects) > 1:
        return "multi-step"
    tables = [t for t in tree.find_all(exp.Table) if t.name and t.name.lower() not in ctes]
    if len(tables) >= 3:
        return "3+ tables"
    if len(tables) == 2:
        return "two tables"
    grouped = any(s.args.get("group") for s in selects)
    if grouped or any(True for _ in tree.find_all(exp.AggFunc)) or any(True for _ in tree.find_all(exp.Window)):
        return "1-table aggregate"
    return "1-table lookup"


def is_complex(sql: str) -> bool:
    return classify(sql) in COMPLEX
