"""Tool feedback "column-hints-v1" (amendment 9): a new SQL tool set beside the unchanged SQLTools.

The four tools keep the same names, arguments and descriptions, so the model sees the same tool
prompt. Only two run_sql observations change:

- a "no such column" error also lists the real columns of the tables the query names, and a
  "no such table" error (or a column error naming no real table) lists the available tables;
- a query that matches an earlier failed run_sql (ignoring whitespace and trailing semicolons) is
  not executed again: the tool repeats the earlier error and asks for a different query.

Scoring is unchanged: refused repeats are logged as failed run_sql calls.
"""
from __future__ import annotations

import functools
import re
from contextlib import closing
from pathlib import Path

from .sqlenv import SQLTools
from .sqlscore import quote, readonly

POLICY = "column-hints-v1"
MAX_COLUMNS_PER_TABLE = 40
MAX_HINT_CHARS = 800


def normalize_query(query: str) -> str:
    return " ".join(query.split()).rstrip("; ").strip()


def named_tables(query: str, names) -> list[str]:
    """Real tables named in the query, in order of first appearance (case-insensitive)."""
    found = []
    for name in names:
        match = re.search(r"(?<![\w$])" + re.escape(name) + r"(?![\w$])", query, flags=re.IGNORECASE)
        if match:
            found.append((match.start(), name))
    return [name for _, name in sorted(found)]


def schema_columns(db_path: Path, names) -> dict[str, list[str]]:
    with closing(readonly(Path(db_path))) as con:
        return {name: [c[1] for c in con.execute(f"PRAGMA table_info({quote(name)})")] for name in names}


def bad_column(error: str) -> str:
    """Bare column name from SQLite's "no such column: alias.name" message."""
    return error.split(":", 1)[1].strip().split(".")[-1].strip('"`[] ')


def column_hint(error: str, query: str, columns: dict[str, list[str]], names) -> str | None:
    if error.startswith("no such column"):
        tables = named_tables(query, names)
        if tables:
            parts = []
            for table in tables:
                cols = columns[table]
                more = ", ..." if len(cols) > MAX_COLUMNS_PER_TABLE else ""
                parts.append(f"{table}({', '.join(cols[:MAX_COLUMNS_PER_TABLE])}{more})")
            text = "Columns of the tables in this query: " + "; ".join(parts)
        else:
            text = "No table in this query exists. Available tables: " + ", ".join(names)
        name = bad_column(error)
        owners = [t for t in names if name and name.lower() in {c.lower() for c in columns[t]}]
        if owners:
            text += f". Column {name} is in: " + ", ".join(owners[:5]) + (", ..." if len(owners) > 5 else "")
    elif error.startswith("no such table"):
        text = "Available tables: " + ", ".join(names)
    else:
        return None
    return text if len(text) <= MAX_HINT_CHARS else text[:MAX_HINT_CHARS] + " [hint truncated]"


class HintedSQLTools(SQLTools):
    feedback = POLICY

    def __init__(self, db_path: Path, task_id: str, discovery_order: list[str] | None = None):
        super().__init__(db_path, task_id, discovery_order)
        self.columns = schema_columns(self.db_path, self.names)
        self.failed: dict[str, str] = {}
        original = self[2]
        if original.__name__ != "run_sql":
            raise RuntimeError("unexpected SQL tool order")

        @functools.wraps(original)
        def run_sql(**kwargs):
            query = kwargs.get("query")
            if self.finished or set(kwargs) != {"query"} or not isinstance(query, str):
                return original(**kwargs)
            key = normalize_query(query)
            if key in self.failed:
                result = ("Error: this exact query already failed with: " + self.failed[key] +
                          ". It was not run again. Write a different query; check the column names first.")
                self.log.append({"task_id": self.task_id, "call_index": len(self.log), "name": "run_sql",
                                 "args": kwargs, "accepted": False, "terminal": False, "result": result,
                                 "error": True, "terminal_after": False, "feedback": "refused_repeat"})
                return result
            result = original(**kwargs)
            entry = self.log[-1]
            if entry["error"] and result.startswith("SQL error: "):
                error = result[len("SQL error: "):]
                hint = column_hint(error, query, self.columns, self.names)
                if hint:
                    result = f"{result}. {hint}"
                    entry.update(result=result, feedback="column_hint")
                self.failed[key] = error
            return result

        self[2] = run_sql


TOOL_SETS = {None: SQLTools, POLICY: HintedSQLTools}


def feedback_counts(log) -> dict:
    return {"column_hints": sum(c.get("feedback") == "column_hint" for c in log),
            "refused_repeats": sum(c.get("feedback") == "refused_repeat" for c in log)}
