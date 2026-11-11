"""Four-tool SQL submission environment, alongside the unchanged shop environment."""
from __future__ import annotations

import functools
import json
from contextlib import closing
from pathlib import Path

from agenttune.agentic.rollout_engines.base import RolloutEngine

from .sqlscore import execute, quote, readonly

SYSTEM_PROMPT = (
    "You are a database analyst. Answer the question by exploring the read-only SQLite database "
    "with the tools below. Inspect tables and columns, execute SQL, and correct execution errors. "
    "Finish by calling submit_sql with the single query that answers the question. "
    "submit_sql ends the episode. Do not invent tool results or reference answers."
)
FALLBACK = (
    "You can call these tools:\n{tools}\n\n"
    "Reply with only one JSON list containing one tool call and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use only the listed parameters. Tool observations arrive as user turns beginning with "
    "Tool result. Continue exploring as needed, then call submit_sql to finish."
)
RESULT_FORMAT = "Tool result for {name}: {content}"


class SQLTools(list):
    def __init__(self, db_path: Path, task_id: str, discovery_order: list[str] | None = None):
        self.db_path, self.task_id = Path(db_path), task_id
        self.log: list[dict] = []
        self.finished = False
        with closing(readonly(db_path)) as con:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        self.names = discovery_order or names
        if set(self.names) != set(names) or len(self.names) != len(names):
            raise ValueError("discovery order must contain every table once")

        def record(fn):
            @functools.wraps(fn)
            def call(**kwargs):
                entry = {"task_id": task_id, "call_index": len(self.log), "name": fn.__name__,
                         "args": kwargs, "accepted": False, "terminal": self.finished}
                if self.finished:
                    result = "Error: episode already ended by submit_sql."
                else:
                    try:
                        result = fn(**kwargs)
                        entry["accepted"] = fn.__name__ == "submit_sql" and self.finished
                    except TypeError:
                        result = "Error: invalid tool arguments."
                entry.update(result=result, error=result.startswith(("Error", "SQL error")),
                             terminal_after=self.finished)
                self.log.append(entry)
                return result
            return call

        @record
        def list_tables() -> str:
            """List every table in this SQLite database."""
            return "Tables: " + ", ".join(self.names)

        @record
        def describe_table(table: str) -> str:
            """Show one table's columns, foreign keys and up to three sample rows.

            Args:
                table: Exact table name returned by list_tables.
            """
            if table not in self.names:
                return "Error: unknown table. Available tables: " + ", ".join(self.names)
            with closing(readonly(db_path)) as con:
                cols = con.execute(f"PRAGMA table_info({quote(table)})").fetchall()
                fks = con.execute(f"PRAGMA foreign_key_list({quote(table)})").fetchall()
            sample = execute(db_path, f"SELECT * FROM {quote(table)} LIMIT 3")
            # Explicit truncation applies only to agent previews, never scoring.
            text = json.dumps({"table": table, "columns": [[c[1], c[2]] for c in cols],
                               "foreign_keys": [[r[3], r[2], r[4]] for r in fks],
                               "sample_rows": sample.rows}, ensure_ascii=False, default=str)
            return text if len(text) <= 2400 else text[:2400] + " [preview truncated]"

        @record
        def run_sql(query: str) -> str:
            """Run a read-only SQLite SELECT, showing at most twenty rows (preview only).

            Args:
                query: One read-only SQLite SELECT statement.
            """
            # The verifier fetches the full result separately. An overflowing full query
            # is not presented as a successful, complete preview.
            result = execute(db_path, query)
            if result.error:
                return "SQL error: " + result.error
            text = json.dumps({"columns": result.columns, "rows": result.rows[:20],
                               "truncated": len(result.rows) > 20}, ensure_ascii=False, default=str)
            return text if len(text) <= 5000 else text[:5000] + " [preview truncated]"

        @record
        def submit_sql(query: str) -> str:
            """Submit the final SQL query and immediately end this episode.

            Args:
                query: One read-only SQLite SELECT answering the question.
            """
            if not isinstance(query, str) or not query.strip():
                return "Error: query must be a non-empty SQL string."
            self.finished = True
            return "SQL submitted. Episode complete."

        super().__init__([list_tables, describe_table, run_sql, submit_sql])

    def close(self):
        # Every query owns/closes its connection; interface matches the legacy Tools.
        pass


class SubmissionEngine(RolloutEngine):
    """Stops model generation immediately after submission without patching AgentTune.

    AgentTune requests one terminal turn; return it locally and never call the model again.
    Tools also reject any calls after submission within the same generated call batch.
    """
    def __init__(self, engine, sql_tools: SQLTools):
        self.engine, self.sql_tools = engine, sql_tools

    def _get_tokenizer(self):
        return self.engine._get_tokenizer()

    def generate(self, prompts, tools, gen_cfg):
        if self.sql_tools.finished:
            return {"completions": [""], "completions_raw": [""], "logprobs": [None], "metadata": {}}
        return self.engine.generate(prompts, tools, gen_cfg)
