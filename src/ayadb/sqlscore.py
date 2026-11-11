"""Full-result SQL verification for sql-submit-v1; independent of the legacy scalar metric."""
from __future__ import annotations

import bisect
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

FORMAT = "sql-submit-v1"
ABS_TOL = REL_TOL = 1e-6
MAX_ROWS = 10_000
MAX_RESULT_BYTES = 8 * 1024 * 1024


@dataclass
class QueryResult:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    reads: set[str] = field(default_factory=set)


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def readonly(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True,
                          check_same_thread=False)
    con.execute("PRAGMA query_only=ON")
    con.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
    con.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 100_000)
    return con


def execute(path: Path, query: str, *, row_limit: int = MAX_ROWS, seconds: float = 5.0,
            max_ops: int = 50_000_000, max_bytes: int = MAX_RESULT_BYTES,
            explain: bool = False) -> QueryResult:
    """One read-only statement, separate connection, complete results or an explicit failure."""
    if not isinstance(query, str) or not query.strip() or len(query) > 100_000:
        return QueryResult(error="invalid_query")
    reads: set[str] = set()
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
    forbidden = {"load_extension", "readfile", "writefile", "random", "randomblob"}
    clock_used = False

    def authorizer(action, arg1, arg2, *_):
        nonlocal clock_used
        if action not in allowed:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            name = (arg2 or arg1 or "").lower()
            if name in {"current_time", "current_date", "current_timestamp"}:
                clock_used = True
                return sqlite3.SQLITE_DENY
            if name in forbidden:
                return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            reads.add(arg1)
        return sqlite3.SQLITE_OK

    try:
        con = readonly(path)
    except sqlite3.Error as e:
        return QueryResult(error=str(e))
    # Inspect actual date-function arguments at execution, including expressions.
    # A literal/comment/column named 'now' never calls these wrappers. Delegate
    # accepted arguments back to the same SQLite library to preserve its semantics.
    date_con = None

    def date_function(name):
        def checked(*args):
            nonlocal clock_used, date_con
            positions = (1,) if name == "strftime" else ((0, 1) if name == "timediff" else (0,))
            if any(i >= len(args) or (isinstance(args[i], str) and args[i].lower() in
                                     {"now", "subsec", "subsecond"}) for i in positions):
                clock_used = True
                raise sqlite3.OperationalError("nondeterministic_time")
            if date_con is None:
                date_con = sqlite3.connect(":memory:")
            return date_con.execute("SELECT " + name + "(" + ",".join("?" for _ in args) + ")", args).fetchone()[0]
        return checked

    for name in ("date", "time", "datetime", "julianday", "unixepoch", "strftime", "timediff"):
        con.create_function(name, -1, date_function(name))
    con.set_authorizer(authorizer)
    start, ops = time.perf_counter(), 0

    def guard():
        nonlocal ops
        ops += 1000
        return int(ops > max_ops or time.perf_counter() - start > seconds)

    con.set_progress_handler(guard, 1000)
    try:
        cur = con.execute(("EXPLAIN " if explain else "") + query)
        if cur.description is None:
            return QueryResult(error="no_result", reads=reads)
        rows, n_bytes = [], 0
        while batch := cur.fetchmany(min(64, row_limit + 1)):
            for row in batch:
                n_bytes += sum(len(v) if isinstance(v, bytes) else len(str(v).encode("utf-8")) for v in row)
                if n_bytes > max_bytes:
                    return QueryResult(error="result_byte_limit", reads=reads)
                rows.append(row)
                if len(rows) > row_limit:
                    return QueryResult(error="row_limit", reads=reads)
            if time.perf_counter() - start > seconds:
                return QueryResult(error="timeout", reads=reads)
        return QueryResult(rows=rows, columns=[c[0] for c in cur.description], reads=reads)
    except sqlite3.Error as e:
        return QueryResult(error="nondeterministic_time" if clock_used else str(e), reads=reads)
    finally:
        con.close()
        if date_con is not None:
            date_con.close()


def top_level_ordered(sql: str) -> bool:
    """Detect ORDER BY outside strings, identifiers, comments and nested SELECTs."""
    tokens = re.findall(r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|"
                        r"`(?:``|[^`])*`|\[[^\]]*\]|[()]|[A-Za-z_]+", sql, re.S)
    depth, last = 0, ""
    for tok in tokens:
        if tok.startswith(("--", "/*")):
            continue
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
        elif depth == 0:
            if last == "ORDER" and tok.upper() == "BY":
                return True
            last = tok.upper()
        else:
            last = ""
    return False


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(rows) -> bool:
    return all(not isinstance(v, float) or math.isfinite(v) for row in rows for v in row)


def _cell_equal(a, b) -> bool:
    if _number(a) and _number(b):
        return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b))
    return type(a) is type(b) and a == b


def row_equal(a, b) -> bool:
    return len(a) == len(b) and all(_cell_equal(x, y) for x, y in zip(a, b))


def rows_equal(predicted, gold, *, ordered: bool = False) -> bool:
    """Multiset comparison; tolerance never reuses one gold row for multiple predictions.

    Exact rows cancel first. Residual numerical rows use a maximum bipartite matching, grouped
    by their exact nonnumeric cells. Bound the ambiguous residual graph and fail closed.
    """
    a, b = [tuple(r) for r in predicted], [tuple(r) for r in gold]
    if len(a) != len(b) or not _finite(a) or not _finite(b):
        return False
    if ordered:
        return all(row_equal(x, y) for x, y in zip(a, b))
    ca, cb = Counter(a), Counter(b)
    if ca == cb:
        return True
    # Exact numeric cancellation can consume a row needed by a tolerant match, so keep
    # complete numeric groups; cancel only groups that match exactly in their entirety.
    def groups(rows):
        out = defaultdict(list)
        for row in rows:
            key = tuple(("number",) if _number(v) else (type(v).__name__, v) for v in row)
            out[key].append(row)
        return out
    ga, gb = groups(a), groups(b)
    if ga.keys() != gb.keys():
        return False
    for key, left in ga.items():
        right = gb[key]
        if len(left) != len(right):
            return False
        if Counter(left) == Counter(right):
            continue
        numeric = [i for i, cell in enumerate(key) if cell == ("number",)]
        if not numeric:
            return False
        col = numeric[0]
        right = sorted(right, key=lambda r: r[col])
        values = [r[col] for r in right]
        edges, n_edges = [], 0
        for row in left:
            # Conservative range for relative tolerance near the prediction.
            delta = (ABS_TOL + REL_TOL * abs(row[col])) / (1 - REL_TOL)
            lo, hi = bisect.bisect_left(values, row[col] - delta), bisect.bisect_right(values, row[col] + delta)
            candidates = [j for j in range(lo, hi) if row_equal(row, right[j])]
            n_edges += len(candidates)
            if not candidates or n_edges > 1_000_000:
                return False
            edges.append(candidates)
        matched: dict[int, int] = {}
        for start in range(len(left)):
            queue, parent, seen_right = deque([start]), {start: None}, set()
            endpoint = None
            while queue and endpoint is None:
                i = queue.popleft()
                for j in edges[i]:
                    if j in seen_right:
                        continue
                    seen_right.add(j)
                    if j not in matched:
                        endpoint = (i, j)
                        break
                    nxt = matched[j]
                    if nxt not in parent:
                        parent[nxt] = (i, j)
                        queue.append(nxt)
            if endpoint is None:
                return False
            i, j = endpoint
            while True:
                matched[j] = i
                if parent[i] is None:
                    break
                i, j = parent[i]
    return True


def score_query(db: Path, query: str | None, reference_sql: str) -> dict:
    gold = execute(db, reference_sql)
    if gold.error:
        return {"strict": False, "bird_set": False, "error": "invalid_gold: " + gold.error}
    if query is None:
        return {"strict": False, "bird_set": False, "error": "unsubmitted"}
    result = execute(db, query)
    if result.error:
        return {"strict": False, "bird_set": False, "error": result.error}
    from .sqlpolicy import ordering_policy
    policy = ordering_policy(db, reference_sql)
    if not policy["eligible"]:
        return {"strict": False, "bird_set": False, "error": "ineligible_reference: " + policy["reason"]}
    return {"strict": rows_equal(result.rows, gold.rows, ordered=policy["ordered"]),
            "bird_set": set(result.rows) == set(gold.rows), "error": None,
            "ordering_policy": policy}


def score_episode(db: Path, reference_sql: str, log: list[dict]) -> dict:
    submitted = next((c["args"].get("query") for c in log
                      if c["name"] == "submit_sql" and c.get("accepted")), None)
    last = next((c["args"].get("query") for c in reversed(log)
                 if c["name"] == "run_sql" and not c.get("error")), None)
    strict = score_query(db, submitted, reference_sql)
    fallback = score_query(db, last, reference_sql)
    return {"success": strict["strict"], "strict": strict["strict"],
            "bird_set": strict["bird_set"], "fallback": fallback["strict"],
            "submitted": submitted is not None, "error": strict["error"]}
