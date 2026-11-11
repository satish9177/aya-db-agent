"""Expert trajectories: correct, executed multi-turn episodes built from the real environment.

Every observation comes from running the call against the task's own database; nothing is
written by hand. A plan per family does schema discovery (list, describe the tables it needs),
dependent lookups (resolve a name to its key, then query by key), unit conversion in SQL
(cents to dollars), and a final `ANSWER:` line formatted from the query result. A share of
episodes contains one realistic mistake (a guessed column or table name) followed by the
genuine error and a recovery; the mistaken turn is context only and is never a training target.

Two kinds of mistake, one per arm of the recovery experiment (experiments/recovery/):
- discovery (the pilot): a name guessed before the schema is read; recovery = describe the table.
- query (`QueryMistake`): after the schema is read, a query fails (undefined alias, wrong column,
  ambiguous join key, missing join, syntax); recovery = the repaired query, no schema lookup. A
  share of these fail twice: the first repair has another mistake.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field

from .env import make_tools
from .tasks import MONTHS, REVENUE, Task, check_answer

RESULT_FORMAT = "Tool result for {name}: {content}"

# Common wrong guesses a model makes before reading the schema.
_WRONG_COLUMNS = [("unit_price_cents", "price"), ("full_name", "name"), ("cust_id", "customer_id"),
                  ("ol.qty", "ol.quantity"), ("title", "name"), ("opened_at", "created_at"),
                  ("order_date", "date")]
_WRONG_TABLES = {"customers": "customer", "orders": "order_table", "products": "product",
                 "order_lines": "order_items", "tickets": "support_tickets"}

# Query-phase wrong column names (arm B). Disjoint from the schema-drift renames (tasks.DRIFT_RENAMES).
QUERY_WRONG_COLUMNS = {"unit_price_cents": "price_cents", "full_name": "customer_name", "order_date": "order_day",
                       "opened_at": "created_at", "resolved_at": "closed_at", "qty": "quantity",
                       "title": "product_name", "city": "city_name", "segment": "customer_segment",
                       "topic": "ticket_topic", "priority": "ticket_priority"}
_TABLE_COLUMNS = {"customers": ("cust_id", "full_name", "city", "segment", "signup_date"),
                  "products": ("sku", "title", "category", "unit_price_cents"),
                  "orders": ("order_no", "cust_id", "order_date", "status"),
                  "order_lines": ("order_no", "sku", "qty", "discount_pct"),
                  "tickets": ("ticket_id", "cust_id", "opened_at", "priority", "topic", "resolved_at")}
_HOPS = {"customer_shipped_year": 2, "distinct_buyers": 2}  # queries per plan; other families: 1


@dataclass
class Call:
    tool: str
    args: dict
    train: bool = True  # False: a deliberate mistake kept as context, never a training target


def _tables(sql: str) -> dict[str, str]:
    """{alias, or the table name if it has none: table} for the FROM/JOIN tables of `sql`."""
    found = re.findall(r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?!WHERE\b|JOIN\b|ON\b|GROUP\b|ORDER\b)([a-z]{1,2})\b)?", sql)
    return {a or t: t for t, a in found}


def _bad_alias(sql: str, rng: random.Random) -> str:
    """A column qualified with an alias the query never defines."""
    tables = _tables(sql)
    refs = list(re.finditer(r"\b([a-z]{1,2})\.(\w+)", sql))
    if refs:
        m = rng.choice(refs)
        new = next(a for a in ("r", "x", "s", "l") if a not in tables)
        return sql[:m.start()] + f"{new}.{m.group(2)}" + sql[m.end():]
    (table,) = tables.values()  # one table, no alias: qualify a WHERE column with the table's initial
    w = sql.index(" WHERE ")
    m = rng.choice([m for m in re.finditer(r"\b\w+\b", sql[w:]) if m.group(0) in _TABLE_COLUMNS[table]])
    return sql[:w + m.start()] + f"{table[0]}.{m.group(0)}" + sql[w + m.end():]


def _bad_column(sql: str, rng: random.Random) -> str | None:
    """A column name that is not in the schema although the table was described."""
    cols = [c for c in QUERY_WRONG_COLUMNS if re.search(rf"\b{c}\b", sql)]
    if not cols:
        return None
    c = rng.choice(cols)
    return re.sub(rf"\b{c}\b", QUERY_WRONG_COLUMNS[c], sql, count=1)


def _ambiguous(sql: str, rng: random.Random) -> str | None:
    """A join key left unqualified although two of the joined tables have it."""
    tables = set(_tables(sql).values())
    refs = [m for m in re.finditer(r"\b[a-z]{1,2}\.(cust_id|order_no|sku)\b", sql)
            if sum(m.group(1) in _TABLE_COLUMNS[t] for t in tables) > 1]
    if not refs:
        return None
    m = rng.choice(refs)
    return sql[:m.start()] + m.group(1) + sql[m.end():]


def _missing_join(sql: str, rng: random.Random) -> str | None:
    """One JOIN dropped while its alias is still used elsewhere in the query."""
    joins = list(re.finditer(r" JOIN \w+ [a-z]{1,2} ON \S+ = \S+", sql))
    if not joins:
        return None
    m = rng.choice(joins)
    return sql[:m.start()] + sql[m.end():]


def _syntax(sql: str, rng: random.Random) -> str:
    """A typo that breaks the statement: a misspelt keyword or a missing closing parenthesis."""
    if ")" not in sql or rng.random() < 0.5:
        kw = rng.choice([k for k in ("SELECT", "FROM", "WHERE") if f"{k} " in sql])
        return sql.replace(f"{kw} ", {"SELECT": "SELCT ", "FROM": "FORM ", "WHERE": "WHER "}[kw], 1)
    i = sql.rindex(")")
    return sql[:i] + sql[i + 1:]


QUERY_MISTAKES = {"alias": _bad_alias, "column": _bad_column, "ambiguous": _ambiguous,
                  "missing_join": _missing_join, "syntax": _syntax}


@dataclass
class QueryMistake:
    """Arm B: the plan's query number `hop` fails once, or twice (`twice`), before the right query."""
    hop: int
    twice: bool
    seed: int
    kinds: list[str] = field(default_factory=list)  # set when the failing queries are made

    def bad(self, sql: str) -> list[str]:
        rng = random.Random(self.seed)
        options = [k for k, f in QUERY_MISTAKES.items() if f(sql, random.Random(0)) is not None]
        self.kinds = rng.sample(options, 2 if self.twice else 1)
        return [QUERY_MISTAKES[k](sql, rng) for k in self.kinds]


def query_mistake_schedule(schedule: list, families: list[str], seed: int,
                           twice_share: float = 1 / 3) -> list[QueryMistake | None]:
    """Arm B's mistakes, at exactly the episodes where `schedule` (the pilot's) has one."""
    rng = random.Random(seed)
    return [None if m is None else QueryMistake(rng.randrange(_HOPS.get(f, 1)), rng.random() < twice_share,
                                                rng.randrange(2**31))
            for m, f in zip(schedule, families)]


@dataclass
class Episode:
    task: Task
    messages: list[dict] = field(default_factory=list)  # assistant/user turns after the task
    calls: list[dict] = field(default_factory=list)     # executed calls with their observations
    answer: str | None = None
    mistake: str | QueryMistake | None = None


def call_text(c: Call) -> str:
    """The exact assistant turn for one call, in the tools_fallback_prompt format."""
    return json.dumps([{"tool_name": c.tool, "parameters": c.args}])


def first_value(observation: str) -> str:
    rows = observation.split("\n")[1:]
    return rows[0].split(" | ")[0] if rows else ""


def _params(task: Task) -> dict:
    q = task.prompt
    pats = {
        "segment_city_count": r"in the '(?P<seg>[^']+)' segment are based in (?P<city>[A-Za-z]+)\?",
        "product_price": r"of the product '(?P<title>[^']+)'",
        "customer_shipped_year": r"did (?P<name>[A-Za-z ]+) place in (?P<year>\d{4})\?",
        "month_revenue": r"placed in (?P<month>[A-Za-z]+) (?P<year>\d{4})\?",
        "top_category_city": r"customers based in (?P<city>[A-Za-z]+)\?",
        "ticket_top_customer": r"most '(?P<prio>[a-z]+)'-priority support tickets in (?P<year>\d{4})\?",
        "avg_resolution": r"of resolved '(?P<topic>[a-z]+)' support tickets",
        "distinct_buyers": r"bought '(?P<title>[^']+)' in shipped orders placed in (?P<year>\d{4})\?",
    }
    m = re.search(pats[task.family], q)
    if not m:
        raise ValueError(f"cannot parse parameters of {task.id}: {q}")
    return m.groupdict()


def _guessed(sql: str) -> str | None:
    """`sql` with one plausible wrong column name, or None if none applies."""
    for good, bad in _WRONG_COLUMNS:
        if good in sql:
            return sql.replace(good, bad, 1)
    return None


def _discover(mistake: str | None, first_sql: str, tables: list[str]):
    """Schema discovery before the first real query. A 'column' mistake runs the query with a
    guessed column first (genuine SQL error), a 'table' mistake describes a wrong table name
    first (genuine error listing the real tables). Mistaken calls are context only."""
    wrong = _guessed(first_sql) if mistake == "column" else None
    if mistake == "column" and wrong is None:
        mistake = "table"  # nothing to guess wrong in this query: use the table mistake instead
    if wrong is not None:
        yield Call("run_sql", {"query": wrong}, train=False)
    for i, t in enumerate(tables):
        if i == 0 and mistake == "table":
            yield Call("describe_table", {"table": _WRONG_TABLES[t]}, train=False)
        yield Call("describe_table", {"table": t})


def _query(sql: str, hop: int, mistake):
    """One query of the plan; a QueryMistake for this hop runs its failing versions first."""
    if isinstance(mistake, QueryMistake) and mistake.hop == hop:
        for bad in mistake.bad(sql):
            yield Call("run_sql", {"query": bad}, train=False)
    return (yield Call("run_sql", {"query": sql}))


def _plan(task: Task, p: dict, mistake):
    """Yields Calls, receives each observation, returns the raw answer value."""
    hops = iter(range(_HOPS.get(task.family, 1)))
    q = lambda sql: _query(sql, next(hops), mistake)  # noqa: E731
    dm = mistake if isinstance(mistake, str) else None  # discovery mistakes (the pilot arm)
    yield Call("list_tables", {})
    f = task.family
    if f == "segment_city_count":
        sql = f"SELECT COUNT(*) FROM customers WHERE segment = '{p['seg']}' AND city = '{p['city']}'"
        yield from _discover(dm, sql, ["customers"])
        return first_value((yield from q(sql)))
    if f == "product_price":
        sql = f"SELECT unit_price_cents / 100.0 AS price_dollars FROM products WHERE title = '{p['title']}'"
        yield from _discover(dm, sql, ["products"])
        return first_value((yield from q(sql)))
    if f == "customer_shipped_year":
        sql = f"SELECT cust_id FROM customers WHERE full_name = '{p['name']}'"
        yield from _discover(dm, sql, ["customers", "orders"])
        cid = first_value((yield from q(sql)))
        return first_value((yield from q(
            f"SELECT COUNT(*) FROM orders WHERE cust_id = {cid} AND status = 'shipped' "
            f"AND order_date LIKE '{p['year']}-%'")))
    if f == "month_revenue":
        mm = MONTHS.index(p["month"]) + 1
        sql = (f"SELECT ROUND(SUM({REVENUE}) / 100.0, 2) AS revenue_dollars FROM orders o "
               "JOIN order_lines ol ON ol.order_no = o.order_no JOIN products p ON p.sku = ol.sku "
               f"WHERE o.status = 'shipped' AND o.order_date LIKE '{p['year']}-{mm:02d}-%'")
        yield from _discover(dm, sql, ["orders", "order_lines", "products"])
        return first_value((yield from q(sql)))
    if f == "top_category_city":
        sql = ("SELECT p.category, SUM(ol.qty) AS units FROM orders o "
               "JOIN customers c ON c.cust_id = o.cust_id JOIN order_lines ol ON ol.order_no = o.order_no "
               "JOIN products p ON p.sku = ol.sku "
               f"WHERE o.status = 'shipped' AND c.city = '{p['city']}' "
               "GROUP BY p.category ORDER BY units DESC LIMIT 3")
        yield from _discover(dm, sql, ["customers", "orders", "order_lines", "products"])
        return first_value((yield from q(sql)))
    if f == "ticket_top_customer":
        sql = ("SELECT c.full_name, COUNT(*) AS n FROM tickets t JOIN customers c ON c.cust_id = t.cust_id "
               f"WHERE t.priority = '{p['prio']}' AND t.opened_at LIKE '{p['year']}-%' "
               "GROUP BY c.cust_id ORDER BY n DESC LIMIT 3")
        yield from _discover(dm, sql, ["tickets", "customers"])
        return first_value((yield from q(sql)))
    if f == "avg_resolution":
        sql = ("SELECT ROUND(AVG(julianday(resolved_at) - julianday(opened_at)), 1) AS avg_days "
               f"FROM tickets WHERE topic = '{p['topic']}' AND resolved_at IS NOT NULL")
        yield from _discover(dm, sql, ["tickets"])
        return first_value((yield from q(sql)))
    if f == "distinct_buyers":
        sql = f"SELECT sku FROM products WHERE title = '{p['title']}'"
        yield from _discover(dm, sql, ["products", "order_lines", "orders"])
        sku = first_value((yield from q(sql)))
        return first_value((yield from q(
            "SELECT COUNT(DISTINCT o.cust_id) FROM orders o "
            "JOIN order_lines ol ON ol.order_no = o.order_no "
            f"WHERE ol.sku = '{sku}' AND o.status = 'shipped' AND o.order_date LIKE '{p['year']}-%'")))
    raise ValueError(f"no expert plan for family {f}")


def format_answer(value: str, kind: str) -> str:
    if kind == "money":
        return f"{float(value):.2f}"
    if kind == "float1":
        return f"{float(value):.1f}"
    if kind == "int":
        return str(int(float(value)))
    return value


def build_episode(task: Task, db_path, mistake: str | QueryMistake | None = None) -> Episode:
    """Run the family's expert plan against the task's database and record the episode."""
    raw = make_tools(db_path)
    tools = {t.__name__: t for t in raw}
    ep = Episode(task=task, mistake=mistake)
    gen = _plan(task, _params(task), mistake)
    obs = None
    try:
        c = gen.send(None)
        while True:
            obs = tools[c.tool](**c.args)
            ep.messages.append({"role": "assistant", "content": call_text(c), "train": c.train})
            ep.messages.append({"role": "user", "content": RESULT_FORMAT.format(name=c.tool, content=obs)})
            ep.calls.append({"name": c.tool, "args": c.args, "result": obs, "train": c.train})
            c = gen.send(obs)
    except StopIteration as stop:
        value = stop.value
    finally:
        raw.close()
    ep.answer = format_answer(value, task.kind)
    ep.messages.append({"role": "assistant", "content": f"ANSWER: {ep.answer}", "train": True})
    return ep


def mistake_schedule(n: int, seed: int, share: float = 0.3) -> list[str | None]:
    """Deterministic mistake assignment: `share` of episodes get one, split column/table 60/40."""
    rng = random.Random(seed)
    return [None if rng.random() >= share else ("column" if rng.random() < 0.6 else "table")
            for _ in range(n)]


def is_verified(ep: Episode) -> bool:
    return check_answer(ep.answer, ep.task.answer, ep.task.kind)
