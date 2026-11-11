"""Deterministic read-only SQLite business database and the agent's tools.

One database per split seed, so train/dev/test never share rows. The schema has
deliberate traps a model can only get past by inspecting it: money is stored in
cents, keys are named `cust_id` / `order_no` / `sku`, products have a `title`
(not a name), and unresolved tickets have NULL `resolved_at`.
"""

from __future__ import annotations

import datetime as dt
import random
import sqlite3
from pathlib import Path

FIRST = ["Asha", "Ravi", "Meera", "Arjun", "Leela", "Kiran", "Nikhil", "Priya", "Sanjay", "Divya",
         "Omar", "Elena", "Tomas", "Yuki", "Grace", "Hassan", "Ines", "Mateo", "Nadia", "Pavel"]
LAST = ["Rao", "Iyer", "Kapoor", "Menon", "Das", "Reddy", "Shah", "Varma", "Nair", "Joshi",
        "Lopez", "Novak", "Sato", "Okafor", "Silva", "Berg", "Haddad", "Kowalski"]
CITIES = ["Pune", "Chennai", "Hyderabad", "Kochi", "Jaipur", "Lucknow", "Indore", "Mysuru"]
SEGMENTS = ["retail", "wholesale", "enterprise"]
CATEGORIES = {
    "office": ["Gel Pen Box", "A4 Paper Ream", "Desk Organizer", "Stapler Pro", "Whiteboard Kit",
               "Sticky Notes Pack", "Archive Box"],
    "electronics": ["USB-C Hub", "Wireless Mouse", "Laptop Stand", "HD Webcam", "Power Bank",
                    "Bluetooth Speaker", "Surge Protector"],
    "furniture": ["Ergo Chair", "Standing Desk", "Filing Cabinet", "Bookshelf Oak", "Meeting Table"],
    "pantry": ["Green Tea Case", "Coffee Beans 1kg", "Snack Crate", "Water Bottles 24", "Sugar Sachets"],
    "cleaning": ["Floor Cleaner", "Hand Wash Refill", "Microfiber Cloths", "Trash Bags XL"],
}
TOPICS = ["billing", "delivery", "returns", "account", "defect"]
PRIORITIES = ["low", "medium", "high"]
STATUSES = ["shipped"] * 15 + ["cancelled"] * 2 + ["returned"] * 2 + ["pending"]

START = dt.date(2025, 1, 1)
END = dt.date(2026, 6, 30)

SCHEMA = """
CREATE TABLE customers (
    cust_id INTEGER PRIMARY KEY, full_name TEXT NOT NULL, city TEXT NOT NULL,
    segment TEXT NOT NULL, signup_date TEXT NOT NULL);
CREATE TABLE products (
    sku TEXT PRIMARY KEY, title TEXT NOT NULL, category TEXT NOT NULL,
    unit_price_cents INTEGER NOT NULL);
CREATE TABLE orders (
    order_no INTEGER PRIMARY KEY, cust_id INTEGER NOT NULL REFERENCES customers(cust_id),
    order_date TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE order_lines (
    order_no INTEGER NOT NULL REFERENCES orders(order_no), sku TEXT NOT NULL REFERENCES products(sku),
    qty INTEGER NOT NULL, discount_pct INTEGER NOT NULL);
CREATE TABLE tickets (
    ticket_id INTEGER PRIMARY KEY, cust_id INTEGER NOT NULL REFERENCES customers(cust_id),
    opened_at TEXT NOT NULL, priority TEXT NOT NULL, topic TEXT NOT NULL, resolved_at TEXT);
"""


def _day(rng: random.Random, lo: dt.date = START, hi: dt.date = END) -> dt.date:
    return lo + dt.timedelta(days=rng.randrange((hi - lo).days + 1))


def build_db(path: Path, seed: int) -> Path:
    """Create the database for `seed` at `path` (overwritten). Same seed, same bytes of data."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)

    names = rng.sample([f"{f} {l}" for f in FIRST for l in LAST], 60)
    customers = []
    for i, name in enumerate(names, start=1):
        customers.append((i, name, rng.choice(CITIES), rng.choice(SEGMENTS),
                          _day(rng, dt.date(2023, 1, 1), dt.date(2024, 12, 31)).isoformat()))
    con.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products, n = [], 0
    for cat, titles in CATEGORIES.items():
        for title in titles:
            n += 1
            products.append((f"{cat[:3].upper()}-{100 + n * 7 % 900:03d}", title, cat,
                             rng.randrange(99, 45000, 1)))
    con.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    orders, lines = [], []
    for order_no in range(10001, 10001 + 640):
        cust = rng.choice(customers)[0]
        orders.append((order_no, cust, _day(rng).isoformat(), rng.choice(STATUSES)))
        for sku in rng.sample([p[0] for p in products], rng.randint(1, 4)):
            lines.append((order_no, sku, rng.randint(1, 12), rng.choice([0, 0, 0, 5, 10, 15])))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
    con.executemany("INSERT INTO order_lines VALUES (?,?,?,?)", lines)

    tickets = []
    for tid in range(1, 281):
        opened = _day(rng)
        resolved = None
        if rng.random() < 0.8:
            resolved = (opened + dt.timedelta(days=rng.randint(0, 20))).isoformat()
        tickets.append((tid, rng.choice(customers)[0], opened.isoformat(), rng.choice(PRIORITIES),
                        rng.choice(TOPICS), resolved))
    con.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?)", tickets)
    con.commit()
    con.close()
    return path


_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
                    getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
MAX_ROWS = 20


def connect_ro(path: Path) -> sqlite3.Connection:
    """Read-only connection: URI mode=ro, query_only, and an authorizer that allows reads only."""
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    con.set_authorizer(lambda action, *_: sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS
                       else sqlite3.SQLITE_DENY)
    return con


class Tools(list):
    """The tool functions, plus close() for their database connections (Windows file locks)."""

    def __init__(self, fns, connections):
        super().__init__(fns)
        self._connections = connections

    def close(self) -> None:
        for c in self._connections:
            c.close()


def make_tools(db_path: Path) -> Tools:
    """The three tools the agent gets, bound to one database. Docstrings become the JSON schemas."""
    con = connect_ro(db_path)
    admin = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True,
                            check_same_thread=False)  # schema introspection only, never agent SQL

    def table_names() -> list[str]:
        return [r[0] for r in admin.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

    def list_tables() -> str:
        """List the tables in the company database with their row counts."""
        out = []
        for name in table_names():
            count = admin.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            out.append(f"{name} ({count} rows)")
        return "Tables: " + ", ".join(out)

    def describe_table(table: str) -> str:
        """Show the columns, foreign keys and three sample rows of one table.

        Args:
            table: Exact table name, as returned by list_tables.
        """
        names = table_names()
        if table not in names:
            return f"Error: no table named {table!r}. Available tables: {', '.join(names)}"
        cols = admin.execute(f'PRAGMA table_info("{table}")').fetchall()
        fks = {r[3]: f"{r[2]}.{r[4]}" for r in admin.execute(f'PRAGMA foreign_key_list("{table}")')}
        parts = [f"{c[1]} {c[2]}" + (f" -> {fks[c[1]]}" if c[1] in fks else "") for c in cols]
        rows = admin.execute(f'SELECT * FROM "{table}" LIMIT 3').fetchall()
        sample = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r) for r in rows)
        return f"Table {table}: " + ", ".join(parts) + "\nSample rows:\n" + sample

    def run_sql(query: str) -> str:
        """Run one read-only SQLite SELECT query and return up to 20 result rows.

        Args:
            query: A single SQLite SELECT statement.
        """
        if not isinstance(query, str) or not query.strip():
            return "Error: query must be a non-empty SQL string."
        steps = [0]

        def guard() -> int:
            steps[0] += 1
            return 1 if steps[0] > 2000 else 0  # ~2M VM ops: stops runaway cross joins

        con.set_progress_handler(guard, 1000)
        try:
            cur = con.execute(query)
            rows = cur.fetchmany(MAX_ROWS + 1)
            header = [d[0] for d in cur.description or []]
        except sqlite3.Error as e:
            return f"SQL error: {e}"
        finally:
            con.set_progress_handler(None, 0)
        more = len(rows) > MAX_ROWS
        body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r)
                         for r in rows[:MAX_ROWS])
        note = f"\n(showing first {MAX_ROWS} rows; more rows exist)" if more else ""
        return " | ".join(header) + "\n" + (body or "(no rows)") + note

    return Tools([list_tables, describe_table, run_sql], [con, admin])
