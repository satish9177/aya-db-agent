"""Task families with hidden gold answers, split generation, and the answer verifier.

Every gold answer comes from a reference SQL query run on the split's own database.
Parameters that give ties or degenerate answers are rejected, so each task has one
correct answer. Three multi-hop families are held out of train/dev entirely and
only appear in the sealed test split.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from .env import CITIES, PRIORITIES, SEGMENTS, TOPICS, build_db

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
REVENUE = "ol.qty * p.unit_price_cents * (100 - ol.discount_pct) / 100.0"
REVENUE_RULE = ("Line revenue = qty x unit price x (1 - discount_pct/100). "
                "Only orders with status 'shipped' count.")


@dataclass
class Task:
    id: str
    split: str
    family: str
    prompt: str
    answer: str          # hidden gold, as text
    kind: str            # int | money | float1 | text
    reference_sql: str


def _one(con, sql, args=()):
    return con.execute(sql, args).fetchall()


def _unique_top(rows):
    """(label, value) rows sorted desc -> label if the top value is not tied, else None."""
    if not rows or rows[0][1] in (None, 0) or (len(rows) > 1 and rows[0][1] == rows[1][1]):
        return None
    return rows[0][0]


# Each family: (con, rng) -> (prompt, answer, kind, sql) or None to reject the draw.
def f_segment_city_count(con, rng):
    seg, city = rng.choice(SEGMENTS), rng.choice(CITIES)
    sql = "SELECT COUNT(*) FROM customers WHERE segment = ? AND city = ?"
    n = _one(con, sql, (seg, city))[0][0]
    if n == 0:
        return None
    return (f"How many customers in the '{seg}' segment are based in {city}?", str(n), "int",
            sql.replace("?", f"'{seg}'", 1).replace("?", f"'{city}'", 1))


def f_product_price(con, rng):
    title = rng.choice(_one(con, "SELECT title FROM products"))[0]
    sql = f"SELECT unit_price_cents / 100.0 FROM products WHERE title = '{title}'"
    return (f"What is the list price, in dollars, of the product '{title}'?",
            f"{_one(con, sql)[0][0]:.2f}", "money", sql)


def f_customer_shipped_year(con, rng):
    name = rng.choice(_one(con, "SELECT full_name FROM customers"))[0]
    year = rng.choice([2025, 2026])
    sql = ("SELECT COUNT(*) FROM orders o JOIN customers c ON c.cust_id = o.cust_id "
           f"WHERE c.full_name = '{name}' AND o.status = 'shipped' AND o.order_date LIKE '{year}-%'")
    n = _one(con, sql)[0][0]
    if n == 0:
        return None
    return (f"How many orders with status 'shipped' did {name} place in {year}?", str(n), "int", sql)


def f_month_revenue(con, rng):
    year = rng.choice([2025, 2026])
    month = rng.randint(1, 12 if year == 2025 else 6)
    sql = (f"SELECT ROUND(SUM({REVENUE}) / 100.0, 2) FROM orders o "
           "JOIN order_lines ol ON ol.order_no = o.order_no JOIN products p ON p.sku = ol.sku "
           f"WHERE o.status = 'shipped' AND o.order_date LIKE '{year}-{month:02d}-%'")
    v = _one(con, sql)[0][0]
    if not v:
        return None
    return (f"What was the total revenue, in dollars, from orders placed in {MONTHS[month - 1]} "
            f"{year}? {REVENUE_RULE} Round to 2 decimals.", f"{v:.2f}", "money", sql)


def f_top_category_city(con, rng):
    city = rng.choice(CITIES)
    sql = ("SELECT p.category, SUM(ol.qty) AS units FROM orders o "
           "JOIN customers c ON c.cust_id = o.cust_id JOIN order_lines ol ON ol.order_no = o.order_no "
           f"JOIN products p ON p.sku = ol.sku WHERE o.status = 'shipped' AND c.city = '{city}' "
           "GROUP BY p.category ORDER BY units DESC")
    top = _unique_top(_one(con, sql))
    if top is None:
        return None
    return (f"Which product category sold the most units in shipped orders to customers based in "
            f"{city}? Answer with the category name.", top, "text", sql)


def f_ticket_top_customer(con, rng):
    prio, year = rng.choice(PRIORITIES), rng.choice([2025, 2026])
    sql = ("SELECT c.full_name, COUNT(*) AS n FROM tickets t JOIN customers c ON c.cust_id = t.cust_id "
           f"WHERE t.priority = '{prio}' AND t.opened_at LIKE '{year}-%' "
           "GROUP BY c.cust_id ORDER BY n DESC")
    top = _unique_top(_one(con, sql))
    if top is None:
        return None
    return (f"Which customer opened the most '{prio}'-priority support tickets in {year}? "
            "Answer with the customer's full name.", top, "text", sql)


def f_avg_resolution(con, rng):
    topic = rng.choice(TOPICS)
    sql = ("SELECT ROUND(AVG(julianday(resolved_at) - julianday(opened_at)), 1) FROM tickets "
           f"WHERE topic = '{topic}' AND resolved_at IS NOT NULL")
    v = _one(con, sql)[0][0]
    return (f"What is the average resolution time, in days, of resolved '{topic}' support tickets? "
            "Round to 1 decimal.", f"{v:.1f}", "float1", sql)


def f_distinct_buyers(con, rng):
    title = rng.choice(_one(con, "SELECT title FROM products"))[0]
    year = rng.choice([2025, 2026])
    sql = ("SELECT COUNT(DISTINCT o.cust_id) FROM orders o JOIN order_lines ol ON ol.order_no = o.order_no "
           f"JOIN products p ON p.sku = ol.sku WHERE p.title = '{title}' AND o.status = 'shipped' "
           f"AND o.order_date LIKE '{year}-%'")
    n = _one(con, sql)[0][0]
    if n == 0:
        return None
    return (f"How many distinct customers bought '{title}' in shipped orders placed in {year}?",
            str(n), "int", sql)


# ---- held-out families: test split only --------------------------------------------------------
def f_cancel_leader_tickets(con, rng):
    year = rng.choice([2025, 2026])
    sql1 = ("SELECT cust_id, COUNT(*) AS n FROM orders WHERE status = 'cancelled' "
            f"AND order_date LIKE '{year}-%' GROUP BY cust_id ORDER BY n DESC")
    cust = _unique_top(_one(con, sql1))
    if cust is None:
        return None
    sql = ("SELECT COUNT(*) FROM tickets WHERE cust_id = (SELECT cust_id FROM orders "
           f"WHERE status = 'cancelled' AND order_date LIKE '{year}-%' GROUP BY cust_id "
           "ORDER BY COUNT(*) DESC LIMIT 1)")
    n = _one(con, sql)[0][0]
    assert n == _one(con, "SELECT COUNT(*) FROM tickets WHERE cust_id = ?", (cust,))[0][0]
    return (f"Find the customer with the most cancelled orders placed in {year}. How many support "
            "tickets has that customer opened in total?", str(n), "int", sql)


def f_largest_order(con, rng):
    name = rng.choice(_one(con, "SELECT full_name FROM customers"))[0]
    sql = (f"SELECT o.order_no, SUM({REVENUE}) AS v FROM orders o "
           "JOIN customers c ON c.cust_id = o.cust_id JOIN order_lines ol ON ol.order_no = o.order_no "
           f"JOIN products p ON p.sku = ol.sku WHERE c.full_name = '{name}' AND o.status = 'shipped' "
           "GROUP BY o.order_no ORDER BY v DESC")
    top = _unique_top(_one(con, sql))
    if top is None:
        return None
    return (f"What is the order number of {name}'s most valuable shipped order? {REVENUE_RULE}",
            str(top), "int", sql)


def f_segment_aov(con, rng):
    seg, year = rng.choice(SEGMENTS), rng.choice([2025, 2026])
    sql = (f"SELECT ROUND(AVG(v) / 100.0, 2) FROM (SELECT o.order_no, SUM({REVENUE}) AS v "
           "FROM orders o JOIN customers c ON c.cust_id = o.cust_id "
           "JOIN order_lines ol ON ol.order_no = o.order_no JOIN products p ON p.sku = ol.sku "
           f"WHERE c.segment = '{seg}' AND o.status = 'shipped' AND o.order_date LIKE '{year}-%' "
           "GROUP BY o.order_no)")
    v = _one(con, sql)[0][0]
    if not v:
        return None
    return (f"What was the average value, in dollars, of shipped orders placed in {year} by "
            f"'{seg}' customers? {REVENUE_RULE} Round to 2 decimals.", f"{v:.2f}", "money", sql)


SHARED = {f.__name__[2:]: f for f in [
    f_segment_city_count, f_product_price, f_customer_shipped_year, f_month_revenue,
    f_top_category_city, f_ticket_top_customer, f_avg_resolution, f_distinct_buyers]}
HELD_OUT = {f.__name__[2:]: f for f in [f_cancel_leader_tickets, f_largest_order, f_segment_aov]}

# seed = database seed; train/dev/test databases never share rows.
SPLITS = {
    "train": {"seed": 101, "families": SHARED, "per_family": 60},
    "dev": {"seed": 202, "families": SHARED, "per_family": 5},
    "test": {"seed": 303, "families": {**SHARED, **HELD_OUT}, "per_family": 12},
}


# The frozen 40-task dev set (2026-11-04). Every dev evaluation checks it regenerates to this.
# The preregistered hash was taken on Windows, where write_text ends lines with CRLF; the same
# records written on Linux (LF) hash to FROZEN_DEV_SHA256_LF. `is_frozen_dev` accepts either.
FROZEN_DEV_SHA256 = "242e89cc07b8f7a79c8dcb04ad629109362b2350d59148f6a2a90f5f5f017433"
FROZEN_DEV_SHA256_LF = "7798e6fe626c51fc4bacf39c8171120885fd78ea029b0627b89e9896e89cdaae"

# Extra training databases for the SFT pool. Disjoint from the dev (202) and test (303) seeds.
TRAIN_POOL_SEEDS = (1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108)


def generate_split(split: str, data_dir: Path) -> tuple[Path, Path]:
    """Build the split's database and task file. Returns (db_path, tasks_path)."""
    cfg = SPLITS[split]
    return generate_tasks(split, cfg["seed"], cfg["families"], cfg["per_family"], data_dir)


def generate_tasks(name: str, seed: int, families: dict, per_family: int,
                   data_dir: Path) -> tuple[Path, Path]:
    """Database `<name>.sqlite` for `seed` and its tasks `<name>.jsonl` (ids `<name>-<family>-NNN`)."""
    assert name in SPLITS or seed not in {c["seed"] for c in SPLITS.values()}, "seed reuse"
    split = name
    cfg = {"seed": seed, "families": families, "per_family": per_family}
    db = build_db(data_dir / f"{split}.sqlite", cfg["seed"])
    con = sqlite3.connect(db)
    rng = random.Random(cfg["seed"] * 7919)
    tasks, seen = [], set()
    for fam, fn in cfg["families"].items():
        made, tries = 0, 0
        while made < cfg["per_family"] and tries < 2000:
            tries += 1
            out = fn(con, rng)
            if out is None or out[0] in seen:
                continue
            seen.add(out[0])
            prompt, answer, kind, sql = out
            tasks.append(Task(f"{split}-{fam}-{made:03d}", split, fam, prompt, answer, kind, sql))
            made += 1
    con.close()
    path = data_dir / f"{split}.jsonl"
    path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks), encoding="utf-8")
    return db, path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    """sha256 of the file with LF line endings: the same on Windows and Linux."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def is_frozen_dev(path: Path) -> bool:
    """True if `path` holds the frozen dev records, whatever the platform's line endings."""
    return lf_sha256(path) == FROZEN_DEV_SHA256_LF


# ---- recovery experiment evaluation splits (experiments/recovery/PREREGISTRATION.md) -------------
# drift: the frozen dev questions and rows with five columns renamed. A query written from memory of
# the training schema fails; one written from the described schema works; gold answers are unchanged.
# mh: the three held-out multi-hop families on three new databases (never the sealed test's).
DRIFT_RENAMES = {("customers", "full_name"): "display_name", ("customers", "city"): "home_city",
                 ("products", "unit_price_cents"): "list_price_cents", ("orders", "order_date"): "placed_on",
                 ("tickets", "opened_at"): "opened_on"}
MH_SEEDS = (404, 405, 406)
MH_PER_FAMILY = 6
EVAL_SPLITS = ("drift", "mh")
PINNED_LF = {  # evaluation splits whose task files are pinned (LF line endings)
    "dev": FROZEN_DEV_SHA256_LF,
    "drift": "eaf5cbaf32120ca4f991629c379fd57d7fbd6edf0242e63254405cd9d9907ace",
    "mh": "d71da41d178eaecc64ccc96e4fbf13ad3bfc7b2056b308e7e96d41a559fcf2e5",
}


def drift_sql(sql: str) -> str:
    for (_, col), new in DRIFT_RENAMES.items():
        sql = re.sub(rf"\b{col}\b", new, sql)
    return sql


def _write_tasks(tasks: list[Task], path: Path) -> Path:
    path.write_text("".join(json.dumps(asdict(t)) + "\n" for t in tasks), encoding="utf-8")
    return path


def generate_eval(split: str, data_dir: Path) -> tuple[Path, dict[str, Path]]:
    """An evaluation split's task file and each task's database."""
    if split in SPLITS:
        db, path = generate_split(split, data_dir)
        return path, {t.id: db for t in load_tasks(path)}
    if split == "drift":
        dev_db, dev_path = generate_split("dev", data_dir)
        assert is_frozen_dev(dev_path), "drift is built from the frozen dev set"
        db = data_dir / "drift.sqlite"
        shutil.copyfile(dev_db, db)
        con = sqlite3.connect(db)
        for (table, col), new in DRIFT_RENAMES.items():
            con.execute(f"ALTER TABLE {table} RENAME COLUMN {col} TO {new}")
        con.commit()
        tasks = [Task(t.id.replace("dev-", "drift-", 1), "drift", t.family, t.prompt, t.answer, t.kind,
                      drift_sql(t.reference_sql)) for t in load_tasks(dev_path)]
        for t in tasks:  # same rows, so the renamed reference queries give the frozen gold answers
            assert check_answer(str(con.execute(t.reference_sql).fetchone()[0]), t.answer, t.kind), t.id
        con.close()
        return _write_tasks(tasks, data_dir / "drift.jsonl"), {t.id: db for t in tasks}
    if split == "mh":
        tasks, dbs = [], {}
        for seed in MH_SEEDS:
            db, path = generate_tasks(f"mh{seed}", seed, HELD_OUT, MH_PER_FAMILY, data_dir)
            for t in load_tasks(path):
                tasks.append(t)
                dbs[t.id] = db
        return _write_tasks(tasks, data_dir / "mh.jsonl"), dbs
    raise ValueError(f"unknown split {split}")


def load_tasks(path: Path) -> list[Task]:
    return [Task(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


# ---- verifier -----------------------------------------------------------------------------------
_MARKER = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(final_text: str | None) -> str | None:
    """Value after the last 'ANSWER:' marker (first line only), or None if there is no marker."""
    if not final_text:
        return None
    hits = _MARKER.findall(final_text)
    if not hits:
        return None
    return hits[-1].strip().splitlines()[0].strip()


def check_answer(value: str | None, gold: str, kind: str) -> bool:
    """Strict check of one extracted value. Hedged answers ('12 or 13') fail by construction."""
    if value is None:
        return False
    v = value.strip().strip("`*\"'").rstrip(".").strip()
    if kind == "text":
        return v.casefold() == gold.casefold()
    nums = _NUM.findall(v.replace("$", ""))
    if len(nums) != 1:
        return False
    try:
        x = float(nums[0].replace(",", ""))
    except ValueError:
        return False
    tol = {"int": 0.0, "money": 0.011, "float1": 0.051}[kind]
    return abs(x - float(gold)) <= tol
