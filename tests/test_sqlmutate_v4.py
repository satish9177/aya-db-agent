"""Amendment 10: realistic column mistakes and step-by-step parts for v4 demos."""
import sqlite3

import pytest

from ayadb.sqlmutate_v4 import first_step_query, make_column_mutation, name_variants
from ayadb.sqlscore import execute


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE users(user_id INTEGER, userName TEXT, signup_date TEXT);"
        "CREATE TABLE orders(order_id INTEGER, user_id INTEGER, total_amount REAL, status TEXT);"
        "INSERT INTO users VALUES (1, 'ann', '2024-01-01'), (2, 'bob', '2024-02-01');"
        "INSERT INTO orders VALUES (10, 1, 5.0, 'paid'), (11, 2, 7.5, 'open');")
    con.close()
    return path


GOLD = ("SELECT u.userName, SUM(o.total_amount) AS spent FROM users u JOIN orders o ON o.user_id = u.user_id "
        "WHERE o.status = 'paid' GROUP BY u.userName")


def test_every_mutation_is_a_real_no_such_column_error_with_its_owner(db):
    seen = set()
    for i in range(40):
        m = make_column_mutation(db, GOLD, f"episode-{i}")
        assert m["error"].startswith("no such column") and execute(db, m["query"]).error == m["error"]
        assert m["owner_table"] in {"users", "orders"} and m["kind"] in {"typo", "naming_style", "other_table_name", "wrong_alias"}
        seen.add(m["kind"])
        assert make_column_mutation(db, GOLD, f"episode-{i}") == m  # deterministic per episode
    assert seen == {"typo", "naming_style", "other_table_name", "wrong_alias"}


def test_wrong_alias_moves_a_real_column_to_a_table_without_it(db):
    m = next(make_column_mutation(db, GOLD, f"e{i}") for i in range(200)
             if make_column_mutation(db, GOLD, f"e{i}")["kind"] == "wrong_alias")
    column = m["original_column"]
    assert m["edited"].startswith("table=") and f"{m['edited'][6:]}.{column}" in m["query"]


def test_name_variants_are_plausible_slips():
    assert {"supevisor", "supervsor"} <= set(name_variants("supervisor", "typo", []))
    # Case-only variants never count: SQLite column names are case-insensitive.
    assert {"user_name", "userName_id"} <= set(name_variants("userName", "naming_style", []))
    assert {"orderId", "orderid", "order"} <= set(name_variants("order_id", "naming_style", []))
    assert next(name_variants("user_id", "other_table_name", ["order_id", "user_email"])) == "user_email"


def test_first_step_is_the_first_cte_body_or_an_uncorrelated_subquery(db):
    cte = ("WITH paid AS (SELECT user_id, SUM(total_amount) AS spent FROM orders WHERE status = 'paid' GROUP BY user_id) "
           "SELECT u.userName, p.spent FROM paid p JOIN users u ON u.user_id = p.user_id")
    part = first_step_query(db, cte)
    assert part.startswith("SELECT user_id, SUM(total_amount)") and not execute(db, part).error
    sub = "SELECT userName FROM users WHERE user_id IN (SELECT user_id FROM orders WHERE status = 'paid')"
    assert first_step_query(db, sub) == "SELECT user_id FROM orders WHERE status = 'paid'"
    assert first_step_query(db, "SELECT userName FROM users") is None
