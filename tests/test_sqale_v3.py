import copy
import sqlite3

import pytest

from ayadb.promptquality import clean_prompt
from ayadb.shopprojection import scalar_reference
from ayadb.sqlmutate import ERRORS, KINDS, make_mutation
from ayadb.sqlpolicy import ordering_policy
from ayadb.sqlscore import execute, score_query
from ayadb.tokenbudget import match_control, size_gate, targets


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "v3.sqlite"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE items(id INTEGER, value INTEGER, label TEXT);"
                      "INSERT INTO items VALUES(1,10,'A'),(2,10,'a'),(3,20,'b');"
                      "CREATE TABLE groups(id INTEGER, label TEXT);")
    con.close()
    return path


def test_tied_order_is_unordered_but_unique_order_is_enforced(db):
    ref = "SELECT id,value FROM items ORDER BY value"
    policy = ordering_policy(db, ref)
    assert policy["eligible"] and not policy["ordered"]
    assert score_query(db, "SELECT id,value FROM items ORDER BY value,id DESC", ref)["strict"]
    assert not score_query(db, "SELECT id,value FROM items ORDER BY id DESC",
                           "SELECT id,value FROM items ORDER BY id")["strict"]
    assert not score_query(db, "SELECT value FROM items GROUP BY value",
                           "SELECT value FROM items ORDER BY value")["strict"]


def test_alias_ordinal_collation_group_and_limit_ties(db):
    for ref in ("SELECT id, label AS name FROM items ORDER BY name COLLATE NOCASE",
                "SELECT id,value FROM items ORDER BY 2",
                "SELECT label, SUM(value) AS n FROM items GROUP BY label ORDER BY n"):
        assert not ordering_policy(db, ref)["ordered"]
    assert not ordering_policy(db, "SELECT id FROM items ORDER BY value LIMIT 1")["eligible"]
    assert not ordering_policy(db, "SELECT id FROM items ORDER BY value LIMIT 2 OFFSET 1")["eligible"]
    assert ordering_policy(db, "SELECT id FROM items ORDER BY value LIMIT 2")["eligible"]
    assert not ordering_policy(db, "SELECT id FROM items LIMIT 1")["eligible"]


def test_now_text_and_comments_are_harmless_and_real_clock_args_blocked(db):
    for sql in ("SELECT 'now'", 'SELECT id AS "now" FROM items',
                "SELECT 'CURRENT_TIMESTAMP' -- date('now')", "SELECT date('2024-01-01','+1 day')",
                "SELECT strftime('%Y','2024-01-01')"):
        assert execute(db, sql).error is None, sql
    for sql in ("SELECT date('now')", "SELECT date()", "SELECT CURRENT_DATE",
                "SELECT datetime('subsec')", "SELECT strftime('%Y')",
                "SELECT date(lower('NOW'))", "SELECT timediff('now','2024-01-01')"):
        assert execute(db, sql).error == "nondeterministic_time", sql


@pytest.mark.parametrize("kind", KINDS)
def test_gold_mutation_gives_intended_native_error(db, kind):
    m = make_mutation(db, "SELECT i.id, SUM(i.value) FROM items i GROUP BY i.id", "test", requested=kind)
    assert m["kind"] == kind and ERRORS[kind] in execute(db, m["query"]).error.lower()
    assert "__ayadb_bad_alias__" not in m["query"]


def example(q, n):
    return {"id": q, "question_id": q, "input_ids": list(range(n + 2)), "labels": [-100, -100] + list(range(n))}


def test_gate_deduplicates_both_sides_before_epochs():
    b, a = example("q", 10), example("a", 100)
    result = size_gate([b, copy.deepcopy(b)], [a, copy.deepcopy(a)])
    assert result["passed"] and result["verified_unique_targets_per_pass"] == 10
    assert result["a_unique_targets_per_pass"] == 100
    assert not size_gate([b] * 4, [example("a", 101)] * 4)["passed"]


def test_global_matching_uses_at_most_one_partial_and_keeps_all_questions():
    control = [example("q1", 5), example("q2", 7), example("q3", 6)]
    verified = [example("q1", 12), example("q2", 9), example("q3", 11)]
    original = copy.deepcopy(control)
    matched, report = match_control(control, verified)
    assert sum(map(targets, matched)) == 32
    assert len(report["mask_adjustments"]) <= 1
    assert {e["question_id"] for e in matched} == {"q1", "q2", "q3"}
    assert control == original
    for ex in matched:
        assert all(label == -100 or label == ex["input_ids"][i] - 2
                   for i, label in enumerate(ex["labels"]))


def test_global_coverage_infeasible_with_only_one_partial():
    with pytest.raises(ValueError, match="infeasible"):
        match_control([example("q1", 10), example("q2", 10)],
                      [example("q1", 1), example("q2", 1)])


def test_heading_generator_and_label_cleanup():
    assert clean_prompt("**Simple 3:** List every item ID.") == (
        "List every item ID.", None, ["stripped_difficulty_number_prefix"])
    assert clean_prompt("**Simple**: List every item ID.")[1] is None
    assert clean_prompt("### Simple Questions")[1]
    assert clean_prompt("Generate three moderate questions about this schema.")[1]
    assert clean_prompt("Find everything, actually, find only the first item.")[1]
    assert clean_prompt("Generate a report of item counts per group.")[1] is None
    for value in ("**Domain Grounding**", "**Domain:** Subscription Management",
                  "*This requires joining the PERSON table.*", "**article_keywords**: Semantic tagging of articles.",
                  "**Retrieve Team and Manager Details for a Specific Employee:**",
                  "*These questions involve basic retrieval.*", "1. Generate straightforward questions."):
        assert clean_prompt(value)[1], value
    for value in ("**Simple (Retrieval with Filter):** Find the item with ID 8.",
                  "**Specific lookup:** Find the item with ID 8.", "**Find the item with ID 8.**"):
        assert clean_prompt(value)[0] == "Find the item with ID 8." and clean_prompt(value)[1] is None


def test_scalar_projection_drops_extra_columns_and_rows_after_alias_resolution(db):
    ref = "SELECT label, SUM(value) AS total FROM items GROUP BY label ORDER BY total DESC"
    converted = scalar_reference(ref)
    assert execute(db, converted).rows == [("b",)]
    assert ordering_policy(db, converted)["eligible"]
    tied = scalar_reference("SELECT id, value AS total FROM items ORDER BY total ASC")
    assert not ordering_policy(db, tied)["eligible"]
