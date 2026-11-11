import json
import pytest

from ayadb.sqalepool import assign_splits, copy_slice, materialize, schema_groups, usable_question
from ayadb.sqlscore import execute
from ayadb.tokenbudget import continuation_forecast, match_control, size_gate, targets


def test_schema_groups_keep_related_schemas_and_phrasings_together():
    ddl = "CREATE TABLE a(x INTEGER); CREATE TABLE b(x INTEGER); CREATE TABLE c(x INTEGER);"
    inventory = [{"schema_id": "train", "full_schema": ddl, "source_split": "train"},
                 {"schema_id": "test", "full_schema": ddl, "source_split": "test"},
                 {"schema_id": "other", "full_schema": "CREATE TABLE unrelated(y TEXT);", "source_split": "train"}]
    groups, links = schema_groups(inventory)
    assert groups["train"] == groups["test"] != groups["other"]
    assert links
    assert assign_splits(inventory, groups)["train"] == "excluded_overlap"
    assert not usable_question("<string>") and not usable_question("...")
    assert usable_question("How many items have a positive quantity?")


def test_materialization_keeps_rows_and_foreign_keys_in_slices(tmp_path):
    row = {"full_schema": "CREATE TABLE a(id INTEGER PRIMARY KEY); CREATE TABLE b(id INTEGER REFERENCES a(id));",
           "schema_content": json.dumps({"a": [{"id": 1}], "b": [{"id": 1}, {"id": 1}]})}
    full, sliced = tmp_path / "full.sqlite", tmp_path / "slice.sqlite"
    materialize(row, full)
    copy_slice(full, sliced, ["a", "b"])
    assert execute(sliced, "SELECT * FROM a JOIN b USING(id)").rows == [(1,), (1,)]


def test_unrepresentable_source_integer_rejects_without_changing_its_value(tmp_path):
    row = {"full_schema": "CREATE TABLE huge(id INTEGER);",
           "schema_content": json.dumps({"huge": [{"id": 2 ** 100}]})}
    path = tmp_path / "unrepresentable.sqlite"
    with pytest.raises(OverflowError):
        materialize(row, path)
    assert not path.exists()


def ex(q, labels):
    return {"question_id": q, "id": q, "input_ids": [1] * len(labels), "labels": labels}


def test_control_exactly_matches_global_target_budget_for_retained_questions():
    control = [ex("q1", [1, 1, 1, -100]), ex("q2", [1, 1, -100])]
    verified = [ex("q1", [1] * 8), ex("q2", [1] * 3)]
    matched, report = match_control(control, verified)
    assert sum(targets(x) for x in matched) == 11 == report["matched_targets"]
    assert {x["question_id"] for x in matched} == {"q1", "q2"}
    assert report["repeated_copies"] and len(report["mask_adjustments"]) <= 1
    assert targets(control[0]) == 3  # original examples were not mutated
    assert size_gate(verified, 110)["passed"]
    assert not size_gate(verified, 111)["passed"]
    forecast = continuation_forecast(.25, 300, 200, 100_000)
    assert forecast["verified_episode_scenario"] == [75, 300]
