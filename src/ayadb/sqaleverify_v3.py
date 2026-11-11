"""Independent structural checks for the private v3 preparation; no model evaluation."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .promptquality import V3, clean_prompt, fingerprint, reviewed_decisions
from .sqalebuild import write_json
from .sqalebuild_v3 import read_lines, verify_v2
from .sqlmutate import ERRORS, KINDS
from .sqlscore import execute, rows_equal
from .tasks import PINNED_LF, check_answer, lf_sha256


def verify(prepared_only=False):
    prep = json.loads((V3 / "preparation.json").read_text())
    decisions = reviewed_decisions()
    pools = {s: read_lines(V3 / f"{s}.tasks.jsonl") for s in ("train", "shop", "q2", "dev", "final")}
    assert {s: len(v) for s, v in pools.items()} == prep["selected_counts"]
    tasks = {t["id"]: t for rows in pools.values() for t in rows}
    assert len(tasks) == sum(map(len, pools.values()))
    groups = {s: {t["schema_group"] for t in pools[s]} for s in ("train", "q2", "dev", "final")}
    for i, split in enumerate(groups):
        for other in list(groups)[i + 1:]:
            assert not groups[split] & groups[other], (split, other)
    for split in ("dev", "final", "q2"):
        for task in pools[split]:
            decision = decisions[task["id"]]
            assert decision["manual_decision"] == "keep" and task["prompt"] == decision["clean_prompt"]
    for task in tasks.values():
        cleaned, reason, edits = clean_prompt(task["prompt"])
        assert not edits and task["prompt"] == cleaned
        assert reason is None or task["id"] in decisions and decisions[task["id"]]["manual_decision"] == "keep"
        assert task["ordering_policy"]["eligible"] and task["ordered"] == task["ordering_policy"]["ordered"]
        path = Path(task["db_path"])
        assert path.resolve().is_relative_to(V3.resolve()) and fingerprint(path) == task["db_sha256"]
    episodes = read_lines(V3 / "train.episodes.jsonl")
    assert len(episodes) == prep["train_episodes"]
    assert {e["task_id"] for e in episodes} == {t["id"] for s in ("shop", "train") for t in pools[s]}
    assert len({e["question_id"] for e in episodes}) == len(episodes)
    mutations = {m["id"]: m for m in read_lines(V3 / "mutations.jsonl")}
    repair_counts = Counter()
    for ep in episodes:
        assert 0 < ep["n_target_tokens"] < ep["n_input_tokens"] <= 4096
        assert "__ayadb_bad_alias__" not in json.dumps(ep)
        assert len(ep["calls"]) <= 8 and ep["calls"][-1]["name"] == "submit_sql" and ep["calls"][-1]["accepted"]
        assistant = [m for m in ep["messages"] if m["role"] == "assistant"]
        assert len(assistant) == len(ep["calls"])
        for message, call in zip(assistant, ep["calls"]):
            if call.get("error"):
                assert not message["train"]
        assert all(not m["train"] for m in ep["messages"] if m["role"] != "assistant")
        if ep["repair"]:
            mutation = mutations[ep["task_id"]]
            assert ep["repair_kind"] == mutation["kind"]
            failure = execute(Path(ep["db_path"]), mutation["query"])
            assert failure.error == mutation["error"] and ERRORS[mutation["kind"]] in failure.error.lower()
            assert any(c["args"].get("query") == mutation["query"] and c.get("error") for c in ep["calls"])
            repair_counts[mutation["kind"]] += 1
    assert set(repair_counts) == set(KINDS) and dict(repair_counts) == prep["repair_kinds"]
    stage_ids = set()
    shop_ids = {t["id"] for t in pools["shop"]}
    for n in sorted(prep["stages"], key=int):
        stage = read_lines(V3 / f"train-{n}.episodes.jsonl")
        ids = {e["task_id"] for e in stage}
        assert len(stage) == len(ids) == int(n) and stage_ids <= ids and shop_ids <= ids
        count = sum(e["n_target_tokens"] for e in stage)
        assert count == prep["stages"][n]["unique_target_tokens_per_pass"]
        assert (count + 9) // 10 == prep["stages"][n]["minimum_verified_unique_targets_per_pass"]
        stage_ids = ids
    retention_count = 0
    review = json.loads((V3 / "retention/manual-review.json").read_text())
    reviewed = {r["id"] for r in review["decisions"] if r["decision"] == "keep"}
    for split in ("dev", "mh"):
        assert lf_sha256(V3 / f"retention/{split}.jsonl") == PINNED_LF[split]
        retained = read_lines(V3 / f"retention/{split}.tasks.jsonl")
        assert len(retained) == 40
        for task in retained:
            assert task["id"] in reviewed and task["id"] not in tasks
            result = execute(Path(task["db_path"]), task["reference_sql"])
            original = execute(Path(task["db_path"]), task["original_reference_sql"])
            stored = task["stored_gold_check"]
            assert result.error is None and len(result.rows) == len(result.rows[0]) == 1
            assert rows_equal(result.rows, task["gold_rows"])
            assert check_answer(str(result.rows[0][0]), stored["stored_gold"], stored["kind"])
            assert not original.error and check_answer(str(original.rows[0][0]), stored["stored_gold"], stored["kind"])
            assert stored["projected_gold_pass"] and task["ordering_policy"]["eligible"]
            retention_count += 1
    code = json.loads((V3 / "code-snapshot.json").read_text())
    for filename, sha in code.items():
        assert fingerprint(V3 / "code" / filename) == sha
        assert fingerprint(Path(filename)) == sha, "active source changed since replay snapshot: " + filename
    result = {"passed": True, "selected_counts": prep["selected_counts"], "training_episodes": len(episodes),
              "manual_sqale_prompts": len(decisions), "manual_retention_prompts": retention_count,
              "repair_kinds": dict(repair_counts), "nested_stages": list(prep["stages"]),
              "max_input_tokens": max(e["n_input_tokens"] for e in episodes), "source_snapshot_matches": True,
              "v2": verify_v2(), "paid_runs": 0, "model_runs": 0}
    if not prepared_only:
        manifest = json.loads((V3 / "manifest.json").read_text())
        for filename, sha in manifest["output_sha256"].items():
            assert fingerprint(V3 / filename) == sha, filename
        count = 0
        with (V3 / "replay-records.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                assert record["metadata"]["format"] == "sql-submit-v1"
                count += 1
        assert count == len(episodes) + sum(len(pools[s]) for s in ("dev", "final", "q2")) + 80
        audit = json.loads((V3 / "auditkit.json").read_text())
        assert len(audit["verdicts"]) == count and all(v == "success" for v in audit["verdicts"].values())
        submissions = read_lines(V3 / "submissions.jsonl")
        assert len(submissions) == 2 * count
        assert len(read_lines(V3 / "preprojection-submissions.jsonl")) == 235
        result.update(replay_audited=count, auditkit_pass=count, final_pool_submission_records=len(submissions),
                      actual_logged_preparation_submissions=manifest.get("actual_logged_preparation_submissions", len(submissions) + 235))
    write_json(V3 / ("verification-prepared.json" if prepared_only else "verification.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-only", action="store_true")
    verify(parser.parse_args().prepared_only)
