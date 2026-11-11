"""Free local v3 preparation from frozen v2 copies; never loads model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from .engine import official_tokenizer
from .promptquality import V2, V3, clean_prompt, fingerprint, reviewed_decisions
from .shopprojection import scalar_reference
from .sftdata import replay_audit
from .sqalebuild import write_json, write_lines
from .sqlenv import FALLBACK, SYSTEM_PROMPT
from .sqlmutate import make_mutation
from .sqlpolicy import ordering_policy
from .sqlscore import execute, rows_equal
from .sqlsftdata import auditkit_check, encode_episode, make_episode
from .tasks import PINNED_LF, check_answer, generate_eval, lf_sha256, load_tasks

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = "sqale-v3-local-20261107-01"


def read_lines(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def verify_v2():
    before = json.loads((V3 / "v2-inventory-before.json").read_text(encoding="utf-8"))
    after = {p.relative_to(V2).as_posix(): {"sha256": fingerprint(p), "bytes": p.stat().st_size}
             for p in sorted(V2.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("v2 inventory changed")
    return {"unchanged": True, "files": len(after), "bytes": sum(r["bytes"] for r in after.values()),
            "inventory_sha256": fingerprint(V3 / "v2-inventory-before.json")}


def prepare():
    if (V3 / "preparation.json").exists() or (V3 / "train.episodes.jsonl").exists():
        raise ValueError("v3 preparation output already exists")
    decisions = reviewed_decisions()
    write_lines(V3 / "prompt-review.jsonl", decisions.values())
    tokenizer = official_tokenizer(str(ROOT / "data/sqale/tokenizer"))
    exclusions, changes, mutation_log, ordering_log = [], [], [], []
    copied = {}

    def copy_db(task):
        old = Path(task["db_path"])
        if old not in copied:
            if not old.resolve().is_relative_to((ROOT / V2).resolve()):
                raise ValueError("only frozen v2 databases may be copied")
            new = V3 / "db" / old.name
            new.parent.mkdir(exist_ok=True)
            if new.exists():
                raise ValueError("database name collision")
            shutil.copy2(old, new)
            if fingerprint(old) != fingerprint(new):
                raise ValueError("copy hash mismatch")
            copied[old] = new
        task["db_path"] = copied[old].as_posix()
        task["db_sha256"] = fingerprint(copied[old])

    def check_task(source):
        task = dict(source)
        manual = decisions.get(task["id"])
        cleaned, reason, edits = clean_prompt(task["prompt"])
        reject = manual["manual_reason"] if manual else reason
        if reject:
            exclusions.append({"scope": "task", "id": task["id"], "split": task["split"],
                               "reason": reject, "raw_prompt": task["prompt"]})
            return None
        if edits:
            changes.append({"id": task["id"], "scope": "primary_prompt", "changes": edits,
                            "raw_prompt": task["prompt"], "clean_prompt": cleaned})
        task["prompt"] = cleaned
        phrasings = {}
        for name, prompt in task.get("phrasings", {}).items():
            text, why, edits = clean_prompt(prompt)
            if why or edits:
                changes.append({"id": task["id"], "scope": "alternative_phrasing", "phrasing": name,
                                "raw_prompt": prompt, "clean_prompt": text, "reason": why, "changes": edits,
                                "decision": "exclude" if why else "strip_prefix"})
            if why is None:
                phrasings[name] = text
        if "phrasings" in task:
            task["phrasings"] = phrasings
        copy_db(task)
        gold = execute(Path(task["db_path"]), task["reference_sql"])
        if gold.error or not rows_equal(gold.rows, task["gold_rows"]):
            raise ValueError("reference result changed: " + task["id"])
        policy = ordering_policy(Path(task["db_path"]), task["reference_sql"])
        ordering_log.append({"id": task["id"], "split": task["split"], **policy})
        if not policy["eligible"]:
            exclusions.append({"scope": "task", "id": task["id"], "split": task["split"],
                               "reason": "ordering: " + policy["reason"]})
            return None
        task["ordered"], task["ordering_policy"] = policy["ordered"], policy
        task["format"] = "sql-submit-v1"
        return task

    heldout = {}
    for split, name in (("dev", "dev-100.tasks.jsonl"), ("final", "final.tasks.jsonl"), ("q2", "q2.tasks.jsonl")):
        heldout[split] = [t for source in read_lines(V2 / name) if (t := check_task(source)) is not None]
        write_lines(V3 / (split + ".tasks.jsonl"), heldout[split])
    old_episodes = {e["task_id"]: e for e in read_lines(V2 / "train.episodes.jsonl")}
    train_tasks, shop_tasks, episodes = [], [], []
    encoded_identities = {}
    for index, source in enumerate(read_lines(V2 / "shop.tasks.jsonl") + read_lines(V2 / "train.tasks.jsonl")):
        task = check_task(source)
        if task is None:
            continue
        old = old_episodes[task["id"]]
        mutation = None
        if old.get("source") == "shop-converted":
            # Preserve the diverse, actually failing shop prefixes; v2's SQaLe
            # bad-alias demos are all regenerated below.
            planned = []
            for message in old["messages"]:
                if message["role"] == "assistant":
                    for call in json.loads(message["content"]):
                        planned.append({"name": call["tool_name"], "args": call["parameters"],
                                        "train": bool(message.get("train"))})
            ep = make_episode(task, calls=planned)
            ep.update(source="shop-converted", original_id=old["original_id"])
        else:
            if old["repair"]:
                mutation = make_mutation(Path(task["db_path"]), task["reference_sql"], task["id"])
                mutation_log.append({"id": task["id"], **mutation})
            ep = make_episode(task, repair=bool(mutation), repair_query=mutation["query"] if mutation else None)
            if mutation:
                ep["repair_kind"] = mutation["kind"]
                ep["repair_error"] = mutation["error"]
        encoded = encode_episode(ep, tokenizer)
        if encoded is None:
            exclusions.append({"scope": "task", "id": task["id"], "split": "train",
                               "reason": "context_over_4096_after_v3_mutation"})
            continue
        ep["n_input_tokens"] = len(encoded["input_ids"])
        ep["n_target_tokens"] = sum(v != -100 for v in encoded["labels"])
        encoded_identities[ep["id"]] = hashlib.sha256(json.dumps(
            [ep["question_id"], encoded["input_ids"], encoded["labels"]], separators=(",", ":")).encode()).hexdigest()
        episodes.append(ep)
        (shop_tasks if ep.get("source") == "shop-converted" else train_tasks).append(task)
        if (index + 1) % 250 == 0:
            print(json.dumps({"prepared_candidates": index + 1, "retained_episodes": len(episodes),
                              "exclusions": len(exclusions)}), flush=True)
    write_lines(V3 / "train.tasks.jsonl", train_tasks)
    write_lines(V3 / "shop.tasks.jsonl", shop_tasks)
    shop_ep = [e for e in episodes if e.get("source") == "shop-converted"]
    sqale_ep = [e for e in episodes if e.get("source") != "shop-converted"]
    stages = {}
    for n in (500, 1500, len(episodes)):
        if n > len(episodes):
            continue
        stage = shop_ep + sqale_ep[:n - len(shop_ep)]
        random.Random(20261107).shuffle(stage)
        write_lines(V3 / f"train-{n}.episodes.jsonl", stage)
        unique = {encoded_identities[e["id"]]: e for e in stage}
        denominator = sum(e["n_target_tokens"] for e in unique.values())
        stages[str(n)] = {"episodes": len(stage), "unique_episodes_per_pass": len(unique),
                          "input_tokens_per_pass": sum(e["n_input_tokens"] for e in stage),
                          "target_tokens_per_pass": sum(e["n_target_tokens"] for e in stage),
                          "unique_target_tokens_per_pass": denominator,
                          "minimum_verified_unique_targets_per_pass": (denominator + 9) // 10}
    random.Random(20261107).shuffle(episodes)
    write_lines(V3 / "train.episodes.jsonl", episodes)
    retention = prepare_retention()
    write_lines(V3 / "exclusions.jsonl", exclusions)
    write_lines(V3 / "prompt-changes.jsonl", changes)
    write_lines(V3 / "mutations.jsonl", mutation_log)
    write_lines(V3 / "ordering-policies.jsonl", ordering_log)
    preparation = {"registration": REGISTRATION, "paid_runs": 0, "model_runs": 0,
                   "selected_counts": {**{s: len(v) for s, v in heldout.items()}, "train": len(train_tasks), "shop": len(shop_tasks)},
                   "train_episodes": len(episodes), "stages": stages,
                   "selected_schema_groups": {**{s: len({t["schema_group"] for t in v}) for s, v in heldout.items()},
                                              "train": len({t["schema_group"] for t in train_tasks})},
                   "manual_prompt_review": {"source_prompts": len(decisions),
                                            "excluded": sum(d["manual_decision"] == "exclude" for d in decisions.values()),
                                            "prefixes_stripped": sum(bool(d["changes"]) for d in decisions.values())},
                   "exclusions": dict(Counter(e["reason"] for e in exclusions)),
                   "repair_kinds": dict(Counter(e.get("repair_kind") for e in episodes if e.get("repair_kind"))),
                   "repair_attempts_not_retained": sum(m["id"] not in {t["id"] for t in train_tasks} for m in mutation_log),
                   "mutation_fallbacks": sum(m["requested_kind"] != m["kind"] for m in mutation_log),
                   "ordering": dict(Counter(p["reason"] for p in ordering_log)),
                   "max_input_tokens": max(e["n_input_tokens"] for e in episodes),
                   "retention": retention, "v2": verify_v2(),
                   "new_dependencies": {"sqlglot": version("sqlglot")},
                   "sealed_shop_test": "not opened, read, generated, evaluated or selected on"}
    write_json(V3 / "preparation.json", preparation)
    print(json.dumps(preparation, indent=2), flush=True)


def prepare_retention():
    directory = V3 / "retention"
    directory.mkdir(exist_ok=False)
    counts, checks = {}, []
    for split in ("dev", "mh"):
        source, dbs = generate_eval(split, directory)
        if lf_sha256(source) != PINNED_LF[split]:
            raise ValueError("retention source tasks do not match original pin: " + split)
        tasks = []
        for original in load_tasks(source):
            db = dbs[original.id]
            result = execute(db, original.reference_sql)
            valid = not result.error and bool(result.rows) and check_answer(str(result.rows[0][0]), original.answer, original.kind)
            check = {"id": original.id, "split": split, "stored_gold": original.answer, "kind": original.kind,
                     "reference_sql": original.reference_sql, "reference_result": result.rows, "error": result.error,
                     "scalar_projection": "first row, first column", "stored_gold_pass": valid,
                     "stored_gold_tolerance": {"int": 0., "money": .011, "float1": .051, "text": "casefold"}[original.kind]}
            checks.append(check)
            if not valid:
                raise ValueError("retention reference fails stored gold: " + original.id)
            projected_sql = scalar_reference(original.reference_sql)
            projected = execute(db, projected_sql)
            if projected.error or len(projected.rows) != 1 or len(projected.rows[0]) != 1 or not check_answer(
                    str(projected.rows[0][0]), original.answer, original.kind):
                raise ValueError("projected retention reference fails gold: " + original.id)
            check.update(projected_reference_sql=projected_sql, projected_result=projected.rows,
                         projected_gold_pass=True, projection_rule="first scalar; unique top row for ranked legacy references")
            policy = ordering_policy(db, projected_sql)
            if not policy["eligible"]:
                raise ValueError("ambiguous retention reference: " + original.id + " " + policy["reason"])
            tasks.append({"id": original.id, "question_id": original.id, "split": "retention-" + split,
                          "schema_id": db.stem, "schema_group": "shop-retention-" + db.stem,
                          "difficulty": original.family, "prompt": original.prompt, "format": "sql-submit-v1",
                          "reference_sql": projected_sql, "original_reference_sql": original.reference_sql,
                          "gold_rows": projected.rows,
                          "gold_tables": sorted(t for t in result.reads if t and not t.startswith("sqlite_")),
                          "db_path": db.as_posix(), "db_sha256": fingerprint(db),
                          "ordering_policy": policy, "ordered": policy["ordered"], "stored_gold_check": check})
        write_lines(directory / f"{split}.tasks.jsonl", tasks)
        counts[split] = {"tasks": len(tasks), "stored_gold_pass": sum(c["stored_gold_pass"] for c in checks if c["split"] == split),
                         "source_lf_sha256": lf_sha256(source), "pinned_lf_sha256": PINNED_LF[split]}
    write_lines(directory / "gold-checks.jsonl", checks)
    return counts


def project_shop():
    """Complete scalar projection before the one replay/audit phase; v2 stays immutable."""
    if (V3 / "shop-projection.json").exists() or (V3 / "replay-records.jsonl").exists():
        raise ValueError("shop projection already completed or replay begun")
    tokenizer = official_tokenizer(str(ROOT / "data/sqale/tokenizer"))
    originals = {e["id"]: e for e in read_lines(Path("data/sft_pilot/episodes.jsonl"))}
    tasks = read_lines(V3 / "shop.tasks.jsonl")
    old_eps = {e["task_id"]: e for e in read_lines(V3 / "train.episodes.jsonl") if e.get("source") == "shop-converted"}
    write_lines(V3 / "preprojection-submissions.jsonl", [
        {"stage": "builder-before-scalar-projection", "episode_id": ep["id"], **call}
        for ep in old_eps.values() for call in ep["calls"] if call["name"] == "submit_sql"])
    replacements, checks = {}, []
    for task in tasks:
        old = old_eps[task["id"]]
        stored = originals[old["original_id"]]
        task["original_reference_sql"] = task["reference_sql"]
        task["reference_sql"] = scalar_reference(task["reference_sql"])
        db = Path(task["db_path"])
        result = execute(db, task["reference_sql"])
        valid = not result.error and len(result.rows) == 1 and len(result.rows[0]) == 1 and check_answer(
            str(result.rows[0][0]), stored["gold"], stored["kind"])
        if not valid:
            raise ValueError("projected shop training reference fails stored gold: " + task["id"])
        policy = ordering_policy(db, task["reference_sql"])
        if not policy["eligible"]:
            raise ValueError("shop training projection ambiguous: " + task["id"])
        task.update(gold_rows=result.rows, ordered=policy["ordered"], ordering_policy=policy)
        calls = []
        for message in old["messages"]:
            if message["role"] == "assistant":
                for c in json.loads(message["content"]):
                    calls.append({"name": c["tool_name"], "args": c["parameters"], "train": bool(message.get("train"))})
        last_run = max(i for i, c in enumerate(calls) if c["name"] == "run_sql")
        calls[last_run]["args"] = {"query": task["reference_sql"]}
        for c in calls:
            if c["name"] == "submit_sql":
                c["args"] = {"query": task["reference_sql"]}
        ep = make_episode(task, calls=calls)
        ep.update(source="shop-converted", original_id=old["original_id"])
        encoded = encode_episode(ep, tokenizer)
        if encoded is None:
            raise ValueError("shop projection context limit")
        ep["n_input_tokens"] = len(encoded["input_ids"])
        ep["n_target_tokens"] = sum(v != -100 for v in encoded["labels"])
        replacements[task["id"]] = ep
        checks.append({"id": task["id"], "stored_gold": stored["gold"], "kind": stored["kind"],
                       "original_reference_sql": task["original_reference_sql"], "reference_sql": task["reference_sql"],
                       "reference_result": result.rows, "stored_gold_pass": valid, "ordering_policy": policy})
    write_lines(V3 / "shop.tasks.jsonl", tasks)
    prep = json.loads((V3 / "preparation.json").read_text(encoding="utf-8"))
    shutil.copy2(V3 / "preparation.json", V3 / "preparation-before-shop-projection.json")
    for filename in ["train.episodes.jsonl"] + [f"train-{n}.episodes.jsonl" for n in prep["stages"]]:
        values = [replacements.get(ep["task_id"], ep) for ep in read_lines(V3 / filename)]
        write_lines(V3 / filename, values)
        n = str(len(values))
        stage = prep["stages"][n]
        stage["input_tokens_per_pass"] = sum(e["n_input_tokens"] for e in values)
        stage["target_tokens_per_pass"] = sum(e["n_target_tokens"] for e in values)
        # All task IDs are unique in each nested stage, verified again before freezing.
        assert len({e["question_id"] for e in values}) == len(values)
        stage["unique_target_tokens_per_pass"] = stage["target_tokens_per_pass"]
        stage["minimum_verified_unique_targets_per_pass"] = (stage["unique_target_tokens_per_pass"] + 9) // 10
    # The prepare phase might have run before the scalar-projection helper was loaded.
    # Rebuild only the v3 retention scorer records, preserving the original generated pins.
    for split in ("dev", "mh"):
        retention_tasks = read_lines(V3 / f"retention/{split}.tasks.jsonl")
        for task in retention_tasks:
            original_sql = task.get("original_reference_sql", task["reference_sql"])
            sql = scalar_reference(original_sql)
            result = execute(Path(task["db_path"]), sql)
            check = task["stored_gold_check"]
            valid = not result.error and len(result.rows) == 1 and len(result.rows[0]) == 1 and check_answer(
                str(result.rows[0][0]), check["stored_gold"], check["kind"])
            if not valid:
                raise ValueError("retention scalar projection failed")
            policy = ordering_policy(Path(task["db_path"]), sql)
            if not policy["eligible"]:
                raise ValueError("retention scalar projection ambiguous")
            task.update(reference_sql=sql, original_reference_sql=original_sql, gold_rows=result.rows,
                        ordered=policy["ordered"], ordering_policy=policy)
            check.update(projected_reference_sql=sql, projected_result=result.rows, projected_gold_pass=True,
                         projection_rule="first scalar; unique top row for ranked legacy references")
        write_lines(V3 / f"retention/{split}.tasks.jsonl", retention_tasks)
    write_lines(V3 / "retention/gold-checks.jsonl", [t["stored_gold_check"] for split in ("dev", "mh")
                                                   for t in read_lines(V3 / f"retention/{split}.tasks.jsonl")])
    write_lines(V3 / "shop-gold-checks.jsonl", checks)
    by_id = {task["id"]: task for task in tasks}
    policies = [{"id": p["id"], "split": p["split"], **by_id[p["id"]]["ordering_policy"]}
                if p["id"] in by_id else p for p in read_lines(V3 / "ordering-policies.jsonl")]
    write_lines(V3 / "ordering-policies.jsonl", policies)
    prep["ordering"] = dict(Counter(p["reason"] for p in policies))
    summary = {"training_tasks": len(checks), "stored_gold_pass": sum(c["stored_gold_pass"] for c in checks),
               "retention_tasks": 80, "scalar_projection": "all v3 shop references match the requested scalar",
               "legacy_records_unchanged": True}
    write_json(V3 / "shop-projection.json", summary)
    prep["shop_projection"] = summary
    write_json(V3 / "preparation.json", prep)
    print(json.dumps(summary))


def audit():
    if (V3 / "manifest.json").exists() or (V3 / "replay-records.jsonl").exists():
        raise ValueError("v3 replay output already exists; do not silently repeat a registration")
    prep = json.loads((V3 / "preparation.json").read_text(encoding="utf-8"))
    if "shop_projection" not in prep:
        raise ValueError("complete the scalar shop projection before auditing")
    review = json.loads((V3 / "retention/manual-review.json").read_text(encoding="utf-8"))
    if review["read_individually"] != 80:
        raise ValueError("retention prompt review incomplete")
    snapshot = V3 / "code"
    snapshot.mkdir(exist_ok=False)
    code_files = [Path("src/ayadb") / name for name in (
        "sqalebuild_v3.py", "promptquality.py", "sqlmutate.py", "sqlpolicy.py", "sqlscore.py",
        "sqlsftdata.py", "sqlenv.py", "sftdata.py", "tokenbudget.py", "shopprojection.py", "agenttune_sft.py")]
    code_files += [Path("experiments/sqale-sft") / name for name in (
        "PREREGISTRATION.md", "AMENDMENT-2-v3-local-fixes.md")]
    code_files += [Path("requirements-sqale-v3.txt"), Path("tests/test_sqale_v3.py")]
    for path in code_files:
        destination = snapshot / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    code_manifest = {p.as_posix(): fingerprint(p) for p in code_files}
    write_json(V3 / "code-snapshot.json", code_manifest)
    tokenizer = official_tokenizer(str(ROOT / "data/sqale/tokenizer"))
    episodes = read_lines(V3 / "train.episodes.jsonl")
    tasks = {t["id"]: t for filename in ("train.tasks.jsonl", "shop.tasks.jsonl", "dev.tasks.jsonl", "final.tasks.jsonl", "q2.tasks.jsonl")
             for t in read_lines(V3 / filename)}
    for split in ("dev", "mh"):
        for task in read_lines(V3 / f"retention/{split}.tasks.jsonl"):
            tasks[task["id"]] = task
    # Also replay every held-out gold path to check the independent protocol and
    # complete result metric; these scripted records are never model evaluations.
    training_ids = {e["task_id"] for e in episodes}
    episodes += [make_episode(task) for tid, task in tasks.items() if tid not in training_ids]
    replay_tasks = {}
    with (V3 / "replay-records.jsonl").open("w", encoding="utf-8", newline="\n") as records, \
         (V3 / "submissions.jsonl").open("w", encoding="utf-8", newline="\n") as submissions:
        for i, ep in enumerate(episodes):
            encoded = encode_episode(ep, tokenizer)
            if encoded is None:
                raise ValueError("held-out expert path exceeds context: " + ep["id"])
            problems = replay_audit(ep, None, ep["db_sha256"], tokenizer, SYSTEM_PROMPT, FALLBACK,
                                    sql_db=Path(ep["db_path"]))
            if problems:
                raise ValueError("replay failed: " + ep["id"] + " " + "; ".join(problems))
            record = ep.pop("_replay_record")
            records.write(json.dumps(record) + "\n")
            for stage, log in (("builder", ep["calls"]), ("AgentTune-replay", record["metadata"]["call_log"])):
                for call in log:
                    if call["name"] == "submit_sql":
                        submissions.write(json.dumps({"stage": stage, "episode_id": ep["id"], **call}) + "\n")
            replay_tasks[ep["id"]] = tasks[ep["task_id"]]
            if (i + 1) % 100 == 0:
                print(json.dumps({"replay_passed": i + 1, "total": len(episodes)}), flush=True)
    audit_result = auditkit_check(V3 / "replay-records.jsonl", replay_tasks)
    write_json(V3 / "auditkit.json", audit_result)
    successes = sum(v == "success" for v in audit_result["verdicts"].values())
    if successes != len(episodes):
        raise ValueError("AuditKIT verification failed")
    manifest = {**prep, "replay_audited": len(episodes), "auditkit_pass": successes,
                "v2": verify_v2(), "retention_manual_review": review, "code_snapshot": code_manifest,
                "amendment_sha256": fingerprint(Path("experiments/sqale-sft/AMENDMENT-2-v3-local-fixes.md")),
                "output_sha256": {p.relative_to(V3).as_posix(): fingerprint(p) for p in sorted(V3.rglob("*.json*"))
                                  if p.name != "v2-inventory-before.json"}}
    write_json(V3 / "manifest.json", manifest)
    print(json.dumps({"replay_audited": len(episodes), "auditkit_pass": successes, "v2": manifest["v2"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "project-shop", "audit", "verify-v2"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "audit":
        audit()
    elif args.phase == "project-shop":
        project_shop()
    else:
        print(json.dumps(verify_v2()))


if __name__ == "__main__":
    main()
