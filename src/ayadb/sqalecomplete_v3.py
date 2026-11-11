"""Finish v3 prompt curation, preserving the initial free audit and rechecking cached replays."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

from .engine import official_tokenizer
from .promptquality import V3, clean_prompt, fingerprint, reviewed_decisions
from .sftdata import conversation, replay_audit, system_message
from .sqalebuild import write_json, write_lines
from .sqalebuild_v3 import read_lines, verify_v2
from .sqlenv import FALLBACK, RESULT_FORMAT, SYSTEM_PROMPT, SQLTools
from .sqlscore import score_episode
from .sqlsftdata import auditkit_check, encode_episode, make_episode

OLD = V3 / "pre-prompt-completion"
REGISTRATION = "sqale-v3-local-20261107-02"


def prepare():
    if not (V3 / "manifest.json").exists() or OLD.exists():
        raise ValueError("initial audit must finish; prompt completion may run only once")
    OLD.mkdir()
    # Every source/destination is checked before moving within the named v3 workspace.
    preserve = {"manual-review-sheet.jsonl", "manual-review-decisions.json", "v2-inventory-before.json"}
    for source in list(V3.glob("*.json*")) + [V3 / "code"]:
        if source.name in preserve:
            continue
        destination = OLD / source.name
        if not source.resolve().is_relative_to(V3.resolve()) or not destination.resolve().is_relative_to(V3.resolve()):
            raise ValueError("archive path outside v3")
        source.rename(destination)
    prep = json.loads((OLD / "preparation.json").read_text())
    prep["registration"] = REGISTRATION
    exclusions = read_lines(OLD / "exclusions.jsonl")
    changes = read_lines(OLD / "prompt-changes.jsonl")
    pools = {}
    delta = []
    for split in ("train", "shop", "dev", "final", "q2"):
        pools[split] = []
        for source in read_lines(OLD / f"{split}.tasks.jsonl"):
            task = copy.deepcopy(source)
            prompt, reason, edits = clean_prompt(task["prompt"])
            if reason:
                if split != "train":
                    raise ValueError("unexpected change to reviewed/retention primary prompt")
                exclusions.append({"id": task["id"], "split": split, "scope": "task", "reason": reason,
                                   "raw_prompt": task["prompt"], "phase": REGISTRATION})
                delta.append({"id": task["id"], "decision": "exclude", "reason": reason,
                              "prompt": task["prompt"], "review_method": "assistant_full_text_individual_read_of_flagged_prompt"})
                continue
            if edits:
                delta.append({"id": task["id"], "decision": "edit", "before": task["prompt"], "after": prompt,
                              "changes": edits, "review_method": "assistant_full_text_individual_read_of_flagged_prompt"})
                changes.append({"id": task["id"], "scope": "primary_prompt", "raw_prompt": task["prompt"],
                                "clean_prompt": prompt, "changes": edits, "phase": REGISTRATION})
            task["prompt"] = prompt
            alternatives = {}
            for key, text in task.get("phrasings", {}).items():
                clean, why, edits = clean_prompt(text)
                if why or edits:
                    changes.append({"id": task["id"], "scope": "alternative_phrasing", "phrasing": key,
                                    "raw_prompt": text, "clean_prompt": clean, "changes": edits, "reason": why,
                                    "decision": "exclude" if why else "strip_prefix", "phase": REGISTRATION})
                if why is None:
                    alternatives[key] = clean
            if "phrasings" in task:
                task["phrasings"] = alternatives
            pools[split].append(task)
        write_lines(V3 / f"{split}.tasks.jsonl", pools[split])
    tasks = {t["id"]: t for rows in pools.values() for t in rows}
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    episodes, fresh_submissions = [], []
    for old in read_lines(OLD / "train.episodes.jsonl"):
        if old["task_id"] not in tasks:
            continue
        task = tasks[old["task_id"]]
        ep = old
        if task["prompt"] != old["prompt"]:
            planned = [{"name": c["tool_name"], "args": c["parameters"], "train": bool(m["train"])}
                       for m in old["messages"] if m["role"] == "assistant" for c in json.loads(m["content"])]
            ep = {**old, **make_episode(task, repair=old["repair"], calls=planned)}
            for call in ep["calls"]:
                if call["name"] == "submit_sql":
                    fresh_submissions.append({"stage": "prompt-completion-builder", "episode_id": ep["id"], **call})
        encoded = encode_episode(ep, tokenizer)
        if encoded is None:
            raise ValueError("context limit after prompt cleanup")
        ep["n_input_tokens"] = len(encoded["input_ids"])
        ep["n_target_tokens"] = sum(v != -100 for v in encoded["labels"])
        episodes.append(ep)
    write_lines(V3 / "train.episodes.jsonl", episodes)
    write_lines(V3 / "new-builder-submissions.jsonl", fresh_submissions)
    by_task = {e["task_id"]: e for e in episodes}
    sqale = [by_task[t["id"]] for t in pools["train"]]
    shop = [by_task[t["id"]] for t in pools["shop"]]
    stages = {}
    for n in (500, 1500, len(episodes)):
        stage = shop + sqale[:n - len(shop)]
        random.Random(20261107).shuffle(stage)
        assert len({e["question_id"] for e in stage}) == len(stage) == n
        write_lines(V3 / f"train-{n}.episodes.jsonl", stage)
        targets = sum(e["n_target_tokens"] for e in stage)
        stages[str(n)] = {"episodes": n, "unique_episodes_per_pass": n,
                          "input_tokens_per_pass": sum(e["n_input_tokens"] for e in stage),
                          "target_tokens_per_pass": targets, "unique_target_tokens_per_pass": targets,
                          "minimum_verified_unique_targets_per_pass": (targets + 9) // 10}
    mutations = read_lines(OLD / "mutations.jsonl")
    for mutation in mutations:
        mutation["retained"] = mutation["id"] in by_task
    policies = [p for p in read_lines(OLD / "ordering-policies.jsonl") if p["id"] in tasks or not p["eligible"]]
    for filename, rows in (("exclusions.jsonl", exclusions), ("prompt-changes.jsonl", changes),
                           ("prompt-completion-decisions.jsonl", delta), ("mutations.jsonl", mutations),
                           ("ordering-policies.jsonl", policies), ("prompt-review.jsonl", reviewed_decisions().values())):
        write_lines(V3 / filename, rows)
    for name in ("shop-gold-checks.jsonl", "shop-projection.json", "preprojection-submissions.jsonl"):
        shutil.copy2(OLD / name, V3 / name)
    prep.update(selected_counts={s: len(v) for s, v in pools.items()}, train_episodes=len(episodes), stages=stages,
                selected_schema_groups={s: len({t["schema_group"] for t in rows}) for s, rows in pools.items() if s != "shop"},
                exclusions=dict(Counter(e["reason"] for e in exclusions)),
                repair_kinds=dict(Counter(e["repair_kind"] for e in episodes if e.get("repair_kind"))),
                repair_attempts_not_retained=sum(not m["retained"] for m in mutations),
                ordering=dict(Counter(p["reason"] for p in policies)),
                max_input_tokens=max(e["n_input_tokens"] for e in episodes),
                prompt_completion={"additional_exclusions": sum(d["decision"] == "exclude" for d in delta),
                                   "additional_primary_edits": len(fresh_submissions), "intermediate_preserved": str(OLD)})
    prep["v2"] = verify_v2()
    write_json(V3 / "preparation.json", prep)
    print(json.dumps({k: prep[k] for k in ("selected_counts", "stages", "repair_kinds", "prompt_completion")}, indent=2))


def audit():
    if (V3 / "replay-records.jsonl").exists():
        raise ValueError("final audit output already exists")
    prep = json.loads((V3 / "preparation.json").read_text())
    if prep["registration"] != REGISTRATION:
        raise ValueError("wrong registration")
    snapshot = V3 / "code"
    snapshot.mkdir(exist_ok=False)
    paths = list(json.loads((OLD / "code-snapshot.json").read_text())) + [
        "src/ayadb/sqalecomplete_v3.py", "src/ayadb/sqaleverify_v3.py",
        "experiments/sqale-sft/V3-PROMPT-COMPLETION-REGISTRATION.md"]
    code = {}
    for name in paths:
        source, destination = Path(name), snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        code[name] = fingerprint(source)
    write_json(V3 / "code-snapshot.json", code)
    tokenizer = official_tokenizer("data/sqale/tokenizer")
    tasks = {t["id"]: t for split in ("train", "shop", "dev", "final", "q2") for t in read_lines(V3 / f"{split}.tasks.jsonl")}
    for split in ("dev", "mh"):
        tasks.update({t["id"]: t for t in read_lines(V3 / f"retention/{split}.tasks.jsonl")})
    episodes = read_lines(V3 / "train.episodes.jsonl")
    trained = {e["task_id"] for e in episodes}
    new_heldout_builders = [make_episode(t) for tid, t in tasks.items() if tid not in trained]
    episodes += new_heldout_builders
    remaining = {e["id"]: e for e in episodes}
    replay_tasks, reused, fresh = {}, 0, 0
    cache_path = OLD / "replay-records.jsonl"
    cache_sha = fingerprint(cache_path)
    with (V3 / "replay-records.jsonl").open("w", encoding="utf-8", newline="\n") as records, \
         (V3 / "submissions.jsonl").open("w", encoding="utf-8", newline="\n") as submissions, \
         (V3 / "fresh-submissions.jsonl").open("w", encoding="utf-8", newline="\n") as new_submissions:
        for row in read_lines(V3 / "new-builder-submissions.jsonl"):
            new_submissions.write(json.dumps(row) + "\n")
        for ep in new_heldout_builders:
            for call in ep["calls"]:
                if call["name"] == "submit_sql":
                    new_submissions.write(json.dumps({"stage": "prompt-completion-heldout-builder",
                                                      "episode_id": ep["id"], **call}) + "\n")

        def accept(ep, record, provenance):
            records.write(json.dumps(record) + "\n")
            for stage, log in (("builder", ep["calls"]), (provenance, record["metadata"]["call_log"])):
                for call in log:
                    if call["name"] == "submit_sql":
                        row = {"stage": stage, "episode_id": ep["id"], **call}
                        submissions.write(json.dumps(row) + "\n")
                        if stage == "fresh-AgentTune-replay":
                            new_submissions.write(json.dumps(row) + "\n")
            replay_tasks[ep["id"]] = tasks[ep["task_id"]]

        with cache_path.open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                identifier = record["trajectory_id"]
                ep = remaining.get(identifier)
                if ep is None:
                    continue
                schemas = [_build_tool_schema(fn) for fn in SQLTools(Path(ep["db_path"]), ep["task_id"], ep.get("discovery_order"))]
                expected = conversation(ep["prompt"], ep["messages"], system_message(SYSTEM_PROMPT, FALLBACK, schemas))
                # AgentTune appends the actual submit observation and an empty final
                # assistant turn. Check this terminal tail too, not just a prefix.
                expected += [{"role": "user", "content": RESULT_FORMAT.format(
                    name="submit_sql", content=ep["calls"][-1]["result"])}, {"role": "assistant", "content": ""}]
                normal = lambda conv: [(m["role"], m.get("content")) for m in conv]
                calls = lambda log: [(c["name"], c["args"], c["result"]) for c in log]
                if (normal(record["metadata"]["conversation"]) != normal(expected)
                        or calls(record["metadata"]["call_log"]) != calls(ep["calls"])):
                    continue
                if fingerprint(Path(ep["db_path"])) != ep["db_sha256"]:
                    raise ValueError("database changed")
                if not score_episode(Path(ep["db_path"]), ep["reference_sql"], record["metadata"]["call_log"])["success"]:
                    raise ValueError("cached final result no longer passes")
                record["metadata"]["replay_cache"] = {"file": cache_path.as_posix(), "sha256": cache_sha,
                                                         "registration": "sqale-v3-local-20261107-01"}
                accept(ep, record, "cached-AgentTune-replay")
                del remaining[identifier]
                reused += 1
                if reused % 500 == 0:
                    print(json.dumps({"reused_after_exact_checks": reused}), flush=True)
        for ep in remaining.values():
            if encode_episode(ep, tokenizer) is None:
                raise ValueError("context limit on fresh replay")
            issues = replay_audit(ep, None, ep["db_sha256"], tokenizer, SYSTEM_PROMPT, FALLBACK,
                                  sql_db=Path(ep["db_path"]))
            if issues:
                raise ValueError("fresh replay failed: " + str(issues))
            record = ep.pop("_replay_record")
            accept(ep, record, "fresh-AgentTune-replay")
            fresh += 1
    result = auditkit_check(V3 / "replay-records.jsonl", replay_tasks)
    write_json(V3 / "auditkit.json", result)
    successes = sum(v == "success" for v in result["verdicts"].values())
    if successes != len(episodes):
        raise ValueError("final AuditKIT failure")
    manifest = {**prep, "code_snapshot": code, "replay_audited": len(episodes), "auditkit_pass": successes,
                "reused_deterministic_replays": reused, "fresh_replays": fresh, "v2": verify_v2(),
                "actual_logged_preparation_submissions": len(read_lines(OLD / "submissions.jsonl")) + 235
                    + len(read_lines(V3 / "fresh-submissions.jsonl")),
                "retention_manual_review": json.loads((V3 / "retention/manual-review.json").read_text()),
                "amendment_sha256": fingerprint(Path("experiments/sqale-sft/AMENDMENT-2-v3-local-fixes.md")),
                "output_sha256": {p.relative_to(V3).as_posix(): fingerprint(p) for p in sorted(V3.glob("*.json*"))}}
    manifest["output_sha256"].update({p.relative_to(V3).as_posix(): fingerprint(p)
                                       for p in sorted((V3 / "retention").glob("*.json*"))})
    write_json(V3 / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("replay_audited", "auditkit_pass", "reused_deterministic_replays",
                                              "fresh_replays", "actual_logged_preparation_submissions")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "audit"))
    (prepare if parser.parse_args().phase == "prepare" else audit)()
