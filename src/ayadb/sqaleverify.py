"""Verify prepared artifacts, split isolation and stages without model calls or the old test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .tasks import file_sha256


def load(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def verify(out: Path, amended_dev: bool = False) -> dict:
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["output_sha256"].items():
        if file_sha256(out / name) != expected:
            raise ValueError("output checksum mismatch: " + name)
    tasks = {split: load(out / (split + ".tasks.jsonl")) for split in ("train", "q2", "dev", "final")}
    if amended_dev:
        change = json.loads((out / "dev-100.manifest.json").read_text(encoding="utf-8"))
        if change["base_manifest_sha256"] != file_sha256(out / "manifest.json"):
            raise ValueError("dev amendment does not reference this base build")
        if change["sha256"] != file_sha256(out / change["file"]):
            raise ValueError("amended dev checksum mismatch")
        amended = load(out / change["file"])
        by_question = {r["question_id"]: r for r in amended}
        if any(by_question.get(r["question_id"]) != r for r in tasks["dev"]):
            raise ValueError("original dev tasks changed in amendment")
        tasks["dev"] = amended
    for split, rows in tasks.items():
        if len({r["question_id"] for r in rows}) != len(rows):
            raise ValueError("duplicate question in " + split)
    groups = {split: {r["schema_group"] for r in rows} for split, rows in tasks.items()}
    for a in groups:
        for b in groups:
            if a != b and groups[a] & groups[b]:
                raise ValueError("schema group crosses " + a + "/" + b)
    episodes = load(out / "train.episodes.jsonl")
    by_id = {ep["id"]: ep for ep in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("duplicate training episode")
    for ep in episodes:
        if not 0 < ep["n_target_tokens"] <= ep["n_input_tokens"] <= 4096:
            raise ValueError("invalid token budget")
        if ep["format"] != "sql-submit-v1" or not ep["calls"][-1].get("accepted"):
            raise ValueError("missing terminal SQL submission")
        if file_sha256(Path(ep["db_path"])) != ep["db_sha256"]:
            raise ValueError("prepared database changed")
    previous = set()
    stages = {}
    for n in (500, 1500, 5000):
        path = out / f"train-{n}.episodes.jsonl"
        if not path.exists():
            continue
        stage = load(path)
        ids = {ep["id"] for ep in stage}
        if len(ids) != n or not previous <= ids <= by_id.keys():
            raise ValueError("stages are not nested subsets")
        if sum(ep.get("source") == "shop-converted" for ep in stage) != 235:
            raise ValueError("stage lost a converted shop episode")
        previous = ids
        stages[n] = len(ids)
    expected = manifest["requested"]
    shortfall = {split: max(0, expected[split] - len(tasks[split])) for split in ("q2", "dev", "final")}
    shortfall["train_episodes"] = max(0, expected["train_episodes"] - len(episodes))
    if manifest["auditkit_pass"] != manifest["replay_audited"]:
        raise ValueError("AuditKIT replay agreement incomplete")
    return {"checks": "passed", "manifest_sha256": file_sha256(out / "manifest.json"),
            "episodes": len(episodes), "max_input_tokens": max(e["n_input_tokens"] for e in episodes),
            "selected_tasks": {s: len(rows) for s, rows in tasks.items()},
            "schema_groups": {s: len(values) for s, values in groups.items()},
            "nested_stages": stages, "shortfall": shortfall, "model_calls": 0,
            "sealed_shop_test": "not read"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("--amended-dev", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.build, args.amended_dev)
    filename = "verification-dev100.json" if args.amended_dev else "verification.json"
    (args.build / filename).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(any(result["shortfall"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
