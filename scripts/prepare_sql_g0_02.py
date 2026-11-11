"""Build the additive G0-02 draft locally; never loads weights, uploads or launches."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.runinputs import eligible_training_rows, input_count, read_jsonl
from ayadb.runledger import atomic_json
from ayadb.sqlsftdata import encode_episode

RUN_ID = "g0-sql-agenttune-20261107-02"
ORIGINAL = Path("experiments/sql-g0/REGISTRATION.json")
REGISTRATION = Path("experiments/sql-g0-02/REGISTRATION.json")
ORIGINAL_SHA = "84799ec8cbdd3a57602b09084e3bf8aec1f82f572b5baabeb71a5d51828c8c81"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-draft", action="store_true")
    args = parser.parse_args()
    receipt = Path("runs/hf-sql/hf-user") / RUN_ID
    if (receipt / "launch.json").exists() or (receipt / "upload.json").exists():
        raise ValueError("G0-02 has external activity; never revise or retry")
    if fingerprint(ORIGINAL) != ORIGINAL_SHA:
        raise ValueError("original G0 registration changed")
    reg = copy.deepcopy(json.loads(ORIGINAL.read_text(encoding="utf-8")))
    out = Path("data/sql-g0-02-v1")
    out.mkdir(parents=True, exist_ok=args.refresh_draft)
    rows, _ = eligible_training_rows(read_jsonl(Path("data/sqale/data-build-v3/train-4888.episodes.jsonl")))
    short = min(rows, key=lambda row: (input_count(row), row["id"]))
    tokenizer = official_tokenizer(reg["model_config"]["tokenizer"])
    for row in (short, read_jsonl(Path("data/sql-g0-v1/longest.episodes.jsonl"))[0]):
        encoded = encode_episode(row, tokenizer, 4096)
        if encoded is None or len(encoded["input_ids"]) != row["n_input_tokens"] or sum(v != -100 for v in encoded["labels"]) != row["n_target_tokens"]:
            raise ValueError("fresh short/long encoding differs from frozen input or masks")
    short_file = out / "short.episodes.jsonl"
    short_file.write_text(json.dumps(short, ensure_ascii=True) + "\n", encoding="utf-8")
    reg["id"] = RUN_ID
    for step in reg["steps"]:
        for key in ("out", "resume", "adapter"):
            if step.get(key):
                step[key] = step[key].replace("g0-sql-agenttune-20261107-01", RUN_ID)
        if step["action"] == "train":
            step["loss_policy"] = "trained-positions-v1"
        else:
            step["inference_logits_to_keep"] = 1
    reg["steps"].insert(0, {"name": "short-loss-equivalence", "action": "check_loss",
        "episodes": short_file.as_posix(), "out": f"runs/sql/{RUN_ID}/short-loss-equivalence",
        "seed": 42, "max_short_input_tokens": short["n_input_tokens"], "rtol": 1e-4, "atol": 1e-4,
        "timeout_seconds": 600})
    for filename, kind in ((short_file, "short_training_episode"), (Path(short["db_path"]), "sqale_database")):
        if not any(a["path"] == filename.as_posix() for a in reg["assets"]):
            reg["assets"].append({"path": filename.as_posix(), "kind": kind,
                                  "sha256": fingerprint(filename), "bytes": filename.stat().st_size})
    reg["assets"].sort(key=lambda a: a["path"])
    source_files = list(Path("src/ayadb").glob("*.py")) + [Path(p) for p in (
        "scripts/sql_job.py", "scripts/prepare_sql_g0_02.py", "requirements-sql-runner.txt", "pyproject.toml",
        "experiments/sql-g0/REGISTRATION.json", "experiments/sqale-sft/RUN-POLICY-v1.json",
        "experiments/sqale-sft/AMENDMENT-4-run-inputs-and-control.md",
        "experiments/sqale-sft/AMENDMENT-5-table-reads-and-manual-audit.md",
        "experiments/sqale-sft/AMENDMENT-6-selected-logits-g0-02.md")]
    reg["source_sha256"] = {p.as_posix(): fingerprint(p) for p in sorted(source_files)}
    reg.update(scorer_policy="table-read-v1", loss_policy="trained-positions-v1", inference_logits_to_keep=1,
        required_environment={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        original_g0_registration_sha256=ORIGINAL_SHA, short_example_id=short["id"],
        short_input_tokens=short["n_input_tokens"], short_target_tokens=short["n_target_tokens"],
        authorization="Draft only: new explicit approval must cover uploads, one job and bounded storage")
    reg["pass_criteria"] = [
        "short whole-example GPU native full/selected losses agree at rtol=1e-4 atol=1e-4 in NF4/fp16 autocast",
        "short-check lm_head projects full inputs versus selected shifted targets; no optimizer updates",
        *reg["pass_criteria"],
        "native selected loss keeps the accumulation-window num_items_in_batch and audited masks",
        "memory snapshots persist after model load, each update and any OOM",
        "inference projects exactly one position on prefill and decode; table-read-v1 AuditKIT agrees"]
    reg["abort_criteria"].extend(["short native loss mismatch/nonfinite or wrong projection count",
                                  "missing memory measurement or full-prompt inference projection"])
    REGISTRATION.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(REGISTRATION, reg)
    metadata = {"registration": REGISTRATION.as_posix(), "sha256": fingerprint(REGISTRATION),
                "asset_files": len(reg["assets"]), "asset_bytes": sum(a["bytes"] for a in reg["assets"]),
                "short_input_tokens": short["n_input_tokens"], "short_target_tokens": short["n_target_tokens"],
                "uploads": 0, "jobs_started": 0}
    atomic_json(out / "preparation.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
