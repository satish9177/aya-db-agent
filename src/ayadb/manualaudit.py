"""Freeze a diagnostic dev-audit sample; labels must be entered by a human reviewer."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .runledger import atomic_json

LABELS = ("really wrong", "correct but scorer too strict", "lucky pass", "correct and scorer agrees")


def freeze_sample(metrics_path: Path, out: Path, *, model_identity: dict):
    from .promptquality import fingerprint
    from .runinputs import read_jsonl

    if out.exists():
        raise ValueError("manual-audit selection already exists; never resample after reading")
    rows = read_jsonl(metrics_path)
    ids = [r["trajectory_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trajectory IDs in manual-audit input")
    selected, coverage = [], {}
    for success, requested in ((False, 20), (True, 10)):
        outcome = "success" if success else "failure"
        available = sorted(r["trajectory_id"] for r in rows if bool(r["strict"]) == success)
        random.Random(42).shuffle(available)
        chosen = available[:requested]
        coverage[outcome] = {"available": len(available), "requested": requested,
                             "selected": len(chosen), "shortfall": max(0, requested - len(chosen))}
        selected.extend({"trajectory_id": identifier, "scored_outcome": outcome, "label": None,
                         "explanation": None} for identifier in chosen)
    result = {"model": model_identity, "metrics_sha256": fingerprint(metrics_path), "seed": 42,
              "coverage": coverage, "selected": selected, "allowed_labels": LABELS,
              "scores_changed": False, "audit_status": "awaiting_manual_reading"}
    atomic_json(out, result)
    return result


def label_counts(path: Path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    counts = {outcome: {label: 0 for label in LABELS} for outcome in ("failure", "success")}
    for row in audit["selected"]:
        if row.get("label") not in LABELS or not row.get("explanation"):
            raise ValueError("manual audit has unread/unexplained rows")
        counts[row["scored_outcome"]][row["label"]] += 1
    return {"counts": counts, "coverage": audit["coverage"], "scores_changed": False}
