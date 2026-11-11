"""Atomic run accounting and durable-checkpoint markers. No model construction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def atomic_json(path: Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def object_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class TokenLedger:
    def __init__(self, out: Path, *, previous=None):
        self.out = Path(out)
        self.pending = []
        self.state = previous or {"optimizer_steps": 0, "skipped_optimizer_steps": 0, "microbatches": 0,
                                  "input_tokens": 0, "target_tokens": 0, "padded_tokens": 0,
                                  "example_visits": 0, "unique_examples_seen": [], "last_global_step": 0}

    def batch(self, identities, lengths, targets, padded):
        self.pending.append({"identities": list(identities), "input_tokens": sum(lengths),
                             "target_tokens": sum(targets), "padded_tokens": padded,
                             "example_visits": len(identities)})

    def commit(self, global_step, *, skipped=False):
        self.state["last_global_step"] = int(global_step)
        self.state["skipped_optimizer_steps" if skipped else "optimizer_steps"] += 1
        if not skipped:
            self.state["microbatches"] += len(self.pending)
            for key in ("input_tokens", "target_tokens", "padded_tokens", "example_visits"):
                self.state[key] += sum(b[key] for b in self.pending)
            seen = set(self.state["unique_examples_seen"])
            seen.update(i for b in self.pending for i in b["identities"])
            self.state["unique_examples_seen"] = sorted(seen)
        self.pending = []
        atomic_json(self.out / "token-ledger.json", self.state)

    def checkpoint(self, directory: Path, train_set_sha: str, training_config_sha=None):
        atomic_json(directory / "token-ledger.json", self.state)
        atomic_json(directory / "_COMPLETE", {"global_step": self.state["last_global_step"],
                                             "train_set_sha256": train_set_sha,
                                             "training_config_sha256": training_config_sha})
