"""Per-pass continuation gates and global supervised-token matching; no model calls."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict


def targets(example: dict) -> int:
    return sum(label != -100 for label in example["labels"])


def identity(example: dict) -> tuple:
    """Identical trajectory/context copies are not new data, regardless of copy ID."""
    return example["question_id"], tuple(example["input_ids"]), tuple(example["labels"])


def unique_examples(examples: list[dict]) -> list[dict]:
    seen, out = set(), []
    for ex in examples:
        key = identity(ex)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def continuation_forecast(dev_success: float, questions: int, mean_target_tokens: float,
                          a_unique_targets_per_pass: int, *, samples: int = 4, cap: int = 2,
                          min_share: float = .1) -> dict:
    if not 0 <= dev_success <= 1 or min(questions, a_unique_targets_per_pass, mean_target_tokens) <= 0:
        raise ValueError("invalid forecast inputs")
    # Scenario range if each sampled attempt had marginal success p. No independence
    # assumption: duplicates/correlation can greatly lower unique retained yield.
    lo = questions * dev_success
    hi = questions * min(samples * dev_success, cap)
    return {"verified_episode_scenario": [lo, hi],
            "verified_unique_target_scenario_per_pass": [lo * mean_target_tokens, hi * mean_target_tokens],
            "a_unique_targets_per_pass": a_unique_targets_per_pass,
            "minimum_unique_verified_targets_per_pass": min_share * a_unique_targets_per_pass,
            "assumption": "dev greedy success is a proxy for sampled success; duplicates/correlation lower unique yield; not a confidence interval"}


def size_gate(verified_examples: list[dict], a_unique_targets_per_pass: int | list[dict],
              threshold: float = .1) -> dict:
    """Apply after verifier/dedup/cap, before epochs, repetition or control matching.

    Prefer A's examples so deduplication is enforced on both sides. An integer must be
    the already deduplicated per-pass count, never an optimizer/epoch total.
    """
    denominator = (sum(targets(ex) for ex in unique_examples(a_unique_targets_per_pass))
                   if isinstance(a_unique_targets_per_pass, list) else a_unique_targets_per_pass)
    if denominator <= 0 or not 0 < threshold <= 1:
        raise ValueError("A's unique supervised-token count per pass must be positive")
    unique = unique_examples(verified_examples)
    total = sum(targets(ex) for ex in unique)
    return {"passed": total >= threshold * denominator,
            "verified_unique_targets_per_pass": total, "a_unique_targets_per_pass": denominator,
            "verified_unique_examples": len(unique), "verified_duplicate_copies_removed": len(verified_examples) - len(unique),
            "threshold": threshold, "share": total / denominator,
            "basis": "unique trajectories per pass on both sides; no epochs/repetition/validation/padding"}


def match_control(control: list[dict], verified: list[dict]) -> tuple[list[dict], dict]:
    """Match B globally with whole examples and at most one partial example overall.

    Every retained question is represented. Distinct alternatives precede repeats.
    Only existing supervised labels can be masked; inputs and B are never mutated.
    """
    by_question, budgets = defaultdict(list), Counter()
    for ex in unique_examples(control):
        if targets(ex) > 0:
            by_question[ex["question_id"]].append(ex)
    for ex in verified:
        budgets[ex["question_id"]] += targets(ex)
    if set(by_question) != set(budgets) or any(n <= 0 for n in budgets.values()):
        raise ValueError("control and verified question IDs must match exactly, with nonempty targets")
    if not budgets:
        raise ValueError("empty continuation set")
    for options in by_question.values():
        options.sort(key=targets)
    budget = sum(budgets.values())
    out = [copy.deepcopy(by_question[q][0]) for q in sorted(by_question)]
    adjustments = []

    def trim(ex, keep):
        supervised = [i for i, label in enumerate(ex["labels"]) if label != -100]
        if not 0 < keep < len(supervised):
            raise ValueError("invalid partial target budget")
        for i in supervised[keep:]:
            ex["labels"][i] = -100
        adjustments.append({"question_id": ex["question_id"], "example_id": ex.get("id"),
                            "masked_targets": len(supervised) - keep, "kept_targets": keep})

    remaining = budget - sum(targets(ex) for ex in out)
    if remaining < 0:
        largest = max(out, key=targets)
        if targets(largest) + remaining < 1:
            raise ValueError("global token match infeasible: question coverage needs more than one partial mask")
        trim(largest, targets(largest) + remaining)
        remaining = 0
    options = [ex for q in sorted(by_question) for ex in by_question[q]]
    used = Counter(identity(ex) for ex in out)
    repeats = 0
    for ex in options:
        if not used[identity(ex)] and targets(ex) <= remaining:
            out.append(copy.deepcopy(ex))
            used[identity(ex)] += 1
            remaining -= targets(ex)
    while remaining:
        fits = [ex for ex in options if targets(ex) <= remaining]
        if fits:
            ex = min(fits, key=lambda e: (used[identity(e)], -targets(e)))
            out.append(copy.deepcopy(ex))
            repeats += int(used[identity(ex)] > 0)
            used[identity(ex)] += 1
            remaining -= targets(ex)
        else:
            ex = copy.deepcopy(min(options, key=lambda e: (used[identity(e)], targets(e))))
            repeats += int(used[identity(ex)] > 0)
            trim(ex, remaining)
            out.append(ex)
            remaining = 0
    count = sum(targets(ex) for ex in out)
    assert count == budget and len(adjustments) <= 1
    return out, {"matched_targets": count, "questions": len(budgets), "repeated_copies": repeats,
                 "mask_adjustments": adjustments, "match_scope": "whole_set",
                 "control_targets_by_question": {q: sum(targets(e) for e in out if e["question_id"] == q) for q in budgets},
                 "verified_targets_by_question": dict(budgets),
                 "control_input_tokens": sum(len(ex["input_ids"]) for ex in out),
                 "verified_input_tokens": sum(len(ex["input_ids"]) for ex in verified)}
