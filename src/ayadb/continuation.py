"""Preregistered global matching with minimum question loss, alongside the v3 utility."""
from __future__ import annotations

from collections import Counter

from .tokenbudget import match_control as whole_match
from .tokenbudget import size_gate, targets, unique_examples


def match_control(control: list[dict], verified: list[dict], *, a_training_examples=None):
    """Return (matched A+, retained B, disclosure), preserving both arms' question set."""
    original_q = {e["question_id"] for e in verified}
    original_unique = unique_examples(verified)
    budgets = Counter()
    minimum = {}
    for ex in verified:
        budgets[ex["question_id"]] += targets(ex)
    for ex in control:
        n, q = targets(ex), ex["question_id"]
        if n > 0:
            minimum[q] = min(minimum.get(q, n), n)
    if not original_q or set(minimum) != original_q or any(b <= 0 for b in budgets.values()):
        raise ValueError("control and verified need exactly the same nonempty question set")
    removed, anchor = [], None
    try:
        matched, report = whole_match(control, verified)
    except ValueError as error:
        if "global token match infeasible" not in str(error):
            raise
        deficit = {q: minimum[q] - budgets[q] for q in sorted(original_q)}
        total = sum(deficit.values())
        choices = []
        ordered = sorted((q for q in deficit if deficit[q] > 0), key=lambda q: (-deficit[q], q))
        for p in sorted(original_q):
            need, amount, dropped = total - (minimum[p] - 1), 0, []
            if need > 0:
                for q in ordered:
                    if q == p:
                        continue
                    dropped.append(q)
                    amount += deficit[q]
                    if amount >= need:
                        break
            if need <= amount:
                choices.append((len(dropped), tuple(sorted(dropped)), p))
        if not choices:
            raise ValueError("no nonempty matched continuation set") from None
        _, selected, anchor = min(choices)
        removed = list(selected)
        retained_q = original_q - set(removed)
        verified = [e for e in verified if e["question_id"] in retained_q]
        control = [e for e in control if e["question_id"] in retained_q]
        matched, report = whole_match(control, verified)
    retained_q = {e["question_id"] for e in verified}
    unique_b = unique_examples(verified)
    before_targets = sum(budgets.values())
    after_targets = sum(targets(e) for e in verified)
    report.update(fallback="minimum-question-drop-v1", original_questions=len(original_q),
                  retained_questions=len(retained_q), question_coverage=len(retained_q) / len(original_q),
                  dropped_question_ids=removed, retained_question_ids=sorted(retained_q),
                  fallback_anchor=anchor, verified_targets_before=before_targets,
                  verified_targets_after=after_targets, verified_target_coverage=after_targets / before_targets,
                  verified_unique_examples_after=len(unique_b),
                  verified_unique_examples_before=len(original_unique),
                  verified_unique_episode_coverage=len(unique_b) / len(original_unique),
                  verified_unique_targets_before=sum(targets(e) for e in original_unique),
                  verified_unique_targets_after=sum(targets(e) for e in unique_b),
                  partial_masked_targets=sum(a["masked_targets"] for a in report["mask_adjustments"]))
    if a_training_examples is not None:
        report["post_fallback_size_gate"] = size_gate(verified, a_training_examples)
        report["continuations_may_train"] = report["post_fallback_size_gate"]["passed"]
    return matched, verified, report
