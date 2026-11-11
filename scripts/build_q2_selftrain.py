"""Build B (A1500's own verified Q2 attempts) and the matched A+ control after sampling (amendment 11).

Free and local; never samples, trains or uploads. Inputs are the three pulled Q2 sampling outputs.
For each sampled attempt:
  1. `selftrainfilter.candidate_eligibility`: clean protocol and guarded strict success.
  2. Exact duplicate call sequences per question are kept once, at most two per question.
  3. Rebuilt with `make_episode` on the same database. Calls whose result was an error are untrained context.
     The rebuilt observations must equal the sampled ones (the job ran SQLite 3.40.1, this machine 3.53.1;
     any difference drops the attempt). It must fit 4,096 tokens and pass the AgentTune replay audit.
Control A+: gold-path episodes (list tables, describe the gold tables, the reference query, submit) for the
same questions, using each question's stored phrasings except the deliberately vague `short_ambiguous`.
`continuation.match_control` matches trained target tokens. Gates: G2 (>=25% of the 577 questions have an
eligible attempt) and size (unique B targets per pass >= 10% of A1500's 330,742).
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ayadb.continuation import match_control
from ayadb.engine import official_tokenizer
from ayadb.promptquality import fingerprint
from ayadb.queryshape import classify
from ayadb.runinputs import read_jsonl
from ayadb.runledger import atomic_json
from ayadb.selftrainfilter import candidate_eligibility
from ayadb.sqlsftdata import encode_episode, make_episode, replay_sql_audit
from ayadb.tokenbudget import size_gate

ACCOUNT = "hf-user"
PARTS = [(f"q2sample-a1500-sql-20261109-p{k}", f"q2-sample-p{k}", Path(f"data/sqale/q2-sampling-v2/part-{k}.tasks.jsonl"))
         for k in (1, 2, 3)]
OUT = Path("data/sqale/selftrain-v1")
A1500_UNIQUE_TARGETS = 330742
A1500_ADAPTER_SHA = "69f0f125c595895bab3b522d450131324fdd5c374e234e62a34b266acf0b62b8"
MAX_PER_QUESTION, MAX_TOKENS, G2_SHARE = 2, 4096, .25
SKIP_PHRASINGS = {"short_ambiguous"}
SETTINGS = {"seed": 42, "epochs": 2, "batch_size": 1, "gradient_accumulation": 16, "learning_rate": 1e-4,
            "loss_policy": "trained-positions-v1"}


def call_key(log):
    return tuple((c["name"], json.dumps(c["args"], sort_keys=True)) for c in log)


def rebuild_attempt(task, record, index, tokenizer):
    """Training episode from one eligible sampled attempt, or (None, reason)."""
    log = record["metadata"]["call_log"]
    plan = [{"name": c["name"], "args": c["args"], "train": not c["error"]} for c in log]
    try:
        ep = make_episode(task, calls=plan)
    except ValueError as error:
        return None, "rebuild_failed: " + str(error)[:80]
    if [(c["name"], c["args"], c["result"]) for c in ep["calls"]] != [(c["name"], c["args"], c["result"]) for c in log]:
        return None, "observation_differs_locally"
    encoded = encode_episode(ep, tokenizer, MAX_TOKENS)
    if encoded is None:
        return None, "over_token_limit"
    ep.update(id=f"{task['id']}-self-{index}", source="q2-self", sampled_trajectory_id=record["trajectory_id"],
              contains_recovery=any(c["error"] for c in log), n_input_tokens=len(encoded["input_ids"]),
              n_target_tokens=sum(v != -100 for v in encoded["labels"]),
              input_ids=encoded["input_ids"], labels=encoded["labels"])
    problems = replay_sql_audit(ep, Path(task["db_path"]), task["db_sha256"], tokenizer)
    ep.pop("_replay_record", None)
    if problems:
        return None, "replay_audit: " + "; ".join(problems)[:80]
    return ep, None


def control_examples(task, tokenizer):
    out = []
    phrasings = task.get("phrasings") or {task.get("phrasing", "verbose"): task["prompt"]}
    for style, prompt in sorted(phrasings.items()):
        if style in SKIP_PHRASINGS or not isinstance(prompt, str) or not prompt.strip():
            continue
        ep = make_episode({**task, "prompt": prompt, "phrasing": style})
        encoded = encode_episode(ep, tokenizer, MAX_TOKENS)
        if encoded is None:
            continue
        ep.update(id=f"{task['id']}-gold-{style}", source="q2-gold-control", n_input_tokens=len(encoded["input_ids"]),
                  n_target_tokens=sum(v != -100 for v in encoded["labels"]),
                  input_ids=encoded["input_ids"], labels=encoded["labels"])
        out.append(ep)
    return out


def build(parts=PARTS, root=Path("remote/hf-sql") / ACCOUNT, out=OUT, tokenizer=None):
    tokenizer = tokenizer or official_tokenizer("data/sqale/tokenizer")
    tasks = {t["id"]: t for _, _, path in parts for t in read_jsonl(path)}
    eligibility, kept, dropped = [], defaultdict(list), Counter()
    seen = defaultdict(set)
    for run_id, step, _ in parts:
        for record in read_jsonl(root / run_id / step / "trajectories.jsonl"):
            tid = record["metadata"]["source_task_id"]
            verdict = candidate_eligibility(tasks[tid], record)
            eligibility.append({"trajectory_id": record["trajectory_id"], "task_id": tid, **verdict})
            if not verdict["eligible"]:
                continue
            key = call_key(record["metadata"]["call_log"])
            if key in seen[tid] or len(kept[tid]) >= MAX_PER_QUESTION:
                dropped["duplicate_or_over_cap"] += 1
                continue
            seen[tid].add(key)
            ep, reason = rebuild_attempt(tasks[tid], record, len(kept[tid]) + 1, tokenizer)
            if ep is None:
                dropped[reason.split(":")[0]] += 1
                continue
            kept[tid].append(ep)
    eligible_questions = {e["task_id"] for e in eligibility if e["eligible"]}
    g2 = {"questions": len(tasks), "with_eligible_attempt": len(eligible_questions),
          "share": len(eligible_questions) / len(tasks), "threshold": G2_SHARE,
          "passed": len(eligible_questions) >= G2_SHARE * len(tasks)}
    kept = {tid: eps for tid, eps in kept.items() if eps}  # drop questions whose eligible attempts were all dropped
    verified = [ep for tid in sorted(kept) for ep in kept[tid]]
    report = {"g2": g2, "eligibility_reasons": dict(Counter(r for e in eligibility for r in e["reasons"])),
              "attempts": len(eligibility), "eligible_attempts": sum(e["eligible"] for e in eligibility),
              "dropped_after_eligibility": dict(dropped), "verified_episodes": len(verified),
              "verified_with_recovery": sum(ep["contains_recovery"] for ep in verified),
              "eligible_questions_by_shape": dict(Counter(classify(tasks[t]["reference_sql"]) for t in eligible_questions)),
              "model_calls": 0}
    if not g2["passed"] or not verified:
        report["continuations_may_train"] = False
        return None, None, report, eligibility
    control = [ex for tid in sorted(kept) for ex in control_examples(tasks[tid], tokenizer)]
    aplus, b, matching = match_control(control, verified)
    gate = size_gate(b, A1500_UNIQUE_TARGETS)
    matching.update(post_fallback_size_gate=gate, continuations_may_train=bool(gate["passed"] and g2["passed"]),
                    a_adapter_sha256=A1500_ADAPTER_SHA, g2=g2, skipped_control_phrasings=sorted(SKIP_PHRASINGS))
    report.update(continuations_may_train=matching["continuations_may_train"], size_gate=gate,
                  matched_targets=matching["matched_targets"], retained_questions=matching["retained_questions"])
    return b, aplus, {**report, "matching": matching}, eligibility


def write(b, aplus, report, eligibility, model_config, out=OUT):
    out.mkdir(parents=True, exist_ok=False)
    with (out / "eligibility.jsonl").open("w", encoding="utf-8", newline="\n") as target:
        for row in eligibility:
            target.write(json.dumps(row) + "\n")
    if b is not None:
        for name, rows in (("b.episodes.jsonl", b), ("aplus.episodes.jsonl", aplus)):
            with (out / name).open("w", encoding="utf-8", newline="\n") as target:
                for row in rows:
                    target.write(json.dumps(row, ensure_ascii=False) + "\n")
        matching = report["matching"]
        matching["datasets"] = {"continuation_self": {"path": (out / "b.episodes.jsonl").as_posix(),
                                                      "sha256": fingerprint(out / "b.episodes.jsonl")},
                                "continuation_control": {"path": (out / "aplus.episodes.jsonl").as_posix(),
                                                         "sha256": fingerprint(out / "aplus.episodes.jsonl")}}
        matching["training_settings"] = {**{k: SETTINGS[k] for k in ("seed", "epochs", "batch_size", "gradient_accumulation",
                                                                     "learning_rate", "loss_policy")},
                                         **{k: model_config[k] for k in ("dtype", "quantization", "rank", "alpha", "dropout")}}
        atomic_json(out / "matching.json", matching)
    atomic_json(out / "summary.json", {k: v for k, v in report.items() if k != "matching"})


def main():
    if (OUT / "summary.json").exists():
        raise ValueError("self-training data already built; never rebuild over it")
    model_config = json.loads(Path("experiments/sql-a1500/REGISTRATION.json").read_text(encoding="utf-8"))["model_config"]
    b, aplus, report, eligibility = build()
    write(b, aplus, report, eligibility, model_config)
    print(json.dumps({k: v for k, v in report.items() if k != "matching"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
