"""Feasibility smoke test: untuned model on verifiable DB tasks through AgentTune's rollout loop.

    python -m ayadb.smoke --split dev --repeats 2 --temperature 0

Writes runs/<run_id>/: config.json, trajectories.jsonl (AgentTune records, readable by
AuditKIT), metrics.jsonl, auditkit.json, summary.json, report.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import (
    DEFAULT_TOOL_RESULT_FORMAT,
    DEFAULT_TOOLS_FALLBACK_PROMPT,
    _run_trajectories,
)

from .engine import HFEngine, OllamaRawEngine, load_hf, load_tokenizer, ollama_blob
from .env import make_tools
from .metrics import logged, summarize, trajectory_metrics
from .protocol import audit_record, summarize_audit
from .tasks import (EVAL_SPLITS, PINNED_LF, SPLITS, check_answer, extract_answer, file_sha256, generate_eval,
                    lf_sha256, load_tasks)

SYSTEM_PROMPT = (
    "You are a data analyst for Kestrel Supplies. You answer questions by querying the "
    "company's SQLite database with the tools below. You cannot see the database: first find "
    "the tables and columns with the tools, then run SQL, and base every answer on query "
    "results. When you know the final answer, reply with exactly one line and nothing else:\n"
    "ANSWER: <value>\n"
    "where <value> is a plain number (no currency symbols or units) or the exact name asked for."
)

# Two documented tool-call formats. `default` is AgentTune's fallback as shipped: it tells the
# model to answer after a tool result, which suits single-call tasks. `multistep` keeps the same
# call syntax but allows chained calls, and stops generation where a model would start writing
# its own (fabricated) tool result, as ReAct harnesses stop at "Observation:".
MULTISTEP_FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call, and only the parameters listed for that tool. "
    'After each call you get a user turn that starts with "Tool result": it is the output of '
    "your call, not a new message from the user. Then either call another tool or, once the "
    "results answer the question, give the final answer. Most questions need several calls: "
    "check the table columns before writing SQL, and fix the query if it returns an error."
)
FORMATS = {
    "default": {"fallback": DEFAULT_TOOLS_FALLBACK_PROMPT, "result": DEFAULT_TOOL_RESULT_FORMAT,
                "stop": []},
    "multistep": {"fallback": MULTISTEP_FALLBACK_PROMPT, "result": "Tool result for {name}: {content}",
                  "stop": ["Tool result"]},
    # One call per turn: generation ends as soon as a call list closes ("}]", re-appended by the
    # engine), so the model cannot script the rest of the episode or invent the tool's output.
    "multistep-cut": {"fallback": MULTISTEP_FALLBACK_PROMPT,
                      "result": "Tool result for {name}: {content}", "stop": ["}]", "\nTool result"]},
    # Same, but a turn that opens with a fake "Tool result: [call]" still gets its call executed.
    "multistep-close": {"fallback": MULTISTEP_FALLBACK_PROMPT,
                        "result": "Tool result for {name}: {content}", "stop": ["}]"]},
}


def _git_rev(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip() + (
            "-dirty" if subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                       text=True).stdout.strip() else "")
    except Exception:
        return None


def auditkit_check(traj_path: Path, tasks_by_tid: dict) -> dict:
    """Score the recorded trajectories with AuditKIT and compare to our verifier."""
    from auditkit.agent_eval import importers
    from auditkit.agent_eval.outcome import CustomPredicate
    from auditkit.agent_eval.runner import AgentEvalRunner, AgentEvalSpec
    from auditkit.agent_eval.types import AgentCase

    def verified(episode, case):
        t = case.metadata["task"]
        return check_answer(extract_answer(episode.final_answer), t["answer"], t["kind"])

    oracle = CustomPredicate(verified, version="answer-v1", name="ayadb.check_answer")
    episodes = importers.episodes_from_agenttune(str(traj_path))
    cases = [AgentCase(id=tid, task=t["prompt"], category=t["family"], outcome=oracle,
                       metadata={"task": t}) for tid, t in tasks_by_tid.items()]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = AgentEvalRunner().run(AgentEvalSpec(
            cases=cases, mode="recorded", episodes=episodes, name="ayadb-smoke",
            scorers=["tool_call_validity", "redundant_tool_calls", "agent_loop_detection"]))
    verdicts = {r.case_id: r.outcome.get("verdict") for r in result.rows}
    return {"n_episodes": len(episodes), "n_rows": len(result.rows), "verdicts": verdicts,
            "summary": result.summary(), "warnings": sorted({str(w.message)[:200] for w in caught}),
            "rows": [r.to_dict() for r in result.rows]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=[*SPLITS, *EVAL_SPLITS])
    ap.add_argument("--unseal", action="store_true", help="required to run the sealed test split")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="hf.co/CohereLabs/tiny-aya-global-GGUF:Q8_0")
    ap.add_argument("--tokenizer", default=None, help="HF repo id; default: the model's GGUF file")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-steps", type=int, default=8, help="tool rounds per task")
    ap.add_argument("--format", default="default", choices=list(FORMATS))
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--engine", default="ollama", choices=["ollama", "hf"])
    ap.add_argument("--hf-model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", default=None, help="hf engine: LoRA adapter directory")
    ap.add_argument("--load-4bit", action="store_true", help="hf engine: nf4 base, as in QLoRA")
    ap.add_argument("--dtype", default="float16", help="hf engine: weights/compute dtype")
    args = ap.parse_args(argv)

    if args.split == "test" and not args.unseal:
        sys.exit("refusing to run the sealed test split without --unseal")
    root = Path(__file__).resolve().parents[2]
    data = root / args.data
    tasks_path, db_of = generate_eval(args.split, data)
    if args.split in PINNED_LF and lf_sha256(tasks_path) != PINNED_LF[args.split]:
        sys.exit(f"{args.split} tasks regenerated to {lf_sha256(tasks_path)} (LF), not the pinned set")
    tasks = load_tasks(tasks_path)
    if args.limit:
        tasks = tasks[:args.limit]

    fmt = FORMATS[args.format]
    if args.engine == "ollama":
        tok_src = args.tokenizer or str(ollama_blob(args.model))
        tokenizer = load_tokenizer(tok_src)
        opts = {"num_ctx": args.context}
        if fmt["stop"]:
            opts["stop"] = fmt["stop"]
        engine = OllamaRawEngine(args.model, tokenizer, options=opts, tool_result_format=fmt["result"])
        model_info = {"engine": "ollama", "model": args.model,
                      "model_blob": Path(ollama_blob(args.model)).name,
                      "tokenizer": args.tokenizer or "gguf:" + Path(tok_src).name,
                      "num_predict": engine.options["num_predict"]}
    else:
        model, tokenizer = load_hf(args.hf_model, args.adapter, args.load_4bit, args.dtype)
        engine = HFEngine(model, tokenizer, fmt["stop"], 512, fmt["result"])
        model_info = {"engine": "hf", "model": args.hf_model, "tokenizer": args.hf_model,
                      "adapter": args.adapter, "load_4bit": args.load_4bit, "dtype": args.dtype,
                      "adapter_sha256": file_sha256(Path(args.adapter) / "adapter_model.safetensors")
                      if args.adapter else None, "num_predict": engine.max_new_tokens}
    call_log: list[dict] = []
    dbs = sorted(set(db_of.values()))
    tools_of = {db: logged(make_tools(db), call_log) for db in dbs}

    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    out = root / args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": run_id, "split": args.split, "n_tasks": len(tasks), "repeats": args.repeats,
        **model_info,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
                     "num_predict": model_info["num_predict"], "num_ctx": args.context},
        "max_steps": args.max_steps, "tasks_sha256": file_sha256(tasks_path),
        "db_sha256": file_sha256(dbs[0]) if len(dbs) == 1 else {db.name: file_sha256(db) for db in dbs},
        "code": _git_rev(root),
        "versions": {p: version(p) for p in ("agenttune", "auditkit", "transformers", "torch")},
        "format": args.format, "system_prompt": SYSTEM_PROMPT,
        "tools_fallback_prompt": fmt["fallback"], "tool_result_format": fmt["result"],
        "stop": fmt["stop"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    traj_f = open(out / "trajectories.jsonl", "w", encoding="utf-8")
    met_f = open(out / "metrics.jsonl", "w", encoding="utf-8")
    rows, tasks_by_tid, caught_warnings, audits = [], {}, set(), []
    t_start = time.time()
    for rep in range(args.repeats):
        for i, task in enumerate(tasks):
            call_log.clear()
            n_calls_before = len(engine.calls)
            t0 = time.time()
            gen = {"temperature": args.temperature, "top_p": args.top_p,
                   "seed": args.seed + 1000 * rep + i}
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                traj = _run_trajectories(
                    [{"prompt": task.prompt, "answer": task.answer}], tools_of[db_of[task.id]], True, engine,
                    args.max_steps, None, None, gen, system_prompt=SYSTEM_PROMPT,
                    context_length=args.context, tools_fallback_prompt=fmt["fallback"],
                    tool_result_format=fmt["result"])[0]
            caught_warnings |= {str(w.message)[:160] for w in caught}
            tid = f"{task.id}#r{rep}"
            traj.trajectory_id = tid
            record = traj.to_dict()
            usage = engine.calls[n_calls_before:]
            m = trajectory_metrics(record, list(call_log), task.answer, task.kind, args.max_steps)
            m.update({"task_id": task.id, "trajectory_id": tid, "family": task.family, "repeat": rep,
                      "gold": task.answer, "wall_s": round(time.time() - t0, 2),
                      "completion_tokens": sum(u["completion_tokens"] or 0 for u in usage),
                      "prompt_tokens_max": max((u["prompt_tokens"] or 0 for u in usage), default=0),
                      "generations": len(usage)})
            record["metadata"].update({"task_id": task.id, "repeat": rep, "call_log": list(call_log),
                                       "usage": usage})
            a = audit_record(record, fmt["result"])
            audits.append(a)
            m.update({k: a[k] for k in ("hallucinated_tool_output", "call_with_text_turns",
                                        "fake_output_turns", "orch_result_mismatch",
                                        "orch_parser_miss", "orch_boundary", "orch_truncated")})
            record["metadata"]["metrics"] = m
            traj_f.write(json.dumps(record, default=str) + "\n")
            met_f.write(json.dumps(m) + "\n")
            traj_f.flush(); met_f.flush()
            rows.append(m)
            tasks_by_tid[tid] = {"prompt": task.prompt, "answer": task.answer, "kind": task.kind,
                                 "family": task.family}
            print(f"[r{rep} {i + 1:>2}/{len(tasks)}] {task.family:<22} ok={int(m['success'])} "
                  f"calls={m['tool_calls']} valid={m['valid_calls']} err={m['exec_errors']} "
                  f"ans={m['answer']!r:.30} gold={task.answer!r} {m['wall_s']}s", flush=True)
    traj_f.close(); met_f.close()

    summary = {"overall": summarize(rows), "wall_s": round(time.time() - t_start, 1),
               "by_repeat": {r: summarize([x for x in rows if x["repeat"] == r])
                             for r in range(args.repeats)},
               "by_family": {}, "warnings": sorted(caught_warnings),
               "protocol": summarize_audit(audits)}
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    summary["by_family"] = {k: {"n": len(v), "success_rate": sum(x["success"] for x in v) / len(v),
                                "mean_tool_calls": sum(x["tool_calls"] for x in v) / len(v)}
                            for k, v in fam.items()}
    if args.repeats >= 2:  # A/A: same tasks, same settings, different repeat
        by = defaultdict(dict)
        for r in rows:
            by[r["task_id"]][r["repeat"]] = r["success"]
        flips = sum(1 for v in by.values() if v.get(0) != v.get(1))
        summary["aa"] = {"tasks": len(by), "flips_r0_r1": flips,
                         "identical_outcomes": flips == 0}
    try:
        ak = auditkit_check(out / "trajectories.jsonl", tasks_by_tid)
        ours = {r["trajectory_id"]: r["correct"] for r in rows}
        agree = sum(1 for tid, v in ak["verdicts"].items() if (v == "success") == ours.get(tid))
        summary["auditkit"] = {"n_episodes": ak["n_episodes"], "n_rows": ak["n_rows"],
                               "verdict_agreement": f"{agree}/{len(ours)}", "summary": ak["summary"],
                               "warnings": ak["warnings"]}
        (out / "auditkit.json").write_text(json.dumps(ak, indent=2, default=str), encoding="utf-8")
    except Exception as e:  # the integration is under test: report, don't hide
        summary["auditkit"] = {"error": f"{type(e).__name__}: {e}"}
    summary["first_prompt"] = engine.first_prompt  # system + AgentTune tool block + task, as sent
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("overall", "aa", "auditkit") if k in summary},
                     indent=2, default=str))
    print(f"run dir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
