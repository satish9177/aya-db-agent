"""New SQL model runner: offline plan by default; registered CUDA execution only."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import _run_trajectories

from .engine import HFEngine, official_tokenizer
from .promptquality import fingerprint
from .protocol import classify_turn
from .runconfig import load_registration, registration_plan, require_execution
from .runinputs import evaluation_tasks
from .runledger import atomic_json
from .sqlenv import FALLBACK, RESULT_FORMAT, SYSTEM_PROMPT, SQLTools, SubmissionEngine
from .sqlfeedback import TOOL_SETS, feedback_counts
from .sqlguard import POLICY, auditkit_check, require_reference_reads, score_episode
from .sqltraining import load_model, run_training, verify_reloaded_adapter
from .gpumemory import GpuMemoryRecorder, record_oom
from .lastposition import LastPositionHFEngine


def check_last_logits(output):
    import torch
    logits = getattr(output, "logits", None)
    if logits is not None and not bool(torch.isfinite(logits[..., -1, :]).all()):
        raise RuntimeError("nonfinite inference logits")


def evaluate_with_engine(tasks, engine, out: Path, *, samples=1, temperature=0., seed=42,
                         context_tokens=8192, tool_steps=8, purpose="compatibility", tool_feedback=None):
    """An injectable engine supports free scripted tests; the CLI supplies a gated HF model.

    tool_feedback=None keeps the original SQLTools; "column-hints-v1" uses amendment 9's tool set.
    """
    if tool_feedback not in TOOL_SETS:
        raise ValueError("unknown tool feedback policy")
    tool_class = TOOL_SETS[tool_feedback]
    out.mkdir(parents=True, exist_ok=False)
    metrics, task_map = [], {}
    start = time.monotonic()
    with (out / "trajectories.jsonl").open("w", encoding="utf-8") as records, \
         (out / "metrics.jsonl").open("w", encoding="utf-8") as scores, \
         (out / "submissions.jsonl").open("w", encoding="utf-8") as submissions:
        for repeat in range(samples):
            for i, task in enumerate(tasks):
                db = Path(task["db_path"])
                if fingerprint(db) != task["db_sha256"]:
                    raise ValueError("evaluation database differs from the frozen input")
                require_reference_reads(db, task["reference_sql"])
                tools = tool_class(db, task["id"], task.get("discovery_order"))
                before = len(engine.calls)
                t0 = time.monotonic()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trajectory = _run_trajectories(
                        [{"prompt": task["prompt"], "answer": ""}], tools, True, SubmissionEngine(engine, tools),
                        tool_steps, None, None, {"temperature": temperature, "top_p": .95,
                        "seed": seed + 1000 * repeat + i}, system_prompt=SYSTEM_PROMPT,
                        context_length=context_tokens, tools_fallback_prompt=FALLBACK,
                        tool_result_format=RESULT_FORMAT)[0]
                identifier = task["id"] + f"#s{repeat}"
                trajectory.trajectory_id = identifier
                record = trajectory.to_dict()
                kinds = Counter(classify_turn(m.get("content", "")) for m in record.get("metadata", {}).get("conversation", [])
                                if m["role"] == "assistant" and m.get("content"))
                usage = engine.calls[before:]
                result = score_episode(db, task["reference_sql"], tools.log)
                result.update(task_id=task["id"], trajectory_id=identifier, sample=repeat,
                              wall_seconds=time.monotonic() - t0, tool_calls=len(tools.log),
                              sql_errors=sum(c["error"] for c in tools.log),
                              generations=len(usage), prompt_tokens=sum(u.get("prompt_tokens") or 0 for u in usage),
                              generated_tokens=sum(u.get("completion_tokens") or 0 for u in usage),
                              assistant_turn_kinds=dict(kinds), malformed_turns=sum(n for k, n in kinds.items() if k != "pure_call"),
                              fabricated_output_turns=kinds["fake_output"],
                              repeated_tool_calls=len(tools.log) - len({json.dumps([c["name"], c["args"]], sort_keys=True) for c in tools.log}))
                if tool_feedback:
                    result.update(tool_feedback=tool_feedback, **feedback_counts(tools.log))
                record.setdefault("metadata", {}).update(task_id=identifier, source_task_id=task["id"],
                    format="sql-submit-v1", purpose=purpose, call_log=tools.log, usage=usage, metrics=result,
                    **({"tool_feedback": tool_feedback} if tool_feedback else {}))
                records.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
                scores.write(json.dumps(result) + "\n")
                for call in tools.log:
                    if call["name"] == "submit_sql":
                        read_check = next(d for d in result["submission_read_diagnostics"]
                                          if d["call_index"] == call["call_index"])
                        submissions.write(json.dumps({"trajectory_id": identifier, **call, "table_read": read_check}) + "\n")
                records.flush(); scores.flush(); submissions.flush()
                metrics.append(result)
                task_map[identifier] = task
                print(json.dumps({"episode": len(metrics), "task_id": task["id"], "strict": result["strict"],
                                  "fallback": result["fallback"], "seconds": round(result["wall_seconds"], 2)}), flush=True)
    audit = auditkit_check(out / "trajectories.jsonl", task_map)
    if len(audit["verdicts"]) != len(metrics) or any((audit["verdicts"][m["trajectory_id"]] == "success") != m["strict"] for m in metrics):
        raise RuntimeError("AuditKIT disagrees with recorded full-result scores")
    summary = {"episodes": len(metrics), "strict_successes": sum(m["strict"] for m in metrics),
               "fallback_successes": sum(m["fallback"] for m in metrics),
               "set_successes": sum(m["bird_set"] for m in metrics), "auditkit_agreement": len(metrics),
               "generated_tokens": sum(m["generated_tokens"] for m in metrics),
               "prompt_tokens": sum(m["prompt_tokens"] for m in metrics),
               "seconds": time.monotonic() - start, "purpose": purpose, "format": "sql-submit-v1",
               "scorer_policy": POLICY,
               **{k: sum(m[k] for m in metrics) for k in ("submission_count", "no_table_submissions",
                   "valid_no_table_submissions", "error_no_table_submissions", "accepted_submission_no_table")}}
    if tool_feedback:
        summary.update(tool_feedback=tool_feedback, column_hints=sum(m["column_hints"] for m in metrics),
                       refused_repeats=sum(m["refused_repeats"] for m in metrics))
    atomic_json(out / "auditkit.json", audit)
    atomic_json(out / "summary.json", summary)
    return summary


def run_evaluation(config, step):
    memory = GpuMemoryRecorder(Path(step["out"]).with_name(Path(step["out"]).name + ".gpu-memory.jsonl"),
                               phase=step.get("name", "evaluation"))
    with record_oom(memory):
        return _run_evaluation(config, step, memory)


def _run_evaluation(config, step, memory):
    selection = Path(step["selection_file"]) if step.get("selection_file") else None
    selection_sha = fingerprint(selection) if selection else None
    if selection:
        frozen = json.loads(selection.read_text(encoding="utf-8"))
        if frozen.get("frozen") is not True or not frozen.get("models"):
            raise ValueError("final evaluation requires a frozen model selection record")
        if step.get("selection_sha256") != selection_sha:
            raise ValueError("selection record changed after registration")
        adapter_sha = fingerprint(Path(step["adapter"]) / "adapter_model.safetensors") if step.get("adapter") else None
        if not any(m.get("model") == config["model"] and m.get("model_revision") == config["model_revision"]
                   and m.get("adapter_sha256") == adapter_sha for m in frozen["models"]):
            raise ValueError("evaluated model/checkpoint is not in the frozen final selection")
    tasks = evaluation_tasks(Path(step["tasks"]), purpose=step["purpose"], selection_sha256=selection_sha)
    for task in tasks:
        require_reference_reads(Path(task["db_path"]), task["reference_sql"])
    tokenizer = official_tokenizer(config["tokenizer"])
    memory.reset()
    model = load_model(config, adapter=step.get("adapter"))
    memory.record("after_model_load")
    model.eval()
    checked_forwards = [0]
    last_position = step.get("inference_logits_to_keep") == 1

    def numerical_check(module, inputs, output):
        if last_position and (output.logits.ndim != 3 or output.logits.shape[-2] != 1):
            raise RuntimeError("inference returned more than the last position")
        check_last_logits(output)
        checked_forwards[0] += 1

    forward_model = model.get_base_model() if hasattr(model, "peft_config") else model
    hook = forward_model.register_forward_hook(numerical_check)
    def check_projection(module, inputs):
        if inputs[0].ndim != 3 or inputs[0].shape[-2] != 1:
            raise RuntimeError("refusing inference vocabulary projection over the full prompt")
    head_hook = forward_model.lm_head.register_forward_pre_hook(check_projection) if last_position else None
    reload_check = None
    if step.get("verify_checkpoint_reload"):
        checkpoint = Path(step["adapter"])
        marker = json.loads((checkpoint / "_COMPLETE").read_text(encoding="utf-8"))
        if marker["global_step"] != step["expected_checkpoint_step"]:
            raise ValueError("wrong checkpoint step for inference reload")
        reload_check = verify_reloaded_adapter(model, checkpoint)
    engine_class = LastPositionHFEngine if last_position else HFEngine
    engine = engine_class(model, tokenizer, ["}]", "Tool result"], config["max_new_tokens"], RESULT_FORMAT)
    try:
        result = evaluate_with_engine(tasks, engine, Path(step["out"]), samples=step["samples"],
                                      temperature=step["temperature"], seed=step["seed"],
                                      context_tokens=config["context_tokens"], tool_steps=config["tool_steps"], purpose=step["purpose"],
                                      tool_feedback=step.get("tool_feedback"))
    finally:
        hook.remove()
        if head_hook is not None:
            head_hook.remove()
    result["finite_logit_forwards"] = checked_forwards[0]
    result["inference_logits_to_keep"] = 1 if last_position else "native_default"
    result["gpu_memory_file"] = memory.path.as_posix()
    memory.record("inference_complete", checked_forwards=checked_forwards[0])
    if not checked_forwards[0]:
        raise RuntimeError("no inference forward pass was numerically checked")
    atomic_json(Path(step["out"]) / "summary.json", result)
    if reload_check:
        atomic_json(Path(step["out"]) / "checkpoint-reload.json", reload_check)
    if result["episodes"] and result["generated_tokens"] <= 0:
        raise RuntimeError("no generated model tokens")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "execute"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--step")
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps(registration_plan(args.registration), indent=2))
        return
    require_execution(args.registration)
    reg = load_registration(args.registration)
    step = next((s for s in reg["steps"] if s["name"] == args.step), None)
    if step is None:
        raise ValueError("step is not in the approved registration")
    if step["action"] == "check_loss":
        from .losscheck import run_gpu_loss_check
        result = run_gpu_loss_check(reg["model_config"], step)
    else:
        result = (run_training if step["action"] == "train" else run_evaluation)(reg["model_config"], step)
    print(json.dumps({"step": step["name"], "completed": True}), flush=True)


if __name__ == "__main__":
    main()
