"""SFT data: expert episodes -> audited multi-turn conversations -> assistant-only token labels.

    python -m ayadb.sftdata --out data/sft_pilot --per-family 30
    python -m ayadb.sftdata --out data/sft_recovery --recovery query   # recovery experiment, arm B

The conversation is the one AgentTune's rollout loop builds at inference (system prompt +
tools_fallback_prompt with the JSON tool schemas, tool results folded into user turns), rendered
by AgentTune's own `_render_chat_template`. Loss covers assistant turns only, including the
closing <|END_RESPONSE|><|END_OF_TURN_TOKEN|>; system, task, tool results and deliberate
mistakes are context. Every episode is replayed through AgentTune's real loop on a freshly
rebuilt database before it is kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

from agenttune.agentic.rollout_engines.base import RolloutEngine
from agenttune.agentic.rollout_engines.rollout_factory import (
    _build_tool_schema,
    _render_chat_template,
    _run_trajectories,
)

from .env import build_db, make_tools
from .expert import (RESULT_FORMAT, Episode, QueryMistake, build_episode, is_verified, mistake_schedule,
                     query_mistake_schedule)
from .metrics import logged, trajectory_metrics
from .protocol import audit_record, classify_turn
from .tasks import (SHARED, SPLITS, TRAIN_POOL_SEEDS, file_sha256, generate_split, generate_tasks,
                    load_tasks)

MAX_STEPS = 8  # the evaluation budget; every expert episode must fit it


def system_message(system_prompt: str, fallback: str, schemas: list[dict]) -> str:
    """Exactly what AgentTune puts in the system turn when the template ignores `tools`."""
    return f"{system_prompt}\n\n{fallback.format(tools=json.dumps(schemas, indent=2))}"


def conversation(prompt: str, messages: list[dict], system: str) -> list[dict]:
    return ([{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            + [{"role": m["role"], "content": m["content"]} for m in messages])


def render(tokenizer, conv: list[dict], schemas: list[dict], generation_prompt: bool) -> str:
    return _render_chat_template(tokenizer, conv, schemas, tool_result_format=RESULT_FORMAT,
                                 tokenize=False, add_generation_prompt=generation_prompt)


class ScriptedEngine(RolloutEngine):
    """Plays back fixed assistant turns and records each prompt AgentTune renders."""

    def __init__(self, tokenizer, outputs: list[str]):
        self.tokenizer, self.outputs, self.prompts = tokenizer, list(outputs), []

    def _get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, tools, gen_cfg):
        self.prompts.append(_render_chat_template(self.tokenizer, prompts, tools,
                                                  tool_result_format=RESULT_FORMAT, tokenize=False))
        out = self.outputs.pop(0) if self.outputs else ""
        return {"completions": [out], "completions_raw": [out], "logprobs": [None], "metadata": {}}


def replay_audit(ep: Episode, seed: int, db_sha: str, tokenizer, system_prompt: str,
                 fallback: str, *, sql_db: Path | None = None) -> list[str]:
    """Problems found when replaying `ep` through AgentTune on a fresh database ([] = clean)."""
    if sql_db is not None:
        # New format is opt-in; the existing shop build and replay path are unchanged.
        from .sqlsftdata import replay_sql_audit
        return replay_sql_audit(ep, sql_db, db_sha, tokenizer)
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        fresh = build_db(Path(tmp) / "fresh.sqlite", seed)
        if file_sha256(fresh) != db_sha:
            problems.append("fresh database differs from the one the episode was built on")
        log: list[dict] = []
        raw_tools = make_tools(fresh)
        tools = logged(raw_tools, log)
        schemas = [_build_tool_schema(t) for t in tools]
        outputs = [m["content"] for m in ep.messages if m["role"] == "assistant"]
        engine = ScriptedEngine(tokenizer, outputs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = _run_trajectories([{"prompt": ep.task.prompt, "answer": ep.task.answer}], tools,
                                     True, engine, MAX_STEPS, None, None, {},
                                     system_prompt=system_prompt, context_length=8192,
                                     tools_fallback_prompt=fallback,
                                     tool_result_format=RESULT_FORMAT)[0]
        record = traj.to_dict()
        record["metadata"]["call_log"] = list(log)
        if engine.outputs:
            problems.append(f"{len(engine.outputs)} scripted turns never requested (episode ended early)")
        replayed = [(c["name"], c["args"], c["result"]) for c in log]
        expected = [(c["name"], c["args"], c["result"]) for c in ep.calls]
        if replayed != expected:
            problems.append("replayed observations differ from the recorded ones")
        m = trajectory_metrics(record, log, ep.task.answer, ep.task.kind, MAX_STEPS)
        if not m["success"]:
            problems.append(f"replay not verified: answer {m['answer']!r} vs {ep.task.answer!r}")
        a = audit_record(record, RESULT_FORMAT)
        if any(a[k] for k in ("orch_result_mismatch", "orch_parser_miss", "orch_boundary")):
            problems.append(f"protocol audit: {a}")
        kinds = a["turn_kinds"]
        if kinds[:-1] != ["pure_call"] * (len(kinds) - 1) or kinds[-1] != "answer":
            problems.append(f"turn kinds {kinds}")
        sys = system_message(system_prompt, fallback, schemas)
        conv = conversation(ep.task.prompt, ep.messages, sys)
        idx = [i for i, msg in enumerate(conv) if msg["role"] == "assistant"]
        sft_prefixes = [render(tokenizer, conv[:i], schemas, True) for i in idx]
        if engine.prompts != sft_prefixes:
            problems.append("AgentTune-rendered prompts differ from the SFT prefixes")
        raw_tools.close()
    return problems


def tokenize(tokenizer, conv: list[dict], train_flags: list[bool], schemas: list[dict],
             max_len: int) -> dict | None:
    """input_ids + labels (assistant-only). None if the episode exceeds max_len (never truncated)."""
    def ids(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    full = ids(render(tokenizer, conv, schemas, False))
    if len(full) > max_len:
        return None
    labels = [-100] * len(full)
    end = [tokenizer.convert_tokens_to_ids(t) for t in ("<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>")]
    for i, msg in enumerate(conv):
        if msg["role"] != "assistant":
            continue
        pre = ids(render(tokenizer, conv[:i], schemas, True))
        upto = ids(render(tokenizer, conv[:i + 1], schemas, False))
        assert full[:len(upto)] == upto and upto[:len(pre)] == pre, "render is not prefix-stable"
        assert upto[-2:] == end, f"assistant turn does not end with {end}: {upto[-4:]}"
        if train_flags[i]:
            labels[len(pre):len(upto)] = upto[len(pre):]
    bos = tokenizer.bos_token_id
    assert full[0] == bos and full.count(bos) == 1, "expected exactly one BOS at position 0"
    return {"input_ids": full, "labels": labels}


def load_episodes(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def main(argv=None) -> int:
    from .engine import load_tokenizer, ollama_blob
    from .smoke import FORMATS, SYSTEM_PROMPT

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_pilot")
    ap.add_argument("--per-family", type=int, default=30)
    ap.add_argument("--mistake-share", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--recovery", default="discovery", choices=["discovery", "query"],
                    help="mistake kind: discovery (the pilot) or query (arm B, same episodes and positions)")
    ap.add_argument("--tokenizer", default=None, help="HF repo id; default: Ollama GGUF metadata")
    args = ap.parse_args(argv)

    fallback = FORMATS["multistep-close"]["fallback"]
    assert FORMATS["multistep-close"]["result"] == RESULT_FORMAT
    root = Path(__file__).resolve().parents[2]
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    tokenizer = load_tokenizer(args.tokenizer or str(ollama_blob("hf.co/CohereLabs/tiny-aya-global-GGUF:Q8_0")))

    # Training sources: the train split and the extra pool databases. Never dev or test.
    _, dev_path = generate_split("dev", data)
    dev = load_tasks(dev_path)
    dev_pairs = {(t.prompt, t.answer) for t in dev}
    dev_prompts = {t.prompt for t in dev}
    sources = [("train", SPLITS["train"]["seed"], *generate_split("train", data))]
    for s in TRAIN_POOL_SEEDS:
        sources.append((f"pool{s}", s, *generate_tasks(f"pool{s}", s, SHARED, 8, data)))
    assert not {s for _, s, _, _ in sources} & {SPLITS["dev"]["seed"], SPLITS["test"]["seed"]}

    candidates, dropped_pairs = defaultdict(list), 0
    for name, seed, db, tasks_path in sources:
        for t in load_tasks(tasks_path):
            if (t.prompt, t.answer) in dev_pairs:
                dropped_pairs += 1  # same question with the same answer as a dev task: excluded
                continue
            candidates[t.family].append((t, seed, db, file_sha256(db)))
    rng = random.Random(args.seed)
    chosen = []
    for fam in SHARED:
        pool = candidates[fam]
        rng.shuffle(pool)
        chosen += pool[: args.per_family]
    mistakes = mistake_schedule(len(chosen), args.seed + 1, args.mistake_share)
    if args.recovery == "query":
        mistakes = query_mistake_schedule(mistakes, [t.family for t, *_ in chosen], args.seed + 2)

    rows, rejected = [], Counter()
    for (task, seed, db, sha), mistake in zip(chosen, mistakes):
        ep = build_episode(task, db, mistake)
        if not is_verified(ep):
            rejected["unverified"] += 1
            continue
        bad = [c for c in ep.calls if not c["train"] and not str(c["result"]).startswith(("Error", "SQL error"))]
        if bad:
            rejected["mistake_without_error"] += 1
            continue
        problems = replay_audit(ep, seed, sha, tokenizer, SYSTEM_PROMPT, fallback)
        if problems:
            rejected["replay: " + problems[0][:60]] += 1
            continue
        eff = ("query:" + "+".join(mistake.kinds) if isinstance(mistake, QueryMistake) else
               "column" if any(not c["train"] and c["name"] == "run_sql" for c in ep.calls) else
               "table" if any(not c["train"] for c in ep.calls) else None)
        rows.append({"id": task.id, "family": task.family, "db_seed": seed, "db_sha256": sha,
                     "mistake": eff, "prompt": task.prompt, "gold": task.answer, "kind": task.kind,
                     "answer": ep.answer, "n_calls": len(ep.calls),
                     "messages": ep.messages})
    tools = make_tools(data / "dev.sqlite")
    schemas = [_build_tool_schema(t) for t in tools]
    lengths = []
    for r in rows:
        conv = conversation(r["prompt"], r["messages"], system_message(SYSTEM_PROMPT, fallback, schemas))
        flags = [False, False] + [m.get("train", False) for m in r["messages"]]
        tok = tokenize(tokenizer, conv, flags, schemas, max_len=4096)
        assert tok is not None, f"{r['id']} exceeds 4096 tokens"
        lengths.append((len(tok["input_ids"]), sum(1 for x in tok["labels"] if x != -100)))
    path = out / "episodes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tot = sorted(n for n, _ in lengths)
    manifest = {
        "episodes": len(rows), "sha256": file_sha256(path), "recovery": args.recovery,
        "per_family": dict(Counter(r["family"] for r in rows)),
        "mistakes": dict(Counter(r["mistake"] for r in rows)), "rejected": dict(rejected),
        "dropped_same_prompt_and_answer_as_dev": dropped_pairs,
        "prompt_text_overlap_with_dev": sum(r["prompt"] in dev_prompts for r in rows),
        "dev_tasks_sha256": file_sha256(dev_path), "train_db_seeds": sorted({r["db_seed"] for r in rows}),
        "test_split": "not generated or read by this script",
        "calls_per_episode": dict(Counter(r["n_calls"] for r in rows)),
        "tokens_per_episode": {"min": tot[0], "median": tot[len(tot) // 2], "max": tot[-1],
                               "total": sum(tot)},
        "target_tokens_total": sum(t for _, t in lengths),
        "tokenizer_used_for_counts": getattr(tokenizer, "name_or_path", "?"),
        "system_message_sha256": hashlib.sha256(
            system_message(SYSTEM_PROMPT, fallback, schemas).encode()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
