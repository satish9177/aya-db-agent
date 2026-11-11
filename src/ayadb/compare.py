"""Paired report on the frozen dev set: the SFT pilot's preregistered comparisons.

    python -m ayadb.compare remote/hf --out experiments/sft-pilot/RESULTS-dev.md
    python -m ayadb.compare remote/hf --split test --out experiments/sft-pilot/RESULTS-test.md

Finds each run as `<root>/**/runs/<timestamp>-<tag>/` (the newest if a tag ran twice) and reports,
overall and per family: verified success, error recovery, malformed calls, fabricated tool output,
loops and tool-call efficiency; exact McNemar on paired task outcomes (primary base-nf4 vs
sft-nf4-r0, secondary base-fp16 vs sft-fp16); A/A flips between sft-nf4-r0 and r1; the
invalid-run checks; and the decision rule. Runs that are missing are skipped.

`--split test` reports the sealed test as preregistered in experiments/sft-pilot/SEALED-TEST.md:
test-base-nf4 vs test-sft-nf4 on the held-out families (primary) and the shared families, the
shared families split by whether the question appears verbatim in training, template replay, and
the interpretation rule.

`--split recovery` reports the recovery experiment (experiments/recovery/PREREGISTRATION.md): arm A
(the pilot adapter) vs arm B on the schema-drift dev variant (primary), the frozen dev set (guard)
and the new multi-hop split (secondary), and what the agent does after each failed call.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .tasks import DRIFT_RENAMES, FROZEN_DEV_SHA256, FROZEN_DEV_SHA256_LF, HELD_OUT, PINNED_LF, extract_answer

TAGS = ("base-nf4", "sft-nf4-r0", "sft-nf4-r1", "base-fp16", "sft-fp16")
PAIRS = (("base-nf4", "sft-nf4-r0", "primary"), ("base-fp16", "sft-fp16", "secondary"))
TEST_TAGS = ("test-base-nf4", "test-sft-nf4")
EPISODES = Path("data/sft_pilot/episodes.jsonl")
TEST_SHA256_LF = "dbe99529da23222b3093552cdd099af9830043531d85ce83c8df5c4366745d14"  # pinned in SEALED-TEST.md
RECOVERY_PAIRS = {"drift": ("drift-sft-a", "drift-sft-b"), "dev": ("sft-nf4-r0", "dev-sft-b"),
                  "mh": ("mh-sft-a", "mh-sft-b")}
RECOVERY_TAGS = tuple(t for pair in RECOVERY_PAIRS.values() for t in pair)
RECOVERY_B_SHA256 = "d0bf1bb35f551e28db0a539a9480aab07ee030791fc8a4550bd7dae4a3de0180"  # data/sft_recovery/episodes.jsonl, pinned in the preregistration


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def load_runs(root: Path, tags=TAGS) -> dict:
    runs = {}
    for tag in tags:
        dirs = sorted(d for d in root.glob(f"**/runs/*-{tag}") if (d / "metrics.jsonl").exists()
                      and re.fullmatch(rf"\d{{8}}-\d{{6}}-{re.escape(tag)}", d.name))
        if dirs:
            d = dirs[-1]
            rows = [json.loads(line) for line in (d / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            traj = [json.loads(line) for line in (d / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
            runs[tag] = {"dir": d, "rows": [r for r in rows if r.get("repeat", 0) == 0],
                         "traj": {t["trajectory_id"].split("#")[0]: t for t in traj if t["trajectory_id"].endswith("#r0")},
                         "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
                         "config": json.loads((d / "config.json").read_text(encoding="utf-8")),
                         "reruns": len(dirs) - 1}
    return runs


def diagnostics(rows: list[dict]) -> dict:
    """The reported diagnostics for a set of episodes (a run or one family of it)."""
    n = len(rows)
    count = lambda f: sum(1 for r in rows if f(r))  # noqa: E731
    mean = lambda k: sum(r[k] for r in rows) / n if n else 0.0  # noqa: E731
    ok = [r for r in rows if r["success"]]
    return {
        "n": n,
        "success": count(lambda r: r["success"]),
        "answer_seen_in_sql": count(lambda r: r["answer_seen_in_sql"]),
        "errored": count(lambda r: r["had_error"]),
        "recovered": count(lambda r: r["recovered"]),
        "malformed": count(lambda r: r["malformed_final"] or r["bad_arg_calls"] or r["unknown_tool_calls"]
                           or r["call_with_text_turns"]),
        "malformed_final": count(lambda r: r["malformed_final"]),
        "call_with_text": count(lambda r: r["call_with_text_turns"]),
        "fabricated": count(lambda r: r["hallucinated_tool_output"] or r["fake_output_turns"]),
        "loop": count(lambda r: r["loop"]),
        "hit_max": count(lambda r: r["hit_max_steps"]),
        "calls": mean("tool_calls"),
        "calls_ok": sum(r["tool_calls"] for r in ok) / len(ok) if ok else None,
        "turns": mean("turns"),
        "tokens": mean("completion_tokens"),
    }


def _row(name: str, d: dict) -> str:
    f = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
    return (f"| {name} | **{d['success']}/{d['n']}** | {d['answer_seen_in_sql']} | {d['recovered']}/{d['errored']} | "
            f"{d['malformed']} ({d['malformed_final']} final, {d['call_with_text']} text) | {d['fabricated']} | "
            f"{d['loop']} ({d['hit_max']} max) | {f(d['calls'])} / {f(d['calls_ok'])} | {f(d['turns'])} | "
            f"{d['tokens']:.0f} |")


HEADER = ("| run | verified success | answer seen in SQL | recovered / errored | malformed episodes | "
          "fabricated output | loops (budget hit) | calls per task / per success | turns | completion tokens |\n"
          "|---|---|---|---|---|---|---|---|---|---|")


def report(runs: dict, train_report: Path | None = None) -> str:
    out = ["# SFT pilot: dev-set results", "",
           "Frozen 40-task dev set, `multistep-close`, greedy, 8 tool rounds, 512 tokens per turn. "
           "Episode counts out of 40 unless stated. Malformed = an unparseable final turn, bad arguments, an "
           "unknown tool, or text around a call. Fabricated = invented tool output. Recovered = a successful query "
           "after the episode's last error, with a verified answer.", ""]

    out += ["## Validity", ""]
    for tag, r in runs.items():
        sha = r["config"].get("tasks_sha256")
        orch = r["summary"]["protocol"]["orchestration"]
        audit = (r["summary"].get("auditkit") or {}).get("verdict_agreement")
        out.append(f"- `{tag}` ({r['dir'].name}): dev frozen {sha in (FROZEN_DEV_SHA256, FROZEN_DEV_SHA256_LF)}, "
                   f"orchestration {orch}, AuditKit agreement {audit}"
                   + (f", **{r['reruns']} earlier run(s) of this tag ignored**" if r["reruns"] else ""))
    if train_report and train_report.exists():
        tr = json.loads(train_report.read_text(encoding="utf-8"))
        finite = all(math.isfinite(h[k]) for h in tr["log_history"] for k in ("loss", "eval_loss") if k in h)
        out.append(f"- training: {tr['steps']} steps, {tr['runtime_s'] / 60:.1f} min, train loss "
                   f"{tr['train_loss']:.3f}, final eval loss {tr['final_eval_loss']:.3f}, all losses finite {finite}, "
                   f"episodes sha256 `{tr['episodes_sha256'][:12]}...`")
    out.append("")

    out += ["## Overall", "", HEADER]
    out += [_row(f"`{tag}`", diagnostics(r["rows"])) for tag, r in runs.items()]
    out.append("")

    fams = sorted({row["family"] for r in runs.values() for row in r["rows"]})
    out += ["## Verified success by family", "", "| family | " + " | ".join(f"`{t}`" for t in runs) + " |",
            "|---|" + "---|" * len(runs)]
    for fam in fams:
        cells = [diagnostics([row for row in r["rows"] if row["family"] == fam]) for r in runs.values()]
        out.append(f"| {fam} | " + " | ".join(f"{c['success']}/{c['n']}" for c in cells) + " |")
    out.append("")

    for tag in ("base-nf4", "sft-nf4-r0"):
        if tag in runs:
            out += [f"## Per family: `{tag}`", "", HEADER.replace("| run |", "| family |", 1)]
            out += [_row(fam, diagnostics([row for row in runs[tag]["rows"] if row["family"] == fam])) for fam in fams]
            out.append("")

    out += ["## Paired comparisons (exact McNemar)", ""]
    for a, b, kind in PAIRS:
        if a in runs and b in runs:
            sa = {r["task_id"]: r["success"] for r in runs[a]["rows"]}
            sb = {r["task_id"]: r["success"] for r in runs[b]["rows"]}
            common = sorted(sa.keys() & sb.keys())
            gained = sum(1 for t in common if sb[t] and not sa[t])
            lost = sum(1 for t in common if sa[t] and not sb[t])
            out.append(f"- {kind}: `{a}` {sum(sa[t] for t in common)}/{len(common)} -> `{b}` "
                       f"{sum(sb[t] for t in common)}/{len(common)}; gained {gained}, lost {lost}, "
                       f"p = {mcnemar_exact(gained, lost):.2g}")
    if "sft-nf4-r0" in runs and "sft-nf4-r1" in runs:
        r0 = {r["task_id"]: r for r in runs["sft-nf4-r0"]["rows"]}
        r1 = {r["task_id"]: r for r in runs["sft-nf4-r1"]["rows"]}
        common = r0.keys() & r1.keys()
        flips = sum(1 for t in common if r0[t]["success"] != r1[t]["success"])
        answers = sum(1 for t in common if r0[t]["answer"] != r1[t]["answer"])
        out.append(f"- A/A (`sft-nf4-r0` vs `sft-nf4-r1`, identical runs): {flips} success flips, "
                   f"{answers} different final answers out of {len(common)}")
    out.append("")

    if "sft-nf4-r0" in runs:
        k = sum(r["success"] for r in runs["sft-nf4-r0"]["rows"])
        verdict = ("proceed to broader evaluation" if k >= 12 else
                   "investigate whether the gains include real multistep behaviour" if k >= 4 else
                   "pivot unless a concrete implementation fault is identified")
        out += ["## Decision rule (preregistered, on `sft-nf4-r0`)", "",
                f"{k}/40 verified successes: **{verdict}** (>= 12 proceed, 4-11 investigate, <= 3 pivot).", ""]
    return "\n".join(out)


def _calls(content: str) -> list:
    try:
        v = json.loads(content)
    except ValueError:
        m = re.search(r"\[\s*\{.*\}\s*\]", content, re.DOTALL)
        try:
            v = json.loads(m.group(0)) if m else []
        except ValueError:
            v = []
    return [c for c in v if isinstance(c, dict)] if isinstance(v, list) else []


def _queries(messages: list) -> list[str]:
    return [str((c.get("parameters") or {}).get("query", "")) for m in messages if m.get("role") == "assistant"
            for c in _calls(m.get("content") or "") if c.get("tool_name") == "run_sql"]


def _template(query: str) -> str:
    """The query with whitespace, case, and string and number literals abstracted."""
    q = re.sub(r"\s+", " ", query.strip().rstrip(";")).lower()
    return re.sub(r"'[^']*'", "'S'", re.sub(r"\b\d+(\.\d+)?\b", "N", q))


def _paired(runs: dict, a: str, b: str, ids: set, label: str) -> dict:
    sa = {r["task_id"]: r["success"] for r in runs[a]["rows"] if r["task_id"] in ids}
    sb = {r["task_id"]: r["success"] for r in runs[b]["rows"] if r["task_id"] in ids}
    common = sorted(sa.keys() & sb.keys())
    d = {"n": len(common), "a": sum(sa[t] for t in common), "b": sum(sb[t] for t in common),
         "gained": sum(1 for t in common if sb[t] and not sa[t]), "lost": sum(1 for t in common if sa[t] and not sb[t])}
    d["p"] = mcnemar_exact(d["gained"], d["lost"])
    d["text"] = (f"- {label}: `{a}` {d['a']}/{d['n']} -> `{b}` {d['b']}/{d['n']}; gained {d['gained']}, "
                 f"lost {d['lost']}, p = {d['p']:.2g}")
    return d


def test_report(runs: dict, episodes: Path = EPISODES) -> str:
    a, b = TEST_TAGS
    out = ["# SFT pilot: sealed test results", "",
           "Preregistered in `experiments/sft-pilot/SEALED-TEST.md` before the split was generated. Same protocol "
           "as dev (`multistep-close`, greedy, 8 tool rounds, 512 tokens per turn), nf4 base, the pilot adapter "
           "unchanged. Column definitions as in `RESULTS-dev.md`.", "", "## Validity", ""]
    for tag, r in runs.items():
        c = r["config"]
        out.append(f"- `{tag}` ({r['dir'].name}): split {c.get('split')}, {c.get('n_tasks')} tasks, tasks sha256 "
                   f"`{c.get('tasks_sha256')}`, adapter `{(c.get('adapter_sha256') or 'none')[:12]}`, orchestration "
                   f"{r['summary']['protocol']['orchestration']}, AuditKit agreement "
                   f"{(r['summary'].get('auditkit') or {}).get('verdict_agreement')}"
                   + (f", **{r['reruns']} earlier run(s) of this tag ignored**" if r["reruns"] else ""))
    if a not in runs or b not in runs:
        return "\n".join(out + ["", "Both runs are needed for the comparison."])
    same = runs[a]["config"]["tasks_sha256"] == runs[b]["config"]["tasks_sha256"]
    pinned = all(r["config"]["tasks_sha256"] == TEST_SHA256_LF for r in runs.values())
    out += [f"- both runs saw the same tasks: {same}; tasks are the pinned split: {pinned}", ""]

    eps = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_prompts = {e["prompt"] for e in eps}
    train_templates = {_template(q) for e in eps for q in _queries(e["messages"])}
    fam = {r["task_id"]: r["family"] for r in runs[a]["rows"]}
    prompt = {tid: t["task"] for tid, t in runs[a]["traj"].items()}
    held = {t for t, f in fam.items() if f in HELD_OUT}
    shared = set(fam) - held
    seen = {t for t in shared if prompt.get(t) in train_prompts}
    groups = {"held-out families": held, "shared families": shared,
              "shared, question verbatim in training": seen, "shared, question not in training": shared - seen}
    order = sorted(set(fam.values()), key=lambda f: (f not in HELD_OUT, f))

    out += ["## Overall", "", HEADER.replace("| run |", "| run, tasks |", 1)]
    for name in ("held-out families", "shared families"):
        for tag, r in runs.items():
            out.append(_row(f"`{tag}`, {name}", diagnostics([row for row in r["rows"] if row["task_id"] in groups[name]])))
    out.append("")

    out += ["## Verified success by family", "", "| family | held out | " + " | ".join(f"`{t}`" for t in runs) + " |",
            "|---|---|" + "---|" * len(runs)]
    for f in order:
        cells = [diagnostics([row for row in r["rows"] if row["family"] == f]) for r in runs.values()]
        out.append(f"| {f} | {'yes' if f in HELD_OUT else ''} | "
                   + " | ".join(f"{c['success']}/{c['n']}" for c in cells) + " |")
    out.append("")

    out += [f"## Per family: `{b}`", "", HEADER.replace("| run |", "| family |", 1)]
    out += [_row(f, diagnostics([row for row in runs[b]["rows"] if row["family"] == f])) for f in order]
    out.append("")

    out += ["## Paired comparisons (exact McNemar)", ""]
    res = {name: _paired(runs, a, b, ids, ("**primary**, " if name == "held-out families" else "") + name)
           for name, ids in groups.items()}
    out += [d["text"] for d in res.values()] + [""]

    out += [f"## Template replay (`{b}`)", "",
            "Episodes whose last query matches a training query once whitespace, case and literals are abstracted.", ""]
    for name in ("held-out families", "shared families"):
        rows = {r["task_id"]: r for r in runs[b]["rows"] if r["task_id"] in groups[name]}
        last = {t: (_queries(runs[b]["traj"][t]["metadata"]["conversation"]) or [""])[-1] for t in rows}
        replay = {t for t, q in last.items() if q and _template(q) in train_templates}
        out.append(f"- {name}: {len(replay)}/{len(rows)} episodes replay a training template "
                   f"({sum(rows[t]['success'] for t in replay)} of them verified); "
                   f"{sum(r['success'] for t, r in rows.items() if t not in replay)} verified successes with a new query")
    out.append("")

    h, s = res["held-out families"], res["shared families"]
    if h["p"] < 0.05 and h["b"] > h["a"] and h["b"] >= 0.5 * h["n"]:
        verdict = "transfer: SFT carries over to unseen multi-hop question types"
    elif h["p"] < 0.05 and h["b"] > h["a"]:
        verdict = "partial transfer to unseen question types"
    else:
        verdict = "no evidence of transfer beyond the trained question types; the claim stays within them"
    out += ["## Interpretation (preregistered)", "",
            f"Held-out families: {h['a']}/{h['n']} -> {h['b']}/{h['n']}, p = {h['p']:.2g}: **{verdict}**."]
    if s["lost"] > s["gained"] and s["p"] < 0.05:
        out.append("Shared families regressed significantly: reported as a failure.")
    if s["n"] and s["b"] < 0.8 * s["n"]:
        out.append(f"Shared families below 80% ({s['b']}/{s['n']}): the dev result overestimates in-family performance.")
    return "\n".join(out)


# ---- recovery experiment (experiments/recovery/PREREGISTRATION.md) --------------------------------

def error_kind(result: str) -> str:
    r = str(result).lower()
    for kind, needle in (("no such column", "no such column"), ("no such table", "no such table"),
                         ("no such table", "no table named"), ("ambiguous column", "ambiguous column"),
                         ("syntax", "syntax error"), ("syntax", "incomplete input"),
                         ("bad arguments", "invalid arguments")):
        if needle in r:
            return kind
    return "other"


def after_errors(traj: dict) -> dict:
    """What the agent did right after each failed call: repeated it, changed the query (and whether
    that worked), looked up the schema, answered, or stopped (budget or unparseable turn)."""
    log = traj["metadata"]["call_log"]
    answered = extract_answer(traj.get("final_response") or "") is not None
    out = {k: 0 for k in AFTER_ERROR}
    kinds = {}
    for i, c in enumerate(log):
        if not c["error"]:
            continue
        kinds[error_kind(c["result"])] = kinds.get(error_kind(c["result"]), 0) + 1
        nxt = log[i + 1] if i + 1 < len(log) else None
        if nxt is None:
            out["answered" if answered else "stopped"] += 1
        elif (nxt["name"], nxt["args"]) == (c["name"], c["args"]):
            out["repeated"] += 1
        elif nxt["name"] == "run_sql" and c["name"] == "run_sql":
            out["edited, failed" if nxt["error"] else "edited, ran"] += 1
        elif nxt["name"] in ("list_tables", "describe_table"):
            out["schema lookup"] += 1
        else:
            out["other"] += 1
    return {"next": out, "kinds": kinds}


AFTER_ERROR = ("edited, ran", "edited, failed", "repeated", "schema lookup", "answered", "stopped", "other")


def _trained_templates(path: Path) -> set:
    """Literal-abstracted queries of the trained (non-mistake) turns of an episode file."""
    eps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {_template(str((c.get("parameters") or {}).get("query", ""))) for e in eps for m in e["messages"]
            if m["role"] == "assistant" and m.get("train", True)
            for c in _calls(m["content"]) if c.get("tool_name") == "run_sql"}


def recovery_report(runs: dict, train_report: Path | None = None) -> str:
    out = ["# Recovery experiment: results", "",
           "Preregistered in `experiments/recovery/PREREGISTRATION.md`. Arm A = the pilot adapter (discovery-phase "
           "mistakes), arm B = the same 235 episodes with query-phase mistakes at the same 66 positions, trained with "
           "the same settings. Protocol as on dev (`multistep-close`, greedy, 8 tool rounds, 512 tokens per turn, nf4). "
           "Column definitions as in `experiments/sft-pilot/RESULTS-dev.md`.", "", "## Validity", ""]
    for tag, r in runs.items():
        c = r["config"]
        split = c.get("split")
        sha = c.get("tasks_sha256")
        pinned = PINNED_LF.get(split)
        ok = sha == pinned or (split == "dev" and sha in (FROZEN_DEV_SHA256, FROZEN_DEV_SHA256_LF))
        out.append(f"- `{tag}` ({r['dir'].name}): split {split}, {c.get('n_tasks')} tasks, pinned tasks {ok}, adapter "
                   f"`{(c.get('adapter_sha256') or 'none')[:12]}`, orchestration "
                   f"{r['summary']['protocol']['orchestration']}, AuditKit agreement "
                   f"{(r['summary'].get('auditkit') or {}).get('verdict_agreement')}"
                   + (f", **{r['reruns']} earlier run(s) of this tag ignored**" if r["reruns"] else ""))
    if train_report and train_report.exists():
        tr = json.loads(train_report.read_text(encoding="utf-8"))
        finite = all(math.isfinite(h[k]) for h in tr["log_history"] for k in ("loss", "eval_loss") if k in h)
        out.append(f"- arm B training: {tr['steps']} steps, {tr['runtime_s'] / 60:.1f} min, train loss "
                   f"{tr['train_loss']:.3f}, final eval loss {tr['final_eval_loss']:.3f}, all losses finite {finite}, "
                   f"episodes sha256 `{tr['episodes_sha256'][:12]}...` (pinned {tr['episodes_sha256'] == RECOVERY_B_SHA256})")
    out.append("")

    out += ["## Overall", "", HEADER.replace("| run |", "| run, split |", 1)]
    out += [_row(f"`{tag}`, {r['config'].get('split')}", diagnostics(r["rows"])) for tag, r in runs.items()]
    out.append("")

    out += ["## Verified success by family", ""]
    for split, (a, b) in (("drift", RECOVERY_PAIRS["drift"]), ("mh", RECOVERY_PAIRS["mh"])):
        if a in runs and b in runs:
            fams = sorted({row["family"] for row in runs[a]["rows"]})
            out += [f"{split}:", "", f"| family | `{a}` | `{b}` |", "|---|---|---|"]
            for f in fams:
                da, db = (diagnostics([x for x in runs[t]["rows"] if x["family"] == f]) for t in (a, b))
                out.append(f"| {f} | {da['success']}/{da['n']} | {db['success']}/{db['n']} |")
            out.append("")

    out += ["## After an error", "",
            "Every failed call, by what the agent did next. Edited = a different query on the next call.", "",
            "| run | failed calls | " + " | ".join(AFTER_ERROR) + " | error kinds |",
            "|---|---|" + "---|" * len(AFTER_ERROR) + "---|"]
    for tag, r in runs.items():
        nxt, kinds = {k: 0 for k in AFTER_ERROR}, {}
        for t in r["traj"].values():
            ae = after_errors(t)
            for k, v in ae["next"].items():
                nxt[k] += v
            for k, v in ae["kinds"].items():
                kinds[k] = kinds.get(k, 0) + v
        out.append(f"| `{tag}` | {sum(nxt.values())} | " + " | ".join(str(nxt[k]) for k in AFTER_ERROR) + " | "
                   + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + " |")
    out.append("")

    trained = _trained_templates(EPISODES)
    renamed = re.compile(r"\b(" + "|".join(c for _, c in DRIFT_RENAMES) + r")\b", re.I)
    out += ["## Memory vs schema on drift", "",
            "First query of each episode: does it replay a training query (training column names, literals "
            "abstracted)? On the drift database such a query fails if it names a renamed column. (Corrected "
            "2026-11-05 after the runs: the preregistered wording said it always fails, but a two-hop family's "
            "first hop uses no renamed column.)", ""]
    for tag in RECOVERY_PAIRS["drift"]:
        if tag in runs:
            firsts = [(_queries(t["metadata"]["conversation"]) or [""])[0] for t in runs[tag]["traj"].values()]
            replays = [q for q in firsts if q and _template(q) in trained]
            out.append(f"- `{tag}`: {len(replays)}/{len(firsts)} first queries replay a training query, "
                       f"{sum(1 for q in replays if renamed.search(q))} of them with a renamed column; "
                       f"{sum(1 for q in firsts if not q)} episodes run no query")
    out.append("")

    out += ["## Paired comparisons (exact McNemar)", ""]
    res = {}
    for name, (a, b) in RECOVERY_PAIRS.items():
        if a in runs and b in runs:
            ids = {x["task_id"] for x in runs[a]["rows"]}
            label = {"drift": "**primary**, drift", "dev": "guard, frozen dev", "mh": "secondary, new multi-hop"}[name]
            res[name] = _paired(runs, a, b, ids, label)
            out.append(res[name]["text"])
    out.append("")

    out += ["## Interpretation (preregistered)", ""]
    if "drift" not in res:
        return "\n".join(out + ["Both drift runs are needed for the primary comparison."])
    d = res["drift"]
    if d["p"] < 0.05 and d["gained"] > d["lost"]:
        verdict = "arm B (query-repair demonstrations) is more robust to schema drift"
    elif d["p"] < 0.05:
        verdict = "arm B (query-repair demonstrations) is less robust to schema drift"
    else:
        verdict = "no evidence of a difference between the two kinds of recovery demonstration"
    out.append(f"Drift, arm A {d['a']}/{d['n']} -> arm B {d['b']}/{d['n']}, p = {d['p']:.2g}: **{verdict}**.")
    a, b = RECOVERY_PAIRS["drift"]
    dg = {t: diagnostics(runs[t]["rows"]) for t in (a, b)}
    out.append(f"How (mechanism): recovered / errored episodes, arm A {dg[a]['recovered']}/{dg[a]['errored']}, "
               f"arm B {dg[b]['recovered']}/{dg[b]['errored']}; successes without any error, arm A "
               f"{dg[a]['success'] - dg[a]['recovered']}, arm B {dg[b]['success'] - dg[b]['recovered']}.")
    errored = {t: dg[t]["errored"] for t in (a, b)}
    if max(errored.values()) < 10:
        out.append(f"Drift exercised recovery in fewer than 10 episodes per arm ({errored[a]} and {errored[b]}): "
                   "the agent read the renamed schema instead of failing, so drift does not test recovery.")
    g = res.get("dev")
    guard = g is not None and g["b"] >= 38 and not (g["p"] < 0.05 and g["lost"] > g["gained"])
    if g is None:
        out.append("Frozen-dev guard not available: arm B cannot be adopted.")
    else:
        out.append(f"Frozen-dev guard (arm B >= 38/40, no significant loss): **{'passed' if guard else 'failed'}** "
                   f"({g['b']}/{g['n']}, lost {g['lost']}, gained {g['gained']}).")
    adopt = guard and d["p"] < 0.05 and d["gained"] > d["lost"]
    out.append(f"Arm B {'replaces' if adopt else 'does not replace'} the pilot adapter as the submission model.")
    if "mh" in res:
        m = res["mh"]
        out.append(f"New multi-hop (secondary, descriptive): {m['a']}/{m['n']} -> {m['b']}/{m['n']}, p = {m['p']:.2g}.")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", choices=["dev", "test", "recovery"], default="dev")
    ap.add_argument("--train-report", type=Path, default=None)
    ap.add_argument("--episodes", type=Path, default=EPISODES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.split == "test":
        text = test_report(load_runs(args.root, TEST_TAGS), args.episodes)
    elif args.split == "recovery":
        text = recovery_report(load_runs(args.root, RECOVERY_TAGS), args.train_report)
    else:
        text = report(load_runs(args.root), args.train_report)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
