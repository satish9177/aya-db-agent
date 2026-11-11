"""Pre-training checks with the official tokenizer (run on the GPU host before any GPU work).

    python -m ayadb.kcheck --gguf-template data/gguf_chat_template.jinja --out check.json

Fails (exit 1) if the frozen dev set does not regenerate byte-identically, if an assistant turn
cannot end with <|END_RESPONSE|><|END_OF_TURN_TOKEN|>, if any SFT episode fails tokenization, or
if the tokenizer we load gives different ids from the model's own tokenizer.json on any probe,
SFT episode or dev prompt (transformers 5.x AutoTokenizer does; see engine.official_tokenizer).
A chat template that differs from the GGUF one used for the Ollama baseline is reported, not fatal:
the official template is the one every GPU comparison uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import HFEngine, official_tokenizer
from .sft import build_examples, tool_schemas
from .sftdata import conversation, load_episodes, render, system_message
from .smoke import FORMATS, SYSTEM_PROMPT
from .tasks import file_sha256, generate_split, is_frozen_dev, load_tasks

# Numbers, SQL, money in cents, Devanagari and code: where the v5 pre-tokenizer diverges.
PROBES = [
    "SELECT SUM(unit_price_cents * qty) FROM order_lines WHERE order_no = 104523;",
    "ANSWER: 1234567 and 12.50 and 2026-11-04 नमस्ते दुनिया",
    "Tool result for run_sql: [(1999,), (20450,), (3,)] open(path): x = 1000000",
]


def tokenizer_json(model: str) -> Path:
    if Path(model).is_dir():
        return Path(model) / "tokenizer.json"
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, "tokenizer.json"))


def main(argv=None) -> int:
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, GenerationConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--episodes", default="data/sft_pilot/episodes.jsonl")
    ap.add_argument("--gguf-template", default="data/gguf_chat_template.jinja")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="check.json")
    args = ap.parse_args(argv)

    tok = official_tokenizer(args.model)
    gen = GenerationConfig.from_pretrained(args.model)
    tj = tokenizer_json(args.model)
    raw = Tokenizer.from_file(str(tj))
    auto = AutoTokenizer.from_pretrained(args.model)  # recorded only, never used
    ends = {t: tok.convert_tokens_to_ids(t) for t in HFEngine.END_TOKENS}
    _, dev = generate_split("dev", Path(args.data))
    template = tok.chat_template.get("default") if isinstance(tok.chat_template, dict) else tok.chat_template
    report = {
        "dev_tasks_sha256": file_sha256(dev), "dev_frozen": is_frozen_dev(dev),
        "template_identical_to_gguf": template == Path(args.gguf_template).read_text(encoding="utf-8"),
        "tokenizer_class": type(tok).__name__, "tokenizer_json_sha256": file_sha256(tj),
        "bos": tok.bos_token, "eos": tok.eos_token, "pad": tok.pad_token,
        "end_token_ids": ends, "generation_eos_token_id": gen.eos_token_id,
        "unk_id": tok.unk_token_id,
    }
    ok = report["dev_frozen"] and all(v is not None and v != tok.unk_token_id for v in ends.values())

    # Ids on every text a run will tokenize: the official loader must equal tokenizer.json exactly.
    fmt = FORMATS["multistep-close"]
    schemas = tool_schemas()
    system = system_message(SYSTEM_PROMPT, fmt["fallback"], schemas)
    rows = load_episodes(Path(args.episodes))
    texts = {f"probe{i}": p for i, p in enumerate(PROBES)}
    texts |= {r["id"]: render(tok, conversation(r["prompt"], r["messages"], system), schemas, False)
              for r in rows}
    texts |= {t.id: render(tok, conversation(t.prompt, [], system), schemas, True) for t in load_tasks(dev)}
    ours = {k: tok(v, add_special_tokens=False)["input_ids"] for k, v in texts.items()}
    ref = {k: raw.encode(v, add_special_tokens=False).ids for k, v in texts.items()}
    v5 = {k: len(auto(v, add_special_tokens=False)["input_ids"]) for k, v in texts.items()}
    mismatched = [k for k in texts if ours[k] != ref[k]]
    report["tokenizer_ids"] = {
        "texts": len(texts), "probes": len(PROBES), "episodes": len(rows), "dev_prompts": len(texts) - len(rows) - len(PROBES),
        "identical_to_tokenizer_json": not mismatched, "mismatched": mismatched[:20],
        "tokens_tokenizer_json": sum(len(x) for x in ref.values()),
        "tokens_v5_autotokenizer": sum(v5.values()), "v5_autotokenizer_class": type(auto).__name__,
        "v5_autotokenizer_texts_differing": sum(1 for k in texts if v5[k] != len(ref[k])),
    }
    ok = ok and not mismatched
    try:
        ex = build_examples(tok, rows, 4096)
        lens = sorted(len(e["input_ids"]) for e in ex)
        report["episodes"] = {"n": len(ex), "tokens_min": lens[0], "tokens_median": lens[len(lens) // 2],
                              "tokens_max": lens[-1], "tokens_total": sum(lens),
                              "target_tokens_total": sum(sum(1 for t in e["labels"] if t != -100) for e in ex)}
    except (AssertionError, ValueError) as e:
        report["episodes"] = {"error": f"{type(e).__name__}: {e}"}
        ok = False
    report["ok"] = ok
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
