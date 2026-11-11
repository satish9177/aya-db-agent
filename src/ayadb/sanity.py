"""Numerical sanity of the official model in a given precision (T4 has no bf16).

    python -m ayadb.sanity --out sanity.json [--load-4bit]

Exit 1 if the model cannot answer a trivial question, which would mean fp16/nf4 inference is
broken and any evaluation in that precision is meaningless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import load_hf

PROBES = [("What is 17 + 25? Reply with the number only.", "42"),
          ("What is the capital of France? Reply with one word.", "Paris")]


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--out", default="sanity.json")
    args = ap.parse_args(argv)
    model, tok = load_hf(args.model, None, args.load_4bit, args.dtype)
    rows = []
    for q, want in PROBES:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=16, do_sample=False)
        text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"question": q, "output": text, "ok": want.lower() in text.lower()})
    report = {"model": args.model, "load_4bit": args.load_4bit, "dtype": args.dtype, "probes": rows,
              "ok": all(r["ok"] for r in rows)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
