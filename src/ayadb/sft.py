"""QLoRA SFT of Tiny Aya on audited expert episodes, loss on assistant turns only.

    python -m ayadb.sft --episodes data/sft_pilot/episodes.jsonl --out runs/sft-pilot

No dev or test data is read. 5% of the training episodes are held out for a validation loss
curve (learning check only, not model selection). Writes the adapter and train_report.json
(configuration, loss history, peak GPU memory, runtime, tokens/sec).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

from .engine import bnb_config, official_tokenizer
from .env import build_db, make_tools
from .sftdata import conversation, load_episodes, system_message, tokenize
from .smoke import FORMATS, SYSTEM_PROMPT
from .tasks import TRAIN_POOL_SEEDS, file_sha256

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tool_schemas() -> list[dict]:
    """Schemas come from the tool docstrings, not the data; built on a training database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tools = make_tools(build_db(Path(tmp) / "schema.sqlite", TRAIN_POOL_SEEDS[0]))
        schemas = [_build_tool_schema(t) for t in tools]
        tools.close()
    return schemas


def build_examples(tokenizer, rows: list[dict], max_len: int) -> list[dict]:
    fmt = FORMATS["multistep-close"]
    schemas = tool_schemas()
    system = system_message(SYSTEM_PROMPT, fmt["fallback"], schemas)
    out = []
    for r in rows:
        conv = conversation(r["prompt"], r["messages"], system)
        flags = [False, False] + [bool(m.get("train", False)) for m in r["messages"]]
        ex = tokenize(tokenizer, conv, flags, schemas, max_len)
        if ex is None:
            raise ValueError(f"{r['id']} is longer than max_len={max_len}; not truncating")
        out.append({**ex, "id": r["id"]})
    return out


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        import torch

        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [self.pad_id] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--episodes", default="data/sft_pilot/episodes.jsonl")
    ap.add_argument("--out", default="runs/sft-pilot")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="preflight: stop after N steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--val-share", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float16", help="T4 has no bf16")
    ap.add_argument("--no-4bit", action="store_true", help="plumbing tests on CPU only")
    args = ap.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = official_tokenizer(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_episodes(Path(args.episodes))
    examples = build_examples(tokenizer, rows, args.max_len)
    rng = random.Random(args.seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    n_val = max(1, round(len(examples) * args.val_share))
    val = [examples[i] for i in order[:n_val]]
    train = [examples[i] for i in order[n_val:]]

    dtype = getattr(torch, args.dtype)
    cuda = torch.cuda.is_available()
    kw = {"dtype": dtype}
    if cuda:
        kw["device_map"] = {"": 0}
    if not args.no_4bit:
        kw["quantization_config"] = bnb_config(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(out / "trainer"), per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=1, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=0.05, weight_decay=0.0, max_grad_norm=1.0,
        logging_steps=2, eval_strategy="steps", eval_steps=10, save_strategy="no",
        fp16=cuda and args.dtype == "float16", bf16=cuda and args.dtype == "bfloat16",
        optim="paged_adamw_8bit" if cuda else "adamw_torch",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", seed=args.seed, data_seed=args.seed, remove_unused_columns=False,
        dataloader_num_workers=0)
    strip = lambda xs: [{k: v for k, v in x.items() if k != "id"} for x in xs]  # noqa: E731
    trainer = Trainer(model=model, args=targs, train_dataset=strip(train), eval_dataset=strip(val),
                      data_collator=Collator(tokenizer.pad_token_id))
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    result = trainer.train()
    runtime = time.time() - t0
    final_eval = trainer.evaluate()
    model.save_pretrained(str(out / "adapter"))

    steps = result.global_step
    seen_examples = steps * args.batch * args.grad_accum
    tok_per_ex = sum(len(x["input_ids"]) for x in train) / len(train)
    tgt_per_ex = sum(sum(1 for t in x["labels"] if t != -100) for x in train) / len(train)
    report = {
        "model": args.model, "episodes_file": args.episodes,
        "episodes_sha256": file_sha256(Path(args.episodes)),
        "n_train": len(train), "n_val": len(val), "val_ids": [x["id"] for x in val],
        "config": {k: v for k, v in vars(args).items()},
        "lora_targets": LORA_TARGETS, "trainable_params": trainable, "total_params": total,
        "quantization": None if args.no_4bit else "nf4 + double quant, compute " + args.dtype,
        "steps": steps, "runtime_s": round(runtime, 1),
        "train_loss": result.training_loss, "final_eval_loss": final_eval.get("eval_loss"),
        "log_history": trainer.state.log_history,
        "tokens_per_s": round(seen_examples * tok_per_ex / runtime, 1),
        "target_tokens_per_s": round(seen_examples * tgt_per_ex / runtime, 1),
        "mean_tokens_per_example": round(tok_per_ex, 1),
        "mean_target_tokens_per_example": round(tgt_per_ex, 1),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "agenttune")},
    }
    if cuda:
        report["gpu"] = torch.cuda.get_device_name(0)
        report["peak_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        report["peak_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        report["versions"]["bitsandbytes"] = version("bitsandbytes")
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("log_history", "val_ids")}, indent=2))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h] + [report["final_eval_loss"]]
    if any(x is None or x != x or abs(x) == float("inf") for x in losses):  # fp16 overflow guard
        print("non-finite loss: training is numerically broken", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
