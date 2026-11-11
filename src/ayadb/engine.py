"""AgentTune rollout engine backed by a local Ollama server in raw mode.

The prompt is rendered by AgentTune's own `_render_chat_template` with the model's real
tokenizer, so the text Ollama sees is the same text a transformers/vLLM rollout would
see during training. Ollama only does the forward passes (GPU on Windows, no vLLM).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from agenttune.agentic.rollout_engines.base import RolloutEngine
from agenttune.agentic.rollout_engines.rollout_factory import (
    DEFAULT_TOOL_RESULT_FORMAT,
    _render_chat_template,
)

OLLAMA = "http://127.0.0.1:11434"


def ollama_blob(model: str) -> Path:
    """Local GGUF file behind an Ollama model tag (e.g. hf.co/Org/Repo-GGUF:Q8_0)."""
    name, _, tag = model.partition(":")
    host, _, repo = name.partition("/") if name.startswith("hf.co/") else ("registry.ollama.ai", "", name)
    if "/" not in repo:
        repo = f"library/{repo}"
    root = Path.home() / ".ollama" / "models"
    manifest = json.loads((root / "manifests" / host / repo / (tag or "latest")).read_text())
    layer = next(l for l in manifest["layers"] if l["mediaType"].endswith("image.model"))
    return root / "blobs" / layer["digest"].replace(":", "-")


def tokenizer_from_gguf(path: Path):
    """Byte-level BPE tokenizer + chat template rebuilt from GGUF metadata.

    transformers cannot load `cohere2` GGUF files. AgentTune only uses the tokenizer here to
    render the chat template, list special tokens and count prompt length, which this covers.
    """
    from gguf import GGUFReader
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    r = GGUFReader(str(path))

    def val(key):
        f = r.fields[key]
        part = f.parts[f.data[0]]
        return bytes(part).decode("utf-8") if f.types[0].name == "STRING" else part.tolist()[0]

    def arr(key, as_str=True):
        f = r.fields[key]
        return [bytes(f.parts[i]).decode("utf-8") if as_str else f.parts[i].tolist()[0]
                for i in f.data]

    tokens = arr("tokenizer.ggml.tokens")
    types = arr("tokenizer.ggml.token_type", as_str=False)
    merges = [tuple(m.split(" ", 1)) for m in arr("tokenizer.ggml.merges")]
    tk = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = [t for t, ty in zip(tokens, types) if ty == 3]  # 3 = control token
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in specials])
    arch = val("general.architecture")
    return PreTrainedTokenizerFast(
        tokenizer_object=tk, chat_template=val("tokenizer.chat_template"),
        bos_token=tokens[val("tokenizer.ggml.bos_token_id")],
        eos_token=tokens[val("tokenizer.ggml.eos_token_id")],
        pad_token=tokens[val("tokenizer.ggml.padding_token_id")]
        if "tokenizer.ggml.padding_token_id" in r.fields else None,
        model_max_length=val(f"{arch}.context_length"), name_or_path=f"gguf:{Path(path).name}")


def bnb_config(dtype: str = "float16"):
    """The one 4-bit quantization used by QLoRA training and by 4-bit evaluation."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=getattr(torch, dtype))


def official_tokenizer(model_id: str):
    """The model's own tokenizer.json, loaded as-is.

    On transformers 5.x AutoTokenizer builds Tiny Aya as `CohereTokenizer`, whose __init__ swaps
    tokenizer.json's pre-tokenizer (digits split in groups of three + a word regex) for a
    hard-coded Digits(individual_digits=True) + ByteLevel one. Text still round-trips, but the
    ids are not the ones the model was trained on (e.g. 74 vs 52 tokens on a SQL probe; llama.cpp
    agrees with tokenizer.json). Upstream fix pending: huggingface/transformers#49192.
    """
    from transformers import TokenizersBackend

    return TokenizersBackend.from_pretrained(model_id)


def load_hf(model_id: str, adapter: str | None = None, load_4bit: bool = False,
            dtype: str = "float16"):
    """Official model + tokenizer (optionally nf4 base and a LoRA adapter) for HFEngine."""
    import torch
    from transformers import AutoModelForCausalLM

    tok = official_tokenizer(model_id)
    kw = {"dtype": getattr(torch, dtype)}
    if torch.cuda.is_available():
        kw["device_map"] = {"": 0}
    if load_4bit:
        kw["quantization_config"] = bnb_config(dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def load_tokenizer(source: str):
    """HF repo id (needs licence access) or a local GGUF file (tokenizer + chat template only)."""
    p = Path(source)
    if p.suffix == ".gguf" or p.name.startswith("sha256-"):
        return tokenizer_from_gguf(p)
    return official_tokenizer(source)


def finish_completion(text: str, stops: list[str]) -> str:
    """Shared stop semantics for every backend: cut before the earliest stop string (as Ollama
    does), then put back a "}]" that closed a tool-call list, so the call stays valid JSON."""
    cut = min((i for i in (text.find(s) for s in stops) if i >= 0), default=-1)
    if cut >= 0:
        text = text[:cut]
    if "}]" in stops and '"tool_name"' in text and not text.rstrip().endswith("]"):
        text = text.rstrip() + "}]"
    return text


class OllamaRawEngine(RolloutEngine):
    def __init__(self, model: str, tokenizer, options: dict | None = None,
                 tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT, host: str = OLLAMA):
        self.model = model
        self.tokenizer = tokenizer
        self.options = {"num_ctx": 8192, "num_predict": 512, **(options or {})}
        self.tool_result_format = tool_result_format
        self.host = host
        self.calls: list[dict] = []  # per-generation usage log (tokens, seconds)
        self.first_prompt: str | None = None  # exact text of the first request, for the record
        bos = getattr(tokenizer, "bos_token", None)
        self._strip_bos = bos  # Ollama adds BOS itself in raw mode; avoid a double BOS

    def _get_tokenizer(self):
        return self.tokenizer

    def render(self, conversation: list[dict], tools: list) -> str:
        text = _render_chat_template(self.tokenizer, conversation, tools,
                                     tool_result_format=self.tool_result_format, tokenize=False)
        if self._strip_bos and text.startswith(self._strip_bos):
            text = text[len(self._strip_bos):]
        return text

    def generate(self, prompts, tools, gen_cfg):
        conversation = prompts  # AgentTune's engine path passes one conversation
        text = self.render(conversation, tools)
        if self.first_prompt is None:
            self.first_prompt = text
        opts = dict(self.options)
        for k in ("temperature", "top_p", "top_k", "seed"):
            if k in gen_cfg:
                opts[k] = gen_cfg[k]
        t0 = time.time()
        r = requests.post(f"{self.host}/api/generate", timeout=600, json={
            "model": self.model, "prompt": text, "raw": True, "stream": False,
            "keep_alive": "30m", "options": opts})
        r.raise_for_status()
        out = r.json()
        self.calls.append({"prompt_tokens": out.get("prompt_eval_count"),
                           "completion_tokens": out.get("eval_count"),
                           "seconds": round(time.time() - t0, 3),
                           "done_reason": out.get("done_reason")})
        completion = finish_completion(out.get("response", ""), opts.get("stop", []))
        return {"completions": [completion], "completions_raw": [completion],
                "logprobs": [None], "metadata": {}}


class HFEngine(RolloutEngine):
    """Same protocol with a transformers model (official weights, optional LoRA adapter)."""

    END_TOKENS = ("<|END_OF_TURN_TOKEN|>", "<|END_RESPONSE|>")

    def __init__(self, model, tokenizer, stops: list[str], max_new_tokens: int = 512,
                 tool_result_format: str = DEFAULT_TOOL_RESULT_FORMAT):
        self.model, self.tokenizer, self.stops = model, tokenizer, list(stops)
        self.max_new_tokens = max_new_tokens
        self.tool_result_format = tool_result_format
        self.calls: list[dict] = []
        self.first_prompt: str | None = None
        ids = [tokenizer.convert_tokens_to_ids(t) for t in self.END_TOKENS]
        self.eos_ids = [i for i in ids + [tokenizer.eos_token_id] if i is not None]

    def _get_tokenizer(self):
        return self.tokenizer

    def render(self, conversation: list[dict], tools: list) -> str:
        return _render_chat_template(self.tokenizer, conversation, tools,
                                     tool_result_format=self.tool_result_format, tokenize=False)

    def generate(self, prompts, tools, gen_cfg):
        import torch

        text = self.render(prompts, tools)
        if self.first_prompt is None:
            self.first_prompt = text
        enc = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        temp = float(gen_cfg.get("temperature", 0.0))
        sampling = ({"do_sample": True, "temperature": temp, "top_p": gen_cfg.get("top_p", 0.95)}
                    if temp > 0 else {"do_sample": False})
        if temp > 0 and "seed" in gen_cfg:
            torch.manual_seed(int(gen_cfg["seed"]))
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      eos_token_id=self.eos_ids, pad_token_id=self.tokenizer.pad_token_id,
                                      stop_strings=self.stops or None, tokenizer=self.tokenizer,
                                      **sampling)
        new = out[0, enc["input_ids"].shape[1]:]
        hit_eos = len(new) > 0 and int(new[-1]) in self.eos_ids
        self.calls.append({"prompt_tokens": int(enc["input_ids"].shape[1]),
                           "completion_tokens": int(len(new)), "seconds": round(time.time() - t0, 3),
                           "done_reason": "stop" if hit_eos or len(new) < self.max_new_tokens else "length"})
        completion = finish_completion(self.tokenizer.decode(new, skip_special_tokens=True), self.stops)
        return {"completions": [completion], "completions_raw": [completion],
                "logprobs": [None], "metadata": {}}
