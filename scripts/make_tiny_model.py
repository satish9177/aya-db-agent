"""Tiny random-weight cohere2 model with Tiny Aya's tokenizer, for CPU plumbing tests only."""

import sys
from pathlib import Path

from transformers import Cohere2Config, Cohere2ForCausalLM

from ayadb.engine import ollama_blob, tokenizer_from_gguf

out = Path(sys.argv[1])
tok = tokenizer_from_gguf(ollama_blob("hf.co/CohereLabs/tiny-aya-global-GGUF:Q8_0"))
tok.save_pretrained(out)
cfg = Cohere2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=8192,
                    bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id)
Cohere2ForCausalLM(cfg).save_pretrained(out)
print("saved", out)
