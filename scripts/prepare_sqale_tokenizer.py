"""Fetch only tokenizer/config files for the existing Tiny Aya model; never model weights."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

repo = "CohereLabs/tiny-aya-global"
out = Path(__file__).resolve().parents[1] / "data" / "sqale" / "tokenizer"
out.mkdir(parents=True, exist_ok=True)
try:
    info = HfApi().model_info(repo, revision="af89d219b53ed9b13b8a4645f9c8028973510324")
    revision = info.sha
    allowed = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json"}
    files = []
    for sibling in info.siblings:
        if sibling.rfilename in allowed:
            path = Path(hf_hub_download(repo, sibling.rfilename, revision=revision, local_dir=out))
            files.append({"file": sibling.rfilename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (out / "source.json").write_text(json.dumps({"repo": repo, "revision": revision, "files": files}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repo": repo, "revision": revision, "files": [r["file"] for r in files]}))
except Exception as e:
    print("Tokenizer preparation stopped: " + type(e).__name__ + "; no model weights requested.")
    raise SystemExit(1) from None
