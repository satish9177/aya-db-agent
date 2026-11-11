"""Prepare, push, follow and pull the private Kaggle SFT pilot job.

    python scripts/kaggle_job.py prepare sft-pilot
    python scripts/kaggle_job.py push sft-pilot      # private kernel, T4x2, internet on
    python scripts/kaggle_job.py status sft-pilot
    python scripts/kaggle_job.py pull sft-pilot      # -> remote/jobs/sft-pilot/output/

The kernel reads the HF token from the Kaggle secret HF_TOKEN (attach it to the notebook once).
"""

import base64
import datetime as dt
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = Path.home() / "AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
USER = "satishkommanaboina"
SHIP = ["pyproject.toml", "README.md", "src", "data/sft_pilot", "data/gguf_chat_template.jinja"]


def kaggle(*args, check=True):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    text = (r.stdout or "") + (r.stderr or "")
    if check and (r.returncode != 0 or "error" in text.lower()):  # the CLI exits 0 on some refusals
        sys.exit(f"kaggle {' '.join(args)} failed:\n{text[-800:]}")
    return text


def job_dir(name):
    return ROOT / "remote" / "jobs" / name


def prepare(name):
    from ayadb.engine import ollama_blob, tokenizer_from_gguf

    tpl = ROOT / "data" / "gguf_chat_template.jinja"
    tpl.write_text(tokenizer_from_gguf(ollama_blob("hf.co/CohereLabs/tiny-aya-global-GGUF:Q8_0")).chat_template,
                   encoding="utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for item in SHIP:
            p = ROOT / item
            for f in ([p] if p.is_file() else sorted(p.rglob("*"))):
                if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".egg-info"):
                    z.write(f, f.relative_to(ROOT).as_posix())
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    job = {"name": name, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "git": git + ("-dirty" if dirty else "")}
    script = ((ROOT / "kaggle" / "job_template.py").read_text(encoding="utf-8")
              .replace("__PAYLOAD__", base64.b64encode(buf.getvalue()).decode())
              .replace("__JOB__", json.dumps(job)))
    d = job_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(script, encoding="utf-8", newline="\n")
    meta = {"id": f"{USER}/aya-db-agent-{name}", "title": f"aya-db-agent-{name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "competition_sources": [],
            "dataset_sources": [], "kernel_sources": [], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (d / "job.json").write_text(json.dumps({**job, "kernel": meta["id"], "payload_kb": len(buf.getvalue()) // 1024},
                                           indent=2), encoding="utf-8")
    print(f"prepared {d} ({len(buf.getvalue()) // 1024} KB payload, code {job['git']})")


def running_gpu_kernels():
    rows = kaggle("kernels", "list", "--mine", "--csv", "--page-size", "50").splitlines()[1:]
    busy = []
    for row in rows:
        ref = row.split(",")[0]
        st = kaggle("kernels", "status", ref, check=False)
        if "running" in st.lower() or "queued" in st.lower():
            busy.append(f"{ref}: {st.strip()[-60:]}")
    return busy


def push(name):
    busy = running_gpu_kernels()
    if busy:
        sys.exit("another kernel is running or queued; not pushing:\n" + "\n".join(busy))
    print(kaggle("kernels", "push", "-p", str(job_dir(name).resolve())).strip())


def status(name):
    meta = json.loads((job_dir(name) / "kernel-metadata.json").read_text(encoding="utf-8"))
    print(kaggle("kernels", "status", meta["id"], check=False).strip())


def pull(name):
    meta = json.loads((job_dir(name) / "kernel-metadata.json").read_text(encoding="utf-8"))
    out = job_dir(name) / "output"
    out.mkdir(exist_ok=True)
    print(kaggle("kernels", "output", meta["id"], "-p", str(out.resolve()), check=False)[-400:])


if __name__ == "__main__":
    cmd, name = sys.argv[1], sys.argv[2]
    {"prepare": prepare, "push": push, "status": status, "pull": pull}[cmd](name)
