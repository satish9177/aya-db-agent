"""Preregistered SFT pilot on Modal T4 (fp16, no bf16). Private; publishes nothing.

    modal run scripts/modal_job.py::check      # download, kcheck (tokenizer ids), fp16/nf4 sanity
    modal run scripts/modal_job.py::baseline   # base-nf4 | base-fp16 on the frozen dev set
    modal run scripts/modal_job.py::smoke      # 2 training steps, then 2 dev tasks with that adapter
    modal run scripts/modal_job.py::pilot      # full SFT, then sft-nf4-r0 | sft-nf4-r1 | sft-fp16
    modal volume get aya-db-agent-runs / remote/modal/   # pull everything

Each stage refuses to start unless the stage it depends on passed (baseline and smoke need check;
pilot needs smoke) and the git tree is clean, so every output maps to one commit. The HF token
comes from the Modal secret `huggingface` (key HF_TOKEN); the official repo is gated.
All evaluations: frozen dev set, multistep-close protocol, greedy, 8 tool rounds, 512 tokens/turn.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
RUNS, HF = "/runs", "/hf"
EVAL = ["-m", "ayadb.smoke", "--engine", "hf", "--hf-model", MODEL, "--split", "dev",
        "--format", "multistep-close", "--temperature", "0", "--max-steps", "8"]
SFT = ["-m", "ayadb.sft", "--model", MODEL, "--episodes", "data/sft_pilot/episodes.jsonl"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu126")
    .uv_pip_install("transformers==5.22.1", "tokenizers==0.23.3", "huggingface_hub==1.33.0",
                    "peft==0.21.2", "trl==1.7.1", "accelerate==1.15.0", "agenttune==1.1.3",
                    "auditkit==1.1.3", "bitsandbytes", "gguf", "requests")
    .env({"PYTHONPATH": "/repo/src", "HF_HOME": HF, "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"})
    .add_local_dir(ROOT / "src", "/repo/src", ignore=["**/__pycache__"])
    .add_local_file(ROOT / "pyproject.toml", "/repo/pyproject.toml")
    .add_local_dir(ROOT / "data" / "sft_pilot", "/repo/data/sft_pilot")
    .add_local_file(ROOT / "data" / "gguf_chat_template.jinja", "/repo/data/gguf_chat_template.jinja")
)
app = modal.App("aya-db-agent", image=image)
runs = modal.Volume.from_name("aya-db-agent-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("aya-db-agent-hf", create_if_missing=True)
SECRET = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
VOLS = {RUNS: runs, HF: hf_cache}


def _run_steps(stage: str, steps: list, job: dict) -> dict:
    """Run (name, argv, timeout_s) steps in order in this container; stop at the first failure."""
    out = Path(RUNS) / stage
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"stage": stage, "steps": {}}
    status.update(job=job, result="RUNNING")
    for name, argv, timeout in steps:
        t0 = time.time()
        with open(out / f"{name}.log", "w") as f:
            try:
                rc = subprocess.run([sys.executable, *argv], cwd="/repo", stdout=f, stderr=subprocess.STDOUT,
                                    timeout=timeout).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
        status["steps"][name] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
        status_path.write_text(json.dumps(status, indent=2))
        runs.commit()
        print(f"{stage}/{name}: rc={rc} in {status['steps'][name]['seconds']}s", flush=True)
        if rc != 0:
            status["result"] = f"FAILED_AT_{name}"
            break
    else:
        status["result"] = "OK"
    status_path.write_text(json.dumps(status, indent=2))
    runs.commit()
    return status


@app.function(volumes=VOLS, secrets=[SECRET], timeout=3600)
def cpu_steps(stage: str, steps: list, job: dict) -> dict:
    return _run_steps(stage, steps, job)


@app.function(gpu="T4", volumes=VOLS, secrets=[SECRET], timeout=4 * 3600)
def gpu_steps(stage: str, steps: list, job: dict) -> dict:
    runs.reload()  # see adapters written by an earlier container
    env = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return _run_steps(stage, steps, {**job, "gpu": env})


@app.function(volumes=VOLS)
def read_status(stage: str) -> dict | None:
    p = Path(RUNS) / stage / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def _job() -> dict:
    """Code identity of this launch; refuses a dirty tree so results map to one commit."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain") and not os.environ.get("AYADB_ALLOW_DIRTY"):
        sys.exit("git tree is dirty: commit first (or set AYADB_ALLOW_DIRTY=1 for a throwaway run)")
    return {"git": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "launched": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _require(stage: str, *steps: str):
    st = read_status.remote(stage)
    if not st or any(st["steps"].get(s, {}).get("rc") != 0 for s in steps):
        sys.exit(f"stage '{stage}' has not passed {steps}: {json.dumps(st, indent=2) if st else 'not run'}")


def _eval(tag: str, *extra: str) -> tuple:
    return (f"eval_{tag}", [*EVAL, "--tag", tag, "--data", f"/tmp/data-{tag}",
                            "--out", f"{RUNS}/{{stage}}/runs", *extra], 3 * 3600)


def _fill(steps, stage):
    return [(n, [a.replace("{stage}", stage) for a in argv], t) for n, argv, t in steps]


@app.local_entrypoint()
def check():
    job = _job()
    st = cpu_steps.remote("check", [
        ("download", ["-c", f"from huggingface_hub import snapshot_download; print(snapshot_download('{MODEL}'))"], 1800),
        ("kcheck", ["-m", "ayadb.kcheck", "--model", MODEL, "--out", f"{RUNS}/check/kcheck.json",
                    "--data", "/tmp/data-kcheck"], 1200),
    ], job)
    if st["result"] != "OK":
        sys.exit(json.dumps(st, indent=2))
    st = gpu_steps.remote("check", [
        ("env_info", ["-c", "import torch, transformers, peft, bitsandbytes, agenttune, tokenizers; "
                      "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "
                      "transformers.__version__, tokenizers.__version__, peft.__version__, "
                      "bitsandbytes.__version__, agenttune.__version__)"], 600),
        ("sanity_fp16", ["-m", "ayadb.sanity", "--out", f"{RUNS}/check/sanity_fp16.json"], 1800),
        ("sanity_nf4", ["-m", "ayadb.sanity", "--load-4bit", "--out", f"{RUNS}/check/sanity_nf4.json"], 1800),
    ], job)
    print(json.dumps(st, indent=2))


@app.local_entrypoint()
def baseline():
    job = _job()
    _require("check", "kcheck", "sanity_nf4")
    calls = [gpu_steps.spawn("baseline", _fill([_eval("base-nf4", "--load-4bit")], "baseline"), job)]
    if read_status.remote("check")["steps"].get("sanity_fp16", {}).get("rc") == 0:
        calls.append(gpu_steps.spawn("baseline-fp16", _fill([_eval("base-fp16")], "baseline"), job))
    for c in calls:
        print(json.dumps(c.get(), indent=2))


@app.local_entrypoint()
def smoke():
    job = _job()
    _require("check", "kcheck", "sanity_nf4")
    st = gpu_steps.remote("smoke", _fill([
        ("mini_train", [*SFT, "--max-steps", "2", "--out", f"{RUNS}/smoke/mini"], 1800),
        _eval("mini", "--limit", "2", "--adapter", f"{RUNS}/smoke/mini/adapter", "--load-4bit"),
    ], "smoke"), job)
    print(json.dumps(st, indent=2))


@app.local_entrypoint()
def pilot():
    job = _job()
    _require("check", "kcheck", "sanity_nf4")
    _require("smoke", "mini_train", "eval_mini")
    st = gpu_steps.remote("pilot", [("train", [*SFT, "--out", f"{RUNS}/pilot/sft-pilot"], 5400)], job)
    if st["result"] != "OK":
        sys.exit(json.dumps(st, indent=2))
    adapter = f"{RUNS}/pilot/sft-pilot/adapter"
    evals = [_eval("sft-nf4-r0", "--adapter", adapter, "--load-4bit"),
             _eval("sft-nf4-r1", "--adapter", adapter, "--load-4bit")]
    if read_status.remote("check")["steps"].get("sanity_fp16", {}).get("rc") == 0:
        evals.append(_eval("sft-fp16", "--adapter", adapter))
    calls = [gpu_steps.spawn(f"pilot-{e[0]}", _fill([e], "pilot"), job) for e in evals]
    for c in calls:
        print(json.dumps(c.get(), indent=2))
