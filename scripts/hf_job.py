"""Preregistered SFT pilot on Hugging Face Jobs, T4 (fp16, no bf16). Private; publishes nothing.

    python scripts/hf_job.py check       # kcheck (tokenizer ids) on CPU, then fp16/nf4 sanity on T4
    python scripts/hf_job.py baseline    # base-nf4 | base-fp16 on the frozen dev set
    python scripts/hf_job.py smoke       # 2 training steps, then 2 dev tasks with that adapter
    python scripts/hf_job.py pilot       # full SFT, then sft-nf4-r0 | sft-nf4-r1 | sft-fp16
    python scripts/hf_job.py pilot_eval  # only the three evaluations (training already passed)
    python scripts/hf_job.py sealed_test # once: base-nf4 | pilot adapter on the sealed test (SEALED-TEST.md)
    python scripts/hf_job.py recovery    # recovery experiment (experiments/recovery/PREREGISTRATION.md):
                                         # train arm B | arm A on drift, mh; then arm B on drift, dev, mh
    python scripts/hf_job.py recovery_eval  # only arm B's three evaluations (its training already passed)
    python scripts/hf_job.py status      # every stage's status.json
    python scripts/hf_job.py pull        # copy all results to remote/hf/

Same stages, gates and step commands as scripts/modal_job.py. Each launch copies the code and the
preregistered episodes to the private bucket {user}/aya-db-agent at code/<git sha>/, mounted
read-only; a job works on local disk and syncs /work/runs to runs/ in the same bucket after every
step. Each stage refuses to start unless the stage it depends on passed and the git tree is clean,
so every output maps to one commit. Needs a local `hf auth login` that can start Jobs, write to your
namespace and read gated repos; the job gets the token as HF_TOKEN (the official repo is gated).

GPU spending (lessons from the Modal runs in Google-Quantum-AI, kb/lessons.md):
- Work that needs no GPU (the tokenizer check) runs on a CPU job, never on the T4.
- Every GPU job first fails fast if CUDA is unusable: the engine silently falls back to CPU, which
  would bill T4 hours at CPU speed.
- A step that prints nothing for STALL_S is killed, so a hang can't bill until the job timeout.
- Ctrl+C in the launcher cancels the jobs it started. Each finished job prints its running time
  and an estimated cost (the billing page lags).
All evaluations: multistep-close protocol, greedy, 8 tool rounds, 512 tokens/turn; frozen dev set
unless the stage says otherwise.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token

ROOT = Path(__file__).resolve().parents[1]
MODEL = "CohereLabs/tiny-aya-global"
RUNS = "/work/runs"
EVAL = ["-m", "ayadb.smoke", "--engine", "hf", "--hf-model", MODEL, "--split", "dev",
        "--format", "multistep-close", "--temperature", "0", "--max-steps", "8"]
SFT = ["-m", "ayadb.sft", "--model", MODEL, "--episodes", "data/sft_pilot/episodes.jsonl"]
EPISODES_SHA256 = {  # preregistered episode files (as written on Windows)
    "data/sft_pilot/episodes.jsonl": "5cb48a813df0238f0cc196037d7e609249bb30698c6fc511759fecea2790eb6b",
    "data/sft_recovery/episodes.jsonl": "d0bf1bb35f551e28db0a539a9480aab07ee030791fc8a4550bd7dae4a3de0180",  # recovery arm B
}
CODE = ["src", "pyproject.toml", "scripts/hf_job.py", "data/sft_pilot", "data/sft_recovery",
        "data/gguf_chat_template.jinja"]

GPU = os.environ.get("AYADB_FLAVOR", "t4-small")  # 1x T4 16 GB, 4 vCPU, 15 GB RAM
CPU = "cpu-basic"  # 2 vCPU, 16 GB RAM
PRICE = {"cpu-basic": 0.01, "cpu-upgrade": 0.03, "t4-small": 0.40, "t4-medium": 0.60, "l4x1": 0.80}  # $/h, 2026-11
STALL_S = 1200
GPU_GUARD = ("gpu_guard", ["-c", "import torch; assert torch.cuda.is_available(), 'CUDA is not usable'; "
                           "print(torch.cuda.get_device_name(0))"], 600)
IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
SETUP = " && ".join([
    "uv pip install --system -q torch==2.11.0 torchvision==0.26.0"  # agenttune needs torchvision; unpinned it
    " --index-url https://download.pytorch.org/whl/cu126",  # comes from PyPI for another torch
    "uv pip install --system -q transformers==5.22.1 tokenizers==0.23.3 huggingface_hub==1.33.0 peft==0.21.2"
    " trl==1.7.1 accelerate==1.15.0 agenttune==1.1.3 auditkit==1.1.3 bitsandbytes==0.50.2 gguf requests",
    "cp -r /code /repo", "cd /repo", "python scripts/hf_job.py _remote",
])
SETUP_S = 900  # image pull and installs, on top of the step timeouts
ENV = {"PYTHONPATH": "/repo/src", "HF_HOME": "/tmp/hf", "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}


# ---- inside the job ------------------------------------------------------------------------------

def _push(bucket: str, final: bool = False):
    """Sync local /work/runs to the bucket; retry, and only a failed final sync fails the job."""
    for attempt in range(5):
        try:
            HfApi().sync_bucket(RUNS, f"hf://buckets/{bucket}/runs", quiet=True)
            return
        except Exception as e:  # noqa: BLE001 - network or hub error; the next step retries anyway
            print(f"sync to bucket failed ({attempt + 1}/5): {e}", flush=True)
            time.sleep(30)
    if final:
        sys.exit("could not sync results to the bucket")


def _tee(argv: list, log: Path, timeout: float):
    """Run argv in /repo, copying its output to `log` and the job log. Kill it after `timeout`s,
    or after STALL_S without output; returns the exit code, "timeout" or "stalled"."""
    p = subprocess.Popen(argv, cwd="/repo", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, errors="replace")
    start, last, why = time.time(), [time.time()], []

    def watch():
        while p.poll() is None:
            now = time.time()
            if now - start > timeout or now - last[0] > STALL_S:
                why.append("timeout" if now - start > timeout else "stalled")
                p.kill()
                return
            time.sleep(10)

    threading.Thread(target=watch, daemon=True).start()
    with open(log, "w") as f:
        for line in p.stdout:
            last[0] = time.time()
            f.write(line)
            sys.stdout.write(line)
    rc = p.wait()
    return why[0] if why else rc


def _run_steps(stage: str, steps: list, job: dict, bucket: str) -> dict:
    """Run (name, argv, timeout_s) steps in order in this container; stop at the first failure."""
    out = Path(RUNS) / stage
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "status.json"
    status = {"stage": stage, "job": job, "result": "RUNNING", "steps": {}}
    status_path.write_text(json.dumps(status, indent=2))
    _push(bucket)
    for name, argv, timeout in steps:
        t0 = time.time()
        rc = _tee([sys.executable, *argv], out / f"{name}.log", timeout)
        status["steps"][name] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
        status_path.write_text(json.dumps(status, indent=2))
        _push(bucket)
        print(f"{stage}/{name}: rc={rc} in {status['steps'][name]['seconds']}s", flush=True)
        if rc != 0:
            status["result"] = f"FAILED_AT_{name}"
            break
    else:
        status["result"] = "OK"
    status_path.write_text(json.dumps(status, indent=2))
    _push(bucket, final=True)
    return status


def _remote():
    spec = json.loads(os.environ["AYADB_SPEC"])
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        gpu = None
    job = {**spec["job"], "hf_job_id": os.environ.get("JOB_ID"), "flavor": os.environ.get("ACCELERATOR"), "gpu": gpu}
    status = _run_steps(spec["stage"], spec["steps"], job, spec["bucket"])
    sys.exit(0 if status["result"] == "OK" else 1)


# ---- local launcher ------------------------------------------------------------------------------

def _job() -> dict:
    """Code identity of this launch; refuses a dirty tree so results map to one commit."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain") and not os.environ.get("AYADB_ALLOW_DIRTY"):
        sys.exit("git tree is dirty: commit first (or set AYADB_ALLOW_DIRTY=1 for a throwaway run)")
    return {"git": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "launched": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _bucket(api: HfApi) -> str:
    bucket = f"{api.whoami()['name']}/aya-db-agent"
    api.create_bucket(bucket, private=True, exist_ok=True)
    if not api.bucket_info(bucket).private:
        sys.exit(f"bucket {bucket} is public; make it private before running anything")
    return bucket


def _upload_code(api: HfApi, bucket: str, job: dict) -> Volume:
    """Copy the code and the preregistered episodes (data/ is gitignored) to code/<sha>/ in the bucket."""
    for rel, sha in EPISODES_SHA256.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != sha:
            sys.exit(f"{rel} is not the preregistered episode set ({sha[:8]}...)")
    name = job["git"] + ("-dirty" if job["dirty"] else "")
    with tempfile.TemporaryDirectory() as d:
        for rel in CODE:
            src, dst = ROOT / rel, Path(d) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            else:
                shutil.copy2(src, dst)
        api.sync_bucket(d, f"hf://buckets/{bucket}/code/{name}", quiet=True)
    return Volume(type="bucket", source=bucket, path=f"code/{name}", mount_path="/code", read_only=True)


def _ctx() -> dict:
    job = _job()
    if not get_token():
        sys.exit("not logged in: run `hf auth login` (see the docstring)")
    api = HfApi()
    return {"api": api, "job": job, "bucket": _bucket(api)}


def _status(ctx: dict, stage: str) -> dict | None:
    with tempfile.TemporaryDirectory() as d:
        ctx["api"].download_bucket_files(ctx["bucket"], [(f"runs/{stage}/status.json", f"{d}/s.json")])
        p = Path(d) / "s.json"
        return json.loads(p.read_text()) if p.exists() else None


def _require(ctx: dict, stage: str, *steps: str):
    st = _status(ctx, stage)
    if not st or any(st["steps"].get(s, {}).get("rc") != 0 for s in steps):
        sys.exit(f"stage '{stage}' has not passed {steps}: {json.dumps(st, indent=2) if st else 'not run'}")


def _checked(ctx: dict):
    _require(ctx, "check-cpu", "kcheck")
    _require(ctx, "check", "gpu_guard", "sanity_nf4")


def _eval(tag: str, *extra: str, timeout: int = 3 * 3600) -> tuple:
    """3 h covers the worst case for 40 tasks (8 rounds x 512 tokens each); the baseline took 16-27 min."""
    return (f"eval_{tag}", [*EVAL, "--tag", tag, "--data", f"/tmp/data-{tag}",
                            "--out", f"{RUNS}/{{stage}}/runs", *extra], timeout)


def _test_eval(tag: str, *extra: str) -> tuple:
    """The sealed test split (97 tasks): same protocol as dev, 4 h for the larger set."""
    argv = ["test" if a == "dev" else a for a in EVAL]
    return (f"eval_{tag}", [*argv, "--unseal", "--tag", tag, "--data", f"/tmp/data-{tag}",
                            "--out", f"{RUNS}/sealed/runs", *extra], 4 * 3600)


def _split_eval(tag: str, split: str, *extra: str) -> tuple:
    """Recovery experiment: one evaluation on `split` (drift, dev or mh), 40 tasks, 3 h cap."""
    argv = [split if a == "dev" else a for a in EVAL]
    return (f"eval_{tag}", [*argv, "--tag", tag, "--data", f"/tmp/data-{tag}",
                            "--out", f"{RUNS}/recovery/runs", *extra], 3 * 3600)


def _fill(steps, stage):
    return [(n, [a.replace("{stage}", stage) for a in argv], t) for n, argv, t in steps]


def _launch(ctx: dict, stage: str, steps: list, volumes: tuple = (), flavor: str = GPU):
    """Start one job that runs `steps` in order; returns without waiting."""
    if flavor != CPU:
        steps = [GPU_GUARD, *steps]
    if "code" not in ctx:
        ctx["code"] = _upload_code(ctx["api"], ctx["bucket"], ctx["job"])
    spec = {"stage": stage, "steps": steps, "job": ctx["job"], "bucket": ctx["bucket"]}
    timeout = sum(t for _, _, t in steps) + SETUP_S
    info = ctx["api"].run_job(
        image=IMAGE, command=["sh", "-c", SETUP], env={**ENV, "AYADB_SPEC": json.dumps(spec)},
        secrets={"HF_TOKEN": get_token()}, flavor=flavor, timeout=timeout,
        name=f"aya-db-agent-{stage}", labels={"project": "aya-db-agent", "git": ctx["job"]["git"][:12]},
        volumes=[ctx["code"], *volumes])
    print(f"{stage}: started on {flavor} (cap {timeout / 3600:.1f} h = ${timeout / 3600 * PRICE.get(flavor, 0):.2f}) "
          f"{info.url}", flush=True)
    return info


def _wait(ctx: dict, launched: dict, others: dict | None = None) -> dict:
    """Wait for {stage: JobInfo}; return {stage: status}, from the bucket when the job wrote one.
    Ctrl+C cancels every job in `launched` and `others` (started earlier, still running), so nothing
    keeps billing unattended."""
    api, out = ctx["api"], {}
    try:
        finals = {stage: api.wait_for_job(info.id, poll_interval=60) for stage, info in launched.items()}
    except KeyboardInterrupt:
        for stage, info in {**launched, **(others or {})}.items():
            try:
                api.cancel_job(job_id=info.id)
                print(f"{stage}: cancelled {info.id}")
            except Exception as e:  # noqa: BLE001 - already finished
                print(f"{stage}: not cancelled ({e})")
        sys.exit("interrupted")
    for stage, final in finals.items():
        st = _status(ctx, stage)
        if not st or st["job"].get("hf_job_id") != final.id:  # died before its first sync (setup, quota)
            st = {"stage": stage, "result": f"NO_STATUS: job {final.status.stage} ({final.status.message})",
                  "url": launched[stage].url}
        elif st["result"] == "RUNNING":  # killed mid-step (job timeout, OOM, preemption)
            st["result"] = f"DIED: job {final.status.stage} ({final.status.message})"
        secs = (final.durations.running_secs if final.durations else None) or 0
        st["billed_estimate"] = {"flavor": final.flavor, "minutes": round(secs / 60, 1),
                                 "usd": round(secs / 3600 * PRICE.get(final.flavor, 0), 3)}
        print(json.dumps(st, indent=2), flush=True)
        out[stage] = st
    return out


def check():
    ctx = _ctx()
    st = _wait(ctx, {"check-cpu": _launch(ctx, "check-cpu", [
        # The whole stack must import before any GPU time is spent (a bad wheel fails here for cents).
        ("imports", ["-c", "import torch, torchvision, transformers, tokenizers, peft, trl, accelerate, bitsandbytes, "
                     "agenttune, auditkit; print(torch.__version__, torchvision.__version__, transformers.__version__, "
                     "peft.__version__, trl.__version__, bitsandbytes.__version__, agenttune.__version__)"], 600),
        ("kcheck", ["-m", "ayadb.kcheck", "--model", MODEL, "--out", f"{RUNS}/check-cpu/kcheck.json",
                    "--data", "/tmp/data-kcheck"], 1200),
    ], flavor=CPU)})
    if st["check-cpu"]["result"] != "OK":
        sys.exit("CPU checks failed; no GPU job started")
    st = _wait(ctx, {"check": _launch(ctx, "check", [
        ("env_info", ["-c", "import torch, transformers, peft, bitsandbytes, agenttune, tokenizers; "
                      "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "
                      "transformers.__version__, tokenizers.__version__, peft.__version__, "
                      "bitsandbytes.__version__, agenttune.__version__)"], 600),
        ("sanity_fp16", ["-m", "ayadb.sanity", "--out", f"{RUNS}/check/sanity_fp16.json"], 1800),
        ("sanity_nf4", ["-m", "ayadb.sanity", "--load-4bit", "--out", f"{RUNS}/check/sanity_nf4.json"], 1800),
    ])})
    if st["check"]["result"] != "OK":
        sys.exit(1)


def _fp16_ok(ctx: dict) -> bool:
    return _status(ctx, "check")["steps"].get("sanity_fp16", {}).get("rc") == 0


def baseline():
    ctx = _ctx()
    _checked(ctx)
    launched = {"baseline": _launch(ctx, "baseline", _fill([_eval("base-nf4", "--load-4bit")], "baseline"))}
    if _fp16_ok(ctx):
        launched["baseline-fp16"] = _launch(ctx, "baseline-fp16", _fill([_eval("base-fp16")], "baseline"))
    _wait(ctx, launched)


def smoke():
    ctx = _ctx()
    _checked(ctx)
    _wait(ctx, {"smoke": _launch(ctx, "smoke", _fill([
        ("mini_train", [*SFT, "--max-steps", "2", "--out", f"{RUNS}/smoke/mini"], 1800),
        _eval("mini", "--limit", "2", "--adapter", f"{RUNS}/smoke/mini/adapter", "--load-4bit", timeout=1800),
    ], "smoke"))})


def pilot_eval(ctx: dict | None = None):
    ctx = ctx or _ctx()
    _require(ctx, "pilot", "train")
    adapter = Volume(type="bucket", source=ctx["bucket"], path="runs/pilot/sft-pilot/adapter",
                     mount_path="/adapter", read_only=True)
    evals = [_eval("sft-nf4-r0", "--adapter", "/adapter", "--load-4bit"),
             _eval("sft-nf4-r1", "--adapter", "/adapter", "--load-4bit")]
    if _fp16_ok(ctx):
        evals.append(_eval("sft-fp16", "--adapter", "/adapter"))
    _wait(ctx, {f"pilot-{e[0]}": _launch(ctx, f"pilot-{e[0]}", _fill([e], "pilot"), (adapter,)) for e in evals})


def sealed_test():
    """Run the sealed test exactly once, as preregistered in experiments/sft-pilot/SEALED-TEST.md."""
    ctx = _ctx()
    _checked(ctx)
    _require(ctx, "pilot", "train")
    _require(ctx, "pilot-eval_sft-nf4-r0", "eval_sft-nf4-r0")
    for stage in ("sealed-base-nf4", "sealed-sft-nf4"):
        if _status(ctx, stage) and not os.environ.get("AYADB_SEALED_RERUN"):
            sys.exit(f"the sealed test already ran ({stage}); it runs once (an invalid run may be rerun once: "
                     "AYADB_SEALED_RERUN=1, reason reported in SEALED-TEST.md)")
    adapter = Volume(type="bucket", source=ctx["bucket"], path="runs/pilot/sft-pilot/adapter",
                     mount_path="/adapter", read_only=True)
    _wait(ctx, {
        "sealed-base-nf4": _launch(ctx, "sealed-base-nf4", [_test_eval("test-base-nf4", "--load-4bit")]),
        "sealed-sft-nf4": _launch(ctx, "sealed-sft-nf4", [_test_eval("test-sft-nf4", "--adapter", "/adapter",
                                                                     "--load-4bit")], (adapter,)),
    })


def _adapter(ctx: dict, path: str) -> Volume:
    return Volume(type="bucket", source=ctx["bucket"], path=path, mount_path="/adapter", read_only=True)


PILOT_ADAPTER, RECOVERY_ADAPTER = "runs/pilot/sft-pilot/adapter", "runs/recovery/sft-recovery/adapter"


def recovery_eval(ctx: dict | None = None, others: dict | None = None):
    """Arm B on drift (primary), the frozen dev set (guard) and mh (secondary)."""
    ctx = ctx or _ctx()
    _require(ctx, "recovery", "train")
    b = _adapter(ctx, RECOVERY_ADAPTER)
    evals = [_split_eval("drift-sft-b", "drift", "--adapter", "/adapter", "--load-4bit"),
             _split_eval("dev-sft-b", "dev", "--adapter", "/adapter", "--load-4bit"),
             _split_eval("mh-sft-b", "mh", "--adapter", "/adapter", "--load-4bit")]
    return _wait(ctx, {f"recovery-{e[0]}": _launch(ctx, f"recovery-{e[0]}", [e], (b,)) for e in evals}, others)


def recovery():
    """The recovery experiment, as preregistered in experiments/recovery/PREREGISTRATION.md. Arm A's
    evaluations need no training, so they run while arm B trains; arm B's start when training passed."""
    ctx = _ctx()
    _checked(ctx)
    _require(ctx, "pilot", "train")
    if _status(ctx, "recovery") and not os.environ.get("AYADB_RECOVERY_RERUN"):
        sys.exit("the recovery experiment already ran; an invalid run may be rerun once "
                 "(AYADB_RECOVERY_RERUN=1, reason reported in the preregistration)")
    a = _adapter(ctx, PILOT_ADAPTER)
    launched = {"recovery": _launch(ctx, "recovery", [("train", [
        "-m", "ayadb.sft", "--model", MODEL, "--episodes", "data/sft_recovery/episodes.jsonl",
        "--out", f"{RUNS}/recovery/sft-recovery"], 5400)])}
    for e in (_split_eval("drift-sft-a", "drift", "--adapter", "/adapter", "--load-4bit"),
              _split_eval("mh-sft-a", "mh", "--adapter", "/adapter", "--load-4bit")):
        launched[f"recovery-{e[0]}"] = _launch(ctx, f"recovery-{e[0]}", [e], (a,))
    st = _wait(ctx, {"recovery": launched.pop("recovery")}, launched)
    if st["recovery"]["result"] == "OK":
        recovery_eval(ctx, launched)
    _wait(ctx, launched)
    if st["recovery"]["result"] != "OK":
        sys.exit("arm B training failed; arm A's evaluations finished, arm B's were not started")


def pilot():
    ctx = _ctx()
    _checked(ctx)
    _require(ctx, "smoke", "mini_train", "eval_mini")
    st = _wait(ctx, {"pilot": _launch(ctx, "pilot", [("train", [*SFT, "--out", f"{RUNS}/pilot/sft-pilot"], 5400)])})
    if st["pilot"]["result"] != "OK":
        sys.exit(1)
    pilot_eval(ctx)


def status():
    api = HfApi()
    bucket = f"{api.whoami()['name']}/aya-db-agent"
    for f in api.list_bucket_tree(bucket, prefix="runs", recursive=True):
        if f.path.endswith("/status.json"):
            st = _status({"api": api, "bucket": bucket}, f.path.split("/")[1])
            print(f"{st['stage']:28} {st['result']:24} {st['job']['git'][:8]} "
                  + " ".join(f"{n}={s['rc']}" for n, s in st["steps"].items()))


def pull():
    api = HfApi()
    bucket = f"{api.whoami()['name']}/aya-db-agent"
    api.sync_bucket(f"hf://buckets/{bucket}/runs", str(ROOT / "remote" / "hf"))


if __name__ == "__main__":
    ENTRIES = {f.__name__: f for f in (check, baseline, smoke, pilot, pilot_eval, sealed_test, recovery, recovery_eval,
                                       status, pull, _remote)}
    if len(sys.argv) != 2 or sys.argv[1] not in ENTRIES:
        sys.exit(__doc__)
    ENTRIES[sys.argv[1]]()
