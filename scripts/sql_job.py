"""New SQL Jobs stages: offline plan, read-only preflight, separately approved upload/launch.

No call from this module starts work on import. Every bucket write requires the exact registration
SHA acknowledgement and --allow-upload; starting one job additionally requires --allow-paid-job.
The legacy hf_job.py and all old stages/results stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_hf_file_metadata, get_token, hf_hub_url
from huggingface_hub.utils import get_stored_tokens

from ayadb.promptquality import fingerprint
from ayadb.runconfig import STAGE_STEP_LIMITS, load_registration, registration_plan
from ayadb.runledger import atomic_json

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "hf-user"
SYNC_LOCK = threading.Lock()
IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
SETUP = " && ".join([
    "uv pip install --system -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126",
    "cp -r /code /repo", "cd /repo",
    "uv pip install --system -q -r requirements-sql-runner.txt",
    "PYTHONPATH=/repo/src python -u scripts/sql_job.py _remote --registration registration.json",
])


def account_token(expected):
    """The active token if it is `expected`, else the stored token that belongs to `expected`.

    Several accounts can be logged in locally (`hf auth login --force` adds one); the remote job has
    only its own HF_TOKEN. Tokens are never printed.
    """
    active = get_token()
    try:
        stored = list(get_stored_tokens().values())
    except Exception:
        stored = []
    for candidate in [active] + [t for t in stored if t != active]:
        if candidate and HfApi(token=candidate).whoami()["name"] == expected:
            return candidate
    return active


def authorized_api(expected=EXPECTED, *, api=None, token=None):
    token = token if token is not None else (getattr(api, "token", None) or account_token(expected))
    if api is None:
        if not token:
            raise PermissionError("HF login required; no writes attempted")
        api = HfApi(token=token)
    name = api.whoami()["name"]
    if name != expected:
        raise PermissionError(f"expected HF account {expected}, found {name}; no writes/jobs attempted")
    return api, token


def preflight(reg, *, api=None):
    api, token = authorized_api(reg["expected_account"], api=api)
    cfg = reg["model_config"]
    model = api.model_info(cfg["model"], revision=cfg["model_revision"])
    weight = next(f.rfilename for f in model.siblings if f.rfilename.endswith(".safetensors"))
    metadata = get_hf_file_metadata(hf_hub_url(cfg["model"], weight, revision=cfg["model_revision"]), token=token)
    if metadata.commit_hash != cfg["model_revision"]:
        raise ValueError("gated-weight metadata revision mismatch")
    hardware = next(h for h in api.list_jobs_hardware() if h.name == reg["hardware"])
    if hardware.unit_label != "minute" or hardware.unit_cost_usd > reg["max_price_per_minute_usd"] + 1e-10:
        raise ValueError("current hardware price exceeds the registered rate")
    return {"account": reg["expected_account"], "model_access_head_ok": True,
            "model_revision": metadata.commit_hash, "price_usd_per_minute": hardware.unit_cost_usd,
            "jobs_started": 0, "bucket_writes": 0}


def approval(path, acknowledged_sha, allow_upload, allow_job=False):
    digest = fingerprint(path)
    if acknowledged_sha != digest or not allow_upload:
        raise PermissionError("explicit user approval for this registration and its uploads is required")
    return digest


def clean_commit():
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise ValueError("commit the reviewed work locally before any upload/job; dirty bypass is disabled")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def private_bucket(api, expected):
    # whoami is deliberately repeated immediately before the first mutation.
    authorized_api(expected, api=api)
    bucket = expected + "/aya-db-agent"
    try:
        info = api.bucket_info(bucket)
    except Exception as error:
        if getattr(getattr(error, "response", None), "status_code", None) != 404:
            raise
        api.create_bucket(bucket, private=True, exist_ok=False)
        info = api.bucket_info(bucket)
    if not info.private:
        raise PermissionError("refusing to write to a public bucket")
    return bucket


def upload_bundle(path, reg, api, bucket, digest):
    prefix = "sql-inputs/" + reg["id"] + "/" + digest
    with tempfile.TemporaryDirectory(prefix="sql-reviewed-") as temp:
        directory = Path(temp)
        files = set(reg["source_sha256"]) | {a["path"] for a in reg["assets"]}
        for filename in sorted(files):
            src, dst = ROOT / filename, directory / filename
            if not src.resolve().is_relative_to(ROOT) or src.is_symlink():
                raise ValueError("bundle path outside workspace")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            expected = reg["source_sha256"].get(filename) or next(a["sha256"] for a in reg["assets"] if a["path"] == filename)
            if fingerprint(dst) != expected:
                raise ValueError("bundle content changed after registration: " + filename)
        shutil.copy2(path, directory / "registration.json")
        api.sync_bucket(str(directory), f"hf://buckets/{bucket}/{prefix}", quiet=True, delete=False)
    return prefix


def upload(path, reg, acknowledged_sha, *, allow_upload=False, api=None):
    digest = approval(path, acknowledged_sha, allow_upload)
    commit = clean_commit()
    api, token = authorized_api(reg["expected_account"], api=api)
    preflight(reg, api=api)
    bucket = private_bucket(api, reg["expected_account"])
    prefix = upload_bundle(path, reg, api, bucket, digest)
    record = {"registration_sha256": digest, "git": commit, "account": reg["expected_account"],
              "bucket": bucket, "prefix": prefix, "files": len(reg["assets"]),
              "data_bytes": sum(a["bytes"] for a in reg["assets"])}
    destination = ROOT / "runs/hf-sql" / reg["expected_account"] / reg["id"]
    atomic_json(destination / "upload.json", record)
    return api, token, record


def launch(path, reg, acknowledged_sha, *, allow_upload=False, allow_job=False, api=None):
    approval(path, acknowledged_sha, allow_upload)
    if not allow_job:
        raise PermissionError("explicit approval for this paid job is required")
    clean_commit()
    api, _ = authorized_api(reg["expected_account"], api=api)
    preflight(reg, api=api)
    state_dir = ROOT / "runs/hf-sql" / reg["expected_account"] / reg["id"]
    state_dir.mkdir(parents=True, exist_ok=True)
    reservation = state_dir / "launch.json"
    # An ambiguous API outcome must never auto-start a second job.
    try:
        with reservation.open("x", encoding="utf-8") as target:
            json.dump({"state": "launch_reserved", "registration_sha256": acknowledged_sha}, target)
    except FileExistsError:
        raise ValueError("this registration already has a launch record; inspect it, never auto-restart") from None
    api, token, record = upload(path, reg, acknowledged_sha, allow_upload=allow_upload, api=api)
    authorized_api(reg["expected_account"], api=api)
    atomic_json(reservation, {"state": "launch_reserved", **record})
    environment = {"PYTHONPATH": "/repo/src", "PYTHONUNBUFFERED": "1", "HF_HOME": "/tmp/hf",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "TOKENIZERS_PARALLELISM": "false",
                   "AYADB_APPROVED_REGISTRATION": record["registration_sha256"],
                   "AYADB_EXPECTED_ACCOUNT": reg["expected_account"], "AYADB_SQL_BUCKET": record["bucket"]}
    info = api.run_job(image=IMAGE, command=["sh", "-c", SETUP], env=environment,
                      secrets={"HF_TOKEN": token}, namespace=reg["expected_account"], flavor=reg["hardware"],
                      timeout=reg["timeout_seconds"], name=reg["id"],
                      labels={"project": "aya-db-agent", "run": reg["id"], "git": record["git"][:12]},
                      volumes=[Volume(type="bucket", source=record["bucket"], path=record["prefix"],
                                      mount_path="/code", read_only=True)])
    atomic_json(reservation, {"state": "launched", **record, "job_id": info.id, "url": info.url})
    return {"job_id": info.id, "url": info.url, "namespace": reg["expected_account"],
            "timeout_seconds": reg["timeout_seconds"], "budget_usd": reg["budget_usd"]}


def redact(text):
    secret = os.environ.get("HF_TOKEN")
    if secret:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "[redacted]", text)
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", text)


def status_or_pull(reg, *, pull=False):
    api, _ = authorized_api(reg["expected_account"])
    receipt = ROOT / "runs/hf-sql" / reg["expected_account"] / reg["id"] / "launch.json"
    launch_record = json.loads(receipt.read_text(encoding="utf-8"))
    if launch_record.get("state") != "launched":
        raise ValueError("no confirmed job ID; inspect the launch record, never restart automatically")
    if pull:
        destination = ROOT / "remote/hf-sql" / reg["expected_account"] / reg["id"]
        destination.mkdir(parents=True, exist_ok=True)
        api.sync_bucket(f"hf://buckets/{reg['expected_account']}/aya-db-agent/runs/sql/{reg['id']}",
                        str(destination), delete=False, quiet=True)
        return {"pulled_to": str(destination), "account": reg["expected_account"]}
    info = api.inspect_job(job_id=launch_record["job_id"], namespace=reg["expected_account"])
    return {"job_id": info.id, "state": info.status.stage, "url": launch_record["url"],
            "account": reg["expected_account"]}


def snapshot_ignore(directory, names):
    """Skip temp files and checkpoint folders without _COMPLETE. Plain files named checkpoint-* still sync."""
    skipped = []
    for name in names:
        path = Path(directory) / name
        if name.endswith(".tmp"):
            skipped.append(name)
        elif name.startswith("checkpoint-") and path.is_dir() and not (path / "_COMPLETE").exists():
            skipped.append(name)
    return skipped


def sync_checkpoint_snapshot(api, reg, out, bucket):
    with SYNC_LOCK:
        return _sync_checkpoint_snapshot(api, reg, out, bucket)


def _sync_checkpoint_snapshot(api, reg, out, bucket):
    if bucket != reg["expected_account"] + "/aya-db-agent" or not api.bucket_info(bucket).private:
        raise PermissionError("invalid/private output bucket required")
    prefix = f"runs/sql/{reg['id']}"
    if not out.resolve().is_relative_to(Path("runs/sql").resolve()) or out.name != reg["id"]:
        raise ValueError("output snapshot path outside this run")
    with tempfile.TemporaryDirectory(prefix="sql-checkpoint-") as temporary:
        snapshot = Path(temporary) / "snapshot"
        shutil.copytree(out, snapshot, ignore=snapshot_ignore)
        count = sum(p.stat().st_size for p in snapshot.rglob("*") if p.is_file())
        if count > reg["max_output_upload_bytes"]:
            raise ValueError("output snapshot exceeds the approved upload byte cap")
        # Only this new, unique run prefix is synchronized/deleted. No legacy run or bucket root.
        api.sync_bucket(str(snapshot), f"hf://buckets/{bucket}/{prefix}", delete=True, quiet=True)


def remote(path):
    from ayadb.runconfig import require_execution
    require_execution(path)
    reg = load_registration(path)
    api, _ = authorized_api(reg["expected_account"])
    bucket = os.environ["AYADB_SQL_BUCKET"]
    out = Path("runs/sql") / reg["id"]
    out.mkdir(parents=True, exist_ok=False)
    # Fail before loading any model if the paid GPU is unusable.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing a billed CPU fallback")
    from importlib.metadata import version
    import sqlite3
    versions = {name: version(name) for name in ("torch", "torchvision", "transformers", "tokenizers",
                "huggingface_hub", "peft", "trl", "accelerate", "agenttune", "auditkit", "bitsandbytes",
                "requests", "safetensors", "sqlglot")}
    atomic_json(out / "environment.json", {"versions": versions, "python": sys.version,
                "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
                "sqlite_version": sqlite3.sqlite_version, "registration_sha256": fingerprint(path)})
    from ayadb.runinputs import verify_runtime_oracles
    atomic_json(out / "oracle-precheck.json", verify_runtime_oracles(reg))
    status = {"id": reg["id"], "state": "running", "gpu": torch.cuda.get_device_name(0), "steps": {}}
    atomic_json(out / "status.json", status)
    for step in reg["steps"]:
        if step["timeout_seconds"] > STAGE_STEP_LIMITS[reg["stage"]]:
            raise ValueError("step exceeds the stage wall-time limit")
        argv = [sys.executable, "-u", "-m", "ayadb.sqlrunner", "execute", "--registration", str(path), "--step", step["name"]]
        start = time.monotonic()
        process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        last = [start]
        reason = []

        def supervise():
            synced, heartbeat = start, start
            while process.poll() is None:
                now = time.monotonic()
                if now - start > step["timeout_seconds"] or now - last[0] > reg["stall_seconds"]:
                    reason.append("step_timeout" if now - start > step["timeout_seconds"] else "stalled")
                    process.kill()
                    return
                if now - synced >= reg["checkpoint_sync_seconds"]:
                    try:
                        sync_checkpoint_snapshot(api, reg, out, bucket)
                    except Exception as error:
                        reason.append("checkpoint_sync_failed:" + type(error).__name__)
                        process.kill()
                        return
                    synced = now
                if now - heartbeat >= 30:
                    print(json.dumps({"step": step["name"], "elapsed_seconds": int(now - start)}), flush=True)
                    heartbeat = now
                time.sleep(2)

        watcher = threading.Thread(target=supervise, daemon=True)
        watcher.start()
        with (out / (step["name"] + ".log")).open("w", encoding="utf-8") as log:
            for line in process.stdout:
                last[0] = time.monotonic()
                safe = redact(line)
                log.write(safe); log.flush()
                print(safe, end="", flush=True)
        code = process.wait()
        watcher.join(timeout=10)
        status["steps"][step["name"]] = {"exit_code": code, "reason": reason,
                                         "seconds": time.monotonic() - start}
        if code or reason:
            status["state"] = "failed"
        atomic_json(out / "status.json", status)
        sync_checkpoint_snapshot(api, reg, out, bucket)
        if code or reason:
            raise RuntimeError("registered step failed; no retry or later stage started")
    status["state"] = "complete"
    atomic_json(out / "status.json", status)
    sync_checkpoint_snapshot(api, reg, out, bucket)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "preflight", "upload", "launch", "status", "pull", "_remote"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--ack-registration-sha256")
    parser.add_argument("--allow-upload", action="store_true")
    parser.add_argument("--allow-paid-job", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "plan":
        print(json.dumps(registration_plan(args.registration), indent=2)); return
    if args.action == "_remote":
        remote(args.registration); return
    if args.action in {"status", "pull"}:
        reg = load_registration(args.registration, check_files=False)
        print(json.dumps(status_or_pull(reg, pull=args.action == "pull"), indent=2)); return
    reg = load_registration(args.registration)
    if args.action == "preflight":
        result = preflight(reg)
    elif args.action == "upload":
        result = upload(args.registration, reg, args.ack_registration_sha256, allow_upload=args.allow_upload)[2]
    else:
        result = launch(args.registration, reg, args.ack_registration_sha256,
                        allow_upload=args.allow_upload, allow_job=args.allow_paid_job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
