"""Offline registration validation and explicit execution gates for new SQL runs."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .promptquality import fingerprint

STAGE_STEP_LIMITS = {"g0": 7200, "train": 24 * 3600, "evaluate": 12 * 3600,
                     "sample": 24 * 3600, "final": 24 * 3600}
# Amendment 9: None keeps the original tools; the name must match ayadb.sqlfeedback.POLICY.
TOOL_FEEDBACK = {None, "column-hints-v1"}
# Amendment 13: the HF account(s) registered jobs may run on. Set this to your own HF username.
ACCOUNTS = {"hf-user"}


def safe_relative(value: str) -> Path:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or ":" in value or not p.parts:
        raise ValueError("registration paths must stay relative to the workspace")
    if p.as_posix() == "data/test.jsonl":
        raise ValueError("sealed test is prohibited")
    return p


def load_registration(path: Path, *, check_files=True):
    if Path(path).resolve() == Path("data/test.jsonl").resolve():
        raise ValueError("sealed test is prohibited")
    reg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,79}", reg["id"]):
        raise ValueError("invalid run ID")
    if reg["format"] != "sql-submit-v1" or reg["expected_account"] not in ACCOUNTS:
        raise ValueError("registration format/account is not the approved scope")
    if reg["stage"] not in {"g0", "train", "evaluate", "sample", "final"}:
        raise ValueError("unknown SQL launcher stage")
    if reg["stage"] == "g0" and (reg["budget_usd"] > 3 or reg["timeout_seconds"] > 7200):
        raise ValueError("G0 is capped at $3 and two job hours")
    if reg["model_config"]["model"] != "CohereLabs/tiny-aya-global":
        raise ValueError("only the existing Tiny Aya model is in scope")
    if reg["hardware"] not in {"t4-small", "l4x1"}:
        raise ValueError("unregistered hardware price")
    if not 0 < reg["timeout_seconds"] <= 48 * 3600 or reg["budget_usd"] <= 0:
        raise ValueError("missing bounded timeout/budget")
    rate = {"t4-small": .40, "l4x1": .80}[reg["hardware"]]
    if reg["timeout_seconds"] / 3600 * rate > reg["budget_usd"]:
        raise ValueError("compute timeout exceeds the registered budget")
    names = [s["name"] for s in reg["steps"]]
    if len(names) != len(set(names)) or not names:
        raise ValueError("step names must be unique/nonempty")
    asset_names = {a["path"] for a in reg["assets"]}
    generated = []
    for step in reg["steps"]:
        if step["action"] not in {"train", "evaluate", "check_loss"}:
            raise ValueError("unknown model action")
        if step["action"] == "check_loss" and reg["stage"] != "g0":
            raise ValueError("short native-loss comparison is a G0-only check")
        if step["action"] == "check_loss" and not (
                0 < step.get("max_short_input_tokens", 0) <= 1536 and
                0 < step.get("rtol", 0) <= 1e-3 and 0 < step.get("atol", 0) <= 1e-3):
            raise ValueError("short native-loss check needs bounded whole input/tolerances")
        if step.get("loss_policy", "full-logits") not in {"full-logits", "trained-positions-v1"}:
            raise ValueError("unknown loss projection policy")
        if step.get("inference_logits_to_keep", 1) != 1:
            raise ValueError("only last-position inference projection is registered")
        if step.get("tool_feedback") not in TOOL_FEEDBACK or (step.get("tool_feedback") and step["action"] != "evaluate"):
            raise ValueError("unknown tool feedback policy or not an evaluation step")
        if not 0 < step["timeout_seconds"] <= min(reg["timeout_seconds"], STAGE_STEP_LIMITS[reg["stage"]]):
            raise ValueError("invalid step runtime limit")
        output = safe_relative(step["out"])
        if not output.is_relative_to(Path("runs/sql") / reg["id"]):
            raise ValueError("run output must stay in its unique namespace")
        for field in ("episodes", "tasks", "selection_file", "continuation_manifest"):
            if step.get(field) and safe_relative(step[field]).as_posix() not in asset_names:
                raise ValueError("step input is not in the pinned asset list: " + field)
        for field in ("adapter", "base_adapter", "resume"):
            if not step.get(field):
                continue
            location = safe_relative(step[field])
            if not any(location.is_relative_to(g) for g in generated):
                if (location / "adapter_model.safetensors").as_posix() not in asset_names:
                    raise ValueError("external adapter/checkpoint is not a pinned asset")
        generated.append(output)
    for item in reg["assets"]:
        relative = safe_relative(item["path"])
        bird = relative.is_relative_to(Path("data/bird"))
        if bird and reg["stage"] != "final":
            raise ValueError("BIRD cannot be uploaded or consumed outside a final run")
        if check_files and (fingerprint(relative) != item["sha256"] or relative.stat().st_size != item["bytes"]):
            raise ValueError("registered asset hash/size mismatch: " + item["path"])
    if sum(a["bytes"] for a in reg["assets"]) > reg["max_input_upload_bytes"]:
        raise ValueError("asset upload exceeds its registered byte limit")
    if reg.get("required_environment") not in (None, {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}):
        raise ValueError("unsupported registered allocator environment")
    if check_files:
        for filename, expected in reg["source_sha256"].items():
            source = safe_relative(filename)
            if source.parts[0] in {"data", "runs", "remote", ".venv", ".git"}:
                raise ValueError("generated/private data must be declared as scoped assets, not source code")
            if fingerprint(source) != expected:
                raise ValueError("registered source changed: " + filename)
    return reg


def require_execution(registration: Path):
    if Path(registration).resolve() == Path("data/test.jsonl").resolve():
        raise ValueError("sealed test is prohibited")
    digest = fingerprint(registration)
    if os.environ.get("AYADB_APPROVED_REGISTRATION") != digest:
        raise PermissionError("model execution requires explicit approval for this registration SHA256")
    expected = json.loads(Path(registration).read_text(encoding="utf-8"))["expected_account"]
    if expected not in ACCOUNTS or os.environ.get("AYADB_EXPECTED_ACCOUNT") != expected:
        raise PermissionError("expected account has not been established by the approved launcher")


def registration_plan(path: Path):
    reg = load_registration(path)
    return {"id": reg["id"], "status": reg["status"], "registration_sha256": fingerprint(path),
            "stage": reg["stage"], "expected_account": reg["expected_account"], "budget_usd": reg["budget_usd"],
            "hardware": reg["hardware"], "timeout_seconds": reg["timeout_seconds"],
            "asset_files": len(reg["assets"]), "asset_bytes": sum(x["bytes"] for x in reg["assets"]),
            "steps": [{k: step[k] for k in ("name", "action", "timeout_seconds")} for step in reg["steps"]],
            "model_execution": False, "bucket_writes": False, "paid_jobs": False}
