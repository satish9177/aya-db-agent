"""The closed-model runner refuses to spend without approval (no network)."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/closed_baseline.py"
    spec = importlib.util.spec_from_file_location("closed_baseline_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.read_api_key = lambda *a, **k: pytest.fail("the key must not be read before approval")
    return module


def test_run_needs_the_approved_registration_sha(runner, tmp_path, monkeypatch):
    path = tmp_path / "REGISTRATION.json"
    path.write_text(json.dumps({"kind": "closed-model-baseline", "out": str(tmp_path / "out")}), encoding="utf-8")
    monkeypatch.delenv("AYADB_APPROVED_REGISTRATION", raising=False)
    with pytest.raises(PermissionError, match="approval"):
        runner.run(path, json.loads(path.read_text(encoding="utf-8")))
    monkeypatch.setenv("AYADB_APPROVED_REGISTRATION", "0" * 64)
    with pytest.raises(PermissionError, match="approval"):
        runner.run(path, json.loads(path.read_text(encoding="utf-8")))
    assert not (tmp_path / "out").exists()


def test_load_rejects_other_registration_kinds(runner, tmp_path):
    path = tmp_path / "REGISTRATION.json"
    path.write_text(json.dumps({"kind": "train"}), encoding="utf-8")
    with pytest.raises(ValueError, match="closed-model"):
        runner.load(path)
