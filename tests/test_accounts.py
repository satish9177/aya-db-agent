"""Amendment 13: a job uses its own account's token, account check and private bucket (fakes only)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayadb.promptquality import fingerprint
from ayadb.runconfig import ACCOUNTS, require_execution


@pytest.fixture
def launcher(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/sql_job.py"
    spec = importlib.util.spec_from_file_location("sql_job_accounts_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owners = {"tok-k": "other-user", "tok-s": "hf-user"}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": owners[self.token]}

        def bucket_info(self, bucket):
            return SimpleNamespace(private=True)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(module, "get_stored_tokens", lambda: {"oauth-other-user": "tok-k", "oauth-hf-user": "tok-s"})
    return module


def test_launcher_uses_the_stored_token_that_matches_the_account(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "get_token", lambda: "tok-k")
    assert launcher.account_token("other-user") == "tok-k"
    assert launcher.account_token("hf-user") == "tok-s"
    api, token = launcher.authorized_api("hf-user")
    assert token == "tok-s" and api.whoami()["name"] == "hf-user"
    monkeypatch.setattr(launcher, "get_stored_tokens", lambda: {"oauth-other-user": "tok-k"})
    with pytest.raises(PermissionError, match="expected HF account hf-user"):
        launcher.authorized_api("hf-user")


def test_remote_execution_requires_the_registrations_own_account(tmp_path, monkeypatch):
    assert ACCOUNTS == {"hf-user"}
    path = tmp_path / "registration.json"
    path.write_text(json.dumps({"expected_account": "hf-user"}), encoding="utf-8")
    monkeypatch.setenv("AYADB_APPROVED_REGISTRATION", fingerprint(path))
    monkeypatch.setenv("AYADB_EXPECTED_ACCOUNT", "hf-user")
    require_execution(path)
    monkeypatch.setenv("AYADB_EXPECTED_ACCOUNT", "other-user")
    with pytest.raises(PermissionError, match="expected account"):
        require_execution(path)
    path.write_text(json.dumps({"expected_account": "someone-else"}), encoding="utf-8")
    monkeypatch.setenv("AYADB_APPROVED_REGISTRATION", fingerprint(path))
    monkeypatch.setenv("AYADB_EXPECTED_ACCOUNT", "someone-else")
    with pytest.raises(PermissionError, match="expected account"):
        require_execution(path)


def test_output_sync_only_writes_to_the_registrations_own_bucket(launcher, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = Path("runs/sql/fixture-run")
    out.mkdir(parents=True)
    (out / "status.json").write_text("{}", encoding="utf-8")
    synced = []
    api = launcher.HfApi("tok-s")
    api.sync_bucket = lambda source, target, **k: synced.append(target)
    reg = {"id": "fixture-run", "expected_account": "hf-user", "max_output_upload_bytes": 10 ** 6}
    with pytest.raises(PermissionError, match="bucket"):
        launcher._sync_checkpoint_snapshot(api, reg, out, "other-user/aya-db-agent")
    launcher._sync_checkpoint_snapshot(api, reg, out, "hf-user/aya-db-agent")
    assert synced == ["hf://buckets/hf-user/aya-db-agent/runs/sql/fixture-run"]
