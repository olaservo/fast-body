"""HF_TOKEN comes from the daemon's own login store when nothing else set it."""

from __future__ import annotations

import json

from fast_body import config


def _store(tmp_path, **fields):
    record = {
        "version": 1,
        "signed_out": False,
        "access_token": "hf_oauth_access",
        "refresh_token": "hf_oauth_refresh",
        "expires_at": 2_000_000_000,
        "lifecycle_generation": 3,
    }
    record.update(fields)
    path = tmp_path / "reachy_mini_daemon_credentials.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_the_daemon_token_is_used(tmp_path):
    assert config.daemon_hf_token(_store(tmp_path), now=1_000_000_000) == "hf_oauth_access"


def test_a_signed_out_store_gives_nothing(tmp_path):
    assert config.daemon_hf_token(_store(tmp_path, signed_out=True), now=1_000_000_000) is None


def test_an_expired_token_gives_nothing(tmp_path):
    assert config.daemon_hf_token(_store(tmp_path, expires_at=999), now=1_000) is None


def test_no_expiry_means_no_expiry(tmp_path):
    assert config.daemon_hf_token(_store(tmp_path, expires_at=None), now=1e12) == "hf_oauth_access"


def test_a_missing_or_broken_store_gives_nothing(tmp_path):
    assert config.daemon_hf_token(tmp_path / "nope.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert config.daemon_hf_token(broken) is None
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"version": 1, "signed_out": False, "access_token": ""}), encoding="utf-8")
    assert config.daemon_hf_token(empty) is None


def test_export_sets_hf_token_from_the_daemon(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(config, "DAEMON_CREDENTIALS", _store(tmp_path))
    source = config.export_hf_token_env()
    assert "daemon" in source
    assert config.os.environ["HF_TOKEN"] == "hf_oauth_access"


def test_an_existing_hf_token_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_mine")
    monkeypatch.setattr(config, "DAEMON_CREDENTIALS", _store(tmp_path))
    assert config.export_hf_token_env() == "environment"
    assert config.os.environ["HF_TOKEN"] == "hf_mine"


def test_no_store_leaves_the_environment_alone(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(config, "DAEMON_CREDENTIALS", tmp_path / "absent.json")
    assert config.export_hf_token_env().startswith("not set")
    assert "HF_TOKEN" not in config.os.environ


def test_the_store_sits_beside_the_cli_token_file():
    assert config.DAEMON_CREDENTIALS.name == "reachy_mini_daemon_credentials.json"
    assert config.DAEMON_CREDENTIALS.parent.name == "huggingface"


def test_the_boot_line_says_where_the_token_came_from():
    assert "HF_TOKEN=" in config.config_sources()
