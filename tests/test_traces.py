"""Trace upload: which conversations go up, where, and what is recorded."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace

import pytest

from fast_body import memory, traces
from fast_body.config import Config


def _info(name: str, minute: int) -> SimpleNamespace:
    return SimpleNamespace(name=name, last_activity=datetime(2026, 11, 27, 5, minute))


class _Manager:
    def __init__(self, *infos: SimpleNamespace) -> None:
        self.infos = list(infos)

    def list_sessions(self, *, include_empty: bool = True) -> list[SimpleNamespace]:
        return self.infos


@pytest.fixture
def robot(monkeypatch, tmp_path):
    """Two saved conversations and a live one, with the Hub calls recorded."""
    manager = _Manager(_info("old", 1), _info("game", 2), _info("live", 3))
    calls = SimpleNamespace(created=[], exported=[], tokens=[], fail_on=None)
    monkeypatch.setattr(traces, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory, "session_manager", lambda: manager)
    monkeypatch.setattr(memory, "live_session_name", lambda: "live")
    monkeypatch.setattr(memory, "session_cards", lambda: {"game": "tavern"})
    monkeypatch.setattr(traces, "_ensure_private", lambda kind, repo, token: calls.created.append((kind, repo)))

    def export(_manager, name, folder, token):
        calls.tokens.append(token)
        if name == calls.fail_on:
            raise RuntimeError("403 Forbidden")
        calls.exported.append((name, folder))
        return 1

    monkeypatch.setattr(traces, "_export", export)
    calls.manager = manager
    return calls


def _run(url: str, token: str = "") -> str:
    cfg = Config()
    cfg.trace_url = url
    cfg.trace_token = token
    return asyncio.run(traces.upload_pending(cfg))


@pytest.mark.parametrize(
    ("url", "parsed"),
    [
        ("hf://buckets/me/traces", ("buckets", "me/traces", "hf://buckets/me/traces")),
        ("hf://datasets/me/traces/robot/", ("datasets", "me/traces", "hf://datasets/me/traces/robot")),
        ("hf://spaces/me/traces", None),
        ("hf://buckets/me", None),
        ("https://huggingface.co/datasets/me/traces", None),
    ],
)
def test_destination_is_a_bucket_or_dataset(url, parsed):
    assert traces.parse_destination(url) == parsed


def test_off_without_a_url(robot):
    assert _run("") == "off"
    assert robot.exported == []


def test_a_url_that_is_not_hub_storage_uploads_nothing(robot):
    assert _run("hf://spaces/me/traces").startswith("off (")
    assert robot.exported == []


def test_saved_conversations_go_up_by_card_and_the_live_one_waits(robot):
    assert _run("hf://buckets/me/traces") == "uploaded 2 to hf://buckets/me/traces"
    assert robot.created == [("buckets", "me/traces")]
    assert robot.exported == [
        ("old", "hf://buckets/me/traces/unknown/"),
        ("game", "hf://buckets/me/traces/tavern/"),
    ]


def test_an_uploaded_conversation_goes_up_again_only_after_it_changes(robot):
    _run("hf://buckets/me/traces")
    robot.exported.clear()
    assert _run("hf://buckets/me/traces") == "nothing new for hf://buckets/me/traces"

    robot.manager.infos[1] = _info("game", 30)
    _run("hf://buckets/me/traces")
    assert [name for name, _ in robot.exported] == ["game"]


def test_a_failed_upload_is_retried_next_start(robot):
    robot.fail_on = "game"
    status = _run("hf://buckets/me/traces")
    assert status.startswith("uploaded 1 of 2") and "403" in status

    robot.fail_on = None
    robot.exported.clear()
    _run("hf://buckets/me/traces")
    assert [name for name, _ in robot.exported] == ["game"]


def test_the_upload_uses_its_own_token_when_one_is_set(robot):
    _run("hf://buckets/me/traces", token="write-token")
    assert set(robot.tokens) == {"write-token"}


def test_without_its_own_token_the_upload_uses_the_apps(robot):
    _run("hf://buckets/me/traces")
    assert set(robot.tokens) == {None}
