"""Companion browser page: access control, and the two-loop message handoff."""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from fast_body.web.hub import _PENDING_LIMIT, WebChatHub
from fast_body.web.server import MAX_MESSAGE_CHARS, WebChatServer


@pytest.fixture
def server():
    return WebChatServer(WebChatHub(), host="127.0.0.1", port=8099, token="s3cret")


@pytest.fixture
def client(server):
    return TestClient(server._build_app())


# ── access control ────────────────────────────────────────────────────────
def test_page_needs_the_token(client):
    assert client.get("/").status_code == 404


def test_page_does_not_hint_that_it_exists(client):
    # 404 rather than 401: an unauthenticated scan learns nothing.
    assert client.get("/?token=wrong").status_code == 404


def test_page_is_served_with_the_token(client):
    response = client.get("/?token=s3cret")
    assert response.status_code == 200
    assert "fast-body" in response.text


def test_socket_needs_the_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass


def test_socket_rejects_a_wrong_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=nope"):
            pass


def test_socket_rejects_a_foreign_origin(client):
    """A page on another site must not be able to drive the robot."""
    with pytest.raises(Exception):
        with client.websocket_connect("/ws?token=s3cret", headers={"origin": "http://evil.example"}):
            pass


def test_socket_accepts_its_own_origin(client):
    with client.websocket_connect("/ws?token=s3cret", headers={"origin": "http://testserver"}) as ws:
        assert ws.receive_json()["role"] == "history"


@pytest.fixture
def open_server():
    """No token configured — the default, so the desktop app's button can open it."""
    return WebChatServer(WebChatHub(), host="127.0.0.1", port=8099)


@pytest.fixture
def open_client(open_server):
    return TestClient(open_server._build_app())


def test_no_token_configured_serves_the_page(open_client):
    assert open_client.get("/").status_code == 200


def test_no_token_configured_accepts_the_socket(open_client):
    with open_client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["role"] == "history"


def test_open_socket_still_rejects_a_foreign_origin(open_client):
    with pytest.raises(Exception):
        with open_client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass


def test_url_carries_the_token(server):
    assert server.url("robot.local") == "http://robot.local:8099/?token=s3cret"


def test_url_without_a_token_is_bare(open_server):
    assert open_server.url("robot.local") == "http://robot.local:8099/"


# ── message handling ──────────────────────────────────────────────────────
def test_history_is_replayed_to_a_new_client(server, client):
    server.hub.echo_user("hello")
    server.hub.echo("hi there")
    with client.websocket_connect("/ws?token=s3cret") as ws:
        lines = ws.receive_json()["lines"]
    assert [line["text"] for line in lines] == ["hello", "hi there"]


def test_oversized_messages_are_truncated(server, client):
    captured: list[str] = []
    server.hub.submit = captured.append  # type: ignore[method-assign]
    with client.websocket_connect("/ws?token=s3cret") as ws:
        ws.receive_json()
        ws.send_json({"text": "x" * (MAX_MESSAGE_CHARS + 500)})
    assert len(captured[0]) == MAX_MESSAGE_CHARS


# ── hub: crossing between the server and conversation loops ───────────────
async def test_submitted_text_reaches_read_line():
    hub = WebChatHub()
    await hub.start()
    hub.submit("  drive forward  ")
    assert await asyncio.wait_for(hub.read_line(), timeout=1) == "drive forward"


async def test_blank_messages_are_ignored():
    hub = WebChatHub()
    await hub.start()
    hub.submit("   ")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(hub.read_line(), timeout=0.05)


async def test_interrupt_sets_the_event():
    hub = WebChatHub()
    await hub.start()
    hub.request_interrupt()
    await asyncio.sleep(0)  # the set is scheduled on the loop
    assert hub.interrupt_event.is_set()


async def test_submitting_before_the_conversation_starts_is_held():
    hub = WebChatHub()  # no conversation loop yet
    hub.submit("hello?")
    hub.submit("still there?")
    await hub.start()
    assert await asyncio.wait_for(hub.read_line(), timeout=1) == "hello?"
    assert await asyncio.wait_for(hub.read_line(), timeout=1) == "still there?"


async def test_messages_held_before_startup_are_bounded():
    hub = WebChatHub()
    for i in range(_PENDING_LIMIT + 10):
        hub.submit(f"m{i}")
    assert len(hub._early) == _PENDING_LIMIT


def test_transcript_is_capped():
    hub = WebChatHub()
    for i in range(400):
        hub.echo(f"line {i}")
    transcript = hub.transcript()
    assert len(transcript) == 200
    assert transcript[-1]["text"] == "line 399"


# ── backend selection ─────────────────────────────────────────────────────
def test_none_backend_requires_the_web_chat():
    from fast_body.config import Config
    from fast_body.voice.base import build_backend

    with pytest.raises(ValueError, match="needs the web chat"):
        build_backend(Config(voice_backend="none"), robot=None)


def test_none_backend_is_wrapped_with_the_hub():
    from fast_body.config import Config
    from fast_body.voice.base import build_backend
    from fast_body.voice.dual_backend import DualVoiceBackend

    backend = build_backend(Config(voice_backend="none"), robot=None, web_hub=WebChatHub())
    assert isinstance(backend, DualVoiceBackend)


def test_unknown_backend_is_rejected():
    from fast_body.config import Config
    from fast_body.voice.base import build_backend

    with pytest.raises(ValueError, match="Unknown VOICE_BACKEND"):
        build_backend(Config(voice_backend="wav2vec"), robot=None)


async def test_a_flood_of_messages_is_bounded():
    """A client holding down Enter must not grow the queue without limit."""
    from fast_body.web.hub import _PENDING_LIMIT

    hub = WebChatHub()
    await hub.start()
    for i in range(_PENDING_LIMIT + 50):
        hub.submit(f"msg {i}")
    await asyncio.sleep(0)
    assert hub._lines.qsize() == _PENDING_LIMIT


# ── settings page ─────────────────────────────────────────────────────────
@pytest.fixture
def env_file(tmp_path, monkeypatch):
    """Point the settings writer at a throwaway .env."""
    from fast_body import config as config_module

    path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "ENV_FILE", path)
    for name in config_module.SETTABLE_ENV:
        monkeypatch.delenv(name, raising=False)
    return path


def test_settings_page_needs_the_token(client):
    assert client.get("/settings").status_code == 404


def test_status_needs_the_token(client):
    assert client.get("/status").status_code == 404


def test_status_reports_keys_without_values(client, env_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    body = client.get("/status?token=s3cret").json()
    assert body["keys"]["OPENAI_API_KEY"] is True
    assert body["keys"]["ANTHROPIC_API_KEY"] is False
    assert "sk-secret-value" not in client.get("/status?token=s3cret").text


def test_status_lists_personalities(client, env_file, monkeypatch):
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "marvin")
    persona = client.get("/status?token=s3cret").json()["personality"]
    assert persona["current"] == "marvin"
    assert "default" in persona["available"]


def test_personality_is_saved_from_the_page(client, env_file):
    r = client.post(
        "/settings?token=s3cret",
        json={"FAST_BODY_PERSONALITY": "marvin"},
        headers={"origin": "http://testserver"},
    )
    assert r.json()["written"] == ["FAST_BODY_PERSONALITY"]
    assert "FAST_BODY_PERSONALITY=marvin" in env_file.read_text()


def test_status_reports_the_voice(client, env_file, monkeypatch):
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "marvin")
    voice = client.get("/status?token=s3cret").json()["voice"]
    assert (voice["current"], voice["override"]) == ("onyx", False)  # the card's voice
    assert "cedar" in voice["available"]

    monkeypatch.setenv("OPENAI_TTS_VOICE", "nova")
    voice = client.get("/status?token=s3cret").json()["voice"]
    assert (voice["current"], voice["override"]) == ("nova", True)


def test_voice_is_saved_from_the_page(client, env_file):
    r = client.post(
        "/settings?token=s3cret",
        json={"OPENAI_TTS_VOICE": "auto"},  # "auto" un-pins: back to the card's voice
        headers={"origin": "http://testserver"},
    )
    assert r.json()["written"] == ["OPENAI_TTS_VOICE"]
    assert "OPENAI_TTS_VOICE=auto" in env_file.read_text()


def test_saving_writes_the_env_file(client, env_file):
    r = client.post(
        "/settings?token=s3cret",
        json={"OPENAI_API_KEY": "sk-written"},
        headers={"origin": "http://testserver"},
    )
    assert r.status_code == 200
    assert r.json()["written"] == ["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY=sk-written" in env_file.read_text()


def test_saving_replaces_rather_than_appends(client, env_file):
    env_file.write_text("OPENAI_API_KEY=old\nOTHER=keep\n")
    client.post("/settings?token=s3cret", json={"OPENAI_API_KEY": "new"},
                headers={"origin": "http://testserver"})
    text = env_file.read_text()
    assert "OPENAI_API_KEY=new" in text
    assert "OPENAI_API_KEY=old" not in text
    assert "OTHER=keep" in text  # unrelated lines survive


def test_saving_needs_the_token(client, env_file):
    r = client.post("/settings", json={"OPENAI_API_KEY": "sk-nope"})
    assert r.status_code == 404
    assert not env_file.exists()


def test_saving_rejects_a_foreign_origin(client, env_file):
    r = client.post("/settings?token=s3cret", json={"OPENAI_API_KEY": "sk-nope"},
                    headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    assert not env_file.exists()


def test_saving_rejects_form_content_type(client, env_file):
    """A plain cross-site form POST can't set keys even with the token."""
    r = client.post("/settings?token=s3cret", data={"OPENAI_API_KEY": "sk-nope"})
    assert r.status_code == 415
    assert not env_file.exists()


def test_only_whitelisted_names_are_written(client, env_file, monkeypatch):
    monkeypatch.setenv("PATH", "/original")
    r = client.post(
        "/settings?token=s3cret",
        json={"PATH": "/evil", "OPENAI_API_KEY": "sk-ok"},
        headers={"origin": "http://testserver"},
    )
    assert r.json()["written"] == ["OPENAI_API_KEY"]
    assert "/evil" not in env_file.read_text()
    import os

    assert os.environ["PATH"] == "/original"


def test_blank_values_leave_existing_keys_alone(client, env_file):
    env_file.write_text("OPENAI_API_KEY=keepme\n")
    r = client.post("/settings?token=s3cret", json={"OPENAI_API_KEY": "   "},
                    headers={"origin": "http://testserver"})
    assert r.json()["written"] == []
    assert "keepme" in env_file.read_text()


# ── the scraped literal ───────────────────────────────────────────────────
def test_daemon_scrape_finds_the_literal():
    """The daemon regex-scrapes main.py for `custom_app_url = "..."` — the
    open button in the desktop app exists only if this pattern matches."""
    import re
    from pathlib import Path

    import fast_body.main as m

    source = Path(m.__file__).read_text(encoding="utf-8")
    # The daemon's exact pattern (local_common_venv._get_custom_app_url_from_file).
    match = re.search(r'custom_app_url\s*(?::\s*[^=]+)?\s*=\s*["\']([^"\']+)["\']', source)
    assert match is not None
    assert match.group(1) == m.custom_app_url


def test_the_literal_matches_the_default_port(monkeypatch):
    from fast_body.config import Config
    from fast_body.main import custom_app_url

    monkeypatch.delenv("WEB_CHAT_PORT", raising=False)
    assert str(Config().web_chat_port) in custom_app_url


# ── the memory page ───────────────────────────────────────────────────────
def test_memory_needs_the_token(client):
    assert client.get("/memory").status_code == 404


def test_memory_lists_what_was_written_down(open_client, monkeypatch):
    from fast_body import memory

    monkeypatch.setattr(memory, "list_sessions", lambda: [{"name": "a", "age_hours": 1.0}])
    body = open_client.get("/memory").json()
    assert body["sessions"] == [{"name": "a", "age_hours": 1.0}]
    assert body["directory"]


def test_forget_needs_a_target(open_client):
    r = open_client.post("/memory/forget", json={})
    assert r.status_code == 400


def test_forget_removes_one(open_client, monkeypatch):
    from fast_body import memory

    deleted: list[str] = []
    monkeypatch.setattr(memory, "delete_session", lambda name: deleted.append(name) or True)
    monkeypatch.setattr(memory, "list_sessions", lambda: [])
    r = open_client.post("/memory/forget", json={"name": "a"})
    assert r.json()["removed"] == 1
    assert deleted == ["a"]


def test_forget_all_removes_everything(open_client, monkeypatch):
    from fast_body import memory

    monkeypatch.setattr(memory, "forget_all", lambda: 3)
    monkeypatch.setattr(memory, "list_sessions", lambda: [])
    assert open_client.post("/memory/forget", json={"all": True}).json()["removed"] == 3


def test_forget_all_clears_the_conversation_in_progress(open_server, open_client, monkeypatch):
    from fast_body import memory

    cleared = []

    async def forget_live(brain):
        cleared.append(brain)
        return True

    async def on_brain(coro, timeout):
        return await coro

    monkeypatch.setattr(memory, "forget_all", lambda: 1)
    monkeypatch.setattr(memory, "live_session_name", lambda: "live")
    monkeypatch.setattr(memory, "forget_live", forget_live)
    monkeypatch.setattr(memory, "list_sessions", lambda: [])
    monkeypatch.setattr(open_server, "_on_brain", on_brain)
    open_server.bind_brain("the brain", None)

    body = open_client.post("/memory/forget", json={"all": True}).json()
    assert body == {"removed": 1, "cleared": True, "sessions": []}
    assert cleared == ["the brain"]


def test_forgetting_another_conversation_leaves_the_live_one_alone(open_server, open_client, monkeypatch):
    from fast_body import memory

    monkeypatch.setattr(memory, "delete_session", lambda name: True)
    monkeypatch.setattr(memory, "live_session_name", lambda: "live")
    monkeypatch.setattr(memory, "list_sessions", lambda: [])
    monkeypatch.setattr(open_server, "_on_brain", None)  # would blow up if called
    open_server.bind_brain("the brain", None)

    body = open_client.post("/memory/forget", json={"name": "old"}).json()
    assert body["cleared"] is False


def test_forget_rejects_a_foreign_origin(open_client):
    r = open_client.post(
        "/memory/forget", json={"all": True}, headers={"origin": "http://evil.example"}
    )
    assert r.status_code == 403


def test_seed_primes_the_transcript_without_a_client(server):
    """A resumed run has no browser attached yet; the page gets it all on connect."""
    server.hub.seed([{"role": "user", "text": "hello"}, {"role": "robot", "text": "hi"}])
    assert server.hub.transcript() == [
        {"role": "user", "text": "hello"},
        {"role": "robot", "text": "hi"},
    ]


def test_seeded_history_is_replayed_to_a_new_client(server, client):
    server.hub.seed([{"role": "robot", "text": "earlier"}])
    with client.websocket_connect("/ws?token=s3cret") as ws:
        assert ws.receive_json()["lines"] == [{"role": "robot", "text": "earlier"}]


def test_activity_is_broadcast_and_not_kept():
    hub = WebChatHub()
    sent = []
    hub._broadcast = sent.append  # type: ignore[method-assign]

    hub.post_activity("thinking")
    hub.post_activity("thinking", step="rolling")
    hub.post_activity("speaking")

    assert sent == [
        {"role": "activity", "state": "thinking"},
        {"role": "activity", "state": "thinking", "step": "rolling"},
        {"role": "activity", "state": "speaking"},
    ]
    assert hub.transcript() == []  # over as soon as the reply lands; a late page is not told
