"""Companion browser page: access control, and the two-loop message handoff."""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient

from fast_body.web.hub import _PENDING_LIMIT, WebChatHub
from fast_body.web.server import MAX_MESSAGE_CHARS, WebChatServer


@pytest.fixture
def server():
    server = WebChatServer(WebChatHub(), host="127.0.0.1", port=8099, token="s3cret")
    server.login_delay = 0
    return server


@pytest.fixture
def client(server):
    return TestClient(server._build_app())


_ids = iter(range(1, 1_000_000))


def call(ws, method, **params):
    """Send one request; return (its reply, the notifications that came first)."""
    request_id = next(_ids)
    ws.send_json({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
    notes = []
    while True:
        message = ws.receive_json()
        if message.get("id") == request_id:
            return message, notes
        notes.append(message)


def result(ws, method, **params):
    reply, _ = call(ws, method, **params)
    assert "result" in reply, reply
    return reply["result"]


def without_seq(params):
    return {k: v for k, v in params.items() if k != "seq"}


# ── access control ────────────────────────────────────────────────────────
def test_page_asks_for_the_password(client):
    """The dashboard's button opens the bare URL, so that has to land somewhere useful."""
    response = client.get("/")
    assert response.status_code == 401
    assert 'action="/login"' in response.text
    assert "Stop" not in response.text  # the chat page itself is not in it


def test_wrong_token_in_the_url_asks_for_the_password(client):
    assert client.get("/?token=wrong").status_code == 401


def test_api_routes_do_not_hint_that_they_exist(client):
    # 404 rather than 401: an unauthenticated scan learns nothing.
    assert client.get("/status?token=wrong").status_code == 404


def test_page_is_served_with_the_token(client):
    response = client.get("/?token=s3cret")
    assert response.status_code == 200
    assert "fast-body" in response.text


def test_token_in_the_url_signs_the_browser_in(client):
    client.get("/?token=s3cret")
    assert client.get("/").status_code == 200
    assert client.get("/status").status_code == 200


def test_login_form_signs_the_browser_in(client):
    response = client.post(
        "/login",
        data={"token": "s3cret", "next": "/settings"},
        headers={"origin": "http://testserver"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/settings"
    assert client.get("/settings").status_code == 200
    assert client.get("/status").status_code == 200


def test_login_form_rejects_a_wrong_password(client):
    response = client.post("/login", data={"token": "nope", "next": "/"}, follow_redirects=False)
    assert response.status_code == 401
    assert "not right" in response.text
    assert client.get("/status").status_code == 404


def test_login_form_rejects_a_foreign_origin(client):
    response = client.post("/login", data={"token": "s3cret"}, headers={"origin": "http://evil.example"})
    assert response.status_code == 403
    assert client.get("/status").status_code == 404


def test_login_only_redirects_within_the_site(client):
    response = client.post("/login", data={"token": "s3cret", "next": "//evil.example/x"}, follow_redirects=False)
    assert response.headers["location"] == "/"


def test_login_form_returns_to_the_page_asked_for(client):
    assert 'value="/settings"' in client.get("/settings").text


def test_cookie_opens_the_socket(client):
    client.post("/login", data={"token": "s3cret"}, follow_redirects=False)
    with client.websocket_connect("/rpc") as ws:
        assert result(ws, "conversation.history") == {"events": []}


def test_wrong_cookie_is_refused(client):
    client.cookies.set("fast_body_session", "stale")
    assert client.get("/status").status_code == 404


def test_logout_forgets_this_browser(client):
    client.post("/login", data={"token": "s3cret"}, follow_redirects=False)
    response = client.post("/logout", json={}, headers={"origin": "http://testserver"})
    assert response.json() == {"signed_out": True}
    assert client.get("/status").status_code == 404


def test_status_reports_the_lock(client, open_client):
    assert client.get("/status?token=s3cret").json()["locked"] is True
    assert open_client.get("/status").json()["locked"] is False


def test_socket_needs_the_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/rpc"):
            pass


def test_socket_rejects_a_wrong_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/rpc?token=nope"):
            pass


def test_socket_rejects_a_foreign_origin(client):
    """A page on another site must not be able to drive the robot."""
    with pytest.raises(Exception):
        with client.websocket_connect("/rpc?token=s3cret", headers={"origin": "http://evil.example"}):
            pass


def test_socket_accepts_its_own_origin(client):
    with client.websocket_connect("/rpc?token=s3cret", headers={"origin": "http://testserver"}) as ws:
        assert result(ws, "conversation.history") == {"events": []}


def test_the_daemon_relay_needs_no_token(server):
    """The relay connects from loopback with no Origin; the daemon authenticated the far end."""
    relay = TestClient(server._build_app(), client=("127.0.0.1", 50000))
    with relay.websocket_connect("/rpc") as ws:
        assert result(ws, "conversation.history") == {"events": []}


def test_a_browser_on_loopback_still_needs_the_token(server):
    """A browser always sends Origin, so a page on this machine is not taken for the relay."""
    local = TestClient(server._build_app(), client=("127.0.0.1", 50000))
    with pytest.raises(Exception):
        with local.websocket_connect("/rpc", headers={"origin": "http://testserver"}):
            pass


def test_another_machine_without_origin_needs_the_token(client):
    with pytest.raises(Exception):
        with client.websocket_connect("/rpc"):
            pass


def test_unknown_methods_are_refused(open_client):
    with open_client.websocket_connect("/rpc") as ws:
        reply, _ = call(ws, "nope")
    assert reply["error"]["code"] == -32601


def test_the_apps_namespace_is_left_to_the_daemon(open_server):
    assert not [name for name in open_server.rpc._methods if name.startswith("apps.")]


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
    with open_client.websocket_connect("/rpc") as ws:
        assert result(ws, "conversation.history") == {"events": []}


def test_open_socket_still_rejects_a_foreign_origin(open_client):
    with pytest.raises(Exception):
        with open_client.websocket_connect("/rpc", headers={"origin": "http://evil.example"}):
            pass


def test_url_carries_the_token(server):
    assert server.url("robot.local") == "http://robot.local:8099/?token=s3cret"


def test_url_without_a_token_is_bare(open_server):
    assert open_server.url("robot.local") == "http://robot.local:8099/"


# ── message handling ──────────────────────────────────────────────────────
def test_history_is_replayed_to_a_new_client(server, client):
    server.hub.echo_user("hello")
    server.hub.echo("hi there")
    with client.websocket_connect("/rpc?token=s3cret") as ws:
        events = result(ws, "conversation.history")["events"]
    assert [(e["method"], e["params"]["role"], e["params"]["text"]) for e in events] == [
        ("conversation.transcript", "user", "hello"),
        ("conversation.transcript", "assistant", "hi there"),
    ]


def test_history_is_capped_for_the_data_channel(open_server, open_client):
    from fast_body.web.hub import HISTORY_LIMIT

    for i in range(HISTORY_LIMIT + 10):
        open_server.hub.echo(f"line {i}")
    with open_client.websocket_connect("/rpc") as ws:
        events = result(ws, "conversation.history")["events"]
    assert len(events) == HISTORY_LIMIT
    assert events[-1]["params"]["text"] == f"line {HISTORY_LIMIT + 9}"


def test_say_reaches_the_conversation_and_every_page(open_server, open_client):
    with open_client.websocket_connect("/rpc") as ws, open_client.websocket_connect("/rpc") as other:
        result(other, "conversation.history")  # connected before the line is said
        assert result(ws, "conversation.say", text="  hello robot ") == {"ok": True}
        note = other.receive_json()
    assert note["method"] == "conversation.transcript"
    assert without_seq(note["params"]) == {"role": "user", "text": "hello robot", "final": True}
    assert open_server.hub._early == ["hello robot"]  # held until the conversation loop starts


def test_say_needs_text(open_client):
    with open_client.websocket_connect("/rpc") as ws:
        reply, _ = call(ws, "conversation.say", text="  ")
    assert reply["error"]["data"]["reason"] == "invalid_params"


def test_oversized_messages_are_truncated(server, client):
    captured: list[str] = []
    server.hub.submit = captured.append  # type: ignore[method-assign]
    with client.websocket_connect("/rpc?token=s3cret") as ws:
        result(ws, "conversation.say", text="x" * (MAX_MESSAGE_CHARS + 500))
    assert len(captured[0]) == MAX_MESSAGE_CHARS


def test_interrupt_stops_the_robot_and_says_so(open_server, open_client):
    interrupted = []
    open_server.hub.request_interrupt = lambda: interrupted.append(True)  # type: ignore[method-assign]
    with open_client.websocket_connect("/rpc") as ws:
        assert result(ws, "conversation.interrupt") == {"ok": True}
        note = ws.receive_json()
    assert interrupted == [True]
    assert note == {
        "jsonrpc": "2.0",
        "method": "conversation.turn",
        "params": {"state": "listening", "reason": "interrupted"},
    }


def test_status_says_when_the_brain_is_connected(open_server, open_client):
    with open_client.websocket_connect("/rpc") as ws:
        before = result(ws, "conversation.status")
        open_server.bind_brain("the brain", None)
        after = result(ws, "conversation.status")
    assert (before["backend_connected"], after["backend_connected"]) == (False, True)
    assert before["locked"] is False
    for field in ("ready", "backend_error", "has_hf_connection", "personality", "voice", "listen_mode"):
        assert field in before


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


def test_the_talk_button_drives_the_gate_and_every_page_sees_it(open_server, open_client):
    from fast_body.listen_gate import PUSH_TO_TALK, ListenGate

    gate = ListenGate(PUSH_TO_TALK)
    open_server.hub.bind_listen_gate(gate)
    with open_client.websocket_connect("/rpc") as ws, open_client.websocket_connect("/rpc") as other:
        # The current gate state, so a late browser shows the right button.
        first = result(ws, "listen.state")
        assert first == {"mode": "push_to_talk", "enabled": False, "held": False, "latched": False}
        result(other, "listen.state")

        assert result(ws, "listen.set", held=True) == {
            "mode": "push_to_talk",
            "enabled": True,
            "held": True,
            "latched": False,
        }
        assert gate.held
        note = other.receive_json()
        assert note["method"] == "listen.state" and note["params"]["held"] is True

        assert result(ws, "listen.set", held=False)["enabled"] is False
        assert result(ws, "listen.set", toggle=True)["latched"] is True
        assert result(ws, "listen.set", latched=False)["latched"] is False


def test_the_mic_maps_onto_the_gate(open_server, open_client):
    from fast_body.listen_gate import PUSH_TO_TALK, ListenGate

    gate = ListenGate(PUSH_TO_TALK)
    open_server.hub.bind_listen_gate(gate)
    with open_client.websocket_connect("/rpc") as ws:
        assert result(ws, "conversation.mic") == {"muted": True}
        assert result(ws, "conversation.mic", muted=False) == {"muted": False}
        assert gate.latched
        assert result(ws, "conversation.mic", muted=True) == {"muted": True}
        assert not gate.enabled


def test_the_mic_cannot_be_muted_without_push_to_talk(open_client):
    with open_client.websocket_connect("/rpc") as ws:
        assert result(ws, "conversation.mic") == {"muted": False}
        reply, _ = call(ws, "conversation.mic", muted=True)
    assert reply["error"]["data"]["reason"] == "unsupported"


def test_a_tab_closing_mid_hold_releases_it(open_server, open_client):
    from fast_body.listen_gate import PUSH_TO_TALK, ListenGate

    gate = ListenGate(PUSH_TO_TALK)
    open_server.hub.bind_listen_gate(gate)
    with open_client.websocket_connect("/rpc") as ws:
        result(ws, "listen.set", held=True)
        assert gate.held
    assert not gate.held


def test_a_tab_closing_leaves_another_tabs_hold_alone(open_server, open_client):
    from fast_body.listen_gate import PUSH_TO_TALK, ListenGate

    gate = ListenGate(PUSH_TO_TALK)
    open_server.hub.bind_listen_gate(gate)
    with open_client.websocket_connect("/rpc") as holder:
        with open_client.websocket_connect("/rpc") as bystander:
            result(bystander, "listen.state")
        result(holder, "listen.set", held=True)
        assert gate.held


def test_without_a_gate_the_button_does_nothing(open_client):
    with open_client.websocket_connect("/rpc") as ws:
        assert result(ws, "listen.state") is None
        assert result(ws, "listen.set", held=True) is None
        assert result(ws, "conversation.say", text="still works") == {"ok": True}


def test_an_antenna_hold_reaches_the_page():
    from fast_body.listen_gate import PUSH_TO_TALK, ListenGate

    hub = WebChatHub()
    gate = ListenGate(PUSH_TO_TALK)
    hub.bind_listen_gate(gate)
    sent: list[tuple[str, dict]] = []
    hub.bind_notifier(lambda method, params: sent.append((method, without_seq(params))))
    gate.set_held(True)  # the antenna loop, not the page
    assert sent == [("listen.state", {"mode": "push_to_talk", "enabled": True, "held": True, "latched": False})]
    assert hub.listen_state()["held"] is True


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
        # setenv first so monkeypatch restores the original afterwards: the
        # writer sets os.environ, and a bare delenv of an absent name records nothing.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return path


def test_settings_page_asks_for_the_password(client):
    response = client.get("/settings")
    assert response.status_code == 401
    assert 'action="/login"' in response.text


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


def test_status_reports_the_listen_mode(client, env_file, monkeypatch):
    listen = client.get("/status?token=s3cret").json()["listen_mode"]
    assert listen == {"current": "always_on", "available": ["always_on", "push_to_talk"]}

    monkeypatch.setenv("LISTEN_MODE", "push_to_talk")
    assert client.get("/status?token=s3cret").json()["listen_mode"]["current"] == "push_to_talk"

    monkeypatch.setenv("LISTEN_MODE", "sometimes")  # not a mode: the page is shown the default
    assert client.get("/status?token=s3cret").json()["listen_mode"]["current"] == "always_on"


def test_listen_mode_is_saved_from_the_page(client, env_file):
    r = client.post(
        "/settings?token=s3cret",
        json={"LISTEN_MODE": "push_to_talk"},
        headers={"origin": "http://testserver"},
    )
    assert r.json()["written"] == ["LISTEN_MODE"]
    assert "LISTEN_MODE=push_to_talk" in env_file.read_text()


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
    client.post("/settings?token=s3cret", json={"OPENAI_API_KEY": "new"}, headers={"origin": "http://testserver"})
    text = env_file.read_text()
    assert "OPENAI_API_KEY=new" in text
    assert "OPENAI_API_KEY=old" not in text
    assert "OTHER=keep" in text  # unrelated lines survive


def test_saving_needs_the_token(client, env_file):
    r = client.post("/settings", json={"OPENAI_API_KEY": "sk-nope"})
    assert r.status_code == 404
    assert not env_file.exists()


def test_saving_rejects_a_foreign_origin(client, env_file):
    r = client.post(
        "/settings?token=s3cret", json={"OPENAI_API_KEY": "sk-nope"}, headers={"origin": "http://evil.example"}
    )
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
    r = client.post("/settings?token=s3cret", json={"OPENAI_API_KEY": "   "}, headers={"origin": "http://testserver"})
    assert r.json()["written"] == []
    assert "keepme" in env_file.read_text()


# ── the lock ──────────────────────────────────────────────────────────────
def test_setting_a_password_locks_the_page_and_signs_in(open_server, open_client, env_file):
    response = open_client.post("/lock", json={"token": "correct horse"}, headers={"origin": "http://testserver"})
    assert response.json() == {"locked": True}
    assert open_server.token == "correct horse"
    assert "WEB_CHAT_TOKEN=correct horse" in env_file.read_text()
    # This browser got the cookie; a fresh one is asked.
    assert open_client.get("/status").status_code == 200
    assert TestClient(open_server._build_app()).get("/status").status_code == 404


def test_short_passwords_are_refused(open_server, open_client, env_file):
    response = open_client.post("/lock", json={"token": "abc"}, headers={"origin": "http://testserver"})
    assert response.status_code == 400
    assert open_server.token is None


def test_passwords_that_would_not_survive_the_env_file_are_refused(open_server, open_client, env_file):
    for bad in ("correct # horse", '"quoted value"'):
        r = open_client.post("/lock", json={"token": bad}, headers={"origin": "http://testserver"})
        assert r.status_code == 400
    assert open_server.token is None


def test_changing_the_password_signs_other_browsers_out(server, client, env_file):
    other = TestClient(server._build_app())
    other.post("/login", data={"token": "s3cret"}, follow_redirects=False)
    assert other.get("/status").status_code == 200
    client.post("/lock?token=s3cret", json={"token": "something new"}, headers={"origin": "http://testserver"})
    assert other.get("/status").status_code == 404
    assert client.get("/status").status_code == 200


def test_removing_the_password_opens_the_page(server, client, env_file):
    env_file.write_text("OPENAI_API_KEY=keepme\nWEB_CHAT_TOKEN=s3cret\n")
    response = client.post("/lock?token=s3cret", json={"remove": True}, headers={"origin": "http://testserver"})
    assert response.json() == {"locked": False}
    assert server.token is None
    assert env_file.read_text() == "OPENAI_API_KEY=keepme\n"
    assert TestClient(server._build_app()).get("/status").status_code == 200


def test_lock_needs_the_token(client, env_file):
    assert client.post("/lock", json={"token": "hijack me"}, headers={"origin": "http://testserver"}).status_code == 404


def test_the_settings_route_does_not_take_the_token(server, client, env_file):
    r = client.post("/settings?token=s3cret", json={"WEB_CHAT_TOKEN": "x"}, headers={"origin": "http://testserver"})
    assert r.json()["written"] == []
    assert server.token == "s3cret"


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
    r = open_client.post("/memory/forget", json={"all": True}, headers={"origin": "http://evil.example"})
    assert r.status_code == 403


def test_seed_primes_the_transcript_without_a_client(server):
    """A resumed run has no browser attached yet; the page gets it all on connect."""
    server.hub.seed([{"role": "user", "text": "hello"}, {"role": "robot", "text": "hi"}])
    assert [without_seq(e) for e in server.hub.transcript()] == [
        {"role": "user", "text": "hello"},
        {"role": "robot", "text": "hi"},
    ]


def test_seeded_history_is_replayed_to_a_new_client(server, client):
    server.hub.seed([{"role": "robot", "text": "earlier"}])
    with client.websocket_connect("/rpc?token=s3cret") as ws:
        (event,) = result(ws, "conversation.history")["events"]
    assert without_seq(event["params"]) == {"role": "assistant", "text": "earlier", "final": True}


def test_activity_is_broadcast_and_not_kept():
    hub = WebChatHub()
    sent = []
    hub.bind_notifier(lambda method, params: sent.append((method, without_seq(params))))

    hub.post_activity("thinking")
    hub.post_activity("thinking", step="rolling")
    hub.post_activity("speaking")
    hub.post_activity("idle")

    assert sent == [
        ("conversation.turn", {"state": "thinking"}),
        ("conversation.activity", {"reason": "tool_call", "step": "rolling"}),
        ("conversation.turn", {"state": "speaking"}),
        ("conversation.turn", {"state": "ready"}),
    ]
    assert hub.transcript() == []  # over as soon as the reply lands; a late page is not told


def test_a_completed_entry_carries_the_seq_of_its_completion():
    """So a page that fetched history after the result can drop the result notification."""
    hub = WebChatHub()
    sent = []
    hub.bind_notifier(lambda method, params: sent.append((method, params)))
    hub._transcript.append({"role": "choice", "id": "c1", "answered": None, "how": None, "seq": 1})
    hub.post_choice_answered("c1", "study", "page")
    (entry,) = hub.history()
    assert entry["params"]["seq"] == sent[-1][1]["seq"]
    assert (entry["method"], sent[-1][0]) == ("choices.ask", "choices.answered")


def test_clearing_tells_every_page():
    hub = WebChatHub()
    sent = []
    hub.bind_notifier(lambda method, params: sent.append(method))
    hub.echo("forgotten")
    hub.clear()
    assert sent == ["conversation.transcript", "conversation.history_cleared"]
    assert hub.history() == []


# ── settings, memory and widgets over /rpc ────────────────────────────────
def test_personalities_list_in_the_conversation_apps_shape(open_client, env_file):
    with open_client.websocket_connect("/rpc") as ws:
        listed = result(ws, "personalities.list")
    assert listed["current"] == "default"
    assert "default" in listed["choices"]
    assert (listed["locked"], listed["locked_to"]) == (False, None)


def test_applying_a_personality_saves_it_for_the_next_start(open_client, env_file):
    with open_client.websocket_connect("/rpc") as ws:
        applied = result(ws, "personalities.apply", name="default", persist=False)
        reply, _ = call(ws, "personalities.apply", name="nobody")
    assert applied["restart_required"] is True
    assert "FAST_BODY_PERSONALITY=default" in env_file.read_text()
    assert reply["error"]["data"]["reason"] == "not_found"


def test_voices_are_listed_read_and_pinned(open_client, env_file):
    with open_client.websocket_connect("/rpc") as ws:
        assert "cedar" in result(ws, "voices.list")
        assert result(ws, "voices.current")["override"] is False
        assert result(ws, "voices.apply", voice="marin") == {"voice": "marin", "restart_required": True}
        assert result(ws, "voices.current") == {"voice": "marin", "override": True}
    assert "OPENAI_TTS_VOICE=marin" in env_file.read_text()


def test_servers_are_listed_while_the_brain_is_down(open_client, monkeypatch):
    from fast_body import mcp_servers

    monkeypatch.setattr(mcp_servers, "list_servers", lambda: [{"name": "dice", "enabled": True}])
    with open_client.websocket_connect("/rpc") as ws:
        listed = result(ws, "servers.list")
    assert listed["live"] is False
    assert listed["servers"] == [{"name": "dice", "enabled": True, "attached": None}]


def test_memory_is_listed_and_forgotten_over_rpc(open_client, monkeypatch):
    from fast_body import memory

    monkeypatch.setattr(memory, "list_sessions", lambda: [])
    monkeypatch.setattr(memory, "delete_session", lambda name: True)
    monkeypatch.setattr(memory, "live_session_name", lambda: "live")
    with open_client.websocket_connect("/rpc") as ws:
        assert result(ws, "memory.list")["sessions"] == []
        assert result(ws, "memory.forget", name="old")["removed"] == 1
        reply, _ = call(ws, "memory.forget")
    assert reply["error"]["code"] == -32602


def test_choices_and_widgets_say_when_the_conversation_is_down(open_client):
    with open_client.websocket_connect("/rpc") as ws:
        answer, _ = call(ws, "choices.answer", id="c1", option="study")
        widget, _ = call(ws, "widgets.call", server="dice", name="roll", arguments={})
        missing, _ = call(ws, "widgets.call", server="dice")
    assert answer["error"]["data"]["reason"] == "not_running"
    assert widget["error"]["data"]["reason"] == "not_running"
    assert missing["error"]["data"]["reason"] == "invalid_params"


def test_http_routes_keep_their_status_codes(open_client):
    r = open_client.post("/apps/call", json={"server": "dice"})
    assert (r.status_code, r.json()) == (400, {"error": "server, name and an arguments object are required"})
    assert open_client.post("/apps/choice", json={"id": "c1", "option": "x"}).status_code == 503
