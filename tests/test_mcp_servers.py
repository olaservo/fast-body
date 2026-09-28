"""User-added MCP servers: the store, the attach policy, and the page routes."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
import yaml
from starlette.testclient import TestClient

from fast_body import mcp_servers
from fast_body.web.hub import WebChatHub
from fast_body.web.server import WebChatServer


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the store at a throwaway file, with no stale attach errors."""
    path = tmp_path / "mcp_servers.yaml"
    monkeypatch.setattr(mcp_servers, "SERVERS_FILE", path)
    monkeypatch.setattr(mcp_servers, "_errors", {})
    return path


class FakeAgent:
    def __init__(self, attached: list[str]):
        self.name = "body"
        self._attached = attached

    def list_attached_mcp_servers(self) -> list[str]:
        return list(self._attached)


class FakeBrain:
    """Records attach/detach calls the way AgentApp receives them."""

    def __init__(self, fail: str | None = None):
        self.agent = FakeAgent([])
        self.fail = fail
        self.attach_calls: list[tuple[str, str, object]] = []
        self.detach_calls: list[tuple[str, str]] = []

    def resolve_agent(self, name):
        return self.agent

    async def attach_mcp_server(self, agent_name, server_name, server_config=None, options=None):
        if self.fail == server_name:
            raise RuntimeError("connection refused")
        self.attach_calls.append((agent_name, server_name, server_config))
        self.agent._attached.append(server_name)
        return SimpleNamespace(tools_total=3, tools_added=[])

    async def detach_mcp_server(self, agent_name, server_name):
        self.detach_calls.append((agent_name, server_name))
        if server_name in self.agent._attached:
            self.agent._attached.remove(server_name)
        return SimpleNamespace(detached=True)


# ── the store ─────────────────────────────────────────────────────────────
def test_http_server_round_trips(store):
    assert mcp_servers.save_server("fetch", transport="http", url="https://x.test/mcp") is None
    (s,) = mcp_servers.list_servers()
    assert (s["name"], s["transport"], s["url"], s["enabled"]) == ("fetch", "http", "https://x.test/mcp", True)


def test_stdio_server_round_trips(store):
    assert mcp_servers.save_server("f", transport="stdio", command="uvx", args="mcp-server-fetch --x") is None
    (s,) = mcp_servers.list_servers()
    assert (s["command"], s["args"]) == ("uvx", "mcp-server-fetch --x")


def test_bearer_token_is_stored_but_never_listed(store):
    mcp_servers.save_server("api", transport="http", url="https://x.test", bearer_token="tok-secret")
    (s,) = mcp_servers.list_servers()
    assert s["has_bearer"] is True
    assert "tok-secret" not in str(mcp_servers.list_servers())
    assert "Bearer tok-secret" in store.read_text()


def test_blank_bearer_keeps_the_existing_one(store):
    mcp_servers.save_server("api", transport="http", url="https://x.test", bearer_token="tok-1")
    mcp_servers.save_server("api", transport="http", url="https://y.test", bearer_token="")
    entry = yaml.safe_load(store.read_text())["servers"]["api"]
    assert entry["headers"]["Authorization"] == "Bearer tok-1"
    assert entry["url"] == "https://y.test"


def test_hand_edited_keys_survive_a_page_edit(store):
    """env/cwd/auth aren't on the form; editing the URL must not drop them."""
    mcp_servers.save_server("api", transport="http", url="https://x.test")
    data = yaml.safe_load(store.read_text())
    data["servers"]["api"]["env"] = {"DEBUG": "1"}
    store.write_text(yaml.safe_dump(data))
    mcp_servers.save_server("api", transport="http", url="https://y.test")
    assert yaml.safe_load(store.read_text())["servers"]["api"]["env"] == {"DEBUG": "1"}


def test_switching_transport_drops_the_other_transports_fields(store):
    mcp_servers.save_server("s", transport="http", url="https://x.test")
    mcp_servers.save_server("s", transport="stdio", command="uvx", args="thing")
    entry = yaml.safe_load(store.read_text())["servers"]["s"]
    assert "url" not in entry
    assert entry["command"] == "uvx"


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(transport="http", url="ftp://x"), "http://"),
        (dict(transport="http", url=""), "http://"),
        (dict(transport="stdio", command=""), "command"),
        (dict(transport="sse", url="https://x.test"), "transport"),
    ],
)
def test_bad_shapes_are_refused(store, kwargs, fragment):
    error = mcp_servers.save_server("s", **kwargs)
    assert error and fragment in error
    assert not store.exists()


def test_bad_names_are_refused(store):
    assert mcp_servers.save_server("a b", transport="http", url="https://x.test")
    assert mcp_servers.save_server("", transport="http", url="https://x.test")


def test_home_assistant_is_an_ordinary_server(store):
    """Home Assistant is saved like any other server; the page's preset only fills the form."""
    assert (
        mcp_servers.save_server(
            "home-assistant", transport="http", url="http://homeassistant.local:8123/api/mcp", bearer_token="llat"
        )
        is None
    )
    (s,) = mcp_servers.list_servers()
    assert (s["name"], s["url"], s["has_bearer"]) == ("home-assistant", "http://homeassistant.local:8123/api/mcp", True)


def test_a_corrupt_file_reads_as_no_servers(store):
    store.write_text("servers: [not, a, mapping")
    assert mcp_servers.list_servers() == []


def test_toggle_and_delete(store):
    mcp_servers.save_server("s", transport="http", url="https://x.test")
    assert mcp_servers.set_enabled("s", False)
    assert mcp_servers.list_servers()[0]["enabled"] is False
    assert mcp_servers.delete_server("s")
    assert mcp_servers.list_servers() == []
    assert not mcp_servers.set_enabled("ghost", True)
    assert not mcp_servers.delete_server("ghost")


# ── attaching to the brain ────────────────────────────────────────────────
async def test_attach_enabled_skips_disabled_servers(store):
    mcp_servers.save_server("on", transport="http", url="https://x.test")
    mcp_servers.save_server("off", transport="http", url="https://x.test")
    mcp_servers.set_enabled("off", False)
    brain = FakeBrain()
    status = await mcp_servers.attach_enabled(brain)
    assert [c[1] for c in brain.attach_calls] == ["on"]
    assert status == "attached on"


async def test_one_failing_server_does_not_stop_the_others(store):
    mcp_servers.save_server("bad", transport="http", url="https://x.test")
    mcp_servers.save_server("good", transport="http", url="https://x.test")
    brain = FakeBrain(fail="bad")
    status = await mcp_servers.attach_enabled(brain)
    assert "attached good" in status and "failed bad" in status
    assert mcp_servers.list_servers()[0]["error"]  # "bad" sorts first


async def test_attach_passes_fast_agents_config_model(store):
    mcp_servers.save_server("api", transport="http", url="https://x.test/mcp", bearer_token="t")
    brain = FakeBrain()
    ok, message = await mcp_servers.attach(brain, "api")
    assert ok and "3 tools" in message
    ((_, _, config),) = brain.attach_calls
    assert config.url == "https://x.test/mcp"
    assert config.headers["Authorization"] == "Bearer t"


async def test_attach_reports_an_unknown_name(store):
    ok, message = await mcp_servers.attach(FakeBrain(), "ghost")
    assert not ok and "ghost" in message


# ── the page routes ───────────────────────────────────────────────────────
@pytest.fixture
def server():
    return WebChatServer(WebChatHub(), host="127.0.0.1", port=8099, token="s3cret")


@pytest.fixture
def client(server):
    return TestClient(server._build_app())


@pytest.fixture
def brain_loop(server):
    """A FakeBrain living on its own loop, bound the way app.run() binds one."""
    brain = FakeBrain()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server.bind_brain(brain, loop)
    yield brain
    server.unbind_brain()
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_mcp_routes_need_the_token(client, store):
    assert client.get("/mcp").status_code == 404
    assert client.post("/mcp/save", json={}).status_code == 404


def test_save_rejects_a_foreign_origin(client, store):
    r = client.post(
        "/mcp/save", params={"token": "s3cret"}, json={"name": "s"}, headers={"origin": "http://evil.example"}
    )
    assert r.status_code == 403


def test_save_without_a_brain_persists_for_the_next_start(client, store):
    r = client.post(
        "/mcp/save?token=s3cret",
        json={"name": "fetch", "transport": "http", "url": "https://x.test"},
        headers={"origin": "http://testserver"},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["live"] is False
    assert "applies when the app starts" in body["note"]
    assert body["servers"][0]["attached"] is None
    assert store.exists()


def test_save_reports_a_shape_error(client, store):
    r = client.post(
        "/mcp/save?token=s3cret",
        json={"name": "fetch", "transport": "http", "url": "nope"},
        headers={"origin": "http://testserver"},
    )
    assert r.status_code == 400
    assert "http://" in r.json()["error"]


def test_save_with_a_live_brain_attaches(client, store, brain_loop):
    r = client.post(
        "/mcp/save?token=s3cret",
        json={"name": "fetch", "transport": "http", "url": "https://x.test"},
        headers={"origin": "http://testserver"},
    )
    body = r.json()
    assert body["live"] is True
    assert "attached" in body["note"]
    assert [c[1] for c in brain_loop.attach_calls] == ["fetch"]
    assert body["servers"][0]["attached"] is True


def test_resaving_an_attached_server_detaches_it_first(client, store, brain_loop):
    """Re-pointing a server's URL while it is attached: fast-agent refuses to
    re-register the name with different settings, so the old one goes first."""
    for url in ("http://laptop.local:3003/mcp", "https://x.hf.space/mcp"):
        r = client.post(
            "/mcp/save?token=s3cret",
            json={"name": "notes", "transport": "http", "url": url},
            headers={"origin": "http://testserver"},
        )
    body = r.json()
    assert "attached" in body["note"]
    assert [c[1] for c in brain_loop.detach_calls] == ["notes"]
    assert [c[1] for c in brain_loop.attach_calls] == ["notes", "notes"]
    assert brain_loop.attach_calls[-1][2].url == "https://x.hf.space/mcp"
    assert body["servers"][0]["attached"] is True


def test_toggle_off_detaches(client, store, brain_loop):
    client.post(
        "/mcp/save?token=s3cret",
        json={"name": "fetch", "transport": "http", "url": "https://x.test"},
        headers={"origin": "http://testserver"},
    )
    r = client.post(
        "/mcp/toggle?token=s3cret", json={"name": "fetch", "enabled": False}, headers={"origin": "http://testserver"}
    )
    body = r.json()
    assert [c[1] for c in brain_loop.detach_calls] == ["fetch"]
    assert body["servers"][0]["enabled"] is False
    assert body["servers"][0]["attached"] is False


def test_delete_detaches_and_forgets(client, store, brain_loop):
    client.post(
        "/mcp/save?token=s3cret",
        json={"name": "fetch", "transport": "http", "url": "https://x.test"},
        headers={"origin": "http://testserver"},
    )
    r = client.post("/mcp/delete?token=s3cret", json={"name": "fetch"}, headers={"origin": "http://testserver"})
    assert r.json()["servers"] == []
    assert [c[1] for c in brain_loop.detach_calls] == ["fetch"]


def test_builtin_servers_are_listed_apart(client, store, brain_loop):
    brain_loop.agent._attached.append("packaged-fetch")
    body = client.get("/mcp?token=s3cret").json()
    assert body["builtin_attached"] == ["packaged-fetch"]
    assert body["servers"] == []


# ── what a stored entry becomes for fast-agent ────────────────────────────
def test_user_servers_default_to_no_elicitation():
    """A form prompt has nobody to answer it on a headless robot; the turn would hang."""
    cfg = mcp_servers._server_config({"transport": "http", "url": "https://x.test/mcp", "enabled": True})
    assert cfg.elicitation is not None and cfg.elicitation.mode == "none"


def test_user_http_servers_do_not_escalate_to_oauth():
    """A 401 would otherwise start a browser flow the robot cannot finish."""
    cfg = mcp_servers._server_config({"transport": "http", "url": "https://x.test/mcp", "enabled": True})
    assert cfg.auth is not None and cfg.auth.oauth is False
    stdio = mcp_servers._server_config({"transport": "stdio", "command": "uvx", "enabled": True})
    assert stdio.auth is None


def test_a_hand_written_elicitation_mode_wins():
    cfg = mcp_servers._server_config(
        {"transport": "http", "url": "https://x.test/mcp", "elicitation": {"mode": "auto-cancel"}}
    )
    assert cfg.elicitation.mode == "auto-cancel"


# ── waking sleeping servers, and retrying the ones that failed at startup ─
class FakeHttpx:
    """Stands in for httpx.AsyncClient: records requests, fails the URLs told to."""

    def __init__(self, fail: set[str] | None = None):
        self.requests: list[tuple[str, dict]] = []
        self.fail = fail or set()

    def AsyncClient(self, **kwargs):  # noqa: N802 - mirrors httpx
        outer = self

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url, headers=None):
                outer.requests.append((url, dict(headers or {})))
                if url in outer.fail:
                    raise ConnectionError("asleep for good")
                return SimpleNamespace(status_code=400)

        return Client()


@pytest.fixture(autouse=True)
def fake_httpx(monkeypatch):
    """Every test: warm-up requests go to this fake, never to the network."""
    import sys

    fake = FakeHttpx()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return fake


def test_warm_up_requests_every_http_server_with_its_bearer(store, fake_httpx):
    mcp_servers.save_server("remote", transport="http", url="https://k.hf.space/mcp", bearer_token="hf_x")
    mcp_servers.save_server("local", transport="stdio", command="uvx", args="mcp-server-fetch")
    mcp_servers.save_server("off", transport="http", url="https://off.test/mcp")
    mcp_servers.set_enabled("off", False)
    assert asyncio.run(mcp_servers.warm_up()) == "1 up"
    assert fake_httpx.requests == [("https://k.hf.space/mcp", {"Authorization": "Bearer hf_x"})]


def test_warm_up_reports_an_unreachable_server_without_raising(store, fake_httpx):
    mcp_servers.save_server("a", transport="http", url="https://a.test/mcp")
    mcp_servers.save_server("b", transport="http", url="https://b.test/mcp")
    fake_httpx.fail.add("https://b.test/mcp")
    assert asyncio.run(mcp_servers.warm_up()) == "b unreachable, 1 up"


def test_warm_up_names_a_server_that_took_a_while(store, fake_httpx, monkeypatch):
    mcp_servers.save_server("remote", transport="http", url="https://k.hf.space/mcp")
    ticks = iter([0.0, 23.0])
    monkeypatch.setattr(mcp_servers, "_clock", lambda: next(ticks))
    assert asyncio.run(mcp_servers.warm_up()) == "remote woke in 23 s"


def test_startup_warms_before_attaching_and_keeps_the_failures(store, monkeypatch):
    mcp_servers.save_server("a", transport="http", url="https://a.test/mcp")
    mcp_servers.save_server("b", transport="http", url="https://b.test/mcp")
    order: list[str] = []

    async def fake_warm(names=None):
        order.append("warm " + ",".join(names or []))
        return "b woke in 12 s, 1 up"

    monkeypatch.setattr(mcp_servers, "warm_up", fake_warm)
    brain = FakeBrain(fail="b")
    assert asyncio.run(mcp_servers.attach_enabled(brain)) == "attached a; failed b; b woke in 12 s, 1 up"
    assert order == ["warm a,b"]
    assert mcp_servers.startup_failures() == ["b"]


def test_retry_attaches_later_and_syncs_skills_once(store, monkeypatch):
    mcp_servers.save_server("b", transport="http", url="https://b.test/mcp")
    monkeypatch.setattr(mcp_servers, "RETRY_DELAYS_S", (0.0, 0.0, 0.0))
    warmed: list[list[str]] = []

    async def fake_warm(names=None):
        warmed.append(list(names or []))
        return ""

    monkeypatch.setattr(mcp_servers, "warm_up", fake_warm)
    brain = FakeBrain(fail="b")
    synced = 0

    async def after():
        nonlocal synced
        synced += 1
        brain.fail = None  # the second round finds it awake

    async def run():
        # The first round still fails; the server comes up before the second.
        task = asyncio.ensure_future(mcp_servers.retry_failed(brain, ["b"], after=after))
        await asyncio.sleep(0)
        brain.fail = None
        await task

    asyncio.run(run())
    assert [c[1] for c in brain.attach_calls] == ["b"]
    assert synced == 1
    assert warmed and warmed[0] == ["b"]


def test_retry_gives_up_after_the_last_delay(store, monkeypatch, caplog):
    mcp_servers.save_server("b", transport="http", url="https://b.test/mcp")
    monkeypatch.setattr(mcp_servers, "RETRY_DELAYS_S", (0.0, 0.0))

    async def fake_warm(names=None):
        return ""

    monkeypatch.setattr(mcp_servers, "warm_up", fake_warm)
    brain = FakeBrain(fail="b")
    with caplog.at_level("WARNING"):
        asyncio.run(mcp_servers.retry_failed(brain, ["b"]))
    assert brain.attach_calls == []
    assert "gave up on b" in caplog.text
