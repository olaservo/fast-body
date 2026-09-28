"""MCP Apps on the chat page: which calls become widgets, and the two routes behind them."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent
from starlette.testclient import TestClient

from fast_body.web import apps
from fast_body.web.hub import WebChatHub
from fast_body.web.server import WebChatServer

SERVER = "dice"
D100 = "ui://dice/d100-roll.html"
SKILL = "skill://dice/guide"


def app_config():
    from fast_agent.mcp.app_integrations import (
        AppIntegrationKind,
        AppResourceConfig,
        AppServerConfig,
        AppToolConfig,
    )

    return AppServerConfig(
        server_name=SERVER,
        supports_resources=True,
        resources=[
            AppResourceConfig(uri=D100, mime_type="text/html;profile=mcp-app", kind=AppIntegrationKind.MCP_APPS)
        ],
        tools=[
            AppToolConfig(
                tool_name="roll_d100",
                namespaced_tool_name="dice__roll_d100",
                resource_uri=D100,
                linked_resource_uri=D100,
                kind=AppIntegrationKind.MCP_APPS,
            ),
            # Declared a UI resource that the server never listed: not a widget.
            AppToolConfig(
                tool_name="opposed_roll",
                namespaced_tool_name="dice__opposed_roll",
                resource_uri="ui://dice/missing.html",
                kind=AppIntegrationKind.MCP_APPS,
                warning="resource not found",
            ),
        ],
    )


class FakeAggregator:
    def __init__(self):
        self.configs = {SERVER: app_config()}

    async def get_app_integration_configs(self):
        return dict(self.configs)

    async def get_app_integration_config(self, server):
        return self.configs.get(server)

    def resolve_tool_name(self, name):
        # Mirrors the aggregator: a known server prefix splits; anything else
        # falls back to the first server with the whole name as the local one.
        if "__" in name:
            server, local = name.split("__", 1)
            if server in self.configs:
                return SimpleNamespace(server_name=server, local_name=local)
        return SimpleNamespace(server_name=SERVER, local_name=name)

    def server_display_name(self, server):
        return server


class FakeAgent:
    def __init__(self):
        self.aggregator = FakeAggregator()
        self.calls: list[tuple[str, dict | None]] = []
        self.fail = False
        self.error = False

    async def call_tool(self, name, arguments=None, tool_use_id=None, **kwargs):
        self.calls.append((name, arguments))
        if self.fail:
            raise RuntimeError("server down")
        if self.error:
            return CallToolResult(isError=True, content=[TextContent(type="text", text="cannot parse dice expression")])
        return CallToolResult(
            content=[TextContent(type="text", text=f"{name} -> 42")],
            structuredContent={"roll": 42, "passed": True},
        )

    async def get_resource(self, uri, server_name=None):
        return SimpleNamespace(
            contents=[SimpleNamespace(uri=uri, mimeType="text/html;profile=mcp-app", text="<html>widget</html>")]
        )


class FakeBrain:
    def __init__(self, agent):
        self.agent = agent

    def resolve_agent(self, name):
        return self.agent


# ── the observer ──────────────────────────────────────────────────────────
async def test_a_tool_with_a_widget_is_posted_as_call_then_result():
    agent, hub = FakeAgent(), WebChatHub()
    apps.observe(agent, hub)

    result = await agent.call_tool("dice__roll_d100", {"skill": 45})

    # The brain reads the content text; the structured payload is the page's.
    assert result.structured_content is None
    assert result.content[0].text == "dice__roll_d100 -> 42"
    (entry,) = hub.transcript()
    assert (entry["role"], entry["server"], entry["tool"], entry["uri"]) == ("app", SERVER, "roll_d100", D100)
    assert entry["arguments"] == {"skill": 45}
    # Completed in place, in the shape the widget reads on the wire.
    assert entry["result"]["structuredContent"] == {"roll": 42, "passed": True}
    assert entry["result"]["isError"] is False
    assert entry["cancelled"] is False


async def test_the_result_is_broadcast_as_it_lands():
    agent, hub = FakeAgent(), WebChatHub()
    sent = []
    hub._broadcast = sent.append  # type: ignore[method-assign]
    apps.observe(agent, hub)

    await agent.call_tool("dice__roll_d100", {"skill": 45})

    # The step mark first, then the widget opens, then its result.
    assert [m["role"] for m in sent] == ["activity", "app", "app-result"]
    assert sent[2]["id"] == sent[1]["id"]


async def test_tools_without_a_widget_pass_straight_through():
    agent, hub = FakeAgent(), WebChatHub()
    apps.observe(agent, hub)

    await agent.call_tool("dice__opposed_roll", {})  # UI declared, resource missing
    await agent.call_tool("emotion", {"name": "scared"})  # a body tool

    assert hub.transcript() == []
    assert [c[0] for c in agent.calls] == ["dice__opposed_roll", "emotion"]


async def test_a_cancelled_call_tells_the_widget_to_stop():
    agent, hub = FakeAgent(), WebChatHub()
    agent.fail = True
    apps.observe(agent, hub)

    with pytest.raises(RuntimeError):
        await agent.call_tool("dice__roll_d100", {"skill": 45})

    (entry,) = hub.transcript()
    assert (entry["result"], entry["cancelled"]) == (None, True)


async def test_an_error_result_closes_the_widget_and_still_reaches_the_brain():
    """A malformed call comes back an error and the brain retries; the retry
    draws its own widget, so the first one closes rather than showing the error."""
    agent, hub = FakeAgent(), WebChatHub()
    agent.error = True
    sent = []
    hub._broadcast = sent.append  # type: ignore[method-assign]
    apps.observe(agent, hub)

    result = await agent.call_tool("dice__roll_d100", {"skill": '"1D4"'})

    assert result.is_error  # the tool loop reads it and the brain calls again
    (entry,) = hub.transcript()
    assert (entry["result"], entry["cancelled"], entry["failed"]) == (None, False, True)
    assert sent[-1]["role"] == "app-result" and sent[-1]["failed"] is True


async def test_every_tool_call_is_a_step_on_the_activity_line():
    agent, hub = FakeAgent(), WebChatHub()
    sent = []
    hub._broadcast = sent.append  # type: ignore[method-assign]
    ticks = []
    apps.observe(agent, hub, steps={"roll_d100": "rolling"}, on_call=lambda: ticks.append(1))

    await agent.call_tool("dice__roll_d100", {"skill": 45})
    await agent.call_tool("emotion", {"name": "scared"})

    # The card named one tool; the other is a step with no name. Neither
    # message carries the arguments.
    assert [m for m in sent if m["role"] == "activity"] == [
        {"role": "activity", "state": "thinking", "step": "rolling"},
        {"role": "activity", "state": "thinking", "step": ""},
    ]
    assert len(ticks) == 2


async def test_a_failing_tick_does_not_cost_the_call():
    def boom():
        raise RuntimeError("no body")

    agent, hub = FakeAgent(), WebChatHub()
    apps.observe(agent, hub, on_call=boom)

    result = await agent.call_tool("emotion", {"name": "scared"})

    assert result.content[0].text == "emotion -> 42"


def test_step_label_matches_the_local_name():
    agent = FakeAgent()
    steps = {"roll_d100": "rolling", "emotion": "feeling"}
    assert apps.step_label(agent, "dice__roll_d100", steps) == "rolling"
    assert apps.step_label(agent, "emotion", steps) == "feeling"
    assert apps.step_label(agent, "dice__opposed_roll", steps) == ""
    assert apps.step_label(agent, "dice__roll_d100", None) == ""


async def test_the_brain_reads_content_not_structured_content_for_every_tool():
    """fast-agent would otherwise send the model json.dumps(structuredContent) and drop the text."""
    agent, hub = FakeAgent(), WebChatHub()
    apps.observe(agent, hub)

    result = await agent.call_tool("emotion", {"name": "scared"})  # no widget, still trimmed

    assert result.structured_content is None
    assert result.content[0].text == "emotion -> 42"


def test_for_brain_keeps_a_result_that_has_nothing_but_structure():
    bare = CallToolResult(content=[], structuredContent={"only": "this"})
    assert apps.for_brain(bare).structured_content == {"only": "this"}
    error = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
    assert apps.for_brain(error) is error


async def test_observe_installs_once():
    agent, hub = FakeAgent(), WebChatHub()
    apps.observe(agent, hub)
    apps.observe(agent, hub)

    await agent.call_tool("dice__roll_d100", {})

    assert len(hub.transcript()) == 1


# ── the resource ──────────────────────────────────────────────────────────
async def test_read_resource_serves_a_known_widget():
    brain = FakeBrain(FakeAgent())
    assert await apps.read_resource(brain, SERVER, D100) == ("<html>widget</html>", "text/html;profile=mcp-app")


async def test_read_resource_refuses_anything_but_a_widget():
    """The route must not become a way to read the server's other resources."""
    brain = FakeBrain(FakeAgent())
    assert await apps.read_resource(brain, SERVER, SKILL) is None
    assert await apps.read_resource(brain, "other", D100) is None


# ── a widget's own call ───────────────────────────────────────────────────
async def test_call_from_widget_runs_the_tool_and_tells_the_table():
    agent, hub = FakeAgent(), WebChatHub()
    apps.observe(agent, hub)
    submitted = []
    hub.submit = submitted.append  # type: ignore[method-assign]

    payload = await apps.call_from_widget(FakeBrain(agent), hub, SERVER, "roll_d100", {"skill": 45, "pushed": True})

    assert payload["structuredContent"] == {"roll": 42, "passed": True}
    assert agent.calls == [("dice__roll_d100", {"skill": 45, "pushed": True})]
    assert submitted == [f"{apps.TABLE_PREFIX} roll_d100: dice__roll_d100 -> 42"]
    assert hub.transcript() == []  # the widget redraws itself; no second widget


async def test_call_from_widget_is_limited_to_its_own_server():
    agent, hub = FakeAgent(), WebChatHub()
    with pytest.raises(ValueError):
        await apps.call_from_widget(FakeBrain(agent), hub, "other", "roll_d100", {})
    assert agent.calls == []


# ── the routes ────────────────────────────────────────────────────────────
@pytest.fixture
def server():
    return WebChatServer(WebChatHub(), host="127.0.0.1", port=8099, token="s3cret")


@pytest.fixture
def client(server):
    return TestClient(server._build_app())


def bind(server):
    async def on_brain(coro, timeout):
        return await coro

    server._on_brain = on_brain  # type: ignore[method-assign]
    agent = FakeAgent()
    server.bind_brain(FakeBrain(agent), None)
    return agent


def test_resource_route_needs_the_token(client):
    assert client.get(f"/apps/resource?server={SERVER}&uri={D100}").status_code == 404


def test_resource_route_waits_for_the_brain(client):
    assert client.get(f"/apps/resource?token=s3cret&server={SERVER}&uri={D100}").status_code == 503


def test_resource_route_returns_the_widget_as_json(server, client):
    """JSON, not a document: opened directly the HTML must not run on this origin."""
    bind(server)
    response = client.get(f"/apps/resource?token=s3cret&server={SERVER}&uri={D100}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {"html": "<html>widget</html>", "mimeType": "text/html;profile=mcp-app"}


def test_resource_route_404s_an_unknown_widget(server, client):
    bind(server)
    assert client.get(f"/apps/resource?token=s3cret&server={SERVER}&uri={SKILL}").status_code == 404


def test_call_route_runs_the_widgets_call(server, client):
    agent = bind(server)
    response = client.post(
        "/apps/call?token=s3cret", json={"server": SERVER, "name": "roll_d100", "arguments": {"skill": 45}}
    )
    assert response.status_code == 200
    assert response.json()["structuredContent"] == {"roll": 42, "passed": True}
    assert agent.calls == [("dice__roll_d100", {"skill": 45})]


def test_call_route_refuses_another_servers_tool(server, client):
    bind(server)
    response = client.post("/apps/call?token=s3cret", json={"server": "other", "name": "roll_d100", "arguments": {}})
    assert response.status_code == 403


def test_call_route_rejects_a_foreign_origin(server, client):
    bind(server)
    response = client.post(
        "/apps/call?token=s3cret",
        json={"server": SERVER, "name": "roll_d100", "arguments": {}},
        headers={"origin": "http://evil.example"},
    )
    assert response.status_code == 403


# ── the hub and the page ──────────────────────────────────────────────────
def test_widgets_are_replayed_finished_to_a_new_client(server, client):
    hub = server.hub
    hub.post_app_call("app-1", widget=apps.Widget(SERVER, "roll_d100", D100), arguments={"skill": 45})
    hub.post_app_result("app-1", {"content": [], "structuredContent": {"roll": 7}})
    with client.websocket_connect("/ws?token=s3cret") as ws:
        (entry,) = ws.receive_json()["lines"]
    assert entry["role"] == "app"
    assert entry["result"]["structuredContent"] == {"roll": 7}


def test_forget_all_clears_the_page(server, client, monkeypatch):
    from fast_body import memory

    monkeypatch.setattr(memory, "forget_all", lambda: 1)
    monkeypatch.setattr(memory, "live_session_name", lambda: "live")
    monkeypatch.setattr(memory, "list_sessions", lambda: [])
    server.hub.echo("a line the robot no longer remembers")

    client.post("/memory/forget?token=s3cret", json={"all": True})

    assert server.hub.transcript() == []


def test_page_speaks_the_host_bridge(client):
    page = client.get("/?token=s3cret").text
    for method in ("ui/initialize", "ui/notifications/tool-input", "ui/notifications/tool-result", "tools/call"):
        assert method in page
    assert "allow-scripts allow-forms" in page


def test_page_has_the_activity_line_and_folds_failed_widgets(client):
    page = client.get("/?token=s3cret").text
    assert 'id="activity"' in page
    assert "didn't go through" in page
