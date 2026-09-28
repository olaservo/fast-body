"""The chat page as an MCP Apps host.

An MCP server can pair a tool with a small web page (MCP Apps, SEP-1865): the
tool's `_meta.ui.resourceUri` names a `ui://` resource holding a self-contained
HTML bundle, and the tool result's `structuredContent` is what that page draws.
fast-agent discovers the pairing when a server attaches
(`get_app_integration_configs`); this module puts it on the chat page:

- `observe` wraps the agent's `call_tool` so a call to a tool with a widget
  reaches every open page twice: as it starts, with the arguments, and again
  with the result. The page opens the widget on the first and feeds it the
  second, so a widget can animate while the server works. A result that is an error
  closes the widget instead: the brain usually corrects its call and tries
  again, and the retry draws its own widget. The same wrapper marks every
  tool call, widget or not, as a step on the page's activity line and as a
  flick of the antennas — activity, never content. The page is told a tool's
  name only when the card lists it under `variables.activity`; arguments and
  results never leave the robot except through a widget.
- `read_resource` serves a widget's HTML, and only for URIs fast-agent
  recognised as app resources. A server's other resources — the skill files,
  for one — are not the page's to read.
- `call_from_widget` runs a call the widget makes itself on the brain's loop,
  and hands the text of the result to the conversation as a `[table]` line, so
  the brain hears about it.
- `for_brain` decides what the model reads. MCP means `content` for the model
  and `structuredContent` for the client, but fast-agent 0.10 sends the model
  the JSON of `structuredContent` whenever there is one and drops the text
  blocks. A sheet or handout result then reaches the model as a truncated
  JSON dump led by a base64 portrait. The wrapper strips `structuredContent`
  from the brain's copy of every result that has content of its own; the page
  was already given the whole thing.

The page's half — the postMessage bridge a widget speaks — is in
static/index.html.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Opens a line the page submits on a widget's behalf.
TABLE_PREFIX = "[table]"

# The MCP Apps kind, as fast-agent's AppIntegrationKind spells it. Compared as a
# string so this module never imports fast-agent on the web thread.
_MCP_APPS = "mcp_apps"

_ORIGINAL = "_fast_body_call_tool"

_ids = itertools.count(1)


@dataclass(frozen=True)
class Widget:
    """A tool that comes with a page to draw its result."""

    server: str
    tool: str
    resource_uri: str


async def widgets(agent: Any) -> dict[tuple[str, str], Widget]:
    """The tools with a widget, keyed by (server, local tool name).

    Read fresh each time rather than cached: a server attached from the
    settings page mid-run brings its widgets with it.
    """
    try:
        configs = await agent.aggregator.get_app_integration_configs()
    except Exception as e:
        logger.debug("could not read app integrations: %s", e)
        return {}
    found: dict[tuple[str, str], Widget] = {}
    for config in configs.values():
        for tool in config.tools:
            if not tool.is_valid or str(tool.kind) != _MCP_APPS or tool.linked_resource_uri is None:
                continue
            found[(config.server_name, tool.tool_name)] = Widget(
                server=config.server_name,
                tool=tool.tool_name,
                resource_uri=str(tool.linked_resource_uri),
            )
    return found


async def widget_for(agent: Any, name: str) -> Widget | None:
    """The widget behind a tool name as the brain calls it, or None."""
    try:
        resolved = agent.aggregator.resolve_tool_name(name)
    except Exception:
        return None
    if resolved.server_name is None:
        return None
    return (await widgets(agent)).get((resolved.server_name, resolved.local_name))


def for_brain(result: Any) -> Any:
    """The result as the model should read it: its `content`, no `structuredContent`.

    A result whose only substance is structured (no content blocks at all)
    is left alone, so the model still gets something. Errors pass through.
    """
    try:
        if getattr(result, "structured_content", None) is None or not getattr(result, "content", None):
            return result
        return result.model_copy(update={"structured_content": None})
    except Exception as e:
        logger.debug("could not trim a tool result for the brain: %s", e)
        return result


def result_payload(result: Any) -> dict[str, Any]:
    """A CallToolResult as the widget expects it on the wire (camelCase, `_meta`)."""
    return dict(result.model_dump(by_alias=True, exclude_none=True, mode="json"))


def step_label(agent: Any, name: str, steps: Mapping[str, str] | None) -> str:
    """What the page may call this tool call: the card's label, or "" for an unnamed step.

    Matched on the tool's local name (the card says `tool`, a server's tool
    arrives as `server__tool`), and on the full name for a body tool.
    """
    if not steps:
        return ""
    local = name
    try:
        resolved = agent.aggregator.resolve_tool_name(name)
        if resolved.server_name is not None and resolved.local_name:
            local = resolved.local_name
    except Exception:
        pass
    return steps.get(local) or steps.get(name) or ""


def observe(
    agent: Any,
    hub: Any,
    *,
    steps: Mapping[str, str] | None = None,
    on_call: Callable[[], None] | None = None,
) -> None:
    """Wrap `agent.call_tool` so every call marks the page, and widget calls draw there.

    Same shape as the TUI hook in app.py: an instance attribute over the
    method, so the tool loop's `self.call_tool(...)` lands here. Installed
    once per agent. `steps` is the card's tool → label map; `on_call` is the
    body's tick, run as each call starts.
    """
    if getattr(agent, _ORIGINAL, None) is not None:
        return
    original = agent.call_tool
    setattr(agent, _ORIGINAL, original)

    async def call_tool(name: str, arguments: dict | None = None, tool_use_id: str | None = None, **kwargs: Any) -> Any:
        hub.post_activity("thinking", step=step_label(agent, name, steps))
        if on_call is not None:
            try:
                on_call()
            except Exception as e:
                logger.debug("tool-call hook failed: %s", e)
        widget = await widget_for(agent, name)
        if widget is None:
            return for_brain(await original(name, arguments, tool_use_id, **kwargs))
        call_id = f"app-{next(_ids)}"
        hub.post_app_call(call_id, widget=widget, arguments=arguments or {})
        try:
            result = await original(name, arguments, tool_use_id, **kwargs)
        except BaseException:
            # A timed-out turn cancels the call mid-flight; the widget should
            # stop tumbling rather than wait forever.
            hub.post_app_result(call_id, None, cancelled=True)
            raise
        if getattr(result, "is_error", False):
            # An error is an ordinary return to the tool loop, so the brain
            # reads it and retries; the retry draws its own widget, so this
            # one closes.
            hub.post_app_result(call_id, None, failed=True)
        else:
            hub.post_app_result(call_id, result_payload(result))
        return for_brain(result)

    agent.call_tool = call_tool
    logger.info("apps: tool calls are marked on the chat page; those with a widget are drawn there")


async def read_resource(brain: Any, server: str, uri: str) -> tuple[str, str] | None:
    """A widget's HTML and MIME type, or None when `uri` is not a known widget.

    The allowlist is what fast-agent listed as app resources for that server,
    so the route cannot be turned on any other resource the server holds.
    """
    agent = brain.resolve_agent(None)
    try:
        config = await agent.aggregator.get_app_integration_config(server)
    except Exception as e:
        logger.debug("no app integrations for %s: %s", server, e)
        return None
    if config is None:
        return None
    allowed = {str(r.uri): r.mime_type for r in config.resources if r.is_valid}
    if uri not in allowed:
        return None
    result = await agent.get_resource(uri, server_name=server)
    for content in result.contents:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            mime = getattr(content, "mime_type", None) or getattr(content, "mimeType", None)
            return text, str(mime or allowed[uri] or "text/html")
    return None


def _text_of(result: Any) -> str:
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)


async def call_from_widget(brain: Any, hub: Any, server: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a tool a widget asked for, and tell the conversation what it got.

    A widget may only call its own server's tools (the MCP Apps host rule),
    and the call skips the observer: the widget draws its own result in
    place, so the page must not open a second one. Raises ValueError for a
    tool the widget may not call.
    """
    from fast_agent.mcp.common import create_namespaced_name

    agent = brain.resolve_agent(None)
    aggregator = agent.aggregator
    name = create_namespaced_name(aggregator.server_display_name(server), tool)
    resolved = aggregator.resolve_tool_name(name)
    if resolved.server_name != server or resolved.local_name != tool:
        raise ValueError(f"{tool} is not a tool of {server}")

    call = getattr(agent, _ORIGINAL, None) or agent.call_tool
    result = await call(name, arguments)
    if not getattr(result, "is_error", False):
        text = _text_of(result)
        if text:
            hub.submit(f"{TABLE_PREFIX} {tool}: {text}")
    return result_payload(result)
