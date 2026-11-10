"""MCP Apps operations: widgets on the chat page, and a server's questions answered there.

Served over HTTP for the chat page and over `/rpc` for everything else; see
settings_routes.py for the pattern.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

from starlette.routing import Route

from fast_body import choices
from fast_body.web import apps
from fast_body.web.access import MAX_MESSAGE_CHARS, RequestError

if TYPE_CHECKING:
    from reachy_mini.apps.jsonrpc_server import JsonRpcServer

    from fast_body.web.server import WebChatServer

logger = logging.getLogger(__name__)


async def app_resource(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """A widget's HTML, as JSON so a browser opening the URL runs nothing.

    The page puts it in a sandboxed iframe's `srcdoc`; served as a
    document it would run with this page's origin.
    """
    server_name = str(params.get("server", "")).strip()
    uri = str(params.get("uri", "")).strip()
    if not server_name or not uri:
        raise RequestError(400, "server and uri are required")
    if server._brain is None:
        raise RequestError.not_running()
    try:
        found = await server._on_brain(apps.read_resource(server._brain, server_name, uri), timeout=20.0)
    except Exception as e:
        logger.warning("could not read widget %s from %s: %s", uri, server_name, e)
        raise RequestError(502, f"could not read the widget: {e}") from e
    if found is None:
        raise RequestError(404, "no such widget")
    html, mime_type = found
    return {"html": html, "mimeType": mime_type}


async def app_call(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """A widget's own tool call, run on the brain."""
    server_name = str(params.get("server", "")).strip()
    name = str(params.get("name", "")).strip()
    arguments = params.get("arguments") or {}
    if not server_name or not name or not isinstance(arguments, dict):
        raise RequestError(400, "server, name and an arguments object are required")
    if server._brain is None:
        raise RequestError.not_running()
    try:
        result: dict[str, Any] = await server._on_brain(
            apps.call_from_widget(server._brain, server.hub, server_name, name, arguments), timeout=60.0
        )
    except ValueError as e:
        raise RequestError(403, str(e)) from e
    except Exception as e:
        logger.warning("widget call %s on %s failed: %s", name, server_name, e)
        raise RequestError(502, f"the call failed: {e}") from e
    return result


async def app_choice(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """A tap on the page answers the question a server asked."""
    choice_id = str(params.get("id", "")).strip()
    option = str(params.get("option", "")).strip()
    elaboration = str(params.get("elaboration", "") or "")[:MAX_MESSAGE_CHARS]
    if not choice_id or not option:
        raise RequestError(400, "id and option are required")
    if server._brain is None:
        raise RequestError.not_running()

    async def answer() -> bool:
        table = choices.TABLE
        return table is not None and table.answer(choice_id, option, elaboration)

    try:
        taken = await server._on_brain(answer(), timeout=5.0)
    except Exception as e:
        raise RequestError(502, f"could not answer: {e}") from e
    if not taken:
        # Already answered from another screen or by voice, or timed out.
        raise RequestError(409, "that question is no longer open")
    return {"ok": True}


def routes(server: WebChatServer) -> list[Route]:
    return [
        Route("/apps/resource", partial(server.http_get, partial(app_resource, server))),
        Route("/apps/call", partial(server.http_post, partial(app_call, server)), methods=["POST"]),
        Route("/apps/choice", partial(server.http_post, partial(app_choice, server)), methods=["POST"]),
    ]


def register(rpc: JsonRpcServer, server: WebChatServer) -> None:
    from fast_body.web.rpc import expose

    expose(rpc, "widgets.resource", partial(app_resource, server))
    expose(rpc, "widgets.call", partial(app_call, server))
    expose(rpc, "choices.answer", partial(app_choice, server))
