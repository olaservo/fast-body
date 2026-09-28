"""MCP Apps routes: widgets on the chat page, and a server's questions answered there."""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse
from starlette.routing import Route

from fast_body import choices
from fast_body.web import apps
from fast_body.web.access import MAX_MESSAGE_CHARS, not_found, not_running, token_ok

if TYPE_CHECKING:
    from fast_body.web.server import WebChatServer

logger = logging.getLogger(__name__)


async def app_resource(server: WebChatServer, request: Any) -> Any:
    """A widget's HTML, as JSON so a browser opening the URL runs nothing.

    The page puts it in a sandboxed iframe's `srcdoc`; served as a
    document it would run with this page's origin.
    """
    if not token_ok(request, server.token):
        return not_found()
    server_name = str(request.query_params.get("server", "")).strip()
    uri = str(request.query_params.get("uri", "")).strip()
    if not server_name or not uri:
        return JSONResponse({"error": "server and uri are required"}, status_code=400)
    if server._brain is None:
        return not_running()
    try:
        found = await server._on_brain(apps.read_resource(server._brain, server_name, uri), timeout=20.0)
    except Exception as e:
        logger.warning("could not read widget %s from %s: %s", uri, server_name, e)
        return JSONResponse({"error": f"could not read the widget: {e}"}, status_code=502)
    if found is None:
        return JSONResponse({"error": "no such widget"}, status_code=404)
    html, mime_type = found
    return JSONResponse({"html": html, "mimeType": mime_type})


async def app_call(server: WebChatServer, request: Any) -> Any:
    """A widget's own tool call, run on the brain."""
    body, error = await server._json_body(request)
    if body is None:
        return error
    server_name = str(body.get("server", "")).strip()
    name = str(body.get("name", "")).strip()
    arguments = body.get("arguments") or {}
    if not server_name or not name or not isinstance(arguments, dict):
        return JSONResponse({"error": "server, name and an arguments object are required"}, status_code=400)
    if server._brain is None:
        return not_running()
    try:
        result = await server._on_brain(
            apps.call_from_widget(server._brain, server.hub, server_name, name, arguments), timeout=60.0
        )
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=403)
    except Exception as e:
        logger.warning("widget call %s on %s failed: %s", name, server_name, e)
        return JSONResponse({"error": f"the call failed: {e}"}, status_code=502)
    return JSONResponse(result)


async def app_choice(server: WebChatServer, request: Any) -> Any:
    """A tap on the page answers the question a server asked."""
    body, error = await server._json_body(request)
    if body is None:
        return error
    choice_id = str(body.get("id", "")).strip()
    option = str(body.get("option", "")).strip()
    elaboration = str(body.get("elaboration", "") or "")[:MAX_MESSAGE_CHARS]
    if not choice_id or not option:
        return JSONResponse({"error": "id and option are required"}, status_code=400)
    if server._brain is None:
        return not_running()

    async def answer() -> bool:
        table = choices.TABLE
        return table is not None and table.answer(choice_id, option, elaboration)

    try:
        taken = await server._on_brain(answer(), timeout=5.0)
    except Exception as e:
        return JSONResponse({"error": f"could not answer: {e}"}, status_code=502)
    if not taken:
        # Already answered from another screen or by voice, or timed out.
        return JSONResponse({"error": "that question is no longer open"}, status_code=409)
    return JSONResponse({"ok": True})


def routes(server: WebChatServer) -> list[Route]:
    return [
        Route("/apps/resource", partial(app_resource, server)),
        Route("/apps/call", partial(app_call, server), methods=["POST"]),
        Route("/apps/choice", partial(app_choice, server), methods=["POST"]),
    ]
