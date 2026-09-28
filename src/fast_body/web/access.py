"""Request guards shared by the page's routes. The reasoning is in server.py."""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlparse

from starlette.responses import JSONResponse, PlainTextResponse

# Long enough to say something to a robot, short enough not to be a payload.
MAX_MESSAGE_CHARS = 2000


def token_ok(request: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    return secrets.compare_digest(str(request.query_params.get("token", "")), expected)


def origin_ok(headers: Any) -> bool:
    """Reject a browser socket opened from another site.

    A non-browser client sends no `Origin`; it still needs the token.
    """
    origin = headers.get("origin")
    if not origin:
        return True
    host = (headers.get("host") or "").split(":")[0]
    return urlparse(origin).hostname == host


def not_found() -> PlainTextResponse:
    return PlainTextResponse("Not found", status_code=404)


def not_running() -> JSONResponse:
    return JSONResponse({"error": "the conversation is not running"}, status_code=503)
