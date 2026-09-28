"""Request guards shared by the page's routes. The reasoning is in server.py."""

from __future__ import annotations

import secrets
from typing import Any
from urllib.parse import urlparse

from starlette.responses import JSONResponse, PlainTextResponse

# Long enough to say something to a robot, short enough not to be a payload.
MAX_MESSAGE_CHARS = 2000

# The session cookie holds the token itself, so changing the token signs every
# browser out. HttpOnly keeps it from page scripts; SameSite keeps another site's
# fetch or form POST from carrying it. Cookies are keyed on the host name, so a
# DNS-rebinding page, which reaches the robot under the attacker's name, never
# has it either. There is no TLS, so no Secure flag.
SESSION_COOKIE = "fast_body_session"
SESSION_MAX_AGE = 365 * 24 * 3600

# Short enough to type on a phone, long enough that guessing over a LAN is slow
# once the failure delay is on it.
MIN_TOKEN_CHARS = 8


def presented_token(request: Any) -> str:
    """The token a request carries: the query string first, else the cookie."""
    return str(request.query_params.get("token") or request.cookies.get(SESSION_COOKIE, ""))


def token_ok(request: Any, expected: str | None) -> bool:
    if expected is None:
        return True
    return secrets.compare_digest(presented_token(request), expected)


def query_token_ok(request: Any, expected: str | None) -> bool:
    """True only when the token arrived in the URL, which is the CLI's logged link."""
    if expected is None:
        return False
    return secrets.compare_digest(str(request.query_params.get("token", "")), expected)


def start_session(response: Any, token: str) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_MAX_AGE, httponly=True, samesite="lax", path="/")


def end_session(response: Any) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def local_path(value: Any) -> str:
    """A path on this server to send the browser to after signing in; `/` otherwise."""
    path = str(value or "")
    if path.startswith("/") and not path.startswith("//") and "\\" not in path and ":" not in path:
        return path
    return "/"


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
