"""HTTP + WebSocket server for the companion page.

We run our own uvicorn rather than the one `ReachyMiniApp` starts for
`custom_app_url`, so the page works the same from the CLI as under the daemon.
`FastBodyApp` sets `dont_start_webserver` to keep the two from fighting over the
port.

Access control and the reasoning behind it are in the README's Companion page
section. There is no TLS.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route, WebSocketRoute

from fast_body import memory
from fast_body.config import (
    AVAILABLE_TTS_VOICES,
    MEMORY_DIR,
    PACKAGE_DIR,
    Config,
    env_status,
    persist_env,
)
from fast_body.personality import list_personalities
from fast_body.web import app_routes, settings_routes
from fast_body.web.access import MAX_MESSAGE_CHARS, not_found, origin_ok, token_ok
from fast_body.web.hub import WebChatHub

__all__ = ["MAX_MESSAGE_CHARS", "WebChatServer"]

logger = logging.getLogger(__name__)

_STATIC = PACKAGE_DIR / "static"
_PAGE = _STATIC / "index.html"
_SETTINGS_PAGE = _STATIC / "settings.html"

# Starlette's code for "policy violation".
_WS_POLICY_VIOLATION = 1008


def _lan_address() -> str:
    """The address other machines can reach this one on.

    Bound to 0.0.0.0 we don't know which interface a person will use, and logging
    `localhost` is useless on a robot — that is the robot talking about itself.
    Opening a UDP socket to a routable address makes the OS pick the outbound
    interface; nothing is sent.
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1, guaranteed unrouted
            return str(s.getsockname()[0])
    except OSError:
        return "localhost"


class WebChatServer:
    """Serves the page and the socket the page talks over."""

    def __init__(self, hub: WebChatHub, *, host: str, port: int, token: str | None = None) -> None:
        self.hub = hub
        self.host = host
        self.port = port
        # None means open — see the README for why that's the default.
        self.token = (token or "").strip() or None
        # False until the conversation loop is actually running. The pages use it
        # to tell "still being configured" apart from "broken".
        self.ready = False

        self._server: Any = None
        self._thread: threading.Thread | None = None

        # The running brain and the loop it lives on, so routes can attach and
        # detach MCP servers live. Bound by the app once `fast.run()` is open;
        # None outside that window, when changes only persist to the file.
        self._brain: Any = None
        self._brain_loop: Any = None

    def bind_brain(self, brain: Any, loop: Any) -> None:
        self._brain, self._brain_loop = brain, loop

    def unbind_brain(self) -> None:
        self._brain = self._brain_loop = None

    async def _on_brain(self, coro: Any, timeout: float) -> Any:
        """Run a coroutine on the brain's loop from a route on the server loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._brain_loop)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout)
        except TimeoutError:
            future.cancel()
            raise

    def url(self, host: str | None = None) -> str:
        """The address to open, token included when one is configured."""
        shown = host or (_lan_address() if self.host in ("0.0.0.0", "::") else self.host)
        base = f"http://{shown}:{self.port}/"
        return f"{base}?token={self.token}" if self.token else base

    async def _json_body(self, request: Any) -> tuple[dict | None, Any]:
        """The guarded JSON object of a POST, or (None, error response).

        Requiring JSON keeps a plain cross-site form POST from reaching a route,
        on top of the token and the origin check.
        """
        if not token_ok(request, self.token):
            return None, not_found()
        if not request.headers.get("content-type", "").startswith("application/json"):
            return None, JSONResponse({"error": "expected application/json"}, status_code=415)
        if not origin_ok(request.headers):
            return None, JSONResponse({"error": "bad origin"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            return None, JSONResponse({"error": "malformed body"}, status_code=400)
        if not isinstance(body, dict):
            return None, JSONResponse({"error": "expected an object"}, status_code=400)
        return body, None

    # ── routes ───────────────────────────────────────────────────────────
    def _serve(self, path: Any) -> Any:
        try:
            return HTMLResponse(path.read_text(encoding="utf-8"))
        except OSError as e:
            logger.error("page missing at %s: %s", path, e)
            return PlainTextResponse("Page unavailable", status_code=500)

    async def _page(self, request: Any) -> Any:
        if not token_ok(request, self.token):
            return not_found()
        return self._serve(_PAGE)

    async def _settings_page(self, request: Any) -> Any:
        if not token_ok(request, self.token):
            return not_found()
        return self._serve(_SETTINGS_PAGE)

    async def _status(self, request: Any) -> Any:
        """What is configured, and what is stopping the conversation starting."""
        if not token_ok(request, self.token):
            return not_found()
        cfg = Config()
        return JSONResponse(
            {
                "keys": env_status(),  # names to booleans; never the values
                "errors": cfg.validate(),
                "ready": self.ready,
                # Not secret, unlike the keys — the page shows and sets these.
                "personality": {"current": cfg.personality, "available": list_personalities()},
                "voice": {
                    "current": cfg.resolve_tts_voice(),
                    "override": cfg.tts_voice_overridden(),
                    "available": list(AVAILABLE_TTS_VOICES),
                },
                "face_tracking": {
                    "current": "on" if cfg.face_tracking else "off",
                    "available": ["off", "on"],
                },
                "memory": {
                    "current": "on" if cfg.enable_memory else "off",
                    "available": ["on", "off"],
                },
            }
        )

    async def _save_settings(self, request: Any) -> Any:
        body, error = await self._json_body(request)
        if body is None:
            return error
        written = persist_env({k: str(v) for k, v in body.items()})
        errors = Config().validate()
        return JSONResponse({"written": written, "keys": env_status(), "errors": errors})

    async def _memory(self, request: Any) -> Any:
        """What the robot has written down. Metadata only, never the transcript."""
        if not token_ok(request, self.token):
            return not_found()
        cfg = Config()
        return JSONResponse(
            {
                "enabled": cfg.enable_memory,
                "directory": str(MEMORY_DIR),
                "resume_hours": cfg.memory_resume_h,
                "sessions": memory.list_sessions(),
            }
        )

    async def _forget(self, request: Any) -> Any:
        """Delete one saved conversation, or all of them."""
        body, error = await self._json_body(request)
        if body is None:
            return error
        live = memory.live_session_name()
        if body.get("all") is True:
            removed = memory.forget_all()
            forgetting_live = True
        else:
            name = str(body.get("name", "")).strip()
            if not name:
                return JSONResponse({"error": "name or all is required"}, status_code=400)
            removed = 1 if memory.delete_session(name) else 0
            forgetting_live = name == live
        # The file is gone; the brain still holds the conversation until told.
        cleared = False
        if forgetting_live and self._brain is not None:
            try:
                cleared = await self._on_brain(memory.forget_live(self._brain), timeout=5.0)
            except Exception as e:
                logger.warning("could not clear the live conversation: %s", e)
        if forgetting_live:
            # The page would otherwise keep showing a conversation the robot
            # no longer has.
            self.hub.clear()
        return JSONResponse({"removed": removed, "cleared": cleared, "sessions": memory.list_sessions()})

    async def _socket(self, websocket: Any) -> None:
        if not token_ok(websocket, self.token) or not origin_ok(websocket.headers):
            await websocket.close(code=_WS_POLICY_VIOLATION)
            return

        self.hub.bind_server_loop()
        await websocket.accept()
        self.hub.add_client(websocket)
        try:
            await websocket.send_json({"role": "history", "lines": self.hub.transcript()})
            while True:
                data = await websocket.receive_json()
                if not isinstance(data, dict):
                    continue
                if data.get("type") == "interrupt":
                    self.hub.request_interrupt()
                    continue
                text = str(data.get("text", ""))[:MAX_MESSAGE_CHARS]
                self.hub.submit(text)
        except Exception:
            pass  # client went away, or sent something unparseable
        finally:
            self.hub.discard_client(websocket)

    def _build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/", self._page),
                Route("/settings", self._settings_page),
                Route("/settings", self._save_settings, methods=["POST"]),
                Route("/status", self._status),
                Route("/memory", self._memory),
                Route("/memory/forget", self._forget, methods=["POST"]),
                *settings_routes.routes(self),
                *app_routes.routes(self),
                WebSocketRoute("/ws", self._socket),
            ]
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        """Run uvicorn on its own thread."""
        import uvicorn

        config = uvicorn.Config(
            self._build_app(),
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="web-chat", daemon=True)
        self._thread.start()
        logger.info("chat page: %s", self.url())

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
