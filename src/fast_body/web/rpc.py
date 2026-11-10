"""The JSON-RPC control surface at `/rpc`.

One protocol for every client: the chat page, a script on the LAN, and a JS
app on a phone, whose requests the daemon relays from WebRTC to this socket.
Method names follow the conversation app's where the meaning matches, so a
client written for it drives fast-body too. The daemon keeps `apps.*` for
itself and never forwards it, so nothing here uses that namespace.

Access: the daemon's relay connects from loopback with no token and no
`Origin`, having already authenticated the remote end, so that peer is let in.
Any other peer needs the page's token and, from a browser, its own origin.

The relay carries any JS app the robot's owner opens, so the methods that
change what runs on the robot (MCP servers, card uploads, card packs) refuse
it. They stay available to the chat page and to clients on the network.
"""

from __future__ import annotations

import contextvars
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reachy_mini.apps.jsonrpc_server import JsonRpcServer
from reachy_mini.io.jsonrpc import INVALID_PARAMS, SERVER_ERROR, JsonRpcError
from starlette.websockets import WebSocket

from fast_body.config import Config
from fast_body.web import app_routes, settings_routes
from fast_body.web.access import MAX_MESSAGE_CHARS, RequestError, origin_ok, token_ok

if TYPE_CHECKING:
    from fast_body.web.server import WebChatServer

RPC_PATH = "/rpc"

# Starlette's code for "policy violation".
_WS_POLICY_VIOLATION = 1008

_LOOPBACK = {"127.0.0.1", "::1"}


@dataclass
class _Peer:
    """One `/rpc` connection's own state: whether it is the relay, and whether it has the Talk button down."""

    relay: bool = False
    holding: bool = False


_PEER: contextvars.ContextVar[_Peer | None] = contextvars.ContextVar("fast_body_rpc_peer", default=None)


# Methods the relay may not call: each can make the robot run code it was not
# shipped with.
LAN_ONLY = frozenset(
    {
        "servers.save",
        "servers.toggle",
        "servers.delete",
        "personalities.upload",
        "personalities.delete",
        "personalities.packs.install",
        "personalities.packs.update",
        "personalities.packs.remove",
    }
)


def is_relay(websocket: Any) -> bool:
    """The daemon's relay: loopback, and no `Origin`, which every browser sends."""
    return not websocket.headers.get("origin") and bool(websocket.client) and websocket.client.host in _LOOPBACK


def peer_allowed(websocket: Any, token: str | None) -> bool:
    if is_relay(websocket):
        return True
    return token_ok(websocket, token) and origin_ok(websocket.headers)


class RpcGate:
    """ASGI middleware in front of `/rpc`: refuses a peer without access, and
    releases the Talk button if a peer that was holding it goes away.

    The SDK's handler accepts every connection it is given and runs each
    request inside the connection's task, so both jobs fit around it here.
    """

    def __init__(self, app: Any, server: WebChatServer) -> None:
        self.app = app
        self.server = server

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "websocket" or scope["path"] != RPC_PATH:
            await self.app(scope, receive, send)
            return
        websocket = WebSocket(scope, receive, send)
        if not peer_allowed(websocket, self.server.token):
            await websocket.close(code=_WS_POLICY_VIOLATION)
            return
        peer = _Peer(relay=is_relay(websocket))
        reset = _PEER.set(peer)
        try:
            await self.app(scope, receive, send)
        finally:
            _PEER.reset(reset)
            if peer.holding:  # a tab closed mid-hold must not leave the mic open
                self.server.hub.set_listen(held=False)


def _invalid(message: str) -> JsonRpcError:
    return JsonRpcError(message, reason="invalid_params", code=INVALID_PARAMS)


def expose(rpc: JsonRpcServer, name: str, op: Callable[[dict[str, Any]], Any]) -> None:
    """Register an operation shared with an HTTP route; its `RequestError` becomes an error with a reason."""

    lan_only = name in LAN_ONLY

    async def handler(params: dict[str, Any]) -> Any:
        peer = _PEER.get()
        if lan_only and peer is not None and peer.relay:
            raise JsonRpcError(
                f"{name} is only available from the chat page on the robot's network",
                reason="forbidden",
                code=SERVER_ERROR,
            )
        try:
            result = op(params)
            return await result if inspect.isawaitable(result) else result
        except RequestError as p:
            code = INVALID_PARAMS if p.status == 400 else SERVER_ERROR
            raise JsonRpcError(p.message, reason=p.reason, code=code) from p

    rpc.register(name, handler)


def build(server: WebChatServer) -> JsonRpcServer:
    """The server's `/rpc` methods."""
    rpc = JsonRpcServer()
    hub = server.hub

    def status(_params: dict[str, Any]) -> dict[str, Any]:
        cfg = Config()
        errors = cfg.validate()
        listen = hub.listen_state()
        return {
            "ready": server.ready,
            # The demo panel waits on this before it enables its controls.
            "backend_connected": server.brain_bound(),
            "backend_error": "; ".join(errors) or None,
            "has_hf_connection": not errors,
            "errors": errors,
            "locked": server.token is not None,
            "personality": cfg.personality,
            "voice": cfg.resolve_tts_voice(),
            "listen_mode": listen["mode"] if listen else cfg.listen_mode,
        }

    def say(params: dict[str, Any]) -> dict[str, Any]:
        text = str(params.get("text", "")).strip()[:MAX_MESSAGE_CHARS]
        if not text:
            raise _invalid("say requires 'text'")
        hub.submit(text)
        return {"ok": True}

    def interrupt(_params: dict[str, Any]) -> dict[str, Any]:
        hub.request_interrupt()
        rpc.broadcast_threadsafe("conversation.turn", {"state": "listening", "reason": "interrupted"})
        return {"ok": True}

    def history(_params: dict[str, Any]) -> dict[str, Any]:
        return {"events": hub.history()}

    def mic(params: dict[str, Any]) -> dict[str, Any]:
        """Muted means the listen gate is shut. Only push-to-talk has a gate to shut."""
        listen = hub.listen_state()
        if "muted" in params:
            if listen is None or listen.get("mode") != "push_to_talk":
                raise JsonRpcError("the microphone is always on in this listen mode", reason="unsupported")
            muted = params["muted"]
            if not isinstance(muted, bool):
                raise _invalid("muted must be a boolean")
            if muted:
                hub.set_listen(held=False, latched=False)
            else:
                hub.set_listen(latched=True)
            listen = hub.listen_state()
        return {"muted": bool(listen is not None and not listen.get("enabled"))}

    def listen_state(_params: dict[str, Any]) -> dict[str, Any] | None:
        return hub.listen_state()

    def listen_set(params: dict[str, Any]) -> dict[str, Any] | None:
        """The page's Talk button: {held?, latched?, toggle?}."""
        for name in ("held", "latched", "toggle"):
            if name in params and not isinstance(params[name], bool):
                raise _invalid(f"{name} must be a boolean")
        held = params.get("held")
        peer = _PEER.get()
        if held is not None and peer is not None:
            peer.holding = held
        hub.set_listen(held=held, latched=params.get("latched"), toggle=params.get("toggle") is True)
        return hub.listen_state()

    rpc.register("conversation.status", status)
    rpc.register("conversation.say", say)
    rpc.register("conversation.interrupt", interrupt)
    rpc.register("conversation.history", history)
    rpc.register("conversation.mic", mic)
    rpc.register("listen.state", listen_state)
    rpc.register("listen.set", listen_set)
    expose(rpc, "memory.list", server.memory_list)
    expose(rpc, "memory.forget", server.memory_forget)
    settings_routes.register(rpc, server)
    app_routes.register(rpc, server)
    return rpc
