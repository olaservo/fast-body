"""The settings page's MCP-server, personality-card and voice operations.

Each operation takes the request's parameters and returns the reply, raising
`RequestError` to refuse. `routes` serves them over HTTP for the settings page and
`register` over `/rpc`, so both callers run the same code.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.routing import Route

from fast_body import mcp_servers, personality_store, skills
from fast_body.config import AVAILABLE_TTS_VOICES, Config, persist_env
from fast_body.personality import list_personalities
from fast_body.web.access import RequestError

if TYPE_CHECKING:
    from reachy_mini.apps.jsonrpc_server import JsonRpcServer

    from fast_body.web.server import WebChatServer

logger = logging.getLogger(__name__)

# Personality and voice are read when the conversation starts.
_RESTART = {"restart_required": True}


# ── MCP servers ──────────────────────────────────────────────────────────
async def mcp_list(server: WebChatServer, params: dict[str, Any], note: str = "") -> dict[str, Any]:
    """The MCP section's whole state, returned by every operation in it."""
    attached: list[str] = []
    live = server._brain is not None
    if live:
        try:
            attached = await server._on_brain(mcp_servers.attached_names(server._brain), timeout=5.0)
        except Exception as e:
            logger.warning("could not list attached servers: %s", e)
    servers = mcp_servers.list_servers()
    for s in servers:
        s["attached"] = s["name"] in attached if live else None
    return {
        "live": live,  # False: changes persist and apply at next start
        "servers": servers,
        # Servers the brain has that this page does not manage — the
        # packaged YAML's (via FAST_BODY_SERVERS) and the personality
        # card's.
        "builtin_attached": [a for a in attached if a not in {s["name"] for s in servers}],
        "note": note,
    }


async def _apply_live(server: WebChatServer, name: str, enabled: bool) -> str:
    """Attach or detach a saved server on the running brain. One-line result."""
    if server._brain is None:
        return "saved; applies when the app starts"
    try:
        if enabled:
            ok, message = await server._on_brain(
                mcp_servers.attach(server._brain, name),
                timeout=mcp_servers.ATTACH_TIMEOUT_S + 5.0,
            )
            if ok:
                # A server may serve skills as well as tools; pull them in
                # the way startup does, so the page doesn't need a restart.
                synced = await server._on_brain(skills.sync_from_servers(server._brain), timeout=60.0)
                message = f"{message}; skills: {synced}"
        else:
            ok, message = await server._on_brain(mcp_servers.detach(server._brain, name), timeout=10.0)
    except Exception as e:
        return f"saved, but applying it live failed: {e}"
    return message if ok else f"saved, but applying it live failed: {message}"


async def mcp_save(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name", "")).strip()
    problem = mcp_servers.save_server(
        name,
        transport=str(params.get("transport", "")),
        url=str(params.get("url", "")),
        command=str(params.get("command", "")),
        args=str(params.get("args", "")),
        bearer_token=str(params.get("bearer_token", "")),
    )
    if problem:
        raise RequestError(400, problem)
    enabled = next((s["enabled"] for s in mcp_servers.list_servers() if s["name"] == name), True)
    note = await _apply_live(server, name, enabled) if enabled else "saved"
    return await mcp_list(server, {}, f"{name}: {note}")


async def mcp_toggle(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name", "")).strip()
    enabled = params.get("enabled") is True
    if not mcp_servers.set_enabled(name, enabled):
        raise RequestError(404, f"no server named {name}")
    note = await _apply_live(server, name, enabled)
    return await mcp_list(server, {}, f"{name}: {note}")


async def mcp_delete(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name", "")).strip()
    if not mcp_servers.delete_server(name):
        raise RequestError(404, f"no server named {name}")
    if server._brain is not None:
        try:
            await server._on_brain(mcp_servers.detach(server._brain, name), timeout=10.0)
        except Exception as e:
            logger.warning("could not detach removed server %s: %s", name, e)
    return await mcp_list(server, {}, f"{name}: removed")


# ── personalities (uploaded cards and card packs) ────────────────────────
def _personalities(note: str = "", **extra: Any) -> dict[str, Any]:
    """The section's whole state, returned by every operation in it.

    `choices`, `locked` and `locked_to` are the conversation app's shape for
    `personalities.list`; the rest is fast-body's own.
    """
    cfg = Config()
    available = list_personalities()
    return {
        "current": cfg.personality,
        "available": available,
        "choices": available,
        "locked": False,
        "locked_to": None,
        "directory": str(personality_store.user_dir()),
        "cards": personality_store.list_user_cards(),
        "packs": personality_store.list_packs(),
        "note": note,
        **extra,
    }


async def personalities(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    return _personalities()


async def personality_apply(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """Save the card to use at the next start. Nothing switches live."""
    name = str(params.get("name", "")).strip()
    if name not in list_personalities():
        raise RequestError(404, f"no personality named {name}")
    persist_env({"FAST_BODY_PERSONALITY": name})
    return {**_personalities(f"{name}: saved; restart the app to use it"), **_RESTART}


async def personality_upload(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """One card file: {filename, content}."""
    filename = str(params.get("filename", "")).strip()
    content = params.get("content")
    if not filename or not isinstance(content, str):
        raise RequestError(400, "filename and content are required")
    problem = personality_store.save_card(filename, content)
    if problem:
        raise RequestError(400, problem)
    return _personalities(f"{Path(filename).stem}: uploaded; pick it above and restart the app")


async def personality_delete(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name", "")).strip()
    if not personality_store.delete_card(name):
        raise RequestError(404, f"no uploaded card named {name}")
    return _personalities(f"{name}: removed")


async def _network(label: str, coro: Any) -> Any:
    """Run a store call that reaches the network."""
    try:
        return await asyncio.wait_for(coro, timeout=personality_store.NETWORK_TIMEOUT)
    except ValueError as e:
        raise RequestError(400, str(e)) from e
    except LookupError as e:
        raise RequestError(404, f"no card pack {e} installed") from e
    except TimeoutError as e:
        raise RequestError(504, f"{label} did not finish in three minutes") from e
    except Exception as e:
        logger.warning("%s failed: %s", label, e)
        raise RequestError(502, f"{label} failed: {e}") from e


async def pack_lookup(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """What a registry offers: {source} → {source, packs: [{name, description, kind}]}."""
    source = str(params.get("source", "")).strip()
    found: dict[str, Any] = await _network(f"looking up {source}", personality_store.lookup(source))
    return found


async def pack_install(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """Install one pack from a registry: {source, name, force?}."""
    source = str(params.get("source", "")).strip()
    name = str(params.get("name", "")).strip()
    force = bool(params.get("force", False))
    done = await _network(f"installing {name}", personality_store.install(source, name, force))
    return _personalities(done["note"], readme=done.get("readme"))


async def pack_update(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """Bring one installed pack up to its source: {name, force?}."""
    name = str(params.get("name", "")).strip()
    force = bool(params.get("force", False))
    done = await _network(f"updating {name}", personality_store.update(name, force))
    return _personalities(done["note"], status=done["status"], pack=name)


async def pack_remove(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    name = str(params.get("name", "")).strip()
    if not personality_store.remove(name):
        raise RequestError(404, f"no card pack {name} installed")
    return _personalities(f"{name}: removed")


# ── voices ───────────────────────────────────────────────────────────────
def voices(server: WebChatServer, params: dict[str, Any]) -> list[str]:
    return list(AVAILABLE_TTS_VOICES)


def voice_current(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    cfg = Config()
    return {"voice": cfg.resolve_tts_voice(), "override": cfg.tts_voice_overridden()}


def voice_apply(server: WebChatServer, params: dict[str, Any]) -> dict[str, Any]:
    """Pin the voice for the next start. Any name passes, as on the settings page."""
    voice = str(params.get("voice", "")).strip()
    if not voice:
        raise RequestError(400, "voice is required")
    persist_env({"OPENAI_TTS_VOICE": voice})
    return {"voice": voice, **_RESTART}


def routes(server: WebChatServer) -> list[Route]:
    get, post = server.http_get, server.http_post
    return [
        Route("/mcp", partial(get, partial(mcp_list, server))),
        Route("/mcp/save", partial(post, partial(mcp_save, server)), methods=["POST"]),
        Route("/mcp/toggle", partial(post, partial(mcp_toggle, server)), methods=["POST"]),
        Route("/mcp/delete", partial(post, partial(mcp_delete, server)), methods=["POST"]),
        Route("/personalities", partial(get, partial(personalities, server))),
        Route("/personalities/upload", partial(post, partial(personality_upload, server)), methods=["POST"]),
        Route("/personalities/delete", partial(post, partial(personality_delete, server)), methods=["POST"]),
        Route("/personalities/packs/lookup", partial(post, partial(pack_lookup, server)), methods=["POST"]),
        Route("/personalities/packs/install", partial(post, partial(pack_install, server)), methods=["POST"]),
        Route("/personalities/packs/update", partial(post, partial(pack_update, server)), methods=["POST"]),
        Route("/personalities/packs/remove", partial(post, partial(pack_remove, server)), methods=["POST"]),
    ]


def register(rpc: JsonRpcServer, server: WebChatServer) -> None:
    from fast_body.web.rpc import expose

    for name, op in {
        "servers.list": mcp_list,
        "servers.save": mcp_save,
        "servers.toggle": mcp_toggle,
        "servers.delete": mcp_delete,
        "personalities.list": personalities,
        "personalities.apply": personality_apply,
        "personalities.upload": personality_upload,
        "personalities.delete": personality_delete,
        "personalities.packs.lookup": pack_lookup,
        "personalities.packs.install": pack_install,
        "personalities.packs.update": pack_update,
        "personalities.packs.remove": pack_remove,
        "voices.list": voices,
        "voices.current": voice_current,
        "voices.apply": voice_apply,
    }.items():
        expose(rpc, name, partial(op, server))
