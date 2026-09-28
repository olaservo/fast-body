"""The settings page's MCP-server and personality-card routes."""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from starlette.responses import JSONResponse
from starlette.routing import Route

from fast_body import mcp_servers, personality_store, skills
from fast_body.config import Config
from fast_body.personality import list_personalities
from fast_body.web.access import not_found, token_ok

if TYPE_CHECKING:
    from fast_body.web.server import WebChatServer

logger = logging.getLogger(__name__)


# ── MCP servers ──────────────────────────────────────────────────────────
async def _mcp_list_response(server: WebChatServer, note: str = "") -> Any:
    """The MCP section's whole state, returned by every route in it."""
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
    return JSONResponse(
        {
            "live": live,  # False: changes persist and apply at next start
            "servers": servers,
            # Servers the brain has that this page does not manage — the
            # packaged YAML's (via FAST_BODY_SERVERS) and the personality
            # card's.
            "builtin_attached": [a for a in attached if a not in {s["name"] for s in servers}],
            "note": note,
        }
    )


async def mcp(server: WebChatServer, request: Any) -> Any:
    if not token_ok(request, server.token):
        return not_found()
    return await _mcp_list_response(server)


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


async def mcp_save(server: WebChatServer, request: Any) -> Any:
    body, error = await server._json_body(request)
    if body is None:
        return error
    name = str(body.get("name", "")).strip()
    problem = mcp_servers.save_server(
        name,
        transport=str(body.get("transport", "")),
        url=str(body.get("url", "")),
        command=str(body.get("command", "")),
        args=str(body.get("args", "")),
        bearer_token=str(body.get("bearer_token", "")),
    )
    if problem:
        return JSONResponse({"error": problem}, status_code=400)
    enabled = next((s["enabled"] for s in mcp_servers.list_servers() if s["name"] == name), True)
    note = await _apply_live(server, name, enabled) if enabled else "saved"
    return await _mcp_list_response(server, f"{name}: {note}")


async def mcp_toggle(server: WebChatServer, request: Any) -> Any:
    body, error = await server._json_body(request)
    if body is None:
        return error
    name = str(body.get("name", "")).strip()
    enabled = body.get("enabled") is True
    if not mcp_servers.set_enabled(name, enabled):
        return JSONResponse({"error": f"no server named {name}"}, status_code=404)
    note = await _apply_live(server, name, enabled)
    return await _mcp_list_response(server, f"{name}: {note}")


async def mcp_delete(server: WebChatServer, request: Any) -> Any:
    body, error = await server._json_body(request)
    if body is None:
        return error
    name = str(body.get("name", "")).strip()
    if not mcp_servers.delete_server(name):
        return JSONResponse({"error": f"no server named {name}"}, status_code=404)
    if server._brain is not None:
        try:
            await server._on_brain(mcp_servers.detach(server._brain, name), timeout=10.0)
        except Exception as e:
            logger.warning("could not detach removed server %s: %s", name, e)
    return await _mcp_list_response(server, f"{name}: removed")


# ── personalities (uploaded cards and card packs) ────────────────────────
def _personalities_response(note: str = "", **extra: Any) -> Any:
    """The section's whole state, returned by every route in it."""
    cfg = Config()
    return JSONResponse(
        {
            "current": cfg.personality,
            "available": list_personalities(),
            "directory": str(personality_store.user_dir()),
            "cards": personality_store.list_user_cards(),
            "packs": personality_store.list_packs(),
            "note": note,
            **extra,
        }
    )


async def personalities(server: WebChatServer, request: Any) -> Any:
    if not token_ok(request, server.token):
        return not_found()
    return _personalities_response()


async def personality_upload(server: WebChatServer, request: Any) -> Any:
    """One card file, sent by the page as JSON: {filename, content}."""
    body, error = await server._json_body(request)
    if body is None:
        return error
    filename = str(body.get("filename", "")).strip()
    content = body.get("content")
    if not filename or not isinstance(content, str):
        return JSONResponse({"error": "filename and content are required"}, status_code=400)
    problem = personality_store.save_card(filename, content)
    if problem:
        return JSONResponse({"error": problem}, status_code=400)
    return _personalities_response(f"{Path(filename).stem}: uploaded; pick it above and restart the app")


async def personality_delete(server: WebChatServer, request: Any) -> Any:
    body, error = await server._json_body(request)
    if body is None:
        return error
    name = str(body.get("name", "")).strip()
    if not personality_store.delete_card(name):
        return JSONResponse({"error": f"no uploaded card named {name}"}, status_code=404)
    return _personalities_response(f"{name}: removed")


async def _network(label: str, coro: Any) -> tuple[Any, Any]:
    """Run a store call that reaches the network; (result, None) or (None, error response)."""
    try:
        return await asyncio.wait_for(coro, timeout=personality_store.NETWORK_TIMEOUT), None
    except ValueError as e:
        return None, JSONResponse({"error": str(e)}, status_code=400)
    except LookupError as e:
        return None, JSONResponse({"error": f"no card pack {e} installed"}, status_code=404)
    except TimeoutError:
        return None, JSONResponse({"error": f"{label} did not finish in three minutes"}, status_code=504)
    except Exception as e:
        logger.warning("%s failed: %s", label, e)
        return None, JSONResponse({"error": f"{label} failed: {e}"}, status_code=502)


async def pack_lookup(server: WebChatServer, request: Any) -> Any:
    """What a registry offers: {source} → {source, packs: [{name, description, kind}]}."""
    body, error = await server._json_body(request)
    if body is None:
        return error
    source = str(body.get("source", "")).strip()
    found, error = await _network(f"looking up {source}", personality_store.lookup(source))
    if found is None:
        return error
    return JSONResponse(found)


async def pack_install(server: WebChatServer, request: Any) -> Any:
    """Install one pack from a registry: {source, name, force?}."""
    body, error = await server._json_body(request)
    if body is None:
        return error
    source = str(body.get("source", "")).strip()
    name = str(body.get("name", "")).strip()
    force = bool(body.get("force", False))
    done, error = await _network(f"installing {name}", personality_store.install(source, name, force))
    if done is None:
        return error
    return _personalities_response(done["note"], readme=done.get("readme"))


async def pack_update(server: WebChatServer, request: Any) -> Any:
    """Bring one installed pack up to its source: {name, force?}."""
    body, error = await server._json_body(request)
    if body is None:
        return error
    name = str(body.get("name", "")).strip()
    force = bool(body.get("force", False))
    done, error = await _network(f"updating {name}", personality_store.update(name, force))
    if done is None:
        return error
    return _personalities_response(done["note"], status=done["status"], pack=name)


async def pack_remove(server: WebChatServer, request: Any) -> Any:
    body, error = await server._json_body(request)
    if body is None:
        return error
    name = str(body.get("name", "")).strip()
    if not personality_store.remove(name):
        return JSONResponse({"error": f"no card pack {name} installed"}, status_code=404)
    return _personalities_response(f"{name}: removed")


def routes(server: WebChatServer) -> list[Route]:
    return [
        Route("/mcp", partial(mcp, server)),
        Route("/mcp/save", partial(mcp_save, server), methods=["POST"]),
        Route("/mcp/toggle", partial(mcp_toggle, server), methods=["POST"]),
        Route("/mcp/delete", partial(mcp_delete, server), methods=["POST"]),
        Route("/personalities", partial(personalities, server)),
        Route("/personalities/upload", partial(personality_upload, server), methods=["POST"]),
        Route("/personalities/delete", partial(personality_delete, server), methods=["POST"]),
        Route("/personalities/packs/lookup", partial(pack_lookup, server), methods=["POST"]),
        Route("/personalities/packs/install", partial(pack_install, server), methods=["POST"]),
        Route("/personalities/packs/update", partial(pack_update, server), methods=["POST"]),
        Route("/personalities/packs/remove", partial(pack_remove, server), methods=["POST"]),
    ]
