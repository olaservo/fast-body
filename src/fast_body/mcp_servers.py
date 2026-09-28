"""MCP servers added from the settings page, kept apart from the packaged YAML.

User-added servers live in `MEMORY_DIR/mcp_servers.yaml` (see config.py);
the packaged `fast-agent.config.yaml` is overwritten by every app update.

fast-agent never reads this file. The servers reach the brain through
`AgentApp.attach_mcp_server(server_config=...)` — at startup for the ones
marked enabled, and live when the page adds or toggles one mid-run. That is
also why validation here is shape-only: fast-agent's own `MCPServerSettings`
model is applied at attach time, on the loop that has fast-agent imported,
so this module never pulls the fast-agent tree onto the web thread.

Everything degrades to "server skipped" rather than failing a startup: a
corrupt config file must not take the robot's voice with it.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
from typing import Any

import yaml

from fast_body.config import MEMORY_DIR

logger = logging.getLogger(__name__)

SERVERS_FILE = MEMORY_DIR / "mcp_servers.yaml"

# Keys the settings page owns. Anything else in a server's entry (env, cwd,
# auth, …) was put there by hand and is preserved across page edits.
_FORM_KEYS = ("transport", "url", "command", "args", "headers")

# What attach_mcp_server may wait for a server before the page hears back.
ATTACH_TIMEOUT_S = 15.0

# A sleeping free HF Space takes longer than ATTACH_TIMEOUT_S to wake.
# warm_up() wakes every http server at once before the attaches; retry_failed()
# keeps trying the rest in the background.
WARM_UP_TIMEOUT_S = 90.0
WARM_UP_SLOW_S = 3.0  # above this the log says the server was asleep
RETRY_DELAYS_S: tuple[float, ...] = (20.0, 40.0, 80.0)

# Servers attach_enabled could not attach at startup, for retry_failed.
_startup_failures: list[str] = []
_clock = time.monotonic  # patched in tests

# Last attach/detach failure per server, for the page. In-memory only: an
# error worth keeping across restarts would happen again at the next attach.
_errors: dict[str, str] = {}


def _valid_name(name: str) -> bool:
    return bool(name) and all(c.isalnum() or c in "-_." for c in name)


def _read() -> dict[str, Any]:
    """The whole file as a mapping. A broken or missing file reads as empty."""
    try:
        if not SERVERS_FILE.is_file():
            return {}
        payload = yaml.safe_load(SERVERS_FILE.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("could not read %s: %s", SERVERS_FILE, e)
        return {}
    return payload if isinstance(payload, dict) else {}


def _load() -> dict[str, dict[str, Any]]:
    """The stored servers, name → entry."""
    servers = _read().get("servers")
    if not isinstance(servers, dict):
        return {}
    return {str(name): entry for name, entry in servers.items() if isinstance(entry, dict) and _valid_name(str(name))}


def _write(servers: dict[str, dict[str, Any]]) -> bool:
    """Write the servers back, keeping any other top-level keys."""
    try:
        payload = _read()
        payload["servers"] = servers
        SERVERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERVERS_FILE.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
        try:  # entries may carry bearer tokens, like the .env
            SERVERS_FILE.chmod(0o600)
        except OSError:
            pass
        return True
    except OSError as e:
        logger.error("could not write %s: %s", SERVERS_FILE, e)
        return False


def _enabled(entry: dict[str, Any]) -> bool:
    return entry.get("enabled", True) is not False


def list_servers() -> list[dict[str, Any]]:
    """The stored servers as the page shows them. Header values never leave."""
    out = []
    for name, entry in sorted(_load().items()):
        out.append(
            {
                "name": name,
                "enabled": _enabled(entry),
                "transport": entry.get("transport") or ("stdio" if entry.get("command") else "http"),
                "url": entry.get("url") or "",
                "command": entry.get("command") or "",
                "args": shlex.join(entry.get("args") or []),
                "has_bearer": bool((entry.get("headers") or {}).get("Authorization")),
                "error": _errors.get(name, ""),
            }
        )
    return out


def save_server(
    name: str,
    *,
    transport: str,
    url: str = "",
    command: str = "",
    args: str = "",
    bearer_token: str = "",
) -> str | None:
    """Add or update a server from the page's flat fields. Returns an error, or None.

    A blank bearer token keeps the existing one — the page never shows secrets
    back, so blank must mean "leave alone", as it does for the keys above it.
    """
    name = name.strip()
    if not _valid_name(name):
        return "server names are letters, digits, - _ ."
    transport = transport.strip().lower()
    url, command = url.strip(), command.strip()
    if transport == "http":
        if not url.startswith(("http://", "https://")):
            return "an http server needs a URL starting with http:// or https://"
    elif transport == "stdio":
        if not command:
            return "a stdio server needs a command"
    else:
        return "transport must be http or stdio"
    try:
        arg_list = shlex.split(args)
    except ValueError as e:
        return f"arguments: {e}"

    servers = _load()
    entry = dict(servers.get(name, {}))
    kept_headers = entry.get("headers")
    if not isinstance(kept_headers, dict):
        kept_headers = {}
    for key in _FORM_KEYS:
        entry.pop(key, None)
    entry["transport"] = transport
    if transport == "http":
        entry["url"] = url
        headers = dict(kept_headers)
        if bearer_token.strip():
            headers["Authorization"] = f"Bearer {bearer_token.strip()}"
        if headers:
            entry["headers"] = headers
    else:
        entry["command"] = command
        if arg_list:
            entry["args"] = arg_list
    entry.setdefault("enabled", True)

    servers[name] = entry
    if not _write(servers):
        return f"could not write {SERVERS_FILE}"
    _errors.pop(name, None)
    return None


def delete_server(name: str) -> bool:
    servers = _load()
    if name not in servers:
        return False
    del servers[name]
    _errors.pop(name, None)
    return _write(servers)


def set_enabled(name: str, enabled: bool) -> bool:
    servers = _load()
    if name not in servers:
        return False
    servers[name]["enabled"] = enabled
    return _write(servers)


def _server_config(entry: dict[str, Any]) -> Any:
    """fast-agent's model of one entry. Imported here, not at module top —
    see the module docstring for why the web thread must not do this."""
    from fast_agent.config import MCPServerSettings

    fields = {k: v for k, v in entry.items() if k != "enabled"}
    # fast-agent's default elicitation is an interactive prompt nobody sees on
    # a headless robot, and the turn hangs on it. choices.py's handler outranks
    # this when bound; a hand-written `elicitation:` still wins.
    fields.setdefault("elicitation", {"mode": "none"})
    # Without an auth block fast-agent answers a 401 by starting an OAuth flow,
    # which needs a browser nobody has on a headless robot. A bearer token in
    # `headers` still goes out; only the escalation is off.
    if fields.get("transport") in ("http", "sse") or fields.get("url"):
        fields.setdefault("auth", {"oauth": False})
    return MCPServerSettings(**fields)


async def attach(brain: Any, name: str) -> tuple[bool, str]:
    """Attach one stored server to the running brain. Returns (ok, message)."""
    entry = _load().get(name)
    if entry is None:
        return False, f"no server named {name}"
    try:
        from fast_agent.mcp.mcp_aggregator import MCPAttachOptions

        agent = brain.resolve_agent(None)
        # fast-agent will not re-register an attached name with new settings,
        # and force_reconnect only reconnects the old ones, so a re-save
        # detaches first.
        if name in agent.list_attached_mcp_servers():
            await brain.detach_mcp_server(agent.name, name)
        result = await brain.attach_mcp_server(
            agent.name,
            name,
            _server_config(entry),
            MCPAttachOptions(startup_timeout_seconds=ATTACH_TIMEOUT_S, force_reconnect=True),
        )
    except Exception as e:
        _errors[name] = str(e)
        logger.warning("could not attach %s: %s", name, e)
        return False, str(e)
    _errors.pop(name, None)
    tools = result.tools_total if result.tools_total is not None else len(result.tools_added)
    return True, f"attached, {tools} tools"


async def detach(brain: Any, name: str) -> tuple[bool, str]:
    try:
        agent = brain.resolve_agent(None)
        await brain.detach_mcp_server(agent.name, name)
    except Exception as e:
        logger.warning("could not detach %s: %s", name, e)
        return False, str(e)
    return True, "detached"


async def attached_names(brain: Any) -> list[str]:
    """What the running brain is actually connected to, or [] off-brain.

    A coroutine even though the read is sync, so the web thread can marshal it
    onto the brain's loop instead of reading agent state across threads.
    """
    try:
        return list(brain.resolve_agent(None).list_attached_mcp_servers())
    except Exception as e:
        logger.debug("could not list attached servers: %s", e)
        return []


def _http_targets(names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Enabled http servers (name -> entry), optionally only the named ones."""
    out = {}
    for name, entry in sorted(_load().items()):
        if names is not None and name not in names:
            continue
        if not _enabled(entry):
            continue
        url = entry.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            out[name] = entry
    return out


async def warm_up(names: list[str] | None = None) -> str:
    """Send one request to every enabled http server, all at once. One line for the log.

    The reply does not matter, only that one arrives: a sleeping Space starts
    on the first request and answers once it is up. The stored bearer goes
    along, since a private Space answers nothing to a stranger. Never raises;
    an unreachable server is a word in the log and the attach reports it.
    """
    import httpx

    targets = _http_targets(names)
    if not targets:
        return "nothing to warm"

    async def one(name: str, entry: dict[str, Any]) -> str:
        headers = {k: str(v) for k, v in (entry.get("headers") or {}).items()}
        started = _clock()
        try:
            async with httpx.AsyncClient(timeout=WARM_UP_TIMEOUT_S, follow_redirects=True) as client:
                await client.get(entry["url"], headers=headers)
        except Exception as e:
            logger.warning("mcp: %s did not answer the warm-up request: %s", name, e)
            return f"{name} unreachable"
        took = _clock() - started
        if took >= WARM_UP_SLOW_S:
            logger.info("mcp: %s woke in %.0f s", name, took)
            return f"{name} woke in {took:.0f} s"
        return ""

    notes = [n for n in await asyncio.gather(*(one(name, entry) for name, entry in targets.items())) if n]
    quick = len(targets) - len(notes)
    if quick:
        notes.append(f"{quick} up")
    return ", ".join(notes)


async def attach_enabled(brain: Any) -> str:
    """Attach every enabled stored server at startup. One line for the log.

    Failures are per-server and never fatal: a robot whose fetch server is
    down should still talk. What failed is kept for retry_failed().
    """
    servers = _load()
    wanted = [n for n, e in sorted(servers.items()) if _enabled(e)]
    _startup_failures.clear()
    if not wanted:
        return "no user servers"
    warmed = await warm_up(wanted)
    attached: list[str] = []
    failed: list[str] = []
    for name in wanted:
        ok, _ = await attach(brain, name)
        (attached if ok else failed).append(name)
    _startup_failures.extend(failed)
    parts = []
    if attached:
        parts.append("attached " + ", ".join(attached))
    if failed:
        parts.append("failed " + ", ".join(failed))
    if "woke" in warmed or "unreachable" in warmed:
        parts.append(warmed)
    return "; ".join(parts)


def startup_failures() -> list[str]:
    return list(_startup_failures)


async def retry_failed(brain: Any, names: list[str], after: Any = None) -> None:
    """Keep trying the servers that failed at startup, in the background.

    Each round warms the pending servers again and re-attaches; `after` (the
    skills sync) runs once any of them made it. Gives up after RETRY_DELAYS_S
    and says so; the settings page can attach by hand from there.
    """
    pending = [n for n in names if n in _load()]
    for delay in RETRY_DELAYS_S:
        if not pending:
            return
        await asyncio.sleep(delay)
        await warm_up(pending)
        still: list[str] = []
        for name in pending:
            ok, message = await attach(brain, name)
            if ok:
                logger.info("mcp: %s %s, after a retry", name, message)
            else:
                still.append(name)
        if len(still) < len(pending) and after is not None:
            try:
                await after()
            except Exception as e:
                logger.warning("mcp: after a retried attach: %s", e)
        pending = still
    if pending:
        logger.warning("mcp: gave up on %s; attach from the settings page", ", ".join(pending))
