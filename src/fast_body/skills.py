"""Agent skills: where they live, and keeping them current from the MCP servers.

fast-agent reads skills (`SKILL.md` plus `references/`, `scripts/`, `assets/`)
from directories, lists them in the instruction under `{{agentSkills}}`, and
gives the brain a `read_skill` tool to open any file under a listed skill. It
can also install a skill from an MCP server that serves one (the SEP-2640
skills extension), verifying every file against the digest the server
declared. What it does not do is decide *where* skills are on a robot or
fetch them without someone typing `/skills install`. Both are handled here.

Directories, most specific first:

- `FAST_BODY_SKILLS`, extra directories separated by the platform's path
  separator, for a robot or a dev machine;
- `<MEMORY_DIR>/skills/<server>/` (`~/.fast-body/skills/…`, see config.py),
  where skills pulled from MCP servers are installed, one directory per server.

`sync_from_servers` runs once the brain is up and its servers are attached:
every skill an attached server offers is installed if it is missing, updated
if the server's copy changed and ours was not edited by hand, and left alone
otherwise. The brain's instruction is then rebuilt so the new skills are
listed. A server that is down, or a skill that fails verification, costs a log
line, never the startup.

The install follows the SEP-2640 spec (ext-skills `specification/stable/
skills.mdx`) where fast-agent lets it: every file is verified
against the digest the server listed, the frontmatter against the entry, and
a served skill is cached at a path that names its server, so two servers'
`guide` skills never collide and the origin is recoverable from the path.
Where it departs: the spec wants files fetched only when read, and this
fetches a skill's files when it is installed, because fast-agent's reader
reads from disk. Names are also not unique across servers; the per-server
directory is what keeps them apart.

Skills configured on an agent make fast-agent switch on its shell runtime, so
the model could run commands on the robot. `keep_shell_off` stops that; it is
applied in agent.py before the brain starts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fast_body.config import MEMORY_DIR

logger = logging.getLogger(__name__)

# Where MCP-served skills are installed, under one directory per server.
# Separate from the user's own directories so a sync never writes into one.
SKILLS_DIR = MEMORY_DIR / "skills"


def _env_dirs() -> list[Path]:
    raw = os.getenv("FAST_BODY_SKILLS", "").strip()
    if not raw:
        return []
    return [Path(p.strip()).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def server_root(server_name: str) -> Path:
    """Where one server's skills are cached. The name is the one in our config,
    not what the server calls itself, as the spec asks."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in server_name).strip(".") or "server"
    return SKILLS_DIR / safe


def _server_roots() -> list[Path]:
    try:
        return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []


def skill_directories() -> list[Path]:
    """The skill directories that exist right now, most specific first.

    Only existing directories: fast-agent warns about a configured directory
    it cannot find, and a server's directory does not exist until the first
    skill lands in it.
    """
    seen: set[Path] = set()
    found: list[Path] = []
    for directory in (*_env_dirs(), *_server_roots()):
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def load_manifests() -> list[Any]:
    """Every skill in `skill_directories()`, as fast-agent manifests."""
    directories = skill_directories()
    if not directories:
        return []
    from fast_agent.skills.registry import SkillRegistry

    registry = SkillRegistry(directories=directories)
    manifests = registry.load_manifests()
    for warning in registry.warnings:
        logger.warning("skills: %s", warning)
    return manifests


def keep_shell_off(app: Any) -> None:
    """Stop skills from switching on fast-agent's shell runtime.

    With skills configured, fast-agent activates a shell tool for the agent
    "because agent skills are configured", unless the context says `no_shell`.
    Only fast-agent's own CLI sets that flag, and the context does not exist
    until `run()` initializes the app, so wrap that step. Without the shell the
    brain reads skill files through `read_skill`, which is what a robot wants.
    """
    original = app.initialize

    async def initialize_without_shell() -> None:
        await original()
        app.context.no_shell = True

    app.initialize = initialize_without_shell


# ── syncing from MCP servers ──────────────────────────────────────────────
async def _sync_one(aggregator: Any, skill: Any) -> str:
    """Install or update one served skill. Returns what happened, for the log."""
    from fast_agent.skills.mcp_registry import (
        install_mcp_registry_skill,
        update_mcp_registry_skill,
    )
    from fast_agent.skills.provenance import (
        compute_skill_content_fingerprint,
        read_installed_skill_source,
    )

    root = server_root(skill.server_name)
    target = root / skill.install_dir_name
    if not target.exists():
        await install_mcp_registry_skill(aggregator, skill, destination_root=root)
        return "installed"

    source = read_installed_skill_source(target).source
    if source is None or source.mcp_server_name != skill.server_name:
        # Not something a sync put there: a hand-written skill, or one from
        # another server. Never overwrite it.
        return "kept"
    if source.installed_revision == skill.revision:
        return "current"
    if compute_skill_content_fingerprint(target) != source.content_fingerprint:
        logger.warning(
            "skills: %s changed on the server but was edited locally; keeping the local copy",
            skill.name,
        )
        return "kept"
    await update_mcp_registry_skill(aggregator, skill, skill_dir=target)
    return "updated"


async def sync_from_servers(brain: Any) -> str:
    """Pull the skills the attached MCP servers offer, then relist them. One log line.

    Runs after the servers are attached. The brain already knows the skill
    directories that existed when it was built; what a sync adds only shows
    up once the instruction is rebuilt, so that happens whenever the set on
    disk changed.
    """
    try:
        agent = brain.resolve_agent(None)
        registries = await agent.aggregator.list_mcp_skill_registries()
    except Exception as e:
        logger.warning("skills: could not list the servers' skills: %s", e)
        return "unavailable"

    counts = {"installed": 0, "updated": 0, "current": 0, "kept": 0, "failed": 0}
    for registry in registries:
        for skill in registry.skills:
            try:
                outcome = await _sync_one(agent.aggregator, skill)
            except Exception as e:
                logger.warning("skills: %s from %s: %s", skill.name, registry.server_name, e)
                outcome = "failed"
            counts[outcome] += 1
            if outcome in ("installed", "updated"):
                logger.info("skills: %s %s from %s", outcome, skill.name, registry.server_name)

    changed = counts["installed"] + counts["updated"] > 0
    if changed or (not agent.skill_manifests and skill_directories()):
        await relist(agent)

    if not registries:
        return "no server offers skills"
    return ", ".join(f"{n} {what}" for what, n in counts.items() if n)


async def relist(agent: Any) -> None:
    """Rebuild the agent's instruction from the skills on disk.

    Also what gives the agent its `read_skill` tool when it started with no
    skills at all: fast-agent only creates the reader when manifests are set.
    """
    from fast_agent.core.instruction_refresh import rebuild_agent_instruction

    manifests = load_manifests()
    await rebuild_agent_instruction(agent, skill_manifests=manifests)
    logger.info("skills: %d listed for the brain", len(manifests))
