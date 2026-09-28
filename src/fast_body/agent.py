"""The fast-agent brain that drives the body.

Creates a single `FastAgent` instance, declares the default conversational agent,
and attaches the robot-body tools. The brain is provider-agnostic — which model
reasons (and which MCP servers it can reach) is decided in `fast-agent.config.yaml`
plus `FAST_BODY_SERVERS`, never here. Who the robot *is* comes from a personality
card (`FAST_BODY_PERSONALITY`, see personality.py); the card may also name a model
and extra MCP servers.

Usage from the app::

    from fast_body.agent import fast
    async with fast.run() as brain:
        reply = await brain.send("hello")
"""

from __future__ import annotations

import logging
from typing import Any

from fast_agent import FastAgent

from fast_body import choices, skills
from fast_body.config import FAST_AGENT_CONFIG, Config, packaged_servers
from fast_body.personality import load_personality
from fast_body.prompts import compose_instruction

logger = logging.getLogger(__name__)

_cfg = Config()
_persona = load_personality(_cfg.personality)
logger.info("personality: %s", _persona.name)

fast = FastAgent(
    "fast-body",
    parse_cli_args=False,  # the body owns argv, not the framework
    # The body owns the terminal — except in the interactive dev console (--console),
    # where we want fast-agent's own Rich panels (prompts, tool calls, results) visible.
    quiet=not _cfg.console,
    config_path=str(FAST_AGENT_CONFIG) if FAST_AGENT_CONFIG else None,
)
# Skills would otherwise hand the brain a shell on the robot — see skills.py.
skills.keep_shell_off(fast.app)

# Skills already on disk are listed from the start; the ones the MCP servers
# serve are pulled in by skills.sync_from_servers once the brain is up. None
# rather than [] when there are none: fast-agent's default would be to scan
# the working directory, which on a robot is the daemon's.
_skills: list[Any] = skills.skill_directories()

# Servers a card asks for extend the env-configured list. fast-agent will only
# build the agent with names its own config defines, so a card's server that
# is not in fast-agent.config.yaml is left out here: if the user added it from
# the settings page it attaches once the brain is up (mcp_servers.attach_enabled),
# and if not, the log says which name went unmet.
_known = packaged_servers()
_servers = _cfg.servers + [s for s in _persona.servers if s not in _cfg.servers]
_deferred = [s for s in _servers if s not in _known]
_servers = [s for s in _servers if s in _known]
if _deferred:
    logger.info(
        "servers not in the packaged config, expected from the settings page: %s",
        ", ".join(_deferred),
    )


@fast.agent(
    name="body",
    instruction=compose_instruction(_persona.instruction, cast=_persona.cast),
    servers=_servers,
    model=_persona.model,
    skills=_skills or None,
    # A server's form (MCP elicitation) is answered at the table: spoken
    # options, a tap on the page or the player's words — see choices.py. This
    # outranks every server's `elicitation:` setting, including the `none`
    # that user-added servers default to.
    elicitation_handler=choices.handler,
    default=True,
)
async def _body_agent() -> None:
    """The robot's voice and mind. Robot tools are registered globally below."""


# Importing the tools module decorates the body-control functions onto `fast`.
# Done last so `fast` exists when the tools module imports it.
from fast_body.embodiment import tools as _tools  # noqa: E402,F401
