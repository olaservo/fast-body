"""Personality cards for the fast-body brain.

A personality is a fast-agent **AgentCard**: a markdown (or YAML) file whose
frontmatter describes the agent and whose body is the persona instruction.
fast-agent's own card loader parses them, so a fast-body personality is also a
valid card anywhere else fast-agent takes one. Cards carry only *character*;
the shared embodiment guide from `prompts.py` is appended when the agent is
built (see `agent.py`).

Cards are searched in this order, most specific first: an optional external
directory (`FAST_BODY_PERSONALITIES_DIR`); `<MEMORY_DIR>/personalities/`
(`~/.fast-body/personalities/`, cards that are not part of the app; see
config.py); each pack installed below it (`packs/<owner>--<repo>/`, see
personality_store.py); and the packaged `personalities/` directory next to
this file. A card with the same stem as a
packaged one wins, so users can override the built-ins without touching the
install.

A broken or missing card never stops the app: selection falls back to the
packaged `default` card with a logged warning, because on a robot there may be
no terminal to fix it from.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fast_body.config import MEMORY_DIR, PACKAGE_DIR
from fast_body.voice.cast import Cast

logger = logging.getLogger(__name__)

PERSONALITIES_DIR = PACKAGE_DIR / "personalities"
# Cards that live with the robot rather than the app (see config.py).
USER_PERSONALITIES_DIR = MEMORY_DIR / "personalities"
DEFAULT_PERSONALITY = "default"

# The extensions fast-agent's card loader accepts.
CARD_SUFFIXES = (".md", ".markdown", ".yaml", ".yml")


@dataclass(frozen=True)
class Personality:
    """What the brain and the voice layer need from a card."""

    name: str
    instruction: str
    model: str | None
    servers: list[str]
    # From the card's free-form `variables` mapping (`variables: {voice: ash}`) —
    # a field the AgentCard schema reserves but fast-agent itself doesn't read.
    voice: str | None
    # How to speak, as opposed to which voice: tone, pace, delivery. Passed to
    # the TTS model as `instructions`, so a card can carry character in the sound
    # and not only in the words.
    delivery: str | None = None
    # Other characters the robot voices, from `variables.cast` — see voice/cast.py.
    # Empty when the card lists none.
    cast: Cast = field(default_factory=lambda: Cast(()))
    # `variables.activity`: tool name → what the chat page may say the robot is
    # doing while that tool runs ("glancing across the room"). A tool not listed
    # shows as an unnamed step, so the card decides which tool names the page shows.
    activity: dict[str, str] = field(default_factory=dict)


def _external_dir() -> Path | None:
    value = os.getenv("FAST_BODY_PERSONALITIES_DIR", "").strip()
    return Path(value) if value else None


def _pack_dirs() -> list[Path]:
    """Installed card packs, in a stable order (personality_store.py fills them)."""
    root = USER_PERSONALITIES_DIR / "packs"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _search_dirs() -> tuple[Path, ...]:
    """Most specific first, so a user card shadows a packaged one of the same name."""
    external = _external_dir()
    dirs = (USER_PERSONALITIES_DIR, *_pack_dirs(), PERSONALITIES_DIR)
    return (external, *dirs) if external else dirs


def is_card_file(path: Path) -> bool:
    """A card, as opposed to a pack's README or a stray file with the right suffix.

    A markdown card must open with frontmatter; fast-agent rejects one that does
    not, so listing it would only offer a name that fails to load.
    """
    if not path.is_file() or path.suffix.lower() not in CARD_SUFFIXES or path.name.startswith("."):
        return False
    if path.suffix.lower() in (".yaml", ".yml"):
        return True
    try:
        with path.open(encoding="utf-8") as f:
            return f.readline().strip() == "---"
    except OSError:
        return False


def _card_path(name: str, directory: Path) -> Path | None:
    for suffix in CARD_SUFFIXES:
        candidate = directory / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def find_card(name: str) -> Path | None:
    """The card file a personality name resolves to, or None."""
    for directory in _search_dirs():
        found = _card_path(name, directory)
        if found is not None:
            return found
    return None


def list_personalities() -> list[str]:
    """Available personality names — card file stems, `default` first."""
    names: set[str] = set()
    for directory in _search_dirs():
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if is_card_file(entry):
                names.add(entry.stem)
    ordered = sorted(names)
    if DEFAULT_PERSONALITY in ordered:
        ordered.remove(DEFAULT_PERSONALITY)
        ordered.insert(0, DEFAULT_PERSONALITY)
    return ordered


def _card_variables(path: Path) -> dict[str, object]:
    """The card's `variables` mapping, read from the file.

    fast-agent's schema accepts the field but its loader doesn't surface it,
    so it has to come straight from the frontmatter (or the YAML document).
    """
    try:
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml  # type: ignore[import-untyped]  # no stubs; a fast-agent dep

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            import frontmatter

            data = frontmatter.loads(path.read_text(encoding="utf-8")).metadata
    except Exception as e:
        logger.debug("could not read variables from %s: %s", path, e)
        return {}
    variables = data.get("variables") if isinstance(data, dict) else None
    return variables if isinstance(variables, dict) else {}


def _load_card(name: str, path: Path) -> Personality:
    # Imported lazily: the web server lists personalities long before the brain
    # (and with it the rest of fast-agent) is needed.
    from fast_agent.core.agent_card_loader import load_agent_cards

    card = load_agent_cards(path)[0]
    if card.agent_data.get("type") != "basic":
        raise ValueError(f"personality card {path} must be a plain agent (type: agent)")
    config = card.agent_data["config"]
    variables = _card_variables(path)
    voice = variables.get("voice")
    delivery = variables.get("delivery")
    return Personality(
        name=name,
        instruction=config.instruction,
        model=config.model,
        servers=list(config.servers),
        voice=voice.strip() if isinstance(voice, str) and voice.strip() else None,
        delivery=delivery.strip() if isinstance(delivery, str) and delivery.strip() else None,
        # The card's own name is a way back to its voice: `[tavern]` on the tavern card.
        cast=Cast.from_card(variables.get("cast"), narrator_names=(name,)),
        activity=_activity_labels(variables.get("activity")),
    )


def _activity_labels(value: Any) -> dict[str, str]:
    """The card's `activity` mapping, kept to string → non-empty string."""
    if not isinstance(value, dict):
        return {}
    return {str(tool): label.strip() for tool, label in value.items() if isinstance(label, str) and label.strip()}


def load_personality(name: str | None) -> Personality:
    """Load a personality by name, falling back to `default` rather than failing."""
    requested = (name or DEFAULT_PERSONALITY).strip() or DEFAULT_PERSONALITY
    path = find_card(requested)
    if path is not None:
        try:
            return _load_card(requested, path)
        except Exception as e:
            logger.warning("personality %r is unusable (%s); using %r", requested, e, DEFAULT_PERSONALITY)
    elif requested != DEFAULT_PERSONALITY:
        logger.warning("personality %r not found; using %r", requested, DEFAULT_PERSONALITY)

    # The packaged default, deliberately not an external override of it — this
    # is the path that must always work.
    default_path = _card_path(DEFAULT_PERSONALITY, PERSONALITIES_DIR)
    if default_path is None:
        raise RuntimeError(f"packaged default personality is missing from {PERSONALITIES_DIR}")
    return _load_card(DEFAULT_PERSONALITY, default_path)
