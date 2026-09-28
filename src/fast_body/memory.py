"""Conversation memory — what the robot still knows after it is switched off.

fast-agent saves the conversation after every turn and can hydrate one back
into an agent. This module adds what a robot needs on top: `config.py` pins
`FAST_AGENT_HOME` so the store is ours and outside site-packages; startup
resumes the last conversation if it is recent enough
(`FAST_BODY_MEMORY_RESUME_H`); and `apply_context_budget` turns the compaction
threshold into a token count.

Everything degrades to "no memory" rather than failing a startup.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fast_body.config import MEMORY_DIR, Config

logger = logging.getLogger(__name__)


_manager: Any | None = None


def bind_manager() -> Any | None:
    """Take fast-agent's session manager for this run, and remember it.

    Call once from the thread running the conversation, inside `fast.run()`.
    `get_current_context()` *builds* a context when there isn't one, so calling
    it from the web thread would race the main thread's fast-agent import.
    Binding here means the web thread only ever reads what this already
    resolved.
    """
    global _manager
    try:
        from fast_agent.context import get_current_context

        _manager = getattr(get_current_context(), "session_manager", None)
    except Exception as e:  # a fast-agent that moved it, or sessions turned off
        logger.debug("no fast-agent context for sessions: %s", e)
        _manager = None
    return _manager


def session_manager() -> Any | None:
    """The bound session manager, or None if sessions are off or not up yet.

    Deliberately does not resolve one: see `bind_manager`.
    """
    return _manager


def _age_hours(when: datetime) -> float:
    return (datetime.now() - when).total_seconds() / 3600.0


@dataclass(frozen=True)
class Resume:
    """What startup did about the last conversation."""

    status: str  # one line for the startup log
    resumed: bool = False


async def resume_recent(brain: Any, cfg: Config) -> Resume:
    """Carry the last conversation forward when it is recent enough to still be it.

    `status` is the only place a person finds out whether it remembered — there
    is no terminal on a robot. `resumed` is what the caller acts on.
    """
    # Bind even when memory is off, so the settings page can still delete what
    # was saved before it was turned off. Otherwise "off" would only hide it.
    manager = bind_manager()

    if not cfg.enable_memory:
        _disable_persistence(brain)
        return Resume("off (ENABLE_MEMORY=0), nothing saved or resumed this run")

    if manager is None:
        return Resume("unavailable (no session manager)")

    if cfg.memory_resume_h <= 0:
        return Resume(f"saving to {MEMORY_DIR}, not resuming (FAST_BODY_MEMORY_RESUME_H=0)")

    try:
        latest = manager.load_latest_session(require_content=True)
    except Exception as e:
        logger.warning("could not read the last session: %s", e)
        return Resume("unreadable, starting fresh")

    if latest is None:
        return Resume(f"nothing to resume, saving to {MEMORY_DIR}")

    age = _age_hours(latest.info.last_activity)
    if age > cfg.memory_resume_h:
        return Resume(f"last conversation was {age:.1f}h ago (over {cfg.memory_resume_h:g}h), starting fresh")

    try:
        agent = brain.resolve_agent(None)  # the default "body" agent
        wanted, wire_before = _brain_model(agent)
        result = await manager.resume_session_agents_async(
            {agent.name: agent}, latest.info.name, fallback_agent_name=agent.name
        )
    except Exception as e:
        # A session written by an older build, or a half-written one. Losing the
        # memory is survivable; refusing to start the conversation is not.
        logger.warning("could not resume %s: %s", latest.info.name, e)
        return Resume("unresumable, starting fresh")

    if result is None:
        return Resume("nothing to resume, starting fresh")

    await _keep_brain_model(agent, wanted, wire_before)
    turns = len(getattr(agent, "message_history", []) or [])
    return Resume(
        f"resumed {result.session.info.name} from {age:.1f}h ago, {turns} messages",
        resumed=True,
    )


def _brain_model(agent: Any) -> tuple[str | None, str | None]:
    """The model this run was configured for, and the wire name it attached."""
    wanted = getattr(getattr(agent, "config", None), "model", None)
    if not wanted:
        try:
            from fast_agent.context import get_current_context

            settings = get_current_context().config
            wanted = settings.default_model if settings is not None else None
        except Exception:
            wanted = None
    wire = getattr(getattr(agent, "llm", None), "model_name", None)
    return wanted, wire


async def _keep_brain_model(agent: Any, wanted: str | None, wire_before: str | None) -> None:
    """Put the configured brain back if resuming swapped it.

    fast-agent restores the model a session was saved with, so a conversation
    held on one brain would carry that brain into the next run. The card and
    the config choose the brain; the memory only supplies the transcript.
    """
    wire_after = getattr(getattr(agent, "llm", None), "model_name", None)
    if not wanted or wire_after == wire_before:
        return
    try:
        await agent.set_model(wanted)
        logger.info("memory: resumed session used %s, back on %s", wire_after, wanted)
    except Exception as e:
        logger.warning("could not put the brain back on %s: %s", wanted, e)


def _disable_persistence(brain: Any) -> None:
    """Stop the after-turn save hook writing anything this run."""
    try:
        agent = brain.resolve_agent(None)
        agent.set_session_history_persistence_enabled(False)
    except Exception as e:
        logger.warning("could not turn session saving off: %s", e)


def apply_context_budget(brain: Any, cfg: Config) -> str:
    """Make auto-compaction fire on a token count instead of a share of the window.

    fast-agent compacts when usage crosses `compaction.threshold`, a *fraction*
    of the model's context window. That unit assumes the window is the budget,
    which it isn't here: the router models report 1M tokens, so the 0.85
    default first compacts at ~891k. Converting a token budget into the
    fraction keeps the knob absolute across a model swap.

    The window is only known once the LLM is attached, hence a call at startup
    rather than a value in the config file.
    """
    if cfg.context_budget_tokens <= 0:
        return "off (FAST_BODY_CONTEXT_BUDGET=0), using fast-agent's own threshold"

    try:
        from fast_agent.context import get_current_context

        settings = get_current_context().config
        agent = brain.resolve_agent(None)
        window = agent.usage_accumulator.context_window_size
    except Exception as e:
        logger.warning("could not read the context window: %s", e)
        return "unknown, using fast-agent's own threshold"

    if settings is None:
        return "unavailable (no fast-agent settings)"

    if not window:
        # `should_auto_compact` returns False on an unknown window, so
        # compaction never runs.
        return "DISABLED — fast-agent does not know this model's context window, so nothing will bound the conversation"

    threshold = min(cfg.context_budget_tokens / window, 1.0)
    settings.compaction.threshold = threshold
    return f"compacting at ~{cfg.context_budget_tokens} tokens ({threshold:.1%} of the model's {window} window)"


# ── picking a conversation back up ───────────────────────────────────────
REOPENING_MAX_CHARS = 160


def _message_text(message: Any) -> str:
    """The readable text of a history message, whichever end it sits at."""
    getter = message.last_text if message.role == "assistant" else message.first_text
    try:
        return (getter() or "").strip()
    except Exception:
        return ""


def _shorten(text: str, max_chars: int) -> str:
    """One sentence at most, and short enough to speak."""
    flat = re.sub(r"\s+", " ", text).strip()
    sentence = re.split(r"(?<=[.!?])\s", flat, maxsplit=1)[0]
    if len(sentence) <= max_chars:
        return sentence
    return sentence[:max_chars].rsplit(" ", 1)[0] + "…"


def reopening_line(history: Sequence[Any], max_chars: int = REOPENING_MAX_CHARS) -> str | None:
    """What the robot says first when it has picked a conversation back up.

    Quotes the robot's own last words rather than asking the brain for a
    greeting. A generated one would cost a turn before anyone has spoken, and
    `brain.send()` takes a *user* message — so producing it would write a line
    the person never said into the very history it is recalling.
    """
    # Not fast-agent's `find_last_assistant_preview_text`: for a turn that ended
    # in a tool call it returns "Pending tool call: …". Walk back to the last
    # words actually spoken.
    text = ""
    for message in reversed(list(history)):
        if getattr(message, "role", None) != "assistant":
            continue
        text = _message_text(message)
        if text:
            break
    if not text:
        return None
    return f"Picking up where we left off. I'd just said: {_shorten(text, max_chars)}"


def resumed_transcript(history: Sequence[Any]) -> list[dict[str, str]]:
    """The resumed conversation, shaped for the companion page's transcript.

    Without this the page opens empty on a resumed run, which reads as a robot
    that forgot — the opposite of what just happened.
    """
    lines = []
    for message in history:
        role = getattr(message, "role", None)
        if role not in ("user", "assistant"):
            continue
        text = _message_text(message)
        if text:
            lines.append({"role": "user" if role == "user" else "robot", "text": text})
    return lines


# ── the settings page's view of it ───────────────────────────────────────
def list_sessions() -> list[dict[str, Any]]:
    """Saved conversations, newest first. Metadata only — never the transcript."""
    manager = session_manager()
    if manager is None:
        # Before the brain is up (the first half-minute of a boot) there is no
        # manager to ask, but the files are on disk. Reading them keeps the
        # settings page from saying "nothing saved" about a conversation the
        # robot is about to resume.
        return _list_sessions_from_disk()
    try:
        infos = manager.list_sessions(include_empty=False)
    except Exception as e:
        logger.warning("could not list sessions: %s", e)
        return []

    from fast_agent.session import extract_session_title

    return [
        {
            "name": info.name,
            "title": extract_session_title(info.metadata) or "",
            "started": info.created_at.isoformat(timespec="seconds"),
            "last_activity": info.last_activity.isoformat(timespec="seconds"),
            "age_hours": round(_age_hours(info.last_activity), 1),
        }
        for info in infos
    ]


def _list_sessions_from_disk() -> list[dict[str, Any]]:
    """The same listing, read from fast-agent's session files.

    One directory per session under `<home>/sessions`, with a `session.json`
    of metadata and a `history_<agent>.json` of messages. A session with no
    history file is one fast-agent opened and never wrote to; it hides those
    too (`include_empty=False`).
    """
    import json

    root = MEMORY_DIR / "sessions"
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for session_dir in root.iterdir():
        meta_file = session_dir / "session.json"
        if not session_dir.is_dir() or not meta_file.is_file():
            continue
        if not any(session_dir.glob("history_*.json")):
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            created = datetime.fromisoformat(meta["created_at"])
            last = datetime.fromisoformat(meta.get("last_activity") or meta["created_at"])
        except Exception as e:
            logger.debug("skipping session %s: %s", session_dir.name, e)
            continue
        metadata = meta.get("metadata") or {}
        title = metadata.get("title") or metadata.get("label") or metadata.get("first_user_preview") or ""
        out.append(
            {
                "name": meta.get("session_id") or session_dir.name,
                "title": str(title),
                "started": created.isoformat(timespec="seconds"),
                "last_activity": last.isoformat(timespec="seconds"),
                "age_hours": round(_age_hours(last), 1),
            }
        )
    out.sort(key=lambda s: s["last_activity"], reverse=True)
    return out


def live_session_name() -> str | None:
    """The conversation fast-agent is writing to right now, if the brain is up."""
    manager = session_manager()
    session = getattr(manager, "current_session", None) if manager is not None else None
    return getattr(getattr(session, "info", None), "name", None)


def delete_session(name: str) -> bool:
    """Forget one conversation. False if it wasn't there.

    fast-agent refuses a plain delete of the session this process is writing
    ("busy"), so the live one goes through `delete_owned_session`; fast-agent
    opens a fresh session at the next save.
    """
    manager = session_manager()
    if manager is None:
        # The listing reads the disk in this state, so the delete does too.
        return _delete_from_disk(name)
    try:
        live = getattr(manager, "current_session", None)
        if live is not None and live.info.name == name:
            return bool(manager.delete_owned_session(live))
        return bool(manager.delete_session(name))
    except Exception as e:
        logger.warning("could not delete session %s: %s", name, e)
        return False


def _delete_from_disk(name: str) -> bool:
    """Remove a session directory by name, with no manager to ask."""
    import shutil

    if not name or Path(name).name != name:
        return False
    target = MEMORY_DIR / "sessions" / name
    if not target.is_dir():
        return False
    try:
        shutil.rmtree(target)
        return True
    except Exception as e:
        logger.warning("could not delete session %s: %s", name, e)
        return False


async def forget_live(brain: Any) -> bool:
    """Drop the conversation the brain is holding in memory.

    Deleting the file is not enough while the app runs: the agent's history is
    what it answers from, and the next turn would save it all back out.
    """
    try:
        brain.resolve_agent(None).clear()
        return True
    except Exception as e:
        logger.warning("could not clear the brain's conversation: %s", e)
        return False


def forget_all() -> int:
    """Forget every saved conversation, and return how many went."""
    removed = 0
    for entry in list_sessions():
        if delete_session(entry["name"]):
            removed += 1
    return removed
