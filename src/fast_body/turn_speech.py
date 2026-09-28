"""Speak a turn's text as it arrives, ahead of each tool call.

A brain turn with tool calls is several model calls: some text, a tool, more
text, another tool, a final reply. `send()` returns only the final message,
so text written before a tool call would otherwise be spoken after the tool
had run.

fast-agent's `ToolRunner` runs its `before_tool_call(runner, request)` hook
once per assistant message that asks for tools, with that message, before the
tools run. `TurnSpeech` installs that hook on the body agent and speaks the
message's text there, awaited, so the tool waits for the words and the next
block follows in order. The text goes through the same `voice.speak` as the
final reply, so the cast splitter and the page echo apply unchanged. What was
spoken is kept per turn, and app.py leaves it out of the end-of-turn reply.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = "_fast_body_turn_speech"


def text_blocks(message: Any) -> list[str]:
    """The text of a message's content blocks, stripped, empty ones dropped."""
    parts = []
    for block in getattr(message, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return parts


class TurnSpeech:
    """Speaks assistant text before each tool call and remembers what it said."""

    def __init__(
        self,
        voice: Any,
        *,
        stopping: asyncio.Event | None = None,
        on_start: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> None:
        self.voice = voice
        self._stopping = stopping
        self._on_start = on_start
        self._on_end = on_end
        self.spoken: Counter[str] = Counter()
        self.speaking = False
        self._finished_at: float | None = None

    def begin(self) -> None:
        """A new turn: nothing said yet."""
        self.spoken = Counter()
        self.speaking = False
        self._finished_at = None

    def quiet_for(self) -> float | None:
        """Seconds since the last block finished, or None if nothing has been spoken this turn."""
        if self._finished_at is None:
            return None
        return asyncio.get_running_loop().time() - self._finished_at

    def install(self, agent: Any) -> None:
        """Put the hook on `agent.tool_runner_hooks`, ahead of any hook already there. Once per agent."""
        if getattr(agent, _INSTALLED, False):
            return
        from dataclasses import replace

        from fast_agent.agents.tool_runner import ToolRunnerHooks

        hooks = getattr(agent, "tool_runner_hooks", None)
        previous = hooks.before_tool_call if hooks is not None else None

        async def before_tool_call(runner: Any, request: Any) -> None:
            if previous is not None:
                await previous(runner, request)
            await self.speak_message(request)

        if hooks is None:
            agent.tool_runner_hooks = ToolRunnerHooks(before_tool_call=before_tool_call)
        else:
            agent.tool_runner_hooks = replace(hooks, before_tool_call=before_tool_call)
        setattr(agent, _INSTALLED, True)
        logger.info("speech: a turn's text is spoken ahead of each tool call")

    async def speak_message(self, message: Any) -> None:
        """Speak an assistant message's text, unless a stop is in. Cancellation cuts playback."""
        parts = text_blocks(message)
        if not parts or (self._stopping is not None and self._stopping.is_set()):
            return
        text = "\n\n".join(parts)
        logger.info("robot: %s", text)
        self.speaking = True
        if self._on_start is not None:
            self._on_start()
        try:
            await self.voice.speak(text)
        except asyncio.CancelledError:
            interrupt = getattr(self.voice, "interrupt", None)
            if interrupt is not None:
                interrupt()
            raise
        except Exception as e:
            logger.warning("could not speak mid-turn: %s", e)
        finally:
            self.speaking = False
            self._finished_at = asyncio.get_running_loop().time()
            self.spoken.update(parts)
            if self._on_end is not None:
                self._on_end()
