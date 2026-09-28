"""Shared keyboard layer for the interactive dev console.

One prompt_toolkit input surface serves both halves of the console:

- **Typed input** — a persistent ``you>`` prompt the user can type into at any
  time; completed lines are queued for :meth:`read_line`.
- **Escape interrupt** — a key binding that sets :attr:`interrupt_event` no matter
  what phase the turn loop is in (listening, thinking, or speaking).

The session runs continuously in its own task under ``patch_stdout`` so the
brain's Rich panels scroll *above* the prompt instead of clobbering it. This is
the raw-key layer the line-buffered ``TextVoiceBackend`` can't provide — a bare
Escape never reaches ``sys.stdin.readline()``.

Bare Escape is, by terminal convention, the prefix of arrow/Alt sequences, so the
binding is intentionally non-eager: prompt_toolkit waits out a short disambiguation
timeout before firing. That keeps arrow-key history working at the cost of a small,
acceptable delay on the interrupt.
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession


class ConsoleInput:
    """A continuously-running prompt_toolkit surface: typed lines + Escape."""

    def __init__(self) -> None:
        self.interrupt_event = asyncio.Event()
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._session: PromptSession | None = None
        self._task: asyncio.Task | None = None
        self._stop_event: threading.Event | None = None

    async def start(self, stop_event: threading.Event) -> None:
        """Begin reading the keyboard; ``stop_event`` is set on EOF/Ctrl-C."""
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings

        self._stop_event = stop_event
        kb = KeyBindings()

        @kb.add("escape")
        def _interrupt(event) -> None:  # noqa: ANN001 — prompt_toolkit event
            event.current_buffer.reset()  # drop the half-typed line
            self.interrupt_event.set()

        self._session = PromptSession(key_bindings=kb)
        self._task = asyncio.ensure_future(self._run())

    async def _run(self) -> None:
        from prompt_toolkit.patch_stdout import patch_stdout

        with patch_stdout():
            while True:
                try:
                    assert self._session is not None
                    line = await self._session.prompt_async("you> ")
                except (EOFError, KeyboardInterrupt):
                    if self._stop_event is not None:
                        self._stop_event.set()  # end the conversation session
                    await self._lines.put(None)
                    return
                text = (line or "").strip()
                if text:  # ignore blank Enters; keep the prompt up
                    await self._lines.put(text)

    async def read_line(self) -> str | None:
        """Return the next submitted line, or ``None`` on EOF/Ctrl-C."""
        return await self._lines.get()

    def echo(self, text: str) -> None:
        """Print through ``patch_stdout`` so it lands above the live prompt."""
        print(f"robot> {text}")

    def echo_user(self, text: str) -> None:
        """Show what the mic heard; typed lines are already on screen."""
        print(f"you> {text}")

    async def aclose(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
