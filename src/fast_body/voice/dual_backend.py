"""Dual voice backend — type *or* speak, interchangeably.

Wraps a voice backend and a typed-input surface: the
terminal (``ConsoleInput``) or the browser page (``WebChatHub``). ``listen()``
races the two and returns whichever produces an utterance first, so a typed line
and a spoken one reach ``brain.send()`` on the same path.

The interrupt is wired through here: :attr:`interrupt_event` is the surface's
event, and :meth:`interrupt` drops in-flight playback on the inner backend. The
turn loop (``app.py``) watches the event and cancels the in-flight task.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fast_body.voice.base import VoiceBackend


class TypedInput(Protocol):
    """A surface the user can type into, alongside the microphone."""

    interrupt_event: asyncio.Event

    async def start(self, stop_event: threading.Event) -> None: ...

    async def read_line(self) -> str | None: ...

    def echo(self, text: str) -> None: ...

    def echo_user(self, text: str) -> None: ...

    async def aclose(self) -> None: ...


class DualVoiceBackend:
    """A `VoiceBackend` that accepts both typed and spoken input."""

    def __init__(self, inner: VoiceBackend, surface: TypedInput) -> None:
        self._inner = inner
        self._surface = surface

    @property
    def interrupt_event(self) -> asyncio.Event:
        return self._surface.interrupt_event

    async def start(self, stop_event: threading.Event) -> None:
        await self._surface.start(stop_event)

    def on_speech_start(self, callback: Callable[[], None] | None) -> None:
        self._inner.on_speech_start(callback)

    async def listen(self) -> str | None:
        # Fresh turn — any earlier Escape is consumed; nothing is in flight yet.
        self.interrupt_event.clear()
        typed = asyncio.ensure_future(self._surface.read_line())
        spoken = asyncio.ensure_future(self._inner.listen())
        try:
            done, pending = await asyncio.wait({typed, spoken}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # A stop cancels the listen; the two reads go with it, or they
            # outlive the loop.
            for task in (typed, spoken):
                task.cancel()
            await asyncio.gather(typed, spoken, return_exceptions=True)
            raise

        # Prefer typed input if both happened to resolve in the same tick.
        result = typed.result() if typed in done else spoken.result()

        for task in pending:  # abandon the loser; its source (queue/mic) persists
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        # Show the surface what the mic heard. Typed input already appears where
        # it was typed.
        if result and typed not in done:
            self._surface.echo_user(result)
        return result

    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        # Raw text: each surface labels its own speaker. A cast's `[Name]` tags
        # stay in as the transcript's speaker labels.
        self._surface.echo(text)
        await self._inner.speak(text, voice=voice, delivery=delivery)

    async def warm_up(self) -> None:
        warm = getattr(self._inner, "warm_up", None)
        if warm is not None:
            await warm()

    def interrupt(self) -> None:
        self._inner.interrupt()

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            await self._surface.aclose()
