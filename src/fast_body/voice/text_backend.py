"""Text voice backend — type instead of speak.

Handy for the simulator and for dev machines with no robot audio: the
conversation loop is identical, only the transport changes. `listen()` reads a
line from stdin; `speak()` prints the reply.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable


class TextVoiceBackend:
    """A `VoiceBackend` that uses the terminal instead of audio I/O."""

    async def listen(self) -> str | None:
        # Read stdin off the event loop so we don't block the brain's tasks.
        sys.stdout.write("you> ")
        sys.stdout.flush()
        line = await asyncio.to_thread(sys.stdin.readline)
        if not line:  # EOF (Ctrl-D / closed pipe) → end the conversation
            return None
        return line.strip() or None

    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        print(f"robot> {text}")

    def on_speech_start(self, callback: Callable[[], None] | None) -> None:
        return None  # no speech detector here

    def interrupt(self) -> None:
        return None  # no audio to drop

    async def aclose(self) -> None:
        return None
