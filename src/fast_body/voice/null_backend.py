"""No local audio — the companion browser page is the only way in and out.

`VOICE_BACKEND=none` for a robot with no usable mic, or when you just want to type
from a laptop. `listen()` never returns, so in `DualVoiceBackend` the browser
always wins the race; `speak()` is silent and the reply reaches you as text.

Only useful with the web chat enabled. `build_backend` rejects the pairing
otherwise, since nothing could reach the robot.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable


class NullVoiceBackend:
    """A `VoiceBackend` with no microphone and no speaker."""

    async def listen(self) -> str | None:
        await asyncio.Event().wait()  # cancelled when the other surface wins
        return None

    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        return None

    def on_speech_start(self, callback: Callable[[], None] | None) -> None:
        return None  # no speech detector here

    def interrupt(self) -> None:
        return None

    async def aclose(self) -> None:
        return None
