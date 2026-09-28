"""The voice transport interface, plus a factory that selects a backend.

A `VoiceBackend` turns the robot's microphone into user text and turns assistant
text into spoken audio. The conversation loop only ever sees text, which is what
keeps the brain provider-agnostic and the voice stack swappable.

Implementations:
- ``openai``   — VAD + OpenAI STT + OpenAI TTS over the robot's audio I/O.
- ``realtime`` — the same TTS, but listening moves to a streaming OpenAI
  transcription session so the server does the endpointing.
- ``text``     — read from stdin / print to stdout (for sim and dev with no audio).
- ``none``     — no audio at all; the companion browser page carries the conversation.

Any of them can be paired with the browser page, which arrives here as
``web_hub`` and gets wrapped in a ``DualVoiceBackend``.

A new backend is one more class and one more branch here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fast_body.config import Config

logger = logging.getLogger(__name__)


class VoiceBackend(Protocol):
    """Bidirectional voice transport between the person and the brain."""

    async def listen(self) -> str | None:
        """Block until the user finishes an utterance; return its text.

        Returns ``None`` if no utterance was captured (timeout / silence) so the
        caller can loop again, or to signal end-of-input (e.g. EOF in text mode).
        """
        ...

    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        """Speak ``text`` aloud (and block until playback finishes).

        ``voice`` and ``delivery`` override, for this utterance only, the ones
        the backend was built with — that is how a card's cast speaks in
        several voices out of one backend. ``None`` means the backend's own.
        Backends with no audio ignore both.
        """
        ...

    def on_speech_start(self, callback: Callable[[], None] | None) -> None:
        """Register a callback for the moment ``listen()`` first hears speech.

        Fired at most once per utterance, from the event loop, before the
        transcript is ready; the body uses it to switch from idle breathing to
        the listening pose. Backends with no speech detector (text, none)
        keep the callback and never fire it.
        """
        ...

    def interrupt(self) -> None:
        """Immediately drop any in-flight playback (the audio half of a barge-in).

        Synchronous and non-blocking — it just clears queued/playing output so the
        caller can return to listening at once. A no-op for backends with no audio.
        """
        ...

    async def aclose(self) -> None:
        """Release any audio / network resources."""
        ...


def build_backend(config: Config, robot, web_hub=None) -> VoiceBackend:
    """Construct the configured voice backend.

    Args:
        config: app configuration (selects the backend and tunes models/voice).
        robot: connected ``ReachyMini`` providing microphone and speaker I/O.
        web_hub: optional ``WebChatHub``; wraps the mic backend so the companion
            browser page can type into the same conversation.
    """
    # Interactive dev console: wrap the mic backend so the user can type *or* speak
    # and Escape can interrupt. Takes precedence over the plain backend selection.
    if config.console:
        from fast_body.voice.console_input import ConsoleInput
        from fast_body.voice.dual_backend import DualVoiceBackend
        from fast_body.voice.openai_backend import OpenAIVoiceBackend

        return DualVoiceBackend(with_cast(OpenAIVoiceBackend(config, robot), config), ConsoleInput())

    backend = config.voice_backend.lower()
    if backend == "text":
        from fast_body.voice.text_backend import TextVoiceBackend

        # Not wrapped: the printed reply keeps its `[Name]` tags, which is the
        # useful thing to see when there is no audio.
        inner: VoiceBackend = TextVoiceBackend()
    elif backend == "openai":
        from fast_body.voice.openai_backend import OpenAIVoiceBackend

        inner = with_cast(OpenAIVoiceBackend(config, robot), config)
    elif backend == "realtime":
        from fast_body.voice.realtime_backend import RealtimeVoiceBackend

        inner = with_cast(RealtimeVoiceBackend(config, robot), config)
    elif backend == "none":
        if web_hub is None:
            raise ValueError("VOICE_BACKEND=none needs the web chat on; nothing else could reach the robot")
        from fast_body.voice.null_backend import NullVoiceBackend

        inner = NullVoiceBackend()
    else:
        raise ValueError(
            f"Unknown VOICE_BACKEND '{config.voice_backend}' (expected 'openai', 'realtime', 'text' or 'none')"
        )

    if web_hub is not None:
        from fast_body.voice.dual_backend import DualVoiceBackend

        return DualVoiceBackend(inner, web_hub)
    return inner


def with_cast(backend: VoiceBackend, config: Config) -> VoiceBackend:
    """Wrap a speaking backend so the card's cast switches voices mid-reply.

    Returns the backend itself when the personality has no cast, so a card
    without one costs nothing.
    """
    from fast_body.personality import load_personality

    cast = load_personality(config.personality).cast
    if not cast:
        return backend
    from fast_body.voice.cast import CastVoiceBackend

    logger.info("cast: %s", ", ".join(cast.names))
    return CastVoiceBackend(backend, cast)
