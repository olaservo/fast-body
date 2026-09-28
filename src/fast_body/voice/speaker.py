"""The speaking half as a provider: text in, PCM out, nothing about the robot.

A `Speaker` synthesizes one utterance as a stream of 16-bit mono PCM at its own
``sample_rate``. Everything after that — resampling to the robot's rate, the
lead-in buffer, pushing to the speaker, waiting out playback, barge-in — is
robot-side and lives in `OpenAIVoiceBackend.speak()`, shared by every provider.

One provider ships: `OpenAISpeaker`, the OpenAI speech API. It also reaches any
server that speaks the same API (a Qwen3-TTS server exposes ``/v1/audio/speech``)
through ``TTS_BASE_URL`` and ``TTS_API_KEY``, so a local or self-hosted voice is
a configuration change. A provider with a different wire — ElevenLabs, Piper —
is one more class here and one more branch in `build_speaker`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fast_body.config import Config

logger = logging.getLogger(__name__)

_OPENAI_TTS_RATE = 24000  # OpenAI's PCM output rate
_CHUNK_BYTES = 8192  # ~170 ms of 24 kHz mono PCM per read


@runtime_checkable
class Speaker(Protocol):
    """Turns text into audio, one utterance at a time."""

    # Rate of the 16-bit mono PCM that `stream()` yields.
    sample_rate: int

    def stream(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM as it is synthesized.

        ``voice``/``delivery`` override the speaker's defaults for this one
        utterance (a cast member's line); ``None`` keeps them.
        """
        ...


class OpenAISpeaker:
    """The OpenAI speech API, or any server that speaks it."""

    sample_rate = _OPENAI_TTS_RATE

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        voice: str,
        delivery: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model
        self._voice = voice
        self._delivery = delivery

    @property
    def voice(self) -> str:
        return self._voice

    async def stream(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> AsyncIterator[bytes]:
        voice = voice or self._voice
        delivery = self._delivery if delivery is None else delivery
        # `instructions` is how a card steers delivery; the older tts-1 models
        # reject it, so it only goes on the wire when there is one.
        steering: dict[str, Any] = {"instructions": delivery} if delivery else {}
        async with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=voice,
            input=text,
            response_format="pcm",  # raw 24 kHz / 16-bit / mono
            **steering,
        ) as response:
            async for data in response.iter_bytes(_CHUNK_BYTES):
                yield data


def build_speaker(config: Config) -> Speaker:
    """The configured speaker: OpenAI, or an OpenAI-compatible server at ``TTS_BASE_URL``."""
    from fast_body.config import AVAILABLE_TTS_VOICES

    # Resolved once (override → personality card → default); a change from the
    # settings page applies on the next app start, like the personality.
    voice = config.resolve_tts_voice()
    base_url = config.tts_base_url or None
    if not base_url and voice not in AVAILABLE_TTS_VOICES:
        logger.warning("TTS voice %r is not in the known list; using it anyway", voice)
    if base_url:
        logger.info("TTS from %s (model %s)", base_url, config.tts_model)
    return OpenAISpeaker(
        api_key=config.tts_api_key or config.openai_api_key,
        model=config.tts_model,
        voice=voice,
        delivery=config.resolve_tts_delivery(),
        base_url=base_url,
    )
