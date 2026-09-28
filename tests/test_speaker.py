"""The speaking half is a provider; the playback path is shared and rate-agnostic."""

from __future__ import annotations

import numpy as np

from fast_body.config import Config
from fast_body.voice.openai_backend import OpenAIVoiceBackend
from fast_body.voice.speaker import OpenAISpeaker, Speaker, build_speaker
from tests.test_realtime_backend import FakeMedia, FakeRobotAudio


class _ToneSpeaker:
    """A provider that synthesizes a fixed-length tone at its own rate."""

    def __init__(self, sample_rate: int, seconds: float) -> None:
        self.sample_rate = sample_rate
        self._seconds = seconds
        self.requests: list[tuple[str, str | None, str | None]] = []

    async def stream(self, text: str, *, voice: str | None = None, delivery: str | None = None):
        self.requests.append((text, voice, delivery))
        n = int(self.sample_rate * self._seconds)
        t = np.arange(n) / self.sample_rate
        pcm = (np.sin(2 * np.pi * 220 * t) * 12000).astype(np.int16).tobytes()
        for i in range(0, len(pcm), 4096):
            yield pcm[i : i + 4096]


class _Media16k(FakeMedia):
    def get_output_audio_samplerate(self) -> int:
        return 16000


async def _play(monkeypatch, speaker: Speaker) -> _Media16k:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TTS_LEAD_IN_S", "0.1")
    media = _Media16k()
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media), speaker=speaker)
    backend._barge_in = False
    await backend.speak("hello there")
    return media


async def test_providers_at_different_rates_play_the_same_duration(monkeypatch) -> None:
    """The point of the seam: resampling, lead-in and push do not depend on the provider."""
    played = {}
    for rate in (24000, 16000, 22050):
        media = await _play(monkeypatch, _ToneSpeaker(rate, seconds=1.0))
        played[rate] = sum(p.size for p in media.pushed)
    for rate, samples in played.items():
        assert abs(samples - 16000) <= 8, f"{rate} Hz provider played {samples / 16000:.3f}s"


async def test_a_provider_conforms_to_the_protocol() -> None:
    assert isinstance(_ToneSpeaker(16000, 0.1), Speaker)


async def test_backend_passes_voice_and_delivery_to_the_provider(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    speaker = _ToneSpeaker(24000, 0.2)
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(_Media16k()), speaker=speaker)
    backend._barge_in = False

    await backend.speak("plain")
    await backend.speak("as someone", voice="onyx", delivery="gravelly")

    assert speaker.requests == [("plain", None, None), ("as someone", "onyx", "gravelly")]


def test_build_speaker_defaults_to_openai(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TTS_BASE_URL", raising=False)
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_TTS_VOICE", "nova")
    speaker = build_speaker(Config())
    assert isinstance(speaker, OpenAISpeaker)
    assert speaker.voice == "nova"
    assert speaker.sample_rate == 24000
    assert "api.openai.com" in str(speaker._client.base_url)


def test_build_speaker_can_point_at_a_compatible_server(monkeypatch) -> None:
    """A Qwen3-TTS server speaks the OpenAI API; only the address and key change."""
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TTS_BASE_URL", "http://laptop.local:8001/v1")
    monkeypatch.setenv("TTS_API_KEY", "server-key")
    monkeypatch.setenv("OPENAI_TTS_VOICE", "gm_narrator")  # not an OpenAI voice, so no warning expected
    speaker = build_speaker(Config())
    assert isinstance(speaker, OpenAISpeaker)
    assert str(speaker._client.base_url).startswith("http://laptop.local:8001/v1")
    assert speaker._client.api_key == "server-key"
    assert speaker.voice == "gm_narrator"


def test_compatible_server_key_falls_back_to_the_openai_one(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("TTS_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.delenv("TTS_API_KEY", raising=False)
    assert build_speaker(Config())._client.api_key == "openai-key"
