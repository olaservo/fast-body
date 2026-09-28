"""Realtime transcription backend — session shape, event handling, mic pump.

Everything here runs against fakes: no websocket, no audio device, no key. The
live session (reconnect, real endpointing latency) still needs the robot.
"""

from __future__ import annotations

import asyncio
import base64

import numpy as np
import pytest
from openai.types.realtime.realtime_transcription_session_create_request import (
    RealtimeTranscriptionSessionCreateRequest,
)
from openai.types.realtime.session_update_event import SessionUpdateEvent

from fast_body.config import Config
from fast_body.voice.base import build_backend
from fast_body.voice.realtime_backend import RealtimeVoiceBackend


class FakeMedia:
    def __init__(self, samples: list[np.ndarray] | None = None, in_rate: int = 16000) -> None:
        self._samples = list(samples or [])
        self._in_rate = in_rate
        self.recording = False
        self.playing = False
        self.pushed: list[np.ndarray] = []
        self.cleared = 0

    def get_input_audio_samplerate(self) -> int:
        return self._in_rate

    def get_output_audio_samplerate(self) -> int:
        return 24000

    def get_audio_sample(self):
        return self._samples.pop(0) if self._samples else None

    def start_recording(self) -> None:
        self.recording = True

    def start_playing(self) -> None:
        self.playing = True

    def stop_recording(self) -> None:
        self.recording = False

    def stop_playing(self) -> None:
        self.playing = False

    def push_audio_sample(self, audio) -> None:
        self.pushed.append(audio)

    def clear_player(self) -> None:
        self.cleared += 1


class FakeRobotAudio:
    def __init__(self, media: FakeMedia) -> None:
        self.media = media


class FakeEvent:
    def __init__(self, type: str, **fields) -> None:
        self.type = type
        for name, value in fields.items():
            setattr(self, name, value)


class FakeInputBuffer:
    def __init__(self) -> None:
        self.appended: list[str] = []

    async def append(self, audio: str) -> None:
        self.appended.append(audio)


class FakeConn:
    """An async-iterable stand-in for a realtime connection."""

    def __init__(self, events: list[FakeEvent] | None = None) -> None:
        self._events = list(events or [])
        self.input_audio_buffer = FakeInputBuffer()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


@pytest.fixture
def backend(monkeypatch) -> RealtimeVoiceBackend:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VOICE_BACKEND", "realtime")
    robot = FakeRobotAudio(FakeMedia())
    made = RealtimeVoiceBackend(Config(), robot)
    # Don't open a websocket; the tests drive _consume/_pump_mic directly.
    made._ensure_session = lambda: None  # type: ignore[method-assign]
    return made


def test_session_is_transcription_only_at_24k(backend: RealtimeVoiceBackend) -> None:
    """No model in the session, and the one input rate the API accepts."""
    session = backend._session_config()
    audio_in = session["audio"]["input"]

    assert session["type"] == "transcription"
    assert "instructions" not in session and "tools" not in session
    assert audio_in["format"] == {"type": "audio/pcm", "rate": 24000}
    assert audio_in["transcription"]["model"] == "gpt-transcribe"
    assert audio_in["transcription"]["language"] == "en"
    assert audio_in["noise_reduction"] == {"type": "far_field"}
    assert audio_in["turn_detection"]["type"] == "server_vad"
    assert audio_in["turn_detection"]["silence_duration_ms"] == 750


def test_session_vad_knobs_follow_config(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("REALTIME_VAD_THRESHOLD", "0.7")
    monkeypatch.setenv("REALTIME_SILENCE_MS", "400")
    monkeypatch.setenv("REALTIME_NOISE_REDUCTION", "near_field")
    monkeypatch.setenv("REALTIME_STT_MODEL", "gpt-live-transcribe")
    made = RealtimeVoiceBackend(Config(), FakeRobotAudio(FakeMedia()))

    turn_detection = made._session_config()["audio"]["input"]["turn_detection"]
    assert turn_detection["threshold"] == 0.7
    assert turn_detection["silence_duration_ms"] == 400
    assert made._session_config()["audio"]["input"]["noise_reduction"] == {"type": "near_field"}
    assert made._session_config()["audio"]["input"]["transcription"]["model"] == "gpt-live-transcribe"


async def test_completed_transcript_reaches_listen(backend: RealtimeVoiceBackend) -> None:
    await backend._consume(
        FakeConn([FakeEvent("conversation.item.input_audio_transcription.completed", transcript="  hello there ")])
    )
    assert await backend.listen() == "hello there"


async def test_stock_filler_and_empties_are_dropped(backend: RealtimeVoiceBackend) -> None:
    await backend._consume(
        FakeConn(
            [
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="Thanks for watching!"),
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="   "),
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="."),
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="turn the light on"),
            ]
        )
    )
    assert backend._heard.qsize() == 1
    assert await backend.listen() == "turn the light on"


async def test_listen_answers_everything_said_since_the_last_turn(backend: RealtimeVoiceBackend) -> None:
    """Talking over a long think or the reply must not put every later answer one turn behind."""
    await backend._consume(
        FakeConn(
            [
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="hello?"),
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="are you there?"),
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="roll for it"),
            ]
        )
    )
    assert await backend.listen() == "hello? are you there? roll for it"
    assert backend._heard.empty()


async def test_listen_returns_a_lone_transcript_unchanged(backend: RealtimeVoiceBackend) -> None:
    await backend._consume(
        FakeConn([FakeEvent("conversation.item.input_audio_transcription.completed", transcript="just this")])
    )
    assert await backend.listen() == "just this"


async def test_speech_events_drive_the_barge_in_signal(backend: RealtimeVoiceBackend) -> None:
    await backend._consume(FakeConn([FakeEvent("input_audio_buffer.speech_started")]))
    assert backend._speech_started.is_set()

    await backend._consume(FakeConn([FakeEvent("input_audio_buffer.speech_stopped")]))
    assert not backend._speech_started.is_set()


async def test_speech_onset_is_reported_only_while_listening(backend: RealtimeVoiceBackend) -> None:
    """speech_started also fires during playback (a barge-in); only the one
    that arrives while listen() is pending is the start of a turn."""
    onsets: list[int] = []
    backend.on_speech_start(lambda: onsets.append(1))

    await backend._consume(FakeConn([FakeEvent("input_audio_buffer.speech_started")]))
    assert onsets == []  # nobody was listening: that was a barge-in, or noise

    pending = asyncio.ensure_future(backend.listen())
    await asyncio.sleep(0)
    await backend._consume(
        FakeConn(
            [
                FakeEvent("input_audio_buffer.speech_started"),
                FakeEvent("conversation.item.input_audio_transcription.completed", transcript="hello"),
            ]
        )
    )
    assert await pending == "hello"
    assert onsets == [1]
    assert not backend._listening


async def test_listen_returns_none_when_nobody_speaks(backend: RealtimeVoiceBackend, monkeypatch) -> None:
    """Silence has to return, not block, or the caller can never re-cue."""
    monkeypatch.setattr("fast_body.voice.realtime_backend._IDLE_TIMEOUT_S", 0.05)
    assert await backend.listen() is None


async def test_cancelling_listen_keeps_the_session(backend: RealtimeVoiceBackend) -> None:
    """DualVoiceBackend cancels the losing task every turn; nothing may be lost."""
    losing = asyncio.ensure_future(backend.listen())
    await asyncio.sleep(0)
    losing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await losing

    await backend._consume(
        FakeConn([FakeEvent("conversation.item.input_audio_transcription.completed", transcript="still here")])
    )
    assert await backend.listen() == "still here"


async def test_pump_resamples_the_mic_to_24k(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # One 100 ms chunk at 16 kHz, past the 50 ms batch size, so it goes out at once.
    media = FakeMedia([np.zeros(1600, dtype=np.float32)], in_rate=16000)
    made = RealtimeVoiceBackend(Config(), FakeRobotAudio(media))
    conn = FakeConn()

    pump = asyncio.ensure_future(made._pump_mic(conn))
    await asyncio.sleep(0.05)
    pump.cancel()

    assert conn.input_audio_buffer.appended
    pcm = np.frombuffer(base64.b64decode(conn.input_audio_buffer.appended[0]), dtype=np.int16)
    assert len(pcm) == 2400  # 100 ms at 24 kHz


async def test_pump_sends_nothing_while_muted(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    media = FakeMedia([np.zeros(1600, dtype=np.float32)], in_rate=16000)
    made = RealtimeVoiceBackend(Config(), FakeRobotAudio(media))
    made._muted.set()
    conn = FakeConn()

    pump = asyncio.ensure_future(made._pump_mic(conn))
    await asyncio.sleep(0.05)
    pump.cancel()

    assert conn.input_audio_buffer.appended == []


async def test_speak_mutes_the_mic_and_drops_what_it_heard(backend: RealtimeVoiceBackend) -> None:
    """No AEC: anything recognized during playback is the robot's own voice."""
    muted_during_playback = False

    async def fake_speak(self, text: str, **_: object) -> None:
        nonlocal muted_during_playback
        muted_during_playback = backend._muted.is_set()
        backend._heard.put_nowait("robot hearing itself")

    backend._barge_in = False
    import fast_body.voice.openai_backend as openai_backend

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(openai_backend.OpenAIVoiceBackend, "speak", fake_speak)
        await backend.speak("hello")

    assert muted_during_playback
    assert not backend._muted.is_set()
    assert backend._heard.empty()


async def test_barge_in_keeps_feeding_the_recognizer(backend: RealtimeVoiceBackend) -> None:
    """With barge-in on, the server needs those frames to hear the interruption."""
    muted_during_playback = None

    async def fake_speak(self, text: str, **_: object) -> None:
        nonlocal muted_during_playback
        muted_during_playback = backend._muted.is_set()

    backend._barge_in = True
    import fast_body.voice.openai_backend as openai_backend

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(openai_backend.OpenAIVoiceBackend, "speak", fake_speak)
        await backend.speak("hello")

    assert muted_during_playback is False


async def test_barge_in_stops_playback_on_server_speech(backend: RealtimeVoiceBackend) -> None:
    playing = asyncio.ensure_future(backend._play_with_barge_in(duration=5.0))
    await asyncio.sleep(0)
    backend._speech_started.set()
    await playing

    assert backend._robot.media.cleared == 1


async def test_barge_in_waits_out_a_quiet_reply(backend: RealtimeVoiceBackend) -> None:
    await backend._play_with_barge_in(duration=0.0)
    assert backend._robot.media.cleared == 0


def test_factory_builds_the_realtime_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VOICE_BACKEND", "realtime")
    made = build_backend(Config(), FakeRobotAudio(FakeMedia()))
    assert isinstance(made, RealtimeVoiceBackend)


def test_realtime_backend_needs_a_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VOICE_BACKEND", "realtime")
    errors = Config().validate()
    assert any("OPENAI_API_KEY" in e and "realtime" in e for e in errors)


async def test_pump_takes_one_channel_of_the_stereo_mic(monkeypatch) -> None:
    """`get_audio_sample()` is (N, 2); flattening it halves the pitch."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    tone = np.sin(2 * np.pi * 440 * np.arange(1600) / 16000).astype(np.float32)
    media = FakeMedia([np.stack([tone, tone], axis=1)], in_rate=16000)
    made = RealtimeVoiceBackend(Config(), FakeRobotAudio(media))
    conn = FakeConn()

    pump = asyncio.ensure_future(made._pump_mic(conn))
    await asyncio.sleep(0.05)
    pump.cancel()

    pcm = np.frombuffer(base64.b64decode(conn.input_audio_buffer.appended[0]), dtype=np.int16)
    assert len(pcm) == 2400  # 100 ms at 24 kHz, not 200 ms of interleaved channels
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float32)))
    assert abs(np.fft.rfftfreq(len(pcm), 1 / 24000)[np.argmax(spectrum)] - 440) < 15


def test_session_config_validates_against_the_sdk(backend: RealtimeVoiceBackend) -> None:
    """A field the SDK renames should fail here, not on the robot."""
    event = SessionUpdateEvent.model_validate(
        {"type": "session.update", "event_id": "test", "session": backend._session_config()}
    )
    assert isinstance(event.session, RealtimeTranscriptionSessionCreateRequest)
    assert event.session.audio.input.turn_detection.silence_duration_ms == 750


async def test_a_rejected_session_config_drops_the_session(backend: RealtimeVoiceBackend) -> None:
    """The server leaves a rejected session open, VAD firing and never transcribing."""
    rejection = FakeEvent(
        "error",
        error=FakeEvent(
            "invalid_request_error",
            param="session.audio.input.turn_detection",
            message="Turn detection is not supported for this transcription model.",
        ),
    )
    with pytest.raises(RuntimeError, match="turn_detection"):
        await backend._consume(FakeConn([rejection]))


async def test_ordinary_errors_do_not_drop_the_session(backend: RealtimeVoiceBackend) -> None:
    await backend._consume(
        FakeConn([FakeEvent("error", error=FakeEvent("server_error", param=None, message="hiccup"))])
    )


async def test_session_is_only_confirmed_by_session_updated(backend: RealtimeVoiceBackend) -> None:
    """Connecting proves nothing — the config is rejected after the socket opens."""
    assert not backend._session_confirmed

    await backend._consume(FakeConn([FakeEvent("session.created")]))
    assert not backend._session_confirmed

    await backend._consume(FakeConn([FakeEvent("session.updated")]))
    assert backend._session_confirmed
