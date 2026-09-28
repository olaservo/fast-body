"""The discrete backend's mic path — channel handling on the SDK audio buffer."""

from __future__ import annotations

import numpy as np

from fast_body.config import Config
from fast_body.voice.openai_backend import OpenAIVoiceBackend, _to_mono
from fast_body.voice.realtime_backend import RealtimeVoiceBackend
from tests.test_realtime_backend import FakeMedia, FakeRobotAudio


def _tone(hz: float, samples: int, rate: int = 16000) -> np.ndarray:
    return np.sin(2 * np.pi * hz * np.arange(samples) / rate).astype(np.float32)


def _peak_hz(x: np.ndarray, rate: int = 16000) -> float:
    spectrum = np.abs(np.fft.rfft(x.astype(np.float32)))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[np.argmax(spectrum)])


def test_stereo_buffer_keeps_its_pitch_and_length() -> None:
    """Interleaving the channels instead would read back an octave down."""
    mono = _tone(440, 16000)
    stereo = np.stack([mono, mono], axis=1)

    got = _to_mono(stereo)

    assert got.shape == (16000,)
    assert abs(_peak_hz(got) - 440) < 5


def test_channels_first_buffers_are_transposed() -> None:
    mono = _tone(440, 16000)
    got = _to_mono(np.stack([mono, mono], axis=0))
    assert got.shape == (16000,)


def test_mono_buffers_pass_through() -> None:
    mono = _tone(440, 16000)
    assert _to_mono(mono).shape == (16000,)


async def test_next_frame_holds_30_ms_of_real_audio(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mono = _tone(440, 1600)  # 100 ms
    media = FakeMedia([np.stack([mono, mono], axis=1)], in_rate=16000)
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media))

    frame, _ = await backend._next_frame()

    assert len(frame) == 480  # 30 ms at 16 kHz
    assert abs(_peak_hz(frame) - 440) < 40  # short window, so a loose bin


async def test_listen_reports_speech_onset_before_the_transcript(monkeypatch) -> None:
    """The body switches to the listening pose when the VAD first triggers,
    not when the transcript lands seconds later."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(FakeMedia([], in_rate=16000)))
    backend._min_speech_frames = 1
    backend._end_silence_frames = 1
    backend._min_rms = 0.0
    loud = (np.ones(480) * 8000).astype(np.int16)
    frames = iter([(loud, False), (loud, True), (loud, True), (loud, False)])

    async def next_frame():
        return next(frames)

    order: list[str] = []

    async def transcribe(_pcm):
        order.append("transcribe")
        return "hello"

    monkeypatch.setattr(backend, "_next_frame", next_frame)
    monkeypatch.setattr(backend, "_transcribe", transcribe)
    monkeypatch.setattr(backend, "_flush_mic", lambda: None)
    monkeypatch.setattr(backend, "_ensure_started", lambda: None)
    backend.on_speech_start(lambda: order.append("onset"))

    assert await backend.listen() == "hello"
    assert order == ["onset", "transcribe"]


def test_tui_speaks_under_the_realtime_backend(monkeypatch) -> None:
    """The TUI is typed, so it wants the speaking half without a session."""
    from fast_body.app import FastBodyCore

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VOICE_BACKEND", "realtime")
    core = FastBodyCore.__new__(FastBodyCore)
    core.config = Config()
    core.robot = FakeRobotAudio(FakeMedia())

    voice = core._build_tui_voice()

    assert isinstance(voice, OpenAIVoiceBackend)
    assert not isinstance(voice, RealtimeVoiceBackend)


def test_tui_stays_silent_without_a_key(monkeypatch) -> None:
    from fast_body.app import FastBodyCore

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("VOICE_BACKEND", "realtime")
    core = FastBodyCore.__new__(FastBodyCore)
    core.config = Config()
    core.robot = FakeRobotAudio(FakeMedia())

    assert core._build_tui_voice() is None


def test_tui_never_listens_for_barge_in(monkeypatch) -> None:
    """Barge-in reads the mic every frame; TUI input is typed, so it must be off."""
    from fast_body.app import FastBodyCore

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ENABLE_BARGE_IN", "true")
    core = FastBodyCore.__new__(FastBodyCore)
    core.config = Config()
    core.robot = FakeRobotAudio(FakeMedia())

    assert core.config.enable_barge_in is True
    assert core._build_tui_voice()._barge_in is False


# ── streaming TTS ────────────────────────────────────────────────────────────


def _pcm_bytes(x: np.ndarray) -> bytes:
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16).tobytes()


def test_streamed_resampling_matches_doing_it_all_at_once() -> None:
    """Chunk seams are where a naive streaming resampler clicks."""
    from scipy.signal import resample_poly

    from fast_body.voice.openai_backend import _StreamResampler

    t = np.arange(24000) / 24000
    sweep = np.sin(2 * np.pi * (200 + 3000 * t) * t).astype(np.float32)
    whole = resample_poly(sweep, 2, 3)

    resampler = _StreamResampler(24000, 16000)
    data = _pcm_bytes(sweep)
    pieces = [resampler.push(data[i : i + 3001]) for i in range(0, len(data), 3001)]
    pieces.append(resampler.flush())
    streamed = np.concatenate([p for p in pieces if p.size])

    n = min(len(whole), len(streamed))
    assert abs(len(whole) - len(streamed)) <= 2
    error = streamed[:n] - whole[:n]
    snr = 20 * np.log10(np.sqrt(np.mean(whole[:n] ** 2)) / np.sqrt(np.mean(error**2)))
    assert snr > 60, f"streaming resample only {snr:.1f} dB — the seams are audible"


def test_stream_resampler_passes_through_at_matching_rates() -> None:
    from fast_body.voice.openai_backend import _StreamResampler

    resampler = _StreamResampler(16000, 16000)
    tone = _tone(440, 800)
    out = np.concatenate([resampler.push(_pcm_bytes(tone)), resampler.flush()])
    assert len(out) == 800
    assert abs(_peak_hz(out) - 440) < 20


def test_stream_resampler_survives_a_split_sample() -> None:
    """A chunk boundary can fall between the two bytes of one sample."""
    from fast_body.voice.openai_backend import _StreamResampler

    resampler = _StreamResampler(24000, 16000)
    data = _pcm_bytes(_tone(440, 6000, rate=24000))
    out = [resampler.push(data[:1001]), resampler.push(data[1001:]), resampler.flush()]
    joined = np.concatenate([p for p in out if p.size])
    assert abs(len(joined) - 4000) <= 2  # 6000 at 24 kHz -> 4000 at 16 kHz


class _FakeSpeechStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def iter_bytes(self, chunk_size=None):
        for chunk in self._chunks:
            yield chunk


class _FakeSpeechClient:
    """Just enough of AsyncOpenAI for `speak()`."""

    def __init__(self, chunks: list[bytes]) -> None:
        outer = self

        class _Streaming:
            def create(self, **kwargs):
                outer.kwargs = kwargs
                return _FakeSpeechStream(chunks)

        class _Speech:
            with_streaming_response = _Streaming()

        class _Audio:
            speech = _Speech()

        self.audio = _Audio()
        self.kwargs: dict = {}


async def test_speak_pushes_audio_before_the_reply_finishes(monkeypatch) -> None:
    """The point of streaming: the robot starts talking mid-synthesis."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class Media16k(FakeMedia):
        def get_output_audio_samplerate(self) -> int:
            return 16000  # the robot's rate, so resampling is actually exercised

    media = Media16k()
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media))
    backend._barge_in = False

    speech = _pcm_bytes(_tone(220, 24000, rate=24000))  # 1s at 24 kHz
    chunks = [speech[i : i + 8192] for i in range(0, len(speech), 8192)]
    backend._speaker._client = _FakeSpeechClient(chunks)

    await backend.speak("hello there")

    assert len(media.pushed) > 1, "pushed once — that is the old buffer-it-all behaviour"
    total = sum(p.size for p in media.pushed)
    assert abs(total - 16000) <= 4  # 1s of 24 kHz becomes 1s of 16 kHz


async def test_speak_ignores_empty_text(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    media = FakeMedia()
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media))
    backend._speaker._client = _FakeSpeechClient([b"never used"])

    await backend.speak("   ")

    assert media.pushed == []


class _Media16k(FakeMedia):
    def get_output_audio_samplerate(self) -> int:
        return 16000


async def _speak_tone(monkeypatch, seconds: float, lead_in: str = "0.6") -> _Media16k:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TTS_LEAD_IN_S", lead_in)
    media = _Media16k()
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media))
    backend._barge_in = False

    speech = _pcm_bytes(_tone(220, int(24000 * seconds), rate=24000))
    backend._speaker._client = _FakeSpeechClient([speech[i : i + 8192] for i in range(0, len(speech), 8192)])
    await backend.speak("hello there")
    return media


async def test_playback_waits_for_a_lead_in(monkeypatch) -> None:
    """Starting the player on 150 ms drains it mid-word — heard on the robot."""
    media = await _speak_tone(monkeypatch, seconds=2.0)

    assert media.pushed, "nothing played"
    assert media.pushed[0].size >= int(0.6 * 16000), (
        f"first push was only {media.pushed[0].size / 16000:.2f}s — the player will run dry"
    )
    assert len(media.pushed) > 1, "still buffering the whole reply"


async def test_a_reply_shorter_than_the_lead_in_still_gets_played(monkeypatch) -> None:
    media = await _speak_tone(monkeypatch, seconds=0.2)

    total = sum(p.size for p in media.pushed)
    assert abs(total - int(0.2 * 16000)) <= 8, f"{total} samples played, expected ~3200"


async def test_lead_in_is_tunable(monkeypatch) -> None:
    media = await _speak_tone(monkeypatch, seconds=2.0, lead_in="0.1")
    assert media.pushed[0].size < int(0.6 * 16000)


async def test_delivery_reaches_the_tts_call(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TTS_DELIVERY", "flat and weary")
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(_Media16k()))
    backend._barge_in = False
    backend._speaker._client = _FakeSpeechClient([_pcm_bytes(_tone(220, 24000, rate=24000))])

    await backend.speak("hello")

    assert backend._speaker._client.kwargs["instructions"] == "flat and weary"


async def test_no_delivery_means_no_instructions(monkeypatch) -> None:
    """tts-1 and tts-1-hd reject the field, so it only goes when a card set one."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("TTS_DELIVERY", raising=False)
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "default")
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(_Media16k()))
    backend._barge_in = False
    backend._speaker._client = _FakeSpeechClient([_pcm_bytes(_tone(220, 24000, rate=24000))])

    await backend.speak("hello")

    assert "instructions" not in backend._speaker._client.kwargs


async def test_warm_up_runs_one_discarded_synthesis(monkeypatch, caplog) -> None:
    from tests.test_speaker import _ToneSpeaker

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    media = FakeMedia([], in_rate=16000)
    speaker = _ToneSpeaker(24000, 0.1)
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media), speaker=speaker)

    with caplog.at_level("INFO", logger="fast_body.voice.openai_backend"):
        await backend.warm_up()

    assert [r[0] for r in speaker.requests] == ["Ready."]
    assert media.playing and media.recording  # the audio path is open for the first listen
    assert "TTS warm-up took" in caplog.text
    # The synthesis is never played; what reaches the player is a beat of
    # silence, so its own start is paid here rather than on the first word.
    assert len(media.pushed) == 1
    assert media.pushed[0].shape == (int(0.2 * 24000),)
    assert media.pushed[0].dtype == np.float32 and not media.pushed[0].any()
    assert "playback warm-up took" in caplog.text


async def test_warm_up_failure_is_logged_not_raised(monkeypatch, caplog) -> None:
    class BrokenSpeaker:
        sample_rate = 24000

        async def stream(self, text, *, voice=None, delivery=None):
            raise RuntimeError("no network")
            yield b""  # makes this an async generator

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(FakeMedia([], in_rate=16000)), speaker=BrokenSpeaker())

    with caplog.at_level("WARNING", logger="fast_body.voice.openai_backend"):
        await backend.warm_up()

    assert "TTS warm-up failed: no network" in caplog.text


async def test_warm_up_silence_matches_the_output_channel_count(monkeypatch) -> None:
    from tests.test_speaker import _ToneSpeaker

    class Stereo(FakeMedia):
        def get_output_channels(self) -> int:
            return 2

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    media = Stereo()
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media), speaker=_ToneSpeaker(24000, 0.1))

    await backend.warm_up()

    assert media.pushed[0].shape == (int(0.2 * 24000), 2)


async def test_warm_up_pushes_nothing_without_an_audio_device(monkeypatch, caplog) -> None:
    """The SDK's media reports -1 for the rates when it came up with no audio."""
    from tests.test_speaker import _ToneSpeaker

    class NoAudio(FakeMedia):
        def get_output_audio_samplerate(self) -> int:
            return -1

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    media = NoAudio()
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(media), speaker=_ToneSpeaker(24000, 0.1))

    with caplog.at_level("INFO", logger="fast_body.voice.openai_backend"):
        await backend.warm_up()

    assert media.pushed == []
    assert "playback warm-up" not in caplog.text


async def test_warm_up_survives_a_player_that_refuses_the_push(monkeypatch) -> None:
    from tests.test_speaker import _ToneSpeaker

    class Refuses(FakeMedia):
        def push_audio_sample(self, audio) -> None:
            raise RuntimeError("pipeline not ready")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(Refuses()), speaker=_ToneSpeaker(24000, 0.1))

    await backend.warm_up()  # logged, not raised
