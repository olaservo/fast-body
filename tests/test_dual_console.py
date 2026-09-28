"""Interactive dev console: dual-input race + Escape interrupt.

These cover the parts that need no robot/audio — the `DualVoiceBackend` race and
`FastBodyCore._interruptible` cancellation — with fakes for the inner backend,
console, and movement. The prompt_toolkit layer (`ConsoleInput`) and the live
end-to-end run still need an interactive terminal.
"""

from __future__ import annotations

import asyncio

from fast_body.app import FastBodyCore
from fast_body.voice.dual_backend import DualVoiceBackend


class FakeConsole:
    def __init__(self, line: str | None, delay: float = 0.0) -> None:
        self.interrupt_event = asyncio.Event()
        self._line = line
        self._delay = delay
        self.closed = False
        self.heard: str | None = None

    async def read_line(self) -> str | None:
        await asyncio.sleep(self._delay)
        return self._line

    def echo(self, text: str) -> None:
        self.echoed = text

    def echo_user(self, text: str) -> None:
        self.heard = text

    async def aclose(self) -> None:
        self.closed = True


class FakeMic:
    def __init__(self, text: str | None, delay: float = 0.0) -> None:
        self._text = text
        self._delay = delay
        self.spoken: str | None = None
        self.interrupted = False
        self.closed = False

    async def listen(self) -> str | None:
        await asyncio.sleep(self._delay)
        return self._text

    async def speak(self, text: str, **_: object) -> None:
        self.spoken = text

    def interrupt(self) -> None:
        self.interrupted = True

    async def aclose(self) -> None:
        self.closed = True


# ── DualVoiceBackend ────────────────────────────────────────────────────────


async def test_cancelling_listen_cancels_both_reads():
    """A stop cancels an idle listen; neither read may be left pending."""
    started: list[asyncio.Task] = []

    class Hanging(FakeConsole):
        async def read_line(self) -> str | None:
            started.append(asyncio.current_task())
            await asyncio.Event().wait()
            return None

    class HangingMic(FakeMic):
        async def listen(self) -> str | None:
            started.append(asyncio.current_task())
            await asyncio.Event().wait()
            return None

    listen = asyncio.ensure_future(DualVoiceBackend(HangingMic(None), Hanging(None)).listen())
    await asyncio.sleep(0.01)
    listen.cancel()
    await asyncio.gather(listen, return_exceptions=True)

    assert len(started) == 2
    assert all(task.cancelled() for task in started)


async def test_listen_prefers_typed_over_slow_mic():
    dual = DualVoiceBackend(FakeMic("speech", delay=5.0), FakeConsole("typed", delay=0.0))
    assert await dual.listen() == "typed"


async def test_listen_returns_spoken_when_nothing_typed():
    dual = DualVoiceBackend(FakeMic("speech", delay=0.0), FakeConsole("typed", delay=5.0))
    assert await dual.listen() == "speech"


async def test_speak_echoes_and_delegates_to_inner():
    mic, console = FakeMic(None), FakeConsole(None)
    await DualVoiceBackend(mic, console).speak("hi there")
    assert mic.spoken == "hi there"
    # Raw — the surface adds its own speaker label.
    assert console.echoed == "hi there"


async def test_interrupt_and_aclose_delegate():
    mic, console = FakeMic(None), FakeConsole(None)
    dual = DualVoiceBackend(mic, console)
    dual.interrupt()
    await dual.aclose()
    assert mic.interrupted and mic.closed and console.closed


# ── FastBodyCore._interruptible ─────────────────────────────────────────────


class FakeMovement:
    def __init__(self) -> None:
        self.cleared = False

    def clear_move_queue(self) -> None:
        self.cleared = True


def _core(voice) -> FastBodyCore:
    core = FastBodyCore.__new__(FastBodyCore)  # skip __init__ (needs a robot)
    core.voice = voice
    core.movement = FakeMovement()
    return core


async def test_interruptible_passthrough_without_event():
    core = _core(voice=object())  # no interrupt_event attribute

    async def work():
        return 42

    assert await core._interruptible(work()) == (42, False)


async def test_interruptible_discards_stale_escape():
    """An Escape from the idle/listen phase must NOT cancel the next turn."""
    mic = FakeMic(None)
    mic.interrupt_event = asyncio.Event()
    mic.interrupt_event.set()  # stale press, set before anything is in flight
    core = _core(mic)

    async def work():
        return "done"

    result, interrupted = await core._interruptible(work())
    assert result == "done" and interrupted is False
    assert not core.movement.cleared  # body was never quiesced


async def test_interruptible_cancels_on_escape():
    mic = FakeMic(None)
    mic.interrupt_event = asyncio.Event()
    core = _core(mic)
    cancelled = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "should not finish"

    fut = asyncio.ensure_future(core._interruptible(work()))
    await asyncio.sleep(0.01)
    mic.interrupt_event.set()  # fresh Escape mid-flight
    result, interrupted = await fut

    assert interrupted is True and result is None
    assert cancelled.is_set()
    assert mic.interrupted and core.movement.cleared
    assert not mic.interrupt_event.is_set()  # cleared for the next turn


# ── transcript filtering ──────────────────────────────────────────────────
def test_stock_subtitle_phrases_are_discarded():
    """Whisper emits these for silence; they came from its subtitle training data."""
    from fast_body.voice.openai_backend import _is_hallucination

    for phrase in ("Thank you.", "Thanks for watching!", "you", " ♪ ", "."):
        assert _is_hallucination(phrase), phrase


def test_real_speech_survives_the_filter():
    from fast_body.voice.openai_backend import _is_hallucination

    for phrase in ("thank you for that", "hello", "what do you see?", "no"):
        assert not _is_hallucination(phrase), phrase


# ── mic flush ─────────────────────────────────────────────────────────────
class _BufferedMedia:
    """Mic buffer holding a robot utterance's worth of chunks."""

    def __init__(self, chunks: int) -> None:
        self.remaining = chunks

    def get_audio_sample(self):
        if self.remaining <= 0:
            return None
        self.remaining -= 1
        return [0.0]


def _backend_with(media):
    from fast_body.voice.openai_backend import OpenAIVoiceBackend

    b = OpenAIVoiceBackend.__new__(OpenAIVoiceBackend)  # skip __init__ (needs openai/webrtcvad)
    b._robot = type("R", (), {"media": media})()
    return b


def test_flush_drains_a_long_utterance():
    """A partial drain leaves the robot's own voice at the head of the next turn."""
    import numpy as np

    media = _BufferedMedia(chunks=500)  # far more than the old 64-chunk cap
    b = _backend_with(media)
    b._leftover = np.ones(10, dtype=np.float32)
    b._flush_mic()
    assert media.remaining == 0
    assert b._leftover.size == 0


def test_flush_is_bounded_if_the_buffer_never_empties():
    from fast_body.voice.openai_backend import _MAX_FLUSH_CHUNKS

    media = _BufferedMedia(chunks=_MAX_FLUSH_CHUNKS + 1000)
    _backend_with(media)._flush_mic()
    assert media.remaining == 1000  # stopped at the cap rather than spinning


async def test_warm_up_delegates_to_the_mic_backend():
    mic, console = FakeMic(None), FakeConsole(None)
    warmed = []

    async def warm():
        warmed.append(True)

    mic.warm_up = warm  # type: ignore[attr-defined]
    await DualVoiceBackend(mic, console).warm_up()
    await DualVoiceBackend(FakeMic(None), console).warm_up()  # a mic without one is fine

    assert warmed == [True]
