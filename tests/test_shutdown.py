"""A stop request ends the run promptly, whatever the loop is waiting on.

The daemon sends SIGINT and force-kills 20 s later. Three stops in a row on
2026-09-26 hit that kill with no output from the app: the KeyboardInterrupt
broke the loop out from under its tasks, so the AnyIO worker thread one of
them owned was never told to stop, and interpreter exit joined it for good.
"""

import asyncio
import threading
from types import SimpleNamespace

from fast_body import app as app_module
from fast_body.app import FastBodyCore, _install_stop_signals
from fast_body.voice.cast import Cast, CastMember, CastVoiceBackend


class NeverSpeaks:
    """A voice whose listen() waits like the realtime backend's 30 s idle wait."""

    def __init__(self):
        self.cancelled = False

    async def listen(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return "never"


class LongReply:
    """A voice whose speak() plays like a reply the stop landed in the middle of."""

    def __init__(self):
        self.started: list[str] = []
        self.cancelled = False
        self.interrupted = 0

    async def listen(self):
        if self.started:  # one turn, then quiet
            await asyncio.sleep(3600)
        return "hello"

    async def speak(self, text, *, voice=None, delivery=None):
        self.started.append(text)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    def interrupt(self):
        self.interrupted += 1


def _core(voice) -> FastBodyCore:
    core = FastBodyCore.__new__(FastBodyCore)  # skip __init__ (needs a robot)
    core.voice = voice
    core.stop_event = threading.Event()
    core._stopping = asyncio.Event()
    core._turn_speech = None
    return core


async def test_a_stop_request_ends_an_idle_listen_at_once():
    voice = NeverSpeaks()
    core = _core(voice)
    listen = asyncio.ensure_future(core._listen_or_stop())
    await asyncio.sleep(0.01)
    assert not listen.done()

    core.request_stop("test")

    assert await asyncio.wait_for(listen, timeout=1.0) is None
    assert voice.cancelled
    assert core.stop_event.is_set()


async def test_a_listen_that_returns_first_is_handed_back():
    class Speaks:
        async def listen(self):
            return "hello"

    core = _core(Speaks())
    assert await core._listen_or_stop() == "hello"
    assert not core.stop_event.is_set()


async def test_the_turn_loop_leaves_when_stopped_mid_listen():
    voice = NeverSpeaks()
    core = _core(voice)
    core.movement = SimpleNamespace(
        current_pose=lambda: (None, None), set_listening=lambda _on: None, queue_move=lambda _m: None
    )
    core.gaze = None
    core.web_hub = None
    core.config = SimpleNamespace(voice_backend="realtime", brain_timeout_s=1.0)
    brain = SimpleNamespace(resolve_agent=lambda _n: None)

    converse = asyncio.ensure_future(core._converse(brain))
    await asyncio.sleep(0.01)
    core.request_stop("test")

    await asyncio.wait_for(converse, timeout=1.0)


async def test_a_stop_request_cuts_a_reply_mid_utterance():
    voice = LongReply()
    core = _core(voice)
    speak = asyncio.ensure_future(core._speak_or_stop("a long reply"))
    await asyncio.sleep(0.01)
    assert not speak.done()

    core.request_stop("test")

    await asyncio.wait_for(speak, timeout=1.0)
    assert voice.cancelled
    assert voice.interrupted == 1


async def test_a_stop_request_drops_the_rest_of_a_cast_script():
    inner = LongReply()
    cast = Cast([CastMember("Lila", "shimmer")])
    core = _core(CastVoiceBackend(inner, cast))
    speak = asyncio.ensure_future(core._speak_or_stop("The door opens. [Lila] Who's there? [narrator] Silence."))
    await asyncio.sleep(0.01)

    core.request_stop("test")

    await asyncio.wait_for(speak, timeout=1.0)
    assert inner.started == ["The door opens."]
    assert inner.interrupted == 1


async def test_a_reply_that_finishes_first_is_left_alone():
    class Quick:
        def __init__(self):
            self.interrupted = 0

        async def speak(self, text, *, voice=None, delivery=None):
            return None

        def interrupt(self):
            self.interrupted += 1

    voice = Quick()
    core = _core(voice)
    await core._speak_or_stop("done already")
    assert voice.interrupted == 0
    assert not core.stop_event.is_set()


async def test_the_turn_loop_leaves_when_stopped_mid_reply():
    voice = LongReply()
    core = _core(voice)
    core.movement = SimpleNamespace(
        current_pose=lambda: (None, None), set_listening=lambda _on: None, queue_move=lambda _m: None
    )
    core.gaze = None
    core.web_hub = None
    core.config = SimpleNamespace(voice_backend="realtime", brain_timeout_s=1.0)

    async def send(_text):
        return "a long reply"

    brain = SimpleNamespace(resolve_agent=lambda _n: None, send=send)

    converse = asyncio.ensure_future(core._converse(brain))
    await asyncio.sleep(0.05)
    assert voice.started == ["a long reply"]
    core.request_stop("test")

    await asyncio.wait_for(converse, timeout=1.0)
    assert voice.cancelled and voice.interrupted == 1


def test_signal_handlers_request_a_stop_and_arm_the_watchdog(monkeypatch):
    armed: list[float] = []
    monkeypatch.setattr(app_module, "_hard_exit_later", lambda s: armed.append(s))
    exits: list[int] = []
    monkeypatch.setattr(app_module, "_hard_exit", lambda code: exits.append(code))

    loop = asyncio.new_event_loop()
    try:
        core = _core(NeverSpeaks())
        # Drive the handler the way the loop would, without raising a real signal.
        handlers: dict = {}
        monkeypatch.setattr(
            loop, "add_signal_handler", lambda signum, cb, *args: handlers.__setitem__(signum, (cb, args))
        )
        monkeypatch.setattr(loop, "remove_signal_handler", lambda signum: handlers.pop(signum))
        restore = _install_stop_signals(loop, core)
        assert handlers, "no handler was installed"

        cb, args = next(iter(handlers.values()))
        cb(*args)
        assert core.stop_event.is_set() and core._stopping.is_set()
        assert armed == [app_module._SHUTDOWN_DEADLINE_S]
        assert exits == []

        cb(*args)  # a second signal exits at once
        assert exits == [1]

        restore()
        assert handlers == {}
    finally:
        loop.close()


def test_the_watchdog_is_a_daemon_thread_and_fires(monkeypatch):
    fired = threading.Event()
    monkeypatch.setattr(app_module, "_hard_exit", lambda code: fired.set())
    timer = app_module._hard_exit_later(0.05)
    assert timer.daemon
    assert fired.wait(1.0)
