"""ConsoleInput driven through prompt_toolkit's pipe-input test harness.

Exercises the *real* prompt_toolkit ``PromptSession`` + key bindings — typed
line, bare Escape, EOF — plus a real ``DualVoiceBackend`` race, all headlessly
(no TTY). The robot, audio, and MuJoCo sim still need a live terminal; see the
README's ``--console`` section.
"""

from __future__ import annotations

import asyncio
import threading

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from fast_body.voice.console_input import ConsoleInput
from fast_body.voice.dual_backend import DualVoiceBackend


async def test_typed_line_is_read():
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        ci = ConsoleInput()
        await ci.start(threading.Event())
        inp.send_text("hello world\n")
        line = await asyncio.wait_for(ci.read_line(), timeout=3)
        await ci.aclose()
    assert line == "hello world"


async def test_blank_lines_are_ignored():
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        ci = ConsoleInput()
        await ci.start(threading.Event())
        inp.send_text("\n   \nreal\n")  # two blank submits, then a real one
        line = await asyncio.wait_for(ci.read_line(), timeout=3)
        await ci.aclose()
    assert line == "real"


async def test_escape_sets_interrupt_event():
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        ci = ConsoleInput()
        await ci.start(threading.Event())
        assert not ci.interrupt_event.is_set()
        inp.send_text("\x1b")  # bare ESC
        await asyncio.wait_for(ci.interrupt_event.wait(), timeout=3)
        await ci.aclose()
    assert ci.interrupt_event.is_set()


async def test_eof_sets_stop_event_and_returns_none():
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        ci = ConsoleInput()
        stop = threading.Event()
        await ci.start(stop)
        inp.close()  # EOF on the pipe (Ctrl-D equivalent)
        line = await asyncio.wait_for(ci.read_line(), timeout=3)
        await ci.aclose()
    assert line is None
    assert stop.is_set()


class _SilentMic:
    """A mic backend that never yields, so the typed path must win the race."""

    async def listen(self) -> str | None:
        await asyncio.sleep(3600)
        return None

    async def speak(self, text: str, **_: object) -> None: ...

    def interrupt(self) -> None: ...

    async def aclose(self) -> None: ...


async def test_dual_backend_reads_typed_via_real_console():
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        dual = DualVoiceBackend(_SilentMic(), ConsoleInput())
        await dual.start(threading.Event())
        inp.send_text("typed via dual\n")
        res = await asyncio.wait_for(dual.listen(), timeout=3)
        await dual.aclose()
    assert res == "typed via dual"
