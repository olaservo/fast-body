"""The brain timeout must end the turn even when the provider eats the cancel.

fast-agent's OpenAI-compatible provider catches CancelledError and returns an
empty assistant message. Under `asyncio.wait_for` that reads as a normal
result: no TimeoutError, no log line, the robot "speaks" an empty string and
the history keeps the failed turn. Observed on the robot 2026-09-04.
"""

import asyncio
from types import SimpleNamespace

from fast_body.app import FastBodyCore


class FakeVoice:
    def __init__(self, lines):
        self.lines = list(lines)
        self.spoken: list[str] = []

    async def listen(self):
        return self.lines.pop(0) if self.lines else None

    async def speak(self, text):
        self.spoken.append(text)


class FakeMovement:
    def current_pose(self):
        return None, None

    def set_listening(self, _on):
        pass

    def queue_move(self, _move):
        pass

    def clear_move_queue(self):
        pass


class SwallowingBrain:
    """A brain whose send() appends to the history, then survives its own cancel."""

    def __init__(self):
        self.agent = SimpleNamespace(name="body", message_history=[])
        self.cancelled = False

    def resolve_agent(self, _name):
        return self.agent

    async def send(self, text):
        self.agent.message_history.append(SimpleNamespace(role="user", content=text))
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.cancelled = True
            return ""  # what fast-agent does: an empty reply, no exception
        return "never"


def _core(voice) -> FastBodyCore:
    core = FastBodyCore.__new__(FastBodyCore)  # skip __init__ (needs a robot)
    core.voice = voice
    core.movement = FakeMovement()
    core.gaze = None
    core.web_hub = None
    core.stop_event = asyncio.Event()
    core._stopping = asyncio.Event()
    core._turn_speech = None
    core.config = SimpleNamespace(voice_backend="text", brain_timeout_s=0.05)
    return core


async def test_a_swallowed_cancel_still_counts_as_a_timeout():
    voice = FakeVoice(["hello"])  # then EOF, which ends a text session
    brain = SwallowingBrain()
    core = _core(voice)

    await core._converse(brain)

    assert brain.cancelled, "the turn was never cancelled"
    assert voice.spoken == ["Sorry, I lost my train of thought there."]
    assert brain.agent.message_history == [], "the failed turn stayed in the history"


class StuckBrain:
    """A brain whose send() never returns, cancelled or not.

    A provider request stalled on the wire did this on the robot 2026-09-25:
    the cancel went in, the task never settled, and `_converse` sat on the
    await of it with no log line and no reply for the rest of the session.
    """

    def __init__(self):
        self.agent = SimpleNamespace(name="body", message_history=[])
        self.cancelled = False

    def resolve_agent(self, _name):
        return self.agent

    async def send(self, text):
        self.agent.message_history.append(SimpleNamespace(role="user", content=text))
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            self.cancelled = True
        # Still not done: the request is stalled. A second cancel (the loop
        # shutting down) is honoured so the test process can exit.
        await asyncio.sleep(5)
        return "never"


async def test_a_turn_that_ignores_its_cancel_is_reported_and_left_behind(monkeypatch):
    import fast_body.app as app_module

    monkeypatch.setattr(app_module, "_CANCEL_GRACE_S", 0.05)
    voice = FakeVoice(["hello"])
    brain = StuckBrain()
    core = _core(voice)

    await asyncio.wait_for(core._converse(brain), timeout=2)

    assert brain.cancelled, "the turn was never cancelled"
    assert voice.spoken == ["Sorry, I lost my train of thought there."]
    assert brain.agent.message_history == [], "the failed turn stayed in the history"
