"""A turn's text is spoken as it arrives, ahead of each tool call, and never twice.

Narration written before a tool call is spoken before the call runs, not
after it.
"""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from types import SimpleNamespace

from fast_agent.agents.tool_runner import ToolRunnerHooks

from fast_body.app import FastBodyCore, _turn_spoken_text
from fast_body.config import Config
from fast_body.turn_speech import TurnSpeech
from fast_body.voice.cast import Cast, CastMember, CastVoiceBackend
from fast_body.voice.dual_backend import DualVoiceBackend
from fast_body.web.hub import WebChatHub
from tests.conftest import FakeRobot
from tests.test_cast import _RecordingVoice
from tests.test_turn_timeout import FakeVoice, _core

BEATS = ["The door creaks open.", "You fail the roll.", "The thing inside turns to look at you."]


def _assistant(text, tool=False):
    return SimpleNamespace(
        role="assistant",
        content=[SimpleNamespace(text=text)],
        tool_calls={"call-1": object()} if tool else None,
    )


class ToolTurnBrain:
    """A brain whose turn is text, tool, text, tool, text, driving the hooks as fast-agent's ToolRunner does.

    Each assistant message is in the history before `before_tool_call` runs,
    the order the OpenAI provider gives (`history.set` before the response is
    returned). Tool runs are recorded in `events` beside what the voice
    records, for ordering assertions.
    """

    def __init__(self, beats, events=None, tool_takes=0.0):
        self.agent = SimpleNamespace(name="body", message_history=[], tool_runner_hooks=None)
        self.beats = list(beats)
        self.events = events if events is not None else []
        self.tool_takes = tool_takes
        self.cancelled = False

    def resolve_agent(self, _name):
        return self.agent

    async def send(self, text):
        history = self.agent.message_history
        history.append(SimpleNamespace(role="user", content=[SimpleNamespace(text=text)]))
        try:
            for i, beat in enumerate(self.beats):
                final = i == len(self.beats) - 1
                message = _assistant(beat, tool=not final)
                history.append(message)
                if final:
                    return beat
                hooks = self.agent.tool_runner_hooks
                if hooks is not None and hooks.before_tool_call is not None:
                    await hooks.before_tool_call(None, message)
                await asyncio.sleep(self.tool_takes)
                self.events.append(("tool", i))
                history.append(SimpleNamespace(role="user", content=[SimpleNamespace(text="<tool result>")]))
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ""


class EventVoice(FakeVoice):
    """Records each utterance in a shared event list; `takes` is how long one lasts."""

    def __init__(self, lines, events, takes=0.0):
        super().__init__(lines)
        self.events = events
        self.takes = takes
        self.interrupted = 0

    async def speak(self, text):
        await asyncio.sleep(self.takes)
        self.spoken.append(text)
        self.events.append(("speak", text))

    def interrupt(self):
        self.interrupted += 1


def _speaking_core(voice, cues=None):
    """A core with mid-turn speech on, as run() builds it with the flag set."""
    core = _core(voice)
    cues = cues if cues is not None else []
    core._turn_speech = TurnSpeech(
        voice,
        stopping=core._stopping,
        on_start=lambda: cues.append("speaking"),
        on_end=lambda: cues.append("thinking"),
    )
    return core


# ── the turn loop ─────────────────────────────────────────────────────────
async def test_each_block_is_spoken_ahead_of_its_tool_call_and_not_again():
    events, cues = [], []
    voice = EventVoice(["hello"], events)  # then EOF, which ends a text session
    brain = ToolTurnBrain(BEATS, events)
    core = _speaking_core(voice, cues)
    core._turn_speech.install(brain.agent)

    await core._converse(brain)

    assert voice.spoken == BEATS
    assert events == [
        ("speak", BEATS[0]),
        ("tool", 0),
        ("speak", BEATS[1]),
        ("tool", 1),
        ("speak", BEATS[2]),
    ]
    # The body is told about each block, and goes back to thinking after it.
    assert cues == ["speaking", "thinking"] * 2


async def test_with_the_flag_off_the_turn_is_spoken_once_at_the_end():
    voice = FakeVoice(["hello"])
    brain = ToolTurnBrain(BEATS)
    core = _core(voice)  # no TurnSpeech, as run() leaves it when SPEAK_BETWEEN_TOOLS is off

    await core._converse(brain)

    assert voice.spoken == ["\n\n".join(BEATS)]
    assert brain.agent.tool_runner_hooks is None


def test_the_flag_decides_whether_mid_turn_speech_is_built(monkeypatch):
    core = FastBodyCore(FakeRobot(), owns_robot=False, stop_event=threading.Event(), config=Config())
    core.voice = FakeVoice([])
    monkeypatch.setenv("SPEAK_BETWEEN_TOOLS", "false")
    core.config = Config()
    assert core._build_turn_speech() is None
    monkeypatch.delenv("SPEAK_BETWEEN_TOOLS")
    core.config = Config()
    assert isinstance(core._build_turn_speech(), TurnSpeech)


async def test_a_turn_without_tool_calls_is_unchanged():
    events = []
    voice = EventVoice(["hello"], events)
    brain = ToolTurnBrain(["Hello there."], events)
    core = _speaking_core(voice)
    core._turn_speech.install(brain.agent)

    await core._converse(brain)

    assert voice.spoken == ["Hello there."]
    assert events == [("speak", "Hello there.")]


async def test_a_line_the_turn_repeats_is_spoken_again_at_the_end():
    voice = FakeVoice(["hello"])
    brain = ToolTurnBrain(["Roll.", "Roll."])
    core = _speaking_core(voice)
    core._turn_speech.install(brain.agent)

    await core._converse(brain)

    assert voice.spoken == ["Roll.", "Roll."]


def test_the_end_of_turn_text_leaves_out_what_was_spoken_once_per_match():
    agent = SimpleNamespace(
        message_history=[
            _assistant("Roll.", tool=True),
            SimpleNamespace(role="user", content=[SimpleNamespace(text="<tool result>")]),
            _assistant("Roll."),
        ]
    )
    assert _turn_spoken_text(agent, 0, "Roll.", Counter({"Roll.": 1})) == "Roll."
    assert _turn_spoken_text(agent, 0, "Roll.", Counter({"Roll.": 2})) == ""
    assert _turn_spoken_text(agent, 0, "Roll.") == "Roll.\n\nRoll."


# ── the same path as the final reply ──────────────────────────────────────
async def test_the_page_shows_each_block_as_it_is_spoken():
    hub, inner = WebChatHub(), _RecordingVoice()
    speech = TurnSpeech(DualVoiceBackend(inner, hub))

    await speech.speak_message(_assistant(BEATS[0], tool=True))
    await speech.speak_message(_assistant(BEATS[1], tool=True))

    assert [(m["role"], m["text"]) for m in hub.transcript()] == [("robot", BEATS[0]), ("robot", BEATS[1])]
    assert [s[0] for s in inner.spoken] == [BEATS[0], BEATS[1]]


async def test_a_block_goes_through_the_cast():
    inner = _RecordingVoice()
    cast = Cast([CastMember("Innkeeper", "onyx", "slow and gravelly")])
    speech = TurnSpeech(CastVoiceBackend(inner, cast))

    await speech.speak_message(_assistant("He looks up. [Innkeeper] Nobody's been out there.", tool=True))

    assert inner.spoken == [
        ("He looks up.", None, None),
        ("Nobody's been out there.", "onyx", "slow and gravelly"),
    ]


async def test_a_message_with_no_text_is_passed_over():
    voice = _RecordingVoice()
    speech = TurnSpeech(voice)
    await speech.speak_message(SimpleNamespace(role="assistant", content=[], tool_calls={"c": object()}))
    assert voice.spoken == [] and speech.quiet_for() is None


async def test_install_keeps_an_existing_hook_and_installs_once():
    seen = []

    async def previous(runner, request):
        seen.append("previous")

    agent = SimpleNamespace(tool_runner_hooks=ToolRunnerHooks(before_tool_call=previous))
    voice = _RecordingVoice()
    speech = TurnSpeech(voice)
    speech.install(agent)
    speech.install(agent)

    await agent.tool_runner_hooks.before_tool_call(None, _assistant("Hm.", tool=True))

    assert seen == ["previous"]
    assert [s[0] for s in voice.spoken] == ["Hm."]


# ── stops and the brain timeout ───────────────────────────────────────────
async def test_a_stop_during_a_tool_call_cancels_the_turn_and_nothing_more_is_spoken():
    events = []
    voice = EventVoice(["hello"], events)
    brain = ToolTurnBrain(BEATS, events, tool_takes=5)
    core = _speaking_core(voice)
    core._turn_speech.install(brain.agent)

    converse = asyncio.ensure_future(core._converse(brain))
    await asyncio.sleep(0.05)  # the first block is out; the tool is running
    core.request_stop("test")
    await asyncio.wait_for(converse, timeout=1.0)

    assert brain.cancelled
    assert voice.spoken == [BEATS[0]]
    assert brain.agent.message_history == [], "the cancelled turn stayed in the history"


async def test_a_stop_while_a_block_is_speaking_cuts_it():
    events = []
    voice = EventVoice(["hello"], events, takes=5)
    brain = ToolTurnBrain(BEATS, events)
    core = _speaking_core(voice)
    core._turn_speech.install(brain.agent)

    converse = asyncio.ensure_future(core._converse(brain))
    await asyncio.sleep(0.05)  # the first block is being spoken
    core.request_stop("test")
    await asyncio.wait_for(converse, timeout=1.0)

    assert brain.cancelled
    assert voice.spoken == []
    assert voice.interrupted == 1


async def test_speaking_a_block_does_not_count_against_the_brain_timeout():
    events = []
    voice = EventVoice(["hello"], events, takes=0.3)  # each block outlasts the timeout
    brain = ToolTurnBrain(["One.", "Two."], events, tool_takes=0.15)
    core = _speaking_core(voice)
    core.config = SimpleNamespace(voice_backend="text", brain_timeout_s=0.2)
    core._turn_speech.install(brain.agent)

    await core._converse(brain)

    assert voice.spoken == ["One.", "Two."]


async def test_a_brain_that_stalls_after_a_block_still_times_out(monkeypatch):
    import fast_body.app as app_module

    monkeypatch.setattr(app_module, "_CANCEL_GRACE_S", 0.05)
    events = []
    voice = EventVoice(["hello"], events, takes=0.05)
    brain = ToolTurnBrain(["One.", "Two."], events, tool_takes=5)
    core = _speaking_core(voice)
    core.config = SimpleNamespace(voice_backend="text", brain_timeout_s=0.1)
    core._turn_speech.install(brain.agent)

    await asyncio.wait_for(core._converse(brain), timeout=2)

    assert brain.cancelled
    assert voice.spoken == ["One.", "Sorry, I lost my train of thought there."]
