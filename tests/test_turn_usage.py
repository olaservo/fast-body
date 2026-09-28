"""The latency line says where a turn went, and the body and page are told it is working."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from reachy_mini.utils import create_head_pose

from fast_body.app import _turn_usage, _usage_index
from fast_body.embodiment import cues
from tests.test_turn_timeout import FakeVoice, _core

_HEAD = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)


def _turn(prompt, cached, completion, reasoning=0, tools=0):
    """One model call as fast-agent's UsageAccumulator records it."""
    return SimpleNamespace(
        prompt=SimpleNamespace(total=prompt, cache_read=cached),
        completion=SimpleNamespace(total=completion, reasoning=reasoning),
        tool_calls=tools,
    )


# ── the usage suffix ──────────────────────────────────────────────────────
def test_usage_suffix_sums_the_turns_model_calls():
    ledger = SimpleNamespace(
        turns=[_turn(1, 1, 1), _turn(48000, 40000, 300, tools=1), _turn(48500, 48000, 120, reasoning=80)]
    )
    agent = SimpleNamespace(usage_accumulator=ledger)

    assert _turn_usage(agent, 1) == (
        "; 2 model calls, prompt 96500 (88000 cached), completion 420, reasoning 80, tool calls 1"
    )


def test_usage_suffix_is_empty_when_there_is_nothing_to_say():
    assert _turn_usage(SimpleNamespace(usage_accumulator=None), 0) == ""
    assert _turn_usage(SimpleNamespace(), 0) == ""
    assert _usage_index(SimpleNamespace()) == 0

    agent = SimpleNamespace(usage_accumulator=SimpleNamespace(turns=[_turn(1, 0, 1)]))
    assert _usage_index(agent) == 1
    assert _turn_usage(agent, 1) == ""  # nothing added this turn


class CountingBrain:
    """A brain whose turn is two model calls, one of them with a tool call."""

    def __init__(self):
        self.agent = SimpleNamespace(name="body", message_history=[], usage_accumulator=SimpleNamespace(turns=[]))

    def resolve_agent(self, _name):
        return self.agent

    async def send(self, text):
        self.agent.message_history.append(SimpleNamespace(role="user", content=text))
        self.agent.usage_accumulator.turns.append(_turn(1000, 800, 50, tools=1))
        self.agent.usage_accumulator.turns.append(_turn(1100, 1000, 40))
        self.agent.message_history.append(SimpleNamespace(role="assistant", content=[SimpleNamespace(text="Roll.")]))
        return "Roll."


async def test_latency_line_carries_the_turns_usage(caplog):
    caplog.set_level(logging.INFO, logger="fast_body.app")
    voice = FakeVoice(["hello"])  # then EOF, which ends a text session
    core = _core(voice)

    await core._converse(CountingBrain())

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("turn latency"))
    assert line.endswith("; 2 model calls, prompt 2100 (1800 cached), completion 90, tool calls 1")
    assert voice.spoken == ["Roll."]


# ── the page and the body are told ────────────────────────────────────────
class RecordingHub:
    def __init__(self):
        self.posted: list[tuple[str, str | None]] = []

    def post_activity(self, state, step=None):
        self.posted.append((state, step))


def test_state_cues_reach_the_page():
    core = _core(FakeVoice([]))
    core.web_hub = RecordingHub()

    core._cue(cues.THINKING)
    core._cue(cues.SPEAKING)

    assert core.web_hub.posted == [("thinking", None), ("speaking", None)]


def test_a_page_that_cannot_be_told_does_not_cost_the_cue():
    core = _core(FakeVoice([]))
    core.web_hub = SimpleNamespace(post_activity=None)  # not callable
    core._cue(cues.THINKING)  # no exception


def test_tool_tick_queues_a_flick_then_the_thinking_pose():
    core = _core(FakeVoice([]))
    core.movement.current_pose = lambda: (_HEAD, (0.0, 0.0))
    queued = []
    core.movement.queue_move = queued.append
    core._tool_calls = 0

    core._tool_tick()
    core._tool_tick()

    assert [m.preset for m in queued] == ["perk_left", "curious", "perk_right", "curious"]
