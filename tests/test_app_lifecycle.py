"""Startup wiring: waking a sleeping robot, and building the gaze controller."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE

from fast_body.app import FastBodyCore
from fast_body.config import Config
from fast_body.embodiment.gaze import GazeController
from tests.conftest import FakeRobot


def _core(robot, **config_kwargs) -> FastBodyCore:
    return FastBodyCore(
        robot,
        owns_robot=False,
        stop_event=threading.Event(),
        config=Config(**config_kwargs),
    )


# ── wake from sleep ───────────────────────────────────────────────────────
def test_wakes_a_sleeping_robot(fake_robot):
    fake_robot.head_pose = SLEEP_HEAD_POSE.copy()
    assert _core(fake_robot)._wake_if_sleeping() is True
    assert fake_robot.woke_up is True


def test_leaves_an_awake_robot_alone(fake_robot):
    assert _core(fake_robot)._wake_if_sleeping() is False
    assert fake_robot.woke_up is False


def test_near_enough_to_the_sleep_pose_still_counts(fake_robot):
    # A real robot settles a few millimetres off the nominal sleep pose.
    pose = SLEEP_HEAD_POSE.copy()
    pose[:3, 3] += np.array([0.01, 0.0, -0.01])
    fake_robot.head_pose = pose
    assert _core(fake_robot)._wake_if_sleeping() is True


def test_unreadable_robot_does_not_raise():
    robot = FakeRobot(readable=False)
    assert _core(robot)._wake_if_sleeping() is False
    assert robot.woke_up is False


def test_motors_are_enabled_before_waking(fake_robot):
    fake_robot.head_pose = SLEEP_HEAD_POSE.copy()
    _core(fake_robot)._prepare_robot()
    # Asleep motors won't move: enable_motors has to land first (upstream #510).
    assert fake_robot.calls == ["enable_motors", "wake_up", "goto_target"]


# ── gaze controller construction ──────────────────────────────────────────
def test_gaze_disabled_without_camera(fake_robot):
    assert _core(fake_robot, enable_camera=False)._build_gaze() is None


def test_gaze_unavailable_on_a_robot_without_daemon_tracking():
    class OldRobot:
        pass

    assert _core(OldRobot())._build_gaze() is None


def test_gaze_uses_configured_weights(fake_robot):
    gaze = _core(fake_robot, head_tracking_weight=0.8, head_tracking_speaking_weight=0.1)._build_gaze()
    assert isinstance(gaze, GazeController)
    assert (gaze.weight, gaze.speaking_weight) == (0.8, 0.1)


@pytest.mark.parametrize("speaking", [True, False])
def test_cue_tracks_the_speaking_state(fake_robot, speaking):
    from fast_body.embodiment import cues

    core = _core(fake_robot)
    core.gaze = GazeController(fake_robot)
    core.gaze.set_enabled(True)
    core._cue(cues.SPEAKING if speaking else cues.LISTENING)
    expected = ("start", core.gaze.speaking_weight if speaking else core.gaze.weight)
    assert fake_robot.tracking_calls[-1] == expected


class _RecordingMovement:
    def __init__(self):
        self.listening: list[bool] = []
        self.queued: list = []

    def current_pose(self):
        return np.eye(4), (0.0, 0.0)

    def set_listening(self, listening):
        self.listening.append(listening)

    def queue_move(self, move):
        self.queued.append(move)


def test_idle_breathes_and_listening_holds_still(fake_robot):
    """The loop rests in IDLE (breathing on, no gesture) and the backend's
    speech-onset callback is what moves it to LISTENING (breathing held,
    antennas forward), the conversation app's freeze-on-speech_started shape."""
    from fast_body.embodiment import cues

    core = _core(fake_robot)
    core.movement = _RecordingMovement()
    core._cue(cues.IDLE)
    assert core.movement.listening == [False]
    assert core.movement.queued == []
    core._cue(cues.LISTENING)
    assert core.movement.listening == [False, True]
    assert [m.preset for m in core.movement.queued] == ["listen"]


# ── speaking the whole turn ───────────────────────────────────────────────
class _Msg:
    def __init__(self, role, texts):
        from types import SimpleNamespace

        self.role = role
        self.content = [SimpleNamespace(text=t) for t in texts]


class _FakeAgent:
    def __init__(self, messages):
        self.message_history = messages


def test_turn_text_includes_speech_before_a_tool_call():
    from fast_body.app import _turn_spoken_text

    agent = _FakeAgent(
        [
            _Msg("user", ["hi"]),
            _Msg("assistant", ["Oh, that is so cool!"]),  # then it called a tool
            _Msg("user", ["<tool result>"]),
            _Msg("assistant", ["Glad the terminal treats you well."]),
        ]
    )
    text = _turn_spoken_text(agent, 0, "fallback")
    assert text == "Oh, that is so cool!\n\nGlad the terminal treats you well."


def test_turn_text_only_reads_from_the_start_index():
    from fast_body.app import _turn_spoken_text

    agent = _FakeAgent([_Msg("assistant", ["old turn"]), _Msg("assistant", ["this turn"])])
    assert _turn_spoken_text(agent, 1, "fallback") == "this turn"


def test_turn_text_falls_back_when_the_turn_has_no_text():
    from fast_body.app import _turn_spoken_text

    agent = _FakeAgent([_Msg("user", ["hi"])])
    assert _turn_spoken_text(agent, 0, "the reply") == "the reply"


def test_turn_text_survives_a_broken_history():
    from fast_body.app import _turn_spoken_text

    class Broken:
        @property
        def message_history(self):
            raise RuntimeError("no history")

    assert _turn_spoken_text(Broken(), 0, "the reply") == "the reply"


# ── picking a conversation back up ────────────────────────────────────────
class _ResumedMsg:
    def __init__(self, role, text):
        self.role, self._text = role, text

    def first_text(self):
        return self._text

    def last_text(self):
        return self._text


class _Voice:
    def __init__(self, fails=False):
        self.spoken: list[str] = []
        self.fails = fails

    async def speak(self, text):
        if self.fails:
            raise RuntimeError("no audio device")
        self.spoken.append(text)


class _Hub:
    def __init__(self):
        self.seeded = None

    def seed(self, lines):
        self.seeded = lines


class _Brain:
    def __init__(self, history=None, broken=False):
        self.history, self.broken = history or [], broken

    def resolve_agent(self, _name):
        if self.broken:
            raise RuntimeError("no agent")
        return type("A", (), {"message_history": self.history})()


def _resumed_core(fake_robot, voice, hub, **cfg):
    core = _core(fake_robot, **cfg)
    core.voice, core.web_hub = voice, hub
    return core


HISTORY = [_ResumedMsg("user", "what's the gate code"), _ResumedMsg("assistant", "It's 4291.")]


async def test_reopen_seeds_the_page_and_speaks(fake_robot):
    voice, hub = _Voice(), _Hub()
    await _resumed_core(fake_robot, voice, hub)._reopen(_Brain(HISTORY))

    assert hub.seeded == [
        {"role": "user", "text": "what's the gate code"},
        {"role": "robot", "text": "It's 4291."},
    ]
    assert voice.spoken and "It's 4291." in voice.spoken[0]


async def test_reopen_still_seeds_the_page_with_the_greeting_off(fake_robot):
    """The page showing the conversation is the part you can't get any other way."""
    voice, hub = _Voice(), _Hub()
    core = _resumed_core(fake_robot, voice, hub)
    core.config.memory_greeting = False
    await core._reopen(_Brain(HISTORY))

    assert hub.seeded is not None
    assert voice.spoken == []


async def test_reopen_survives_a_voice_that_cannot_speak(fake_robot):
    """Losing the greeting must not cost the memory it just recovered."""
    hub = _Hub()
    await _resumed_core(fake_robot, _Voice(fails=True), hub)._reopen(_Brain(HISTORY))
    assert hub.seeded is not None


async def test_reopen_gives_up_quietly_on_an_unreadable_history(fake_robot):
    voice, hub = _Voice(), _Hub()
    await _resumed_core(fake_robot, voice, hub)._reopen(_Brain(broken=True))
    assert hub.seeded is None
    assert voice.spoken == []


async def test_reopen_says_nothing_when_the_robot_never_spoke(fake_robot):
    voice, hub = _Voice(), _Hub()
    await _resumed_core(fake_robot, voice, hub)._reopen(_Brain([_ResumedMsg("user", "hello")]))
    assert hub.seeded == [{"role": "user", "text": "hello"}]
    assert voice.spoken == []
