"""Push-to-talk in the app: the gate's wiring, the cue policy and the antenna hold loop."""

from __future__ import annotations

import asyncio
import logging
import threading

import pytest

from fast_body.app import FastBodyCore
from fast_body.config import Config
from fast_body.embodiment import cues
from fast_body.embodiment.moves import AntennaMove
from fast_body.listen_gate import ALWAYS_ON, PUSH_TO_TALK
from tests.conftest import FakeRobot


class AntennaRobot(FakeRobot):
    """A robot whose measured antenna positions the test can move."""

    def __init__(self) -> None:
        super().__init__()
        self.antennas = [0.0, 0.0]

    def get_current_joint_positions(self):
        return [0.0] * 7, list(self.antennas)


class GatedVoice:
    """A backend with the push-to-talk hooks, recording what the gate tells it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def supports_push_to_talk(self) -> bool:
        return True

    async def on_listen_open(self, *, continuous: bool) -> None:
        self.calls.append(("open", continuous))

    async def on_listen_close(self, *, commit: bool) -> None:
        self.calls.append(("close", commit))

    async def on_listen_mode_changed(self, mode: str) -> None:
        self.calls.append(("mode", mode))

    async def aclose(self) -> None:
        self.calls.append(("closed",))


class PlainVoice:
    """The discrete backend's shape: no hooks."""


def _core(robot, voice, **env) -> FastBodyCore:
    core = FastBodyCore(robot, owns_robot=False, stop_event=threading.Event(), config=Config())
    core.voice = voice
    return core


@pytest.fixture
def ptt_env(monkeypatch):
    monkeypatch.setenv("LISTEN_MODE", "push_to_talk")
    monkeypatch.setenv("PTT_ANTENNA", "true")


def _queued_moves(core: FastBodyCore) -> list:
    core.movement._drain_commands()
    return list(core.movement._queue)


# ── binding ───────────────────────────────────────────────────────────────
async def test_push_to_talk_binds_the_gate_and_starts_the_antenna_loop(ptt_env):
    core = _core(AntennaRobot(), GatedVoice())
    core._bind_listen_gate(None)
    assert core.listen_gate is not None
    assert core.listen_gate.mode == PUSH_TO_TALK
    assert not core.listen_gate.enabled
    assert core._antenna_task is not None and not core._antenna_task.done()
    await core._stop_antenna_loop()
    assert core._antenna_task is None


async def test_antenna_loop_can_be_turned_off(ptt_env, monkeypatch):
    monkeypatch.setenv("PTT_ANTENNA", "false")
    core = _core(AntennaRobot(), GatedVoice())
    core._bind_listen_gate(None)
    assert core.listen_gate is not None and core.listen_gate.mode == PUSH_TO_TALK
    assert core._antenna_task is None


async def test_a_backend_without_the_hooks_falls_back_to_always_on(ptt_env, caplog):
    core = _core(AntennaRobot(), PlainVoice())
    with caplog.at_level(logging.WARNING):
        core._bind_listen_gate(None)
    assert core.listen_gate is not None
    assert core.listen_gate.mode == ALWAYS_ON
    assert core.listen_gate.enabled
    assert core._antenna_task is None
    assert any("VOICE_BACKEND=realtime" in r.message for r in caplog.records)


async def test_always_on_leaves_everything_as_it_was(monkeypatch):
    monkeypatch.setenv("LISTEN_MODE", "always_on")
    core = _core(AntennaRobot(), GatedVoice())
    core._bind_listen_gate(None)
    assert core.listen_gate is not None and core.listen_gate.enabled
    assert core._antenna_task is None


async def test_the_gate_reaches_the_voice_and_the_page(ptt_env):
    class Hub:
        def __init__(self) -> None:
            self.gate = None

        def bind_listen_gate(self, gate) -> None:
            self.gate = gate

    voice, hub = GatedVoice(), Hub()
    core = _core(AntennaRobot(), voice)
    core._bind_listen_gate(hub)
    await core._stop_antenna_loop()
    assert hub.gate is core.listen_gate

    core.listen_gate.set_held(True)
    core.listen_gate.set_held(False)
    for _ in range(3):
        await asyncio.sleep(0)
    assert voice.calls == [("open", True), ("close", True)]


# ── cue policy ────────────────────────────────────────────────────────────
async def test_antenna_cues_are_held_back_while_the_gate_is_open(ptt_env, monkeypatch):
    monkeypatch.setenv("PTT_ANTENNA", "false")
    core = _core(AntennaRobot(), GatedVoice())
    core._bind_listen_gate(None)

    core.listen_gate.set_held(True)
    core._cue(cues.LISTENING)
    assert _queued_moves(core) == []  # no listen pose...
    assert core.movement._listening is True  # ...but breathing is still frozen

    core._tool_tick()
    assert _queued_moves(core) == []

    core.listen_gate.set_held(False)
    core._cue(cues.THINKING)
    moves = _queued_moves(core)
    assert len(moves) == 1 and isinstance(moves[0], AntennaMove) and moves[0].preset == "curious"


async def test_cues_play_as_usual_in_always_on(monkeypatch):
    monkeypatch.setenv("LISTEN_MODE", "always_on")
    core = _core(AntennaRobot(), GatedVoice())
    core._bind_listen_gate(None)
    core._cue(cues.LISTENING)
    moves = _queued_moves(core)
    assert len(moves) == 1 and moves[0].preset == "listen"


# ── the antenna hold loop ─────────────────────────────────────────────────
async def _until(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def test_a_deflected_antenna_opens_the_gate_and_letting_go_closes_it(ptt_env, caplog):
    robot = AntennaRobot()
    voice = GatedVoice()
    core = _core(robot, voice)
    with caplog.at_level(logging.INFO, logger="fast_body.app"):
        core._bind_listen_gate(None)
        gate = core.listen_gate
        await asyncio.sleep(0.2)  # the baseline settles at rest
        assert not gate.held

        robot.antennas = [0.4, 0.0]  # a hand, as the residual: commanded is (0, 0)
        await _until(lambda: gate.held)
        robot.antennas = [0.0, 0.0]
        await _until(lambda: not gate.held)
        await core._stop_antenna_loop()

    assert voice.calls == [("open", True), ("close", True)]
    started = [r.message for r in caplog.records if "antenna hold started" in r.message]
    assert started and "residual [0.4, 0.0]" in started[0] and "commanded [0.0, 0.0]" in started[0]


async def test_a_brush_shorter_than_the_debounce_is_ignored(ptt_env):
    robot = AntennaRobot()
    core = _core(robot, GatedVoice())
    core._bind_listen_gate(None)
    gate = core.listen_gate
    await asyncio.sleep(0.2)
    robot.antennas = [0.4, 0.0]
    await asyncio.sleep(0.05)  # one or two polls at 30 Hz
    robot.antennas = [0.0, 0.0]
    await asyncio.sleep(0.2)
    await core._stop_antenna_loop()
    assert not gate.held


async def test_a_move_starting_mid_hold_releases_it_and_pauses_detection(ptt_env):
    robot = AntennaRobot()
    voice = GatedVoice()
    core = _core(robot, voice)
    core._bind_listen_gate(None)
    gate = core.listen_gate
    await asyncio.sleep(0.2)
    robot.antennas = [0.0, -0.5]
    await _until(lambda: gate.held)

    core.movement.move_active = lambda: True  # type: ignore[method-assign]
    await _until(lambda: not gate.held)
    await asyncio.sleep(0.01)  # the hook lands on the loop a tick later
    assert voice.calls[-1] == ("close", True)  # what was said so far is still sent

    await asyncio.sleep(0.2)
    assert not gate.held  # the deflection is still there, but detection is paused
    core.movement.move_active = lambda: False  # type: ignore[method-assign]
    await asyncio.sleep(0.2)
    assert not gate.held  # a new baseline was taken from the held position; only a change counts
    await core._stop_antenna_loop()


async def test_a_sleeping_robot_opens_nothing_until_it_wakes(ptt_env):
    robot = AntennaRobot()
    voice = GatedVoice()
    core = _core(robot, voice)
    core._bind_listen_gate(None)
    gate = core.listen_gate
    await asyncio.sleep(0.2)

    # Put to sleep from outside the app: the antennas rest at the sleep pose
    # and no longer follow what the app commands.
    core.movement.asleep = lambda: True  # type: ignore[method-assign]
    robot.antennas = [-3.05, 3.05]
    await asyncio.sleep(0.2)
    robot.antennas = [-2.7, 2.7]  # a hand on a sleeping robot
    await asyncio.sleep(0.2)
    assert not gate.held
    assert voice.calls == []

    core.movement.asleep = lambda: False  # type: ignore[method-assign]
    robot.antennas = [0.0, 0.0]
    await asyncio.sleep(0.2)
    assert not gate.held  # waking is a new baseline, not a hold
    robot.antennas = [0.4, 0.0]
    await _until(lambda: gate.held)
    await core._stop_antenna_loop()


async def test_falling_asleep_mid_hold_drops_it_unsent(ptt_env):
    robot = AntennaRobot()
    voice = GatedVoice()
    core = _core(robot, voice)
    core._bind_listen_gate(None)
    gate = core.listen_gate
    await asyncio.sleep(0.2)
    robot.antennas = [0.4, 0.0]  # the daemon's sleep move pulling the antennas away
    await _until(lambda: gate.held)

    core.movement.asleep = lambda: True  # type: ignore[method-assign]
    await _until(lambda: not gate.held)
    await asyncio.sleep(0.01)
    assert voice.calls == [("open", True), ("close", False)]
    await core._stop_antenna_loop()


async def test_the_loop_survives_an_unreadable_robot(ptt_env):
    robot = AntennaRobot()
    robot.readable = False  # FakeRobot.get_current_head_pose raises; make joints raise too

    def boom():
        raise RuntimeError("not connected")

    robot.get_current_joint_positions = boom  # type: ignore[method-assign]
    core = _core(robot, GatedVoice())
    core._bind_listen_gate(None)
    await asyncio.sleep(0.2)
    assert core._antenna_task is not None and not core._antenna_task.done()
    await core._stop_antenna_loop()


async def test_shutdown_stops_the_loop(ptt_env):
    core = _core(AntennaRobot(), GatedVoice())
    core._bind_listen_gate(None)
    task = core._antenna_task
    assert task is not None
    await core._shutdown()
    assert task.cancelled() or task.done()
