"""Daemon-side face tracking: the GazeController and the look_at_me tool."""

from __future__ import annotations

import pytest

from fast_body.embodiment import tools
from fast_body.embodiment.context import RobotContext
from fast_body.embodiment.gaze import GazeController
from tests.conftest import FakeLibrary, FakeMovement, FakeRobot


@pytest.fixture
def gaze(fake_robot):
    return GazeController(fake_robot, weight=1.0, speaking_weight=0.3)


# ── GazeController ────────────────────────────────────────────────────────
def test_starts_disabled(gaze):
    assert gaze.is_enabled is False
    assert gaze.robot.tracking_calls == []


def test_enable_starts_daemon_tracking_at_full_weight(gaze):
    assert gaze.set_enabled(True) is True
    assert gaze.is_enabled is True
    assert gaze.robot.tracking_calls == [("start", 1.0)]


def test_disable_stops_daemon_tracking(gaze):
    gaze.set_enabled(True)
    gaze.set_enabled(False)
    assert gaze.is_enabled is False
    assert gaze.robot.tracking_calls[-1] == ("stop", None)


def test_repeated_enable_does_not_re_send(gaze):
    gaze.set_enabled(True)
    gaze.set_enabled(True)
    assert gaze.robot.tracking_calls == [("start", 1.0)]


def test_speaking_lowers_the_weight_without_pausing_detection(gaze):
    gaze.set_enabled(True)
    gaze.set_speaking(True)
    # A non-zero weight keeps the daemon's detector running, so the head is
    # still locked on when speech ends.
    assert gaze.robot.tracking_calls[-1] == ("start", 0.3)
    gaze.set_speaking(False)
    assert gaze.robot.tracking_calls[-1] == ("start", 1.0)


def test_speaking_is_ignored_while_tracking_is_off(gaze):
    gaze.set_speaking(True)
    assert gaze.robot.tracking_calls == []


def test_re_enabling_after_speaking_resumes_at_full_weight(gaze):
    gaze.set_enabled(True)
    gaze.set_speaking(True)
    gaze.set_enabled(False)
    gaze.set_enabled(True)
    assert gaze.robot.tracking_calls[-1] == ("start", 1.0)


def test_weights_are_clamped(fake_robot):
    gaze = GazeController(fake_robot, weight=5.0, speaking_weight=-1.0)
    gaze.set_enabled(True)
    gaze.set_speaking(True)
    assert gaze.robot.tracking_calls == [("start", 1.0), ("start", 0.0)]


def test_enable_failure_is_reported_and_leaves_tracking_off(fake_robot):
    fake_robot.tracking_fails = True
    gaze = GazeController(fake_robot)
    assert gaze.set_enabled(True) is False
    assert gaze.is_enabled is False


def test_stop_is_safe_when_never_enabled(gaze):
    gaze.stop()
    assert gaze.robot.tracking_calls == []


def test_stop_hands_the_head_back(gaze):
    gaze.set_enabled(True)
    gaze.stop()
    assert gaze.robot.tracking_calls[-1] == ("stop", None)


def test_is_supported_detects_an_older_robot(fake_robot):
    assert GazeController.is_supported(fake_robot) is True
    assert GazeController.is_supported(object()) is False


# ── look_at_me tool ────────────────────────────────────────────────────────
@pytest.fixture
def ctx_with_gaze():
    context = RobotContext(
        robot=FakeRobot(),
        movement=FakeMovement(),
        emotions=FakeLibrary([]),
        dances=FakeLibrary([]),
        gaze=GazeController(FakeRobot()),
    )
    RobotContext.set(context)
    yield context
    RobotContext.clear()


def test_look_at_me_enables_tracking(ctx_with_gaze):
    result = tools.look_at_me(True)
    assert ctx_with_gaze.gaze.is_enabled is True
    assert "watching" in result.lower()


def test_look_at_me_disables_tracking(ctx_with_gaze):
    tools.look_at_me(True)
    result = tools.look_at_me(False)
    assert ctx_with_gaze.gaze.is_enabled is False
    assert "no longer" in result.lower()


def test_look_at_me_without_camera_reports_unavailable():
    context = RobotContext(robot=FakeRobot(), movement=FakeMovement(), gaze=None)
    RobotContext.set(context)
    try:
        result = tools.look_at_me(True)
        assert "isn't available" in result.lower()
    finally:
        RobotContext.clear()
