"""MovementManager queue + breathing logic, exercised without the 100 Hz thread."""

from __future__ import annotations

from fast_body.embodiment.moves import (
    AntennaMove,
    BreathingMove,
    HeadLookMove,
    MovementManager,
)


def _mgr(fake_robot):
    return MovementManager(fake_robot, idle_delay=0.3)


def test_queue_move_is_drained_into_internal_queue(fake_robot):
    mgr = _mgr(fake_robot)
    head, ant = mgr.current_pose()
    mgr.queue_move(HeadLookMove("left", head, ant))
    assert len(mgr._queue) == 0  # not yet drained (command queue only)
    mgr._drain_commands()
    assert len(mgr._queue) == 1


def test_advance_queue_starts_next_move(fake_robot):
    mgr = _mgr(fake_robot)
    head, ant = mgr.current_pose()
    mgr.queue_move(AntennaMove("curious", head, ant))
    mgr._drain_commands()
    mgr._advance_queue(now=10.0)
    assert isinstance(mgr.state.current_move, AntennaMove)
    assert mgr.state.move_start_time == 10.0


def test_clear_empties_queue_and_current(fake_robot):
    mgr = _mgr(fake_robot)
    head, ant = mgr.current_pose()
    mgr.queue_move(HeadLookMove("up", head, ant))
    mgr._drain_commands()
    mgr._advance_queue(now=0.0)
    mgr.clear_move_queue()
    mgr._drain_commands()
    assert mgr.state.current_move is None
    assert len(mgr._queue) == 0


def test_idle_triggers_breathing(fake_robot):
    mgr = _mgr(fake_robot)
    mgr.state.last_activity_time = 0.0
    mgr._maybe_breathe(now=5.0)  # well past idle_delay
    assert any(isinstance(m, BreathingMove) for m in mgr._queue)


def test_queued_move_interrupts_breathing(fake_robot):
    mgr = _mgr(fake_robot)
    head, ant = mgr.current_pose()
    mgr.state.current_move = BreathingMove(head, ant)
    mgr.queue_move(HeadLookMove("left", head, ant))
    mgr._drain_commands()
    mgr._advance_queue(now=1.0)
    assert isinstance(mgr.state.current_move, HeadLookMove)


def test_listening_suppresses_breathing(fake_robot):
    mgr = _mgr(fake_robot)
    mgr.set_listening(True)
    mgr._drain_commands()
    mgr.state.last_activity_time = 0.0
    mgr._maybe_breathe(now=5.0)
    assert not any(isinstance(m, BreathingMove) for m in mgr._queue)

    mgr.set_listening(False)
    mgr._drain_commands()
    mgr.state.last_activity_time = 0.0
    mgr._maybe_breathe(now=5.0)
    assert any(isinstance(m, BreathingMove) for m in mgr._queue)


def test_current_pose_falls_back_when_robot_unreadable():
    from tests.conftest import FakeRobot

    mgr = MovementManager(FakeRobot(readable=False))
    head, antennas = mgr.current_pose()  # should not raise
    assert head.shape == (4, 4)
    assert antennas == (0.0, 0.0)


# ── what the control thread publishes for other threads ───────────────────
def test_commanded_antennas_follow_the_pose_just_sent(fake_robot):
    mgr = _mgr(fake_robot)
    head, ant = mgr.current_pose()
    mgr.queue_move(AntennaMove("curious", head, ant, duration=1.0))
    mgr._step(now=0.0)
    mgr._step(now=0.999)  # the move has all but reached its target
    left, right = mgr.commanded_antennas()
    sent = fake_robot.set_target_calls[-1]["antennas"]
    assert (left, right) == (sent[0], sent[1])
    assert abs(left - 0.8727) < 1e-3  # 50 degrees


def test_move_active_excludes_breathing(fake_robot):
    mgr = _mgr(fake_robot)
    mgr.state.last_activity_time = -10.0
    mgr._step(now=0.0)  # idle long enough: a breath is queued...
    mgr._step(now=0.01)  # ...and starts
    assert isinstance(mgr.state.current_move, BreathingMove)
    assert mgr.move_active() is False

    head, ant = mgr.current_pose()
    mgr.queue_move(AntennaMove("curious", head, ant, duration=0.6))
    mgr._step(now=0.01)
    assert mgr.move_active() is True

    mgr._step(now=0.7)  # the cue is over
    assert mgr.move_active() is False


def test_listening_ends_a_running_breath_and_holds_the_antennas(fake_robot):
    mgr = _mgr(fake_robot)
    mgr.state.last_activity_time = -10.0
    mgr._step(now=0.0)
    mgr._step(now=0.01)  # the breath starts
    mgr._step(now=1.51)  # past the blend, at the peak of the sway
    assert isinstance(mgr.state.current_move, BreathingMove)
    before = mgr.commanded_antennas()
    assert before != (0.0, 0.0)

    mgr.set_listening(True)
    mgr._step(now=1.52)
    assert mgr.state.current_move is None
    assert mgr.commanded_antennas() == before  # frozen where the sway left them
    mgr._step(now=5.0)
    assert mgr.commanded_antennas() == before  # and no new breath while listening
