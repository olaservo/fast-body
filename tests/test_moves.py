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
