"""Shared test doubles — no hardware, no network, no LLM."""

from __future__ import annotations

import numpy as np
import pytest
from reachy_mini.utils import create_head_pose


@pytest.fixture(autouse=True)
def _no_user_personalities(tmp_path, monkeypatch):
    """Keep the developer's own ~/.fast-body/agent-cards out of every test.

    The parent of this directory stands in for the fast-agent home, so card
    packs land under it too (personality_store.home_paths).
    """
    from fast_body import personality

    monkeypatch.setattr(personality, "USER_PERSONALITIES_DIR", tmp_path / "home" / "agent-cards")


class FakeRobot:
    """Minimal stand-in for ReachyMini covering what the movement system reads."""

    def __init__(self, readable: bool = True):
        self.readable = readable
        self.set_target_calls: list[dict] = []
        self.goto_target_calls: list[dict] = []
        # ("start", weight) / ("stop", None), in call order.
        self.tracking_calls: list[tuple[str, float | None]] = []
        self.tracking_fails = False
        self.woke_up = False
        # Lifecycle method names in call order, for ordering assertions.
        self.calls: list[str] = []
        self.head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)

    def get_current_head_pose(self):
        if not self.readable:
            raise RuntimeError("not readable")
        return self.head_pose

    def get_current_joint_positions(self):
        if not self.readable:
            raise RuntimeError("not readable")
        return [0.0] * 7, [0.0, 0.0]

    def set_target(self, **kwargs):
        self.set_target_calls.append(kwargs)

    def goto_target(self, **kwargs):
        self.calls.append("goto_target")
        self.goto_target_calls.append(kwargs)

    def enable_motors(self):
        self.calls.append("enable_motors")

    def wake_up(self):
        self.calls.append("wake_up")
        self.woke_up = True

    def start_head_tracking(self, weight: float = 1.0):
        if self.tracking_fails:
            raise RuntimeError("daemon refused")
        self.tracking_calls.append(("start", weight))

    def stop_head_tracking(self):
        if self.tracking_fails:
            raise RuntimeError("daemon refused")
        self.tracking_calls.append(("stop", None))


class FakeMovement:
    """Records queued moves so tool tests can assert on them."""

    def __init__(self):
        self.queued: list = []
        self.cleared = 0

    def queue_move(self, move):
        self.queued.append(move)

    def clear_move_queue(self):
        self.cleared += 1

    def current_pose(self):
        return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.0, 0.0)


class FakeLibrary:
    """Stand-in for RecordedMoves (emotions / dances)."""

    def __init__(self, names: list[str]):
        self._names = names

    def list_moves(self):
        return list(self._names)

    def get(self, name):
        if name not in self._names:
            raise KeyError(name)

        class _Move:
            duration = 1.0

            def evaluate(self, t):
                return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), np.array([0.0, 0.0]), 0.0

        return _Move()


@pytest.fixture
def fake_robot():
    return FakeRobot()
