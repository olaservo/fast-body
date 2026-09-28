"""Movement system for expressive robot control.

A single 100 Hz control loop owns the robot's pose. Primary moves (head looks,
antenna flicks, recorded emotions and dances) are queued and played one at a
time; when the queue is empty the robot drifts into a gentle idle breathing
animation so it always looks alive.

This is a trimmed, self-contained descendant of the movement system in
`pollen-robotics/reachy_mini_conversation_app`: a queue plus a breathing core
that is easy to test and extend.

Face tracking is deliberately *not* here: the daemon blends the look-at aim into
whatever pose we command (see `embodiment/gaze.py`), so the head can follow a
person while these moves play without this loop knowing about it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any

import numpy as np
from numpy.typing import NDArray
from reachy_mini import ReachyMini
from reachy_mini.motion.move import Move
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation

logger = logging.getLogger(__name__)

CONTROL_LOOP_FREQUENCY_HZ = 100.0

FullBodyPose = tuple[NDArray[np.float64], tuple[float, float], float]

_NEUTRAL_HEAD = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)


def _smoothstep(alpha: float) -> float:
    """Ease-in/ease-out interpolation factor for organic motion."""
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3 - 2 * alpha)


class BreathingMove(Move):
    """Continuous, gentle idle animation. Runs forever until interrupted."""

    def __init__(
        self,
        start_pose: NDArray[np.float64],
        start_antennas: tuple[float, float],
        blend_duration: float = 1.0,
    ):
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas, dtype=np.float64)
        self.blend_duration = blend_duration

        self.neutral_pose = _NEUTRAL_HEAD
        self.neutral_antennas = np.array([0.0, 0.0])

        self.z_amplitude = 0.005  # 5 mm
        self.breath_freq = 0.1  # Hz
        self.antenna_amplitude = np.deg2rad(15)
        self.antenna_freq = 0.5  # Hz

    @property
    def duration(self) -> float:
        return float("inf")

    def evaluate(self, t: float) -> tuple:
        if t < self.blend_duration:
            alpha = t / self.blend_duration
            head = linear_pose_interpolation(self.start_pose, self.neutral_pose, alpha, yaw_as_scalar=True)
            antennas = (1 - alpha) * self.start_antennas + alpha * self.neutral_antennas
            return (head, antennas.astype(np.float64), 0.0)

        bt = t - self.blend_duration
        z = self.z_amplitude * np.sin(2 * np.pi * self.breath_freq * bt)
        head = create_head_pose(x=0, y=0, z=z, roll=0, pitch=0, yaw=0, degrees=True, mm=False)
        sway = self.antenna_amplitude * np.sin(2 * np.pi * self.antenna_freq * bt)
        antennas = np.array([sway, -sway], dtype=np.float64)
        return (head, antennas, 0.0)


class HeadLookMove(Move):
    """Turn the head to look in a named direction (with expressive antennas)."""

    DIRECTIONS = {
        "left": (0, 0, 0, 0, 0, 30),
        "right": (0, 0, 0, 0, 0, -30),
        "up": (0, 0, 10, 0, 15, 0),
        "down": (0, 0, -5, 0, -15, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }
    ANTENNAS_MAP = {
        "left": (-50.0, 20.0),
        "right": (20.0, -50.0),
        "up": (40.0, 40.0),
        "down": (-30.0, -30.0),
        "front": (0.0, 0.0),
    }

    def __init__(
        self,
        direction: str,
        start_pose: NDArray[np.float64],
        start_antennas: tuple[float, float],
        duration: float = 1.0,
    ):
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas, dtype=np.float64)
        self._duration = duration

        p = self.DIRECTIONS.get(direction, self.DIRECTIONS["front"])
        self.target_pose = create_head_pose(
            x=p[0], y=p[1], z=p[2], roll=p[3], pitch=p[4], yaw=p[5], degrees=True, mm=True
        )
        a = self.ANTENNAS_MAP.get(direction, (0.0, 0.0))
        self.target_antennas = np.array([np.deg2rad(a[0]), np.deg2rad(a[1])], dtype=np.float64)

    @property
    def duration(self) -> float:
        return self._duration

    def update_start(self, pose: NDArray[np.float64], antennas: tuple[float, float]) -> None:
        """Re-anchor the start pose when the move actually begins, so chained
        moves blend smoothly instead of snapping back."""
        self.start_pose = pose
        self.start_antennas = np.array(antennas, dtype=np.float64)

    def evaluate(self, t: float) -> tuple:
        alpha = _smoothstep(t / self._duration)
        # yaw_as_scalar keeps the turn going through the front: the geodesic
        # shortest path can swing a wide yaw round through ±180°, which the head
        # cannot reach anyway.
        head = linear_pose_interpolation(self.start_pose, self.target_pose, alpha, yaw_as_scalar=True)
        antennas = (1 - alpha) * self.start_antennas + alpha * self.target_antennas
        return (head, antennas.astype(np.float64), 0.0)


class AntennaMove(Move):
    """Move only the antennas to an expressive preset; head holds still."""

    PRESETS = {
        "curious": (50.0, 30.0),
        "excited": (60.0, 60.0),
        "sad": (-40.0, -40.0),
        "point_left": (-70.0, 20.0),
        "point_right": (20.0, -70.0),
        "listen": (15.0, 15.0),
        "surprised": (70.0, 70.0),
        "shy": (-30.0, -30.0),
        "angry": (-50.0, 50.0),
        "confused": (30.0, -30.0),
        "neutral": (0.0, 0.0),
        "wiggle": (40.0, -40.0),
        "perk_left": (50.0, 0.0),
        "perk_right": (0.0, 50.0),
        "droop": (-50.0, -50.0),
    }

    def __init__(
        self,
        preset: str,
        start_pose: NDArray[np.float64],
        start_antennas: tuple[float, float],
        duration: float = 0.6,
    ):
        self.preset = preset
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas, dtype=np.float64)
        self._duration = duration
        a = self.PRESETS.get(preset, (0.0, 0.0))
        self.target_antennas = np.array([np.deg2rad(a[0]), np.deg2rad(a[1])], dtype=np.float64)

    @property
    def duration(self) -> float:
        return self._duration

    def update_start(self, pose: NDArray[np.float64], antennas: tuple[float, float]) -> None:
        self.start_pose = pose
        self.start_antennas = np.array(antennas, dtype=np.float64)

    def evaluate(self, t: float) -> tuple:
        alpha = _smoothstep(t / self._duration)
        antennas = (1 - alpha) * self.start_antennas + alpha * self.target_antennas
        return (self.start_pose, antennas.astype(np.float64), 0.0)


class RecordedQueueMove(Move):
    """Adapt a pre-recorded move (emotion or dance) for the queue.

    Wraps any object with `.duration` and `.evaluate(t) -> (head, antennas, yaw)`,
    normalising tuple antennas to numpy and falling back to neutral on error.
    """

    def __init__(self, name: str, recorded: Any):
        self._move = recorded
        self.name = name

    @property
    def duration(self) -> float:
        return float(self._move.duration)

    def evaluate(self, t: float) -> tuple:
        try:
            head, antennas, body_yaw = self._move.evaluate(t)
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]], dtype=np.float64)
            return (head, antennas, body_yaw)
        except Exception as e:
            logger.error("Recorded move '%s' failed at t=%.3f: %s", self.name, t, e)
            return (_NEUTRAL_HEAD, np.array([0.0, 0.0], dtype=np.float64), 0.0)


def recorded_move(name: str, library: RecordedMoves) -> RecordedQueueMove:
    """Build a queueable move from a recorded emotion/dance library entry."""
    return RecordedQueueMove(name, library.get(name))


@dataclass
class _State:
    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0
    last_primary_pose: FullBodyPose | None = None


class MovementManager:
    """Coordinate robot movement on a single 100 Hz control thread.

    Thread-safe: `queue_move` / `clear_move_queue` may be called from any thread
    (e.g. the brain's tool calls). The control loop is the only writer to the
    robot.

        mgr = MovementManager(robot)
        mgr.start()
        mgr.queue_move(HeadLookMove("left", *mgr.current_pose()))
        mgr.stop()
    """

    def __init__(self, robot: ReachyMini, idle_delay: float = 0.3):
        self.robot = robot
        self.idle_delay = idle_delay
        self.period = 1.0 / CONTROL_LOOP_FREQUENCY_HZ

        self._now = time.monotonic
        self.state = _State(last_activity_time=self._now())
        self.state.last_primary_pose = (_NEUTRAL_HEAD, (0.0, 0.0), 0.0)

        self._queue: deque[Move] = deque()
        self._commands: Queue[tuple[str, Any]] = Queue()
        self._breathing = False
        self._listening = False  # while True, hold still (no idle breathing)

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── public, thread-safe API ──────────────────────────────────────────
    def queue_move(self, move: Move) -> None:
        self._commands.put(("queue", move))

    def clear_move_queue(self) -> None:
        self._commands.put(("clear", None))

    def set_listening(self, listening: bool) -> None:
        """While listening, suppress idle breathing so the robot holds still and
        visibly 'pays attention'. Thread-safe."""
        self._commands.put(("listening", bool(listening)))

    def current_pose(self) -> tuple[NDArray[np.float64], tuple[float, float]]:
        """Best-effort current (head_pose, antennas) for seeding a new move.

        Reads live state from the robot; falls back to the last commanded
        primary pose if the robot isn't readable (e.g. in tests).
        """
        try:
            head = self.robot.get_current_head_pose()
            _, antennas = self.robot.get_current_joint_positions()
            return head, (float(antennas[0]), float(antennas[1]))
        except Exception:
            pose = self.state.last_primary_pose
            assert pose is not None
            return pose[0], pose[1]

    # ── control thread ───────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="movement", daemon=True)
        self._thread.start()
        logger.info("MovementManager started")

    def stop(self) -> None:
        if not self._thread or not self._thread.is_alive():
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        try:
            self.robot.goto_target(head=_NEUTRAL_HEAD, antennas=[0.0, 0.0], duration=1.5, body_yaw=0.0)
        except Exception as e:
            logger.debug("reset to neutral failed: %s", e)
        logger.info("MovementManager stopped")

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd, payload = self._commands.get_nowait()
            except Empty:
                return
            if cmd == "queue" and isinstance(payload, Move):
                self._queue.append(payload)
                self.state.last_activity_time = self._now()
            elif cmd == "clear":
                self._queue.clear()
                self.state.current_move = None
                self.state.move_start_time = None
                self._breathing = False
            elif cmd == "listening":
                self._listening = payload
                self.state.last_activity_time = self._now()

    def _advance_queue(self, now: float) -> None:
        cur = self.state.current_move
        if cur is not None and self.state.move_start_time is not None:
            if now - self.state.move_start_time >= cur.duration:
                self.state.current_move = None
                self.state.move_start_time = None

        # A queued real move always interrupts idle breathing.
        if isinstance(self.state.current_move, BreathingMove) and self._queue:
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing = False

        if self.state.current_move is None and self._queue:
            move = self._queue.popleft()
            self.state.current_move = move
            self.state.move_start_time = now
            self._breathing = isinstance(move, BreathingMove)
            if hasattr(move, "update_start"):
                head, antennas = self.current_pose()
                move.update_start(head, antennas)

    def _maybe_breathe(self, now: float) -> None:
        if (
            self.state.current_move is None
            and not self._queue
            and not self._breathing
            and not self._listening
            and now - self.state.last_activity_time >= self.idle_delay
        ):
            head, antennas = self.current_pose()
            self._queue.append(BreathingMove(head, antennas, blend_duration=1.0))
            self._breathing = True

    def _primary_pose(self, now: float) -> FullBodyPose:
        move = self.state.current_move
        if move is not None and self.state.move_start_time is not None:
            head, antennas, body_yaw = move.evaluate(now - self.state.move_start_time)
            if head is None:
                head = _NEUTRAL_HEAD
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            pose = (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw or 0.0))
            self.state.last_primary_pose = pose
            return pose
        last = self.state.last_primary_pose
        assert last is not None
        return (last[0].copy(), last[1], last[2])

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            start = self._now()
            self._drain_commands()
            self._advance_queue(start)
            self._maybe_breathe(start)
            head, antennas, body_yaw = self._primary_pose(start)
            try:
                self.robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            except Exception as e:
                logger.debug("set_target failed: %s", e)
            time.sleep(max(0.0, self.period - (self._now() - start)))
