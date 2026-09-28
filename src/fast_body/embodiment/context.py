"""Process-wide handle the in-process robot tools use to reach the live robot.

`@fast.tool` functions are module-level — the brain only sees their signature, so
runtime dependencies (the connected robot, the movement manager, the recorded-move
libraries) can't be passed as arguments. Instead the app installs a single
`RobotContext` at startup and the tools read it via `RobotContext.current()`.

In tests, install a fake context with `RobotContext.set(...)`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class RobotContext:
    """Everything the robot tools need to act on the body."""

    robot: Any  # reachy_mini.ReachyMini (or a stub in tests)
    movement: Any  # embodiment.moves.MovementManager (or a stub)
    emotions: Any = None  # reachy_mini.motion.recorded_move.RecordedMoves | None
    dances: Any = None  # reachy_mini.motion.recorded_move.RecordedMoves | None
    gaze: Any = None  # embodiment.gaze.GazeController | None
    vision: Any = None  # embodiment.vision.VisionReader | None, behind `examine(question)`
    camera: bool = True  # ENABLE_CAMERA: False and camera()/examine() decline politely
    # Set by `emotion(hold=True)`: keep the expression through the spoken reply
    # instead of relaxing to neutral. Cleared when the next turn starts listening.
    hold_expression: bool = False

    _current: ClassVar[RobotContext | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def set(cls, ctx: RobotContext) -> None:
        with cls._lock:
            cls._current = ctx

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._current = None

    @classmethod
    def current(cls) -> RobotContext:
        with cls._lock:
            if cls._current is None:
                raise RuntimeError("RobotContext is not installed — tools called before the body started")
            return cls._current
