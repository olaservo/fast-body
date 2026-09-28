"""Gaze control — following the person's face, run by the daemon.

The *daemon* owns face tracking: it runs the detector on
the camera stream and blends the resulting look-at aim into the head pose on its
own control loop. Following a face therefore costs this process nothing per
frame — we only say *how strongly* to track, and the daemon interpolates between
the pose we command and the aim.

At a partial weight the active emotion, dance or breathing still shows in the
head, biased toward the person.

Tracking starts **off**; the brain turns it on with the `look_at_me` tool.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


def _clamp(weight: float) -> float:
    return min(1.0, max(0.0, float(weight)))


class GazeController:
    """Enable/disable daemon-side face tracking, with a lower weight while speaking.

    Thread-safe: `set_enabled` is called from the brain's tool thread while
    `set_speaking` is called from the conversation loop.
    """

    def __init__(self, robot: Any, *, weight: float = 1.0, speaking_weight: float = 0.3) -> None:
        self.robot = robot
        self.weight = _clamp(weight)
        self.speaking_weight = _clamp(speaking_weight)

        self._enabled = False
        self._speaking = False
        self._lock = threading.Lock()

    @staticmethod
    def is_supported(robot: Any) -> bool:
        """True when the connected robot exposes daemon-side head tracking."""
        return callable(getattr(robot, "start_head_tracking", None))

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    # ── public, thread-safe API ──────────────────────────────────────────
    def set_enabled(self, enabled: bool) -> bool:
        """Start or stop following the face. Returns False if the daemon refused."""
        with self._lock:
            if enabled == self._enabled:
                return True
            # Speaking state only means anything while tracking; clear it on the
            # way out so a later toggle doesn't resume at the speaking weight.
            self._speaking = False
            if not self._apply(enabled):
                return False
            self._enabled = enabled
        logger.info("face tracking %s", "enabled" if enabled else "disabled")
        return True

    def set_speaking(self, speaking: bool) -> None:
        """Bias tracking down while the robot talks, so its own expression reads.

        A no-op while tracking is off. The weight stays non-zero so the detector
        keeps its lock on the face.
        """
        with self._lock:
            if not self._enabled or speaking == self._speaking:
                return
            self._speaking = speaking
            self._apply(True)

    def stop(self) -> None:
        """Hand the head back to the app (shutdown path); safe to call twice."""
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            self._speaking = False
            self._apply(False)

    # ── internals ────────────────────────────────────────────────────────
    def _apply(self, enabled: bool) -> bool:
        """Push the current state to the daemon. The caller holds the lock."""
        try:
            if enabled:
                self.robot.start_head_tracking(weight=self.speaking_weight if self._speaking else self.weight)
            else:
                self.robot.stop_head_tracking()
        except Exception as e:
            logger.warning("head tracking %s failed: %s", "start" if enabled else "stop", e)
            return False
        return True
