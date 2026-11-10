"""Physical push-to-talk trigger: hold an antenna to talk.

A hysteresis detector that reports whether an antenna is currently deflected past
a threshold ("held") relative to a slowly-tracked baseline, so the antenna poll
loop in `app.py` can drive a :class:`~fast_body.listen_gate.ListenGate` hold
state: deflect an antenna to open the mic (hold-to-talk), release to send.

Feed it the residual, measured minus commanded, when the commanded pose is known
(`MovementManager` sends every antenna target, so it is): the robot moving its
own antennas then reads as zero and only a hand shows up. Fed raw positions it
still works on a robot that is not animating, and on this robot that is never:
idle breathing sways the antennas ±0.262 rad, over the press threshold on its
own.

This mirrors the daemon's one-shot ``AntennaTouchDetector`` (same calibrated
deltas) but reports a *level* rather than a single edge, which is what hold-to-talk
needs. The baseline drifts only while released, so it absorbs slow drift without
ever chasing an active hold. A press has to persist for ``press_samples`` polls,
which rides out the servo lagging a fast commanded move.

Copied from the conversation-app port (`physical_trigger.py` on its
`feat/push-to-talk` branch), where it was tuned against the robot.
"""

from __future__ import annotations

# Match the daemon's calibrated antenna-touch thresholds
# (reachy_mini.daemon.app.startup_app.AntennaTouchDetector).
DEFAULT_PRESS_DELTA_RAD = 0.25  # deflection (rad) that starts a hold
DEFAULT_RELEASE_DELTA_RAD = 0.10  # deflection (rad) at/below which the hold releases
DEFAULT_BASELINE_ALPHA = 0.02  # resting-baseline tracking per update, while released
DEFAULT_PRESS_SAMPLES = 3  # consecutive over-threshold updates before a hold starts


class AntennaHoldDetector:
    """Level (held True/False) detector on antenna deflection from a resting baseline."""

    def __init__(
        self,
        press_delta_rad: float = DEFAULT_PRESS_DELTA_RAD,
        release_delta_rad: float = DEFAULT_RELEASE_DELTA_RAD,
        baseline_alpha: float = DEFAULT_BASELINE_ALPHA,
        press_samples: int = DEFAULT_PRESS_SAMPLES,
    ) -> None:
        """Create a detector with the given hysteresis thresholds (radians)."""
        self.press_delta_rad = press_delta_rad
        self.release_delta_rad = release_delta_rad
        self.baseline_alpha = baseline_alpha
        self.press_samples = max(1, int(press_samples))
        self._baseline: list[float] | None = None
        self._held = False
        self._over = 0

    def reset(self) -> None:
        """Forget the baseline and release. Call whenever detection is paused."""
        self._baseline = None
        self._held = False
        self._over = 0

    @property
    def held(self) -> bool:
        """Whether an antenna is currently deflected past the press threshold."""
        return self._held

    def update(self, present: list[float]) -> bool:
        """Feed the latest present antenna positions (rad); return the held state."""
        if not present:
            return self._held
        if self._baseline is None:
            self._baseline = list(present)
            self._held = False
            return False

        deflection = max(abs(p - b) for p, b in zip(present, self._baseline))
        if not self._held:
            if deflection >= self.press_delta_rad:
                self._over += 1
                if self._over >= self.press_samples:
                    self._held = True
            else:
                self._over = 0
                # Track the resting baseline only while released — never chase a hold.
                self._baseline = [b + (p - b) * self.baseline_alpha for p, b in zip(present, self._baseline)]
        elif deflection <= self.release_delta_rad:
            self._held = False
            self._over = 0
        return self._held
