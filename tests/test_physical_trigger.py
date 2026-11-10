"""Antenna hold detector, replayed against the body's own antenna motion.

The robot's cues and idle breathing are what the detector has to ignore, so
each one is played through it twice: as the residual (measured minus
commanded, which is what the app feeds it) and as raw positions (which is what
a naive port would feed it).
"""

from __future__ import annotations

import numpy as np

from fast_body.embodiment import cues
from fast_body.embodiment.moves import _NEUTRAL_HEAD, AntennaMove, BreathingMove
from fast_body.physical_trigger import AntennaHoldDetector

POLL_HZ = 30
DT = 1.0 / POLL_HZ
REST = (0.0, 0.0)


def _commanded(moves, seconds: float) -> list[list[float]]:
    """Antenna targets of `moves` played back to back, sampled at the poll rate, then held."""
    samples: list[list[float]] = []
    t = 0.0
    for move in moves:
        duration = move.duration if move.duration != float("inf") else seconds
        start = t
        while t - start < duration:
            _, antennas, _ = move.evaluate(t - start)
            samples.append([float(antennas[0]), float(antennas[1])])
            t += DT
    while len(samples) < int(seconds * POLL_HZ):
        samples.append(list(samples[-1]))
    return samples


def _lagged(commanded: list[list[float]], alpha: float = 0.5) -> list[list[float]]:
    """A servo that closes half the gap to its target every poll."""
    measured = list(commanded[0])
    out = []
    for target in commanded:
        measured = [m + (c - m) * alpha for m, c in zip(measured, target)]
        out.append(list(measured))
    return out


def _ever_held(samples: list[list[float]]) -> bool:
    detector = AntennaHoldDetector()
    return any(detector.update(s) for s in samples)


def _breathing(seconds: float = 4.0) -> list[list[float]]:
    # Past the blend, so the full ±15° sway is in the replay.
    move = BreathingMove(_NEUTRAL_HEAD, REST, blend_duration=0.0)
    return _commanded([move], seconds)


def _cue(state: str) -> list[list[float]]:
    move = cues.antenna_cue(state, _NEUTRAL_HEAD, REST)
    assert move is not None
    return _commanded([move], 2.0)


def _tool_tick() -> list[list[float]]:
    return _commanded(cues.tool_tick(1, _NEUTRAL_HEAD, REST), 2.0)


# ── raw positions: the body's own motion reads as a hand ──────────────────
def test_breathing_alone_trips_the_detector_on_raw_positions():
    """The idle sway is ±0.262 rad, over the 0.25 rad press threshold by itself."""
    assert _ever_held(_breathing())


def test_the_thinking_cue_trips_it_on_raw_positions():
    assert _ever_held(_cue(cues.THINKING))  # 0.87 rad


def test_the_listening_cue_is_borderline_on_raw_positions():
    """0.262 rad is just over the threshold, and the baseline drifts up during the
    0.4 s ramp, so on its own it does not trip. It would from any other start."""
    assert not _ever_held(_cue(cues.LISTENING))
    from_thinking = _commanded([AntennaMove("listen", _NEUTRAL_HEAD, (0.87, 0.52))], 2.0)
    assert _ever_held(from_thinking)


def test_a_tool_tick_trips_it_on_raw_positions():
    assert _ever_held(_tool_tick())


# ── the residual: the same motion, commanded, cancels out ─────────────────
def test_a_perfect_servo_leaves_no_residual():
    for commanded in (_breathing(), _cue(cues.LISTENING), _cue(cues.THINKING), _tool_tick()):
        residual = [[m - c for m, c in zip(sample, sample)] for sample in commanded]
        assert not _ever_held(residual)


def test_servo_lag_on_the_cues_stays_under_the_press_threshold():
    """A lagging servo shows a residual during each cue; the debounce and threshold ride it out."""
    for commanded in (_breathing(), _cue(cues.LISTENING), _cue(cues.THINKING), _tool_tick()):
        measured = _lagged(commanded)
        residual = [[m - c for m, c in zip(ms, cs)] for ms, cs in zip(measured, commanded)]
        assert not _ever_held(residual)


def test_a_hand_on_the_antenna_shows_through_the_breathing():
    """Breathing plus a hand: the residual is the hand, and only the hand."""
    commanded = _breathing(4.0)
    hand = np.zeros((len(commanded), 2))
    hand[45:75, 0] = 0.4  # one second on the left antenna, from t = 1.5 s
    measured = [[c0 + h0, c1 + h1] for (c0, c1), (h0, h1) in zip(commanded, hand)]
    residual = [[m - c for m, c in zip(ms, cs)] for ms, cs in zip(measured, commanded)]

    detector = AntennaHoldDetector()
    held = [detector.update(s) for s in residual]
    assert not any(held[:45])
    assert held[47]  # the third over-threshold sample starts the hold
    assert all(held[47:75])
    assert not any(held[76:])


# ── the detector itself ───────────────────────────────────────────────────
def test_a_press_needs_three_consecutive_samples():
    detector = AntennaHoldDetector()
    detector.update([0.0, 0.0])
    assert detector.update([0.4, 0.0]) is False
    assert detector.update([0.4, 0.0]) is False
    assert detector.update([0.4, 0.0]) is True


def test_a_two_sample_blip_is_not_a_press():
    detector = AntennaHoldDetector()
    detector.update([0.0, 0.0])
    detector.update([0.4, 0.0])
    detector.update([0.4, 0.0])
    assert detector.update([0.0, 0.0]) is False
    assert detector.update([0.4, 0.0]) is False  # the count restarted


def test_release_has_hysteresis():
    detector = AntennaHoldDetector()
    detector.update([0.0, 0.0])
    for _ in range(3):
        detector.update([0.0, 0.4])
    assert detector.held
    assert detector.update([0.0, 0.2]) is True  # between the thresholds: still held
    assert detector.update([0.0, 0.1]) is False  # at the release threshold


def test_baseline_tracks_slow_drift_but_not_a_hold():
    detector = AntennaHoldDetector(baseline_alpha=0.5)
    detector.update([0.0, 0.0])
    for _ in range(20):
        detector.update([0.2, 0.2])  # under the threshold: the baseline follows
    assert not detector.held
    assert detector.update([0.2, 0.2]) is False
    for _ in range(3):
        detector.update([0.6, 0.2])
    assert detector.held
    for _ in range(50):
        detector.update([0.6, 0.2])
    assert detector.held  # the baseline never chased the hold


def test_reset_forgets_the_baseline_and_releases():
    detector = AntennaHoldDetector()
    detector.update([0.0, 0.0])
    for _ in range(3):
        detector.update([0.4, 0.0])
    assert detector.held
    detector.reset()
    assert not detector.held
    assert detector.update([0.4, 0.0]) is False  # a new baseline, taken from this sample


def test_empty_input_is_ignored():
    detector = AntennaHoldDetector()
    assert detector.update([]) is False
