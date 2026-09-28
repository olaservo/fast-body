"""Conversational state cues — let the body show which turn it's in.

A small, hardware-free mapping from conversation state to an antenna gesture:

- ``idle``      — waiting for someone to speak: no gesture, idle breathing runs
  and the antennas sway with it. This is the resting state between turns.
- ``listening`` — someone started speaking: antennas forward, attentive (the
  manager also freezes idle breathing so the robot holds still).
- ``thinking``  — one antenna up, pondering, while the brain works.
- ``speaking``  — back to neutral; the audio-reactive wobble carries liveness
  from here.

A long turn is tool calls with silence between them, and the thinking pose,
set once, cannot tell "still working" from "stuck". `tool_tick` is the body's
tick per call: one antenna flicks, alternating sides, and settles back to the
thinking pose. What the call is stays private; only that there was one shows.

Listening starts at speech onset, not when the mic opens: holding the pose
through silence looks like staring.

Keeping this pure (state + pose in, a Move or None out) makes the turn-taking
behaviour unit-testable without a robot. The orchestration (when to call it)
lives in `FastBodyCore`.
"""

from __future__ import annotations

from numpy.typing import NDArray

from fast_body.embodiment.moves import AntennaMove

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"

_PRESETS = {LISTENING: "listen", THINKING: "curious", SPEAKING: "neutral"}


def antenna_cue(
    state: str,
    head: NDArray,
    antennas: tuple[float, float],
    hold_expression: bool = False,
) -> AntennaMove | None:
    """Antenna gesture for a conversation state, or None if there's no cue.

    `hold_expression` suppresses the speaking cue, so an emotion the brain chose
    to hold survives the reply instead of being flattened as speech begins.
    """
    if hold_expression and state == SPEAKING:
        return None
    preset = _PRESETS.get(state)
    if preset is None:
        return None
    return AntennaMove(preset, head, antennas, duration=0.4)


def tool_tick(count: int, head: NDArray, antennas: tuple[float, float]) -> list[AntennaMove]:
    """The flick for the `count`-th tool call of a turn, then back to thinking."""
    side = "perk_left" if count % 2 else "perk_right"
    return [
        AntennaMove(side, head, antennas, duration=0.25),
        AntennaMove(_PRESETS[THINKING], head, antennas, duration=0.35),
    ]
