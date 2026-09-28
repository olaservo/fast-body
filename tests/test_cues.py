"""State cues map conversation states to the right antenna gesture."""

from __future__ import annotations

from reachy_mini.utils import create_head_pose

from fast_body.embodiment import cues
from fast_body.embodiment.moves import AntennaMove

_HEAD = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)


def test_listening_cue_uses_listen_preset():
    move = cues.antenna_cue(cues.LISTENING, _HEAD, (0.0, 0.0))
    assert isinstance(move, AntennaMove)
    assert move.preset == "listen"


def test_thinking_cue_uses_curious_preset():
    assert cues.antenna_cue(cues.THINKING, _HEAD, (0.0, 0.0)).preset == "curious"


def test_speaking_cue_resets_to_neutral():
    assert cues.antenna_cue(cues.SPEAKING, _HEAD, (0.0, 0.0)).preset == "neutral"


def test_idle_has_no_cue():
    """Between turns the antennas belong to idle breathing, not a held pose."""
    assert cues.antenna_cue(cues.IDLE, _HEAD, (0.0, 0.0)) is None


def test_unknown_state_has_no_cue():
    assert cues.antenna_cue("bogus", _HEAD, (0.0, 0.0)) is None


def test_a_held_expression_survives_the_reply():
    """emotion(hold=True) means the mood should still be on the face while talking."""
    head, antennas = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.5, 0.5)

    assert cues.antenna_cue(cues.SPEAKING, head, antennas, hold_expression=True) is None
    assert cues.antenna_cue(cues.SPEAKING, head, antennas) is not None


def test_holding_does_not_suppress_the_other_cues():
    """Only speech flattens an expression; listening and thinking still show."""
    head, antennas = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), (0.5, 0.5)

    assert cues.antenna_cue(cues.LISTENING, head, antennas, hold_expression=True) is not None
    assert cues.antenna_cue(cues.THINKING, head, antennas, hold_expression=True) is not None


def test_the_prompt_forbids_writing_a_scene():
    """Observed: a reply invented the person leaving, then narrated waiting for them.

    Every assistant text block in a turn is joined and spoken as one utterance
    (`app._turn_spoken_text`), so a multi-beat monologue is spoken in full.
    """
    from fast_body.prompts import EMBODIMENT_INSTRUCTION

    text = EMBODIMENT_INSTRUCTION.lower()
    assert "one continuous utterance" in text
    assert "not narrate waiting or silence" in text


def test_tool_tick_flicks_alternate_sides_and_returns_to_thinking():
    first = cues.tool_tick(1, _HEAD, (0.0, 0.0))
    second = cues.tool_tick(2, _HEAD, (0.0, 0.0))

    assert [m.preset for m in first] == ["perk_left", "curious"]
    assert [m.preset for m in second] == ["perk_right", "curious"]
    assert sum(m.duration for m in first) < 1.0  # a tick, not a gesture
