"""A card's cast: tagged replies come out one voice at a time, in order."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from fast_body.config import Config
from fast_body.personality import load_personality
from fast_body.prompts import cast_instruction, compose_instruction
from fast_body.voice.base import build_backend, with_cast
from fast_body.voice.cast import Cast, CastMember, CastVoiceBackend, Utterance, normalize_name
from fast_body.voice.openai_backend import OpenAIVoiceBackend
from tests.test_realtime_backend import FakeMedia, FakeRobotAudio


@pytest.fixture
def cast() -> Cast:
    return Cast(
        [
            CastMember("Innkeeper", "onyx", "slow and gravelly", aliases=("Bram", "the innkeeper")),
            CastMember("Bard", "shimmer", aliases=("Wren the Bard",)),
        ],
        narrator_names=("tavern",),
    )


class _RecordingVoice:
    """A speaking backend that records what it was asked to say, and how."""

    def __init__(self) -> None:
        self.spoken: list[tuple[str, str | None, str | None]] = []
        self.interrupted = 0
        self.closed = False

    async def listen(self) -> str | None:
        return "heard"

    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        self.spoken.append((text, voice, delivery))

    def interrupt(self) -> None:
        self.interrupted += 1

    async def aclose(self) -> None:
        self.closed = True


# ── name matching ─────────────────────────────────────────────────────────
def test_names_match_loosely(cast: Cast) -> None:
    assert normalize_name("Wren, the Bard!") == "wren the bard"
    member = cast.resolve("wren, the bard")
    assert member is not None and member.voice == "shimmer"
    assert cast.resolve("THE INNKEEPER") is cast.resolve("Bram")
    assert cast.resolve("nobody") is None


def test_narrator_names_include_the_card(cast: Cast) -> None:
    for name in ("narrator", "Narrator", "me", "tavern", "Tavern"):
        assert cast.is_narrator(name)
    assert not cast.is_narrator("Bram")


# ── splitting a reply ─────────────────────────────────────────────────────
def test_script_splits_on_tags_in_reading_order(cast: Cast) -> None:
    text = (
        "The innkeeper looks up. [Bram] Nobody's been down that road since spring. "
        "[narrator] He goes back to polishing. [Wren the Bard] Well, I never."
    )
    assert cast.script(text) == [
        Utterance("The innkeeper looks up.", None, None, None),
        Utterance("Nobody's been down that road since spring.", "Innkeeper", "onyx", "slow and gravelly"),
        Utterance("He goes back to polishing.", None, None, None),
        Utterance("Well, I never.", "Bard", "shimmer", None),
    ]


def test_a_line_break_hands_the_voice_back_to_the_narrator(cast: Cast) -> None:
    # How the brain actually writes a scene: paragraphs, the character's one
    # tagged, the narration after it not.
    text = (
        "Wren goes pale reading the note over your shoulder.\n\n"
        "[Bram] Friends below? What does that mean?\n\n"
        "He's looking at you like you might have an answer. What do you do?"
    )
    assert cast.script(text) == [
        Utterance("Wren goes pale reading the note over your shoulder.", None, None, None),
        Utterance("Friends below? What does that mean?", "Innkeeper", "onyx", "slow and gravelly"),
        Utterance("He's looking at you like you might have an answer. What do you do?", None, None, None),
    ]


def test_a_character_speaking_two_paragraphs_is_tagged_in_both(cast: Cast) -> None:
    script = cast.script("[Bram] One.\n[Bram] Two.\nThree.")
    assert [(u.text, u.speaker) for u in script] == [("One. Two.", "Innkeeper"), ("Three.", None)]


def test_untagged_reply_is_one_narrator_utterance(cast: Cast) -> None:
    assert cast.script("Just narration.") == [Utterance("Just narration.")]


def test_unknown_name_falls_back_to_the_narrator(cast: Cast, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        script = cast.script("[Garrick] Who goes there?")
    assert script == [Utterance("Who goes there?")]
    assert "Garrick" in caplog.text


def test_same_voice_in_a_row_is_one_utterance(cast: Cast) -> None:
    script = cast.script("[Bram] One. [the innkeeper] Two.")
    assert [u.text for u in script] == ["One. Two."]
    assert script[0].speaker == "Innkeeper"


def test_empty_stretches_are_dropped(cast: Cast) -> None:
    assert cast.script("[Bram]   [Wren the Bard] Hm.") == [Utterance("Hm.", "Bard", "shimmer", None)]


# ── the wrapper ───────────────────────────────────────────────────────────
async def test_wrapper_speaks_each_utterance_in_its_voice(cast: Cast) -> None:
    inner = _RecordingVoice()
    backend = CastVoiceBackend(inner, cast)

    await backend.speak("Dusk. [Bram] Go home, son. [narrator] You don't.")

    assert inner.spoken == [
        ("Dusk.", None, None),
        ("Go home, son.", "onyx", "slow and gravelly"),
        ("You don't.", None, None),
    ]


async def test_wrapper_keeps_a_callers_voice_for_narration_only(cast: Cast) -> None:
    inner = _RecordingVoice()
    await CastVoiceBackend(inner, cast).speak("A. [Bram] B.", voice="nova", delivery="brisk")
    assert inner.spoken == [("A.", "nova", "brisk"), ("B.", "onyx", "slow and gravelly")]


async def test_wrapper_passes_the_rest_through(cast: Cast) -> None:
    inner = _RecordingVoice()
    backend = CastVoiceBackend(inner, cast)
    assert await backend.listen() == "heard"
    backend.interrupt()
    await backend.aclose()
    assert (inner.interrupted, inner.closed) == (1, True)


# ── the OpenAI backend takes a voice per call ─────────────────────────────
class _FakeSpeechClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        outer = self

        class _Stream:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc) -> None:
                return None

            async def iter_bytes(self, chunk_size=None):
                yield np.zeros(2400, dtype=np.int16).tobytes()

        class _Streaming:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                return _Stream()

        class _Speech:
            with_streaming_response = _Streaming()

        class _Audio:
            speech = _Speech()

        self.audio = _Audio()


async def test_openai_backend_switches_voice_and_delivery_per_call(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TTS_VOICE", "cedar")
    monkeypatch.setenv("TTS_DELIVERY", "even")
    monkeypatch.setenv("TTS_LEAD_IN_S", "0")
    backend = OpenAIVoiceBackend(Config(), FakeRobotAudio(FakeMedia()))
    backend._barge_in = False
    backend._speaker._client = _FakeSpeechClient()

    await backend.speak("narration")
    await backend.speak("a line", voice="onyx", delivery="gravelly")
    await backend.speak("more narration")

    assert [(c["voice"], c.get("instructions")) for c in backend._speaker._client.calls] == [
        ("cedar", "even"),
        ("onyx", "gravelly"),
        ("cedar", "even"),
    ]


# ── cards ─────────────────────────────────────────────────────────────────
def test_card_cast_comes_from_variables(tmp_path, monkeypatch) -> None:
    (tmp_path / "troupe.md").write_text(
        "---\ntype: agent\nvariables:\n  voice: fable\n  cast:\n"
        "    Innkeeper:\n      voice: onyx\n      delivery: gruff\n      aliases: [the innkeeper]\n"
        "    Bard: nova\n"
        "    Ghost:\n      delivery: no voice, so skipped\n"
        "---\nYou run an inn.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    p = load_personality("troupe")
    assert p.voice == "fable"
    assert p.cast.names == ("Innkeeper", "Bard")
    innkeeper = p.cast.resolve("the innkeeper")
    assert innkeeper is not None and (innkeeper.voice, innkeeper.delivery) == ("onyx", "gruff")
    assert p.cast.is_narrator("troupe")


def test_cards_without_a_cast_have_an_empty_one(monkeypatch) -> None:
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    assert not load_personality("default").cast
    assert not load_personality("marvin").cast


def test_shipped_tavern_card_has_the_readme_cast(monkeypatch) -> None:
    """The packaged example of a cast: the innkeeper and the bard the README shows."""
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    p = load_personality("tavern")
    assert p.voice == "fable"
    assert p.cast.names == ("Innkeeper", "Bard")
    innkeeper = p.cast.resolve("the innkeeper")
    bard = p.cast.resolve("bard")
    assert innkeeper is not None and innkeeper.voice == "onyx" and innkeeper.delivery
    assert bard is not None and bard.voice == "nova" and bard.delivery
    assert p.cast.is_narrator("tavern")
    assert p.activity  # the page may name its steps


def test_user_dir_cards_are_found_without_an_env_var(tmp_path, monkeypatch) -> None:
    """A robot keeps its own cards under ~/.fast-body/personalities, no config needed."""
    from fast_body import personality

    user_dir = tmp_path / "user-personalities"
    user_dir.mkdir()
    (user_dir / "troupe.md").write_text(
        "---\ntype: agent\nvariables:\n  cast:\n    Innkeeper: onyx\n---\nYou run an inn.\n", encoding="utf-8"
    )
    monkeypatch.setattr(personality, "USER_PERSONALITIES_DIR", user_dir)
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    assert "troupe" in personality.list_personalities()
    assert load_personality("troupe").cast.names == ("Innkeeper",)


def test_a_broken_cast_does_not_break_the_card(tmp_path, monkeypatch, caplog) -> None:
    (tmp_path / "odd.md").write_text(
        "---\ntype: agent\nvariables:\n  cast: [not, a, mapping]\n---\nHi.\n", encoding="utf-8"
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING):
        p = load_personality("odd")
    assert p.name == "odd" and not p.cast
    assert "cast" in caplog.text


# ── the brain is told the rule ────────────────────────────────────────────
def test_cast_instruction_names_the_cast_and_the_tag_form(cast: Cast) -> None:
    text = cast_instruction(cast)
    assert "[Innkeeper]" in text
    assert "Innkeeper, Bard" in text
    assert "[narrator]" in text
    assert "end of that line" in text


def test_compose_adds_the_voices_only_for_a_cast(cast: Cast) -> None:
    assert "square brackets" not in compose_instruction("Persona.")
    with_voices = compose_instruction("Persona.", cast=cast)
    assert with_voices.startswith("Persona.\n\n")
    assert "square brackets" in with_voices
    assert with_voices.index("square brackets") < with_voices.index("{{agentSkills}}")


# ── the factory wraps only when there is a cast ───────────────────────────
def test_with_cast_leaves_castless_cards_alone(monkeypatch) -> None:
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "default")
    inner = _RecordingVoice()
    assert with_cast(inner, Config()) is inner


@pytest.fixture
def troupe_card(tmp_path, monkeypatch) -> None:
    (tmp_path / "troupe.md").write_text(
        "---\ntype: agent\nvariables:\n  cast:\n    Innkeeper: onyx\n---\nYou run an inn.\n", encoding="utf-8"
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "troupe")


def test_with_cast_wraps_a_card_with_a_cast(troupe_card) -> None:
    wrapped = with_cast(_RecordingVoice(), Config())
    assert isinstance(wrapped, CastVoiceBackend)


def test_build_backend_keeps_the_text_backend_unwrapped(troupe_card, monkeypatch) -> None:
    monkeypatch.setenv("VOICE_BACKEND", "text")
    assert not isinstance(build_backend(Config(), robot=None), CastVoiceBackend)


async def test_wrapper_forwards_warm_up(cast: Cast) -> None:
    inner = _RecordingVoice()
    warmed = []
    inner.warm_up = lambda: _record(warmed)  # type: ignore[attr-defined]

    await CastVoiceBackend(inner, cast).warm_up()
    await CastVoiceBackend(_RecordingVoice(), cast).warm_up()  # an inner without one is fine

    assert warmed == [True]


async def _record(into: list) -> None:
    into.append(True)
