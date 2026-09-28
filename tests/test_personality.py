"""Personality cards load, list, override, and fall back."""

from __future__ import annotations

import logging

from fast_body.personality import (
    DEFAULT_PERSONALITY,
    PERSONALITIES_DIR,
    list_personalities,
    load_personality,
)
from fast_body.prompts import EMBODIMENT_INSTRUCTION, SKILLS_PLACEHOLDER, compose_instruction


# ── packaged cards ────────────────────────────────────────────────────────
def test_packaged_cards_exist_and_load(monkeypatch):
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    names = list_personalities()
    assert names[0] == DEFAULT_PERSONALITY  # the fallback reads first in the UI
    assert {"default", "marvin", "tavern"} <= set(names)
    for name in names:
        assert load_personality(name).instruction.strip()


def test_default_card_has_no_overrides(monkeypatch):
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    p = load_personality("default")
    assert (p.name, p.model, p.servers, p.voice) == ("default", None, [], None)


def test_shipped_cards_declare_voices(monkeypatch):
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    assert load_personality("marvin").voice == "onyx"
    assert load_personality("tavern").voice == "fable"


# ── fallback: a card must never stop the app ──────────────────────────────
def test_unknown_name_falls_back_to_default(monkeypatch, caplog):
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    with caplog.at_level(logging.WARNING):
        p = load_personality("does-not-exist")
    assert p.name == DEFAULT_PERSONALITY
    assert "not found" in caplog.text


def test_blank_name_means_default(monkeypatch):
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    assert load_personality("").name == DEFAULT_PERSONALITY
    assert load_personality(None).name == DEFAULT_PERSONALITY


def test_broken_card_falls_back(tmp_path, monkeypatch, caplog):
    (tmp_path / "broken.md").write_text("---\ntype: nonsense\n---\nhi\n", encoding="utf-8")
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING):
        p = load_personality("broken")
    assert p.name == DEFAULT_PERSONALITY
    assert "unusable" in caplog.text


def test_non_basic_card_is_rejected(tmp_path, monkeypatch, caplog):
    """A chain/router card parses fine but isn't a personality."""
    (tmp_path / "chainy.md").write_text("---\ntype: chain\nsequence:\n  - a\n---\n", encoding="utf-8")
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    with caplog.at_level(logging.WARNING):
        p = load_personality("chainy")
    assert p.name == DEFAULT_PERSONALITY


# ── external directory ────────────────────────────────────────────────────
def test_external_dir_adds_and_overrides(tmp_path, monkeypatch):
    (tmp_path / "captain.md").write_text("---\ntype: agent\n---\nYou are a captain.\n", encoding="utf-8")
    (tmp_path / "default.md").write_text("---\ntype: agent\nmodel: haiku\n---\nOverridden.\n", encoding="utf-8")
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))

    names = list_personalities()
    assert "captain" in names
    assert names.count("default") == 1  # merged, not duplicated

    assert load_personality("captain").instruction == "You are a captain."
    overridden = load_personality("default")
    assert (overridden.model, overridden.instruction) == ("haiku", "Overridden.")


def test_card_frontmatter_model_and_servers(tmp_path, monkeypatch):
    (tmp_path / "connected.md").write_text(
        "---\ntype: agent\nmodel: sonnet\nservers:\n  - fetch\n---\nBody.\n", encoding="utf-8"
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    p = load_personality("connected")
    assert (p.model, p.servers) == ("sonnet", ["fetch"])


# ── instruction composition ───────────────────────────────────────────────
def test_compose_puts_persona_first_then_guide_then_skills():
    text = compose_instruction("You are a test persona.")
    assert text.startswith("You are a test persona.\n\n")
    assert EMBODIMENT_INSTRUCTION in text
    # fast-agent fills this with the installed skills, or with nothing.
    assert text.endswith(SKILLS_PLACEHOLDER)


def test_compose_with_empty_persona_is_the_guide_and_the_skills():
    text = compose_instruction("  ")
    assert text.startswith(EMBODIMENT_INSTRUCTION)
    assert text.endswith(SKILLS_PLACEHOLDER)


def test_shipped_cards_are_character_only():
    """The body facts live in the shared guide, not in every card."""
    for name in ("default", "marvin", "tavern"):
        instruction = load_personality(name).instruction
        assert "antennas(" not in instruction  # no duplicated tool guide
        assert "spoken aloud" not in instruction


def test_default_dir_is_package_data():
    assert PERSONALITIES_DIR.name == "personalities"
    assert PERSONALITIES_DIR.parent.name == "fast_body"


# ── voice resolution ──────────────────────────────────────────────────────
def test_card_voice_comes_from_variables(tmp_path, monkeypatch):
    (tmp_path / "gruff.md").write_text("---\ntype: agent\nvariables:\n  voice: onyx\n---\nGruff.\n", encoding="utf-8")
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    assert load_personality("gruff").voice == "onyx"


def test_voice_resolution_precedence(monkeypatch):
    from fast_body.config import Config

    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "marvin")

    monkeypatch.setenv("OPENAI_TTS_VOICE", "nova")  # explicit override wins
    assert Config().resolve_tts_voice() == "nova"

    monkeypatch.setenv("OPENAI_TTS_VOICE", "auto")  # auto = follow the card
    assert Config().resolve_tts_voice() == "onyx"

    monkeypatch.delenv("OPENAI_TTS_VOICE")  # unset = follow the card too
    assert Config().resolve_tts_voice() == "onyx"

    monkeypatch.setenv("FAST_BODY_PERSONALITY", "default")  # card without a voice
    assert Config().resolve_tts_voice() == "cedar"


def test_voice_override_flag(monkeypatch):
    from fast_body.config import Config

    monkeypatch.setenv("OPENAI_TTS_VOICE", "nova")
    assert Config().tts_voice_overridden() is True
    monkeypatch.setenv("OPENAI_TTS_VOICE", "auto")
    assert Config().tts_voice_overridden() is False
    monkeypatch.delenv("OPENAI_TTS_VOICE")
    assert Config().tts_voice_overridden() is False


# ── config plumbing ───────────────────────────────────────────────────────
def test_config_reads_personality(monkeypatch):
    from fast_body.config import Config

    monkeypatch.setenv("FAST_BODY_PERSONALITY", "marvin")
    assert Config().personality == "marvin"
    monkeypatch.delenv("FAST_BODY_PERSONALITY")
    assert Config().personality == "default"


def test_personality_and_voice_are_settable_from_the_page():
    from fast_body.config import SETTABLE_ENV

    assert "FAST_BODY_PERSONALITY" in SETTABLE_ENV
    assert "OPENAI_TTS_VOICE" in SETTABLE_ENV


def test_a_card_can_say_how_to_speak():
    """Character lives in the sound as well as the words."""
    from fast_body.personality import load_personality

    delivery = load_personality("marvin").delivery
    assert delivery and "weary" in delivery.lower()


def test_a_card_without_delivery_leaves_it_unset():
    from fast_body.personality import load_personality

    assert load_personality("default").delivery is None


def test_delivery_precedence(monkeypatch):
    from fast_body.config import Config

    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    monkeypatch.setenv("FAST_BODY_PERSONALITY", "marvin")

    monkeypatch.delenv("TTS_DELIVERY", raising=False)
    assert "weary" in (Config().resolve_tts_delivery() or "").lower()

    monkeypatch.setenv("TTS_DELIVERY", "brisk and cheerful")
    assert Config().resolve_tts_delivery() == "brisk and cheerful"

    monkeypatch.setenv("FAST_BODY_PERSONALITY", "default")
    monkeypatch.delenv("TTS_DELIVERY")
    assert Config().resolve_tts_delivery() is None


def test_card_servers_are_read(tmp_path, monkeypatch):
    """On the robot nothing sets FAST_BODY_SERVERS, so a card has to ask for its servers."""
    (tmp_path / "gm.md").write_text(
        "---\ntype: agent\nservers:\n  - dice\n  - game-state\n---\nYou run the game.\n", encoding="utf-8"
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    assert load_personality("gm").servers == ["dice", "game-state"]


def test_card_activity_labels_come_from_variables(tmp_path, monkeypatch):
    (tmp_path / "gm.md").write_text(
        "---\ntype: agent\nvariables:\n  activity:\n    dance: striking up a tune\n    set_fact: ''\n    adjust: 3\n---\nGM.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))

    # Only the string labels; a blank or non-string entry names nothing.
    assert load_personality("gm").activity == {"dance": "striking up a tune"}


def test_a_card_without_activity_names_no_tools(tmp_path, monkeypatch):
    (tmp_path / "quiet.md").write_text("---\ntype: agent\n---\nQuiet.\n", encoding="utf-8")
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    assert load_personality("quiet").activity == {}
