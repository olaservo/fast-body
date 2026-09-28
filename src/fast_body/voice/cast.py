"""A cast of voices: several characters out of one brain and one TTS backend.

A personality card may list a ``cast`` under ``variables``: the characters the
robot voices besides itself, each with a TTS voice and, optionally, a delivery.
The brain marks a character's line by writing the name in square brackets in
front of it::

    The innkeeper looks up from the bar. [Innkeeper] Nobody's been down that
    road since spring. [narrator] He goes back to polishing a glass.

``Cast.script`` cuts a reply into utterances on those tags, and
``CastVoiceBackend`` speaks them in order, each in its own voice. Text before
the first tag, ``[narrator]``, and the line after a character's line are the
robot's own voice (the card's ``voice`` and ``delivery``, which the inner
backend already holds).

Why a tag and not a tool: a tool call is a full round trip to the model, and a
scene with three spoken lines would cost three more brain turns on top of a
reply that already takes ten seconds. A tag costs nothing.

Names match loosely (case, punctuation and spacing ignored, plus the aliases
the card lists), and an unknown name falls back to the narrator with a logged
warning. The line still gets spoken, which at a table beats a refusal.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fast_body.voice.base import VoiceBackend

logger = logging.getLogger(__name__)

# Names that always mean "the robot's own voice", on any card.
NARRATOR_NAMES = frozenset({"narrator", "me", "myself"})

# `[Name]` on one line, no nesting, short enough that a bracketed sentence is
# left alone. A line break also ends a character's line: the brain tags the
# character's paragraph, never the narration after it.
_BOUNDARY = re.compile(r"\[([^\[\]\n]{1,60})\]|\n")


@dataclass(frozen=True)
class CastMember:
    """One character the robot can speak as."""

    name: str
    voice: str
    delivery: str | None = None
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class Utterance:
    """A stretch of a reply and who says it.

    ``voice``/``delivery`` are ``None`` for the narrator: the backend then uses
    the voice it was built with, so the card's own settings (and any
    ``OPENAI_TTS_VOICE`` override) keep applying to narration.
    """

    text: str
    speaker: str | None = None
    voice: str | None = None
    delivery: str | None = None


def normalize_name(text: str) -> str:
    """Lower-case, punctuation dropped, whitespace collapsed.

    So ``Wren, the Bard!``, ``wren the bard`` and ``"Wren" the bard`` land on
    the same key; the card's aliases cover the rest.
    """
    return " ".join(re.sub(r"[^\w\s]", " ", text).split()).lower()


class Cast:
    """The card's characters, resolved by loose name match."""

    def __init__(self, members: Iterable[CastMember], narrator_names: Iterable[str] = ()) -> None:
        self._members: list[CastMember] = []
        self._by_alias: dict[str, CastMember] = {}
        self._narrator = {normalize_name(n) for n in (*NARRATOR_NAMES, *narrator_names) if n and n.strip()}
        for member in members:
            self._members.append(member)
            for alias in (member.name, *member.aliases):
                key = normalize_name(alias)
                if key and key not in self._by_alias:
                    self._by_alias[key] = member

    @classmethod
    def from_card(cls, value: Any, narrator_names: Iterable[str] = ()) -> Cast:
        """Build a cast from a card's ``variables.cast``; bad entries are logged and skipped.

        Two shapes per entry: a bare voice name (``Innkeeper: onyx``) or a
        mapping with ``voice`` and optional ``delivery`` / ``aliases``. A card
        with a broken cast still loads — it just has fewer voices.
        """
        members: list[CastMember] = []
        if value is None:
            return cls(members, narrator_names)
        if not isinstance(value, Mapping):
            logger.warning("cast must be a mapping of character name to voice; ignoring %r", type(value).__name__)
            return cls(members, narrator_names)
        for raw_name, spec in value.items():
            name = str(raw_name).strip()
            if not name:
                continue
            member = _member_from_spec(name, spec)
            if member is None:
                logger.warning("cast entry %r has no voice; skipping it", name)
                continue
            members.append(member)
        return cls(members, narrator_names)

    def __bool__(self) -> bool:
        return bool(self._members)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._members)

    def is_narrator(self, name: str) -> bool:
        return normalize_name(name) in self._narrator

    def resolve(self, name: str) -> CastMember | None:
        """The member a bracketed name refers to, or ``None`` (narrator or unknown)."""
        return self._by_alias.get(normalize_name(name))

    def script(self, text: str) -> list[Utterance]:
        """Cut a reply into utterances at the ``[Name]`` tags, in reading order.

        Consecutive stretches in the same voice are merged, so a reply that
        tags the same character twice in a row costs one synthesis, not two.
        """
        utterances: list[Utterance] = []
        pos = 0
        speaker: CastMember | None = None
        for match in _BOUNDARY.finditer(text):
            _append(utterances, text[pos : match.start()], speaker)
            pos = match.end()
            name = match.group(1)
            if name is None:  # a line break: whoever was speaking has finished
                speaker = None
                continue
            name = name.strip()
            if self.is_narrator(name):
                speaker = None
                continue
            member = self.resolve(name)
            if member is None:
                logger.warning("no voice in the cast for %r; speaking it as the narrator", name)
            speaker = member
        _append(utterances, text[pos:], speaker)
        return utterances


def _member_from_spec(name: str, spec: Any) -> CastMember | None:
    if isinstance(spec, str):
        voice = spec.strip()
        return CastMember(name, voice) if voice else None
    if not isinstance(spec, Mapping):
        return None
    voice = str(spec.get("voice") or "").strip()
    if not voice:
        return None
    delivery = spec.get("delivery")
    aliases_raw = spec.get("aliases") or ()
    if isinstance(aliases_raw, str):
        aliases_raw = [aliases_raw]
    aliases: tuple[str, ...] = ()
    if isinstance(aliases_raw, Iterable):
        aliases = tuple(str(a).strip() for a in aliases_raw if str(a).strip())
    return CastMember(
        name=name,
        voice=voice,
        delivery=delivery.strip() if isinstance(delivery, str) and delivery.strip() else None,
        aliases=aliases,
    )


def _append(utterances: list[Utterance], text: str, member: CastMember | None) -> None:
    text = text.strip()
    if not text:
        return
    voice = member.voice if member else None
    delivery = member.delivery if member else None
    if utterances and (utterances[-1].voice, utterances[-1].delivery) == (voice, delivery):
        last = utterances[-1]
        utterances[-1] = Utterance(f"{last.text} {text}", last.speaker, voice, delivery)
        return
    utterances.append(Utterance(text, member.name if member else None, voice, delivery))


class CastVoiceBackend:
    """A `VoiceBackend` that speaks a tagged reply one character at a time.

    Wraps the speaking backend (the OpenAI or realtime one), so it sits inside
    a `DualVoiceBackend`: the browser page and console still get the reply
    once, tags and all, and only the audio is split.
    """

    def __init__(self, inner: VoiceBackend, cast: Cast) -> None:
        self._inner = inner
        self._cast = cast

    async def listen(self) -> str | None:
        return await self._inner.listen()

    def on_speech_start(self, callback: Callable[[], None] | None) -> None:
        self._inner.on_speech_start(callback)

    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        # A caller's explicit voice applies to the narrator's stretches; a cast
        # member's own voice and delivery always win for theirs.
        for utterance in self._cast.script(text):
            if utterance.speaker:
                logger.info("speaking as %s (%s)", utterance.speaker, utterance.voice)
                await self._inner.speak(utterance.text, voice=utterance.voice, delivery=utterance.delivery)
            else:
                await self._inner.speak(utterance.text, voice=voice, delivery=delivery)

    async def warm_up(self) -> None:
        warm = getattr(self._inner, "warm_up", None)
        if warm is not None:
            await warm()

    def interrupt(self) -> None:
        self._inner.interrupt()

    async def aclose(self) -> None:
        await self._inner.aclose()
