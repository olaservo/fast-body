"""A server's question, answered at the table: MCP elicitation as a spoken choice.

An MCP server can pause a tool call on an elicitation form: a `choice` from a
fixed list of option ids and an optional `elaboration`. The MCP client drives the round trips itself (mcp 2.x
retries the tool call with the answer); all fast-body supplies is the handler,
passed to the body agent's decorator, and the handler's answer is the table's:

1. the page shows the options as buttons (a `choice` transcript entry),
2. the robot speaks the prompt and reads the options out,
3. whichever comes first — a tap on the page, a typed line, or the player's
   spoken words matched to an option — is the form's content.

An answer that matches no option is a decline; the words are kept and become
the next turn, so the player is still heard. Any server whose form is a single enum field gets the same treatment, and a
form that is anything else is declined, which every server has to handle.

The handler is registered at import time, before the voice or the page exist,
so it looks the table up in a module-level slot the app fills at startup.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# How long the table gets by default (CHOICE_TIMEOUT_S overrides). Players
# deliberate. Past this the form is cancelled, and a server turns that into
# "revisit the moment with the player".
DEFAULT_TIMEOUT_S = 300.0

_ORDINALS = {
    "1": 0, "one": 0, "first": 0,
    "2": 1, "two": 1, "second": 1,
    "3": 2, "three": 2, "third": 2,
    "4": 3, "four": 3, "fourth": 3,
    "5": 4, "five": 4, "fifth": 4,
    "6": 5, "six": 5, "sixth": 5,
}  # fmt: skip
_STOPWORDS = frozenset("the a an to of and or in on at it i we go let lets option".split())

_ids = itertools.count(1)


@dataclass(frozen=True)
class Option:
    id: str
    label: str
    description: str = ""


@dataclass
class Choice:
    """One pending question."""

    id: str
    prompt: str
    options: list[Option]
    allow_free_text: bool
    field: str  # the schema property the answer goes in
    elaboration_field: str | None
    # Created by the table when the question goes out, on the loop that waits.
    future: asyncio.Future | None = dataclasses.field(default=None)

    def option(self, option_id: str) -> Option | None:
        return next((o for o in self.options if o.id == option_id), None)


@dataclass(frozen=True)
class Answer:
    option: Option | None  # None: the player said something that fits no option
    elaboration: str = ""
    how: str = "page"  # page | typed | spoken | timeout


# ── the form ─────────────────────────────────────────────────────────────
def _enum_field(schema: Any) -> tuple[str, list[Option]] | None:
    """The one enum-shaped property of an elicitation form, or None."""
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        return None
    for name, prop in schema["properties"].items():
        if not isinstance(prop, dict):
            continue
        options: list[Option] = []
        if isinstance(prop.get("oneOf"), list):
            for item in prop["oneOf"]:
                if isinstance(item, dict) and isinstance(item.get("const"), str):
                    options.append(Option(item["const"], str(item.get("title") or item["const"])))
        elif isinstance(prop.get("enum"), list):
            raw_names = prop.get("enumNames")
            names: list[Any] = raw_names if isinstance(raw_names, list) else []
            for i, value in enumerate(prop["enum"]):
                if isinstance(value, str):
                    options.append(Option(value, str(names[i]) if i < len(names) else value))
        if len(options) >= 2:
            return str(name), options
    return None


def _text_field(schema: Any, skip: str) -> str | None:
    for name, prop in schema.get("properties", {}).items():
        if name != skip and isinstance(prop, dict) and prop.get("type") == "string" and "enum" not in prop:
            return str(name)
    return None


def _split_message(message: str, options: list[Option]) -> tuple[str, list[Option]]:
    """The prompt without the option list the server appended, plus descriptions.

    A server may append `<prompt>\\n\\nOptions:\\n- label: description`. The
    schema has no descriptions, so they are read back off the message.
    """
    head, sep, tail = message.partition("\n\nOptions:")
    if not sep:
        return message.strip(), options
    described = []
    lines = [ln.strip()[2:] for ln in tail.splitlines() if ln.strip().startswith("- ")]
    by_label = {}
    for line in lines:
        label, _, description = line.partition(": ")
        by_label[label.strip()] = description.strip()
    for option in options:
        described.append(Option(option.id, option.label, by_label.get(option.label, "")))
    return head.strip(), described


def parse(params: Any) -> Choice | None:
    """A Choice from form-mode elicitation params, or None if the form is not a choice."""
    schema = getattr(params, "requested_schema", None)
    found = _enum_field(schema)
    if found is None:
        return None
    name, options = found
    prompt, options = _split_message(str(getattr(params, "message", "") or ""), options)
    elaboration_field = _text_field(schema, skip=name)
    return Choice(
        id=f"choice-{next(_ids)}",
        prompt=prompt,
        options=options,
        allow_free_text=elaboration_field is not None,
        field=name,
        elaboration_field=elaboration_field,
    )


# ── matching what the player said ────────────────────────────────────────
def _words(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split() if w]


def match(text: str, options: list[Option]) -> Option | None:
    """The option a spoken or typed answer names, or None.

    In order of trust: the option id itself, an ordinal ("the second one",
    "two"), the whole label inside the answer, then most of the label's
    content words. Ties go to the earlier option, as read out.
    """
    words = _words(text)
    if not words:
        return None
    lowered = " ".join(words)
    for option in options:
        if option.id.lower() in words:
            return option
    for w in words:
        index = _ORDINALS.get(w)
        if index is not None and index < len(options):
            return options[index]
    for option in options:
        label = " ".join(_words(option.label))
        if label and label in lowered:
            return option
    best, best_score = None, 0.0
    for option in options:
        content = [w for w in _words(option.label) if w not in _STOPWORDS]
        if not content:
            continue
        hits = sum(1 for w in content if w in words)
        score = hits / len(content)
        if score > best_score:
            best, best_score = option, score
    return best if best_score >= 0.6 else None


def spoken_options(choice: Choice) -> str:
    """The question as the robot reads it: prompt, then numbered options."""
    parts = [choice.prompt] if choice.prompt else []
    for i, option in enumerate(choice.options, 1):
        line = f"{i}: {option.label}"
        if option.description:
            line += f", {option.description}"
        parts.append(line + ".")
    return " ".join(parts)


# ── the table ────────────────────────────────────────────────────────────
class Table:
    """Where a question goes: the page, the speaker, and the microphone."""

    def __init__(self, voice: Any, hub: Any, *, timeout_s: float = DEFAULT_TIMEOUT_S, cue: Any = None) -> None:
        self.voice = voice
        self.hub = hub
        self.timeout_s = timeout_s
        self.cue = cue
        self.pending: Choice | None = None

    def answer(self, choice_id: str, option_id: str, elaboration: str = "") -> bool:
        """The page answered. Called on the loop the question is waiting on."""
        choice = self.pending
        if choice is None or choice.id != choice_id or choice.future is None or choice.future.done():
            return False
        option = choice.option(option_id)
        if option is None:
            return False
        choice.future.set_result(Answer(option, elaboration.strip(), "page"))
        return True

    async def _listen(self, choice: Choice) -> Answer:
        """The player's words, matched to an option; the words are kept either way."""
        while True:
            text = await self.voice.listen()
            if not text or not text.strip():
                continue
            return Answer(match(text, choice.options), text.strip(), "spoken")

    async def ask(self, choice: Choice) -> Answer:
        loop = asyncio.get_running_loop()
        choice.future = loop.create_future()
        self.pending = choice
        try:
            if self.hub is not None:
                self.hub.post_choice(choice)
            if self.voice is not None:
                try:
                    await self.voice.speak(spoken_options(choice))
                except Exception as e:
                    logger.warning("could not read the options aloud: %s", e)
            if self.cue is not None:
                self.cue()
            deadline = loop.time() + self.timeout_s
            listening = asyncio.ensure_future(self._listen(choice)) if self.voice is not None else None
            try:
                while True:
                    waits: set[asyncio.Future] = {choice.future}
                    if listening is not None and not listening.done():
                        waits.add(listening)
                    done, _ = await asyncio.wait(
                        waits, timeout=max(0.0, deadline - loop.time()), return_when=asyncio.FIRST_COMPLETED
                    )
                    if not done:
                        return Answer(None, "", "timeout")
                    if choice.future in done:
                        return choice.future.result()
                    # The microphone spoke first, or failed. A failure is not an
                    # answer: keep waiting on the page until the deadline.
                    assert listening is not None
                    if listening.exception() is None:
                        return listening.result()
                    logger.warning("listening for the answer failed: %s", listening.exception())
                    listening = None
            finally:
                if listening is not None and not listening.done():
                    listening.cancel()
                    await asyncio.gather(listening, return_exceptions=True)
        finally:
            self.pending = None

    def waiting(self) -> bool:
        pending = self.pending
        return pending is not None and pending.future is not None and not pending.future.done()


TABLE: Table | None = None


def bind(table: Table | None) -> None:
    """Set the table the handler answers at. None detaches it."""
    global TABLE
    TABLE = table


def waiting() -> bool:
    """True while a question is out at the table; the turn is not stuck."""
    return TABLE is not None and TABLE.waiting()


async def handler(context: Any, params: Any) -> Any:
    """fast-agent's `ElicitationFnT`: answer a choice form from the table."""
    from mcp.types import ElicitResult

    table = TABLE
    if table is None or getattr(params, "url", None) is not None:
        return ElicitResult(action="decline")
    choice = parse(params)
    if choice is None:
        logger.info("declining an elicitation that is not a choice: %s", str(getattr(params, "message", ""))[:80])
        return ElicitResult(action="decline")

    answer = await table.ask(choice)
    hub = table.hub
    if answer.how == "timeout":
        logger.info("choice %s: nobody answered in %.0fs", choice.id, table.timeout_s)
        if hub is not None:
            hub.post_choice_answered(choice.id, None, "timeout")
        return ElicitResult(action="cancel")
    if answer.option is None:
        # Not one of the options. Decline the form and keep the words for
        # the next turn, so the brain still hears what was said.
        logger.info("choice %s: no option matched %r", choice.id, answer.elaboration)
        if hub is not None:
            hub.post_choice_answered(choice.id, None, answer.how)
            if answer.elaboration:
                hub.requeue(answer.elaboration)
        return ElicitResult(action="decline")
    content: dict[str, Any] = {choice.field: answer.option.id}
    if choice.elaboration_field and answer.elaboration and answer.elaboration.lower() != answer.option.label.lower():
        content[choice.elaboration_field] = answer.elaboration
    logger.info("choice %s: %s (%s)", choice.id, answer.option.label, answer.how)
    if hub is not None:
        hub.post_choice_answered(choice.id, answer.option.id, answer.how)
    return ElicitResult(action="accept", content=content)
