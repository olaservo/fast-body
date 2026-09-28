"""A server's elicitation form answered at the table: parsing, matching, the race, the route."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from fast_body import choices
from fast_body.web.hub import WebChatHub
from fast_body.web.server import WebChatServer

SCHEMA = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "string",
            "title": "Your choice",
            "oneOf": [
                {"const": "cellar", "title": "Go down to the cellar"},
                {"const": "study", "title": "Search the study"},
                {"const": "leave", "title": "Leave the house"},
            ],
        },
        "elaboration": {"type": "string", "title": "Anything to add? (optional)"},
    },
    "required": ["choice"],
}
MESSAGE = (
    "The house is silent. Where do you go?\n\nOptions:\n"
    "- Go down to the cellar: the smell is coming from there\n"
    "- Search the study\n"
    "- Leave the house: it is not too late"
)


def params(message=MESSAGE, schema=SCHEMA):
    return SimpleNamespace(message=message, requested_schema=schema, url=None)


class FakeVoice:
    """Speaks into a list; listens from a queue."""

    def __init__(self, heard: list[str | None] | None = None):
        self.spoken: list[str] = []
        self.heard: asyncio.Queue = asyncio.Queue()
        for h in heard or []:
            self.heard.put_nowait(h)

    async def speak(self, text, **kwargs):
        self.spoken.append(text)

    async def listen(self):
        return await self.heard.get()


# ── the form ──────────────────────────────────────────────────────────────
def test_parse_reads_the_options_and_their_descriptions():
    choice = choices.parse(params())
    assert choice is not None
    assert choice.prompt == "The house is silent. Where do you go?"
    assert [(o.id, o.label, o.description) for o in choice.options] == [
        ("cellar", "Go down to the cellar", "the smell is coming from there"),
        ("study", "Search the study", ""),
        ("leave", "Leave the house", "it is not too late"),
    ]
    assert (choice.field, choice.elaboration_field, choice.allow_free_text) == ("choice", "elaboration", True)


def test_parse_accepts_a_plain_enum_too():
    schema = {
        "type": "object",
        "properties": {"pick": {"type": "string", "enum": ["a", "b"], "enumNames": ["Ay", "Bee"]}},
    }
    choice = choices.parse(params("Pick one", schema))
    assert choice is not None
    assert [(o.id, o.label) for o in choice.options] == [("a", "Ay"), ("b", "Bee")]
    assert choice.allow_free_text is False


def test_parse_declines_a_form_that_is_not_a_choice():
    assert choices.parse(params("Your name?", {"type": "object", "properties": {"name": {"type": "string"}}})) is None
    assert choices.parse(params("?", None)) is None


def test_spoken_options_are_numbered():
    text = choices.spoken_options(choices.parse(params()))
    assert text.startswith(
        "The house is silent. Where do you go? 1: Go down to the cellar, the smell is coming from there."
    )
    assert "2: Search the study." in text and "3: Leave the house, it is not too late." in text


# ── matching ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "said, expected",
    [
        ("the second one", "study"),
        ("Two.", "study"),
        ("let's go down to the cellar", "cellar"),
        ("I search the study for the letters", "study"),
        ("we leave. This house is wrong.", "leave"),
        ("cellar", "cellar"),
        ("I light a cigarette and wait", None),
        ("", None),
    ],
)
def test_match(said, expected):
    options = choices.parse(params()).options
    found = choices.match(said, options)
    assert (found.id if found else None) == expected


# ── the table ─────────────────────────────────────────────────────────────
async def test_a_tap_on_the_page_answers():
    from mcp.types import ElicitResult

    hub = WebChatHub()
    sent = []
    hub._broadcast = sent.append  # type: ignore[method-assign]
    voice = FakeVoice()  # hears nothing
    table = choices.Table(voice, hub, timeout_s=5)
    choices.bind(table)
    try:
        task = asyncio.ensure_future(choices.handler(None, params()))
        await asyncio.sleep(0.05)
        assert choices.waiting()
        assert table.answer(table.pending.id, "study", "quietly") is True
        result = await task
    finally:
        choices.bind(None)
    assert isinstance(result, ElicitResult)
    assert (result.action, result.content) == ("accept", {"choice": "study", "elaboration": "quietly"})
    assert voice.spoken and voice.spoken[0].startswith("The house is silent.")
    assert [m["role"] for m in sent] == ["choice", "choice-answered"]
    assert sent[1]["answered"] == "study"
    (entry,) = hub.transcript()
    assert (entry["role"], entry["answered"], entry["how"]) == ("choice", "study", "page")
    assert not choices.waiting()


async def test_the_spoken_answer_wins_when_it_names_an_option():
    hub = WebChatHub()
    voice = FakeVoice([None, "  ", "we go down to the cellar, carefully"])
    choices.bind(choices.Table(voice, hub, timeout_s=5))
    try:
        result = await choices.handler(None, params())
    finally:
        choices.bind(None)
    assert result.action == "accept"
    assert result.content == {"choice": "cellar", "elaboration": "we go down to the cellar, carefully"}
    assert hub.transcript()[0]["how"] == "spoken"


async def test_words_that_fit_no_option_decline_and_are_kept_for_the_next_turn():
    hub = WebChatHub()
    await hub.start()
    voice = FakeVoice(["I light a cigarette and wait"])
    choices.bind(choices.Table(voice, hub, timeout_s=5))
    try:
        result = await choices.handler(None, params())
    finally:
        choices.bind(None)
    assert result.action == "decline"
    assert await asyncio.wait_for(hub.read_line(), 1) == "I light a cigarette and wait"
    assert hub.transcript()[0]["answered"] is None


async def test_nobody_answering_cancels():
    hub = WebChatHub()
    choices.bind(choices.Table(FakeVoice(), hub, timeout_s=0.05))
    try:
        result = await choices.handler(None, params())
    finally:
        choices.bind(None)
    assert result.action == "cancel"
    assert hub.transcript()[0]["how"] == "timeout"


async def test_a_failing_microphone_still_lets_the_page_answer():
    class DeafVoice(FakeVoice):
        async def listen(self):
            raise RuntimeError("no mic")

    hub = WebChatHub()
    table = choices.Table(DeafVoice(), hub, timeout_s=5)
    choices.bind(table)
    try:
        task = asyncio.ensure_future(choices.handler(None, params()))
        await asyncio.sleep(0.05)
        assert table.answer(table.pending.id, "leave") is True
        result = await task
    finally:
        choices.bind(None)
    assert result.content == {"choice": "leave"}


async def test_without_a_table_or_for_other_forms_the_handler_declines():
    choices.bind(None)
    assert (await choices.handler(None, params())).action == "decline"
    choices.bind(choices.Table(FakeVoice(), WebChatHub(), timeout_s=1))
    try:
        not_a_choice = params("Your name?", {"type": "object", "properties": {"name": {"type": "string"}}})
        assert (await choices.handler(None, not_a_choice)).action == "decline"
        url = SimpleNamespace(message="log in", url="https://x.test", requested_schema=None)
        assert (await choices.handler(None, url)).action == "decline"
    finally:
        choices.bind(None)


def test_answer_rejects_a_stale_or_unknown_option():
    table = choices.Table(None, None)
    assert table.answer("choice-x", "study") is False


# ── the route ─────────────────────────────────────────────────────────────
@pytest.fixture
def server():
    return WebChatServer(WebChatHub(), host="127.0.0.1", port=8099, token="s3cret")


@pytest.fixture
def client(server):
    return TestClient(server._build_app())


def test_choice_route_needs_a_running_brain(client):
    response = client.post("/apps/choice?token=s3cret", json={"id": "choice-1", "option": "study"})
    assert response.status_code == 503


def test_choice_route_answers_the_open_question(server, client):
    taken = []

    class FakeTable:
        def answer(self, choice_id, option, elaboration=""):
            taken.append((choice_id, option, elaboration))
            return True

    async def on_brain(coro, timeout):
        return await coro

    server._on_brain = on_brain  # type: ignore[method-assign]
    server.bind_brain("brain", None)
    choices.bind(FakeTable())  # type: ignore[arg-type]
    try:
        response = client.post(
            "/apps/choice?token=s3cret", json={"id": "choice-1", "option": "study", "elaboration": "x"}
        )
    finally:
        choices.bind(None)
    assert response.status_code == 200
    assert taken == [("choice-1", "study", "x")]


def test_choice_route_reports_a_question_already_closed(server, client):
    async def on_brain(coro, timeout):
        return await coro

    server._on_brain = on_brain  # type: ignore[method-assign]
    server.bind_brain("brain", None)
    choices.bind(None)
    response = client.post("/apps/choice?token=s3cret", json={"id": "choice-1", "option": "study"})
    assert response.status_code == 409


def test_page_draws_choices(client):
    page = client.get("/?token=s3cret").text
    assert "addChoice" in page and "/apps/choice" in page
