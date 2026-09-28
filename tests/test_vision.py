"""The vision reader sends one question and one JPEG, and hands back words."""

from __future__ import annotations

import base64
from types import SimpleNamespace

from fast_body.embodiment.vision import VisionReader


class _FakeResponses:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


def _reader(text: str = "  A six.  ") -> tuple[VisionReader, _FakeResponses]:
    responses = _FakeResponses(text)
    client = SimpleNamespace(responses=responses)
    return VisionReader(api_key="k", model="fake-vision", client=client), responses


async def test_describe_sends_the_frame_as_a_data_url_with_the_question() -> None:
    reader, responses = _reader()
    assert await reader.describe(b"\xff\xd8jpeg", "what number is on the die?") == "A six."
    call = responses.calls[0]
    assert call["model"] == "fake-vision"
    content = call["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "what number is on the die?"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"] == "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()
    assert content[1]["detail"] == "high"


async def test_an_empty_question_still_asks_something() -> None:
    reader, responses = _reader()
    await reader.describe(b"jpeg", "   ")
    assert responses.calls[0]["input"][0]["content"][0]["text"]


async def test_no_output_text_is_an_empty_answer() -> None:
    reader, responses = _reader()
    responses.create = _none_create  # type: ignore[assignment]
    assert await reader.describe(b"jpeg", "anything") == ""


async def _none_create(**kwargs):
    return SimpleNamespace(output_text=None)


def test_reader_builds_its_own_client_from_the_key(monkeypatch) -> None:
    import openai

    built = {}

    class _Client:
        def __init__(self, *, api_key, base_url):
            built["api_key"], built["base_url"] = api_key, base_url

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    reader = VisionReader(api_key="sk-test", model="m")
    assert reader.model == "m"
    assert built == {"api_key": "sk-test", "base_url": None}
