"""A second pair of eyes: answer one question about a camera frame.

`examine(question)` sends the frame to a small vision model and returns its
words, so the picture itself never enters the brain's conversation. A frame
that goes into the history is replayed on every later call and has to be
summarized away at compaction; a sentence about it costs a few tokens once.
The question makes the read specific — dice pips, the numbers on a sheet — which
a general "what do you see" turn is worse at.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You are the eyes of a small desk robot. Answer the question about the "
    "photo in one or two plain sentences. Read text and numbers exactly as "
    "printed. If the photo does not show what is asked about, say what it "
    "shows instead. Do not describe things the question did not ask about."
)


class VisionReader:
    """Asks an OpenAI vision model one question about a JPEG."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def describe(self, jpeg: bytes, question: str) -> str:
        data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
        response = await self._client.responses.create(
            model=self._model,
            instructions=_INSTRUCTIONS,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": question.strip() or "What is in front of you?"},
                        # High detail: the point is reading small print and pips.
                        {"type": "input_image", "image_url": data_url, "detail": "high"},
                    ],
                }
            ],
        )
        return (getattr(response, "output_text", None) or "").strip()
