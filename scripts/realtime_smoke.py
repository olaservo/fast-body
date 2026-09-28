"""Exercise the realtime voice backend against OpenAI, with no robot.

Synthesizes a sentence with OpenAI TTS, feeds it to the real
`RealtimeVoiceBackend` through a fake mic that yields what the SDK yields —
stereo float32 at 16 kHz, 30 ms at a time — and prints what `listen()` returns
plus the session events and their timings.

Covers the session handshake, our exact session config, the 24 kHz framing,
server VAD endpointing and the transcript events. What it cannot cover is
acoustic: the real mic, the room, and the robot hearing its own voice.

    OPENAI_API_KEY=… uv run python scripts/realtime_smoke.py [model …]

Costs a few cents a run (one TTS call plus a few seconds of transcription).
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any

import numpy as np

SENTENCE = "The quick brown fox jumps over the lazy dog, and then asks what time it is."
MODELS = ("gpt-transcribe", "gpt-live-transcribe", "gpt-4o-transcribe")


class FakeMedia:
    """The mic side of `MediaManager`: (N, 2) float32 at 16 kHz, 30 ms a time."""

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = list(chunks)

    def extend(self, chunks: list[np.ndarray]) -> None:
        self._chunks.extend(chunks)

    def get_input_audio_samplerate(self) -> int:
        return 16000

    def get_output_audio_samplerate(self) -> int:
        return 16000

    def get_audio_sample(self):
        # The SDK's read blocks for up to 20 ms; this also paces playback.
        time.sleep(0.03)
        return self._chunks.pop(0) if self._chunks else None

    def start_recording(self) -> None: ...
    def start_playing(self) -> None: ...
    def stop_recording(self) -> None: ...
    def stop_playing(self) -> None: ...
    def push_audio_sample(self, audio) -> None: ...
    def clear_player(self) -> None: ...


class FakeRobot:
    def __init__(self, media: FakeMedia) -> None:
        self.media = media


async def synthesize(sentence: str) -> np.ndarray:
    """Return `sentence` spoken, as mono float32 at 16 kHz."""
    from openai import AsyncOpenAI
    from scipy.signal import resample_poly

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = await client.audio.speech.create(
        model="gpt-4o-mini-tts", voice="cedar", input=sentence, response_format="pcm"
    )
    pcm24 = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32768.0
    return resample_poly(pcm24, 2, 3).astype(np.float32)


def as_mic_chunks(mono16k: np.ndarray, trailing_silence_s: float = 2.0) -> list[np.ndarray]:
    """Frame the utterance as mic chunks, with silence after so the VAD endpoints."""
    padded = np.concatenate([mono16k, np.zeros(int(16000 * trailing_silence_s), dtype=np.float32)])
    stereo = np.stack([padded, padded], axis=1)  # the ReSpeaker is 2-channel
    return [stereo[i : i + 480] for i in range(0, len(stereo), 480)]


def report_events(seen: list[tuple[float, str]], started: float) -> None:
    if not seen:
        print("    no session events at all")
        return
    counts: dict[str, int] = {}
    first: dict[str, float] = {}
    for at, kind in seen:
        counts[kind] = counts.get(kind, 0) + 1
        first.setdefault(kind, at - started)
    for kind in sorted(counts, key=lambda k: first[k]):
        short = kind.replace("conversation.item.input_audio_transcription", "transcription")
        print(f"    {first[kind]:6.1f}s  {short.replace('input_audio_buffer', 'buffer'):<34} x{counts[kind]}")


async def run_reconnect(chunks: list[np.ndarray]) -> None:
    """Drop a working session mid-run and check listening comes back.

    An app that goes deaf after one network blip looks exactly like a bad room,
    so this is worth knowing before blaming the acoustics.
    """
    os.environ["VOICE_BACKEND"] = "realtime"

    from fast_body.config import Config
    from fast_body.voice.realtime_backend import RealtimeVoiceBackend

    captured: list[Any] = []

    class Capturing(RealtimeVoiceBackend):
        async def _consume(self, conn):
            captured.append(conn)
            await super()._consume(conn)

    media = FakeMedia(list(chunks))
    backend = Capturing(Config(), FakeRobot(media))
    try:
        print(f"  first utterance:  {await backend.listen()!r}")

        print("  closing the websocket under it")
        await captured[-1].close()
        before = len(captured)

        media.extend(list(chunks))
        second = await asyncio.wait_for(backend.listen(), timeout=60)
        print(f"  second utterance: {second!r}")
        print(f"  sessions opened: {len(captured)} (reconnected: {len(captured) > before})")
    finally:
        await backend.aclose()


async def run_one(model: str, chunks: list[np.ndarray]) -> None:
    os.environ["VOICE_BACKEND"] = "realtime"
    os.environ["REALTIME_STT_MODEL"] = model

    from fast_body.config import Config
    from fast_body.voice.realtime_backend import RealtimeVoiceBackend

    seen: list[tuple[float, str]] = []

    class Traced(RealtimeVoiceBackend):
        """Records every server event on its way through, timings included."""

        async def _consume(self, conn):
            events = conn.__aiter__()

            class Tee:
                # Only closes over `events`/`seen`/`conn`, so `self` here is the
                # Tee instance and never the backend.
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    event = await events.__anext__()
                    seen.append((time.perf_counter(), event.type))
                    return event

                def __getattr__(self, name):
                    return getattr(conn, name)

            await super()._consume(Tee())

    backend = Traced(Config(), FakeRobot(FakeMedia(chunks)))
    started = time.perf_counter()
    try:
        heard = await asyncio.wait_for(backend.listen(), timeout=45)
    finally:
        await backend.aclose()

    if heard is None:
        print(f"  {model}: nothing heard (listen() hit its idle timeout)")
    else:
        audio_s = (len(chunks) * 480) / 16000
        print(f"  {model}: {heard!r}")
        print(f"    {time.perf_counter() - started:.1f}s wall for {audio_s:.1f}s of audio")
    report_events(seen, started)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="    %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "openai", "websockets", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("set OPENAI_API_KEY")

    print(f"synthesizing: {SENTENCE!r}")
    chunks = as_mic_chunks(await synthesize(SENTENCE))
    print(f"  {len(chunks)} mic chunks\n")

    if "--reconnect" in sys.argv:
        await run_reconnect(chunks)
        return

    for model in sys.argv[1:] or MODELS:
        print(f"model {model}")
        try:
            await run_one(model, list(chunks))
        except Exception as e:
            print(f"  {model}: FAILED {type(e).__name__}: {e}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
