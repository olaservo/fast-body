"""Exercise the realtime voice backend against OpenAI, with no robot.

Synthesizes a sentence with OpenAI TTS, feeds it to the real
`RealtimeVoiceBackend` through a fake mic that yields what the SDK yields —
stereo float32 at 16 kHz, 30 ms at a time — and prints what `listen()` returns
plus the session events and their timings.

Covers the session handshake, our exact session config, the 24 kHz framing,
server VAD endpointing and the transcript events. What it cannot cover is
acoustic: the real mic, the room, and the robot hearing its own voice.

    OPENAI_API_KEY=… uv run python scripts/realtime_smoke.py [model …]
    OPENAI_API_KEY=… uv run python scripts/realtime_smoke.py --reconnect
    OPENAI_API_KEY=… uv run python scripts/realtime_smoke.py --ptt [model]
    OPENAI_API_KEY=… uv run python scripts/realtime_smoke.py --ptt-silent

`--ptt` runs the push-to-talk session (no turn detection): the gate opens, the
sentence plays, the gate closes and commits, and the transcript should follow.
It also says whether the server ran a VAD of its own despite the config, which
is what decides whether the release has anything left to commit. `--ptt-silent`
holds the gate open over a second of silence and expects nothing to be sent.

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


class HeldMedia(FakeMedia):
    """A mic that yields silence until the hold starts, then the utterance.

    The gated pump still reads the mic (and drops the audio), so the utterance
    has to be withheld until the gate is open or it would be read into the void.
    """

    def __init__(self, chunks: list[np.ndarray]) -> None:
        super().__init__(chunks)
        self.holding = False
        self.silence = np.zeros((480, 2), dtype=np.float32)

    def get_audio_sample(self):
        time.sleep(0.03)
        if not self.holding:
            return self.silence
        return self._chunks.pop(0) if self._chunks else self.silence

    def drained(self) -> bool:
        return not self._chunks


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


async def run_ptt(chunks: list[np.ndarray], model: str | None, *, silent: bool) -> None:
    """Hold the gate open over the utterance (or over silence), release, and see what comes back."""
    os.environ["VOICE_BACKEND"] = "realtime"
    os.environ["LISTEN_MODE"] = "push_to_talk"
    if model:
        os.environ["REALTIME_PTT_STT_MODEL"] = model

    from fast_body.config import Config
    from fast_body.voice.realtime_backend import RealtimeVoiceBackend

    seen: list[tuple[float, str]] = []

    class Traced(RealtimeVoiceBackend):
        async def _consume(self, conn):
            events = conn.__aiter__()

            class Tee:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    event = await events.__anext__()
                    seen.append((time.perf_counter(), event.type))
                    return event

                def __getattr__(self, name):
                    return getattr(conn, name)

            await super()._consume(Tee())

    held_chunks = [np.zeros((480, 2), dtype=np.float32)] * 33 if silent else list(chunks)
    media = HeldMedia(held_chunks)
    backend = Traced(Config(), FakeRobot(media))
    print(f"  model {Config().realtime_ptt_stt_model}, {'a second of silence' if silent else 'the sentence'}")
    started = time.perf_counter()
    try:
        heard = asyncio.ensure_future(backend.listen())
        for _ in range(300):  # the session has to be up before the hold means anything
            if backend._session_confirmed:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("no session.updated within 15 s")
        print(f"  session open after {time.perf_counter() - started:.1f}s; opening the gate")

        await backend.on_listen_open(continuous=True)
        media.holding = True
        while not media.drained():
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.3)  # the last pump batch
        print(
            f"  hold: {backend._ptt_ms_since_open:.0f} ms pumped, "
            f"local VAD heard speech: {backend._ptt_speech_since_open}, "
            f"server ended the turn itself: {backend._ptt_server_turn_since_open}"
        )
        released = time.perf_counter()
        await backend.on_listen_close(commit=True)

        try:
            text = await asyncio.wait_for(heard, timeout=8.0 if silent else 30.0)
        except TimeoutError:
            heard.cancel()
            text = None
        if text is None:
            print(
                "  no transcript" + (" (correct: nothing was said)" if silent else " — the release committed nothing?")
            )
        else:
            print(f"  transcript {time.perf_counter() - released:.1f}s after release: {text!r}")
    finally:
        await backend.aclose()
    report_events(seen, started)


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
    if "--ptt" in sys.argv or "--ptt-silent" in sys.argv:
        rest = [a for a in sys.argv[1:] if not a.startswith("--")]
        await run_ptt(chunks, rest[0] if rest else None, silent="--ptt-silent" in sys.argv)
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
