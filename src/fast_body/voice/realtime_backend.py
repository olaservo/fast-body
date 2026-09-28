"""Realtime voice backend: streaming recognition in, discrete TTS out.

`listen()` reads from an OpenAI *transcription session* — a realtime WebSocket
that carries audio up and transcripts down, with no model in it. The server does
the endpointing; the brain stays fast-agent's. `speak()` is inherited from
`OpenAIVoiceBackend` unchanged.

    mic ─► 24 kHz PCM ─► ws ─► server VAD ─► transcript ─► queue ─► listen()

A streaming recognizer endpoints on audio it is already receiving, so it is
never handed an isolated clip of near-silence to interpret — the failure the
discrete backend guards against with three thresholds and a phrase list.

The Hugging Face realtime endpoint only serves full `realtime` sessions, so it
cannot be used for transcription alone.

The robot's LOCAL audio path cancels echo (the SDK builds webrtcdsp +
webrtcechoprobe into it), so with barge-in on the mic stays live through the
reply and you can talk over it. With barge-in off the mic is muted for the
duration instead — the fallback for external speakers, at the cost of a deaf
window as long as whatever the robot just said.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from typing import Any

import numpy as np

from fast_body.voice.openai_backend import OpenAIVoiceBackend, _is_hallucination, _resample, _to_int16, _to_mono

logger = logging.getLogger(__name__)

_SESSION_RATE = 24000  # the only input rate the realtime API accepts
_PUMP_MS = 50  # how much mic audio to batch per websocket frame
_IDLE_TIMEOUT_S = 30.0  # return None after this so the caller can loop
_RECONNECT_MAX_S = 30.0
_FAILURES_BEFORE_ALARM = 3


class RealtimeVoiceBackend(OpenAIVoiceBackend):
    """A `VoiceBackend` whose listening half is a realtime transcription session."""

    def __init__(self, config: Any, robot: Any):
        super().__init__(config, robot)
        self._heard: asyncio.Queue[str] = asyncio.Queue()
        self._speech_started = asyncio.Event()
        self._muted = asyncio.Event()
        self._session_task: asyncio.Task[None] | None = None
        self._session_confirmed = False
        self._closing = False
        self._pump_samples = max(1, int(self._in_rate * _PUMP_MS / 1000))
        self._listening = False  # a listen() is pending; speech_started then means a turn, not a barge-in

    # ── listening ────────────────────────────────────────────────────────
    async def listen(self) -> str | None:
        self._ensure_started()
        self._ensure_session()
        self._listening = True
        try:
            text = await asyncio.wait_for(self._heard.get(), timeout=_IDLE_TIMEOUT_S)
        except TimeoutError:
            return None  # nobody's talking; let the caller loop
        finally:
            self._listening = False
        # With barge-in on, anything said while the brain was thinking or the
        # robot was talking is still queued. Hand it over with this transcript
        # as one turn; taking only the oldest would answer each utterance one
        # turn late from then on, until the person waited a whole turn out.
        later = self._drain_heard()
        if later:
            logger.info("heard %d more utterance(s) since the last turn; answering them together", len(later))
            text = " ".join([text, *later])
        return text

    def _ensure_session(self) -> None:
        if self._session_task is None or self._session_task.done():
            self._session_task = asyncio.create_task(self._run_session())

    def _session_config(self) -> dict:
        """The transcription-session config sent on connect.

        `noise_reduction` filters before the server's VAD, which is the lever the
        discrete backend never had — it could only reject a clip after the fact.
        """
        transcription: dict = {"model": self._cfg.realtime_stt_model}
        if self._cfg.stt_language:
            transcription["language"] = self._cfg.stt_language
        return {
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": _SESSION_RATE},
                    "noise_reduction": {"type": self._cfg.realtime_noise_reduction},
                    "transcription": transcription,
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self._cfg.realtime_vad_threshold,
                        "prefix_padding_ms": self._cfg.realtime_prefix_padding_ms,
                        "silence_duration_ms": self._cfg.realtime_silence_ms,
                    },
                }
            },
        }

    async def _run_session(self) -> None:
        """Hold a transcription session open, reconnecting until `aclose()`."""
        delay = 1.0
        failures = 0
        while not self._closing:
            # Connecting proves nothing — the server accepts the socket and only
            # then rejects the config. `session.updated` is what says it works, so
            # the backoff resets on that rather than on connect.
            self._session_confirmed = False
            try:
                # Without intent the endpoint demands a `model` query parameter and
                # closes with `missing_model`; the transcription model belongs in
                # the session config, not the URL.
                async with self._client.realtime.connect(extra_query={"intent": "transcription"}) as conn:
                    await conn.session.update(session=self._session_config())  # type: ignore[arg-type]
                    pump = asyncio.create_task(self._pump_mic(conn))
                    try:
                        await self._consume(conn)
                    finally:
                        pump.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await pump
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("realtime session dropped: %s", e)
            if self._closing:
                return

            if self._session_confirmed:
                delay, failures = 1.0, 0
            else:
                failures += 1
                # A config the server rejects fails identically every time, and
                # listen() only ever returns None — which reads as a quiet room
                # rather than a broken app. Say so once.
                if failures == _FAILURES_BEFORE_ALARM:
                    logger.error(
                        "realtime voice input has failed %d times running without a working session; "
                        "nothing said to the robot is being heard. Set VOICE_BACKEND=openai to fall back.",
                        failures,
                    )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _RECONNECT_MAX_S)

    async def _pump_mic(self, conn: Any) -> None:
        """Feed the mic to the session in `_PUMP_MS` batches, resampled to 24 kHz.

        The SDK read is a blocking 20 ms GStreamer pull, so it runs in a thread:
        this pump shares the loop with the websocket, the web page and the brain
        for the whole session.
        """
        buf = np.zeros(0, dtype=np.float32)
        while True:
            if self._muted.is_set():
                await asyncio.sleep(0.02)
                continue
            sample = await asyncio.to_thread(self._robot.media.get_audio_sample)
            if sample is None:
                await asyncio.sleep(0.01)
                continue
            buf = np.concatenate([buf, _to_mono(sample)])
            if len(buf) < self._pump_samples:
                continue

            chunk, buf = buf, np.zeros(0, dtype=np.float32)
            pcm = _to_int16(_resample(chunk, self._in_rate, _SESSION_RATE))
            await conn.input_audio_buffer.append(audio=base64.b64encode(pcm.tobytes()).decode("utf-8"))

    async def _consume(self, conn: Any) -> None:
        """Turn session events into transcripts and the barge-in signal."""
        async for event in conn:
            if event.type == "session.updated":
                self._session_confirmed = True
                logger.info("realtime transcription session open (%s)", self._cfg.realtime_stt_model)
            elif event.type == "input_audio_buffer.speech_started":
                self._speech_started.set()
                if self._listening:
                    self._notify_speech_start()
            elif event.type == "input_audio_buffer.speech_stopped":
                self._speech_started.clear()
            elif event.type == "conversation.item.input_audio_transcription.delta":
                logger.debug("partial: %s", event.delta)
            elif event.type == "conversation.item.input_audio_transcription.completed":
                text = (event.transcript or "").strip()
                # Endpointing should make stock filler rare, but the phrase list
                # costs nothing and the failure it catches is loud.
                if not text or _is_hallucination(text):
                    logger.debug("discarding transcript: %r", text)
                    continue
                self._heard.put_nowait(text)
            elif event.type == "conversation.item.input_audio_transcription.failed":
                logger.warning("transcription failed: %s", getattr(event, "error", None))
            elif event.type == "error":
                error = getattr(event, "error", None)
                param = getattr(error, "param", "") or ""
                if param.startswith("session."):
                    # The server keeps the session open with our config rejected,
                    # so VAD still fires and no transcript ever arrives. Drop it.
                    raise RuntimeError(f"session config rejected at {param}: {getattr(error, 'message', error)}")
                logger.error("realtime error: %s", error)

    # ── speaking ─────────────────────────────────────────────────────────
    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        # The mic hears the robot, so stop feeding the recognizer while it talks
        # — unless barge-in wants those frames.
        mute = not self._barge_in
        if mute:
            self._muted.set()
        try:
            await super().speak(text, voice=voice, delivery=delivery)
        finally:
            if mute:
                self._flush_mic()
                dropped = self._drain_heard()
                if dropped:
                    logger.debug("dropped %d transcript(s) heard during playback", len(dropped))
                self._muted.clear()

    async def _play_with_barge_in(self, duration: float) -> None:
        """Wait out playback, but cut it off when the server hears speech."""
        self._speech_started.clear()
        try:
            await asyncio.wait_for(self._speech_started.wait(), timeout=duration + 0.2)
        except TimeoutError:
            return
        self._flush_playback()
        logger.info("barge-in: user interrupted, stopping playback")

    def _drain_heard(self) -> list[str]:
        """Take every queued transcript, oldest first."""
        taken: list[str] = []
        while True:
            try:
                taken.append(self._heard.get_nowait())
            except asyncio.QueueEmpty:
                return taken

    async def aclose(self) -> None:
        self._closing = True
        if self._session_task is not None:
            self._session_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._session_task
            self._session_task = None
        await super().aclose()
