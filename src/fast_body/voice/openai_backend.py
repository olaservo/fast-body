"""OpenAI voice backend: VAD + speech-to-text + text-to-speech.

Turn-based loop over the robot's audio I/O:

    listen()  mic float32 ─► resample 16 kHz ─► webrtcvad endpointing
              ─► WAV ─► OpenAI STT ─► user text
    speak()   text ─► Speaker.stream() (PCM at the provider's rate)
              ─► resample to speaker rate ─► robot.media.push_audio_sample

Discrete STT and TTS calls, so the same shape fits a local backend; the
realtime backend (realtime_backend.py) swaps only the listening half. The TTS
provider is a `Speaker` (voice/speaker.py); everything after the bytes arrive
is here and shared by every provider, barge-in included.

The mic buffer is flushed at the start of each `listen()` so the robot doesn't
transcribe the tail of its own speech. Echo cancellation itself comes from the
SDK's LOCAL audio path (webrtcdsp + webrtcechoprobe); it does not cover external
or Bluetooth speakers.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import time
import wave
from collections.abc import Callable
from typing import Any

import numpy as np

from fast_body.voice.speaker import Speaker, build_speaker

logger = logging.getLogger(__name__)

_VAD_RATE = 16000  # webrtcvad supports 8/16/32/48 kHz
_FRAME_MS = 30  # webrtcvad supports 10/20/30 ms
_FRAME_SAMPLES = _VAD_RATE * _FRAME_MS // 1000  # 480 samples / frame
# Enough to cover the longest reply the robot could have been playing while the
# mic kept recording; only a backstop against a buffer that never reports empty.
_MAX_FLUSH_CHUNKS = 10_000
# Silence pushed at warm-up so the player's own start is paid before the first reply.
_PLAYER_WARMUP_S = 0.2


class OpenAIVoiceBackend:
    """A `VoiceBackend` using OpenAI for STT and TTS over the robot's audio."""

    def __init__(self, config: Any, robot: Any, speaker: Speaker | None = None):
        import webrtcvad
        from openai import AsyncOpenAI

        self._cfg = config
        self._robot = robot
        self._client = AsyncOpenAI(api_key=config.openai_api_key)
        # Default 3, the most aggressive: the mic sits next to fans and motors,
        # so an invented sentence costs more than a clipped syllable.
        self._vad = webrtcvad.Vad(int(config.vad_aggressiveness))
        self._on_speech_start: Callable[[], None] | None = None

        self._in_rate = max(1, robot.media.get_input_audio_samplerate())
        self._out_rate = max(1, robot.media.get_output_audio_samplerate())
        self._in_frame = int(self._in_rate * _FRAME_MS / 1000)
        self._leftover = np.zeros(0, dtype=np.float32)
        self._started = False

        # Endpointing thresholds. Minimum voiced audio before anything is sent;
        # less is a click, not a word.
        self._min_speech_frames = max(1, int(config.stt_min_speech_ms / _FRAME_MS))
        self._min_rms = float(config.stt_min_rms)
        self._end_silence_frames = 25  # ~750 ms of silence ends the turn
        self._max_utterance_frames = int(15_000 / _FRAME_MS)  # 15 s safety cap
        self._idle_timeout_frames = int(30_000 / _FRAME_MS)  # give up after 30 s of pure silence
        # Barge-in: sustained speech (~450 ms) detected during playback cuts it off.
        self._barge_in = config.enable_barge_in
        self._lead_in_s = float(config.tts_lead_in_s)
        self._barge_in_frames = int(450 / _FRAME_MS)

        # The speaking half is a provider (voice/speaker.py); the voice and
        # delivery it defaults to are resolved there, once, at construction.
        self._speaker: Speaker = speaker if speaker is not None else build_speaker(config)

    def _ensure_started(self) -> None:
        if not self._started:
            self._robot.media.start_recording()
            self._robot.media.start_playing()
            self._started = True

    async def warm_up(self) -> None:
        """Pay the TTS and player cold starts before the first reply instead of on it.

        One discarded synthesis opens the provider connection while the brain is
        still attaching its servers, then a beat of silence goes through the
        same push the reply will take. Failure is logged and otherwise ignored.
        """
        started = time.perf_counter()
        try:
            self._ensure_started()
            async for _ in self._speaker.stream("Ready."):
                pass
        except Exception as e:
            logger.warning("TTS warm-up failed: %s", e)
            return
        logger.info("TTS warm-up took %.0f ms", (time.perf_counter() - started) * 1000)
        self._warm_up_player()

    def _warm_up_player(self) -> None:
        """Push `_PLAYER_WARMUP_S` of silence, shaped like the SDK's output; a no-op without audio."""
        media = self._robot.media
        samples = int(_PLAYER_WARMUP_S * self._out_rate)
        if samples < 2:  # no audio device: the SDK's -1 rate was clamped to 1 above
            return
        try:
            channels = int(media.get_output_channels())
        except Exception:
            channels = 1
        shape = (samples, channels) if channels > 1 else (samples,)
        started = time.perf_counter()
        try:
            media.push_audio_sample(np.zeros(shape, dtype=np.float32))
        except Exception as e:
            logger.debug("playback warm-up skipped: %s", e)
            return
        logger.info(
            "playback warm-up took %.0f ms (%.0f ms of silence)",
            (time.perf_counter() - started) * 1000,
            _PLAYER_WARMUP_S * 1000,
        )

    # ── listening ────────────────────────────────────────────────────────
    def on_speech_start(self, callback: Callable[[], None] | None) -> None:
        self._on_speech_start = callback

    def _notify_speech_start(self) -> None:
        """Tell the body someone began talking. A failing callback must not cost the turn."""
        if self._on_speech_start is None:
            return
        try:
            self._on_speech_start()
        except Exception as e:
            logger.debug("speech-start callback failed: %s", e)

    async def listen(self) -> str | None:
        self._ensure_started()
        self._flush_mic()

        voiced: list[np.ndarray] = []
        triggered = False
        silence_run = 0
        idle_run = 0

        while True:
            frame16k, is_speech = await self._next_frame()
            if not triggered:
                idle_run += 1
                if idle_run > self._idle_timeout_frames:
                    return None  # nobody's talking; let the caller loop
                if is_speech:
                    triggered = True
                    self._notify_speech_start()
                    voiced.append(frame16k)
                    silence_run = 0
                continue

            voiced.append(frame16k)
            if is_speech:
                silence_run = 0
            else:
                silence_run += 1
                if silence_run >= self._end_silence_frames:
                    break
            if len(voiced) >= self._max_utterance_frames:
                break

        # Require enough real speech to be a word. Whisper-family models answer
        # non-speech with confident invented text, often in another language, so
        # a fan blip that reaches the API comes back as a sentence.
        speech_frames = len(voiced) - silence_run
        if speech_frames < self._min_speech_frames:
            logger.debug("ignoring %d ms of voiced audio", speech_frames * _FRAME_MS)
            return None

        samples = np.concatenate(voiced).astype(np.int16)
        # webrtcvad keys on spectral shape, so steady machine noise can pass it
        # while carrying almost no level. Drop anything too quiet to be speech.
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        if rms < self._min_rms:
            logger.debug("ignoring quiet audio (rms %.4f < %.4f)", rms, self._min_rms)
            return None

        return await self._transcribe(samples.tobytes())

    def _flush_mic(self) -> None:
        """Drop buffered mic audio — mainly the robot's own last utterance.

        Recording runs throughout playback, so the buffer holds everything the mic
        heard while the robot talked. Drain it all: a partial drain leaves the
        robot's own voice at the head of the next turn, where it gets transcribed
        as if someone had spoken. The cap only stops an unbounded loop if the
        backend never reports empty.
        """
        self._leftover = np.zeros(0, dtype=np.float32)
        dropped = 0
        while dropped < _MAX_FLUSH_CHUNKS:
            if self._robot.media.get_audio_sample() is None:
                break
            dropped += 1
        else:
            logger.warning("mic buffer still had audio after %d chunks", dropped)
        if dropped:
            logger.debug("dropped %d buffered mic chunks", dropped)

    async def _next_frame(self) -> tuple[np.ndarray, bool]:
        """Return the next 30 ms frame as 16 kHz int16, plus its VAD verdict."""
        while len(self._leftover) < self._in_frame:
            sample = self._robot.media.get_audio_sample()
            if sample is None:
                await asyncio.sleep(0.01)
                continue
            self._leftover = np.concatenate([self._leftover, _to_mono(sample)])

        frame = self._leftover[: self._in_frame]
        self._leftover = self._leftover[self._in_frame :]

        frame16k = _resample(frame, self._in_rate, _VAD_RATE)
        frame16k = _fit(frame16k, _FRAME_SAMPLES)
        i16 = _to_int16(frame16k)
        try:
            is_speech = self._vad.is_speech(i16.tobytes(), _VAD_RATE)
        except Exception:
            is_speech = False
        return i16, is_speech

    async def _transcribe(self, pcm_int16: bytes) -> str | None:
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_VAD_RATE)
            w.writeframes(pcm_int16)
        kwargs: dict = {}
        if self._cfg.stt_language:
            # Pinning the language stops the model answering noise in Georgian.
            kwargs["language"] = self._cfg.stt_language
        try:
            resp = await self._client.audio.transcriptions.create(
                model=self._cfg.stt_model,
                file=("speech.wav", wav.getvalue(), "audio/wav"),
                **kwargs,
            )
            text = (resp.text or "").strip()
        except Exception as e:
            logger.error("transcription failed: %s", e)
            return None

        if not text or _is_hallucination(text):
            logger.debug("discarding transcript: %r", text)
            return None
        return text

    # ── speaking ───────────────────────────────────────────────────────────
    async def speak(self, text: str, *, voice: str | None = None, delivery: str | None = None) -> None:
        if not text.strip():
            return
        self._ensure_started()
        resampler = _StreamResampler(self._speaker.sample_rate, self._out_rate)
        requested = time.perf_counter()
        first_audio_at: float | None = None
        played = 0
        # Hold a lead-in before the first push. The player starts as soon as it has
        # audio, so handing it 150 ms and hoping the next chunk beats the network
        # drains it mid-word — heard on the robot as the first word glitching.
        lead_in = int(self._lead_in_s * self._out_rate)
        pending: list[np.ndarray] = []
        buffered = 0

        def _play(chunk: np.ndarray, final: bool = False) -> None:
            """Push a chunk, holding everything until the lead-in; `final` plays a reply shorter than it."""
            nonlocal first_audio_at, played, buffered
            if first_audio_at is None:
                if chunk.size:
                    pending.append(chunk)
                    buffered += chunk.size
                if not pending or (buffered < lead_in and not final):
                    return
                chunk = np.concatenate(pending)
                pending.clear()
                first_audio_at = time.perf_counter()
                logger.info("first audio %.0f ms after the TTS request", (first_audio_at - requested) * 1000)
            if not chunk.size:
                return
            self._robot.media.push_audio_sample(chunk)
            played += chunk.size

        try:
            # Push as it arrives, so the robot starts talking before the whole
            # reply is synthesized. Matters most on long replies.
            async for data in self._speaker.stream(text, voice=voice, delivery=delivery):
                _play(resampler.push(data))
            _play(resampler.flush(), final=True)
        except Exception as e:
            logger.error("TTS failed: %s", e)
            return
        if first_audio_at is None:
            return

        # Playback began at the first chunk, so only the unplayed remainder is
        # left to wait out.
        remaining = max(0.0, played / self._out_rate - (time.perf_counter() - first_audio_at))
        if not self._barge_in:
            # Block until playback should be finished so we don't hear ourselves.
            await asyncio.sleep(remaining + 0.2)
            return
        await self._play_with_barge_in(remaining)

    async def _play_with_barge_in(self, duration: float) -> None:
        """Wait out playback, but cut it off if the user talks over it.

        The robot's own audio path cancels the echo, so the open mic does not
        normally hear the robot itself. On an external or Bluetooth speaker it
        can, and self-interrupt — that is what `ENABLE_BARGE_IN=false` is for.
        """
        self._flush_mic()
        frame_dt = _FRAME_MS / 1000
        elapsed = 0.0
        voiced_run = 0
        while elapsed < duration + 0.2:
            _, is_speech = await self._next_frame()
            elapsed += frame_dt
            voiced_run = voiced_run + 1 if is_speech else 0
            if voiced_run >= self._barge_in_frames:
                self._flush_playback()
                logger.info("barge-in: user interrupted, stopping playback")
                return

    def interrupt(self) -> None:
        """Drop in-flight playback now (keyboard Escape / barge-in, audio half)."""
        self._flush_playback()

    def _flush_playback(self) -> None:
        """Drop any queued/playing output audio immediately."""
        media = self._robot.media
        for attempt in (
            lambda: media.clear_player(),
            lambda: media.audio.clear_player(),
            lambda: (media.stop_playing(), media.start_playing()),
        ):
            try:
                attempt()
                return
            except Exception:
                continue

    async def aclose(self) -> None:
        if not self._started:
            return
        try:
            self._robot.media.stop_recording()
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug("audio stop failed: %s", e)


def _to_mono(sample: Any) -> np.ndarray:
    """Take the mic channel out of an SDK audio buffer.

    `MediaManager.get_audio_sample()` returns `(num_samples, 2)`. Flattening that
    interleaves the two channels: a 440 Hz tone reads back as 220 Hz over twice
    the duration, so speech reaches the recognizer an octave down at half speed.
    The conversation app takes channel 0 here for the same reason.
    """
    audio = np.asarray(sample, dtype=np.float32)
    if audio.ndim == 1:
        return audio
    if audio.shape[0] < audio.shape[1]:  # channels-first
        audio = audio.T
    return np.ascontiguousarray(audio[:, 0])


def _to_int16(x: np.ndarray) -> np.ndarray:
    """Float audio in [-1, 1] as clipped 16-bit PCM."""
    return np.clip(x * 32768.0, -32768, 32767).astype(np.int16)


class _StreamResampler:
    """Resample a PCM stream chunk by chunk without clicking at the seams.

    `resample_poly` is FIR-based, so a chunk resampled on its own carries an edge
    transient at both ends. This keeps a margin of input either side of every
    block it emits — the leading margin is history already played, the trailing
    margin is held back until more arrives — so every sample returned was
    computed with real context on both sides.
    """

    # Comfortably longer than resample_poly's filter for the rates in play here.
    # Rounded up to a whole number of input samples per output sample: the output
    # grid is relative to the start of each array handed to resample_poly, so a
    # block that advances the stream by a fraction of `down` shifts the phase and
    # the pieces no longer line up.
    _MIN_MARGIN = 128

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self._src = src_rate
        self._dst = dst_rate
        g = math.gcd(src_rate, dst_rate)
        self._up = dst_rate // g
        self._down = src_rate // g
        self._margin = self._down * -(-self._MIN_MARGIN // self._down)
        self._buf = np.zeros(0, dtype=np.float32)
        self._head = 0  # leading samples of _buf already emitted, kept as history
        self._odd = b""  # a chunk can split a 16-bit sample

    def push(self, pcm_bytes: bytes) -> np.ndarray:
        """Feed 16-bit PCM; return whatever can safely be played now."""
        data = self._odd + pcm_bytes
        whole = len(data) - len(data) % 2
        self._odd = data[whole:]
        if whole:
            samples = np.frombuffer(data[:whole], dtype=np.int16).astype(np.float32) / 32768.0
            self._buf = np.concatenate([self._buf, samples])
        if self._src == self._dst:
            return self._drain()
        if len(self._buf) - self._head < 3 * self._margin:
            return np.zeros(0, dtype=np.float32)  # not enough for a margin either side
        return self._emit(hold_back=True)

    def flush(self) -> np.ndarray:
        """Return the remainder, including the held-back trailing margin."""
        if self._src == self._dst:
            return self._drain()
        if len(self._buf) <= self._head:
            return np.zeros(0, dtype=np.float32)
        return self._emit(hold_back=False)

    def _drain(self) -> np.ndarray:
        out, self._buf = self._buf, np.zeros(0, dtype=np.float32)
        return out

    def _emit(self, hold_back: bool) -> np.ndarray:
        from scipy.signal import resample_poly

        out = resample_poly(self._buf, self._up, self._down)
        if hold_back:
            # Keep the stream offset of _buf[0] a multiple of `down` so every
            # block resamples onto the same output grid.
            end_in = ((len(self._buf) - self._margin) // self._down) * self._down
            if end_in <= self._head:
                return np.zeros(0, dtype=np.float32)
        else:
            end_in = len(self._buf)
        start = self._head * self._up // self._down
        end = min(end_in * self._up // self._down, len(out))
        emitted = np.asarray(out[start:end], dtype=np.float32)

        if hold_back:
            self._buf = self._buf[end_in - self._margin :]
            self._head = self._margin
        else:
            self._buf = np.zeros(0, dtype=np.float32)
            self._head = 0
        return emitted


def _resample(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 audio with a rational factor (no-op if rates match)."""
    if src_rate == dst_rate or x.size == 0:
        return x.astype(np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(src_rate, dst_rate)
    return resample_poly(x, dst_rate // g, src_rate // g).astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples (VAD needs fixed-size frames)."""
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - x.size, dtype=np.float32)])


# Stock phrases Whisper-family models emit for non-speech audio — they come from
# the subtitle corpora the models were trained on. Matched on the whole
# transcript only, so someone can still say "thank you" to the robot.
_HALLUCINATIONS = frozenset(
    {
        "thank you.",
        "thanks for watching!",
        "thank you for watching.",
        "thank you for watching!",
        "you",
        "bye.",
        "please subscribe.",
        "subtitles by the amara.org community",
        "amara.org",
        "♪",
    }
)


def _is_hallucination(text: str) -> bool:
    """True for transcripts that are stock filler rather than something said."""
    stripped = text.strip().lower()
    if stripped in _HALLUCINATIONS:
        return True
    # A lone punctuation mark or single character is never a real utterance.
    return len(stripped.strip(".,!?…-–—'\"")) < 2
