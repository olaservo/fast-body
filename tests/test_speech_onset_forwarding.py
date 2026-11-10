"""The speech-onset hook reaches the audio backend through the wrappers."""

from __future__ import annotations

from types import SimpleNamespace

from fast_body.voice.base import supports_push_to_talk
from fast_body.voice.cast import CastVoiceBackend
from fast_body.voice.dual_backend import DualVoiceBackend
from fast_body.voice.null_backend import NullVoiceBackend
from fast_body.voice.text_backend import TextVoiceBackend


class _Inner:
    def __init__(self):
        self.callback = None

    def on_speech_start(self, callback):
        self.callback = callback


def test_wrappers_forward_the_hook_to_the_audio_backend():
    inner = _Inner()
    cb = lambda: None  # noqa: E731
    DualVoiceBackend(CastVoiceBackend(inner, cast=None), surface=SimpleNamespace()).on_speech_start(cb)
    assert inner.callback is cb


def test_backends_without_a_detector_accept_the_hook():
    for backend in (NullVoiceBackend(), TextVoiceBackend()):
        backend.on_speech_start(lambda: None)  # kept, never fired


# ── the push-to-talk hooks take the same path ─────────────────────────────
class _Gated(_Inner):
    def __init__(self):
        super().__init__()
        self.calls = []

    def supports_push_to_talk(self):
        return True

    async def on_listen_open(self, *, continuous):
        self.calls.append(("open", continuous))

    async def on_listen_close(self, *, commit):
        self.calls.append(("close", commit))

    async def on_listen_mode_changed(self, mode):
        self.calls.append(("mode", mode))


async def test_wrappers_forward_the_push_to_talk_hooks():
    inner = _Gated()
    outer = DualVoiceBackend(CastVoiceBackend(inner, cast=None), surface=SimpleNamespace())
    assert supports_push_to_talk(outer)
    await outer.on_listen_open(continuous=True)
    await outer.on_listen_close(commit=True)
    await outer.on_listen_mode_changed("always_on")
    assert inner.calls == [("open", True), ("close", True), ("mode", "always_on")]


async def test_wrappers_around_a_plain_backend_report_no_push_to_talk():
    outer = DualVoiceBackend(CastVoiceBackend(_Inner(), cast=None), surface=SimpleNamespace())
    assert not supports_push_to_talk(outer)
    await outer.on_listen_open(continuous=True)  # nothing to forward to; nothing raised
    assert not supports_push_to_talk(TextVoiceBackend())
    assert not supports_push_to_talk(NullVoiceBackend())
