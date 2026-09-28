"""The speech-onset hook reaches the audio backend through the wrappers."""

from __future__ import annotations

from types import SimpleNamespace

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
