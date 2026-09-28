"""Voice layer — a swappable transport for hearing and speaking.

The brain (fast-agent) is provider-agnostic and never touches audio. Everything
audio lives behind the `VoiceBackend` protocol; `build_backend` picks the one
`VOICE_BACKEND` names.
"""

from fast_body.voice.base import VoiceBackend, build_backend

__all__ = ["VoiceBackend", "build_backend"]
