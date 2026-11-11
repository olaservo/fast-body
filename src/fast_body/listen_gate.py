"""Single source of truth for whether the robot is listening (push-to-talk).

``always_on`` (the default) keeps the gate permanently open, so the app behaves
as it always has. In ``push_to_talk`` the gate is driven by control sources: a
hand holding an antenna (`app.py`'s hold loop) and the hold/latch button on the
chat page (`web/hub.py`). They all funnel into one boolean:

    enabled = (mode == always_on) OR latched OR held

- ``held``    — an antenna, or the page's button, is physically down (hold-to-talk).
- ``latched`` — the page's toggle is on: tap to open, tap again to send.

The voice backend reads ``enabled`` (through the hooks below) to drop mic frames
while the gate is closed. Each false<->true transition schedules the backend's
``on_listen_open`` / ``on_listen_close`` so it can finalise the turn: the
realtime backend commits the audio buffer on close. A mode change fires
``on_listen_mode_changed`` as well, since server-side endpointing is fixed when
the session connects.

Ported from the conversation app's `listen_gate.py`; the handler there is the
voice backend here, and the hooks are looked up rather than assumed, so a
backend without them (text, the discrete OpenAI one) just never hears from it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

ALWAYS_ON = "always_on"
PUSH_TO_TALK = "push_to_talk"
LISTEN_MODES = (ALWAYS_ON, PUSH_TO_TALK)


def normalize_mode(value: str | None) -> str:
    """Coerce an env/UI string to a known listen mode (defaulting to always_on)."""
    return PUSH_TO_TALK if (value or "").strip().lower() == PUSH_TO_TALK else ALWAYS_ON


class ListenGate:
    """Thread-safe gate that decides when mic audio reaches the voice backend."""

    def __init__(self, mode: str = ALWAYS_ON) -> None:
        self._mode = normalize_mode(mode)
        self._held = False
        self._latched = False
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._get_backend: Callable[[], Any] | None = None
        self._on_change: Callable[[dict[str, Any]], None] | None = None
        self._enabled = self._compute_enabled()

    def bind(self, loop: asyncio.AbstractEventLoop, get_backend: Callable[[], Any]) -> None:
        """Attach the app loop and the accessor for the live voice backend.

        Until bound, the gate still tracks state but fires no backend hooks.
        """
        self._loop = loop
        self._get_backend = get_backend

    def set_change_listener(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """Call ``callback(snapshot())`` after every visible change (any thread)."""
        self._on_change = callback

    # ── state ────────────────────────────────────────────────────────────
    def _compute_enabled(self) -> bool:
        return self._mode == ALWAYS_ON or self._latched or self._held

    @property
    def enabled(self) -> bool:
        """Whether mic audio should currently flow to the backend."""
        return self._enabled

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def held(self) -> bool:
        return self._held

    @property
    def latched(self) -> bool:
        return self._latched

    def snapshot(self) -> dict[str, Any]:
        """The state the page shows."""
        return {"mode": self._mode, "enabled": self._enabled, "held": self._held, "latched": self._latched}

    # ── mutations (safe to call from any thread) ──────────────────────────
    def set_mode(self, mode: str) -> None:
        """Switch listen mode; leaving push_to_talk clears the momentary hold."""
        with self._lock:
            new_mode = normalize_mode(mode)
            if new_mode == self._mode:
                return
            logger.info("listen mode -> %s", new_mode)
            self._mode = new_mode
            if new_mode == ALWAYS_ON:
                self._held = False
            # A mode change is not a release: nothing was captured to send.
            self._apply_locked(continuous=False, commit=False)
            self._schedule("on_listen_mode_changed", new_mode)
        self._notify()

    def set_held(self, held: bool, *, commit: bool = True) -> None:
        """Momentary push-to-talk control down (True) / up (False).

        ``commit=False`` on a release drops what the hold captured.
        """
        with self._lock:
            if bool(held) == self._held:
                return
            self._held = bool(held)
            # A momentary press with no latch => capture continuously until release.
            self._apply_locked(continuous=self._held and not self._latched, commit=commit)
        self._notify()  # the page shows the hold itself, not only the gate

    def set_latched(self, latched: bool) -> None:
        """Set the latching toggle explicitly (the page's tap)."""
        with self._lock:
            if bool(latched) == self._latched:
                return
            self._latched = bool(latched)
            self._apply_locked(continuous=False)
        self._notify()

    def toggle(self) -> None:
        """Flip the latching toggle."""
        with self._lock:
            self._latched = not self._latched
            logger.info("listen latch -> %s", self._latched)
            self._apply_locked(continuous=False)
        self._notify()

    # ── internals ────────────────────────────────────────────────────────
    def _notify(self) -> None:
        callback = self._on_change
        if callback is None:
            return
        try:
            callback(self.snapshot())
        except Exception as e:
            logger.debug("listen gate change listener failed: %s", e)

    def _apply_locked(self, *, continuous: bool, commit: bool = True) -> bool:
        new_enabled = self._compute_enabled()
        if new_enabled == self._enabled:
            return False
        self._enabled = new_enabled
        if new_enabled:
            self._schedule("on_listen_open", continuous=continuous)
        else:
            self._schedule("on_listen_close", commit=commit)
        return True

    def _schedule(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Run the backend's `hook` on the app loop, if it is bound and has one."""
        loop, get_backend = self._loop, self._get_backend
        if loop is None or get_backend is None:
            return
        backend = get_backend()
        method = getattr(backend, hook, None)
        if backend is None or method is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(method(*args, **kwargs), loop)
        except Exception as e:  # never let a control source crash on a dead loop
            logger.debug("listen gate hook %s not scheduled: %s", hook, e)
