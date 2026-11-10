"""The listen gate: one boolean out of mode, hold and latch, with backend hooks on each edge."""

from __future__ import annotations

import asyncio

from fast_body.listen_gate import ALWAYS_ON, PUSH_TO_TALK, ListenGate, normalize_mode


class _Backend:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def on_listen_open(self, *, continuous: bool) -> None:
        self.calls.append(("open", continuous))

    async def on_listen_close(self, *, commit: bool) -> None:
        self.calls.append(("close", commit))

    async def on_listen_mode_changed(self, mode: str) -> None:
        self.calls.append(("mode", mode))


async def _settle() -> None:
    # run_coroutine_threadsafe lands on the loop a tick later.
    for _ in range(3):
        await asyncio.sleep(0)


def test_modes_normalize_to_the_two_known_values():
    assert normalize_mode(" Push_To_Talk ") == PUSH_TO_TALK
    assert normalize_mode("always_on") == ALWAYS_ON
    assert normalize_mode("whatever") == ALWAYS_ON
    assert normalize_mode(None) == ALWAYS_ON


def test_always_on_is_open_whatever_the_controls_do():
    gate = ListenGate(ALWAYS_ON)
    assert gate.enabled
    gate.set_held(True)
    gate.set_held(False)
    gate.set_latched(True)
    gate.set_latched(False)
    assert gate.enabled


def test_push_to_talk_starts_closed_and_opens_on_hold_or_latch():
    gate = ListenGate(PUSH_TO_TALK)
    assert not gate.enabled
    gate.set_held(True)
    assert gate.enabled and gate.held
    gate.set_held(False)
    assert not gate.enabled
    gate.toggle()
    assert gate.enabled and gate.latched
    gate.toggle()
    assert not gate.enabled


def test_a_hold_under_a_latch_does_not_close_the_gate_when_released():
    gate = ListenGate(PUSH_TO_TALK)
    gate.set_latched(True)
    gate.set_held(True)
    gate.set_held(False)
    assert gate.enabled


async def test_edges_reach_the_backend_hooks_on_the_loop():
    gate = ListenGate(PUSH_TO_TALK)
    backend = _Backend()
    gate.bind(asyncio.get_running_loop(), lambda: backend)

    gate.set_held(True)
    gate.set_held(True)  # no edge
    gate.set_held(False)
    await _settle()
    assert backend.calls == [("open", True), ("close", True)]

    backend.calls.clear()
    gate.set_latched(True)
    gate.set_held(True)  # already open: no edge
    gate.set_held(False)
    gate.set_latched(False)
    await _settle()
    assert backend.calls == [("open", False), ("close", True)]


async def test_a_mode_change_is_not_a_release():
    """Switching away from push_to_talk mid-hold must not commit the half-captured audio."""
    gate = ListenGate(PUSH_TO_TALK)
    backend = _Backend()
    gate.bind(asyncio.get_running_loop(), lambda: backend)

    gate.set_mode(ALWAYS_ON)  # opens the gate, with nothing to commit
    await _settle()
    assert backend.calls == [("open", False), ("mode", ALWAYS_ON)]
    assert gate.enabled and not gate.held

    backend.calls.clear()
    gate.set_mode(PUSH_TO_TALK)
    await _settle()
    assert backend.calls == [("close", False), ("mode", PUSH_TO_TALK)]
    assert not gate.enabled


async def test_backends_without_the_hooks_are_left_alone():
    gate = ListenGate(PUSH_TO_TALK)
    gate.bind(asyncio.get_running_loop(), lambda: object())
    gate.set_held(True)
    gate.set_held(False)
    await _settle()  # nothing raised, nothing scheduled


def test_unbound_gate_still_tracks_state():
    gate = ListenGate(PUSH_TO_TALK)
    gate.set_held(True)
    assert gate.enabled


def test_change_listener_sees_every_visible_change():
    gate = ListenGate(PUSH_TO_TALK)
    seen: list[dict] = []
    gate.set_change_listener(seen.append)
    gate.set_held(True)
    gate.set_held(True)  # nothing changed, nothing reported
    gate.set_latched(True)
    gate.set_held(False)  # still enabled through the latch, but the hold changed
    assert [(s["enabled"], s["held"], s["latched"]) for s in seen] == [
        (True, True, False),
        (True, True, True),
        (True, False, True),
    ]
    assert seen[-1]["mode"] == PUSH_TO_TALK


def test_a_failing_change_listener_does_not_break_the_gate():
    gate = ListenGate(PUSH_TO_TALK)

    def boom(_: dict) -> None:
        raise RuntimeError("page went away")

    gate.set_change_listener(boom)
    gate.set_held(True)
    assert gate.enabled
