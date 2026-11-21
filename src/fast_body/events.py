"""MCP Events: the card's subscriptions, and the turns an event starts.

A card lists the events it wants under `variables.events`:

    variables:
      events:
        - server: forge
          name: forge.your_move
          wake: true
        - server: forge
          name: forge.chat

fast-agent keeps each subscription, by push when the server offers it and by
poll otherwise, and hands what arrived to the model at the
start of the next turn, as a user message marked untrusted. An event from a
subscription with `wake: true` also starts a turn when nobody is speaking;
without it the event waits for the next turn the user starts. Wake is off by
default because an event's content comes from the server, not the user.

`EventWakes` tells the conversation loop when to start such a turn, and keeps
count of the events a failed turn drained so the loop tries again.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# What the brain is sent when an event starts the turn. The events themselves
# follow it as their own message.
WAKE_PROMPT = (
    "(Nobody spoke. An event arrived from a server you subscribe to, and it follows this message. "
    "Read the current state with the server's tools before acting, since the event may be stale. "
    "Act if it needs you, say aloud only what the people in the room should hear, then end your turn.)"
)
# The first line of the message fast-agent builds from the events.
EVENT_MESSAGE_PREFIX = "MCP event notifications from subscribed servers follow."

# Waits before a turn that failed with events in hand is tried again.
RETRY_DELAYS_S: tuple[float, ...] = (5.0, 15.0)


@dataclass(frozen=True)
class EventSpec:
    """One `variables.events` entry."""

    server: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    wake: bool = False


def parse_specs(value: Any) -> list[EventSpec]:
    """The card's `events` list, kept to the entries that name a server and an event."""
    if not isinstance(value, list):
        return []
    specs = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        server, name = entry.get("server"), entry.get("name")
        if not isinstance(server, str) or not server.strip() or not isinstance(name, str) or not name.strip():
            logger.warning("events: skipping an entry without a server and a name: %r", entry)
            continue
        arguments = entry.get("arguments")
        specs.append(
            EventSpec(
                server=server.strip(),
                name=name.strip(),
                arguments=dict(arguments) if isinstance(arguments, dict) else {},
                wake=entry.get("wake") is True,
            )
        )
    return specs


def is_event_text(text: str) -> bool:
    """Whether a user message is one this module or fast-agent wrote, not something a person said."""
    return text.startswith(WAKE_PROMPT) or text.startswith(EVENT_MESSAGE_PREFIX)


def manager_for(agent: Any) -> Any:
    """The agent's event subscription manager, or None on a fast-agent without one."""
    try:
        return agent.aggregator.events
    except Exception as e:
        logger.warning("events: this fast-agent has no MCP Events support (%s)", e)
        return None


class EventWakes:
    """The card's subscriptions on one manager, and when the loop should take a turn for them."""

    def __init__(self, manager: Any, specs: Iterable[EventSpec], *, clock: Any = time.monotonic) -> None:
        self._manager = manager
        self._specs: list[EventSpec | None] = list(specs)
        self._held: dict[int, Any] = {}  # spec index -> the subscription it got
        self._wake_keys: set[str] = set()
        self._owed = False  # a wake event arrived and no turn has finished since
        self._failures = 0
        self._retry_at: float | None = None
        self._clock = clock
        self._lock = asyncio.Lock()
        manager.add_listener(self._note)

    def _note(self, item: Any) -> None:
        if getattr(item, "subscription_key", None) in self._wake_keys:
            self._owed = True
        logger.info("event: %s %s from %s", getattr(item, "kind", "?"), getattr(item, "name", "?"), item.server_name)

    async def subscribe(self, attached: Iterable[str]) -> str:
        """Subscribe the entries whose server is attached. One line for the log.

        Called again after each later attach, so a server that was asleep at
        startup is picked up when it arrives, and one that was detached and
        attached again gets its subscriptions back. An entry the server does
        not offer is dropped; any other failure leaves it for the next call.
        """
        servers = set(attached)
        async with self._lock:
            done, dropped, waiting = [], [], []
            for index, spec in enumerate(self._specs):
                if spec is None:
                    continue
                label = f"{spec.server}/{spec.name}"
                if spec.server not in servers:
                    waiting.append(label)
                    continue
                try:
                    subscription = await self._manager.subscribe(
                        spec.server, spec.name, spec.arguments or None, wake=spec.wake
                    )
                except ValueError as e:
                    logger.warning("events: %s", e)
                    self._specs[index] = None
                    dropped.append(label)
                    continue
                except Exception as e:
                    logger.warning("events: could not subscribe to %s on %s: %s", spec.name, spec.server, e)
                    waiting.append(label)
                    continue
                key = subscription.key
                if spec.wake:
                    self._wake_keys.add(key)
                if self._held.get(index) is subscription:
                    # Still running from before. Its server may have just come
                    # back, so end any wait between polls.
                    self._manager.poll_now(key)
                    continue
                self._held[index] = subscription
                done.append(label + (" (wake)" if spec.wake else ""))
        parts = []
        if done:
            parts.append("subscribed " + ", ".join(done))
        if dropped:
            parts.append("not offered " + ", ".join(dropped))
        if waiting:
            parts.append("waiting for " + ", ".join(waiting))
        return "; ".join(parts) or "nothing to subscribe"

    async def wait(self) -> None:
        """Return when a turn should start: a wake event is in, or a failed turn is due again."""
        if self._retry_at is not None:
            await asyncio.sleep(max(0.0, self._retry_at - self._clock()))
            return
        await self._manager.wait_for_wake()

    def turn_ok(self) -> None:
        """A turn finished, so whatever it drained reached the brain."""
        self._owed = False
        self._failures = 0
        self._retry_at = None

    def turn_failed(self) -> bool:
        """A turn failed. True when events it drained are still owed a turn.

        The drain happens inside the turn, so a turn that timed out or came
        back as a provider error has usually taken the events with it. After
        the last retry they are dropped, or `wait()` would return at once
        for ever.
        """
        if not self._owed:
            self._retry_at = None
            return False
        if self._failures >= len(RETRY_DELAYS_S):
            dropped = len(self._manager.take())
            logger.warning("events: giving up on a turn after %d tries; %d events dropped", self._failures + 1, dropped)
            self.turn_ok()
            return False
        self._retry_at = self._clock() + RETRY_DELAYS_S[self._failures]
        self._failures += 1
        return True

    def drop(self) -> None:
        """Forget what is waiting, for a turn the user cut short."""
        self._manager.take()
        self.turn_ok()
