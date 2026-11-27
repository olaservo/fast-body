"""MCP Events: the card's subscriptions and the turns an event starts."""

import asyncio
from types import SimpleNamespace

from fast_body import events
from fast_body.app import FastBodyCore
from fast_body.events import EventSpec, EventWakes, parse_specs
from fast_body.personality import load_personality
from tests.test_turn_timeout import FakeMovement


class FakeManager:
    """The parts of fast-agent's EventSubscriptionManager the loop uses."""

    def __init__(self, offered=("forge.your_move", "forge.chat")):
        self.offered = set(offered)
        self.subscribed: list[tuple] = []
        self.running: dict = {}
        self.polled: list = []
        self.inbox: list = []
        self.listeners: list = []
        self._wake = asyncio.Event()

    def add_listener(self, listener):
        self.listeners.append(listener)

    async def subscribe(self, server, name, arguments=None, *, wake=False):
        if name not in self.offered:
            raise ValueError(f"Server '{server}' does not offer event '{name}'")
        key = f"{server}/{name}"
        if key not in self.running:
            self.subscribed.append((server, name, arguments, wake))
            self.running[key] = SimpleNamespace(key=key, server_name=server)
        return self.running[key]

    def poll_now(self, key):
        self.polled.append(key)
        return key in self.running

    def detach(self, server):
        self.running = {k: s for k, s in self.running.items() if s.server_name != server}

    async def wait_for_wake(self):
        await self._wake.wait()

    def take(self):
        items, self.inbox = self.inbox, []
        self._wake.clear()
        return items

    def deliver(self, server, name, *, wake=True):
        item = SimpleNamespace(kind="event", name=name, server_name=server, subscription_key=f"{server}/{name}")
        self.inbox.append(item)
        if wake:
            self._wake.set()
        for listener in self.listeners:
            listener(item)


def test_specs_keep_only_entries_with_a_server_and_a_name():
    specs = parse_specs(
        [
            {"server": "forge", "name": "forge.your_move", "wake": True},
            {"server": "forge", "name": "forge.chat", "arguments": {"room": "a"}, "wake": "yes"},
            {"server": "forge"},
            "forge.chat",
        ]
    )
    assert specs == [
        EventSpec("forge", "forge.your_move", {}, True),
        EventSpec("forge", "forge.chat", {"room": "a"}, False),  # wake is on only for a literal true
    ]
    assert parse_specs(None) == []


def test_card_events_come_from_variables(tmp_path, monkeypatch):
    (tmp_path / "seat.md").write_text(
        "---\ntype: agent\nvariables:\n  events:\n    - server: forge\n      name: forge.your_move\n      wake: true\n---\nSeat.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_BODY_PERSONALITIES_DIR", str(tmp_path))
    assert load_personality("seat").events == [EventSpec("forge", "forge.your_move", {}, True)]


async def test_subscribing_waits_for_the_server_and_drops_what_it_does_not_offer():
    manager = FakeManager()
    wakes = EventWakes(manager, [EventSpec("forge", "forge.your_move", wake=True), EventSpec("forge", "forge.nothing")])

    assert await wakes.subscribe([]) == "waiting for forge/forge.your_move, forge/forge.nothing"
    assert manager.subscribed == []

    line = await wakes.subscribe(["forge"])
    assert line == "subscribed forge/forge.your_move (wake); not offered forge/forge.nothing"
    assert manager.subscribed == [("forge", "forge.your_move", None, True)]
    assert await wakes.subscribe(["forge"]) == "nothing to subscribe"
    assert manager.polled == ["forge/forge.your_move"]


async def test_the_subscribe_line_names_the_delivery_mode():
    manager = FakeManager()
    wakes = EventWakes(manager, [EventSpec("forge", "forge.your_move", wake=True), EventSpec("forge", "forge.chat")])
    subscribe = manager.subscribe

    async def with_delivery(server, name, arguments=None, *, wake=False):
        subscription = await subscribe(server, name, arguments, wake=wake)
        subscription.delivery = "push" if wake else "poll"
        return subscription

    manager.subscribe = with_delivery
    assert await wakes.subscribe(["forge"]) == "subscribed forge/forge.your_move (push, wake), forge/forge.chat (poll)"


async def test_a_server_attached_again_gets_its_subscriptions_back():
    manager = FakeManager()
    wakes = EventWakes(manager, [EventSpec("forge", "forge.your_move", wake=True)])
    await wakes.subscribe(["forge"])

    manager.detach("forge")  # fast-agent ends a detached server's subscriptions
    assert await wakes.subscribe([]) == "waiting for forge/forge.your_move"
    assert await wakes.subscribe(["forge"]) == "subscribed forge/forge.your_move (wake)"
    assert len(manager.subscribed) == 2


async def test_a_failed_turn_is_retried_then_the_events_are_dropped(monkeypatch):
    monkeypatch.setattr(events, "RETRY_DELAYS_S", (0.0, 0.0))
    manager = FakeManager()
    wakes = EventWakes(manager, [EventSpec("forge", "forge.your_move", wake=True)])
    await wakes.subscribe(["forge"])

    assert wakes.turn_failed() is False  # nothing arrived, so nothing is owed

    manager.deliver("forge", "forge.your_move")
    manager.take()  # the turn drained it, then failed
    assert wakes.turn_failed() is True
    await asyncio.wait_for(wakes.wait(), 1)  # due again although the inbox is empty
    assert wakes.turn_failed() is True
    manager.deliver("forge", "forge.your_move")
    assert wakes.turn_failed() is False  # third failure: give up
    assert manager.inbox == [] and not manager._wake.is_set()


class WakeVoice:
    """A voice whose listen blocks until a line is fed."""

    def __init__(self):
        self.lines: asyncio.Queue = asyncio.Queue()
        self.spoken: list[str] = []
        self.cancelled = 0

    async def listen(self):
        try:
            return await self.lines.get()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    async def speak(self, text, **_):
        self.spoken.append(text)

    def interrupt(self):
        pass


class Brain:
    def __init__(self, manager, replies):
        self.agent = SimpleNamespace(name="body", message_history=[])
        self.manager = manager
        self.replies = list(replies)
        self.sent: list[str] = []

    def resolve_agent(self, _name):
        return self.agent

    async def send(self, text):
        self.sent.append(text)
        self.manager.take()  # fast-agent drains the inbox before the model call
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _core(voice, manager) -> FastBodyCore:
    core = FastBodyCore.__new__(FastBodyCore)
    core.voice = voice
    core.movement = FakeMovement()
    core.gaze = None
    core.web_hub = None
    core.stop_event = asyncio.Event()
    core._stopping = asyncio.Event()
    core._turn_speech = None
    core.config = SimpleNamespace(personality="default", voice_backend="realtime", brain_timeout_s=5)
    core._wakes = EventWakes(manager, [EventSpec("forge", "forge.your_move", wake=True)])
    return core


async def _run(core, brain, until, timeout=2.0):
    task = asyncio.ensure_future(core._converse(brain))
    try:
        async with asyncio.timeout(timeout):
            while not until():
                await asyncio.sleep(0.01)
    finally:
        core.request_stop()
        await asyncio.wait_for(task, 2)


async def test_an_event_starts_a_turn_while_nobody_is_speaking():
    manager, voice = FakeManager(), WakeVoice()
    core = _core(voice, manager)
    await core._wakes.subscribe(["forge"])
    brain = Brain(manager, ["I attack with everything."])

    asyncio.get_running_loop().call_later(0.05, manager.deliver, "forge", "forge.your_move")
    await _run(core, brain, lambda: voice.spoken)

    assert brain.sent == [events.WAKE_PROMPT]
    assert voice.spoken == ["I attack with everything."]
    assert voice.cancelled >= 1, "the listen kept running under the event turn"


async def test_an_event_mid_sentence_waits_for_the_words():
    manager, voice = FakeManager(), WakeVoice()
    core = _core(voice, manager)
    await core._wakes.subscribe(["forge"])
    brain = Brain(manager, ["Your turn it is."])
    loop = asyncio.get_running_loop()

    loop.call_later(0.02, core._on_speech_start)
    loop.call_later(0.05, manager.deliver, "forge", "forge.your_move")
    loop.call_later(0.15, voice.lines.put_nowait, "whose turn is it")
    await _run(core, brain, lambda: voice.spoken)

    assert brain.sent == ["whose turn is it"]  # one turn; the event rode along
    assert manager.inbox == []


async def test_a_failed_event_turn_is_silent_and_retried(monkeypatch):
    monkeypatch.setattr(events, "RETRY_DELAYS_S", (0.0, 0.0))
    manager, voice = FakeManager(), WakeVoice()
    core = _core(voice, manager)
    await core._wakes.subscribe(["forge"])
    brain = Brain(manager, [RuntimeError("provider down"), "Done."])

    asyncio.get_running_loop().call_later(0.05, manager.deliver, "forge", "forge.your_move")
    await _run(core, brain, lambda: voice.spoken)

    assert brain.sent == [events.WAKE_PROMPT, events.WAKE_PROMPT]
    assert voice.spoken == ["Done."]


async def test_without_events_the_loop_listens_as_before():
    voice = WakeVoice()
    core = _core(voice, FakeManager())
    core._wakes = None
    voice.lines.put_nowait("hello")
    assert await core._listen_or_wake() == ("hello", False)


async def test_the_cards_missing_servers_are_said_once_at_start():
    voice = WakeVoice()
    core = _core(voice, FakeManager())

    await core._say_card_problems()  # nothing missing, nothing said
    assert voice.spoken == []

    core._card_problems = ({"name": "keeper", "reason": "503"}, {"name": "game-state", "reason": "not set up"})
    await core._say_card_problems()
    assert voice.spoken == ["I can't reach keeper and game-state right now, so some of what I do won't work."]
