"""fast-body application — wires the brain, the body, and the voice loop together.

`FastBodyCore` owns the conversation loop. `FastBodyApp` is the thin
`ReachyMiniApp` entry point the Reachy Mini daemon launches; the standalone CLI
in `main.py` drives the same core.

The loop is intentionally simple — that's the point of pushing all the cleverness
into the brain (fast-agent) and the voice layer:

    listen → brain.send(text) → speak    (repeat until stopped)

The brain calls the robot-body tools (look/emotion/dance/…) on its own while it
reasons; those run on the MovementManager thread, so motion and speech overlap.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import threading
import time
from collections import Counter
from collections.abc import Callable, Coroutine
from typing import Any

import numpy as np
from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini.reachy_mini import SLEEP_HEAD_POSE
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import distance_between_poses

from fast_body import choices, mcp_servers, memory, skills
from fast_body.config import ENV_FILE, Config, config_sources
from fast_body.embodiment import cues
from fast_body.embodiment.context import RobotContext
from fast_body.embodiment.gaze import GazeController
from fast_body.embodiment.moves import MovementManager
from fast_body.turn_speech import TurnSpeech, text_blocks

logger = logging.getLogger(__name__)

_EMOTIONS_REPO = "pollen-robotics/reachy-mini-emotions-library"
_DANCES_REPO = "pollen-robotics/reachy-mini-dances-library"

# How close to the sleep pose still counts as asleep (matches the conversation
# app's tolerances, which are tuned against the real robot's resting position).
_SLEEP_TRANSLATION_TOLERANCE_M = 0.05
_SLEEP_ROTATION_TOLERANCE_RAD = 0.35

# How long a cancelled turn gets to wind down before the timeout is reported
# anyway. A provider request stalled on the wire can swallow the cancel and
# never return.
_CANCEL_GRACE_S = 5.0
# The daemon force-kills an app 20 s after SIGINT. Exit hard before that if a
# clean shutdown has not finished, so the daemon never has to.
_SHUTDOWN_DEADLINE_S = 15.0


def _installed_versions() -> str:
    """What the brain actually runs on, for the boot line.

    The daemon installs from pyproject's pins and never reads the lock, so
    this line, not the lock, says which fast-agent the robot has.
    """
    from importlib.metadata import PackageNotFoundError, version

    parts = []
    for name in ("fast-agent-mcp", "mcp", "openai"):
        try:
            parts.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            parts.append(f"{name} missing")
    return ", ".join(parts)


def _turn_spoken_text(agent: Any, start_index: int, fallback: str, spoken: Counter[str] | None = None) -> str:
    """Every assistant text block the turn produced, oldest first, less what `spoken` already said.

    `send()` returns only the *final* assistant message, so in a turn with tool
    calls anything the model said before calling a tool would go unspoken.
    Thinking stays silent: providers put it in a message's `channels`, and
    only `content` is read here. `spoken` counts the blocks `TurnSpeech` said
    mid-turn, matched by text and each used once, so a repeated line is still
    spoken again. Returns "" when the turn had text and all of it was spoken.
    """
    try:
        parts = []
        for msg in agent.message_history[start_index:]:
            if getattr(msg, "role", None) == "assistant":
                parts.extend(text_blocks(msg))
        if parts:
            left = Counter(spoken or ())
            remaining = []
            for part in parts:
                if left[part] > 0:
                    left[part] -= 1
                else:
                    remaining.append(part)
            return "\n\n".join(remaining)
    except Exception as e:
        logger.debug("collecting turn text failed, speaking the reply only: %s", e)
    return fallback


def _usage_index(agent: Any) -> int:
    """Where the turn starts in the agent's usage ledger, or 0 when there is none."""
    try:
        return len(agent.usage_accumulator.turns)
    except Exception:
        return 0


def _turn_usage(agent: Any, start_index: int) -> str:
    """The turn's model calls and tokens, as a suffix for the latency line; "" when unknown.

    A turn with tool calls is several model calls, each sending the whole
    prompt. The counts say where a slow turn went:
    a large prompt is the card and the tool schemas, a low cache share is
    the cache missing, and many calls is the shape of the turn itself.
    """
    try:
        turns = list(agent.usage_accumulator.turns[start_index:])
    except Exception:
        return ""
    if not turns:
        return ""
    prompt = sum(t.prompt.total or 0 for t in turns)
    cached = sum(t.prompt.cache_read or 0 for t in turns)
    completion = sum(t.completion.total or 0 for t in turns)
    reasoning = sum(t.completion.reasoning or 0 for t in turns)
    tools = sum(t.tool_calls for t in turns)
    parts = [
        f"{len(turns)} model call{'s' if len(turns) != 1 else ''}",
        f"prompt {prompt} ({cached} cached)",
        f"completion {completion}",
    ]
    if reasoning:
        parts.append(f"reasoning {reasoning}")
    if tools:
        parts.append(f"tool calls {tools}")
    return "; " + ", ".join(parts)


class FastBodyCore:
    """Orchestrates one embodied conversation session."""

    def __init__(
        self,
        robot: Any,
        *,
        owns_robot: bool,
        stop_event: threading.Event,
        config: Config | None = None,
    ):
        self.robot = robot
        self.owns_robot = owns_robot
        self.stop_event = stop_event
        self._stopping = asyncio.Event()  # set with stop_event; awaitable in the loop
        self.config = config or Config()
        self.movement = MovementManager(robot)
        self.voice: Any = None
        self._tool_calls = 0
        self.gaze: Any = None
        self.web: Any = None
        self.web_hub: Any = None
        self._turn_speech: TurnSpeech | None = None
        self._mcp_retry: asyncio.Task | None = None
        self._wobbling = False

    # ── lifecycle ────────────────────────────────────────────────────────
    def _prepare_robot(self) -> None:
        try:
            self.robot.enable_motors()  # before wake_up: asleep motors won't move
            self._wake_if_sleeping()
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            # goto_target blocks until the daemon reports the task done, so there's
            # nothing to sleep off afterwards — that just delayed the first turn.
            self.robot.goto_target(head=neutral, antennas=[0.0, 0.0], duration=1.5, body_yaw=0.0)
        except Exception as e:
            logger.warning("robot pose init failed (continuing): %s", e)

    def _wake_if_sleeping(self) -> bool:
        """Play the wake-up movement when we're launched onto a sleeping robot.

        The daemon only auto-wakes an app from a *proper* sleep pose, so an app
        started while Reachy naps otherwise begins mid-slump. Detection mirrors
        the conversation app: near enough to `SLEEP_HEAD_POSE` in both position
        and orientation.
        """
        try:
            head_pose = np.asarray(self.robot.get_current_head_pose(), dtype=np.float64)
            if head_pose.shape != (4, 4):
                return False
            # (translation in m, rotation in rad, a combined "magic-mm" figure)
            distances = distance_between_poses(head_pose, SLEEP_HEAD_POSE)
            if float(distances[0]) > _SLEEP_TRANSLATION_TOLERANCE_M:
                return False
            if float(distances[1]) > _SLEEP_ROTATION_TOLERANCE_RAD:
                return False
            logger.info("robot is asleep — waking up")
            self.robot.wake_up()
            return True
        except Exception as e:
            logger.warning("wake-up check failed (continuing): %s", e)
            return False

    def _load_libraries(self) -> tuple[Any, Any]:
        loaded: list[Any] = []
        for kind, repo in (("emotion", _EMOTIONS_REPO), ("dance", _DANCES_REPO)):
            try:
                from reachy_mini.motion.recorded_move import RecordedMoves

                loaded.append(RecordedMoves(repo))
            except Exception as e:
                logger.warning("%s library unavailable: %s", kind, e)
                loaded.append(None)
        return loaded[0], loaded[1]

    def _build_vision(self):
        """The reader behind `examine(question)`; None without an OpenAI key."""
        if not self.config.openai_api_key:
            return None
        from fast_body.embodiment.vision import VisionReader

        return VisionReader(api_key=self.config.openai_api_key, model=self.config.vision_model)

    def _build_gaze(self) -> Any:
        """Build the gaze controller, or None if face tracking isn't available.

        Gated on `enable_camera`. An older daemon without daemon-side head
        tracking must not take the whole app down — we log and run without it;
        the `look_at_me` tool then reports that it's unavailable.
        """
        if not self.config.enable_camera:
            return None
        if not GazeController.is_supported(self.robot):
            logger.warning("face tracking unavailable: this robot has no daemon-side head tracking")
            return None
        return GazeController(
            self.robot,
            weight=self.config.head_tracking_weight,
            speaking_weight=self.config.head_tracking_speaking_weight,
        )

    def _start_web_chat(self) -> Any:
        """Start the companion page, or None if it's off or won't bind."""
        if not self.config.enable_web_chat:
            return None
        try:
            from fast_body.web import WebChatHub, WebChatServer

            self.web_hub = WebChatHub()
            self.web = WebChatServer(
                self.web_hub,
                host=self.config.web_chat_host,
                port=self.config.web_chat_port,
                token=self.config.web_chat_token,
            )
            self.web.start()
            return self.web_hub
        except Exception as e:
            logger.warning("web chat unavailable (continuing without it): %s", e)
            self.web = self.web_hub = None
            return None

    async def _wait_until_configured(self) -> bool:
        """Hold until the config validates, so the settings page can fix it.

        Returns False if we were stopped first. Nothing is built before this
        returns, so the brain and voice backend read the settings just saved.
        """
        errors = self.config.validate()
        if not errors:
            return True

        for e in errors:
            logger.error("config error: %s", e)
        if self.web is None:
            logger.error("no way to fix it from here — set the keys in %s", ENV_FILE)
            return False
        logger.error("waiting for settings at %s", self.web.url() + "  → /settings")

        while not self.stop_event.is_set():
            await asyncio.sleep(1.0)
            self.config = Config()  # picks up whatever the settings page wrote
            if not self.config.validate():
                logger.info("configured; starting up")
                return True
        return False

    def _build_tui_voice(self) -> Any:
        """A speak-only backend for `--tui`, or None to run silent.

        `text` is skipped because its printing fights the TUI, and `realtime`
        gets the discrete backend because input is typed. Barge-in is off: it
        would read room noise from the mic and clip the robot.
        """
        if self.config.voice_backend.lower() not in ("openai", "realtime") or not self.config.openai_api_key:
            logger.info("TUI running silent (no OPENAI_API_KEY for speech)")
            return None
        try:
            from dataclasses import replace

            from fast_body.voice.base import with_cast
            from fast_body.voice.openai_backend import OpenAIVoiceBackend

            return with_cast(OpenAIVoiceBackend(replace(self.config, enable_barge_in=False), self.robot), self.config)
        except Exception as e:
            logger.warning("TUI speech unavailable (continuing silent): %s", e)
            return None

    def _speak_tui_replies(self, brain: Any) -> None:
        """Speak whatever fast-agent's prompt loop returns for each turn.

        `interactive()` wraps `_send_interactive_message`, whose return value is
        the assistant's reply, so wrapping it catches every turn without
        reimplementing the loop. Private, so a fast-agent upgrade could move it —
        hence the check rather than an assumption.
        """
        if self.voice is None:
            return
        original = getattr(brain, "_send_interactive_message", None)
        if not callable(original):
            logger.warning("can't hook the TUI's replies; the robot will stay silent")
            return

        async def speaking(*args: Any, **kwargs: Any) -> Any:
            # (message, agent_name, ...) — capture where this turn starts in the
            # agent's history so every assistant message in it gets spoken, not
            # just the final one `_send_interactive_message` returns.
            agent_name = args[1] if len(args) > 1 else kwargs.get("agent_name")
            try:
                agent = brain.resolve_agent(agent_name)
                start = len(agent.message_history)
            except Exception:
                agent, start = None, 0
            speech = self._turn_speech
            if speech is not None:
                speech.begin()
            reply = await original(*args, **kwargs)
            text = reply if isinstance(reply, str) else ""
            if agent is not None:
                text = _turn_spoken_text(agent, start, text, speech.spoken if speech is not None else None)
            if text.strip():
                self._cue(cues.SPEAKING)
                try:
                    # Awaited, not fired off, so the prompt returns when the robot
                    # has finished talking rather than over the top of it.
                    await self.voice.speak(text)
                except Exception as e:
                    logger.warning("TUI speech failed: %s", e)
                self._cue(cues.IDLE)
            return reply

        brain._send_interactive_message = speaking

    async def run(self) -> None:
        # Web chat comes up first so a misconfigured app is still reachable.
        # When the config is already valid, import the brain first: /status
        # imports the same module tree, and a concurrent import sees a
        # half-initialised module.
        if not self.config.validate():
            import fast_body.agent  # noqa: F401
        web_hub = self._start_web_chat()
        if not await self._wait_until_configured():
            await self._shutdown()
            return

        from fast_body.agent import fast

        warm_up = await self._start_body(web_hub)
        try:
            async with fast.run() as brain:
                resume = await self._attach_brain(brain)
                if warm_up is not None:
                    await warm_up
                if resume.resumed:
                    await self._reopen(brain)
                if self.config.tui:
                    # fast-agent's own prompt loop, with our body tools attached.
                    self._speak_tui_replies(brain)
                    await brain.interactive()
                else:
                    await self._converse(brain)
        finally:
            choices.bind(None)
            retry = self._mcp_retry
            if retry is not None and not retry.done():
                retry.cancel()
            if self.web is not None:
                self.web.unbind_brain()
            await self._shutdown()

    async def _start_body(self, web_hub: Any) -> asyncio.Task | None:
        """Pose, movement, voice and the choice handler; returns the TTS warm-up task."""
        from fast_body.voice import build_backend

        self._prepare_robot()

        emotions, dances = self._load_libraries()
        self.gaze = self._build_gaze()
        if self.gaze is not None and self.config.face_tracking:
            self.gaze.set_enabled(True)
        RobotContext.set(
            RobotContext(
                robot=self.robot,
                movement=self.movement,
                emotions=emotions,
                dances=dances,
                gaze=self.gaze,
                vision=self._build_vision(),
                camera=self.config.enable_camera,
            )
        )

        self.movement.start()
        if self.config.tui:
            # The TUI reads input; the robot still speaks the replies if it can.
            self.voice = self._build_tui_voice()
        else:
            self.voice = build_backend(self.config, self.robot, web_hub=web_hub)
            if hasattr(self.voice, "start"):  # backends with an input task of their own
                await self.voice.start(self.stop_event)
        if hasattr(self.voice, "on_speech_start"):
            self.voice.on_speech_start(lambda: self._cue(cues.LISTENING))
        self._turn_speech = self._build_turn_speech()
        warm_up = self._start_tts_warm_up()

        if self.config.enable_wobble:
            try:
                self.robot.enable_wobbling()  # daemon composes wobble onto our targets
                self._wobbling = True
            except Exception as e:
                logger.debug("wobble unavailable (non-LOCAL audio backend?): %s", e)

        if self.web is not None:
            self.web.ready = True
        # A server's question goes to the page, the speaker and the microphone.
        # Bound here because all three now exist.
        choices.bind(
            choices.Table(
                self.voice,
                web_hub,
                timeout_s=self.config.choice_timeout_s,
                cue=lambda: self._cue(cues.LISTENING),
            )
        )
        logger.info(
            "fast-body ready (voice=%s, wobble=%s, barge_in=%s; %s)",
            self.config.voice_backend,
            self._wobbling,
            self.config.enable_barge_in,
            _installed_versions(),
        )
        return warm_up

    async def _attach_brain(self, brain: Any) -> memory.Resume:
        """Memory, context budget, MCP servers, skills and the page hooks, once the brain is open."""
        # Both memory calls need the brain open: the session manager lives on
        # the fast-agent context, and the context window is only known once
        # the LLM is attached.
        resume = await memory.resume_recent(brain, self.config)
        logger.info("memory: %s", resume.status)
        logger.info("context: %s", memory.apply_context_budget(brain, self.config))
        logger.info("mcp: %s", await mcp_servers.attach_enabled(brain))
        logger.info("skills: %s", await skills.sync_from_servers(brain))
        if mcp_servers.startup_failures():
            # A Space that was asleep, or a laptop not yet on the network:
            # keep trying while the robot talks, rather than losing the
            # server for the whole session.
            self._mcp_retry = asyncio.create_task(
                mcp_servers.retry_failed(
                    brain,
                    mcp_servers.startup_failures(),
                    after=lambda: skills.sync_from_servers(brain),
                )
            )
        if self.web is not None:
            # From here the settings page can attach and detach live.
            self.web.bind_brain(brain, asyncio.get_running_loop())
            # Every tool call marks the page and flicks the antennas;
            # those with a widget draw on the page too.
            from fast_body.personality import load_personality
            from fast_body.web import apps

            apps.observe(
                brain.resolve_agent(None),
                self.web_hub,
                steps=load_personality(self.config.personality).activity,
                on_call=self._tool_tick,
            )
        if self._turn_speech is not None:
            self._turn_speech.install(brain.resolve_agent(None))
        return resume

    async def _reopen(self, brain: Any) -> None:
        """Show and say that a conversation was picked back up.

        Both halves exist because resuming is otherwise invisible: the page
        would open empty and the robot would carry on mid-thought, which reads
        as a robot that forgot rather than one that remembered.
        """
        try:
            history = list(brain.resolve_agent(None).message_history)
        except Exception as e:
            logger.warning("could not read the resumed history: %s", e)
            return

        if self.web_hub is not None:
            self.web_hub.seed(memory.resumed_transcript(history))

        if not self.config.memory_greeting or self.voice is None:
            return
        line = memory.reopening_line(history)
        if not line:
            return
        logger.info("robot: %s", line)
        self._cue(cues.SPEAKING)
        try:
            await self._speak_or_stop(line)
        except Exception as e:
            # Speaking the reopening is a nicety; failing it must not cost the
            # conversation the memory it just recovered.
            logger.warning("could not speak the reopening line: %s", e)

    async def _converse(self, brain: Any) -> None:
        is_text = self.config.voice_backend.lower() == "text"
        try:
            agent = brain.resolve_agent(None)  # the default "body" agent
        except Exception:
            agent = None
        while not self.stop_event.is_set():
            # Breathe while waiting; the backend cues LISTENING at speech onset
            # (on_speech_start, bound in run), so the pose reads as a reaction.
            self._cue(cues.IDLE)
            text = await self._listen_or_stop()
            if self.stop_event.is_set():
                break
            if not text:
                if is_text:  # EOF on stdin → end the session
                    break
                continue  # silence / no speech → keep listening
            user_done = time.perf_counter()
            logger.info("user: %s", text)

            self._cue(cues.THINKING)
            self._tool_calls = 0
            speech = self._turn_speech
            if speech is not None:
                speech.begin()
            start = len(agent.message_history) if agent is not None else 0
            usage_start = _usage_index(agent)
            # Not asyncio.wait_for: fast-agent's OpenAI-compatible provider
            # catches the cancellation and returns an empty assistant message,
            # which wait_for reports as a result. Time the task here and
            # discard what a cancelled one returns.
            turn = asyncio.ensure_future(self._interruptible(brain.send(text)))
            if await self._await_turn(turn, speech):
                # Stopped mid-turn. The daemon gives an app 20 s, so the turn
                # is cancelled rather than finished; a block being spoken is
                # cut by the hook's own cancel handling.
                turn.cancel()
                await asyncio.wait({turn}, timeout=_CANCEL_GRACE_S)
                self._drop_turn(agent, start)
                break
            if not turn.done():
                await self._abandon_turn(turn)
                self._drop_turn(agent, start)
                reply, interrupted = "Sorry, I lost my train of thought there.", False
            else:
                try:
                    reply, interrupted = turn.result()
                except Exception as e:
                    logger.error("brain error: %s", e)
                    self._drop_turn(agent, start)
                    reply, interrupted = "Sorry, my mind went blank for a second.", False
            if interrupted:  # Escape during reasoning → drop the turn, back to input
                continue
            if self._stopping.is_set():
                break  # a stop came in while the turn finished; say nothing more
            if agent is not None and isinstance(reply, str):
                # The whole turn, less what was said between its tool calls.
                reply = _turn_spoken_text(agent, start, reply, speech.spoken if speech is not None else None)
            silent = isinstance(reply, str) and not reply.strip()
            logger.info(
                "turn latency: %.0f ms (user done → %s)%s",
                (time.perf_counter() - user_done) * 1000,
                "turn done" if silent else "speaking",
                _turn_usage(agent, usage_start),
            )
            if silent:
                continue
            logger.info("robot: %s", reply)
            self._cue(cues.SPEAKING)
            await self._speak_or_stop(reply)

    async def _abandon_turn(self, turn: asyncio.Future) -> None:
        """Cancel a timed-out turn and give it a bounded time to wind down."""
        turn.cancel()
        # Bounded: a cancel the provider swallows would otherwise hold the loop
        # here for good, with no log line and no reply.
        settled, _ = await asyncio.wait({turn}, timeout=_CANCEL_GRACE_S)
        if not settled:
            logger.error("the cancelled turn is still running after %.0fs; leaving it", _CANCEL_GRACE_S)
        logger.error("brain timed out after %.0fs; dropping the turn", self.config.brain_timeout_s)

    async def _await_turn(self, turn: asyncio.Future, speech: TurnSpeech | None) -> bool:
        """Wait for the turn, the brain timeout, or a stop. True when the stop came first.

        The timeout bounds the provider, so time the turn spends at the table
        (a server's question to the players has its own clock) or speaking a
        block does not count: the clock restarts when a block finishes.
        """
        stop = asyncio.ensure_future(self._stopping.wait())
        timeout = self.config.brain_timeout_s
        try:
            while True:
                await asyncio.wait({turn, stop}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if turn.done():
                    return False
                if stop.done():
                    return True
                if choices.waiting() or (speech is not None and speech.speaking):
                    continue
                quiet = speech.quiet_for() if speech is not None else None
                if quiet is None or quiet >= self.config.brain_timeout_s:
                    return False  # timed out
                timeout = self.config.brain_timeout_s - quiet
        finally:
            stop.cancel()

    @staticmethod
    def _drop_turn(agent: Any, start: int) -> None:
        """Forget what a failed turn left in the history.

        A timeout cancels the turn wherever it is, and mid-tool-loop that is
        after a tool result was appended and before the assistant message that
        called it. The provider then rejects every later turn (e.g.
        `orphan_tool_call_id`). Cutting back to where the turn began costs the
        failed turn and nothing else.
        """
        if agent is None:
            return
        try:
            history = agent.message_history
            if len(history) > start:
                del history[start:]
        except Exception as e:
            logger.warning("could not roll back the failed turn: %s", e)

    def request_stop(self, reason: str = "stop requested") -> None:
        """End the run from inside the loop: the next idle wait returns at once.

        Called by the signal handlers `run_core` installs. Sets both events:
        `stop_event` is what the daemon and the sync helpers watch, `_stopping`
        is what the awaits in this loop can race against.
        """
        if not self.stop_event.is_set():
            logger.info("%s; finishing up", reason)
        self.stop_event.set()
        self._stopping.set()

    async def _listen_or_stop(self) -> str | None:
        """`voice.listen()`, abandoned the moment a stop is requested.

        A listen waits up to its idle timeout (30 s on the realtime backend),
        longer than the 20 s the daemon gives an app to stop.
        """
        text, stopped = await self._unless_stopped(self.voice.listen())
        return None if stopped else text

    async def _speak_or_stop(self, text: str) -> None:
        """`voice.speak()`, cut off the moment a stop is requested.

        Cancelling the speak drops the rest of the reply, a cast's remaining
        lines included; `interrupt()` then empties the player of what was
        already pushed.
        """
        _, stopped = await self._unless_stopped(self._interruptible(self.voice.speak(text)))
        if stopped:
            self.voice.interrupt()

    async def _unless_stopped(self, coro: Coroutine[Any, Any, Any]) -> tuple[Any, bool]:
        """Await ``coro``, or cancel it when a stop is requested first.

        Returns ``(result, stopped)``. A cancelled task is awaited so nothing
        it was doing outlives the loop.
        """
        if self._stopping.is_set():
            coro.close()
            return None, True
        task = asyncio.ensure_future(coro)
        stop = asyncio.ensure_future(self._stopping.wait())
        done, _ = await asyncio.wait({task, stop}, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            stop.cancel()
            return task.result(), False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        return None, True

    async def _interruptible(self, coro: Any) -> tuple[Any, bool]:
        """Await ``coro``, but let an Escape press cancel it mid-flight.

        Returns ``(result, interrupted)``. Outside the interactive console there is
        no interrupt event, so this just awaits ``coro`` and the loop is unchanged.
        On interrupt it cancels the task and quiesces the body — stops playback
        (``voice.interrupt``) and movement (``clear_move_queue``) — then returns
        ``(None, True)`` so the caller can fall back to listening.
        """
        event = getattr(self.voice, "interrupt_event", None)
        if event is None:
            return await coro, False

        # Discard any Escape pressed while idle/listening — interrupt only applies
        # to what's in flight *now*. A fresh Escape during the await re-sets it.
        event.clear()
        task = asyncio.ensure_future(coro)
        waiter = asyncio.ensure_future(event.wait())
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            task.cancel()  # a timeout upstream must not leave the turn running
            raise
        finally:
            waiter.cancel()

        if not task.done():  # the interrupt won the race
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            self.voice.interrupt()
            self.movement.clear_move_queue()
            event.clear()
            return None, True
        return task.result(), False

    def _build_turn_speech(self) -> TurnSpeech | None:
        """Mid-turn speech, or None when the flag is off (the turn is then spoken once, at the end)."""
        if not self.config.speak_between_tools or self.voice is None:
            return None
        return TurnSpeech(
            self.voice,
            stopping=self._stopping,
            on_start=lambda: self._cue(cues.SPEAKING),
            on_end=lambda: self._cue(cues.THINKING),
        )

    def _start_tts_warm_up(self) -> asyncio.Task | None:
        """Run the speech warm-up beside the brain's startup, not before it."""
        warm = getattr(self.voice, "warm_up", None)
        if not self.config.tts_warmup or warm is None:
            return None
        return asyncio.create_task(warm())

    def _tool_tick(self) -> None:
        """A flick of the antennas per tool call: the turn is working, not stuck."""
        self._tool_calls += 1
        try:
            head, antennas = self.movement.current_pose()
            for move in cues.tool_tick(self._tool_calls, head, antennas):
                self.movement.queue_move(move)
        except Exception as e:
            logger.debug("tool tick failed: %s", e)

    def _cue(self, state: str) -> None:
        """Signal the conversation state on the body (antennas + breathing hold) and on the page."""
        if self.web_hub is not None:
            try:
                self.web_hub.post_activity(state)
            except Exception as e:
                logger.debug("activity %s not posted: %s", state, e)
        try:
            head, antennas = self.movement.current_pose()
            self.movement.set_listening(state == cues.LISTENING)
            if self.gaze is not None:
                # Ease off the face while talking so emotions read as expression.
                self.gaze.set_speaking(state == cues.SPEAKING)
            # Defensively: a missing context must cost the held expression, not
            # every cue — this whole method swallows exceptions at debug level.
            try:
                ctx = RobotContext.current()
            except RuntimeError:
                ctx = None
            if ctx is not None and state in (cues.IDLE, cues.LISTENING):
                ctx.hold_expression = False  # each turn starts with a clean face
            hold = bool(ctx.hold_expression) if ctx is not None else False
            move = cues.antenna_cue(state, head, antennas, hold_expression=hold)
            if move is not None:
                self.movement.queue_move(move)
        except Exception as e:
            logger.debug("cue %s failed: %s", state, e)

    async def _shutdown(self) -> None:
        logger.info("shutting down…")
        steps: list[tuple[str, Callable[[], Any] | None]] = [
            ("disable wobble", self.robot.disable_wobbling if self._wobbling else None),
            ("voice close", self.voice.aclose if self.voice is not None else None),
            ("web chat stop", self.web.stop if self.web is not None else None),
            # Hand the head back before the movement manager parks it.
            ("gaze stop", self.gaze.stop if self.gaze is not None else None),
            ("movement stop", self.movement.stop),
            ("context clear", RobotContext.clear),
            ("media close", self.robot.media.close if self.owns_robot else None),
        ]
        for name, step in steps:
            if step is None:
                continue
            try:
                result = step()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.debug("%s: %s", name, e)


def run_core(
    robot: Any,
    *,
    owns_robot: bool,
    stop_event: threading.Event,
    config: Config | None = None,
) -> None:
    """Run a `FastBodyCore` to completion in a fresh event loop (sync entry).

    SIGINT and SIGTERM become a stop request inside the loop. A
    KeyboardInterrupt would break the loop out from under its tasks and leave
    a non-daemon AnyIO worker thread that interpreter exit then joins.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    core = FastBodyCore(robot, owns_robot=owns_robot, stop_event=stop_event, config=config)
    restore = _install_stop_signals(loop, core)
    try:
        loop.run_until_complete(core.run())
    finally:
        restore()
        loop.close()


def _install_stop_signals(loop: asyncio.AbstractEventLoop, core: FastBodyCore) -> Callable[[], None]:
    """Route SIGINT/SIGTERM to `core.request_stop`; returns the undo.

    The first signal asks for a clean stop and arms `_hard_exit_later`. A
    second one exits at once, so a stuck shutdown still answers Ctrl-C.
    """
    state = {"armed": False}

    def on_signal(signame: str) -> None:
        if state["armed"]:
            logger.warning("%s again; exiting now", signame)
            _hard_exit(1)
        state["armed"] = True
        core.request_stop(f"{signame} received")
        _hard_exit_later(_SHUTDOWN_DEADLINE_S)

    installed: list[tuple[int, Any]] = []
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is None:
            continue
        try:
            loop.add_signal_handler(signum, on_signal, signame)
            installed.append((signum, None))
        except (NotImplementedError, RuntimeError):
            # Windows, or not the main thread: a plain handler, hopping onto the loop.
            try:
                previous = signal.signal(signum, lambda *_: loop.call_soon_threadsafe(on_signal, signame))
            except (ValueError, OSError):
                continue
            installed.append((signum, previous))

    def restore() -> None:
        for signum, previous in installed:
            try:
                if previous is None:
                    loop.remove_signal_handler(signum)
                else:
                    signal.signal(signum, previous)
            except Exception:
                pass

    return restore


def _hard_exit(code: int) -> None:
    logging.shutdown()
    os._exit(code)


def _hard_exit_later(deadline_s: float) -> threading.Timer:
    """Exit the process if it is still here after `deadline_s`.

    A daemon thread, so it never keeps the process alive itself; it only
    catches the case where a clean shutdown, or interpreter teardown after
    one, is what hangs.
    """

    def fire() -> None:
        logger.error("shutdown did not finish in %.0f s; exiting hard", deadline_s)
        _hard_exit(0)

    timer = threading.Timer(deadline_s, fire)
    timer.daemon = True
    timer.start()
    return timer


class FastBodyApp(ReachyMiniApp):
    """Reachy Mini Apps entry point (registered in pyproject.toml).

    The daemon instantiates this class and calls `wrapped_run()` (inherited),
    which opens the robot connection and hands it to our `run()`. Standalone use
    goes through `main.py`, which constructs its own `ReachyMini`.
    """

    # The daemon reads the URL by regex from main.py; this attribute only
    # matters for SDK-mode listings. We serve the page ourselves
    # (web/server.py), hence dont_start_webserver.
    custom_app_url: str | None = "http://0.0.0.0:8080/"
    dont_start_webserver: bool = True

    def run(self, reachy_mini: Any, stop_event: threading.Event) -> None:
        from fast_body.main import setup_logging

        config = Config()
        setup_logging(config.debug)
        logger.info("fast-body launched by Reachy Mini daemon")
        logger.info("config: %s", config_sources())
        # A missing key doesn't end the run: the core serves the settings page and
        # waits, which is the only way to fix it on a robot with no terminal.
        run_core(reachy_mini, owns_robot=False, stop_event=stop_event, config=config)


if __name__ == "__main__":
    # `python -m fast_body.app` is how the daemon launches the app: wrapped_run()
    # opens/closes the ReachyMini connection inside the daemon's lifecycle.
    app = FastBodyApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
