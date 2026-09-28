"""The browser end of the conversation, as a typed-input surface.

`WebChatHub` offers the same interface as `ConsoleInput` (`read_line`, `echo`,
`interrupt_event`), so `DualVoiceBackend` can race a browser message against the
microphone without knowing which surface it has.

The web server runs uvicorn on its own thread and loop; the conversation runs on
the app's loop. Each direction crosses that boundary once:

- browser → conversation: `call_soon_threadsafe` onto the conversation loop's queue.
- conversation → browser: `run_coroutine_threadsafe` onto the server loop, without
  waiting for delivery, so a dead socket can't stall a turn.

Recent lines are buffered and replayed to a browser that connects or reloads
mid-conversation. Besides spoken lines the transcript carries widgets: a tool
call whose server paired it with an MCP App page (see web/apps.py) is one
entry, posted when the call starts and completed in place when the result
lands, so a late browser gets the finished widget.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# Lines replayed to a browser that connects late. Enough to see the thread of the
# conversation, small enough to send in one frame.
_TRANSCRIPT_LIMIT = 200

# The robot answers one message per turn, so a backlog past this is someone
# holding down Enter. Bounded to keep the queue from growing without limit.
_PENDING_LIMIT = 32


class WebChatHub:
    """Shared state between the web server thread and the conversation loop."""

    def __init__(self) -> None:
        self.interrupt_event = asyncio.Event()

        self._lines: asyncio.Queue[str | None] = asyncio.Queue(maxsize=_PENDING_LIMIT)
        # Typed before the conversation loop exists; handed over in start().
        self._early: list[str] = []
        self._transcript: deque[dict[str, Any]] = deque(maxlen=_TRANSCRIPT_LIMIT)

        # Bound when each side starts; both are read from the other's thread.
        self._app_loop: asyncio.AbstractEventLoop | None = None
        self._server_loop: asyncio.AbstractEventLoop | None = None
        self._clients: set[Any] = set()

    # ── conversation-loop side (the VoiceBackend interface) ──────────────
    async def start(self, stop_event: Any = None) -> None:
        """Adopt the conversation loop as the one `read_line` is awaited on."""
        self._app_loop = asyncio.get_running_loop()
        early, self._early = self._early, []
        for text in early:
            self._enqueue(text)

    async def read_line(self) -> str | None:
        """Await the next message typed in a browser."""
        return await self._lines.get()

    def echo(self, text: str) -> None:
        """Post a robot line to every open browser. The page labels it itself."""
        self._post("robot", text)

    def echo_user(self, text: str) -> None:
        """Post a user line — what the microphone heard, or what was typed."""
        self._post("user", text)

    def post_app_call(self, call_id: str, *, widget: Any, arguments: dict[str, Any]) -> None:
        """A tool with a widget has been called; the page opens the widget now."""
        message = {
            "role": "app",
            "id": call_id,
            "server": widget.server,
            "tool": widget.tool,
            "uri": widget.resource_uri,
            "arguments": arguments,
            "result": None,
            "cancelled": False,
            "failed": False,
        }
        self._transcript.append(message)
        self._broadcast(message)

    def post_app_result(
        self, call_id: str, result: dict[str, Any] | None, *, cancelled: bool = False, failed: bool = False
    ) -> None:
        """The call finished (or was cut short, or came back an error); completes the entry in place."""
        self._complete("app", call_id, result=result, cancelled=cancelled, failed=failed)
        self._broadcast(
            {"role": "app-result", "id": call_id, "result": result, "cancelled": cancelled, "failed": failed}
        )

    def post_activity(self, state: str, step: str | None = None) -> None:
        """What the robot is doing right now, for the page to show while a reply is on its way.

        Not a transcript entry: it is over as soon as the reply lands. `step`
        marks one tool call — the label the card allows for it, or "" for an
        unnamed one; None means the state changed with no call attached.
        """
        message: dict[str, Any] = {"role": "activity", "state": state}
        if step is not None:
            message["step"] = step
        self._broadcast(message)

    def post_choice(self, choice: Any) -> None:
        """A server is asking the table to choose; the page shows the options as buttons."""
        message = {
            "role": "choice",
            "id": choice.id,
            "prompt": choice.prompt,
            "options": [{"id": o.id, "label": o.label, "description": o.description} for o in choice.options],
            "allow_free_text": choice.allow_free_text,
            "answered": None,
            "how": None,
        }
        self._transcript.append(message)
        self._broadcast(message)

    def post_choice_answered(self, choice_id: str, option_id: str | None, how: str) -> None:
        """The question was answered (or ran out); every page's buttons settle."""
        self._complete("choice", choice_id, answered=option_id, how=how)
        self._broadcast({"role": "choice-answered", "id": choice_id, "answered": option_id, "how": how})

    def requeue(self, text: str) -> None:
        """Hand words already shown on the page back to the conversation as its next turn."""
        loop = self._app_loop
        if loop is None:
            self._early.append(text)
            return
        loop.call_soon_threadsafe(self._enqueue, text)

    def clear(self) -> None:
        """Drop the transcript, here and on every open page — the conversation was forgotten."""
        self._transcript.clear()
        self._broadcast({"role": "history", "lines": []})

    async def aclose(self) -> None:
        self._clients.clear()

    # ── web-server side ──────────────────────────────────────────────────
    def bind_server_loop(self) -> None:
        """Adopt the running loop as the server's; called from a route handler."""
        self._server_loop = asyncio.get_running_loop()

    def seed(self, lines: list[dict[str, str]]) -> None:
        """Prime the transcript with a conversation resumed from an earlier run.

        Not broadcast: this runs at startup, before any browser has connected,
        and a client is sent the whole transcript when it does connect.
        """
        for line in lines:
            self._post(line["role"], line["text"], broadcast=False)

    def transcript(self) -> list[dict[str, Any]]:
        """The recent conversation, for replay into a newly opened browser."""
        return list(self._transcript)

    def add_client(self, websocket: Any) -> None:
        self._clients.add(websocket)

    def discard_client(self, websocket: Any) -> None:
        self._clients.discard(websocket)

    def submit(self, text: str) -> None:
        """Hand a browser message to the conversation loop. Called on the server loop."""
        text = text.strip()
        if not text:
            return
        self._post("user", text, broadcast=True)
        loop = self._app_loop
        if loop is None:
            # The page is served well before the conversation loop is up. Hold
            # the message rather than drop it.
            if len(self._early) >= _PENDING_LIMIT:
                logger.warning("dropping browser message — %d already waiting for startup", _PENDING_LIMIT)
                return
            self._early.append(text)
            logger.info("holding browser message until the conversation starts")
            return
        loop.call_soon_threadsafe(self._enqueue, text)

    def _enqueue(self, text: str) -> None:
        """Queue a message, dropping it if the backlog is full. On the app loop."""
        try:
            self._lines.put_nowait(text)
        except asyncio.QueueFull:
            logger.warning("dropping browser message — %d already queued", _PENDING_LIMIT)

    def request_interrupt(self) -> None:
        """Let the browser stop the robot mid-reply, as Escape does in the console."""
        loop = self._app_loop
        if loop is not None:
            loop.call_soon_threadsafe(self.interrupt_event.set)

    # ── internals ────────────────────────────────────────────────────────
    def _complete(self, role: str, entry_id: str, **fields: Any) -> None:
        """Update the latest transcript entry with this role and id."""
        for entry in reversed(self._transcript):
            if entry.get("role") == role and entry.get("id") == entry_id:
                entry.update(fields)
                break

    def _post(self, role: str, text: str, broadcast: bool = True) -> None:
        message = {"role": role, "text": text}
        self._transcript.append(message)
        if broadcast:
            self._broadcast(message)

    def _broadcast(self, message: dict[str, Any]) -> None:
        """Fan a message out to open sockets without blocking the caller."""
        loop = self._server_loop
        if loop is None or not self._clients:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._send_all(message), loop)
        except RuntimeError:  # server loop already closing
            logger.debug("web chat broadcast skipped — server loop is gone")

    async def _send_all(self, message: dict[str, str]) -> None:
        for websocket in list(self._clients):
            try:
                await websocket.send_json(message)
            except Exception:
                # A browser tab that went away is ordinary, not an error.
                self._clients.discard(websocket)
