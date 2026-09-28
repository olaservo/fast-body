"""Drive a fast-body session on the robot with typed turns and read the journal back.

Each step goes in through the chat page's websocket, the same path a typed
line takes, so the reply is spoken and the cast voices switch as they would
for a spoken turn. The page's `activity` frames (thinking / speaking / idle,
plus a step label per tool call) are printed as they arrive, with the seconds
since the step was sent. The daemon journal is read at the same time, so every
step reports what the app logged: turn latency, first audio per utterance,
which cast member spoke, and any name it could not place.

    uv run python scripts/session_probe.py --from steps.txt --out runs/
    uv run python scripts/session_probe.py --say "I ask the innkeeper about the road"

Only the typed input differs from a real session; the mic stays live.

Needs `websockets`, which comes in with fast-agent rather than as a direct
dependency.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import websockets

APP_PREFIX = re.compile(
    r"^(\S+) reachy-mini launcher\.sh\[\d+\]: reachy_mini\.apps\.manager\.runner - WARNING - "
    r"\S+ \[(\w+)\] (\S+): (.*)$"
)
NOISE = ("socket.send() raised", "TURN credentials")

Line = tuple[float, str, str, str, str]  # (t_local, robot stamp, level, logger, text)


class Journal:
    """Live daemon-journal lines from the app, with the robot's timestamp."""

    def __init__(self, host: str):
        self.url = f"ws://{host}:8000/logs/ws/daemon"
        self.origin = f"http://{host}:8000"
        self.lines: list[Line] = []
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        async with websockets.connect(self.url, additional_headers={"origin": self.origin}, max_size=None) as ws:
            async for msg in ws:
                now = time.time()
                for raw in str(msg).splitlines():
                    if any(n in raw for n in NOISE):
                        continue
                    m = APP_PREFIX.match(raw)
                    if m:
                        stamp, level, logger, text = m.groups()
                        self.lines.append((now, stamp, level, logger, text))

    def since(self, t: float) -> list[Line]:
        return [line for line in self.lines if line[0] >= t]

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()


def summarize(lines: list[Line]) -> dict:
    out: dict = {"turn_ms": None, "first_audio_ms": [], "spoke_as": [], "unplaced": [], "errors": [], "other": []}
    for _, _, level, logger, text in lines:
        if m := re.match(r"turn latency: (\d+) ms", text):
            out["turn_ms"] = int(m.group(1))
        elif m := re.match(r"first audio (\d+) ms", text):
            out["first_audio_ms"].append(int(m.group(1)))
        elif m := re.match(r"speaking as (.+?) \((\w+)\)", text):
            out["spoke_as"].append(f"{m.group(1)} ({m.group(2)})")
        elif m := re.match(r"no voice in the cast for '(.+?)'", text):
            out["unplaced"].append(m.group(1))
        elif level in ("ERROR", "WARNING"):
            out["errors"].append(text)
        elif not text.startswith(("user: ", "robot: ")):
            out["other"].append(f"{logger}: {text}")
    return out


def print_activity(frame: dict, t0: float) -> None:
    """One `activity` frame: what the robot is doing, how long after the step was sent."""
    line = f"    activity ({time.time() - t0:.1f}s): {frame.get('state')}"
    if "step" in frame:  # a tool call; "" when the card gives it no label
        line += f" - {frame['step'] or 'unnamed step'}"
    print(line)


async def watch_activity(chat, t0: float, until: float) -> None:
    """Keep printing activity frames until `until`; the `idle` one marks the end of playback."""
    while (left := until - time.time()) > 0:
        try:
            frame = json.loads(await asyncio.wait_for(chat.recv(), timeout=left))
        except TimeoutError:
            return
        if frame.get("role") == "activity":
            print_activity(frame, t0)


async def one_step(chat, i: int, n: int, text: str, reply_timeout: float) -> tuple[str, float]:
    """Send one typed line and read the whole turn back.

    The turn ends at the `idle` activity frame, which the app posts when the
    last utterance has played. Since text is spoken ahead of each tool call,
    a turn can carry several `robot` lines; every one is printed and they are
    joined for the report.
    """
    t0 = time.time()
    print(f"\n[{i}/{n}] you: {text}")
    await chat.send(json.dumps({"text": text}))
    blocks: list[str] = []
    deadline = t0 + reply_timeout
    # A present_player_choice call posts a `choice` card and then speaks the
    # options as a `robot` line before the turn's reply exists. Skip that line,
    # or the options are taken for the reply and the next typed step answers
    # the choice instead of the turn.
    options_pending = False
    started = False
    while time.time() < deadline:
        try:
            frame = json.loads(await asyncio.wait_for(chat.recv(), timeout=deadline - time.time()))
        except TimeoutError:
            break
        role = frame.get("role")
        if role == "activity":
            print_activity(frame, t0)
            state = frame.get("state")
            if state in ("thinking", "speaking"):
                started = True
            elif state == "idle" and started:
                break
        elif role == "choice":
            options_pending = True
            print("    choice offered mid-turn (options read aloud before the reply)")
        elif role == "robot":
            if options_pending:
                options_pending = False
                continue
            block = frame.get("text", "")
            blocks.append(block)
            tail = "..." if len(block) > 300 else ""
            print(f"    robot ({time.time() - t0:.1f}s): {block[:300]}{tail}")
    if not blocks:
        print(f"    no reply within {reply_timeout:.0f}s")
    return "\n\n".join(blocks), t0


async def run(host: str, steps: list[str], out_dir: Path, reply_timeout: float, pause: float) -> int:
    journal = Journal(host)
    await journal.start()
    await asyncio.sleep(1.0)  # let the backlog drain before the first step

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"probe-{stamp}.md"
    rows: list[str] = []
    detail: list[str] = []

    url = f"ws://{host}:8080/ws"
    async with websockets.connect(url, max_size=None) as chat:
        first = json.loads(await chat.recv())
        if first.get("role") != "history":
            print("unexpected first frame:", first, file=sys.stderr)
        print(f"connected to {url}; {len(first.get('lines', []))} lines of transcript already on the page")

        for i, text in enumerate(steps, 1):
            reply, t0 = await one_step(chat, i, len(steps), text, reply_timeout)

            # The idle frame ended the step; give the last voice lines a moment to land.
            await watch_activity(chat, t0, time.time() + pause)
            lines = journal.since(t0)
            s = summarize(lines)
            spoke = s["spoke_as"] or ["narrator only"]
            print(f"    turn {s['turn_ms']} ms | first audio {s['first_audio_ms']} ms | spoke as {spoke}")
            if s["unplaced"]:
                print(f"    NOT IN CAST: {s['unplaced']}")
            for e in s["errors"]:
                print(f"    ! {e[:200]}")
            for o in s["other"]:
                print(f"    . {o[:160]}")

            rows.append(
                f"| {i} | {s['turn_ms']} | {', '.join(map(str, s['first_audio_ms']))} | "
                f"{', '.join(s['spoke_as']) or '-'} | {', '.join(s['unplaced']) or '-'} | {len(s['errors'])} |"
            )
            journal_text = "\n".join(f"{st} [{lv}] {lg}: {tx}" for _, st, lv, lg, tx in lines)
            detail.append(f"## Step {i}\n\n**You:** {text}\n\n**Robot:** {reply}\n\n```\n{journal_text}\n```\n")

    await journal.stop()
    report.write_text(
        f"# Session probe {stamp}\n\nRobot `{host}`. {len(steps)} steps.\n\n"
        "| step | turn ms | first audio ms | spoke as | not in cast | errors |\n|---|---|---|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n"
        + "\n".join(detail),
        encoding="utf-8",
    )
    print(f"\nreport: {report}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="reachy-mini.local")
    ap.add_argument("--from", dest="steps_file", type=Path, help="one typed turn per line; # comments skipped")
    ap.add_argument("--say", action="append", default=[], help="a single typed turn (repeatable)")
    ap.add_argument("--out", type=Path, default=Path("probe-runs"))
    ap.add_argument("--reply-timeout", type=float, default=150.0)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds to keep reading activity after the idle frame")
    args = ap.parse_args()

    steps = list(args.say)
    if args.steps_file:
        for line in args.steps_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                steps.append(line)
    if not steps:
        ap.error("give --from FILE or --say TEXT")
    return asyncio.run(run(args.host, steps, args.out, args.reply_timeout, args.pause))


if __name__ == "__main__":
    sys.exit(main())
