"""Robot-body tools exposed to the fast-agent brain.

Each function is registered with `@fast.tool`, so the brain can call it during a
turn to move the body. The functions are deliberately tiny — they look up the
live robot via `RobotContext.current()`, queue a move (returning immediately so
speech and motion overlap), and report a short status string back to the brain.

Registration happens at import time. `agent.py` imports this module after creating
the `fast` instance, so the tools attach to it before `fast.run()`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastmcp.utilities.types import Image

from fast_body.agent import fast
from fast_body.embodiment.context import RobotContext
from fast_body.embodiment.moves import AntennaMove, HeadLookMove, recorded_move

logger = logging.getLogger(__name__)

# Ceiling on one camera capture (run in a thread; see `_frame`).
_CAMERA_TIMEOUT_S = 10.0
# The capture above plus one vision call.
_EXAMINE_TIMEOUT_S = 30.0

Direction = Literal["left", "right", "up", "down", "front"]
AntennaPreset = Literal[
    "curious",
    "excited",
    "sad",
    "point_left",
    "point_right",
    "listen",
    "surprised",
    "shy",
    "angry",
    "confused",
    "neutral",
    "wiggle",
    "perk_left",
    "perk_right",
    "droop",
]

# Simple emotion names the brain uses → best-matching recorded animation.
_EMOTION_MAP: dict[str, str] = {
    "happy": "cheerful1",
    "sad": "sad1",
    "surprised": "surprised1",
    "curious": "curious1",
    "thinking": "thoughtful1",
    "confused": "confused1",
    "excited": "enthusiastic1",
    "scared": "scared1",
    "shy": "shy1",
    "angry": "irritated1",
    "bored": "boredom1",
    "proud": "proud1",
    "grateful": "grateful1",
    "tired": "tired1",
    "loving": "loving1",
    "fear": "fear1",
    "disgusted": "disgusted1",
    "relieved": "relief1",
    "impatient": "impatient1",
    "frustrated": "frustrated1",
    "success": "success1",
    "laughing": "laughing1",
    "welcoming": "welcoming1",
    "calming": "calming1",
}


@fast.tool
def look(direction: Direction) -> str:
    """Turn your head to look in a direction (left, right, up, down, or front).

    Use it to direct attention, glance around, or punctuate a thought. 'front'
    returns to a neutral, forward-facing pose.
    """
    ctx = RobotContext.current()
    head, antennas = ctx.movement.current_pose()
    ctx.movement.queue_move(HeadLookMove(direction, head, antennas, duration=1.0))
    return f"looking {direction}"


@fast.tool
def antennas(preset: AntennaPreset) -> str:
    """Flick your two antennas to a preset to show mood or emphasis.

    Quick and cheap — use it often. Presets: curious, excited, sad, point_left,
    point_right, listen, surprised, shy, angry, confused, neutral, wiggle,
    perk_left, perk_right, droop.
    """
    ctx = RobotContext.current()
    head, ant = ctx.movement.current_pose()
    ctx.movement.queue_move(AntennaMove(preset, head, ant, duration=0.6))
    return f"antennas: {preset}"


@fast.tool
def emotion(name: str, hold: bool = False) -> str:
    """Play a full-body emotional animation.

    Use simple names: happy, sad, surprised, curious, thinking, confused, excited,
    scared, shy, angry, bored, proud, grateful, tired, loving, fear, disgusted,
    relieved, impatient, frustrated, success, laughing, welcoming, calming.

    Set `hold=True` to keep the feeling on your face while you speak the reply:
    your antennas stay where the animation left them instead of relaxing as you
    start talking. Use it when the mood *is* the answer — sympathy for bad news,
    delight at good news, visible confusion at a question you can't parse. Leave
    it off for a passing reaction, or you will look stuck in one mood.
    """
    ctx = RobotContext.current()
    if ctx.emotions is None:
        return "emotions unavailable"

    try:
        available = list(ctx.emotions.list_moves())
    except Exception:
        available = []

    target = name if name in available else _EMOTION_MAP.get(name.lower())
    if not target or (available and target not in available):
        options = ", ".join(sorted(_EMOTION_MAP))
        return f"unknown emotion '{name}'. Try: {options}"

    try:
        ctx.movement.queue_move(recorded_move(target, ctx.emotions))
        # Only on success — a refused emotion must not silently suppress the cue.
        ctx.hold_expression = bool(hold)
        return f"feeling {name}" + (", and holding it while I answer" if hold else "")
    except Exception as e:
        logger.error("emotion '%s' failed: %s", name, e)
        return f"couldn't play emotion '{name}'"


@fast.tool
def dance(name: str) -> str:
    """Perform a short choreographed dance with your whole body.

    Use it to celebrate or respond to music. Call with an empty/unknown name to
    discover what's available.
    """
    ctx = RobotContext.current()
    if ctx.dances is None:
        return "dances unavailable"

    try:
        available = list(ctx.dances.list_moves())
    except Exception:
        available = []

    if name not in available:
        options = ", ".join(available[:20]) if available else "(library not loaded)"
        return f"unknown dance '{name}'. Available: {options}"

    try:
        ctx.movement.queue_move(recorded_move(name, ctx.dances))
        return f"dancing: {name}"
    except Exception as e:
        logger.error("dance '%s' failed: %s", name, e)
        return f"couldn't dance '{name}'"


@fast.tool
async def camera() -> Image | str:
    """Take a picture to see what is in front of you.

    Use this when asked to look at something, to see what the person is holding
    or wearing, to describe the scene, or for your opinion on how something
    looks. The camera is live — every call captures the current moment. If you
    are asked to look without being told what at, don't ask what to look at:
    take the picture and say what you see.

    The picture comes back to you directly, so describe it in your own voice —
    briefly, the way you would if you had just glanced up.
    """
    jpeg = await _frame(RobotContext.current())
    if isinstance(jpeg, str):
        return jpeg
    # Returned as image content, so the brain looks at the frame itself. Needs a
    # vision-capable model — the default is one.
    return Image(data=jpeg, format="jpeg")


@fast.tool
async def examine(question: str) -> str:
    """Read something in front of you and get the answer in words.

    Ask a precise question — "what number is on the die?", "what are the
    characteristics on this sheet?", "which handout is this?" — and a separate
    pair of eyes reads the current camera frame and answers. Unlike `camera()`,
    the picture itself never enters your conversation, so use this whenever you
    only need a fact from the scene, and every time when you are running a game.
    Tell the person what it said in your own words.
    """
    ctx = RobotContext.current()
    if ctx.vision is None:
        return "I can't examine things right now — my second pair of eyes isn't configured."
    jpeg = await _frame(ctx)
    if isinstance(jpeg, str):
        return jpeg
    try:
        answer = await asyncio.wait_for(ctx.vision.describe(jpeg, question), timeout=_EXAMINE_TIMEOUT_S)
    except TimeoutError:
        logger.error("examine: %s did not answer within %.0fs", ctx.vision.model, _EXAMINE_TIMEOUT_S)
        return "I looked, but my eyes didn't report back in time."
    except Exception as e:
        logger.error("examine: %s failed: %s", ctx.vision.model, e)
        return "I looked, but I couldn't make sense of what I saw."
    if not answer:
        return "I looked, but I have nothing to report."
    return answer


@fast.tool
def stop_moves() -> str:
    """Stop all current movement and hold still."""
    RobotContext.current().movement.clear_move_queue()
    return "holding still"


@fast.tool
def look_at_me(enable: bool = True) -> str:
    """Follow the person's face with your head, or stop following.

    Off by default. Turn it on when the person asks for eye contact or the moment
    calls for it, and off again before a big dance or when you want to look away.
    It shares the camera with `camera()` and moves your head while a picture is
    taken, so leave it off when you are mostly being asked to look at things.
    """
    ctx = RobotContext.current()
    if ctx.gaze is None:
        return "I can't track faces right now — my camera isn't available."
    if not ctx.gaze.set_enabled(enable):
        return "I couldn't change how I'm watching you just now."
    return "watching you" if enable else "no longer following your face"


async def _frame(ctx) -> bytes | str:
    """The current frame, or what to tell the brain instead.

    The capture blocks (a frame grab plus a pipeline state change), so it runs
    in a thread with its own ceiling; a blocked event loop would stop the app
    answering its shutdown signal.
    """
    if not ctx.camera:
        return "My camera is switched off."
    try:
        jpeg = await asyncio.wait_for(asyncio.to_thread(ctx.robot.media.get_frame_jpeg), timeout=_CAMERA_TIMEOUT_S)
    except TimeoutError:
        logger.error("camera capture timed out after %.0fs", _CAMERA_TIMEOUT_S)
        return "My camera didn't answer in time."
    except Exception as e:
        logger.error("camera capture failed: %s", e)
        return "I couldn't get an image from my camera right now."
    if jpeg is None:
        return "I couldn't see anything — no image from my camera."
    return jpeg
