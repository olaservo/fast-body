"""Robot tools queue the right moves against a fake RobotContext."""

from __future__ import annotations

import asyncio

import pytest

from fast_body.embodiment import tools
from fast_body.embodiment.context import RobotContext
from fast_body.embodiment.moves import AntennaMove, HeadLookMove, RecordedQueueMove
from tests.conftest import FakeLibrary, FakeMovement, FakeRobot


@pytest.fixture
def ctx():
    movement = FakeMovement()
    robot = FakeRobot()
    context = RobotContext(
        robot=robot,
        movement=movement,
        emotions=FakeLibrary(["cheerful1", "sad1"]),
        dances=FakeLibrary(["yeah_nod", "chicken_peck"]),
    )
    RobotContext.set(context)
    yield context
    RobotContext.clear()


def test_look_queues_head_move(ctx):
    result = tools.look("left")
    assert "left" in result
    assert isinstance(ctx.movement.queued[-1], HeadLookMove)


def test_antennas_queues_antenna_move(ctx):
    tools.antennas("curious")
    assert isinstance(ctx.movement.queued[-1], AntennaMove)


def test_emotion_maps_simple_name(ctx):
    result = tools.emotion("happy")  # maps to cheerful1
    assert isinstance(ctx.movement.queued[-1], RecordedQueueMove)
    assert ctx.movement.queued[-1].name == "cheerful1"
    assert "happy" in result


def test_emotion_unknown_reports_options(ctx):
    result = tools.emotion("nonsense-emotion")
    assert "unknown emotion" in result
    assert ctx.movement.queued == []


def test_dance_known_name(ctx):
    tools.dance("yeah_nod")
    assert ctx.movement.queued[-1].name == "yeah_nod"


def test_dance_unknown_lists_available(ctx):
    result = tools.dance("moonwalk")
    assert "unknown dance" in result
    assert "yeah_nod" in result


def test_stop_moves_clears(ctx):
    tools.stop_moves()
    assert ctx.movement.cleared == 1


def test_tools_require_installed_context():
    RobotContext.clear()
    with pytest.raises(RuntimeError):
        tools.look("front")


def _fake_media(**members):
    return type("M", (), {k: staticmethod(v) for k, v in members.items()})()


async def test_camera_handles_no_frame(ctx):
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: None)
    assert "couldn't see" in (await tools.camera()).lower()


async def test_camera_reports_a_capture_failure(ctx):
    def _boom():
        raise RuntimeError("camera is gone")

    ctx.robot.media = _fake_media(get_frame_jpeg=_boom)
    assert "couldn't get an image" in (await tools.camera()).lower()


def test_camera_is_a_coroutine_tool():
    """fast-agent awaits coroutine tools; a plain def would block the loop."""
    import inspect

    assert inspect.iscoroutinefunction(tools.camera)


async def test_camera_gives_up_rather_than_holding_the_turn(ctx, monkeypatch):
    """A stalled GStreamer capture must not hold the conversation open."""
    import threading

    started = threading.Event()

    def _hang():
        started.set()
        threading.Event().wait(2)  # far past the timeout, short enough not to linger

    monkeypatch.setattr(tools, "_CAMERA_TIMEOUT_S", 0.2)
    ctx.robot.media = _fake_media(get_frame_jpeg=_hang)

    result = await tools.camera()

    assert started.is_set()
    assert "in time" in result.lower()


async def test_a_stalled_camera_leaves_the_event_loop_running(ctx, monkeypatch):
    """The capture runs in a thread, so the loop keeps turning while it hangs."""
    import asyncio
    import threading

    monkeypatch.setattr(tools, "_CAMERA_TIMEOUT_S", 0.3)
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: threading.Event().wait(2))

    ticks = 0
    capture = asyncio.ensure_future(tools.camera())
    while not capture.done():
        await asyncio.sleep(0.01)
        ticks += 1
    await capture

    assert ticks > 5, f"loop only turned {ticks} times — the capture blocked it"


async def test_camera_hands_the_frame_to_the_brain(ctx):
    """The picture goes back as image content, not a description from elsewhere."""
    from fastmcp.utilities.types import Image

    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    result = await tools.camera()
    assert isinstance(result, Image)
    assert result.data == b"jpeg-bytes"


async def test_camera_needs_no_openai_key(ctx, monkeypatch):
    """camera() hands the frame to the brain; no OpenAI key is involved."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    assert await tools.camera() is not None


class _FakeVision:
    model = "fake-vision"

    def __init__(self, answer="a six", fail=False, hang=False):
        self.answer, self.fail, self.hang = answer, fail, hang
        self.calls: list[tuple[bytes, str]] = []

    async def describe(self, jpeg, question):
        self.calls.append((jpeg, question))
        if self.hang:
            await asyncio.sleep(3600)
        if self.fail:
            raise RuntimeError("vision is down")
        return self.answer


async def test_examine_answers_in_words_and_keeps_the_frame_out(ctx):
    ctx.vision = _FakeVision(answer="The die shows a six.")
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    result = await tools.examine("what number is on the die?")
    assert result == "The die shows a six."
    assert ctx.vision.calls == [(b"jpeg-bytes", "what number is on the die?")]


async def test_examine_without_a_reader_says_so(ctx):
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    assert "configured" in (await tools.examine("anything")).lower()


async def test_examine_handles_no_frame(ctx):
    ctx.vision = _FakeVision()
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: None)
    assert "couldn't see" in (await tools.examine("anything")).lower()
    assert ctx.vision.calls == []


async def test_examine_reports_a_vision_failure(ctx):
    ctx.vision = _FakeVision(fail=True)
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    assert "couldn't make sense" in (await tools.examine("anything")).lower()


async def test_examine_gives_up_on_a_stalled_reader(ctx, monkeypatch):
    monkeypatch.setattr(tools, "_EXAMINE_TIMEOUT_S", 0.05)
    ctx.vision = _FakeVision(hang=True)
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    assert "in time" in (await tools.examine("anything")).lower()


def test_examine_is_a_coroutine_tool():
    import inspect

    assert inspect.iscoroutinefunction(tools.examine)


def test_emotion_can_hold_the_expression(ctx):
    result = tools.emotion("sad", hold=True)
    assert ctx.hold_expression is True
    assert "holding" in result


def test_emotion_does_not_hold_by_default(ctx):
    tools.emotion("sad")
    assert ctx.hold_expression is False


def test_a_refused_emotion_does_not_hold(ctx):
    """A hold that suppresses the speaking cue with nothing on the face is worse."""
    ctx.hold_expression = False
    tools.emotion("not-an-emotion", hold=True)
    assert ctx.hold_expression is False


def test_nothing_turns_face_tracking_on_by_itself(ctx):
    """It costs camera contention and head movement, so it waits to be asked."""
    from fast_body.embodiment.gaze import GazeController

    assert "start at the start" not in tools.look_at_me.__doc__.lower()
    assert "off by default" in tools.look_at_me.__doc__.lower()
    assert GazeController.__module__  # the controller itself already starts off


async def test_camera_tools_decline_when_the_camera_is_off(ctx, monkeypatch):
    ctx.camera = False
    ctx.robot.media = _fake_media(get_frame_jpeg=lambda: b"jpeg-bytes")
    ctx.vision = _FakeVision()
    assert await tools.camera() == "My camera is switched off."
    assert await tools.examine("what is this?") == "My camera is switched off."
