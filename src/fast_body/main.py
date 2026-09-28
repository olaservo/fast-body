"""fast-body — standalone CLI entry point.

Constructs a `ReachyMini` (real, networked, or a spawned simulator) and runs the
same `FastBodyCore` the daemon uses. The daemon path lives in `app.py`.

    fast-body                # connect to a robot (or local daemon), use the mic
    fast-body --sim          # spawn a MuJoCo simulator
    fast-body --text         # type instead of speaking (no audio needed)
    fast-body --console      # dev console: see the brain, type or speak, Esc to interrupt
    fast-body --tui          # fast-agent's own TUI; the robot speaks the replies
    fast-body --debug        # verbose logging
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

from fast_body.config import Config

# The daemon regex-scrapes main.py for the first quoted assignment to this name
# (comments included) and puts the URL behind the desktop app's open button.
# Keep it a plain literal, the only such line in this file, and in sync with
# WEB_CHAT_PORT's default in config.py.
custom_app_url = "http://0.0.0.0:8080/"


def setup_logging(debug: bool = False, console: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.DEBUG if debug else logging.WARNING)
    # Never below WARNING, even in debug: at DEBUG the realtime session logs a
    # base64 line per audio frame, which buries everything else in the journal.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    # In the dev console the brain renders the turn, and our INFO lines on
    # stderr break prompt_toolkit's prompt.
    if console and not debug:
        logging.getLogger("fast_body").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="fast-body — an embodied fast-agent for Reachy Mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sim", action="store_true", help="Spawn a MuJoCo simulator instead of a real robot")
    parser.add_argument("--text", action="store_true", help="Type instead of speaking (VOICE_BACKEND=text)")
    parser.add_argument(
        "--console",
        action="store_true",
        help="Interactive dev console: see the brain + type or speak + Escape to interrupt",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Drive the robot from fast-agent's own TUI; it speaks the replies aloud",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--robot-name", type=str, default=None, help="Robot name (default: auto-discover)")
    parser.add_argument(
        "--personality",
        type=str,
        default=None,
        metavar="NAME",
        help="Personality card to play (overrides FAST_BODY_PERSONALITY)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        metavar="NAME",
        help="TTS voice (overrides OPENAI_TTS_VOICE; 'auto' follows the personality card)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        metavar="ADDRESS",
        help="Robot address; forces the network connection instead of trying localhost first",
    )
    parser.add_argument(
        "--take-over",
        action="store_true",
        help="If the daemon is already running an app, stop it and take control",
    )
    return parser.parse_args()


def _running_app_name(status: object) -> str | None:
    """The app named by a current-app-status payload, or None if idle."""
    if not isinstance(status, dict) or status.get("state") not in ("starting", "running"):
        return None
    info = status.get("info")
    name = info.get("name") if isinstance(info, dict) else None
    return name or "an app"


def _daemon_app_in_control(host: str, port: int) -> str | None:
    """Name of the app the daemon is currently running, or None. Best-effort:
    an unreachable or older daemon just means no answer, not an error."""
    import json
    from urllib.request import urlopen

    try:
        with urlopen(f"http://{host}:{port}/api/apps/current-app-status", timeout=3.0) as r:
            return _running_app_name(json.load(r))
    except Exception:
        return None


def _stop_daemon_app(host: str, port: int) -> bool:
    from urllib.request import Request, urlopen

    try:
        with urlopen(Request(f"http://{host}:{port}/api/apps/stop-current-app", method="POST"), timeout=30.0):
            return True
    except Exception as e:
        logging.getLogger(__name__).error("could not stop the daemon's app: %s", e)
        return False


def main() -> None:
    args = parse_args()

    # Env overrides go in before Config(): agent.py reads them at import time
    # (console and TUI decide `quiet`, the personality card loads there), and
    # the voice backend resolves the voice from the environment.
    if args.console:
        os.environ["FAST_BODY_CONSOLE"] = "1"
    if args.tui:
        os.environ["FAST_BODY_TUI"] = "1"
        os.environ["FAST_BODY_CONSOLE"] = "1"
    if args.personality:
        os.environ["FAST_BODY_PERSONALITY"] = args.personality
    if args.voice:
        os.environ["OPENAI_TTS_VOICE"] = args.voice

    config = Config()
    setup_logging(config.debug or args.debug, console=args.console)
    logger = logging.getLogger(__name__)
    if args.text:
        config.voice_backend = "text"

    errors = config.validate()
    if errors:
        for e in errors:
            logger.error("config error: %s", e)
        sys.exit(1)

    from reachy_mini import ReachyMini

    from fast_body.app import run_core

    kwargs: dict = {}
    if args.robot_name:
        kwargs["robot_name"] = args.robot_name
    elif config.robot_name:
        kwargs["robot_name"] = config.robot_name
    if args.sim:
        # localhost_only, not the SDK's auto mode: auto falls back to
        # reachy-mini.local when localhost refuses, so a sim run whose daemon is
        # slow to come up (or never spawned — the SDK's "already running" check
        # matches any process whose command line mentions the daemon) would
        # wake and drive the real robot on the LAN.
        kwargs.update(spawn_daemon=True, use_sim=True, host="localhost", connection_mode="localhost_only")
    elif args.host:
        # Skip the SDK's localhost-first auto mode, which connects to a daemon
        # on this machine (the desktop app runs one) instead of the robot.
        kwargs.update(host=args.host, connection_mode="network")

    logger.info("connecting to Reachy Mini%s…", " (sim)" if args.sim else "")
    try:
        robot = ReachyMini(**kwargs)
    except Exception as e:
        logger.error("robot connection failed: %s", e)
        sys.exit(1)

    # Exactly one process may command the body — a second one interleaves its
    # targets with ours (observed as the antennas thrashing between two streams).
    client = getattr(robot, "client", None)
    daemon_host = getattr(client, "host", None)
    daemon_port = int(getattr(client, "port", 8000))
    logger.info("connected to daemon at %s:%s (%s)", daemon_host, daemon_port, getattr(robot, "connection_mode", "?"))
    if not args.sim and daemon_host:
        # localhost on Windows/macOS can only be the desktop app's daemon, not a
        # robot; on Linux it may be the robot itself (running on it over SSH).
        if daemon_host == "localhost" and sys.platform in ("win32", "darwin"):
            logger.warning(
                "this is a daemon on THIS machine, not the robot — quit the desktop app, or pass --host <robot-address>"
            )
        running = _daemon_app_in_control(daemon_host, daemon_port)
        if running:
            if args.take_over:
                logger.info("stopping the daemon's running app (%s)…", running)
                if not _stop_daemon_app(daemon_host, daemon_port):
                    sys.exit(1)
            else:
                logger.error(
                    "the daemon is already running an app (%s); both would command the body at once. "
                    "Stop it from the dashboard, or rerun with --take-over.",
                    running,
                )
                sys.exit(1)

    stop_event = threading.Event()
    try:
        run_core(robot, owns_robot=True, stop_event=stop_event, config=config)
    except KeyboardInterrupt:
        logger.info("interrupted")
        stop_event.set()


if __name__ == "__main__":
    main()
