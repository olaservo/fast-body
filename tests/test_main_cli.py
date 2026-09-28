"""CLI guards: one controller at a time, and flag plumbing."""

from __future__ import annotations

from fast_body.main import _running_app_name


def test_running_app_is_named():
    status = {"info": {"name": "fast_body"}, "state": "running", "error": None}
    assert _running_app_name(status) == "fast_body"


def test_starting_app_also_counts():
    status = {"info": {"name": "fast_body"}, "state": "starting", "error": None}
    assert _running_app_name(status) == "fast_body"


def test_idle_daemon_is_none():
    assert _running_app_name(None) is None  # the endpoint returns null when idle
    assert _running_app_name({"info": None, "state": "stopped"}) is None


def test_nameless_running_app_still_blocks():
    assert _running_app_name({"info": {}, "state": "running"}) == "an app"


def test_cli_flags_parse(monkeypatch):
    import sys

    from fast_body.main import parse_args

    monkeypatch.setattr(
        sys, "argv",
        ["fast-body", "--tui", "--personality", "butler", "--voice", "marin",
         "--host", "reachy-mini.local", "--take-over"],
    )
    args = parse_args()
    assert (args.personality, args.voice) == ("butler", "marin")
    assert (args.host, args.take_over) == ("reachy-mini.local", True)


def test_debug_logging_keeps_websockets_quiet():
    import logging

    from fast_body.main import setup_logging

    setup_logging(debug=True)
    assert logging.getLogger("websockets").getEffectiveLevel() >= logging.WARNING
