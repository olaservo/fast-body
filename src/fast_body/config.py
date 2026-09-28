"""Typed configuration for fast-body.

Loads `.env` and exposes a single `Config` dataclass. The *brain* provider is
configured in `fast-agent.config.yaml` (read by fast-agent itself), so this file
only covers what the body needs: voice, robot, and feature flags.

**Where config files are looked for.** Two very different layouts have to work:

- a source checkout, where `.env` sits at the repo root beside `pyproject.toml`;
- an installed app on the robot, where the daemon has pip-installed us into
  `apps_venv` and there is no repo root at all.

So we search the *package directory* first — `.../site-packages/fast_body/` — and
fall back to the repo root. The package directory is the platform's per-app config
convention: the daemon launches apps with a copy of its own environment
(`os.environ.copy()`), which offers no per-app hook, and the conversation app
answers this the same way, with an instance `.env` next to its installed package.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .../site-packages/fast_body/ when installed; <repo>/src/fast_body/ from source.
PACKAGE_DIR = Path(__file__).resolve().parent
# Only meaningful for a source checkout (src layout: src/fast_body -> src -> root).
PROJECT_ROOT = PACKAGE_DIR.parent.parent

# Most specific first: a deployed instance's own config beats the checkout's.
_CONFIG_DIRS = (PACKAGE_DIR, PROJECT_ROOT)


def _find_config(filename: str) -> Path | None:
    """First existing `filename` across the config directories, or None."""
    for directory in _CONFIG_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


# fast-agent would otherwise look next to the working directory; we pass an
# explicit path so the app works however it was launched (daemon, CLI, sim).
FAST_AGENT_CONFIG = _find_config("fast-agent.config.yaml")


def packaged_servers() -> set[str]:
    """Names of the MCP servers the packaged fast-agent config defines.

    fast-agent refuses to build an agent whose `servers` names one it does not
    know, so a personality card can only ask for these by name at build time.
    Servers the user adds from the settings page live elsewhere
    (`mcp_servers.py`) and attach once the brain is up; agent.py uses this to
    tell the two apart rather than fail the startup.
    """
    if FAST_AGENT_CONFIG is None:
        return set()
    try:
        import yaml  # type: ignore[import-untyped]  # a fast-agent dep

        data = yaml.safe_load(FAST_AGENT_CONFIG.read_text(encoding="utf-8")) or {}
    except Exception as e:  # a broken YAML fails later, with fast-agent's own message
        logger.debug("could not read %s: %s", FAST_AGENT_CONFIG, e)
        return set()
    servers = (data.get("mcp") or {}).get("servers") if isinstance(data, dict) else None
    return set(servers) if isinstance(servers, dict) else set()


_DOTENV = _find_config(".env")
if _DOTENV is not None:
    # override=False: a real environment variable always beats the file.
    load_dotenv(_DOTENV, override=False)


# ── Memory ────────────────────────────────────────────────────────────────
# fast-agent saves a session after every turn under a home it resolves from
# the working directory, which for a daemon-launched app is the daemon's.
# Pinning FAST_AGENT_HOME makes saving work.
#
# Outside site-packages because an app remove wipes the package directory. The
# same home also carries skills, agent cards and user-added MCP servers.
DEFAULT_MEMORY_DIR = Path.home() / ".fast-body"


def export_memory_env() -> Path:
    """Pin fast-agent's home, and return it.

    Runs at import, before agent.py constructs the FastAgent — settings are read
    then, and a home resolved later would be the wrong one. An already-set
    FAST_AGENT_HOME wins, so a developer can point a run somewhere else.
    """
    existing = os.getenv("FAST_AGENT_HOME", "").strip()
    if existing:
        return Path(existing).expanduser()
    configured = os.getenv("FAST_BODY_MEMORY_DIR", "").strip()
    home = Path(configured).expanduser() if configured else DEFAULT_MEMORY_DIR
    os.environ["FAST_AGENT_HOME"] = str(home)
    return home


MEMORY_DIR = export_memory_env()


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _number(name: str, default: float) -> float:
    """Read a float env var, falling back to the default if it isn't a number."""
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _port(name: str, default: int) -> int:
    """Read a TCP port env var, falling back to the default if it isn't one."""
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return value if 1 <= value <= 65535 else default


def config_sources() -> str:
    """Which config files were found — worth logging where there is no terminal."""
    return (
        f".env={_DOTENV or 'not found'}; "
        f"fast-agent.config.yaml={FAST_AGENT_CONFIG or 'not found (fast-agent will search)'}; "
        f"memory={MEMORY_DIR}; skills={MEMORY_DIR / 'skills'}; personalities={MEMORY_DIR / 'personalities'}"
    )


# Where the settings page writes. An existing file wins so a source checkout keeps
# using its own .env; otherwise we create one beside the installed package.
ENV_FILE = _DOTENV or (PACKAGE_DIR / ".env")

# Only these may be written from the settings page. A whitelist, so a request
# can't set PATH or anything else the process reads.
SETTABLE_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "FAST_BODY_PERSONALITY",
    "OPENAI_TTS_VOICE",
    "FACE_TRACKING",
    "ENABLE_MEMORY",
)

# Voices the OpenAI TTS API accepts, for the settings page's picker. The list
# is for the UI, not a gate — an unrecognized value is passed through (with a
# logged warning) so a newly released voice works before we hear about it.
AVAILABLE_TTS_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)

DEFAULT_TTS_VOICE = "cedar"


def persist_env(updates: dict[str, str]) -> list[str]:
    """Write settings to `ENV_FILE` and apply them to this process.

    Returns the names actually written. Unknown names and blank values are
    ignored — blank means "leave alone", so the page never has to send a secret
    back to clear one by accident.
    """
    written = {name: value.strip() for name, value in updates.items() if name in SETTABLE_ENV and (value or "").strip()}
    if not written:
        return []

    for name, value in written.items():
        os.environ[name] = value

    try:
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.is_file() else []
        for name, value in written.items():
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{name}="):
                    lines[i] = f"{name}={value}"
                    break
            else:
                lines.append(f"{name}={value}")
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:  # keep it readable only by the account running the app
            ENV_FILE.chmod(0o600)
        except OSError:
            pass  # not supported everywhere; the write itself still happened
        logger.info("wrote %s to %s", ", ".join(sorted(written)), ENV_FILE)
    except OSError as e:
        # The values are live in this process even if the file write failed, so
        # the app can run now and the user can retry persisting.
        logger.error("could not write %s: %s", ENV_FILE, e)

    return sorted(written)


def env_status() -> dict[str, bool]:
    """Which settable keys currently have a value. Never returns the values."""
    return {name: bool(os.getenv(name, "").strip()) for name in SETTABLE_ENV}


@dataclass
class Config:
    """Runtime configuration loaded from environment variables."""

    # Voice — `openai` (STT + TTS), `realtime` (streaming recognition + TTS),
    # `text` (stdin), or `none` (browser page only).
    voice_backend: str = field(default_factory=lambda: os.getenv("VOICE_BACKEND", "openai"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    stt_model: str = field(default_factory=lambda: os.getenv("OPENAI_STT_MODEL", "gpt-transcribe"))
    tts_model: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    # Raw override. Blank or "auto" = follow the personality card's voice —
    # resolution lives in resolve_tts_voice().
    tts_voice: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_VOICE", "").strip())
    # Delivery steering — tone, pace, emotion. Blank = follow the personality card.
    # Not accepted by tts-1 / tts-1-hd, only by the gpt-4o-mini-tts family.
    tts_delivery: str = field(default_factory=lambda: os.getenv("TTS_DELIVERY", "").strip())
    # An OpenAI-compatible speech server instead of OpenAI (a Qwen3-TTS server
    # exposes /v1/audio/speech). Blank = OpenAI. The key defaults to
    # OPENAI_API_KEY, which STT still needs either way.
    tts_base_url: str = field(default_factory=lambda: os.getenv("TTS_BASE_URL", "").strip())
    tts_api_key: str = field(default_factory=lambda: os.getenv("TTS_API_KEY", "").strip())
    # Whisper-family models answer non-speech audio with confident invented text,
    # usually a stock subtitle phrase and often in another language. Three guards,
    # because none of them is reliable alone:
    #   - the mic sits beside fans and motors, so run the VAD at its strictest;
    #   - require enough voiced audio to be a word, not a click;
    #   - require enough level, since steady noise can satisfy the VAD quietly.
    vad_aggressiveness: int = field(default_factory=lambda: int(_number("VAD_AGGRESSIVENESS", 3)))
    stt_min_speech_ms: int = field(default_factory=lambda: int(_number("STT_MIN_SPEECH_MS", 400)))
    stt_min_rms: float = field(default_factory=lambda: _number("STT_MIN_RMS", 0.012))
    # Pin the transcription language.
    # Blank lets the model guess, and on noise it guesses wildly.
    stt_language: str = field(default_factory=lambda: os.getenv("STT_LANGUAGE", "en").strip())

    # Realtime backend — the server endpoints the speech, so the three guards
    # above don't apply. What replaces them is noise reduction ahead of the VAD
    # (`far_field` suits a mic across the room) and the VAD's own threshold.
    # `gpt-live-transcribe` streams deltas sooner but rejects turn detection
    # ("Turn detection is not supported for this transcription model"), which
    # would put endpointing back on us — the thing this backend exists to avoid.
    realtime_stt_model: str = field(default_factory=lambda: os.getenv("REALTIME_STT_MODEL", "gpt-transcribe"))
    realtime_noise_reduction: str = field(
        default_factory=lambda: os.getenv("REALTIME_NOISE_REDUCTION", "far_field").strip()
    )
    realtime_vad_threshold: float = field(default_factory=lambda: _number("REALTIME_VAD_THRESHOLD", 0.5))
    realtime_prefix_padding_ms: int = field(default_factory=lambda: int(_number("REALTIME_PREFIX_PADDING_MS", 300)))
    # 750 ms of silence ends the turn, matching the discrete backend.
    realtime_silence_ms: int = field(default_factory=lambda: int(_number("REALTIME_SILENCE_MS", 750)))
    # `examine(question)` reads a camera frame with this model (OpenAI, via the
    # same key as the voice), so the picture never enters the brain's history.
    vision_model: str = field(default_factory=lambda: os.getenv("FAST_BODY_VISION_MODEL", "gpt-5-mini").strip())

    # Brain — MCP servers (defined in fast-agent.config.yaml) the body attaches to.
    servers: list[str] = field(
        default_factory=lambda: [s.strip() for s in os.getenv("FAST_BODY_SERVERS", "").split(",") if s.strip()]
    )
    # Which personality card the robot plays — see personality.py. "default" is
    # the packaged fallback card (personality.DEFAULT_PERSONALITY; a literal here
    # because importing personality.py from config.py would be circular).
    personality: str = field(default_factory=lambda: os.getenv("FAST_BODY_PERSONALITY", "").strip() or "default")

    # Memory. fast-agent saves the conversation after every turn once it has a
    # home to write to (MEMORY_DIR above); these decide whether we let it, and
    # whether a new run picks the last one back up.
    enable_memory: bool = field(default_factory=lambda: _flag("ENABLE_MEMORY", True))
    # How stale the last session may be and still be resumed, in hours. 0 saves
    # but never resumes, so each run starts fresh and the record is still there.
    # A robot is switched on and off around a day, not a task: the default
    # carries a morning into an afternoon and lets last week go.
    memory_resume_h: float = field(default_factory=lambda: _number("FAST_BODY_MEMORY_RESUME_H", 6))
    # Say something on a resumed run, rather than carrying on as if nothing had
    # happened. fast-agent's CLI prints the last assistant message on --resume;
    # a robot has no terminal, so it speaks instead. Off means it just continues.
    memory_greeting: bool = field(default_factory=lambda: _flag("FAST_BODY_MEMORY_GREETING", True))
    # Ceiling on how much conversation is replayed to the brain each turn, in
    # tokens. fast-agent compacts on a *fraction* of the model's context window,
    # the wrong unit when the router models report a 1M window.
    # memory.apply_context_budget converts this budget into that fraction once
    # it knows the window. 0 leaves fast-agent's own threshold alone.
    context_budget_tokens: int = field(default_factory=lambda: int(_number("FAST_BODY_CONTEXT_BUDGET", 60000)))

    # Robot
    robot_name: str | None = field(default_factory=lambda: os.getenv("ROBOT_NAME") or None)

    # Feature flags
    enable_camera: bool = field(default_factory=lambda: _flag("ENABLE_CAMERA", True))
    # How strongly daemon-side face tracking owns the head while `look_at_me` is on.
    # The daemon interpolates between our commanded pose and the look-at aim, so
    # 1.0 is full eye contact and lower values let the active move show through.
    head_tracking_weight: float = field(default_factory=lambda: _number("HEAD_TRACKING_WEIGHT", 1.0))
    # …and while the robot is speaking, when its own expression matters most.
    head_tracking_speaking_weight: float = field(default_factory=lambda: _number("HEAD_TRACKING_SPEAKING_WEIGHT", 0.3))
    # Companion browser page: a chat transcript you can also type into, served by
    # the app itself. Works in both run modes, which is the point — it is the only
    # way to talk to the robot while the daemon runs it headless.
    enable_web_chat: bool = field(default_factory=lambda: _flag("ENABLE_WEB_CHAT", True))
    web_chat_port: int = field(default_factory=lambda: _port("WEB_CHAT_PORT", 8080))
    # 0.0.0.0 so you can reach a headless robot from your laptop. Set 127.0.0.1 to
    # keep it on the robot itself.
    web_chat_host: str = field(default_factory=lambda: os.getenv("WEB_CHAT_HOST", "0.0.0.0"))
    # Optional lock. Unset (the default) the page is open to the network it's
    # bound to, like the daemon on :8000 and the conversation app on :7860 —
    # which is what lets the desktop app's open button work, since the URL the
    # daemon scrapes for it can't carry a token. Set a value to require it.
    web_chat_token: str | None = field(default_factory=lambda: os.getenv("WEB_CHAT_TOKEN", "").strip() or None)
    # Audio-reactive head wobble while speaking (on-robot LOCAL audio backend only).
    enable_wobble: bool = field(default_factory=lambda: _flag("ENABLE_WOBBLE", True))
    # Follow the person's face from the moment the app starts. Off by default:
    # it shares the camera with `camera()` and moves the head while a still is
    # taken. The brain can still turn it on mid-conversation with `look_at_me`.
    face_tracking: bool = field(default_factory=lambda: _flag("FACE_TRACKING", False))
    # Let the user interrupt the robot by talking over it, and keep the mic live
    # while it speaks. On by default: the SDK builds webrtcdsp + webrtcechoprobe
    # into the robot's LOCAL audio path, so the mic doesn't hear the robot itself.
    # Turn it off for external or Bluetooth speakers, where the echo probe's
    # far-end reference no longer matches what is actually played — there the mic
    # is muted for the reply instead, at the cost of not hearing you during it.
    enable_barge_in: bool = field(default_factory=lambda: _flag("ENABLE_BARGE_IN", True))
    # Speak a turn's text as it arrives, ahead of each tool call, instead of once
    # after the last one.
    speak_between_tools: bool = field(default_factory=lambda: _flag("SPEAK_BETWEEN_TOOLS", True))
    # Ceiling on one brain turn, tool calls included. A stalled provider request
    # would otherwise hold the loop open forever. High enough never to fire on a
    # healthy turn.
    brain_timeout_s: float = field(default_factory=lambda: _number("BRAIN_TIMEOUT_S", 90))
    # How long a server's question (MCP elicitation) stays open before it is
    # cancelled. Longer than the brain timeout because people deliberate. See
    # choices.py.
    choice_timeout_s: float = field(default_factory=lambda: _number("CHOICE_TIMEOUT_S", 300))
    # How much speech to hold before playback starts. The reply streams in, and
    # the player empties whatever it has — too small a lead-in drains mid-word.
    # Raise it if speech stutters; lower it to start talking sooner.
    tts_lead_in_s: float = field(default_factory=lambda: _number("TTS_LEAD_IN_S", 0.6))
    # One discarded synthesis at startup, so the first reply is not the slow one.
    tts_warmup: bool = field(default_factory=lambda: _flag("TTS_WARMUP", True))
    debug: bool = field(default_factory=lambda: _flag("FAST_BODY_DEBUG", False))
    # Interactive dev console (CLI/dev only, never the headless daemon): un-blind the
    # brain's console, accept typed *or* spoken input, and let Escape interrupt Reachy.
    # The `--console` flag sets FAST_BODY_CONSOLE=1 so agent.py sees it at import time.
    console: bool = field(default_factory=lambda: _flag("FAST_BODY_CONSOLE", False))
    # Hand the terminal to fast-agent's own TUI instead of running our turn loop.
    # The body tools are registered either way, so typing there drives the robot.
    # The robot speaks the replies when an OpenAI key is set.
    tui: bool = field(default_factory=lambda: _flag("FAST_BODY_TUI", False))

    def tts_voice_overridden(self) -> bool:
        """Whether OPENAI_TTS_VOICE pins the voice regardless of personality."""
        return bool(self.tts_voice) and self.tts_voice.lower() != "auto"

    def resolve_tts_voice(self) -> str:
        """The voice to speak with: explicit override → personality card → default."""
        if self.tts_voice_overridden():
            return self.tts_voice
        # Lazy: personality.py imports this module, so the cycle only closes at
        # call time, and the card loader (fast-agent) stays unimported until a
        # voice is actually needed.
        from fast_body.personality import load_personality

        return load_personality(self.personality).voice or DEFAULT_TTS_VOICE

    def resolve_tts_delivery(self) -> str | None:
        """How to speak it: explicit override → personality card → nothing."""
        if self.tts_delivery:
            return self.tts_delivery
        from fast_body.personality import load_personality

        return load_personality(self.personality).delivery

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration errors (empty = OK)."""
        errors: list[str] = []
        if self.tui:
            return errors  # speech is optional in the TUI
        if self.voice_backend in ("openai", "realtime") and not self.openai_api_key:
            errors.append(
                f"OPENAI_API_KEY is required for the {self.voice_backend} voice backend (or set VOICE_BACKEND=text)"
            )
        return errors
