"""Config loads from the environment and validates the voice backend."""

from __future__ import annotations

from fast_body.config import Config


def test_servers_parsed_from_csv(monkeypatch):
    monkeypatch.setenv("FAST_BODY_SERVERS", "fetch, memory ,home-assistant")
    cfg = Config()
    assert cfg.servers == ["fetch", "memory", "home-assistant"]


def test_servers_empty_by_default(monkeypatch):
    monkeypatch.delenv("FAST_BODY_SERVERS", raising=False)
    assert Config().servers == []


def test_flags_and_voice(monkeypatch):
    monkeypatch.setenv("ENABLE_CAMERA", "false")
    monkeypatch.setenv("VOICE_BACKEND", "text")
    cfg = Config()
    assert cfg.enable_camera is False
    assert cfg.voice_backend == "text"


def test_speak_between_tools_is_on_unless_turned_off(monkeypatch):
    monkeypatch.delenv("SPEAK_BETWEEN_TOOLS", raising=False)
    assert Config().speak_between_tools is True
    monkeypatch.setenv("SPEAK_BETWEEN_TOOLS", "false")
    assert Config().speak_between_tools is False


def test_validate_requires_openai_key_for_openai_backend(monkeypatch):
    monkeypatch.setenv("VOICE_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    errors = Config().validate()
    assert any("OPENAI_API_KEY" in e for e in errors)


def test_validate_ok_for_text_backend(monkeypatch):
    monkeypatch.setenv("VOICE_BACKEND", "text")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert Config().validate() == []


# ── where config files are looked for ─────────────────────────────────────
def test_package_dir_is_the_installed_package():
    from fast_body import config as config_module

    assert config_module.PACKAGE_DIR.name == "fast_body"
    assert (config_module.PACKAGE_DIR / "config.py").is_file()


def test_config_is_searched_in_the_package_dir_first(tmp_path, monkeypatch):
    """An installed app keeps its .env beside the package, with no repo root."""
    from fast_body import config as config_module

    package_dir = tmp_path / "site-packages" / "fast_body"
    package_dir.mkdir(parents=True)
    (package_dir / ".env").write_text("OPENAI_API_KEY=from-package\n")

    monkeypatch.setattr(config_module, "_CONFIG_DIRS", (package_dir, tmp_path / "repo"))
    assert config_module._find_config(".env") == package_dir / ".env"


def test_config_falls_back_to_the_repo_root(tmp_path, monkeypatch):
    from fast_body import config as config_module

    package_dir = tmp_path / "src" / "fast_body"
    repo_root = tmp_path
    package_dir.mkdir(parents=True)
    (repo_root / ".env").write_text("OPENAI_API_KEY=from-repo\n")

    monkeypatch.setattr(config_module, "_CONFIG_DIRS", (package_dir, repo_root))
    assert config_module._find_config(".env") == repo_root / ".env"


def test_missing_config_is_not_an_error(tmp_path, monkeypatch):
    from fast_body import config as config_module

    monkeypatch.setattr(config_module, "_CONFIG_DIRS", (tmp_path,))
    assert config_module._find_config(".env") is None


def test_web_chat_port_falls_back_when_invalid(monkeypatch):
    from fast_body.config import Config

    monkeypatch.setenv("WEB_CHAT_PORT", "not-a-port")
    assert Config().web_chat_port == 8080
    monkeypatch.setenv("WEB_CHAT_PORT", "99999")
    assert Config().web_chat_port == 8080
    monkeypatch.setenv("WEB_CHAT_PORT", "9000")
    assert Config().web_chat_port == 9000


# ── speech gating (Whisper invents text for non-speech audio) ─────────────
def test_speech_gates_have_sane_defaults():
    from fast_body.config import Config

    c = Config()
    assert c.vad_aggressiveness == 3  # strictest; the mic is next to fans
    assert c.stt_min_speech_ms == 400  # a click is not a word
    assert c.stt_min_rms > 0
    assert c.stt_language == "en"  # unpinned, noise gets answered in any language


def test_speech_gates_are_tunable(monkeypatch):
    from fast_body.config import Config

    monkeypatch.setenv("VAD_AGGRESSIVENESS", "1")
    monkeypatch.setenv("STT_MIN_SPEECH_MS", "250")
    monkeypatch.setenv("STT_LANGUAGE", "")
    c = Config()
    assert (c.vad_aggressiveness, c.stt_min_speech_ms, c.stt_language) == (1, 250, "")


# ── fast-agent TUI mode ───────────────────────────────────────────────────
def test_tui_needs_no_voice_credentials(monkeypatch):
    """The TUI is its own input and output, so no STT/TTS key is required."""
    from fast_body.config import Config

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Config(voice_backend="openai").validate()  # normally an error
    assert Config(voice_backend="openai", tui=True).validate() == []


def test_tui_is_off_by_default():
    from fast_body.config import Config

    assert Config().tui is False


# ── --tui speech ──────────────────────────────────────────────────────────
def test_tui_runs_silent_without_a_key(monkeypatch):
    """No credentials means no speech, not a broken TUI."""
    import threading

    from fast_body.app import FastBodyCore
    from fast_body.config import Config
    from tests.conftest import FakeRobot

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    core = FastBodyCore(
        FakeRobot(),
        owns_robot=False,
        stop_event=threading.Event(),
        config=Config(tui=True, voice_backend="openai", openai_api_key=""),
    )
    assert core._build_tui_voice() is None


async def test_tui_speaks_each_reply():
    """interactive() returns each reply through _send_interactive_message."""
    import threading

    from fast_body.app import FastBodyCore
    from fast_body.config import Config
    from tests.conftest import FakeRobot

    spoken: list[str] = []

    class FakeVoice:
        async def speak(self, text):
            spoken.append(text)

    class FakeBrain:
        async def _send_interactive_message(self, message, agent_name=None):
            return f"reply to {message}"

    core = FastBodyCore(FakeRobot(), owns_robot=False, stop_event=threading.Event(), config=Config(tui=True))
    core.voice = FakeVoice()
    brain = FakeBrain()
    core._speak_tui_replies(brain)

    assert await brain._send_interactive_message("hello") == "reply to hello"
    assert spoken == ["reply to hello"]


async def test_tui_hook_survives_a_missing_seam(caplog):
    """A fast-agent upgrade could move the private method; don't crash."""
    import threading

    from fast_body.app import FastBodyCore
    from fast_body.config import Config
    from tests.conftest import FakeRobot

    core = FastBodyCore(FakeRobot(), owns_robot=False, stop_event=threading.Event(), config=Config(tui=True))
    core.voice = object()
    core._speak_tui_replies(object())  # no _send_interactive_message
    assert "stay silent" in caplog.text


def test_barge_in_is_on_by_default(monkeypatch):
    """The robot's own audio path cancels echo, so the mic can stay live."""
    monkeypatch.delenv("ENABLE_BARGE_IN", raising=False)
    assert Config().enable_barge_in is True


def test_barge_in_can_be_turned_off_for_external_speakers(monkeypatch):
    monkeypatch.setenv("ENABLE_BARGE_IN", "false")
    assert Config().enable_barge_in is False


def test_brain_timeout_defaults_high_enough_not_to_fire(monkeypatch):
    """A ceiling on a stalled provider, not a latency budget — turns run 5-20s."""
    monkeypatch.delenv("BRAIN_TIMEOUT_S", raising=False)
    assert Config().brain_timeout_s == 90


def test_brain_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("BRAIN_TIMEOUT_S", "15")
    assert Config().brain_timeout_s == 15


def test_face_tracking_waits_to_be_asked(monkeypatch):
    """It shares the camera with stills and moves the head mid-capture."""
    monkeypatch.delenv("FACE_TRACKING", raising=False)
    assert Config().face_tracking is False


def test_face_tracking_can_be_turned_on(monkeypatch):
    monkeypatch.setenv("FACE_TRACKING", "on")
    assert Config().face_tracking is True


def test_face_tracking_is_settable_from_the_page():
    from fast_body.config import SETTABLE_ENV

    assert "FACE_TRACKING" in SETTABLE_ENV


# ── which servers a card may name at build time ───────────────────────────
def test_packaged_servers_are_the_yaml_ones():
    """fast-agent refuses names it does not know, so agent.py filters card servers by this.
    Per-install servers are not packaged."""
    from fast_body.config import packaged_servers

    names = packaged_servers()
    assert names == set()  # every server, Home Assistant included, comes from the settings page
