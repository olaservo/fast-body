"""Memory: where the store goes, when a conversation is picked back up, what bounds it."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from fast_body import memory
from fast_body.config import Config, export_memory_env


# ── where the store goes ──────────────────────────────────────────────────
def test_home_defaults_beside_the_user_not_the_package(monkeypatch):
    """Outside site-packages, so an app remove doesn't take the memory with it."""
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.delenv("FAST_BODY_MEMORY_DIR", raising=False)
    assert export_memory_env() == Path.home() / ".fast-body"


def test_home_is_exported_so_fast_agent_stops_following_the_cwd(monkeypatch):
    """The whole point: a daemon-launched app inherits the daemon's directory."""
    monkeypatch.delenv("FAST_AGENT_HOME", raising=False)
    monkeypatch.setenv("FAST_BODY_MEMORY_DIR", "/tmp/somewhere")
    home = export_memory_env()
    assert home == Path("/tmp/somewhere")
    import os

    assert os.environ["FAST_AGENT_HOME"] == str(home)


def test_an_explicit_fast_agent_home_wins(monkeypatch):
    monkeypatch.setenv("FAST_AGENT_HOME", "/tmp/chosen")
    monkeypatch.setenv("FAST_BODY_MEMORY_DIR", "/tmp/ignored")
    assert export_memory_env() == Path("/tmp/chosen")


# ── test doubles ──────────────────────────────────────────────────────────
class FakeAgent:
    def __init__(self, window: int | None = 1_000_000):
        self.name = "body"
        self.message_history: list[object] = []
        self.persistence = True
        self.usage_accumulator = SimpleNamespace(context_window_size=window)

    def set_session_history_persistence_enabled(self, enabled: bool) -> None:
        self.persistence = enabled


class FakeBrain:
    def __init__(self, agent: FakeAgent | None = None):
        self.agent = agent or FakeAgent()

    def resolve_agent(self, _name):
        return self.agent


def _session(name="2608272127-abc", hours_ago=1.0, title=""):
    when = datetime.now() - timedelta(hours=hours_ago)
    info = SimpleNamespace(
        name=name,
        created_at=when,
        last_activity=when,
        metadata={"title": title} if title else {},
    )
    return SimpleNamespace(info=info)


class FakeManager:
    def __init__(self, latest=None, sessions=None):
        self.latest = latest
        self.sessions = sessions if sessions is not None else []
        self.resumed: list[str] = []
        self.deleted: list[str] = []

    def load_latest_session(self, *, require_content=False):
        return self.latest

    async def resume_session_agents_async(self, agents, name, fallback_agent_name=None):
        self.resumed.append(name)
        return SimpleNamespace(session=self.latest)

    def list_sessions(self, *, include_empty=True):
        return [s.info for s in self.sessions]

    def delete_session(self, name):
        self.deleted.append(name)
        return any(s.info.name == name for s in self.sessions)


class LeasingManager(FakeManager):
    """fast-agent 0.10.16: the session being written is leased, and a plain
    delete of it comes back False ("busy")."""

    def __init__(self, live, sessions=None):
        super().__init__(sessions=sessions)
        self.current_session = live
        self.owned_deleted: list[str] = []

    def delete_session(self, name):
        self.deleted.append(name)
        if name == self.current_session.info.name:
            return False
        return any(s.info.name == name for s in self.sessions)

    def delete_owned_session(self, session):
        self.owned_deleted.append(session.info.name)
        return True


@pytest.fixture
def bound(monkeypatch):
    """Bind a fake session manager, bypassing the real fast-agent context."""

    def _bind(manager):
        monkeypatch.setattr(memory, "_manager", manager)
        monkeypatch.setattr(memory, "bind_manager", lambda: manager)
        return manager

    return _bind


# ── the resume policy ─────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_resumes_a_recent_conversation(bound, monkeypatch):
    monkeypatch.setenv("FAST_BODY_MEMORY_RESUME_H", "6")
    manager = bound(FakeManager(latest=_session(hours_ago=2.0)))
    brain = FakeBrain()

    outcome = await memory.resume_recent(brain, Config())

    assert manager.resumed == ["2608272127-abc"]
    assert outcome.resumed is True
    assert "resumed" in outcome.status


class ModelledAgent(FakeAgent):
    """An agent that knows which brain it is on, the way a real one does."""

    def __init__(self, model):
        super().__init__()
        self.config = SimpleNamespace(model=model)
        self.llm = SimpleNamespace(model_name=model)
        self.models_set: list[str] = []

    async def set_model(self, model):
        self.models_set.append(model)
        self.config.model = model
        self.llm.model_name = model


class BrainSwappingManager(FakeManager):
    """Hydrates like fast-agent does: the session's saved model goes on the agent."""

    async def resume_session_agents_async(self, agents, name, fallback_agent_name=None):
        for agent in agents.values():
            await agent.set_model("moonshotai/Kimi-K3:together")
        return await super().resume_session_agents_async(agents, name, fallback_agent_name)


@pytest.mark.asyncio
async def test_resuming_keeps_the_cards_brain(bound, monkeypatch):
    """The transcript comes back; the model it was saved on does not."""
    monkeypatch.setenv("FAST_BODY_MEMORY_RESUME_H", "6")
    bound(BrainSwappingManager(latest=_session(hours_ago=1.0)))
    agent = ModelledAgent("sonnet")

    outcome = await memory.resume_recent(FakeBrain(agent), Config())

    assert outcome.resumed is True
    assert agent.llm.model_name == "sonnet"
    assert agent.models_set == ["moonshotai/Kimi-K3:together", "sonnet"]


@pytest.mark.asyncio
async def test_resuming_on_the_same_brain_does_not_reattach(bound, monkeypatch):
    monkeypatch.setenv("FAST_BODY_MEMORY_RESUME_H", "6")
    bound(FakeManager(latest=_session(hours_ago=1.0)))
    agent = ModelledAgent("sonnet")

    await memory.resume_recent(FakeBrain(agent), Config())

    assert agent.models_set == []


@pytest.mark.asyncio
async def test_a_stale_conversation_is_left_behind(bound, monkeypatch):
    """A robot switched on the next morning starts a new conversation, not last night's."""
    monkeypatch.setenv("FAST_BODY_MEMORY_RESUME_H", "6")
    manager = bound(FakeManager(latest=_session(hours_ago=20.0)))

    outcome = await memory.resume_recent(FakeBrain(), Config())

    assert manager.resumed == []
    assert outcome.resumed is False
    assert "starting fresh" in outcome.status


@pytest.mark.asyncio
async def test_zero_hours_saves_without_resuming(bound, monkeypatch):
    monkeypatch.setenv("FAST_BODY_MEMORY_RESUME_H", "0")
    manager = bound(FakeManager(latest=_session(hours_ago=0.1)))

    outcome = await memory.resume_recent(FakeBrain(), Config())

    assert manager.resumed == []
    assert outcome.resumed is False
    assert "not resuming" in outcome.status


@pytest.mark.asyncio
async def test_memory_off_stops_the_save_hook(bound, monkeypatch):
    monkeypatch.setenv("ENABLE_MEMORY", "false")
    manager = bound(FakeManager(latest=_session(hours_ago=0.1)))
    brain = FakeBrain()

    outcome = await memory.resume_recent(brain, Config())

    assert brain.agent.persistence is False
    assert manager.resumed == []
    assert outcome.resumed is False
    assert "off" in outcome.status


@pytest.mark.asyncio
async def test_memory_off_still_binds_so_the_page_can_delete(bound, monkeypatch):
    """Turning memory off must not merely hide what was already written down."""
    monkeypatch.setenv("ENABLE_MEMORY", "false")
    saved = _session()
    bound(FakeManager(latest=saved, sessions=[saved]))

    await memory.resume_recent(FakeBrain(), Config())

    assert [s["name"] for s in memory.list_sessions()] == [saved.info.name]


@pytest.mark.asyncio
async def test_an_unresumable_session_does_not_stop_the_conversation(bound, monkeypatch):
    """A session from an older build is worth losing; a robot that won't talk isn't."""
    monkeypatch.setenv("FAST_BODY_MEMORY_RESUME_H", "6")

    class Broken(FakeManager):
        async def resume_session_agents_async(self, *a, **k):
            raise RuntimeError("snapshot from another version")

    bound(Broken(latest=_session(hours_ago=1.0)))

    outcome = await memory.resume_recent(FakeBrain(), Config())
    assert outcome.resumed is False  # so nothing tries to reopen a conversation it lost
    assert "starting fresh" in outcome.status


@pytest.mark.asyncio
async def test_nothing_saved_yet(bound):
    bound(FakeManager(latest=None))
    assert "nothing to resume" in (await memory.resume_recent(FakeBrain(), Config())).status


# ── the context budget ────────────────────────────────────────────────────
def test_budget_becomes_a_fraction_of_the_window(monkeypatch):
    """The knob is absolute tokens; fast-agent wants a share of the window."""
    monkeypatch.setenv("FAST_BODY_CONTEXT_BUDGET", "60000")
    settings = SimpleNamespace(compaction=SimpleNamespace(threshold=0.85))
    brain = FakeBrain(FakeAgent(window=1_000_000))
    _patch_context(monkeypatch, settings)

    status = memory.apply_context_budget(brain, Config())

    assert settings.compaction.threshold == pytest.approx(0.06)
    assert "60000 tokens" in status


def test_an_unknown_window_is_reported_not_swallowed(monkeypatch):
    """`should_auto_compact` skips silently on an unknown window — say so instead."""
    monkeypatch.setenv("FAST_BODY_CONTEXT_BUDGET", "60000")
    settings = SimpleNamespace(compaction=SimpleNamespace(threshold=0.85))
    _patch_context(monkeypatch, settings)

    status = memory.apply_context_budget(FakeBrain(FakeAgent(window=None)), Config())

    assert "DISABLED" in status
    assert settings.compaction.threshold == 0.85  # left alone


def test_zero_budget_leaves_fast_agents_own_threshold(monkeypatch):
    monkeypatch.setenv("FAST_BODY_CONTEXT_BUDGET", "0")
    settings = SimpleNamespace(compaction=SimpleNamespace(threshold=0.85))
    _patch_context(monkeypatch, settings)

    status = memory.apply_context_budget(FakeBrain(), Config())

    assert settings.compaction.threshold == 0.85
    assert "off" in status


def _patch_context(monkeypatch, settings):
    """Stand in for fast_agent.context.get_current_context()."""
    import fast_agent.context as fa_context

    monkeypatch.setattr(fa_context, "get_current_context", lambda: SimpleNamespace(config=settings))


# ── the settings page's view ──────────────────────────────────────────────
def test_list_sessions_reports_age_and_never_the_transcript(bound):
    bound(FakeManager(sessions=[_session(hours_ago=3.0, title="about the garden")]))

    entries = memory.list_sessions()

    assert entries[0]["title"] == "about the garden"
    assert entries[0]["age_hours"] == pytest.approx(3.0, abs=0.1)
    assert not any("message" in k or "history" in k for k in entries[0])


def test_forget_all_removes_every_saved_conversation(bound):
    manager = bound(FakeManager(sessions=[_session("a"), _session("b")]))

    assert memory.forget_all() == 2
    assert manager.deleted == ["a", "b"]


def test_forgetting_the_live_conversation_goes_through_the_owner(bound):
    """Observed on the robot 2026-09-05: Forget everything removed nothing,
    because the running app held the lease on the session it was deleting."""
    live = _session("live")
    manager = bound(LeasingManager(live, sessions=[_session("old"), live]))

    assert memory.forget_all() == 2
    assert manager.owned_deleted == ["live"]
    assert manager.deleted == ["old"]
    assert memory.live_session_name() == "live"


def test_forgetting_with_no_manager_removes_the_files(bound, tmp_path, monkeypatch):
    """Before the brain is up the listing reads the disk, so the delete must too."""
    bound(None)
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    session_dir = tmp_path / "sessions" / "2609050225-abc"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}")

    assert memory.delete_session("2609050225-abc") is True
    assert not session_dir.exists()
    assert memory.delete_session("2609050225-abc") is False
    assert memory.delete_session("../sessions") is False


@pytest.mark.asyncio
async def test_forget_live_clears_what_the_brain_is_holding():
    cleared = []
    agent = SimpleNamespace(clear=lambda: cleared.append(True))
    assert await memory.forget_live(SimpleNamespace(resolve_agent=lambda _n: agent)) is True
    assert cleared == [True]


def test_list_is_empty_when_sessions_are_unavailable(bound, tmp_path, monkeypatch):
    bound(None)
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)  # no sessions on disk either
    assert memory.list_sessions() == []
    assert memory.delete_session("a") is False


def test_list_reads_the_disk_before_the_brain_is_up(bound, tmp_path, monkeypatch):
    """The page loads during the boot, when no manager is bound yet; the files are there."""
    import json

    bound(None)
    monkeypatch.setattr(memory, "MEMORY_DIR", tmp_path)
    when = (datetime.now() - timedelta(minutes=20)).isoformat()
    saved = tmp_path / "sessions" / "2609050225-gzuLZv"
    saved.mkdir(parents=True)
    (saved / "session.json").write_text(
        json.dumps(
            {
                "session_id": "2609050225-gzuLZv",
                "created_at": when,
                "last_activity": when,
                "metadata": {"title": None, "first_user_preview": "What's the gate code again?"},
            }
        ),
        encoding="utf-8",
    )
    (saved / "history_body.json").write_text('{"messages": [{"role": "user"}]}', encoding="utf-8")
    empty = tmp_path / "sessions" / "2609050300-empty"  # opened, never written to
    empty.mkdir()
    (empty / "session.json").write_text(json.dumps({"session_id": "x", "created_at": when}), encoding="utf-8")

    (entry,) = memory.list_sessions()
    assert entry["name"] == "2609050225-gzuLZv"
    assert entry["title"] == "What's the gate code again?"
    assert entry["age_hours"] == pytest.approx(0.3, abs=0.1)
    assert not any("message" in k or "history" in k for k in entry)


# ── picking a conversation back up ────────────────────────────────────────
class FakeMessage:
    def __init__(self, role: str, text: str):
        self.role = role
        self._text = text

    def first_text(self) -> str:
        return self._text

    def last_text(self) -> str:
        return self._text


def test_reopening_quotes_the_robots_own_last_words():
    history = [
        FakeMessage("user", "what's the gate code"),
        FakeMessage("assistant", "It's 4291."),
    ]
    assert memory.reopening_line(history) == "Picking up where we left off. I'd just said: It's 4291."


def test_reopening_takes_one_sentence_not_the_whole_reply():
    history = [FakeMessage("assistant", "First bit. Second bit. Third bit.")]
    assert memory.reopening_line(history).endswith("First bit.")


def test_reopening_shortens_a_long_sentence():
    history = [FakeMessage("assistant", "word " * 200)]
    line = memory.reopening_line(history)
    assert len(line) < 250
    assert line.endswith("…")


def test_reopening_skips_a_turn_that_ended_in_a_tool_call():
    """The robot said "Pending tool call: read_skill" out loud (2026-09-05):
    fast-agent's preview labels such a turn for a session picker."""
    history = [
        FakeMessage("assistant", "You push the vestry door open."),
        FakeMessage("user", "I search the desk"),
        FakeMessage("assistant", ""),  # stopped for a tool call, no text
    ]
    assert memory.reopening_line(history).endswith("You push the vestry door open.")


def test_nothing_to_reopen_with_when_the_robot_never_spoke():
    assert memory.reopening_line([FakeMessage("user", "hello")]) is None
    assert memory.reopening_line([]) is None


def test_transcript_carries_both_sides_for_the_page():
    history = [
        FakeMessage("user", "hello"),
        FakeMessage("assistant", "hi"),
    ]
    assert memory.resumed_transcript(history) == [
        {"role": "user", "text": "hello"},
        {"role": "robot", "text": "hi"},
    ]


def test_transcript_skips_empty_and_unknown_roles():
    history = [
        FakeMessage("assistant", "   "),
        FakeMessage("system", "ignored"),
        FakeMessage("user", "kept"),
    ]
    assert memory.resumed_transcript(history) == [{"role": "user", "text": "kept"}]
