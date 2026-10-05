"""Skills: where they are looked for, how they arrive from MCP servers, what they must not switch on."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from fast_body import skills
from fast_body.prompts import SKILLS_PLACEHOLDER, compose_instruction


# ── where skills are looked for ───────────────────────────────────────────
@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Env and install locations under tmp; none exist yet."""
    installed = tmp_path / "home" / "skills"
    monkeypatch.setattr(skills, "SKILLS_DIR", installed)
    monkeypatch.delenv("FAST_BODY_SKILLS", raising=False)
    return SimpleNamespace(installed=installed, root=tmp_path)


def test_no_directories_when_none_exist(dirs):
    """So agent.py passes None and fast-agent doesn't scan the working directory."""
    assert skills.skill_directories() == []


def test_env_then_each_servers_cache(dirs, monkeypatch):
    served = [dirs.installed / "guide-server", dirs.installed / "home-assistant"]
    for d in served:
        d.mkdir(parents=True)
    extra = dirs.root / "extra"
    extra.mkdir()
    missing = dirs.root / "missing"
    monkeypatch.setenv("FAST_BODY_SKILLS", os.pathsep.join([str(extra), str(missing)]))
    found = skills.skill_directories()
    # Served skills are listed per server, never the cache root itself.
    assert found == [p.resolve() for p in (extra, *sorted(served))]


def test_a_directory_named_twice_is_listed_once(dirs, monkeypatch):
    cache = dirs.installed / "guide-server"
    cache.mkdir(parents=True)
    monkeypatch.setenv("FAST_BODY_SKILLS", str(cache))
    assert skills.skill_directories() == [cache.resolve()]


def test_server_root_names_the_server_and_stays_a_single_segment(dirs):
    assert skills.server_root("guide-server") == dirs.installed / "guide-server"
    # Separators and leading dots go, so the name is one plain, visible segment.
    assert skills.server_root("../x/y") == dirs.installed / "_x_y"


def test_manifests_come_from_the_directories(dirs):
    _write_skill(dirs.installed / "guide-server" / "greet", "greet", "Say hello", body="Wave first.")
    (manifest,) = skills.load_manifests()
    assert manifest.name == "greet"
    assert manifest.path == (dirs.installed / "guide-server" / "greet" / "SKILL.md").resolve()


# ── the instruction ───────────────────────────────────────────────────────
async def test_placeholder_renders_to_nothing_without_skills():
    """fast-agent substitutes the placeholder, so no `{{agentSkills}}` reaches the model."""
    from fast_agent.core.instruction_refresh import build_instruction

    text = await build_instruction(compose_instruction("Persona."), skill_manifests=[])
    assert SKILLS_PLACEHOLDER not in text
    assert "<available_skills>" not in text


async def test_placeholder_lists_the_skills_and_the_reader_tool(dirs):
    from fast_agent.core.instruction_refresh import build_instruction

    _write_skill(dirs.installed / "guide-server" / "greet", "greet", "Say hello")
    text = await build_instruction(compose_instruction("Persona."), skill_manifests=skills.load_manifests())
    assert "<name>greet</name>" in text
    assert "read_skill" in text
    # The origin is visible: the path names the server, and the note says what that means.
    assert str(dirs.installed / "guide-server") in text
    assert "was served by the MCP server" in text


# ── the shell stays off ───────────────────────────────────────────────────
async def test_keep_shell_off_sets_no_shell_once_the_context_exists():
    class FakeApp:
        def __init__(self):
            self.context = None

        async def initialize(self):
            self.context = SimpleNamespace(no_shell=False)

    app = FakeApp()
    skills.keep_shell_off(app)
    await app.initialize()
    assert app.context.no_shell is True


# ── syncing from an MCP server ────────────────────────────────────────────
def _digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _skill_md(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body}\n"


def _write_skill(directory: Path, name: str, description: str, body: str = "Do it.") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(_skill_md(name, description, body), encoding="utf-8")


class FakeServer:
    """One MCP server serving skills over the SEP-2640 extension, in memory.

    Files are `relative path → text`. The digests and the listing are derived
    from them the way the real server does, so an edited file changes the
    skill's revision.
    """

    def __init__(self, name: str, skill: str, files: dict[str, str]):
        self.name = name
        self.skill = skill
        self.files = dict(files)
        self.reads: list[str] = []

    @property
    def root(self) -> str:
        return f"skill://served/{self.skill}"

    def _uri(self, relative: str) -> str:
        return f"{self.root}/{relative}"

    def _resources(self):
        from fast_agent.mcp.skills_extension import SkillResource

        return [SkillResource(uri=self._uri(rel), digest=_digest(text), size=len(text.encode())) for rel, text in self.files.items()]

    def _entry(self):
        import frontmatter
        from fast_agent.mcp.skills_extension import SkillEntry

        meta = frontmatter.loads(self.files["SKILL.md"]).metadata
        return SkillEntry(uri=self._uri("SKILL.md"), frontmatter=dict(meta), resources=self._resources())

    def registry(self):
        from fast_agent.skills.mcp_registry import McpSkillRegistry, _registry_skill

        skill = _registry_skill(self._entry(), server_name=self.name, server_version="0.1.0")
        return McpSkillRegistry(server_name=self.name, server_version="0.1.0", skills=[skill])


class FakeAggregator:
    """The slice of MCPAggregator the installer and the sync use."""

    def __init__(self, *servers: FakeServer, fail_listing: bool = False):
        self.servers = {s.name: s for s in servers}
        self.fail_listing = fail_listing

    async def list_mcp_skill_registries(self):
        if self.fail_listing:
            raise RuntimeError("server down")
        return [s.registry() for s in self.servers.values()]

    async def get_skill(self, uri: str, server_name: str):
        from fast_agent.mcp.skills_extension import GetSkillResult

        return GetSkillResult(skill=self.servers[server_name]._entry())

    async def get_resource(self, resource_uri: str, *, server_name=None, cache_mode="use"):
        from mcp_types import ReadResourceResult, TextResourceContents

        server = self.servers[server_name]
        server.reads.append(resource_uri)
        relative = resource_uri[len(server.root) + 1 :]
        return ReadResourceResult(
            contents=[TextResourceContents(uri=resource_uri, mimeType="text/markdown", text=server.files[relative])]
        )


class FakeAgent:
    def __init__(self, aggregator):
        self.name = "body"
        self.aggregator = aggregator
        self.skill_manifests: list = []


class FakeBrain:
    def __init__(self, aggregator):
        self.agent = FakeAgent(aggregator)

    def resolve_agent(self, _name):
        return self.agent


@pytest.fixture
def relisted(monkeypatch):
    """Record relists instead of rebuilding a real agent's instruction."""
    calls: list = []

    async def fake_relist(agent):
        calls.append(agent)
        agent.skill_manifests = skills.load_manifests()

    monkeypatch.setattr(skills, "relist", fake_relist)
    return calls


def _guide_server(body: str = "Read rules.md first.", rules: str = "d100 under skill.") -> FakeServer:
    return FakeServer(
        "guide-server",
        "guide",
        {"SKILL.md": _skill_md("guide", "How to run the game", body), "references/rules.md": rules},
    )


async def test_first_sync_installs_every_file_and_lists_the_skill(dirs, relisted):
    server = _guide_server()
    brain = FakeBrain(FakeAggregator(server))
    assert await skills.sync_from_servers(brain) == "1 installed"
    installed = dirs.installed / "guide-server" / "guide"
    assert (installed / "references" / "rules.md").read_text(encoding="utf-8") == "d100 under skill."
    assert (installed / "SKILL.md").exists()
    # Provenance sidecar, so a later sync knows this copy is the server's.
    sidecar = json.loads((installed / ".skill-source.json").read_text(encoding="utf-8"))
    assert sidecar["mcp_server_name"] == "guide-server"
    assert len(relisted) == 1
    assert [m.name for m in brain.agent.skill_manifests] == ["guide"]


async def test_second_sync_reads_nothing_and_does_not_relist(dirs, relisted):
    server = _guide_server()
    brain = FakeBrain(FakeAggregator(server))
    await skills.sync_from_servers(brain)
    server.reads.clear()
    assert await skills.sync_from_servers(brain) == "1 current"
    assert server.reads == []
    assert len(relisted) == 1


async def test_a_changed_server_copy_replaces_ours(dirs, relisted):
    server = _guide_server()
    brain = FakeBrain(FakeAggregator(server))
    await skills.sync_from_servers(brain)
    server.files["references/rules.md"] = "d100 under skill; 01 is a critical."
    assert await skills.sync_from_servers(brain) == "1 updated"
    rules = dirs.installed / "guide-server" / "guide" / "references" / "rules.md"
    assert rules.read_text(encoding="utf-8").endswith("critical.")
    assert len(relisted) == 2


async def test_a_local_edit_is_never_overwritten(dirs, relisted, caplog):
    server = _guide_server()
    brain = FakeBrain(FakeAggregator(server))
    await skills.sync_from_servers(brain)
    rules = dirs.installed / "guide-server" / "guide" / "references" / "rules.md"
    rules.write_text("house rule: 96 fumbles", encoding="utf-8")
    server.files["references/rules.md"] = "server changed too"
    assert await skills.sync_from_servers(brain) == "1 kept"
    assert rules.read_text(encoding="utf-8") == "house rule: 96 fumbles"
    assert "edited locally" in caplog.text


async def test_a_hand_written_skill_of_the_same_name_is_left_alone(dirs, relisted):
    _write_skill(dirs.installed / "guide-server" / "guide", "guide", "Mine", body="My own guide.")
    brain = FakeBrain(FakeAggregator(_guide_server()))
    assert await skills.sync_from_servers(brain) == "1 kept"
    assert "My own guide." in (dirs.installed / "guide-server" / "guide" / "SKILL.md").read_text(encoding="utf-8")
    # Nothing new on disk, but the brain started without manifests: list what is there.
    assert len(relisted) == 1


async def test_a_file_that_fails_verification_installs_nothing(dirs, relisted):
    server = _guide_server()
    aggregator = FakeAggregator(server)
    listed = server.registry()

    async def stale_listing():
        return [listed]

    aggregator.list_mcp_skill_registries = stale_listing  # type: ignore[method-assign]
    server.files["references/rules.md"] = "changed after the listing"
    brain = FakeBrain(aggregator)
    assert await skills.sync_from_servers(brain) == "1 failed"
    assert not (dirs.installed / "guide-server" / "guide").exists()
    assert brain.agent.skill_manifests == []


async def test_a_server_that_cannot_be_listed_costs_one_line(dirs, relisted, caplog):
    brain = FakeBrain(FakeAggregator(fail_listing=True))
    assert await skills.sync_from_servers(brain) == "unavailable"
    assert "server down" in caplog.text
    assert relisted == []


async def test_servers_without_skills_say_so(dirs, relisted):
    brain = FakeBrain(FakeAggregator())
    assert await skills.sync_from_servers(brain) == "no server offers skills"


async def test_relist_gives_the_agent_its_manifests_and_a_reader(dirs):
    """The real rebuild: manifests set, instruction rebuilt, read_skill available."""
    from fast_agent.core.instruction_refresh import McpInstructionCapable

    _write_skill(dirs.installed / "guide-server" / "greet", "greet", "Say hello")

    class Agent:
        """Just enough of McpAgent for rebuild_agent_instruction."""

        name = "body"
        aggregator = None
        skill_registry = None
        skill_read_tool_name = "read_skill"
        instruction_template = compose_instruction("Persona.")
        instruction_context: dict = {}

        def __init__(self):
            self.skill_manifests: list = []
            self.instruction = ""

        def set_skill_manifests(self, manifests):
            self.skill_manifests = list(manifests)

        def set_instruction_context(self, context):
            self.instruction_context = dict(context)

        def set_instruction(self, text):
            self.instruction = text

    agent = Agent()
    assert isinstance(agent, McpInstructionCapable)
    await skills.relist(agent)
    assert [m.name for m in agent.skill_manifests] == ["greet"]
    assert "<name>greet</name>" in agent.instruction
    assert SKILLS_PLACEHOLDER not in agent.instruction
