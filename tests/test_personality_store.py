"""Cards that reach the robot from the settings page: uploads and card packs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from fast_body import personality, personality_store
from fast_body.personality import list_personalities, load_personality
from fast_body.web.hub import WebChatHub
from fast_body.web.server import WebChatServer

CARD = "---\ntype: agent\nvariables:\n  voice: onyx\n---\nYou keep an inn.\n"


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    """`agent-cards/` under a scratch fast-agent home."""
    path = tmp_path / "home" / "agent-cards"
    monkeypatch.setattr(personality, "USER_PERSONALITIES_DIR", path)
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    return path


# ── uploads ───────────────────────────────────────────────────────────────
def test_an_uploaded_card_is_stored_and_listed(user_dir):
    assert personality_store.save_card("innkeeper.md", CARD) is None
    assert (user_dir / "innkeeper.md").read_text(encoding="utf-8") == CARD
    assert personality_store.list_user_cards() == [
        {"name": "innkeeper", "file": "innkeeper.md", "shadows_builtin": False, "pack": None}
    ]
    assert "innkeeper" in list_personalities()
    assert load_personality("innkeeper").voice == "onyx"


def test_upload_rejects_what_is_not_a_card(user_dir):
    assert "file" in (personality_store.save_card("notes.txt", "hi") or "")
    assert "name" in (personality_store.save_card("../evil.md", CARD) or "")
    assert "usable card" in (personality_store.save_card("broken.md", "no frontmatter here\n") or "")
    assert "KB" in (personality_store.save_card("huge.md", CARD + "x" * 300_000) or "")
    # nothing half-written is left behind
    assert not user_dir.is_dir() or not [p for p in user_dir.iterdir() if not p.name.startswith(".")]


def test_a_rejected_card_leaves_no_staging_file(user_dir):
    personality_store.save_card("broken.md", "no frontmatter here\n")
    assert not list(user_dir.glob(".*"))


def test_an_upload_named_like_a_builtin_shadows_it(user_dir):
    assert personality_store.save_card("marvin.md", CARD) is None
    assert personality_store.list_user_cards()[0]["shadows_builtin"] is True
    assert load_personality("marvin").voice == "onyx"
    assert personality_store.delete_card("marvin")
    assert load_personality("marvin").instruction.startswith("You are a robot of staggering intelligence")


def test_delete_only_touches_uploads(user_dir):
    assert not personality_store.delete_card("marvin")  # packaged, not ours to delete
    assert not personality_store.delete_card("nobody")


# ── card packs ────────────────────────────────────────────────────────────
def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env).stdout


def _pack(repo: Path, name: str, cards: dict[str, str], manifest_extra: str = "", readme: str | None = None) -> None:
    root = repo / "packs" / name
    root.mkdir(parents=True)
    for file, text in cards.items():
        (root / file).write_text(text, encoding="utf-8")
    listed = "\n".join(f"    - {f}" for f in cards if personality.is_card_file(root / f))
    (root / "card-pack.yaml").write_text(
        f"schema_version: 1\nname: {name}\nkind: card\ninstall:\n  agent_cards:\n{listed}\n{manifest_extra}",
        encoding="utf-8",
    )
    if readme:
        (root / "README.md").write_text(readme, encoding="utf-8")


@pytest.fixture
def registry(tmp_path):
    """A git repo of card packs with a marketplace.json, the shape fast-agent reads.

    Returned as a `file:` URL, which is what the store makes of a local path.
    """
    repo = tmp_path / "cards-repo"
    repo.mkdir()
    _pack(repo, "bard", {"bard.md": CARD}, readme="# bard\n\nSings.\n")
    _pack(repo, "sage", {"sage.yaml": "type: agent\ninstruction: Be wise.\n"})
    _pack(
        repo,
        "sneaky",
        {"gm.md": CARD, "mcp_servers.yaml": "servers: {}\n"},
        manifest_extra="  files:\n    - mcp_servers.yaml\n",
    )
    entries = [
        {
            "name": n,
            "description": d,
            "kind": "card",
            "repo_url": str(repo),
            "repo_ref": "main",
            "repo_path": f"packs/{n}",
        }
        for n, d in (("bard", "a singer"), ("sage", "a thinker"), ("sneaky", "brings a file"))
    ]
    (repo / "marketplace.json").write_text(json.dumps({"entries": entries}), encoding="utf-8")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "packs")
    return repo


async def test_lookup_lists_what_a_registry_offers(user_dir, registry):
    found = await personality_store.lookup(str(registry))
    assert [p["name"] for p in found["packs"]] == ["bard", "sage", "sneaky"]
    assert found["packs"][0]["description"] == "a singer"
    assert found["source"].endswith("marketplace.json")  # fast-agent hands back the file it read


async def test_lookup_needs_a_registry(user_dir, tmp_path):
    with pytest.raises(ValueError, match="enter a"):
        await personality_store.lookup("")
    with pytest.raises(ValueError, match="no registry"):
        await personality_store.lookup(str(tmp_path / "nowhere"))


async def test_a_pack_installs_its_cards_through_fast_agent(user_dir, registry):
    done = await personality_store.install(str(registry), "bard")
    assert done["note"].startswith("installed bard (") and done["note"].endswith("): bard")
    assert done["readme"].startswith("# bard")
    assert (user_dir / "bard.md").read_text(encoding="utf-8") == CARD
    assert (user_dir.parent / "card-packs" / "bard" / "card-pack.yaml").is_file()
    assert "bard" in list_personalities() and load_personality("bard").voice == "onyx"
    (listed,) = personality_store.list_packs()
    assert (listed["name"], listed["cards"], len(listed["revision"])) == ("bard", ["bard"], 7)
    # the card is the pack's: not an upload to remove or overwrite one at a time
    assert personality_store.list_user_cards()[0]["pack"] == "bard"
    assert not personality_store.delete_card("bard")
    assert "belongs to the card pack bard" in (personality_store.save_card("bard.md", CARD) or "")


async def test_a_pack_that_brings_more_than_cards_is_refused(user_dir, registry):
    with pytest.raises(ValueError, match="more than cards"):
        await personality_store.install(str(registry), "sneaky")
    assert personality_store.list_packs() == []
    assert not (user_dir / "gm.md").exists()
    assert not (user_dir.parent / "mcp_servers.yaml").exists()


async def test_unknown_pack_is_refused(user_dir, registry):
    with pytest.raises(ValueError, match="no pack called"):
        await personality_store.install(str(registry), "nobody")


async def test_update_follows_the_source_and_keeps_local_edits(user_dir, registry):
    await personality_store.install(str(registry), "bard")
    assert (await personality_store.update("bard"))["status"] == "up_to_date"
    (registry / "packs" / "bard" / "bard.md").write_text(CARD.replace("onyx", "ash"), encoding="utf-8")
    _git(registry, "commit", "-qam", "new voice")
    done = await personality_store.update("bard")
    assert done["status"] == "updated" and load_personality("bard").voice == "ash"
    # an edit on the robot is not overwritten unless asked
    (user_dir / "bard.md").write_text(CARD.replace("onyx", "coral"), encoding="utf-8")
    (registry / "packs" / "bard" / "bard.md").write_text(CARD.replace("onyx", "sage"), encoding="utf-8")
    _git(registry, "commit", "-qam", "another")
    assert (await personality_store.update("bard"))["status"] == "skipped_dirty"
    assert load_personality("bard").voice == "coral"
    assert (await personality_store.update("bard", force=True))["status"] == "updated"
    assert load_personality("bard").voice == "sage"
    with pytest.raises(LookupError):
        await personality_store.update("nobody")


async def test_remove_drops_the_pack_and_its_cards(user_dir, registry):
    await personality_store.install(str(registry), "bard")
    assert personality_store.remove("bard")
    assert not (user_dir / "bard.md").exists()
    assert "bard" not in list_personalities()
    assert personality_store.list_packs() == []
    assert not personality_store.remove("bard")


async def test_a_hub_repo_id_is_read_with_the_robots_login(user_dir, registry, tmp_path, monkeypatch):
    """`owner/name` resolves through the Hub API, and its marketplace.json is fetched with the token."""
    import huggingface_hub
    from huggingface_hub.errors import RepositoryNotFoundError

    calls: list[tuple[str, str | None]] = []

    def repo_info(self, repo_id, repo_type=None, **_):
        calls.append((repo_id, repo_type))
        if (repo_id, repo_type) != ("ola/cards", "model"):
            import httpx

            raise RepositoryNotFoundError(
                "404", response=httpx.Response(404, request=httpx.Request("GET", "https://x"))
            )

    def download(repo_id, filename, repo_type=None, **_):
        assert (repo_id, filename, repo_type) == ("ola/cards", "marketplace.json", "model")
        return str(registry / "marketplace.json")

    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", repo_info)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    found = await personality_store.lookup("ola/cards")
    assert found["source"] == "https://huggingface.co/ola/cards"  # a model repo's git URL has no prefix
    assert calls == [("ola/cards", "dataset"), ("ola/cards", "model")]
    assert (await personality_store.install("ola/cards", "bard"))["note"].startswith("installed bard")
    # a dataset URL is taken as such
    assert personality_store._hf_repo("https://huggingface.co/datasets/ola/cards") == ("ola/cards", "dataset")
    assert personality_store._hf_repo("https://github.com/ola/cards") is None


def test_git_is_given_the_hub_login_through_the_environment(monkeypatch, tmp_path):
    import huggingface_hub

    monkeypatch.setattr(personality_store, "_git_env_done", False)
    for key in [k for k in os.environ if k.startswith("GIT_CONFIG") or k == "GIT_TERMINAL_PROMPT"]:
        monkeypatch.delenv(key)
    token_file = tmp_path / "token"
    monkeypatch.setattr(huggingface_hub.constants, "HF_TOKEN_PATH", str(token_file))
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_secret")
    personality_store.export_git_env()
    assert os.environ["GIT_TERMINAL_PROMPT"] == "0"
    assert os.environ["GIT_CONFIG_COUNT"] == "1"
    assert os.environ["GIT_CONFIG_KEY_0"] == "credential.https://huggingface.co.helper"
    helper = os.environ["GIT_CONFIG_VALUE_0"]
    assert token_file.as_posix() in helper and "hf_secret" not in helper
    # once per process
    personality_store.export_git_env()
    assert os.environ["GIT_CONFIG_COUNT"] == "1"


def test_no_token_means_no_helper(monkeypatch):
    monkeypatch.setattr(personality_store, "_git_env_done", False)
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    personality_store.export_git_env()
    assert "GIT_CONFIG_COUNT" not in os.environ


# ── the routes ────────────────────────────────────────────────────────────
@pytest.fixture
def client(user_dir):
    return TestClient(WebChatServer(WebChatHub(), host="127.0.0.1", port=8099, token="s3cret")._build_app())


def _post(client, path, body):
    return client.post(path + "?token=s3cret", json=body, headers={"origin": "http://testserver"})


def test_routes_upload_list_and_delete(client, user_dir):
    r = _post(client, "/personalities/upload", {"filename": "bard.md", "content": CARD})
    assert r.status_code == 200
    data = r.json()
    assert data["cards"][0]["name"] == "bard" and "bard" in data["available"] and "uploaded" in data["note"]
    assert client.get("/personalities?token=s3cret").json()["cards"][0]["name"] == "bard"
    assert _post(client, "/personalities/upload", {"filename": "x.txt", "content": "no"}).status_code == 400
    assert _post(client, "/personalities/delete", {"name": "bard"}).json()["cards"] == []
    assert _post(client, "/personalities/delete", {"name": "bard"}).status_code == 404


def test_routes_need_the_token(client):
    assert client.get("/personalities").status_code == 404
    assert client.post("/personalities/upload", json={}).status_code == 404
    assert client.post("/personalities/packs/lookup", json={}).status_code == 404


def test_routes_look_up_install_update_and_remove_a_pack(client, user_dir, registry):
    found = _post(client, "/personalities/packs/lookup", {"source": str(registry)})
    assert found.status_code == 200 and [p["name"] for p in found.json()["packs"]] == ["bard", "sage", "sneaky"]
    bad = _post(client, "/personalities/packs/lookup", {"source": str(registry.parent / "nowhere")})
    assert bad.status_code == 400 and "no registry" in bad.json()["error"]
    r = _post(client, "/personalities/packs/install", {"source": found.json()["source"], "name": "bard"})
    assert r.status_code == 200
    assert r.json()["packs"][0]["cards"] == ["bard"] and r.json()["readme"].startswith("# bard")
    assert _post(client, "/personalities/packs/install", {"source": str(registry), "name": "sneaky"}).status_code == 400
    up = _post(client, "/personalities/packs/update", {"name": "bard"})
    assert up.status_code == 200 and up.json()["status"] == "up_to_date" and up.json()["pack"] == "bard"
    assert _post(client, "/personalities/packs/update", {"name": "nobody"}).status_code == 404
    assert _post(client, "/personalities/packs/remove", {"name": "bard"}).json()["packs"] == []
    assert _post(client, "/personalities/packs/remove", {"name": "bard"}).status_code == 404
