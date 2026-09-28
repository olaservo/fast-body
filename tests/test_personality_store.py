"""Cards that reach the robot from the settings page: uploads and Hub packs."""

from __future__ import annotations

import types

import pytest
import yaml
from starlette.testclient import TestClient

from fast_body import personality, personality_store
from fast_body.personality import list_personalities, load_personality
from fast_body.web.hub import WebChatHub
from fast_body.web.server import WebChatServer

CARD = "---\ntype: agent\nvariables:\n  voice: onyx\n---\nYou keep an inn.\n"


@pytest.fixture
def user_dir(tmp_path, monkeypatch):
    path = tmp_path / "personalities"
    monkeypatch.setattr(personality, "USER_PERSONALITIES_DIR", path)
    monkeypatch.delenv("FAST_BODY_PERSONALITIES_DIR", raising=False)
    return path


# ── uploads ───────────────────────────────────────────────────────────────
def test_an_uploaded_card_is_stored_and_listed(user_dir):
    assert personality_store.save_card("innkeeper.md", CARD) is None
    assert (user_dir / "innkeeper.md").read_text(encoding="utf-8") == CARD
    assert personality_store.list_user_cards() == [
        {"name": "innkeeper", "file": "innkeeper.md", "shadows_builtin": False}
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


# ── packs ─────────────────────────────────────────────────────────────────
class FakeHub:
    """Stands in for huggingface_hub: one dataset repo holding two cards and a README."""

    def __init__(self, repos: dict[tuple[str, str], dict[str, str]], sha: str = "abc1234def"):
        self.repos = repos
        self.sha = sha
        self.downloads: list[tuple[str, str]] = []

    def repo_info(self, repo_id, repo_type=None, **_):
        from huggingface_hub.errors import RepositoryNotFoundError

        if (repo_id, repo_type) not in self.repos:
            import httpx

            response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/api"))
            raise RepositoryNotFoundError(f"404 {repo_id}", response=response)
        return types.SimpleNamespace(sha=self.sha)

    def snapshot_download(self, repo_id, repo_type=None, local_dir=None, allow_patterns=None, **_):
        import fnmatch
        from pathlib import Path

        self.downloads.append((repo_id, repo_type))
        for name, text in self.repos[(repo_id, repo_type)].items():
            if any(fnmatch.fnmatch(name, pat) for pat in allow_patterns or ["*"]):
                (Path(local_dir) / name).write_text(text, encoding="utf-8")
        return local_dir


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub

    fake = FakeHub(
        {
            ("ola/cards", "dataset"): {"README.md": "# cards\n", "bard.md": CARD, "sage.yaml": "type: agent\ninstruction: Be wise.\n"},
            ("ola/model-cards", "model"): {"gm.md": CARD},
            ("ola/empty", "dataset"): {"README.md": "# nothing\n"},
        }
    )
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    monkeypatch.setattr(huggingface_hub.HfApi, "repo_info", lambda self, *a, **k: fake.repo_info(*a, **k))
    return fake


def test_a_pack_installs_its_cards_and_is_recorded(user_dir, hub):
    note = personality_store.install_pack("ola/cards")
    assert note.startswith("installed ola/cards (abc1234)") and "bard, sage" in note
    pack = user_dir / "packs" / "ola--cards"
    assert (pack / "bard.md").is_file() and (pack / "sage.yaml").is_file()
    assert (pack / "README.md").exists()  # has the suffix, comes down, is not a card
    recorded = yaml.safe_load((user_dir / "packs" / "packs.yaml").read_text())["packs"]["ola/cards"]
    assert (recorded["repo_type"], recorded["sha"]) == ("dataset", "abc1234def")
    assert {"bard", "sage"} <= set(list_personalities())
    assert load_personality("bard").voice == "onyx"
    (listed,) = personality_store.list_packs()
    assert (listed["repo_id"], listed["cards"], listed["revision"]) == ("ola/cards", ["bard", "sage"], "abc1234")


def test_repo_type_is_found_by_trying_in_order(user_dir, hub):
    personality_store.install_pack("ola/model-cards")
    assert hub.downloads == [("ola/model-cards", "model")]


def test_a_pack_with_no_cards_is_refused_and_not_kept(user_dir, hub):
    with pytest.raises(ValueError, match="no card files"):
        personality_store.install_pack("ola/empty")
    assert not (user_dir / "packs" / "ola--empty").exists()
    assert personality_store.list_packs() == []


def test_unknown_repo_and_bad_ids_are_refused(user_dir, hub):
    with pytest.raises(ValueError, match="no repo called"):
        personality_store.install_pack("ola/nowhere")
    with pytest.raises(ValueError, match="owner/name"):
        personality_store.install_pack("not a repo")
    with pytest.raises(ValueError, match="owner/name"):
        personality_store.install_pack("../../etc")


def test_installing_again_reports_current_or_updated(user_dir, hub):
    personality_store.install_pack("ola/cards")
    assert personality_store.install_pack("ola/cards").startswith("ola/cards is current")
    hub.sha = "def5678abc"
    assert personality_store.install_pack("ola/cards").startswith("updated ola/cards (def5678)")


def test_remove_pack_drops_directory_and_record(user_dir, hub):
    personality_store.install_pack("ola/cards")
    assert personality_store.remove_pack("ola/cards")
    assert not (user_dir / "packs" / "ola--cards").exists()
    assert "bard" not in list_personalities()
    assert not personality_store.remove_pack("ola/cards")


def test_a_readme_in_a_pack_is_not_a_personality(user_dir):
    pack = user_dir / "packs" / "x--y"
    pack.mkdir(parents=True)
    (pack / "README.md").write_text("# not a card\n", encoding="utf-8")
    (pack / "bard.md").write_text(CARD, encoding="utf-8")
    names = list_personalities()
    assert "bard" in names and "README" not in names


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


def test_routes_install_and_remove_a_pack(client, user_dir, hub):
    r = _post(client, "/personalities/packs/install", {"repo_id": "ola/cards"})
    assert r.status_code == 200 and r.json()["packs"][0]["cards"] == ["bard", "sage"]
    bad = _post(client, "/personalities/packs/install", {"repo_id": "ola/nowhere"})
    assert bad.status_code == 400 and "no repo called" in bad.json()["error"]
    assert _post(client, "/personalities/packs/remove", {"repo_id": "ola/cards"}).json()["packs"] == []
    assert _post(client, "/personalities/packs/remove", {"repo_id": "ola/cards"}).status_code == 404
