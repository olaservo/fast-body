"""Cards a robot holds that are not part of the app: uploads and card packs.

Both live in fast-agent's home, which fast-body pins to `~/.fast-body`
(config.py), so they survive an app update or remove:

- **upload**: one card file from the browser, saved in `agent-cards/` and
  listed with a Remove button;
- **card pack**: fast-agent's own packaging of cards — a `card-pack.yaml` in a
  git repo, listed by a `marketplace.json` registry. Installing goes through
  `fast_agent.cards.manager`, which copies the pack's cards into
  `agent-cards/`, keeps the source and revision in `card-packs/<name>/`, and
  does update (skipping a card edited by hand unless forced) and remove. The
  same repo installs on a laptop with `fast-agent cards add`.

fast-body adds two things on top of the manager. A Hugging Face repo id
(`owner/name`) is accepted as a registry: its `marketplace.json` is read with
the robot's own Hub login, where fast-agent fetches registries anonymously.
And git is handed that login for huggingface.co through the environment, so a
private repo clones without a token on the command line or in the pack's
record on disk.

A pack may only bring cards. fast-agent lets a manifest install tool cards and
arbitrary files under the home; a pack that asks for either is removed again
and refused, since a robot has no one watching what lands beside
`mcp_servers.yaml`.

The loader (personality.py) reads `agent-cards/` like any other directory.
Network calls block, so the routes run them in a thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fast_body import personality

if TYPE_CHECKING:
    from fast_agent.cards.manager import MarketplaceCardPack
    from fast_agent.paths import HomePaths

logger = logging.getLogger(__name__)

MAX_CARD_BYTES = 256_000
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_HF_URL_RE = re.compile(r"^https?://huggingface\.co/(?:(datasets|spaces)/)?([^/\s]+/[^/\s]+?)(?:\.git)?/?$")
# In this order when the id does not say: a repo of cards is a dataset in Hub
# terms, and the other two are where someone may have put one anyway.
_REPO_TYPES = ("dataset", "model", "space")
_HF_GIT_PREFIX = {"dataset": "datasets/", "space": "spaces/", "model": ""}
NETWORK_TIMEOUT = 180.0


def user_dir() -> Path:
    """Read at call time, so tests can point the loader elsewhere."""
    return personality.USER_PERSONALITIES_DIR


def home_paths() -> HomePaths:
    """fast-agent's view of the home. `agent-cards/` sits directly under it."""
    from fast_agent.paths import resolve_home_paths

    return resolve_home_paths(override=user_dir().parent)


# ── uploaded cards ────────────────────────────────────────────────────────
def list_user_cards() -> list[dict[str, Any]]:
    """Cards in `agent-cards/`, by name, saying which pack put each one there."""
    root = user_dir()
    if not root.is_dir():
        return []
    owners = _pack_owned()
    out = []
    for path in sorted(root.iterdir()):
        if personality.is_card_file(path):
            out.append(
                {
                    "name": path.stem,
                    "file": path.name,
                    "shadows_builtin": _is_builtin(path.stem),
                    "pack": owners.get(path.name),
                }
            )
    return out


def _is_builtin(name: str) -> bool:
    return personality._card_path(name, personality.PERSONALITIES_DIR) is not None


def _pack_owned() -> dict[str, str]:
    """Card file name → the installed pack that owns it, from fast-agent's records."""
    from fast_agent.cards import manager

    owned: dict[str, str] = {}
    try:
        packs = manager.list_local_card_packs(home_paths=home_paths())
    except OSError:
        return owned
    for pack in packs:
        if pack.source is None:
            continue
        for relative in pack.source.installed_files:
            parent, _, name = relative.rpartition("/")
            if parent == user_dir().name:
                owned[name] = pack.name
    return owned


def save_card(filename: str, content: str) -> str | None:
    """Store an uploaded card. Returns a problem to show the page, or None."""
    name = (filename or "").strip()
    suffix = Path(name).suffix.lower()
    stem = Path(name).stem
    if suffix not in personality.CARD_SUFFIXES:
        return f"a card is a {', '.join(personality.CARD_SUFFIXES)} file"
    if name != Path(name).name or not _NAME_RE.match(stem):
        return "the file name becomes the personality's name: letters, digits, . _ - only"
    if len(content.encode("utf-8")) > MAX_CARD_BYTES:
        return f"the card is over {MAX_CARD_BYTES // 1000} KB"
    if (owner := _pack_owned().get(f"{stem}{suffix}")) is not None:
        return f"{stem} belongs to the card pack {owner}; update or remove the pack instead"
    root = user_dir()
    try:
        root.mkdir(parents=True, exist_ok=True)
        staged = root / f".{stem}.upload{suffix}"
        staged.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"could not write to {root}: {e}"
    try:
        # The same parse the app does at startup: a card that fails here would
        # fall back to `default` with a warning nobody sees.
        personality._load_card(stem, staged)
    except Exception as e:
        staged.unlink(missing_ok=True)
        return f"not a usable card: {_first_line(e)}"
    final = root / f"{stem}{suffix}"
    staged.replace(final)
    logger.info("personalities: uploaded %s", final)
    return None


def delete_card(name: str) -> bool:
    """Remove an uploaded card. False if there is none by that name, or a pack owns it."""
    path = personality._card_path(name, user_dir())
    if path is None or path.name in _pack_owned():
        return False
    try:
        path.unlink()
    except OSError as e:
        logger.warning("could not delete %s: %s", path, e)
        return False
    return True


def _first_line(e: BaseException) -> str:
    return str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__


# ── git credentials ───────────────────────────────────────────────────────
_git_env_done = False


def export_git_env() -> None:
    """Let fast-agent's `git clone` reach a private Hugging Face repo.

    fast-agent runs git with this process's environment, so a credential helper
    passed through `GIT_CONFIG_*` reaches it without touching the robot's git
    config. The helper reads the Hub token file when git asks for huggingface.co,
    and nothing else; the token itself never enters the environment or a
    command line. `GIT_TERMINAL_PROMPT=0` makes a missing credential fail at
    once instead of waiting on a prompt no one will answer.
    """
    global _git_env_done
    if _git_env_done:
        return
    _git_env_done = True
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    from huggingface_hub import constants, get_token

    if not get_token():
        return
    token_path = Path(constants.HF_TOKEN_PATH).as_posix()
    helper = (
        "!f() { echo username=hf_user; "
        f'echo "password=$(cat \'{token_path}\' 2>/dev/null || printf %s "$HF_TOKEN")"; }}; f'
    )
    _append_git_config("credential.https://huggingface.co.helper", helper)


def _append_git_config(key: str, value: str) -> None:
    try:
        count = int(os.environ.get("GIT_CONFIG_COUNT", "0") or 0)
    except ValueError:
        count = 0
    os.environ[f"GIT_CONFIG_KEY_{count}"] = key
    os.environ[f"GIT_CONFIG_VALUE_{count}"] = value
    os.environ["GIT_CONFIG_COUNT"] = str(count + 1)


# ── card packs ────────────────────────────────────────────────────────────
def list_packs() -> list[dict[str, Any]]:
    """Installed packs as the page shows them, with the cards each provides."""
    from fast_agent.cards import manager

    out: list[dict[str, Any]] = []
    for pack in manager.list_local_card_packs(home_paths=home_paths()):
        if pack.source is None:
            out.append({"name": pack.name, "error": pack.metadata_error or "unreadable"})
            continue
        cards = sorted(Path(f).stem for f in pack.source.installed_files if f.startswith(user_dir().name + "/"))
        out.append(
            {
                "name": pack.name,
                "revision": pack.source.installed_revision[:7],
                "source": pack.source.repo_url,
                "installed": pack.source.installed_at[:16].replace("T", " "),
                "cards": cards,
            }
        )
    return out


async def lookup(source: str) -> dict[str, Any]:
    """The packs a registry offers: `{"source": ..., "packs": [{name, description, kind}]}`.

    Raises ValueError with a message the page can show.
    """
    normalized, packs = await _registry(source)
    return {
        "source": normalized,
        "packs": [{"name": p.name, "description": p.description or "", "kind": p.kind} for p in packs],
    }


async def install(source: str, name: str, force: bool = False) -> dict[str, Any]:
    """Install a pack from a registry. Returns `{"note": ..., "readme": ...}`.

    Raises ValueError with a message the page can show.
    """
    from fast_agent.cards import manager

    normalized, packs = await _registry(source)
    pack = manager.select_card_pack_by_name_or_index(packs, name)
    if pack is None:
        raise ValueError(f"{normalized} offers no pack called {name}")
    export_git_env()
    paths = home_paths()
    try:
        result = await manager.install_marketplace_card_pack(pack, home_paths=paths, force=force)
    except Exception as e:
        raise ValueError(f"could not install {name}: {_first_line(e)}") from e
    manifest = manager.load_card_pack_manifest(result.pack_dir)
    extra = [*manifest.tool_cards, *manifest.files]
    if extra:
        manager.remove_local_card_pack(result.pack_dir.name, home_paths=paths)
        raise ValueError(f"{name} wants to install more than cards ({', '.join(extra)}); fast-body only takes cards")
    cards = [Path(f).stem for f in result.installed_files if f.startswith(user_dir().name + "/")]
    broken = [c for c in cards if _unusable(c)]
    note = f"installed {name} ({result.source.installed_revision[:7]}): {', '.join(cards) or 'no cards'}"
    if broken:
        note += f"; not usable as a personality: {', '.join(broken)}"
    logger.info("personalities: %s", note)
    return {"note": note, "readme": manager.load_card_pack_readme(result.pack_dir)}


def _unusable(name: str) -> str | None:
    path = personality._card_path(name, user_dir())
    if path is None:
        return "missing"
    try:
        personality._load_card(name, path)
    except Exception as e:
        return _first_line(e)
    return None


async def update(name: str, force: bool = False) -> dict[str, Any]:
    """Bring one pack up to its source. Returns `{"status": ..., "note": ...}`.

    `status` is fast-agent's: `up_to_date`, `updated`, `skipped_dirty` (a card
    was edited by hand; `force` overwrites it), or an error status.
    """
    from fast_agent.cards import manager

    export_git_env()
    paths = home_paths()
    known = {p.name for p in manager.list_local_card_packs(home_paths=paths)}
    if name not in known:
        raise LookupError(name)
    try:
        checks = await asyncio.to_thread(manager.check_card_pack_updates, home_paths=paths)
    except Exception as e:
        raise ValueError(f"could not check {name}: {_first_line(e)}") from e
    mine = [u for u in checks if u.name == name]
    if not mine:
        raise LookupError(name)
    (info,) = mine
    if info.status != "update_available":
        return {"status": info.status, "note": f"{name}: {info.detail or info.status}"}
    try:
        (applied,) = await asyncio.to_thread(manager.apply_card_pack_updates, mine, home_paths=paths, force=force)
    except Exception as e:
        raise ValueError(f"could not update {name}: {_first_line(e)}") from e
    detail = applied.detail or applied.status
    if applied.status == "updated" and applied.available_revision:
        detail = f"{detail} to {applied.available_revision[:7]}"
    logger.info("personalities: pack %s %s", name, detail)
    return {"status": applied.status, "note": f"{name}: {detail}"}


def remove(name: str) -> bool:
    """Remove a pack and the cards it installed. False if it is not installed."""
    from fast_agent.cards import manager

    try:
        result = manager.remove_local_card_pack(name, home_paths=home_paths())
    except FileNotFoundError:
        return False
    logger.info("personalities: removed pack %s (%s)", name, ", ".join(result.removed_paths) or "no files")
    return True


# ── registries ────────────────────────────────────────────────────────────
async def _registry(source: str) -> tuple[str, list[MarketplaceCardPack]]:
    """Resolve what the page typed to the packs it offers."""
    from fast_agent.cards import manager

    source = (source or "").strip()
    if not source:
        raise ValueError("enter a Hugging Face repo (owner/name), a git repository URL, or a registry URL")
    hf = _hf_repo(source)
    if hf is not None:
        return await asyncio.to_thread(_hf_registry, *hf)
    if Path(source).exists():
        source = Path(source).resolve().as_uri()  # a drive letter reads as a URL scheme otherwise
    try:
        packs, normalized = await manager.fetch_marketplace_card_packs_with_source(source)
    except Exception as e:
        raise ValueError(f"no registry at {source}: {_first_line(e)}") from e
    if not packs:
        raise ValueError(f"{normalized} lists no card packs")
    return normalized, packs


def _hf_repo(source: str) -> tuple[str, str | None] | None:
    """(repo_id, repo_type or None) when the source names a Hub repo."""
    if _REPO_RE.match(source):
        return source, None
    m = _HF_URL_RE.match(source)
    if m is None:
        return None
    kind = m.group(1)
    return m.group(2), {"datasets": "dataset", "spaces": "space", None: "model"}[kind]


def _hf_registry(repo_id: str, repo_type: str | None) -> tuple[str, list[MarketplaceCardPack]]:
    """Read a Hub repo's `marketplace.json` with the robot's login."""
    from fast_agent.cards.manager import MarketplaceCardPack
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

    api = HfApi()
    found = None
    for candidate in (repo_type,) if repo_type else _REPO_TYPES:
        try:
            api.repo_info(repo_id, repo_type=candidate)
            found = candidate
            break
        except RepositoryNotFoundError:
            continue
        except HfHubHTTPError as e:
            raise ValueError(f"Hugging Face said: {_first_line(e)}") from e
    if found is None:
        raise ValueError(
            f"no repo called {repo_id} that this robot can read (a private one needs the robot's HF login to have access)"
        )
    git_url = f"https://huggingface.co/{_HF_GIT_PREFIX[found]}{repo_id}"
    try:
        path = hf_hub_download(repo_id, "marketplace.json", repo_type=found)
        entries = json.loads(Path(path).read_text(encoding="utf-8")).get("entries", [])
    except EntryNotFoundError as e:
        raise ValueError(f"{repo_id} has no marketplace.json listing its card packs") from e
    except (OSError, ValueError, AttributeError) as e:
        raise ValueError(f"could not read {repo_id}'s marketplace.json: {_first_line(e)}") from e
    packs = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("repo_path"):
            continue
        packs.append(
            MarketplaceCardPack(
                name=str(entry["name"]),
                description=str(entry.get("description") or "") or None,
                kind="bundle" if entry.get("kind") == "bundle" else "card",
                repo_url=str(entry.get("repo_url") or entry.get("repo") or git_url),
                repo_ref=str(entry.get("repo_ref") or "main"),
                repo_path=str(entry["repo_path"]),
                source_url=git_url,
            )
        )
    if not packs:
        raise ValueError(f"{repo_id}'s marketplace.json lists no card packs")
    return git_url, packs
