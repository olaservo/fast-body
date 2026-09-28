"""Cards a robot holds that are not part of the app: uploads and packs.

Two ways a card reaches `~/.fast-body/personalities/` without a shell, both
driven from the settings page:

- **upload**: one card file from the browser, saved at the top of that
  directory and listed there with a Remove button;
- **pack**: a Hugging Face repository of cards, downloaded with the robot's own
  HF login into `packs/<owner>--<repo>/`, so a private repo works the way a
  private Space does for an MCP server. Packs are recorded in `packs.yaml`
  beside them with the revision installed, and can be updated or removed.

The loader (personality.py) searches the directory and each pack below it.

Downloads block on the network, so the routes run them in a thread.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from fast_body import personality

logger = logging.getLogger(__name__)

PACKS_SUBDIR = "packs"
PACKS_FILE = "packs.yaml"
MAX_CARD_BYTES = 256_000
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# Only files with a card suffix come down (a README.md among them, which the
# loader's frontmatter check keeps out of the list); images and scripts stay on the Hub.
_CARD_PATTERNS = [f"*{suffix}" for suffix in personality.CARD_SUFFIXES]
# In this order when the page does not say: a pack of cards is a dataset in
# Hub terms, and the other two are where someone may have put one anyway.
_REPO_TYPES = ("dataset", "model", "space")


def user_dir() -> Path:
    """Read at call time, so tests can point the loader elsewhere."""
    return personality.USER_PERSONALITIES_DIR


def packs_dir() -> Path:
    return user_dir() / PACKS_SUBDIR


def pack_dir(repo_id: str) -> Path:
    return packs_dir() / repo_id.replace("/", "--")


# ── uploaded cards ────────────────────────────────────────────────────────
def list_user_cards() -> list[dict[str, Any]]:
    """Cards at the top of the user directory (uploads), by name."""
    root = user_dir()
    if not root.is_dir():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if personality.is_card_file(path):
            out.append({"name": path.stem, "file": path.name, "shadows_builtin": _is_builtin(path.stem)})
    return out


def _is_builtin(name: str) -> bool:
    return personality._card_path(name, personality.PERSONALITIES_DIR) is not None


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
    """Remove an uploaded card. False if there is none by that name."""
    path = personality._card_path(name, user_dir())
    if path is None:
        return False
    try:
        path.unlink()
    except OSError as e:
        logger.warning("could not delete %s: %s", path, e)
        return False
    return True


def _first_line(e: BaseException) -> str:
    return str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__


# ── packs ─────────────────────────────────────────────────────────────────
def _read_packs() -> dict[str, dict[str, Any]]:
    path = packs_dir() / PACKS_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (OSError, yaml.YAMLError) as e:
        logger.warning("could not read %s: %s", path, e)
        return {}
    packs = data.get("packs") if isinstance(data, dict) else None
    return {k: v for k, v in packs.items() if isinstance(v, dict)} if isinstance(packs, dict) else {}


def _write_packs(packs: dict[str, dict[str, Any]]) -> None:
    root = packs_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / PACKS_FILE).write_text(yaml.safe_dump({"packs": packs}, sort_keys=True), encoding="utf-8")


def list_packs() -> list[dict[str, Any]]:
    """Installed packs as the page shows them, with the cards each provides."""
    out = []
    for repo_id, entry in sorted(_read_packs().items()):
        directory = pack_dir(repo_id)
        cards = sorted(p.stem for p in directory.iterdir() if personality.is_card_file(p)) if directory.is_dir() else []
        out.append(
            {
                "repo_id": repo_id,
                "repo_type": entry.get("repo_type", "dataset"),
                "revision": str(entry.get("sha", ""))[:7],
                "installed": entry.get("installed"),
                "cards": cards,
            }
        )
    return out


def install_pack(repo_id: str, repo_type: str | None = None) -> str:
    """Download (or refresh) a pack with the robot's HF login. Returns a note for the page.

    Raises ValueError with a message the page can show when the repo cannot be
    found or holds no cards.
    """
    repo_id = (repo_id or "").strip().strip("/")
    if not _REPO_RE.match(repo_id):
        raise ValueError("a pack is a Hugging Face repo, written owner/name")
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    api = HfApi()
    types = (repo_type,) if repo_type else _REPO_TYPES
    info = None
    for candidate in types:
        try:
            info = api.repo_info(repo_id, repo_type=candidate)
            repo_type = candidate
            break
        except RepositoryNotFoundError:
            continue
        except HfHubHTTPError as e:
            raise ValueError(f"Hugging Face said: {_first_line(e)}") from e
    if info is None:
        raise ValueError(
            f"no repo called {repo_id} that this robot can read (a private one needs the robot's HF login to have access)"
        )
    target = pack_dir(repo_id)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id, repo_type=repo_type, local_dir=str(target), allow_patterns=_CARD_PATTERNS)
    cards = sorted(p.stem for p in target.iterdir() if personality.is_card_file(p))
    if not cards:
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError(f"{repo_id} holds no card files at its top level")
    packs = _read_packs()
    previous = packs.get(repo_id, {}).get("sha")
    packs[repo_id] = {
        "repo_type": repo_type,
        "sha": getattr(info, "sha", None),
        "installed": time.strftime("%Y-%m-%d %H:%M"),
    }
    _write_packs(packs)
    sha = str(getattr(info, "sha", "") or "")[:7]
    what = ", ".join(cards)
    if previous and previous == getattr(info, "sha", None):
        return f"{repo_id} is current ({sha}): {what}"
    logger.info("personalities: pack %s at %s: %s", repo_id, sha, what)
    return f"{'updated' if previous else 'installed'} {repo_id} ({sha}): {what}"


def remove_pack(repo_id: str) -> bool:
    """Delete a pack's directory and its record. False if it was not installed."""
    packs = _read_packs()
    if repo_id not in packs:
        return False
    shutil.rmtree(pack_dir(repo_id), ignore_errors=True)
    del packs[repo_id]
    _write_packs(packs)
    return True
