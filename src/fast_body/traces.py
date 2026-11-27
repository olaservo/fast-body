"""Trace upload: copy saved conversations to Hugging Face storage.

fast-agent keeps a rolling window of sessions under its home and can export
one as a trace. With `FAST_BODY_TRACE_URL` set, each start exports the saved
conversations that changed since their last upload and writes them under that
URL, one folder per personality card:

    <url>/<card>/<session>.json

The format is ATIF, because fast-agent's ATIF export restores the messages a
compaction moved out of the live history. The live conversation is skipped;
it goes up on a later start, once it is no longer being written.

Traces hold everything said to the robot and every tool result. The bucket or
dataset is created private when it does not exist, and nothing here makes one
public.

The upload uses `FAST_BODY_TRACE_TOKEN` when it is set, so the app's main
Hugging Face token can stay limited to inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fast_body import memory
from fast_body.config import MEMORY_DIR, Config

logger = logging.getLogger(__name__)

_KINDS = ("buckets", "datasets")
_NO_CARD = "unknown"


def parse_destination(url: str) -> tuple[str, str, str] | None:
    """Split `hf://buckets/owner/name[/prefix]` into (kind, repo id, root URL).

    None when the URL is not a bucket or dataset on the Hub.
    """
    parsed = urlparse(url.strip())
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.scheme != "hf" or parsed.netloc not in _KINDS or len(parts) < 2:
        return None
    return parsed.netloc, f"{parts[0]}/{parts[1]}", f"hf://{parsed.netloc}/" + "/".join(parts)


def _ledger_file() -> Path:
    return MEMORY_DIR / "trace_uploads.json"


def _ledger() -> dict[str, str]:
    """Session name -> the `last_activity` it had when it was uploaded."""
    try:
        data = json.loads(_ledger_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def _save_ledger(ledger: dict[str, str]) -> None:
    path = _ledger_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    tmp.replace(path)


def pending(manager: Any, ledger: dict[str, str], live: str | None) -> list[tuple[str, str]]:
    """(session name, last activity) for each conversation that needs uploading."""
    out: list[tuple[str, str]] = []
    for info in manager.list_sessions(include_empty=False):
        stamp = info.last_activity.isoformat()
        if info.name != live and ledger.get(info.name) != stamp:
            out.append((info.name, stamp))
    return out


def _ensure_private(kind: str, repo_id: str, token: str | None) -> None:
    """Create the destination private if it is missing; an existing one is left as it is.

    A token that may write to the destination but not create one fails here.
    That is not an error: the upload that follows says whether it exists.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        if kind == "buckets":
            api.create_bucket(repo_id, private=True, exist_ok=True)
        else:
            api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    except Exception as e:
        logger.debug("could not create %s %s: %s", kind, repo_id, e)


def _export(manager: Any, name: str, folder_url: str, token: str | None) -> int:
    """Export one session and upload it; returns the record count."""
    from fast_agent.session.trace_export_hf import HuggingFaceDatasetTraceUploader
    from fast_agent.session.trace_export_models import ExportRequest
    from fast_agent.session.trace_exporter import SessionTraceExporter
    from huggingface_hub import HfApi, HfFileSystem

    uploader = HuggingFaceDatasetTraceUploader(api=HfApi(token=token), filesystem=HfFileSystem(token=token))
    with tempfile.TemporaryDirectory(prefix="fast-body-trace-") as tmp:
        result = SessionTraceExporter(session_manager=manager, dataset_uploader=uploader).export(
            ExportRequest(
                target=name,
                agent_name=None,
                output_path=Path(tmp) / f"{name}.json",
                hf_url=folder_url,
                format="atif",
            )
        )
    return result.record_count


async def upload_pending(cfg: Config) -> str:
    """Upload the conversations that changed since their last upload; one status line."""
    if not cfg.trace_url:
        return "off"
    destination = parse_destination(cfg.trace_url)
    if destination is None:
        return "off (FAST_BODY_TRACE_URL must be hf://buckets/<owner>/<name> or hf://datasets/<owner>/<name>)"
    manager = memory.session_manager()
    if manager is None:
        return "off (no session manager)"
    kind, repo_id, root = destination
    ledger = _ledger()
    todo = pending(manager, ledger, memory.live_session_name())
    if not todo:
        return f"nothing new for {root}"
    cards = memory.session_cards()
    done = 0
    try:
        token = cfg.trace_token or None  # None = the token the rest of the app uses
        await asyncio.to_thread(_ensure_private, kind, repo_id, token)
        for name, stamp in todo:
            folder = f"{root}/{cards.get(name, _NO_CARD)}/"
            await asyncio.to_thread(_export, manager, name, folder, token)
            ledger[name] = stamp
            _save_ledger(ledger)
            done += 1
    except Exception as e:  # no token, no write access, no network: try again next start
        reason = " ".join(str(e).split())
        return f"uploaded {done} of {len(todo)} to {root}, then stopped: {reason}"
    return f"uploaded {done} to {root}"


async def upload_and_log(cfg: Config) -> None:
    """`upload_pending` as a background task: its result goes to the log."""
    try:
        logger.info("traces: %s", await upload_pending(cfg))
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("traces: upload failed: %s", e)
