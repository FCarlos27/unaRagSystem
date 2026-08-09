import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger("unanswered_log")

_lock = threading.Lock()


def log_unanswered(
    path: str,
    query: str,
    session_id: str | None,
    sources: list[str] | None = None,
    reason: str = "no_context",
) -> None:
    if not path:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "query": query,
        "sources": sources or [],
        "reason": reason,
    }
    try:
        with _lock:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.error("Failed to write unanswered log %s: %s", path, exc)


def ensure_log_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
