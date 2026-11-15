"""Thread-safe JSON session history and append-only JSONL LGPD audit persistence."""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SESSION_ID_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]")


class SessionStore:
    """File-backed conversation memory and LGPD audit logger."""

    def __init__(self, sessions_dir: Path, audit_log_path: Path) -> None:
        self.sessions_dir = Path(sessions_dir).resolve()
        self.audit_log_path = Path(audit_log_path).resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def sanitize_session_id(session_id: str) -> str:
        """Strip unsafe characters to prevent path traversal."""
        cleaned = _SESSION_ID_SANITIZER.sub("", session_id or "")
        return cleaned[:64] or "default"

    def _session_file(self, session_id: str) -> Path:
        safe_id = self.sanitize_session_id(session_id)
        candidate = (self.sessions_dir / f"{safe_id}.json").resolve()
        if candidate.parent != self.sessions_dir:
            raise ValueError(f"Unsafe session path resolved for session_id={session_id!r}")
        return candidate

    def get_full_history(self, session_id: str) -> list[dict[str, str]]:
        """Read all stored turns for a session in chronological order."""
        path = self._session_file(session_id)
        with self._lock:
            if not path.exists():
                return []
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return list(data) if isinstance(data, list) else []

    def get_recent_history(self, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        """Return up to the last `limit` conversation turns in chronological order."""
        if limit <= 0:
            return []
        history = self.get_full_history(session_id)
        return history[-limit:]

    def append_turns(self, session_id: str, human_msg: str, ai_msg: str) -> None:
        """Atomically append human and AI turns for the given session."""
        path = self._session_file(session_id)
        now_iso = datetime.now(UTC).isoformat()
        with self._lock:
            history: list[dict[str, str]] = []
            if path.exists():
                with path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                    if isinstance(loaded, list):
                        history = loaded
            history.extend(
                [
                    {"role": "human", "content": human_msg, "timestamp": now_iso},
                    {"role": "ai", "content": ai_msg, "timestamp": now_iso},
                ]
            )
            tmp_path = path.with_suffix(".json.tmp")
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(history, fh, ensure_ascii=False, indent=2)
            tmp_path.replace(path)

    def append_audit_log(self, entry: dict[str, Any]) -> None:
        """Append one anonymized JSON line to rag_audit_logs.jsonl."""
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            **entry,
        }
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock, self.audit_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
