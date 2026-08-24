"""Atomic JSON checkpoint persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any
from uuid import uuid4

from .serialization import sanitize_json_value


class JsonCheckpointStore:
    """Atomic JSON store safe for concurrent callers in one process."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def save(self, state: dict[str, Any]) -> None:
        temporary = self.path.with_name(
            f".{self.path.name}.{uuid4().hex}.tmp"
        )
        with self._lock:
            payload = sanitize_json_value(state)
            try:
                temporary.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=True),
                    encoding="utf-8",
                )
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)

    def load(self) -> dict[str, Any] | None:
        with self._lock:
            if not self.path.exists():
                return None
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Checkpoint must contain a JSON object")
            return payload
