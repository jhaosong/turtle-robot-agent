"""Small JSON checkpoint store; later backends can replace this implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .serialization import sanitize_json_value


class JsonCheckpointStore:
    """Atomic JSON store assuming one writer per session.

    Concurrent callers require external locking; the current runtime invokes
    each goal sequentially in one process.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = sanitize_json_value(state)
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        os.replace(temporary, self.path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Checkpoint must contain a JSON object")
        return payload
