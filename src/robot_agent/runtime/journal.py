"""JSONL journal for durable, inspectable agent traces."""

from __future__ import annotations

import json
from pathlib import Path

from .events import RuntimeEvent


class RunJournal:
    def __init__(self, path: Path, run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = self._load_last_sequence()

    def _load_last_sequence(self) -> int:
        if not self.path.exists():
            return 0
        last_sequence = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            last_sequence = max(last_sequence, int(event.get("sequence", 0)))
        return last_sequence

    def append(self, event: RuntimeEvent) -> RuntimeEvent:
        self._sequence += 1
        event = RuntimeEvent(
            run_id=self.run_id,
            type=event.type,
            payload=event.payload,
            category=event.category,
            sequence=self._sequence,
            timestamp=event.timestamp,
        )
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event
