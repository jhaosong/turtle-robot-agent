from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.runtime.checkpoint import JsonCheckpointStore
from robot_agent.runtime.events import RuntimeEvent
from robot_agent.runtime.journal import RunJournal
from robot_agent.runtime.serialization import sanitize_text


class UnicodeSerializationTest(unittest.TestCase):
    def test_sanitize_text_preserves_valid_unicode_and_replaces_surrogates(self):
        self.assertEqual(sanitize_text("寻找灭火器"), "寻找灭火器")
        self.assertEqual(sanitize_text("location\udce42"), "location\ufffd2")

    def test_journal_persists_malformed_terminal_text_as_valid_utf8_json(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "events.jsonl"
            journal = RunJournal(path, "run-1")

            journal.append(
                RuntimeEvent(
                    run_id="run-1",
                    type="run_started",
                    payload={"goal": "locatin\udce4 2", "nested": ["寻找"]},
                )
            )

            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["payload"]["goal"], "locatin\ufffd 2")
            self.assertEqual(saved["payload"]["nested"], ["寻找"])

    def test_checkpoint_persists_surrogate_free_state(self):
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "checkpoint.json"
            store = JsonCheckpointStore(path)

            store.save({"goal": "bad\udce4input"})

            self.assertEqual(store.load(), {"goal": "bad\ufffdinput"})


if __name__ == "__main__":
    unittest.main()
