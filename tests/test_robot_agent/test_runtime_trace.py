from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from robot_agent.config import RobotAgentSettings
from robot_agent.runtime import RobotAgentRuntime


class RuntimeTraceTest(unittest.TestCase):
    def test_detector_samples_are_journaled_without_console_noise(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            location_file = root / "locations.yaml"
            location_file.write_text("location1: [0.0, 0.0, 0.0]\n", encoding="utf-8")
            settings = RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                trace=True,
            )

            with patch("builtins.print") as print_mock:
                runtime = RobotAgentRuntime(settings, "find an extinguisher")
                runtime.emit("detector_sampled", {"samples": 5}, category="perception")
                runtime.emit("visual_alignment_started", {"phase": "rotate"})

            printed = [call.args[0] for call in print_mock.call_args_list]
            self.assertFalse(any("detector_sampled" in line for line in printed))
            self.assertTrue(any("visual_alignment_started" in line for line in printed))

            events = [
                json.loads(line)
                for line in runtime.journal.path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(event["type"] == "detector_sampled" for event in events))


if __name__ == "__main__":
    unittest.main()
