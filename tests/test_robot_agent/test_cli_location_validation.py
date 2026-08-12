from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from robot_agent.cli import validate_location_file
from robot_agent.config import RobotAgentSettings


class CliLocationValidationTest(unittest.TestCase):
    def test_missing_location_file_has_actionable_error(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = RobotAgentSettings(
                location_file=root / "missing.yaml",
                run_directory=root / "runs",
                trace=False,
            )

            with self.assertRaisesRegex(FileNotFoundError, "ROBOT_AGENT_LOCATION_FILE"):
                validate_location_file(settings)


if __name__ == "__main__":
    unittest.main()
