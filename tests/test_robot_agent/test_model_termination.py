from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from langchain_core.messages import AIMessage

from robot_agent.config import RobotAgentSettings
from robot_agent.middlewares.model_termination import ModelTerminationMiddleware
from robot_agent.runtime import RobotAgentRuntime


class ModelTerminationTest(unittest.TestCase):
    def _runtime(self, root: Path) -> RobotAgentRuntime:
        location_file = root / "locations.yaml"
        location_file.write_text("kitchen: [1.0, 2.0, 0.0]\n", encoding="utf-8")
        return RobotAgentRuntime(
            RobotAgentSettings(
                location_file=location_file,
                run_directory=root / "runs",
                trace=False,
            ),
            "test provider termination",
        )

    def test_length_finish_reason_marks_run_without_removing_content(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory))
            middleware = ModelTerminationMiddleware(runtime)
            message = AIMessage(content="partial but useful", response_metadata={"finish_reason": "length"})

            update = middleware.after_model({"messages": [message]}, runtime=None)

            self.assertIsNone(update)
            self.assertEqual(message.content, "partial but useful")
            self.assertEqual(runtime.state.model_stop_reason, "model_length_capped")

    def test_safety_finish_reason_strips_partial_tool_calls(self):
        with TemporaryDirectory() as temporary_directory:
            runtime = self._runtime(Path(temporary_directory))
            middleware = ModelTerminationMiddleware(runtime)
            message = AIMessage(
                content="",
                tool_calls=[
                    {"name": "navigate_to", "args": {"location": ""}, "id": "partial", "type": "tool_call"}
                ],
                additional_kwargs={"tool_calls": [{"function": {"name": "navigate_to", "arguments": "{\"location\":"}}]},
                response_metadata={"finish_reason": "content_filter"},
            )

            update = middleware.after_model({"messages": [message]}, runtime=None)
            cleaned = update["messages"][0]

            self.assertEqual(cleaned.tool_calls, [])
            self.assertNotIn("tool_calls", cleaned.additional_kwargs)
            self.assertNotIn("function_call", cleaned.additional_kwargs)
            self.assertEqual(runtime.state.model_stop_reason, "model_safety_capped")


if __name__ == "__main__":
    unittest.main()
