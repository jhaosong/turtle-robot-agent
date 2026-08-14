from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from robot_agent.models import (
    get_azure_deployment_name,
    get_azure_v1_base_url,
    get_env_variable,
)


class ModelConfigurationTest(unittest.TestCase):
    def test_azure_v1_url_is_normalized(self):
        with patch.dict(
            os.environ,
            {"AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/"},
            clear=True,
        ):
            self.assertEqual(
                get_azure_v1_base_url(),
                "https://example.openai.azure.com/openai/v1/",
            )

    def test_azure_deployment_accepts_shared_model_fallback(self):
        with patch.dict(os.environ, {"OPENAI_MODEL": "robot-model"}, clear=True):
            self.assertEqual(get_azure_deployment_name(), "robot-model")

    def test_required_environment_variable_rejects_empty_value(self):
        with patch.dict(os.environ, {"TOKEN": ""}, clear=True):
            with self.assertRaisesRegex(ValueError, "TOKEN"):
                get_env_variable("TOKEN")


if __name__ == "__main__":
    unittest.main()
