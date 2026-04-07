from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from inference import build_client, load_api_key, resolve_baseline_mode


class InferenceConfigTests(unittest.TestCase):
    def test_build_client_uses_injected_proxy_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://proxy.example/v1",
                "API_KEY": "proxy-key",
            },
            clear=False,
        ):
            with patch("inference.OpenAI") as mock_openai:
                build_client()

        mock_openai.assert_called_once_with(
            api_key="proxy-key",
            base_url="https://proxy.example/v1",
        )

    def test_load_api_key_prefers_api_key_over_hf_token(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_KEY": "proxy-key",
                "HF_TOKEN": "legacy-token",
            },
            clear=False,
        ):
            self.assertEqual(load_api_key(), "proxy-key")

    def test_resolve_baseline_mode_defaults_to_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_baseline_mode(), "model")

    def test_build_client_requires_api_base_url(self) -> None:
        with patch.dict(os.environ, {"API_KEY": "proxy-key"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "API_BASE_URL"):
                build_client()

    def test_build_client_requires_api_key(self) -> None:
        with patch.dict(os.environ, {"API_BASE_URL": "https://proxy.example/v1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "API_KEY"):
                build_client()


if __name__ == "__main__":
    unittest.main()
