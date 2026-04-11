from __future__ import annotations

import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

from env import Action
from inference import (
    _safe_reward,
    build_client,
    load_api_key,
    load_hf_token,
    resolve_baseline_mode,
    resolve_task_names,
    run_episode,
)


class InferenceConfigTests(unittest.TestCase):
    def test_build_client_uses_injected_proxy_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://proxy.example/v1",
                "HF_TOKEN": "hf-token",
            },
            clear=False,
        ):
            with patch("inference.OpenAI") as mock_openai:
                build_client()

        mock_openai.assert_called_once_with(
            api_key="hf-token",
            base_url="https://proxy.example/v1",
        )

    def test_load_hf_token_reads_hf_token(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "hf-token"}, clear=True):
            self.assertEqual(load_hf_token(), "hf-token")
            self.assertEqual(load_api_key(), "hf-token")

    def test_resolve_baseline_mode_defaults_to_model(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_baseline_mode(), "model")

    def test_build_client_uses_default_api_base_url(self) -> None:
        with patch.dict(os.environ, {"HF_TOKEN": "hf-token"}, clear=True):
            with patch("inference.OpenAI") as mock_openai:
                build_client()

        mock_openai.assert_called_once_with(
            api_key="hf-token",
            base_url="https://router.huggingface.co/v1",
        )

    def test_build_client_requires_hf_token(self) -> None:
        with patch.dict(os.environ, {"API_BASE_URL": "https://proxy.example/v1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                build_client()

    def test_build_client_does_not_accept_api_key_only(self) -> None:
        with patch.dict(
            os.environ,
            {
                "API_BASE_URL": "https://proxy.example/v1",
                "API_KEY": "legacy-token",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
                build_client()

    def test_resolve_task_names_defaults_to_all_three_tasks(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_task_names(),
                [
                    "invoice_extract_easy",
                    "invoice_validate_medium",
                    "po_reconcile_hard",
                ],
            )

    def test_safe_reward_clamps_to_two_decimal_safe_interval(self) -> None:
        self.assertEqual(_safe_reward(0.0), 0.01)
        self.assertEqual(_safe_reward(1.0), 0.99)
        self.assertEqual(_safe_reward(0.5), 0.5)

    def test_run_episode_logs_safe_fallback_reward_when_exception_happens_before_first_step(self) -> None:
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        with patch("inference.choose_action", side_effect=RuntimeError("boom")):
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                run_episode(
                    task_name="invoice_extract_easy",
                    benchmark="finance-ops-openenv",
                    success_score_threshold=0.1,
                    model_name="unit-test-model",
                    client=None,
                )

        end_lines = [
            line
            for line in stdout_buffer.getvalue().splitlines()
            if line.startswith("[END]")
        ]
        self.assertEqual(len(end_lines), 1)
        reward_text = end_lines[0].split("rewards=", 1)[1]
        rewards = [float(value) for value in reward_text.split(",") if value]

        self.assertEqual(len(rewards), 1)
        self.assertGreater(rewards[0], 0.0)
        self.assertLess(rewards[0], 1.0)
        self.assertNotIn(rewards[0], {0.0, 1.0})

    def test_run_episode_uses_raw_score_snapshot_for_success_threshold(self) -> None:
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()

        with patch("inference.choose_action", return_value=Action(action_type="submit")):
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                run_episode(
                    task_name="invoice_extract_easy",
                    benchmark="finance-ops-openenv",
                    success_score_threshold=0.1,
                    model_name="unit-test-model",
                    client=None,
                )

        end_lines = [
            line
            for line in stdout_buffer.getvalue().splitlines()
            if line.startswith("[END]")
        ]
        self.assertEqual(len(end_lines), 1)
        self.assertIn("success=false", end_lines[0])


if __name__ == "__main__":
    unittest.main()
