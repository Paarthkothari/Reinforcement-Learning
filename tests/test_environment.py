from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO

from env import Action, FinanceOpsEnv
from env.data import generate_task
from inference import log_end


class FinanceOpsEnvironmentTests(unittest.TestCase):
    def test_reset_returns_clean_state(self) -> None:
        env = FinanceOpsEnv()
        env.reset("easy")
        env.step(
            Action(
                action_type="extract",
                field_name="vendor_name",
                field_value="Sunrise Office Supplies Pvt Ltd",
            )
        )

        observation = env.reset("medium")
        state = env.state()

        self.assertEqual(observation.task_id, "invoice_validate_medium")
        self.assertEqual(state["step_count"], 0)
        self.assertFalse(state["done"])
        self.assertEqual(state["extracted_fields"], {})
        self.assertEqual(state["flagged_issues"], [])
        self.assertEqual(state["matches"], {})
        self.assertEqual(state["action_history"], [])
        self.assertIsNone(state["last_action_error"])

    def test_step_updates_state_for_easy_extraction(self) -> None:
        env = FinanceOpsEnv()
        env.reset("easy")
        expected_vendor_name = env.current_task["ground_truth"]["vendor_name"]

        _, reward, done, _ = env.step(
            Action(
                action_type="extract",
                field_name="vendor_name",
                field_value=expected_vendor_name,
            )
        )
        state = env.state()

        self.assertFalse(done)
        self.assertGreater(reward.score, 0.0)
        self.assertEqual(state["step_count"], 1)
        self.assertEqual(state["extracted_fields"]["vendor_name"], expected_vendor_name)
        self.assertEqual(state["last_action_error"], None)

    def test_invalid_medium_flag_is_penalized(self) -> None:
        env = FinanceOpsEnv()
        env.reset("medium")

        _, reward, done, info = env.step(Action(action_type="flag"))

        self.assertFalse(done)
        self.assertEqual(reward.score, 0.01)
        self.assertEqual(info["last_action_error"], "missing_issue_code")

    def test_generate_task_is_deterministic_for_same_episode_index(self) -> None:
        first = generate_task("hard", episode_index=2)
        second = generate_task("hard", episode_index=2)

        self.assertEqual(first["task_id"], second["task_id"])
        self.assertEqual(first["variant_id"], second["variant_id"])
        self.assertEqual(first["ground_truth"], second["ground_truth"])
        self.assertEqual(first["document"], second["document"])

    def test_generate_task_changes_across_episode_index(self) -> None:
        first = generate_task("easy", episode_index=0)
        second = generate_task("easy", episode_index=1)

        self.assertNotEqual(first["variant_id"], second["variant_id"])
        self.assertNotEqual(first["document"], second["document"])

    def test_hard_task_applies_late_step_penalty_after_turn_six(self) -> None:
        env = FinanceOpsEnv()
        env.reset("hard")
        first_invoice_id, first_po_id = next(iter(env.current_task["ground_truth"]["matches"].items()))

        _, reward, done, _ = env.step(
            Action(action_type="match", invoice_id=first_invoice_id, po_id=first_po_id)
        )
        self.assertFalse(done)
        self.assertAlmostEqual(reward.score, 0.16, places=4)

        env.reset("hard")
        first_invoice_id, first_po_id = next(iter(env.current_task["ground_truth"]["matches"].items()))

        for _ in range(6):
            _, reward, done, _ = env.step(Action(action_type="skip"))
            self.assertFalse(done)
            self.assertAlmostEqual(reward.score, 0.01, places=4)

        _, reward, done, _ = env.step(
            Action(action_type="match", invoice_id=first_invoice_id, po_id=first_po_id)
        )

        self.assertFalse(done)
        self.assertAlmostEqual(reward.score, 0.11, places=4)

    def test_log_end_matches_required_contract(self) -> None:
        buffer = StringIO()
        with redirect_stdout(buffer):
            log_end(success=True, steps=3, rewards=[0.2, 0.0, 1.0])

        self.assertEqual(
            buffer.getvalue().strip(),
            "[END] success=true steps=3 rewards=0.20,0.00,1.00",
        )

    def test_submit_returns_strict_task_score_bounds(self) -> None:
        env = FinanceOpsEnv()
        observation = env.reset("easy")
        observation, reward, done, _ = env.step(Action(action_type="submit"))

        self.assertTrue(done)
        self.assertGreater(reward.score, 0.0)
        self.assertLess(reward.score, 1.0)
        self.assertEqual(observation.task_id, "invoice_extract_easy")

    def test_invalid_step_rewards_also_stay_inside_open_interval(self) -> None:
        env = FinanceOpsEnv()
        env.reset("easy")
        _, reward, done, _ = env.step(
            Action(action_type="extract", field_name="vendor_name", field_value="wrong")
        )

        self.assertFalse(done)
        self.assertGreater(reward.score, 0.0)
        self.assertLess(reward.score, 1.0)


if __name__ == "__main__":
    unittest.main()
