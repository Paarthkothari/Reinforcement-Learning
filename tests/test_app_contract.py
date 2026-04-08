from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app import app


class AppContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_metadata_exposes_all_three_tasks_with_graders(self) -> None:
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        tasks = payload.get("tasks", [])
        self.assertEqual(len(tasks), 3)
        self.assertEqual(
            [task["id"] for task in tasks],
            [
                "invoice_extract_easy",
                "invoice_validate_medium",
                "po_reconcile_hard",
            ],
        )
        self.assertEqual(
            [task["grader"] for task in tasks],
            [
                "env.tasks.easy:grade_easy_task",
                "env.tasks.medium:grade_medium_task",
                "env.tasks.hard:grade_hard_task",
            ],
        )

    def test_schema_exposes_task_manifest(self) -> None:
        response = self.client.get("/schema")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        tasks = payload.get("tasks", [])
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            low, high = task["reward_range"]
            self.assertGreater(low, 0.0)
            self.assertLess(high, 1.0)


if __name__ == "__main__":
    unittest.main()
