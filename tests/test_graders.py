from __future__ import annotations

import unittest

from env.tasks import grade_easy_submission, grade_hard_submission, grade_medium_submission


class GraderBoundsTests(unittest.TestCase):
    def test_easy_grader_bounds(self) -> None:
        score = grade_easy_submission({"vendor_name": "wrong"}, {"vendor_name": "right", "currency": "INR"})
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_medium_grader_bounds(self) -> None:
        score = grade_medium_submission(
            flagged_issues=["invalid_invoice_date", "false_positive"],
            ground_truth=["invalid_invoice_date", "missing_gstin"],
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hard_grader_bounds(self) -> None:
        score = grade_hard_submission(
            predicted_matches={"INV-1": "PO-1"},
            predicted_unmatched=["INV-2", "INV-3"],
            predicted_discrepancies=["INV-1"],
            predicted_duplicates=["INV-4"],
            ground_truth_matches={"INV-1": "PO-1", "INV-5": "PO-5"},
            ground_truth_unmatched=["INV-2"],
            ground_truth_discrepancies={"INV-1": "10.00"},
            ground_truth_duplicates=["INV-4"],
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_hard_grader_accepts_rich_discrepancy_payloads(self) -> None:
        score = grade_hard_submission(
            predicted_matches={"INV-1": "PO-1"},
            predicted_unmatched=["INV-2"],
            predicted_discrepancies=["INV-3", "INV-4"],
            predicted_duplicates=["INV-5"],
            ground_truth_matches={"INV-1": "PO-1"},
            ground_truth_unmatched=["INV-2"],
            ground_truth_discrepancies={
                "INV-3": {"issue_code": "amount_mismatch", "delta": "75.00"},
                "INV-4": {"issue_code": "split_po", "po_references": ["PO-7", "PO-8"]},
            },
            ground_truth_duplicates=["INV-5"],
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
