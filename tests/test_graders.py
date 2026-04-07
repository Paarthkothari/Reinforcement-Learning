from __future__ import annotations

import inspect
import unittest

from env.tasks import (
    grade_easy_submission,
    grade_easy_task,
    grade_hard_submission,
    grade_hard_task,
    grade_medium_submission,
    grade_medium_task,
)


class GraderBoundsTests(unittest.TestCase):
    def test_easy_grader_bounds(self) -> None:
        score = grade_easy_submission({"vendor_name": "wrong"}, {"vendor_name": "right", "currency": "INR"})
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_medium_grader_bounds(self) -> None:
        score = grade_medium_submission(
            flagged_issues=["invalid_invoice_date", "false_positive"],
            ground_truth=["invalid_invoice_date", "missing_gstin"],
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

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
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

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
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_perfect_scores_are_kept_strictly_below_one(self) -> None:
        easy_score = grade_easy_submission(
            {"vendor_name": "Acme", "currency": "INR"},
            {"vendor_name": "Acme", "currency": "INR"},
        )
        medium_score = grade_medium_submission(
            flagged_issues=["invalid_invoice_date"],
            ground_truth=["invalid_invoice_date"],
        )
        hard_score = grade_hard_submission(
            predicted_matches={"INV-1": "PO-1"},
            predicted_unmatched=["INV-2"],
            predicted_discrepancies=["INV-3"],
            predicted_duplicates=["INV-4"],
            ground_truth_matches={"INV-1": "PO-1"},
            ground_truth_unmatched=["INV-2"],
            ground_truth_discrepancies={"INV-3": "10.00"},
            ground_truth_duplicates=["INV-4"],
        )

        self.assertLess(easy_score, 1.0)
        self.assertLess(medium_score, 1.0)
        self.assertLess(hard_score, 1.0)

    def test_empty_scores_are_kept_strictly_above_zero(self) -> None:
        easy_score = grade_easy_submission({}, {"vendor_name": "Acme"})
        medium_score = grade_medium_submission(flagged_issues=[], ground_truth=["invalid_invoice_date"])
        hard_score = grade_hard_submission(
            predicted_matches={},
            predicted_unmatched=[],
            predicted_discrepancies=[],
            predicted_duplicates=[],
            ground_truth_matches={"INV-1": "PO-1"},
            ground_truth_unmatched=["INV-2"],
            ground_truth_discrepancies={"INV-3": "10.00"},
            ground_truth_duplicates=["INV-4"],
        )

        self.assertGreater(easy_score, 0.0)
        self.assertGreater(medium_score, 0.0)
        self.assertGreater(hard_score, 0.0)

    def test_manifest_graders_use_standard_two_argument_signature(self) -> None:
        self.assertEqual(len(inspect.signature(grade_easy_task).parameters), 2)
        self.assertEqual(len(inspect.signature(grade_medium_task).parameters), 2)
        self.assertEqual(len(inspect.signature(grade_hard_task).parameters), 2)

    def test_manifest_graders_accept_structured_payloads(self) -> None:
        easy_score = grade_easy_task(
            {"current_submission": {"vendor_name": "Acme", "currency": "INR"}},
            {"ground_truth": {"vendor_name": "Acme", "currency": "INR"}},
        )
        medium_score = grade_medium_task(
            {"flagged_issues": ["invalid_invoice_date"]},
            {"issues": ["invalid_invoice_date"]},
        )
        hard_score = grade_hard_task(
            {
                "current_matches": {"INV-1": "PO-1"},
                "current_unmatched_invoices": ["INV-2"],
                "current_flagged_discrepancies": ["INV-3"],
                "current_flagged_duplicate_invoices": ["INV-4"],
            },
            {
                "matches": {"INV-1": "PO-1"},
                "unmatched_invoices": ["INV-2"],
                "discrepancies": {"INV-3": "10.00"},
                "duplicate_invoices": ["INV-4"],
            },
        )

        self.assertGreater(easy_score, 0.0)
        self.assertLess(easy_score, 1.0)
        self.assertGreater(medium_score, 0.0)
        self.assertLess(medium_score, 1.0)
        self.assertGreater(hard_score, 0.0)
        self.assertLess(hard_score, 1.0)


if __name__ == "__main__":
    unittest.main()
