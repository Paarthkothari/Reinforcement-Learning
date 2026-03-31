from __future__ import annotations

from typing import Dict, Iterable


def grade_hard_submission(
    predicted_matches: Dict[str, str],
    predicted_unmatched: Iterable[str],
    predicted_discrepancies: Iterable[str],
    ground_truth_matches: Dict[str, str],
    ground_truth_unmatched: Iterable[str],
    ground_truth_discrepancies: Dict[str, str],
) -> float:
    match_total = len(ground_truth_matches)
    matched_correct = sum(
        1
        for invoice_id, po_id in ground_truth_matches.items()
        if predicted_matches.get(invoice_id) == po_id
    )
    match_score = matched_correct / match_total if match_total else 0.0

    predicted_unmatched_set = set(predicted_unmatched)
    ground_truth_unmatched_set = set(ground_truth_unmatched)
    if ground_truth_unmatched_set:
        unmatched_score = len(predicted_unmatched_set & ground_truth_unmatched_set) / len(
            ground_truth_unmatched_set
        )
    else:
        unmatched_score = 1.0

    false_unmatched = len(predicted_unmatched_set - ground_truth_unmatched_set)
    unmatched_score = max(0.0, unmatched_score - (0.1 * false_unmatched))

    predicted_discrepancy_set = set(predicted_discrepancies)
    ground_truth_discrepancy_set = set(ground_truth_discrepancies.keys())
    if ground_truth_discrepancy_set:
        discrepancy_score = len(predicted_discrepancy_set & ground_truth_discrepancy_set) / len(
            ground_truth_discrepancy_set
        )
    else:
        discrepancy_score = 1.0

    false_discrepancies = len(predicted_discrepancy_set - ground_truth_discrepancy_set)
    discrepancy_score = max(0.0, discrepancy_score - (0.1 * false_discrepancies))

    weighted_score = (0.6 * match_score) + (0.2 * unmatched_score) + (0.2 * discrepancy_score)
    return round(max(0.0, min(1.0, weighted_score)), 4)
