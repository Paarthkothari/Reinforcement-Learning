from __future__ import annotations

from typing import Dict, Iterable

from .scoring import clamp_open_unit_interval


def grade_hard_submission(
    predicted_matches: Dict[str, str],
    predicted_unmatched: Iterable[str],
    predicted_discrepancies: Iterable[str],
    predicted_duplicates: Iterable[str],
    ground_truth_matches: Dict[str, str],
    ground_truth_unmatched: Iterable[str],
    ground_truth_discrepancies: Dict[str, str],
    ground_truth_duplicates: Iterable[str],
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

    predicted_duplicate_set = set(predicted_duplicates)
    ground_truth_duplicate_set = set(ground_truth_duplicates)
    if ground_truth_duplicate_set:
        duplicate_score = len(predicted_duplicate_set & ground_truth_duplicate_set) / len(
            ground_truth_duplicate_set
        )
    else:
        duplicate_score = 1.0

    false_duplicates = len(predicted_duplicate_set - ground_truth_duplicate_set)
    duplicate_score = max(0.0, duplicate_score - (0.1 * false_duplicates))

    weighted_score = (
        (0.5 * match_score)
        + (0.15 * unmatched_score)
        + (0.2 * discrepancy_score)
        + (0.15 * duplicate_score)
    )
    return clamp_open_unit_interval(max(0.0, min(1.0, weighted_score)))
