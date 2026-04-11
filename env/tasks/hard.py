from __future__ import annotations

from typing import Any, Dict, Iterable

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


def _first_present(payload: Dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    content = payload.get("content")
    if isinstance(content, dict):
        for key in keys:
            if key in content and content[key] is not None:
                return content[key]
    nested_ground_truth = payload.get("ground_truth")
    if isinstance(nested_ground_truth, dict):
        for key in keys:
            if key in nested_ground_truth and nested_ground_truth[key] is not None:
                return nested_ground_truth[key]
    return default


def grade_hard_task(agent_output: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    predicted_matches = _first_present(
        agent_output,
        "predicted_matches",
        "matches",
        "current_matches",
        default={},
    )
    predicted_unmatched = _first_present(
        agent_output,
        "predicted_unmatched",
        "unmatched_invoices",
        "current_unmatched_invoices",
        default=[],
    )
    predicted_discrepancies = set(
        _first_present(
            agent_output,
            "predicted_discrepancies",
            "flagged_discrepancies",
            "current_flagged_discrepancies",
            default=[],
        )
    )
    predicted_discrepancies.update(
        _first_present(
            agent_output,
            "predicted_split_flags",
            "flagged_split_invoices",
            "current_flagged_split_invoices",
            default=[],
        )
    )
    predicted_duplicates = _first_present(
        agent_output,
        "predicted_duplicates",
        "duplicate_invoices",
        "flagged_duplicate_invoices",
        "current_flagged_duplicate_invoices",
        default=[],
    )

    expected_matches = _first_present(ground_truth, "matches", default={})
    expected_unmatched = _first_present(
        ground_truth,
        "unmatched_invoices",
        default=[],
    )
    expected_discrepancies = _first_present(
        ground_truth,
        "discrepancies",
        default={},
    )
    expected_duplicates = _first_present(
        ground_truth,
        "duplicate_invoices",
        default=[],
    )

    return grade_hard_submission(
        predicted_matches=dict(predicted_matches or {}),
        predicted_unmatched=list(predicted_unmatched or []),
        predicted_discrepancies=list(predicted_discrepancies),
        predicted_duplicates=list(predicted_duplicates or []),
        ground_truth_matches=dict(expected_matches or {}),
        ground_truth_unmatched=list(expected_unmatched or []),
        ground_truth_discrepancies=dict(expected_discrepancies or {}),
        ground_truth_duplicates=list(expected_duplicates or []),
    )
