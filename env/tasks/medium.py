from __future__ import annotations

from typing import Iterable, Set

from .scoring import clamp_open_unit_interval


def grade_medium_submission(flagged_issues: Iterable[str], ground_truth: Iterable[str]) -> float:
    predicted: Set[str] = set(flagged_issues)
    actual: Set[str] = set(ground_truth)
    if not actual:
        return clamp_open_unit_interval(0.0)

    true_positive = len(predicted & actual)
    false_positive = len(predicted - actual)
    score = (true_positive / len(actual)) - (0.1 * false_positive)
    return clamp_open_unit_interval(max(0.0, min(1.0, score)))
