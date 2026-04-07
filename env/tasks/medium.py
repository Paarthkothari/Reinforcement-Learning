from __future__ import annotations

from typing import Any, Iterable, Set

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


def grade_medium_task(agent_output: Any, ground_truth: Any) -> float:
    predicted = agent_output
    if isinstance(agent_output, dict):
        predicted = (
            agent_output.get("flagged_issues")
            or agent_output.get("issues")
            or agent_output.get("predicted_issues")
            or []
        )

    expected = ground_truth
    if isinstance(ground_truth, dict):
        expected = (
            ground_truth.get("issues")
            or ground_truth.get("ground_truth")
            or []
        )

    return grade_medium_submission(predicted or [], expected or [])
