from __future__ import annotations

from typing import Any, Dict

from .scoring import clamp_open_unit_interval


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def grade_easy_submission(agent_output: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    fields = list(ground_truth.keys())
    if not fields:
        return clamp_open_unit_interval(0.0)

    correct = sum(
        1
        for field in fields
        if _normalize(agent_output.get(field)) == _normalize(ground_truth.get(field))
    )
    return clamp_open_unit_interval(correct / len(fields))


def grade_easy_task(agent_output: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    predicted_fields = agent_output
    if isinstance(agent_output, dict):
        predicted_fields = (
            agent_output.get("current_submission")
            or agent_output.get("extracted_fields")
            or agent_output.get("submission")
            or agent_output
        )

    expected_fields = ground_truth
    if isinstance(ground_truth, dict):
        expected_fields = ground_truth.get("ground_truth") or ground_truth

    return grade_easy_submission(
        dict(predicted_fields or {}),
        dict(expected_fields or {}),
    )
