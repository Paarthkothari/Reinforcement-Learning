from __future__ import annotations

from typing import Dict

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
