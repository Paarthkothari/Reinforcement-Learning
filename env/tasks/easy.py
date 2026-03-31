from __future__ import annotations

from typing import Dict


def _normalize(value: str | None) -> str:
    return (value or "").strip().lower()


def grade_easy_submission(agent_output: Dict[str, str], ground_truth: Dict[str, str]) -> float:
    fields = list(ground_truth.keys())
    if not fields:
        return 0.0

    correct = sum(
        1
        for field in fields
        if _normalize(agent_output.get(field)) == _normalize(ground_truth.get(field))
    )
    return round(correct / len(fields), 4)
