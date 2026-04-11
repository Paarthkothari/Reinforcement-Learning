from __future__ import annotations

SCORE_FLOOR = 0.01
SCORE_CEILING = 0.99
MANIFEST_SCORE_FLOOR = 0.1
MANIFEST_SCORE_CEILING = 0.9


def clamp_open_unit_interval(score: float) -> float:
    if score <= SCORE_FLOOR:
        return SCORE_FLOOR
    if score >= SCORE_CEILING:
        return SCORE_CEILING
    return round(score, 4)


def normalize_manifest_task_score(score: float) -> float:
    """
    Remap manifest grader outputs into a one-decimal-safe open interval.

    Phase 2 validation appears to coerce task scores more aggressively than the
    raw environment reward path. Keeping manifest graders inside 0.1..0.9
    preserves ordering while staying comfortably away from 0.0 and 1.0.
    """
    bounded_score = clamp_open_unit_interval(score)
    normalized_score = (bounded_score - SCORE_FLOOR) / (SCORE_CEILING - SCORE_FLOOR)
    remapped_score = MANIFEST_SCORE_FLOOR + (
        normalized_score * (MANIFEST_SCORE_CEILING - MANIFEST_SCORE_FLOOR)
    )
    return round(remapped_score, 4)
