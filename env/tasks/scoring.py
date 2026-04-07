from __future__ import annotations

SCORE_FLOOR = 0.0001
SCORE_CEILING = 0.9999


def clamp_open_unit_interval(score: float) -> float:
    if score <= SCORE_FLOOR:
        return SCORE_FLOOR
    if score >= SCORE_CEILING:
        return SCORE_CEILING
    return round(score, 4)
