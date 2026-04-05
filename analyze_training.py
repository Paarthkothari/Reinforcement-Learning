from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LOG_PATH = Path("training_log.jsonl")


def moving_average(values: list[float], window: int) -> list[float]:
    averaged: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        window_values = values[start : index + 1]
        averaged.append(round(sum(window_values) / len(window_values), 4))
    return averaged


def analyze(window: int = 20) -> None:
    if not LOG_PATH.exists():
        print(f"No training log found at {LOG_PATH}. Run train_loop.py first.")
        return

    episodes_by_difficulty: dict[str, list[dict]] = defaultdict(list)
    with LOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line.strip())
            episodes_by_difficulty[payload["difficulty"]].append(payload)

    print(f"\n{'=' * 64}")
    print(f"Training Analysis (window={window})")
    print(f"{'=' * 64}")
    for difficulty, episodes in sorted(episodes_by_difficulty.items()):
        scores = [episode["final_score"] for episode in episodes]
        smoothed = moving_average(scores, window)
        thresholds = {0.5: None, 0.7: None, 0.9: None}
        for index, value in enumerate(smoothed, start=1):
            for threshold in list(thresholds):
                if thresholds[threshold] is None and value >= threshold:
                    thresholds[threshold] = index

        print(f"\n[{difficulty.upper()}] episodes={len(episodes)}")
        print(f"first={scores[0]:.3f} last={scores[-1]:.3f} best={max(scores):.3f} worst={min(scores):.3f}")
        print(f"avg_last_{min(window, len(scores))}={sum(scores[-window:]) / min(window, len(scores)):.3f}")
        for threshold, episode_number in thresholds.items():
            status = episode_number if episode_number is not None else "not reached"
            print(f"avg>={threshold:.1f}: {status}")

        print("curve:")
        for offset, value in enumerate(smoothed[::10], start=1):
            filled = int(value * 30)
            bar = ("#" * filled) + ("." * (30 - filled))
            print(f"ep {(offset - 1) * 10 + 1:4d} {bar} {value:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze FinanceOps OpenEnv training logs")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()
    analyze(args.window)
