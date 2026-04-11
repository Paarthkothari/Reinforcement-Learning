from __future__ import annotations

import re
import sys


START_RE = re.compile(r"^\[START\] task=(?P<task>\S+) env=(?P<env>\S+) model=(?P<model>.+)$")
STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) action=(?P<action>.+) reward=(?P<reward>\d+\.\d{2}) "
    r"done=(?P<done>true|false) error=(?P<error>.*)$"
)
END_RE = re.compile(
    r"^\[END\] success=(?P<success>true|false) steps=(?P<steps>\d+) rewards=(?P<rewards>(\d+\.\d{2})(,\d+\.\d{2})*|)$"
)


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def validate(lines: list[str]) -> None:
    if not lines:
        fail("expected evaluator output, but stdin was empty")

    index = 0
    episode_count = 0

    while index < len(lines):
        start_line = lines[index]
        if not START_RE.fullmatch(start_line):
            fail(f"line {index + 1} is not a valid [START] line: {start_line}")
        index += 1
        episode_count += 1

        step_count = 0
        while index < len(lines) and lines[index].startswith("[STEP]"):
            step_line = lines[index]
            match = STEP_RE.fullmatch(step_line)
            if not match:
                fail(f"line {index + 1} is not a valid [STEP] line: {step_line}")

            expected_step = step_count + 1
            actual_step = int(match.group("step"))
            if actual_step != expected_step:
                fail(
                    f"line {index + 1} has step={actual_step}, expected step={expected_step}"
                )

            step_count += 1
            index += 1

        if index >= len(lines):
            fail("missing [END] line at end of episode")

        end_line = lines[index]
        match = END_RE.fullmatch(end_line)
        if not match:
            fail(f"line {index + 1} is not a valid [END] line: {end_line}")

        declared_steps = int(match.group("steps"))
        if declared_steps != step_count:
            fail(
                f"line {index + 1} declares steps={declared_steps}, but counted {step_count} [STEP] lines"
            )

        rewards_field = match.group("rewards")
        rewards = rewards_field.split(",") if rewards_field else []
        if len(rewards) != step_count:
            fail(
                f"line {index + 1} contains {len(rewards)} rewards, expected {step_count}"
            )

        index += 1

    print(f"OK: validated {episode_count} episode(s)")


def main() -> None:
    validate(sys.stdin.read().splitlines())


if __name__ == "__main__":
    main()
