from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .data import TASKS
from .models import Action, Observation, Reward
from .tasks import grade_easy_submission, grade_hard_submission, grade_medium_submission


class FinanceOpsEnv:
    def __init__(self) -> None:
        self.current_task_key: str = "easy"
        self.current_task: Dict[str, Any] = {}
        self.step_count: int = 0
        self.done: bool = False
        self.extracted_fields: Dict[str, str] = {}
        self.flagged_issues: set[str] = set()
        self.matches: Dict[str, str] = {}
        self.unmatched_invoices: set[str] = set()
        self.flagged_discrepancies: set[str] = set()
        self.action_history: list[Dict[str, Any]] = []
        self.last_reward: Optional[Reward] = None

    def reset(self, difficulty: Optional[str] = None) -> Observation:
        task_key = difficulty or self.current_task_key or "easy"
        if task_key not in TASKS:
            raise ValueError(f"Unknown difficulty '{task_key}'. Expected one of {list(TASKS)}")

        self.current_task_key = task_key
        self.current_task = deepcopy(TASKS[task_key])
        self.step_count = 0
        self.done = False
        self.extracted_fields = {}
        self.flagged_issues = set()
        self.matches = {}
        self.unmatched_invoices = set()
        self.flagged_discrepancies = set()
        self.action_history = []
        self.last_reward = None
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if not self.current_task:
            self.reset(self.current_task_key)

        if self.done:
            reward = Reward(score=-0.2, reason="Episode already finished. Reset before new actions.", partial_credit=0.0)
            return self._make_observation(), reward, True, {"error": "episode_done"}

        self.step_count += 1
        reward = self._apply_action(action)
        self.last_reward = reward
        self.action_history.append(action.model_dump())

        if self.step_count >= self.current_task["max_steps"] and not self.done:
            self.done = True
            reward = Reward(
                score=max(-1.0, reward.score - 0.2),
                reason=f"{reward.reason} Max steps reached before successful submission.",
                partial_credit=reward.partial_credit,
            )
            self.last_reward = reward

        return self._make_observation(), reward, self.done, {
            "task_id": self.current_task["task_id"],
            "difficulty": self.current_task["difficulty"],
            "step_count": self.step_count,
            "score_snapshot": self._current_score_snapshot(),
        }

    def state(self) -> Dict[str, Any]:
        return {
            "task_key": self.current_task_key,
            "task_id": self.current_task.get("task_id"),
            "difficulty": self.current_task.get("difficulty"),
            "step_count": self.step_count,
            "done": self.done,
            "extracted_fields": deepcopy(self.extracted_fields),
            "flagged_issues": sorted(self.flagged_issues),
            "matches": deepcopy(self.matches),
            "unmatched_invoices": sorted(self.unmatched_invoices),
            "flagged_discrepancies": sorted(self.flagged_discrepancies),
            "last_reward": self.last_reward.model_dump() if self.last_reward else None,
            "action_history": deepcopy(self.action_history),
        }

    def _make_observation(self) -> Observation:
        if not self.current_task:
            self.reset(self.current_task_key)

        task_type = self.current_task["type"]
        content: Dict[str, Any] = {
            "instructions": self.current_task["instructions"],
            "document": deepcopy(self.current_task["document"]),
        }
        if task_type == "extraction":
            content["current_submission"] = deepcopy(self.extracted_fields)
        elif task_type == "validation":
            content["flagged_issues"] = sorted(self.flagged_issues)
        elif task_type == "reconciliation":
            content["current_matches"] = deepcopy(self.matches)
            content["current_unmatched_invoices"] = sorted(self.unmatched_invoices)
            content["current_flagged_discrepancies"] = sorted(self.flagged_discrepancies)

        return Observation(
            task_id=self.current_task["task_id"],
            difficulty=self.current_task["difficulty"],
            content=content,
            available_actions=["extract", "flag", "match", "submit", "skip"],
            step_number=self.step_count,
            max_steps=self.current_task["max_steps"],
            context=f"Finance operations task: {task_type}",
        )

    def _apply_action(self, action: Action) -> Reward:
        action_type = action.action_type
        if action_type == "extract":
            return self._handle_extract(action)
        if action_type == "flag":
            return self._handle_flag(action)
        if action_type == "match":
            return self._handle_match(action)
        if action_type == "submit":
            return self._handle_submit()
        if action_type == "skip":
            return Reward(score=-0.03, reason="Skipped a turn.", partial_credit=0.0)
        return Reward(score=-0.1, reason="Invalid action.", partial_credit=0.0)

    def _handle_extract(self, action: Action) -> Reward:
        if self.current_task["type"] != "extraction":
            return Reward(score=-0.08, reason="Extract is not useful for this task.", partial_credit=0.0)

        if not action.field_name or action.field_value is None:
            return Reward(score=-0.1, reason="Extraction needs field_name and field_value.", partial_credit=0.0)

        ground_truth = self.current_task["ground_truth"]
        expected_value = str(ground_truth.get(action.field_name, ""))
        submitted_value = str(action.field_value)
        self.extracted_fields[action.field_name] = submitted_value

        if action.field_name not in ground_truth:
            return Reward(score=-0.05, reason=f"Field '{action.field_name}' is not required.", partial_credit=0.0)

        if submitted_value.strip().lower() == expected_value.strip().lower():
            return Reward(score=0.2, reason=f"Correct extraction for '{action.field_name}'.", partial_credit=0.2)

        if submitted_value.strip() and expected_value.strip() and submitted_value.strip().lower() in expected_value.strip().lower():
            return Reward(score=0.08, reason=f"Partial extraction for '{action.field_name}'.", partial_credit=0.08)

        return Reward(score=-0.04, reason=f"Incorrect value for '{action.field_name}'.", partial_credit=0.0)

    def _handle_flag(self, action: Action) -> Reward:
        if self.current_task["type"] == "validation":
            if not action.issue_code:
                return Reward(score=-0.1, reason="Flag action needs issue_code.", partial_credit=0.0)

            actual_issues = set(self.current_task["ground_truth"]["issues"])
            already_flagged = action.issue_code in self.flagged_issues
            self.flagged_issues.add(action.issue_code)

            if already_flagged:
                return Reward(score=-0.03, reason=f"Issue '{action.issue_code}' was already flagged.", partial_credit=0.0)
            if action.issue_code in actual_issues:
                return Reward(score=0.15, reason=f"Correctly flagged '{action.issue_code}'.", partial_credit=0.15)
            return Reward(score=-0.05, reason=f"'{action.issue_code}' is a false positive.", partial_credit=0.0)

        if self.current_task["type"] == "reconciliation":
            if not action.invoice_id:
                return Reward(score=-0.1, reason="Flag action needs invoice_id for reconciliation.", partial_credit=0.0)

            true_unmatched = set(self.current_task["ground_truth"]["unmatched_invoices"])
            true_discrepancies = set(self.current_task["ground_truth"].get("discrepancies", {}).keys())

            if action.issue_code == "amount_mismatch":
                if action.invoice_id in self.flagged_discrepancies:
                    return Reward(
                        score=-0.03,
                        reason=f"Invoice '{action.invoice_id}' discrepancy was already flagged.",
                        partial_credit=0.0,
                    )

                self.flagged_discrepancies.add(action.invoice_id)
                if action.invoice_id in true_discrepancies:
                    return Reward(
                        score=0.14,
                        reason=f"Correctly flagged amount mismatch for '{action.invoice_id}'.",
                        partial_credit=0.14,
                    )
                return Reward(
                    score=-0.06,
                    reason=f"Invoice '{action.invoice_id}' does not have a reportable amount mismatch.",
                    partial_credit=0.0,
                )

            if action.invoice_id in self.unmatched_invoices:
                return Reward(score=-0.03, reason=f"Invoice '{action.invoice_id}' was already flagged.", partial_credit=0.0)

            self.unmatched_invoices.add(action.invoice_id)
            if action.invoice_id in true_unmatched:
                return Reward(score=0.18, reason=f"Correctly flagged unmatched invoice '{action.invoice_id}'.", partial_credit=0.18)
            return Reward(score=-0.06, reason=f"Invoice '{action.invoice_id}' should not be flagged unmatched.", partial_credit=0.0)

        return Reward(score=-0.08, reason="Flag is not useful for this task.", partial_credit=0.0)

    def _handle_match(self, action: Action) -> Reward:
        if self.current_task["type"] != "reconciliation":
            return Reward(score=-0.08, reason="Match is not useful for this task.", partial_credit=0.0)

        if not action.invoice_id or not action.po_id:
            return Reward(score=-0.1, reason="Match action needs invoice_id and po_id.", partial_credit=0.0)

        ground_truth_matches = self.current_task["ground_truth"]["matches"]
        previous_po = self.matches.get(action.invoice_id)
        self.matches[action.invoice_id] = action.po_id

        if previous_po == action.po_id:
            return Reward(score=-0.03, reason=f"Match for '{action.invoice_id}' already recorded.", partial_credit=0.0)
        if ground_truth_matches.get(action.invoice_id) == action.po_id:
            return Reward(score=0.16, reason=f"Correct match {action.invoice_id} -> {action.po_id}.", partial_credit=0.16)
        return Reward(score=-0.05, reason=f"Incorrect match {action.invoice_id} -> {action.po_id}.", partial_credit=0.0)

    def _handle_submit(self) -> Reward:
        final_score = self._current_score_snapshot()
        self.done = True
        step_penalty = min(0.25, 0.02 * max(0, self.step_count - 1))
        score = max(-1.0, min(1.0, final_score - step_penalty))
        return Reward(
            score=round(score, 4),
            reason=f"Submitted episode with final task score {final_score:.4f} and step penalty {step_penalty:.2f}.",
            partial_credit=round(final_score, 4),
        )

    def _current_score_snapshot(self) -> float:
        task_type = self.current_task["type"]
        if task_type == "extraction":
            return grade_easy_submission(self.extracted_fields, self.current_task["ground_truth"])
        if task_type == "validation":
            return grade_medium_submission(self.flagged_issues, self.current_task["ground_truth"]["issues"])
        return grade_hard_submission(
            self.matches,
            self.unmatched_invoices,
            self.flagged_discrepancies,
            self.current_task["ground_truth"]["matches"],
            self.current_task["ground_truth"]["unmatched_invoices"],
            self.current_task["ground_truth"].get("discrepancies", {}),
        )
