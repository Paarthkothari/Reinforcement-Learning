from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Observation(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    content: Dict[str, Any]
    available_actions: List[str]
    step_number: int = Field(ge=0)
    max_steps: int = Field(ge=1)
    context: Optional[str] = None


class Action(BaseModel):
    action_type: Literal["extract", "flag", "match", "submit", "skip"]
    field_name: Optional[str] = None
    field_value: Optional[str] = None
    issue_code: Optional[str] = None
    invoice_id: Optional[str] = None
    po_id: Optional[str] = None
    notes: Optional[str] = None


class Reward(BaseModel):
    score: float = Field(gt=0.0, lt=1.0)
    reason: str
    partial_credit: float = Field(gt=0.0, lt=1.0)
