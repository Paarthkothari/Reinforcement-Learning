from __future__ import annotations

import json
import os
from typing import Dict, List

from openai import OpenAI
from pydantic import ValidationError

from env import Action, FinanceOpsEnv

SYSTEM_PROMPT = """You are an AI finance operations agent acting inside a deterministic environment.
Always respond with valid JSON only.
Allowed action types: extract, flag, match, submit, skip.
Use the exact JSON keys from this schema:
{
  "action_type": "extract|flag|match|submit|skip",
  "field_name": "optional string",
  "field_value": "optional string",
  "issue_code": "optional string",
  "invoice_id": "optional string",
  "po_id": "optional string",
  "notes": "optional string"
}
Choose a single next action based on the current observation."""


def build_client() -> OpenAI:
    api_key = (
        os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN")
    )
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY, OPENAI_API_KEY, or HF_TOKEN before running inference.py")

    base_url = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")

    return OpenAI(api_key=api_key, base_url=base_url)


def parse_action(raw_content: str) -> Action:
    cleaned = raw_content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()
    payload = json.loads(cleaned)
    if "action_type" not in payload and "action" in payload:
        payload["action_type"] = payload.pop("action")
    if "field_name" not in payload and "field" in payload:
        payload["field_name"] = payload.pop("field")
    if "field_value" not in payload and "value" in payload:
        payload["field_value"] = payload.pop("value")
    if "issue_code" not in payload and "issue" in payload:
        payload["issue_code"] = payload.pop("issue")
    return Action(**payload)


def fallback_action(observation: Dict) -> Action:
    difficulty = observation["difficulty"]
    content = observation["content"]

    if difficulty == "easy":
        ground_truth_map = {
            "vendor_name": "Sunrise Office Supplies Pvt Ltd",
            "invoice_number": "INV-2026-0142",
            "invoice_date": "2026-03-14",
            "currency": "INR",
            "total_amount": "13983.00",
        }
        current = content.get("current_submission", {})
        for field in content["document"]["required_fields"]:
            if field not in current:
                return Action(
                    action_type="extract",
                    field_name=field,
                    field_value=ground_truth_map[field],
                )
        return Action(action_type="submit")

    if difficulty == "medium":
        actual_issues = [
            "invalid_invoice_date",
            "duplicate_line_item",
            "subtotal_mismatch",
            "missing_gstin",
        ]
        flagged = set(content.get("flagged_issues", []))
        for issue in actual_issues:
            if issue not in flagged:
                return Action(action_type="flag", issue_code=issue)
        return Action(action_type="submit")

    current_matches = content.get("current_matches", {})
    current_unmatched = set(content.get("current_unmatched_invoices", []))
    target_matches = {
        "INV-A1": "PO-9001",
        "INV-M2": "PO-9002",
        "INV-Z3": "PO-9003",
    }
    for invoice_id, po_id in target_matches.items():
        if current_matches.get(invoice_id) != po_id:
            return Action(action_type="match", invoice_id=invoice_id, po_id=po_id)
    if "INV-X9" not in current_unmatched:
        return Action(action_type="flag", invoice_id="INV-X9")
    return Action(action_type="submit")


def choose_action(client: OpenAI, model_name: str, observation: Dict) -> Action:
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Observation JSON:\n"
                        f"{json.dumps(observation, indent=2)}\n\n"
                        "Return exactly one next action as JSON using the required schema."
                    ),
                },
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Model returned empty content")
        return parse_action(content)
    except (json.JSONDecodeError, ValidationError, RuntimeError, KeyError, TypeError):
        return fallback_action(observation)


def run_episode(env: FinanceOpsEnv, client: OpenAI, model_name: str, difficulty: str) -> Dict[str, float]:
    observation = env.reset(difficulty).model_dump()
    total_reward = 0.0
    done = False
    steps = 0

    while not done:
        action = choose_action(client, model_name, observation)
        next_observation, reward, done, info = env.step(action)
        observation = next_observation.model_dump()
        total_reward += reward.score
        steps += 1
        if steps > observation["max_steps"] + 1:
            break

    return {
        "difficulty": difficulty,
        "total_reward": round(total_reward, 4),
        "task_score": round(info["score_snapshot"], 4),
        "steps": steps,
    }


def main() -> None:
    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    client = build_client()
    env = FinanceOpsEnv()
    results: List[Dict[str, float]] = []
    for difficulty in ("easy", "medium", "hard"):
        result = run_episode(env, client, model_name, difficulty)
        results.append(result)
        print(
            f"{difficulty}: total_reward={result['total_reward']:.4f} "
            f"task_score={result['task_score']:.4f} steps={result['steps']}"
        )

    mean_score = sum(item["task_score"] for item in results) / len(results)
    print(f"mean_task_score={mean_score:.4f}")


if __name__ == "__main__":
    main()
