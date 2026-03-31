from __future__ import annotations

import json
import os
import re
from datetime import datetime
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running inference.py")

    base_url = os.getenv("API_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


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


def _extract_label_value(text: str, label: str) -> str | None:
    pattern = rf"^{re.escape(label)}:\s*(.+)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _safe_decimal(value: str) -> float:
    return float(value.replace(",", "").strip())


def heuristic_action(observation: Dict) -> Action:
    difficulty = observation["difficulty"]
    content = observation["content"]

    if difficulty == "easy":
        invoice_text = content["document"]["invoice_text"]
        extracted_map = {
            "vendor_name": _extract_label_value(invoice_text, "Vendor"),
            "invoice_number": _extract_label_value(invoice_text, "Invoice Number"),
            "invoice_date": _extract_label_value(invoice_text, "Invoice Date"),
            "currency": _extract_label_value(invoice_text, "Currency"),
            "total_amount": _extract_label_value(invoice_text, "Total Amount"),
        }
        current = content.get("current_submission", {})
        for field in content["document"]["required_fields"]:
            if field not in current and extracted_map.get(field):
                return Action(
                    action_type="extract",
                    field_name=field,
                    field_value=extracted_map[field],
                )
        return Action(action_type="submit")

    if difficulty == "medium":
        invoice_text = content["document"]["invoice_text"]
        actual_issues: List[str] = []
        invoice_date = _extract_label_value(invoice_text, "Invoice Date")
        subtotal = _extract_label_value(invoice_text, "Subtotal")
        line_item_matches = re.findall(r"^- (.+)$", invoice_text, flags=re.MULTILINE)
        gstin = _extract_label_value(invoice_text, "GSTIN")
        amount_matches = re.findall(r"= ([0-9.]+)", invoice_text)

        if invoice_date:
            try:
                datetime.strptime(invoice_date, "%d/%m/%Y")
            except ValueError:
                actual_issues.append("invalid_invoice_date")

        if len(line_item_matches) != len(set(line_item_matches)):
            actual_issues.append("duplicate_line_item")

        if subtotal and amount_matches:
            computed_subtotal = round(sum(_safe_decimal(amount) for amount in amount_matches), 2)
            if abs(computed_subtotal - _safe_decimal(subtotal)) > 0.009:
                actual_issues.append("subtotal_mismatch")

        if gstin and gstin.upper() == "MISSING":
            actual_issues.append("missing_gstin")

        flagged = set(content.get("flagged_issues", []))
        for issue in actual_issues:
            if issue not in flagged:
                return Action(action_type="flag", issue_code=issue)
        return Action(action_type="submit")

    current_matches = content.get("current_matches", {})
    current_unmatched = set(content.get("current_unmatched_invoices", []))
    current_discrepancies = set(content.get("current_flagged_discrepancies", []))
    purchase_orders = content["document"]["purchase_orders"]
    invoices = content["document"]["invoices"]

    po_by_vendor = {po["vendor"]: po for po in purchase_orders}
    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        matching_po = po_by_vendor.get(invoice["vendor"])
        if matching_po and current_matches.get(invoice_id) != matching_po["po_id"]:
            return Action(action_type="match", invoice_id=invoice_id, po_id=matching_po["po_id"])

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        matching_po = po_by_vendor.get(invoice["vendor"])
        if not matching_po and invoice_id not in current_unmatched:
            return Action(action_type="flag", invoice_id=invoice_id)

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        matching_po = po_by_vendor.get(invoice["vendor"])
        if not matching_po:
            continue
        if abs(_safe_decimal(invoice["amount"]) - _safe_decimal(matching_po["amount"])) > 0.009:
            if invoice_id not in current_discrepancies:
                return Action(action_type="flag", invoice_id=invoice_id, issue_code="amount_mismatch")

    return Action(action_type="submit")


def choose_action(client: OpenAI | None, model_name: str, observation: Dict) -> Action:
    if client is None:
        return heuristic_action(observation)

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
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
        return heuristic_action(observation)


def run_episode(env: FinanceOpsEnv, client: OpenAI | None, model_name: str, difficulty: str) -> Dict[str, float]:
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
    baseline_mode = os.getenv("BASELINE_MODE", "heuristic").strip().lower()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")

    client = build_client() if baseline_mode == "model" else None
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
