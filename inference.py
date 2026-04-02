from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import ValidationError

from env import Action, FinanceOpsEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("FINANCE_OPS_TASK", "invoice_extract_easy")
BENCHMARK = os.getenv("FINANCE_OPS_BENCHMARK", "finance-ops-openenv")
BASELINE_MODE = os.getenv("BASELINE_MODE", "heuristic").strip().lower()
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))

TASK_NAME_TO_DIFFICULTY = {
    "invoice_extract_easy": "easy",
    "invoice_validate_medium": "medium",
    "po_reconcile_hard": "hard",
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}

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
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN before running inference.py in model mode")
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


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


def _extract_first_matching_label(text: str, labels: List[str]) -> str | None:
    for label in labels:
        value = _extract_label_value(text, label)
        if value is not None:
            return value
    return None


def _safe_decimal(value: str) -> float:
    return float(value.replace(",", "").strip())


def _normalize_vendor_name(value: str | None, aliases: Dict[str, str]) -> str | None:
    if value is None:
        return None
    return aliases.get(value, value)


def _build_vendor_lookup(document: Dict) -> Dict[str, Dict[str, Dict[str, str]] | Dict[str, str]]:
    po_by_id: Dict[str, Dict[str, str]] = {}
    alias_map: Dict[str, str] = {}
    for po in document.get("purchase_orders", []):
        po_by_id[po["po_id"]] = po

    for vendor_entry in document.get("vendor_directory", []):
        canonical_vendor = vendor_entry["canonical_vendor"]
        alias_map[canonical_vendor] = canonical_vendor
        for alias in vendor_entry.get("aliases", []):
            alias_map[alias] = canonical_vendor

    return {"po_by_id": po_by_id, "alias_map": alias_map}


def _find_duplicate_invoice_ids(invoices: List[Dict[str, str]]) -> set[str]:
    seen_by_external_number: Dict[str, str] = {}
    duplicate_ids: set[str] = set()
    for invoice in invoices:
        external_number = invoice.get("external_invoice_number")
        if not external_number:
            continue
        if external_number in seen_by_external_number:
            duplicate_ids.add(invoice["invoice_id"])
        else:
            seen_by_external_number[external_number] = invoice["invoice_id"]
    return duplicate_ids


def heuristic_action(observation: Dict) -> Action:
    difficulty = observation["difficulty"]
    content = observation["content"]

    if difficulty == "easy":
        invoice_text = content["document"]["invoice_text"]
        field_aliases = content["document"].get("field_aliases", {})
        vendor_aliases = content["document"].get("normalization_hints", {}).get("vendor_aliases", {})
        extracted_map = {
            "vendor_name": _normalize_vendor_name(
                _extract_first_matching_label(invoice_text, field_aliases.get("vendor_name", ["Vendor"])),
                vendor_aliases,
            ),
            "invoice_number": _extract_first_matching_label(
                invoice_text,
                field_aliases.get("invoice_number", ["Invoice Number"]),
            ),
            "invoice_date": _extract_first_matching_label(
                invoice_text,
                field_aliases.get("invoice_date", ["Invoice Date"]),
            ),
            "currency": _extract_first_matching_label(
                invoice_text,
                field_aliases.get("currency", ["Currency"]),
            ),
            "total_amount": _extract_first_matching_label(
                invoice_text,
                field_aliases.get("total_amount", ["Total Amount"]),
            ),
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
    document = content["document"]
    purchase_orders = document["purchase_orders"]
    invoices = document["invoices"]
    matching_policy = document.get("matching_policy", {})
    amount_tolerance = float(matching_policy.get("amount_tolerance", "0.00"))
    vendor_lookup = _build_vendor_lookup(document)
    po_by_id = vendor_lookup["po_by_id"]
    alias_map = vendor_lookup["alias_map"]
    po_by_vendor = {po["vendor"]: po for po in purchase_orders}
    current_duplicate_flags = set(content.get("current_flagged_duplicate_invoices", []))
    duplicate_invoice_ids = _find_duplicate_invoice_ids(invoices)
    po_by_reference = {
        po["po_reference"]: po for po in purchase_orders if po.get("po_reference")
    }

    def resolve_po(invoice: Dict[str, str]) -> Dict[str, str] | None:
        canonical_vendor = alias_map.get(invoice["vendor"], invoice["vendor"])
        po_reference = invoice.get("po_reference")
        if po_reference and po_reference in po_by_reference:
            referenced_po = po_by_reference[po_reference]
            if referenced_po["vendor"] == canonical_vendor:
                return referenced_po
        return po_by_vendor.get(canonical_vendor)

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids and invoice_id not in current_duplicate_flags:
            return Action(action_type="flag", invoice_id=invoice_id, issue_code="duplicate_invoice")

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids:
            continue
        matching_po = resolve_po(invoice)
        if matching_po and current_matches.get(invoice_id) != matching_po["po_id"]:
            return Action(action_type="match", invoice_id=invoice_id, po_id=matching_po["po_id"])

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids:
            continue
        matching_po = resolve_po(invoice)
        if not matching_po and invoice_id not in current_unmatched:
            return Action(action_type="flag", invoice_id=invoice_id)

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids:
            continue
        matched_po_id = current_matches.get(invoice_id)
        matching_po = po_by_id.get(matched_po_id) if matched_po_id else resolve_po(invoice)
        if not matching_po:
            continue

    invoices_by_po: Dict[str, List[Dict[str, str]]] = {}
    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids:
            continue
        matched_po_id = current_matches.get(invoice_id)
        matching_po = po_by_id.get(matched_po_id) if matched_po_id else resolve_po(invoice)
        if not matching_po:
            continue
        invoices_by_po.setdefault(matching_po["po_id"], []).append(invoice)

    for po_id, po_invoices in invoices_by_po.items():
        matching_po = po_by_id[po_id]
        invoice_total = sum(_safe_decimal(invoice["amount"]) for invoice in po_invoices)
        po_total = _safe_decimal(matching_po["amount"])
        if abs(invoice_total - po_total) <= amount_tolerance:
            continue

        for invoice in po_invoices:
            invoice_id = invoice["invoice_id"]
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


def action_to_log_string(action: Action) -> str:
    if action.action_type == "extract":
        return f"extract({action.field_name!r},{action.field_value!r})"
    if action.action_type == "flag":
        if action.issue_code and action.invoice_id:
            return f"flag({action.invoice_id!r},{action.issue_code!r})"
        if action.issue_code:
            return f"flag({action.issue_code!r})"
        if action.invoice_id:
            return f"flag({action.invoice_id!r})"
    if action.action_type == "match":
        return f"match({action.invoice_id!r},{action.po_id!r})"
    if action.action_type == "submit":
        return "submit()"
    if action.action_type == "skip":
        return "skip()"
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def resolve_difficulty(task_name: str) -> str:
    try:
        return TASK_NAME_TO_DIFFICULTY[task_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(TASK_NAME_TO_DIFFICULTY))
        raise RuntimeError(f"Unsupported FINANCE_OPS_TASK '{task_name}'. Expected one of: {valid_names}") from exc


def main() -> None:
    difficulty = resolve_difficulty(TASK_NAME)
    client = build_client() if BASELINE_MODE == "model" else None
    env = FinanceOpsEnv()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(difficulty).model_dump()
        done = False

        while not done:
            action = choose_action(client, MODEL_NAME, observation)
            next_observation, reward, done, info = env.step(action)

            rewards.append(reward.score)
            steps_taken += 1
            score = max(0.0, min(1.0, float(info.get("score_snapshot", 0.0))))

            log_step(
                step=steps_taken,
                action=action_to_log_string(action),
                reward=reward.score,
                done=done,
                error=info.get("last_action_error"),
            )

            observation = next_observation.model_dump()
            if steps_taken > observation["max_steps"] + 1:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        print(f"[DEBUG] inference error: {exc}", file=sys.stderr, flush=True)
        success = False
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
