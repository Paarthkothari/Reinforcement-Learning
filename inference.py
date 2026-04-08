from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

from env import Action, FinanceOpsEnv


def load_api_key() -> str | None:
    return os.getenv("API_KEY") or os.getenv("HF_TOKEN")


def get_api_base_url() -> str | None:
    return os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").strip()


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1:fastest").strip()


def resolve_baseline_mode() -> str:
    mode = os.getenv("BASELINE_MODE", "model").strip().lower()
    if mode not in {"heuristic", "model"}:
        raise RuntimeError("BASELINE_MODE must be either 'heuristic' or 'model'")
    return mode


def _safe_reward(value: float) -> float:
    return round(max(0.01, min(0.99, float(value))), 4)


TASK_NAME = os.getenv("FINANCE_OPS_TASK", "invoice_extract_easy")
BENCHMARK = os.getenv("FINANCE_OPS_BENCHMARK", "finance-ops-openenv")
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

TASK_GUIDANCE = {
    "easy": {
        "goal": (
            "Extract exactly these fields from the invoice text: vendor_name, invoice_number, "
            "invoice_date, currency, total_amount."
        ),
        "rules": [
            "Only use action_type='extract' until all required fields are present, then use submit.",
            "For extract actions, both field_name and field_value are required.",
            "Do not invent field names outside the required fields.",
        ],
        "examples": [
            {"action_type": "extract", "field_name": "vendor_name", "field_value": "Sunrise Office Supplies Pvt Ltd"},
            {"action_type": "submit"},
        ],
    },
    "medium": {
        "goal": (
            "Review the invoice and flag anomalies from this catalog only: invalid_invoice_date, "
            "duplicate_line_item, subtotal_mismatch, missing_gstin."
        ),
        "rules": [
            "Only use action_type='flag' with issue_code for anomalies, then use submit when finished.",
            "For medium tasks, do not send invoice_id or po_id.",
            "A valid flag example is {'action_type':'flag','issue_code':'invalid_invoice_date'}.",
        ],
        "examples": [
            {"action_type": "flag", "issue_code": "invalid_invoice_date"},
            {"action_type": "submit"},
        ],
    },
    "hard": {
        "goal": (
            "Reconcile invoices to POs, flag unmatched invoices, flag amount mismatches, detect duplicate invoices, "
            "and flag invoices that reference multiple POs."
        ),
        "rules": [
            "Use match with both invoice_id and po_id when an invoice maps to a purchase order.",
            "Use flag with invoice_id only to mark an unmatched invoice.",
            "Use flag with invoice_id and issue_code='amount_mismatch' for reportable mismatches.",
            "Use flag with invoice_id and issue_code='duplicate_invoice' for duplicate submissions.",
            "Use flag with invoice_id and issue_code='split_po' when a single invoice references multiple POs.",
            "Apply the document FX rates and +/-2% tolerance before deciding whether an amount is a valid match.",
        ],
        "examples": [
            {"action_type": "match", "invoice_id": "INV-A1", "po_id": "PO-9001"},
            {"action_type": "flag", "invoice_id": "INV-X9"},
            {"action_type": "flag", "invoice_id": "INV-M2", "issue_code": "amount_mismatch"},
            {"action_type": "flag", "invoice_id": "INV-S1", "issue_code": "split_po"},
            {"action_type": "submit"},
        ],
    },
}


def build_client() -> OpenAI:
    api_base_url = get_api_base_url()
    api_key = load_api_key()
    if not api_base_url:
        raise RuntimeError("Set API_BASE_URL before running inference.py in model mode")
    if not api_key:
        raise RuntimeError("Set API_KEY before running inference.py in model mode")
    return OpenAI(api_key=api_key, base_url=api_base_url)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={_safe_reward(reward):.2f} done={done_value} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{_safe_reward(reward):.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={_safe_reward(score):.2f} rewards={rewards_str}",
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


def _extract_po_reference_candidates(invoice: Dict[str, str]) -> List[str]:
    references: List[str] = []
    if invoice.get("po_references"):
        references.extend(str(reference).strip() for reference in invoice["po_references"] if str(reference).strip())

    raw_reference = str(invoice.get("po_reference", "")).strip()
    if raw_reference:
        split_values = [raw_reference]
        for separator in ["/", "|", ";", ","]:
            next_values: List[str] = []
            for value in split_values:
                next_values.extend(value.split(separator))
            split_values = next_values
        references.extend(value.strip() for value in split_values if value.strip())

    deduped: List[str] = []
    seen: set[str] = set()
    for reference in references:
        if reference not in seen:
            seen.add(reference)
            deduped.append(reference)
    return deduped


def _amount_in_base_currency(amount: str, currency: str, fx_rates: Dict[str, float]) -> float:
    return _safe_decimal(amount) * float(fx_rates.get(currency, 1.0))


def _within_amount_tolerance(invoice: Dict[str, str], po: Dict[str, str], matching_policy: Dict) -> bool:
    fx_rates = matching_policy.get("fx_rates_to_inr", {})
    invoice_base = _amount_in_base_currency(invoice["amount"], invoice["currency"], fx_rates)
    po_base = _amount_in_base_currency(po["amount"], po["currency"], fx_rates)
    if po_base <= 0:
        return False

    tolerance_percent = float(matching_policy.get("amount_tolerance_percent", 0.0))
    if tolerance_percent > 0.0:
        return abs(invoice_base - po_base) <= (po_base * tolerance_percent / 100.0)

    absolute_tolerance = float(matching_policy.get("amount_tolerance", "0.0"))
    return abs(invoice_base - po_base) <= absolute_tolerance


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


def _task_specific_prompt(observation: Dict) -> str:
    difficulty = observation["difficulty"]
    guidance = TASK_GUIDANCE[difficulty]
    examples = "\n".join(json.dumps(example, separators=(",", ":")) for example in guidance["examples"])
    rules = "\n".join(f"- {rule}" for rule in guidance["rules"])
    review_memory = observation["content"].get("review_memory", [])
    last_action_error = observation.get("last_action_error")
    return (
        f"Task difficulty: {difficulty}\n"
        f"Goal: {guidance['goal']}\n"
        f"Rules:\n{rules}\n"
        f"Valid JSON examples:\n{examples}\n"
        f"Recent review memory: {json.dumps(review_memory, ensure_ascii=True)}\n"
        f"Last action error: {last_action_error or 'null'}"
    )


def _validate_action_for_observation(action: Action, observation: Dict) -> Tuple[bool, str | None]:
    difficulty = observation["difficulty"]

    if action.action_type == "extract":
        if difficulty != "easy":
            return False, "extract is only valid for easy extraction tasks"
        if not action.field_name or action.field_value is None:
            return False, "extract actions require both field_name and field_value"
        required_fields = set(observation["content"]["document"].get("required_fields", []))
        if action.field_name not in required_fields:
            return False, f"field_name must be one of {sorted(required_fields)}"
        return True, None

    if action.action_type == "flag":
        if difficulty == "easy":
            return False, "flag is not valid for easy extraction tasks"
        if difficulty == "medium":
            valid_issues = set(observation["content"]["document"].get("known_issue_catalog", []))
            if not action.issue_code:
                return False, "medium validation flags require issue_code"
            if action.issue_code not in valid_issues:
                return False, f"issue_code must be one of {sorted(valid_issues)}"
            if action.invoice_id or action.po_id:
                return False, "medium validation flags must not include invoice_id or po_id"
            return True, None
        if not action.invoice_id:
            return False, "hard reconciliation flags require invoice_id"
        if action.po_id:
            return False, "hard reconciliation flags must not include po_id"
        valid_issues = set(observation["content"]["document"].get("known_issue_catalog", []))
        if action.issue_code and action.issue_code not in valid_issues:
            return False, f"hard reconciliation issue_code must be one of {sorted(valid_issues)} or omitted"
        return True, None

    if action.action_type == "match":
        if difficulty != "hard":
            return False, "match is only valid for hard reconciliation tasks"
        if not action.invoice_id or not action.po_id:
            return False, "match actions require both invoice_id and po_id"
        return True, None

    if action.action_type in {"submit", "skip"}:
        return True, None

    return False, "unknown action_type"


def _build_messages(observation: Dict, corrective_note: str | None = None, raw_attempt: str | None = None) -> List[Dict[str, str]]:
    prompt_parts = [
        _task_specific_prompt(observation),
        "Observation JSON:",
        json.dumps(observation, indent=2),
        "Return exactly one next action as JSON using the required schema.",
    ]
    if corrective_note:
        prompt_parts.extend(
            [
                "Your previous action proposal was invalid for this task.",
                f"Correction: {corrective_note}",
            ]
        )
    if raw_attempt:
        prompt_parts.extend(["Invalid prior raw output:", raw_attempt])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(prompt_parts)},
    ]


def _request_model_action(
    client: OpenAI,
    model_name: str,
    observation: Dict,
    corrective_note: str | None = None,
    raw_attempt: str | None = None,
) -> Action:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=_build_messages(observation, corrective_note=corrective_note, raw_attempt=raw_attempt),
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Model returned empty content")
    action = parse_action(content)
    is_valid, validation_error = _validate_action_for_observation(action, observation)
    if not is_valid:
        raise RuntimeError(validation_error or "invalid action for task")
    return action


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
    current_split_flags = set(content.get("current_flagged_split_invoices", []))
    document = content["document"]
    purchase_orders = document["purchase_orders"]
    invoices = document["invoices"]
    matching_policy = document.get("matching_policy", {})
    vendor_lookup = _build_vendor_lookup(document)
    po_by_id = vendor_lookup["po_by_id"]
    alias_map = vendor_lookup["alias_map"]
    po_by_vendor = {po["vendor"]: po for po in purchase_orders}
    current_duplicate_flags = set(content.get("current_flagged_duplicate_invoices", []))
    duplicate_invoice_ids = _find_duplicate_invoice_ids(invoices)
    po_by_reference = {po["po_reference"]: po for po in purchase_orders if po.get("po_reference")}

    def resolve_po(invoice: Dict[str, str]) -> tuple[str, Dict[str, str] | None]:
        po_references = _extract_po_reference_candidates(invoice)
        if len(po_references) > 1:
            return "split_po", None

        canonical_vendor = alias_map.get(invoice["vendor"], invoice["vendor"])
        if po_references and po_references[0] in po_by_reference:
            referenced_po = po_by_reference[po_references[0]]
            if referenced_po["vendor"] == canonical_vendor:
                return "candidate", referenced_po
        return "candidate", po_by_vendor.get(canonical_vendor)

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids and invoice_id not in current_duplicate_flags:
            return Action(action_type="flag", invoice_id=invoice_id, issue_code="duplicate_invoice")

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids or invoice_id in current_duplicate_flags:
            continue
        resolution, _ = resolve_po(invoice)
        if resolution == "split_po" and invoice_id not in current_split_flags:
            return Action(action_type="flag", invoice_id=invoice_id, issue_code="split_po")

    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids or invoice_id in current_duplicate_flags or invoice_id in current_split_flags:
            continue

        _, matching_po = resolve_po(invoice)
        if matching_po is None:
            if invoice_id not in current_unmatched:
                return Action(action_type="flag", invoice_id=invoice_id)
            continue

        if _within_amount_tolerance(invoice, matching_po, matching_policy):
            if current_matches.get(invoice_id) != matching_po["po_id"]:
                return Action(action_type="match", invoice_id=invoice_id, po_id=matching_po["po_id"])
            continue

        if invoice_id not in current_discrepancies:
            return Action(action_type="flag", invoice_id=invoice_id, issue_code="amount_mismatch")

    invoices_by_po: Dict[str, List[Dict[str, str]]] = {}
    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids or invoice_id in current_duplicate_flags or invoice_id in current_split_flags:
            continue
        matched_po_id = current_matches.get(invoice_id)
        _, resolved_po = resolve_po(invoice)
        matching_po = po_by_id.get(matched_po_id) if matched_po_id else resolved_po
        if not matching_po:
            continue
        invoices_by_po.setdefault(matching_po["po_id"], []).append(invoice)

    tolerance_percent = float(matching_policy.get("amount_tolerance_percent", 0.0))
    fx_rates = matching_policy.get("fx_rates_to_inr", {})
    for po_id, po_invoices in invoices_by_po.items():
        matching_po = po_by_id[po_id]
        invoice_total = sum(
            _amount_in_base_currency(invoice["amount"], invoice["currency"], fx_rates)
            for invoice in po_invoices
        )
        po_total = _amount_in_base_currency(matching_po["amount"], matching_po["currency"], fx_rates)
        tolerance_value = po_total * (tolerance_percent / 100.0)
        if abs(invoice_total - po_total) <= tolerance_value:
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
        return _request_model_action(client, model_name, observation)
    except Exception as first_error:
        try:
            return _request_model_action(
                client,
                model_name,
                observation,
                corrective_note=str(first_error),
                raw_attempt=getattr(first_error, "doc", None),
            )
        except Exception:
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


def resolve_task_names() -> List[str]:
    raw_value = os.getenv("FINANCE_OPS_TASK", "").strip()
    if not raw_value:
        return [
            "invoice_extract_easy",
            "invoice_validate_medium",
            "po_reconcile_hard",
        ]

    return [task_name.strip() for task_name in raw_value.split(",") if task_name.strip()]


def run_episode(
    task_name: str,
    benchmark: str,
    success_score_threshold: float,
    model_name: str,
    client: OpenAI | None,
) -> None:
    difficulty = resolve_difficulty(task_name)
    env = FinanceOpsEnv()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.5
    success = False

    log_start(task=task_name, env=benchmark, model=model_name)

    try:
        observation = env.reset(difficulty).model_dump()
        observation["last_action_error"] = None
        done = False

        while not done:
            action = choose_action(client, model_name, observation)
            next_observation, reward, done, info = env.step(action)

            rewards.append(_safe_reward(reward.score))
            steps_taken += 1
            score = _safe_reward(info.get("score_snapshot", 0.5))

            log_step(
                step=steps_taken,
                action=action_to_log_string(action),
                reward=reward.score,
                done=done,
                error=info.get("last_action_error"),
            )

            observation = next_observation.model_dump()
            observation["last_action_error"] = info.get("last_action_error")
            if steps_taken > observation["max_steps"] + 1:
                break

        success = score >= success_score_threshold
    except Exception as exc:
        print(f"[DEBUG] inference error on task {task_name}: {exc}", file=sys.stderr, flush=True)
        success = False
    finally:
        if not rewards:
            try:
                last_reward = getattr(env, "last_reward", None)
                if last_reward is not None and getattr(last_reward, "score", None) is not None:
                    fallback = _safe_reward(last_reward.score)
                else:
                    fallback = _safe_reward(env._current_score_snapshot())
            except Exception:
                fallback = _safe_reward(0.5)
            rewards.append(fallback)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    benchmark = os.getenv("FINANCE_OPS_BENCHMARK", BENCHMARK)
    success_score_threshold = float(os.getenv("SUCCESS_SCORE_THRESHOLD", str(SUCCESS_SCORE_THRESHOLD)))
    model_name = get_model_name()
    baseline_mode = resolve_baseline_mode()
    task_names = resolve_task_names()

    client: OpenAI | None = None
    if baseline_mode == "model":
        try:
            client = build_client()
        except Exception as exc:
            print(f"[DEBUG] model mode unavailable, falling back to heuristic: {exc}", file=sys.stderr, flush=True)
            client = None

    for task_name in task_names:
        run_episode(
            task_name=task_name,
            benchmark=benchmark,
            success_score_threshold=success_score_threshold,
            model_name=model_name,
            client=client,
        )


if __name__ == "__main__":
    main()
