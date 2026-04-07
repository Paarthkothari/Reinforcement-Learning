from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

from env import Action, FinanceOpsEnv
from inference import (
    _build_vendor_lookup,
    _extract_first_matching_label,
    _extract_po_reference_candidates,
    _find_duplicate_invoice_ids,
    _normalize_vendor_name,
    _within_amount_tolerance,
    build_client,
    choose_action,
    heuristic_action,
)

QTABLE_PATH = Path("qtable.json")
LOG_PATH = Path("training_log.jsonl")
RESULTS_PATH = Path("training_results.json")

DIFFICULTIES = ["easy", "medium", "hard"]
ALPHA = 0.15
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995


class QTable:
    def __init__(self) -> None:
        self._table: Dict[str, Dict[str, float]] = {}

    def get(self, state_key: str, action_key: str) -> float:
        return self._table.get(state_key, {}).get(action_key, 0.0)

    def update(self, state_key: str, action_key: str, value: float) -> None:
        self._table.setdefault(state_key, {})[action_key] = round(value, 6)

    def best_action(self, state_key: str, action_keys: List[str]) -> str | None:
        if not action_keys:
            return None
        q_values = {action_key: self.get(state_key, action_key) for action_key in action_keys}
        return max(q_values, key=q_values.__getitem__)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._table, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QTable":
        qtable = cls()
        if path.exists():
            qtable._table = json.loads(path.read_text(encoding="utf-8"))
        return qtable

    def stats(self) -> Dict[str, int]:
        return {
            "states": len(self._table),
            "q_entries": sum(len(action_map) for action_map in self._table.values()),
        }


def _action_key(action: Action) -> str:
    return json.dumps(action.model_dump(exclude_none=True), sort_keys=True)


def _dedupe_actions(actions: List[Action]) -> List[Action]:
    deduped: List[Action] = []
    seen: set[str] = set()
    for action in actions:
        action_key = _action_key(action)
        if action_key not in seen:
            seen.add(action_key)
            deduped.append(action)
    return deduped


def _encode_state(observation: Dict) -> str:
    content = observation["content"]
    document = content["document"]
    parts = [
        observation["task_id"],
        observation["difficulty"],
        f"s{observation['step_number']}",
    ]

    if observation["difficulty"] == "easy":
        current = content.get("current_submission", {})
        parts.append("ext=" + "|".join(sorted(current.keys())))
        parts.append("req=" + "|".join(sorted(document.get("required_fields", []))))
    elif observation["difficulty"] == "medium":
        parts.append("flags=" + "|".join(sorted(content.get("flagged_issues", []))))
        parts.append("catalog=" + "|".join(sorted(document.get("known_issue_catalog", []))))
    else:
        parts.append("matches=" + "|".join(sorted(content.get("current_matches", {}).keys())))
        parts.append("unmatched=" + "|".join(sorted(content.get("current_unmatched_invoices", []))))
        parts.append("disc=" + "|".join(sorted(content.get("current_flagged_discrepancies", []))))
        parts.append("dup=" + "|".join(sorted(content.get("current_flagged_duplicate_invoices", []))))
        parts.append("split=" + "|".join(sorted(content.get("current_flagged_split_invoices", []))))
    return ":".join(parts)


def _easy_candidates(observation: Dict) -> List[Action]:
    content = observation["content"]
    document = content["document"]
    invoice_text = document["invoice_text"]
    field_aliases = document.get("field_aliases", {})
    vendor_aliases = document.get("normalization_hints", {}).get("vendor_aliases", {})
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
    actions = [
        Action(action_type="extract", field_name=field, field_value=value)
        for field, value in extracted_map.items()
        if value and field not in current
    ]
    actions.append(Action(action_type="submit"))
    return _dedupe_actions(actions)


def _medium_candidates(observation: Dict) -> List[Action]:
    content = observation["content"]
    document = content["document"]
    flagged = set(content.get("flagged_issues", []))
    actions = [
        Action(action_type="flag", issue_code=issue)
        for issue in document.get("known_issue_catalog", [])
        if issue not in flagged
    ]
    actions.append(Action(action_type="submit"))
    return _dedupe_actions(actions)


def _hard_candidates(observation: Dict) -> List[Action]:
    content = observation["content"]
    document = content["document"]
    purchase_orders = document["purchase_orders"]
    invoices = document["invoices"]
    current_matches = content.get("current_matches", {})
    current_unmatched = set(content.get("current_unmatched_invoices", []))
    current_discrepancies = set(content.get("current_flagged_discrepancies", []))
    current_duplicate_flags = set(content.get("current_flagged_duplicate_invoices", []))
    current_split_flags = set(content.get("current_flagged_split_invoices", []))
    vendor_lookup = _build_vendor_lookup(document)
    alias_map = vendor_lookup["alias_map"]
    po_by_vendor = {po["vendor"]: po for po in purchase_orders}
    po_by_reference = {po["po_reference"]: po for po in purchase_orders if po.get("po_reference")}
    duplicate_invoice_ids = _find_duplicate_invoice_ids(invoices)
    matching_policy = document.get("matching_policy", {})

    actions: List[Action] = []
    for invoice in invoices:
        invoice_id = invoice["invoice_id"]
        if invoice_id in duplicate_invoice_ids and invoice_id not in current_duplicate_flags:
            actions.append(Action(action_type="flag", invoice_id=invoice_id, issue_code="duplicate_invoice"))

        po_references = _extract_po_reference_candidates(invoice)
        if len(po_references) > 1 and invoice_id not in current_split_flags:
            actions.append(Action(action_type="flag", invoice_id=invoice_id, issue_code="split_po"))
            continue

        canonical_vendor = alias_map.get(invoice["vendor"], invoice["vendor"])
        candidate_po = None
        if len(po_references) == 1 and po_references[0] in po_by_reference:
            referenced_po = po_by_reference[po_references[0]]
            if referenced_po["vendor"] == canonical_vendor:
                candidate_po = referenced_po
        if candidate_po is None:
            candidate_po = po_by_vendor.get(canonical_vendor)

        if candidate_po is not None:
            if current_matches.get(invoice_id) != candidate_po["po_id"]:
                actions.append(Action(action_type="match", invoice_id=invoice_id, po_id=candidate_po["po_id"]))
            if not _within_amount_tolerance(invoice, candidate_po, matching_policy) and invoice_id not in current_discrepancies:
                actions.append(Action(action_type="flag", invoice_id=invoice_id, issue_code="amount_mismatch"))
        elif invoice_id not in current_unmatched:
            actions.append(Action(action_type="flag", invoice_id=invoice_id))

    actions.append(Action(action_type="submit"))
    return _dedupe_actions(actions)


def build_candidate_actions(observation: Dict) -> List[Action]:
    difficulty = observation["difficulty"]
    if difficulty == "easy":
        return _easy_candidates(observation)
    if difficulty == "medium":
        return _medium_candidates(observation)
    return _hard_candidates(observation)


def _select_action(
    qtable: QTable,
    state_key: str,
    observation: Dict,
    epsilon: float,
    policy: str,
    client,
    model_name: str,
) -> Action:
    base_action = choose_action(client if policy == "model" else None, model_name, observation)
    candidates = build_candidate_actions(observation)
    if _action_key(base_action) not in {_action_key(candidate) for candidate in candidates}:
        candidates.append(base_action)

    if random.random() < epsilon:
        return random.choice(candidates)

    candidate_map = {_action_key(candidate): candidate for candidate in candidates}
    best_action_key = qtable.best_action(state_key, list(candidate_map))
    if best_action_key is None:
        return base_action

    best_value = qtable.get(state_key, best_action_key)
    if best_value == 0.0 and _action_key(base_action) in candidate_map:
        return candidate_map[_action_key(base_action)]
    return candidate_map[best_action_key]


def run_episode(
    env: FinanceOpsEnv,
    difficulty: str,
    qtable: QTable,
    epsilon: float,
    policy: str,
    client,
    model_name: str,
) -> Dict:
    observation = env.reset(difficulty).model_dump()
    observation["last_action_error"] = None
    done = False
    total_reward = 0.0
    transitions: List[Dict] = []
    steps = 0
    final_score = 0.0

    while not done and steps < 20:
        steps += 1
        state_key = _encode_state(observation)
        action = _select_action(qtable, state_key, observation, epsilon, policy, client, model_name)
        next_observation, reward, done, info = env.step(action)
        next_observation_payload = next_observation.model_dump()
        next_observation_payload["last_action_error"] = info.get("last_action_error")
        next_state_key = _encode_state(next_observation_payload)
        next_candidates = [] if done else build_candidate_actions(next_observation_payload)
        next_q = 0.0
        if next_candidates:
            next_q = max(qtable.get(next_state_key, _action_key(candidate)) for candidate in next_candidates)

        action_key = _action_key(action)
        old_q = qtable.get(state_key, action_key)
        updated_q = old_q + ALPHA * (reward.score + (GAMMA * next_q) - old_q)
        qtable.update(state_key, action_key, updated_q)

        total_reward += reward.score
        final_score = float(info.get("score_snapshot", final_score))
        transitions.append(
            {
                "state": state_key,
                "action": action.model_dump(exclude_none=True),
                "reward": round(reward.score, 4),
                "next_state": next_state_key,
                "done": done,
                "error": info.get("last_action_error"),
            }
        )
        observation = next_observation_payload

    return {
        "task_id": observation["task_id"],
        "difficulty": difficulty,
        "steps": steps,
        "total_reward": round(total_reward, 4),
        "final_score": round(max(0.0, min(1.0, final_score)), 4),
        "epsilon": round(epsilon, 4),
        "transitions": transitions,
        "success": done,
    }


def train(difficulties: List[str], n_episodes: int, policy: str, resume: bool) -> Dict[str, Dict[str, float]]:
    qtable = QTable.load(QTABLE_PATH) if resume else QTable()
    policy = policy.strip().lower()
    if policy not in {"heuristic", "model"}:
        raise ValueError("policy must be either 'heuristic' or 'model'")

    client = None
    model_name = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1:fastest")
    if policy == "model":
        client = build_client()

    env = FinanceOpsEnv()
    results: Dict[str, List[float]] = {difficulty: [] for difficulty in difficulties}
    epsilon = EPSILON_START
    episode_number = 0

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        for episode_index in range(n_episodes):
            for difficulty in difficulties:
                episode_number += 1
                episode = run_episode(env, difficulty, qtable, epsilon, policy, client, model_name)
                results[difficulty].append(episode["final_score"])
                log_file.write(json.dumps({k: v for k, v in episode.items() if k != "transitions"}) + "\n")

                if episode_number % 10 == 0 or episode_index == 0:
                    recent_scores = results[difficulty][-20:]
                    average_score = sum(recent_scores) / len(recent_scores)
                    print(
                        f"ep={episode_number:4d} | {difficulty:6s} | "
                        f"score={episode['final_score']:.3f} | avg20={average_score:.3f} | "
                        f"steps={episode['steps']:2d} | eps={epsilon:.3f}"
                    )

            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            if (episode_index + 1) % 50 == 0:
                qtable.save(QTABLE_PATH)
                print(f"[checkpoint] saved {QTABLE_PATH} with {qtable.stats()}")

    qtable.save(QTABLE_PATH)
    summary: Dict[str, Dict[str, float]] = {}
    for difficulty, scores in results.items():
        tail = scores[-50:] if len(scores) >= 50 else scores
        summary[difficulty] = {
            "avg_last_window": round(sum(tail) / len(tail), 4),
            "best": round(max(scores), 4),
            "worst": round(min(scores), 4),
        }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] saved {QTABLE_PATH} with {qtable.stats()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FinanceOps OpenEnv training loop")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--policy", choices=["heuristic", "model"], default=os.getenv("POLICY", "model"))
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    selected_difficulties = DIFFICULTIES if arguments.task == "all" else [arguments.task]
    train(
        difficulties=selected_difficulties,
        n_episodes=arguments.episodes,
        policy=arguments.policy,
        resume=arguments.resume,
    )
