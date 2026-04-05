from __future__ import annotations

import hashlib
import os
import random
import string
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Tuple

TASK_KEYS = ("easy", "medium", "hard")
_ANCHOR_DATE = date(2026, 4, 1)
_CURRENCIES = ("INR", "USD", "EUR")
_FX_TO_INR = {"INR": 1.0, "USD": 83.5, "EUR": 90.2}

_VENDOR_PROFILES = [
    {
        "canonical": "Sunrise Office Supplies Pvt Ltd",
        "aliases": ["Sunrise Office Supplies Pvt. Ltd.", "Sunrise Office Supplies", "Sunrise Ops Supply"],
        "customers": ["Artha Retail LLP", "Helios Manufacturing Inc"],
        "invoice_prefix": "SOS",
        "po_prefix": "SUN",
    },
    {
        "canonical": "Apex Logistics Solutions",
        "aliases": ["Apex Logistics", "Apex Log Solutions", "Apex Logistics Solutions Pvt Ltd"],
        "customers": ["Nova Commerce Pvt Ltd", "Cedar Procurement GmbH"],
        "invoice_prefix": "APX",
        "po_prefix": "APX",
    },
    {
        "canonical": "BluePeak Shared Services Private Limited",
        "aliases": ["BluePeak Shared Services Pvt Ltd", "BluePeak Services", "BluePeak Shared Services"],
        "customers": ["North Harbor Freight", "Crest Retail Systems"],
        "invoice_prefix": "BPS",
        "po_prefix": "BPS",
    },
    {
        "canonical": "Metro Cable Works",
        "aliases": ["Metro Cable Works Ltd", "Metro Cables", "Metro Cables Ltd"],
        "customers": ["Vertex Cooling Systems", "Aster Buildtech LLP"],
        "invoice_prefix": "MET",
        "po_prefix": "MET",
    },
    {
        "canonical": "Zenith Packaging Co",
        "aliases": ["Zenith Packaging", "Zenith Pack", "Zenith Packaging Company"],
        "customers": ["Riverfront Packaging", "Cedar & Oak Commerce GmbH"],
        "invoice_prefix": "ZEN",
        "po_prefix": "ZEN",
    },
    {
        "canonical": "North Harbor Freight",
        "aliases": ["North Harbour Freight", "NH Freight", "North Harbor Freight Ltd"],
        "customers": ["Atlas Controls", "Meridian Paper Works"],
        "invoice_prefix": "NHF",
        "po_prefix": "NHF",
    },
    {
        "canonical": "Cedar Logistics LLC",
        "aliases": ["Cedar Logistics", "Cedar Logix", "Cedar Logistics LLP"],
        "customers": ["Urban Office Mart", "Helio Maintenance Co."],
        "invoice_prefix": "CED",
        "po_prefix": "CED",
    },
    {
        "canonical": "Atlas Controls Inc",
        "aliases": ["Atlas Controls", "Atlas Control Systems", "Atlas Controls Incorporated"],
        "customers": ["Northwind Industrial Services", "Pinnacle Stationery Hub"],
        "invoice_prefix": "ATL",
        "po_prefix": "ATL",
    },
    {
        "canonical": "Riverfront Packaging Co",
        "aliases": ["Riverfront Packaging", "Riverfront Pack", "Riverfront Packaging Company"],
        "customers": ["Maple Industrial Equipments", "Oakline Maintenance"],
        "invoice_prefix": "RIV",
        "po_prefix": "RIV",
    },
    {
        "canonical": "Maple Industrial Equipments",
        "aliases": ["Maple Industrial Equipment", "Maple Ind Equip", "Maple Industrial Equipments Ltd"],
        "customers": ["Silverline Packaging", "Vertex Cooling Systems"],
        "invoice_prefix": "MAP",
        "po_prefix": "MAP",
    },
]

_EASY_LABEL_VARIANTS = {
    "vendor": ["Vendor", "Vendor Name", "Supplier", "Seller"],
    "invoice_number": ["Invoice Number", "Invoice No", "Inv No", "Bill Number"],
    "invoice_date": ["Invoice Date", "Date", "Date of Invoice", "Bill Date"],
    "currency": ["Currency", "Curr.", "Currency Code", "CCY"],
    "subtotal": ["Subtotal", "Net Amount", "Pre-Tax Amount"],
    "tax": ["Tax", "GST", "VAT"],
    "total_amount": ["Total Amount", "Grand Total", "Amount Due", "Total Due"],
    "bill_to": ["Bill To", "Billed To", "Customer"],
}

_MEDIUM_ISSUES = [
    "invalid_invoice_date",
    "duplicate_line_item",
    "subtotal_mismatch",
    "missing_gstin",
]

_MEDIUM_REVIEW_NOTES = [
    "Note: rounding differences below 0.50 are acceptable and should not be flagged.",
    "Note: early-payment discounts can be valid when approved by procurement.",
    "Note: OCR artifacts may alter punctuation without changing invoice meaning.",
]

_MEDIUM_LINE_ITEMS = [
    ("Machine Parts A", 4500.00),
    ("Site Visit Fee", 2500.00),
    ("Warehouse Transfer", 1200.00),
    ("Fuel Surcharge", 450.00),
    ("Preventive Maintenance", 1800.00),
    ("Calibration Charge", 700.00),
    ("Packing Material", 950.00),
    ("Compliance Review", 1600.00),
]


def _seeded_rng(task_key: str, episode_index: int) -> random.Random:
    base_seed = os.getenv("FINANCE_OPS_SEED", "finance-ops-openenv")
    payload = f"{base_seed}:{task_key}:{episode_index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return random.Random(int(digest[:16], 16))


def _money(value: float) -> str:
    return f"{value:.2f}"


def _random_date(rng: random.Random, *, fmt: str = "iso") -> str:
    invoice_day = _ANCHOR_DATE - timedelta(days=rng.randint(5, 180))
    if fmt == "dd/mm/YYYY":
        return invoice_day.strftime("%d/%m/%Y")
    return invoice_day.isoformat()


def _invalid_invoice_date(rng: random.Random) -> str:
    invalid_templates = [
        "31/02/2026",
        "29/13/2026",
        "2026/14/03",
        "32/01/2026",
        "00/11/2026",
    ]
    return rng.choice(invalid_templates)


def _random_invoice_number(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(1000, 9999)}"


def _random_external_number(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randint(1000, 9999)}-{rng.choice(string.ascii_uppercase)}"


def _make_alias_map(profile: Dict[str, Any]) -> Dict[str, str]:
    aliases = {profile["canonical"]: profile["canonical"]}
    for alias in profile.get("aliases", []):
        aliases[alias] = profile["canonical"]
    return aliases


def _shuffle_unique(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _inr_to_currency(amount_in_inr: float, currency: str) -> float:
    return round(amount_in_inr / _FX_TO_INR[currency], 2)


def _currency_to_inr(amount: float, currency: str) -> float:
    return round(amount * _FX_TO_INR[currency], 2)


def _build_easy_task(episode_index: int) -> Dict[str, Any]:
    rng = _seeded_rng("easy", episode_index)
    vendor = rng.choice(_VENDOR_PROFILES)
    vendor_alias = rng.choice(vendor["aliases"])
    distractor_vendor = rng.choice(
        [profile["canonical"] for profile in _VENDOR_PROFILES if profile["canonical"] != vendor["canonical"]]
    )
    labels = {key: rng.choice(options) for key, options in _EASY_LABEL_VARIANTS.items()}
    currency = rng.choice(_CURRENCIES)
    subtotal = round(rng.uniform(1200.0, 24000.0), 2)
    tax_rate = rng.choice([0.05, 0.12, 0.18])
    tax = round(subtotal * tax_rate, 2)
    total_amount = round(subtotal + tax, 2)
    invoice_number = _random_invoice_number(rng, vendor["invoice_prefix"])
    invoice_date = _random_date(rng, fmt="iso")

    invoice_text = "\n".join(
        [
            "TAX INVOICE",
            f"{labels['vendor']}: {vendor_alias}",
            f"Account Manager Note: coordinated with {distractor_vendor} on the freight leg.",
            f"{labels['invoice_number']}: {invoice_number}",
            f"{labels['invoice_date']}: {invoice_date}",
            f"{labels['currency']}: {currency}",
            f"{labels['subtotal']}: {_money(subtotal)}",
            f"{labels['tax']}: {_money(tax)}",
            f"{labels['total_amount']}: {_money(total_amount)}",
            f"{labels['bill_to']}: {rng.choice(vendor['customers'])}",
        ]
    )

    return {
        "task_id": "invoice_extract_easy",
        "variant_id": f"easy_seeded_{episode_index:04d}",
        "difficulty": "easy",
        "type": "extraction",
        "max_steps": 8,
        "instructions": (
            "Extract the required invoice fields. Labels may vary and the invoice body may mention unrelated "
            "vendors that should not be extracted."
        ),
        "document": {
            "invoice_text": invoice_text,
            "required_fields": [
                "vendor_name",
                "invoice_number",
                "invoice_date",
                "currency",
                "total_amount",
            ],
            "field_aliases": {
                "vendor_name": _shuffle_unique([labels["vendor"], "Vendor", "Supplier", "Vendor Name", "Seller"]),
                "invoice_number": _shuffle_unique(
                    [labels["invoice_number"], "Invoice Number", "Invoice No", "Inv No", "Bill Number"]
                ),
                "invoice_date": _shuffle_unique(
                    [labels["invoice_date"], "Invoice Date", "Date", "Date of Invoice", "Bill Date"]
                ),
                "currency": _shuffle_unique([labels["currency"], "Currency", "Curr.", "Currency Code", "CCY"]),
                "total_amount": _shuffle_unique(
                    [labels["total_amount"], "Total Amount", "Grand Total", "Amount Due", "Total Due"]
                ),
            },
            "normalization_hints": {
                "vendor_aliases": _make_alias_map(vendor),
            },
            "noise_hints": [
                "Labels may use supplier or billing synonyms.",
                "Body notes can mention vendors unrelated to the billing entity.",
            ],
        },
        "ground_truth": {
            "vendor_name": vendor["canonical"],
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "currency": currency,
            "total_amount": _money(total_amount),
        },
    }


def _build_medium_task(episode_index: int) -> Dict[str, Any]:
    rng = _seeded_rng("medium", episode_index)
    vendor = rng.choice(_VENDOR_PROFILES)
    issues = sorted(rng.sample(_MEDIUM_ISSUES, k=rng.randint(2, len(_MEDIUM_ISSUES))))

    chosen_items = rng.sample(_MEDIUM_LINE_ITEMS, k=3)
    line_entries: List[Tuple[str, int, float, float]] = []
    for description, unit_price in chosen_items:
        quantity = rng.randint(1, 3)
        line_total = round(quantity * unit_price, 2)
        line_entries.append((description, quantity, unit_price, line_total))

    rendered_lines = [f"{desc} x{qty} @ {_money(unit)} = {_money(total)}" for desc, qty, unit, total in line_entries]
    if "duplicate_line_item" in issues:
        rendered_lines.append(rendered_lines[rng.randrange(len(rendered_lines))])

    computed_subtotal = round(sum(float(line.split("= ")[-1]) for line in rendered_lines), 2)
    if "subtotal_mismatch" in issues:
        displayed_subtotal = round(computed_subtotal + rng.choice([-250.0, -175.0, 225.0, 350.0]), 2)
    else:
        displayed_subtotal = computed_subtotal

    displayed_tax = round(displayed_subtotal * rng.choice([0.08, 0.12, 0.18]), 2)
    displayed_total = round(displayed_subtotal + displayed_tax, 2)
    invoice_date = _invalid_invoice_date(rng) if "invalid_invoice_date" in issues else _random_date(rng, fmt="dd/mm/YYYY")
    gstin = "MISSING" if "missing_gstin" in issues else f"27AAC{rng.randint(1000,9999)}F1Z{rng.randint(0,9)}"

    invoice_text = "\n".join(
        [
            f"Vendor: {vendor['canonical']}",
            f"Invoice Number: {_random_invoice_number(rng, vendor['invoice_prefix'])}",
            f"Invoice Date: {invoice_date}",
            f"Currency: {rng.choice(_CURRENCIES)}",
            "Line Items:",
            *[f"- {line}" for line in rendered_lines],
            f"Subtotal: {_money(displayed_subtotal)}",
            f"Tax: {_money(displayed_tax)}",
            f"Total Amount: {_money(displayed_total)}",
            f"GSTIN: {gstin}",
        ]
    )

    return {
        "task_id": "invoice_validate_medium",
        "variant_id": f"medium_seeded_{episode_index:04d}",
        "difficulty": "medium",
        "type": "validation",
        "max_steps": 10,
        "instructions": "Review the invoice and flag every genuine anomaly. False positives reduce score.",
        "document": {
            "invoice_text": invoice_text,
            "known_issue_catalog": list(_MEDIUM_ISSUES),
            "review_notes": rng.sample(_MEDIUM_REVIEW_NOTES, k=rng.randint(1, 2)),
        },
        "ground_truth": {
            "issues": issues,
        },
    }


def _make_purchase_order(rng: random.Random, profile: Dict[str, Any], sequence: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    currency = rng.choice(_CURRENCIES)
    amount_in_inr = round(rng.uniform(6000.0, 28000.0), 2)
    po_id = f"PO-{8200 + sequence:04d}"
    po_reference = f"{profile['po_prefix']}-{10 + sequence:02d}"
    po = {
        "po_id": po_id,
        "vendor": profile["canonical"],
        "aliases": list(profile["aliases"]),
        "amount": _money(_inr_to_currency(amount_in_inr, currency)),
        "currency": currency,
        "po_reference": po_reference,
    }
    return po, {"amount_in_inr": amount_in_inr}


def _make_invoice_id(rng: random.Random, prefix: str, sequence: int) -> str:
    suffix = "".join(rng.choices(string.ascii_uppercase, k=2))
    return f"INV-{prefix}{suffix}{sequence}"


def _make_valid_invoice(
    rng: random.Random,
    profile: Dict[str, Any],
    po: Dict[str, Any],
    po_amount_in_inr: float,
    *,
    sequence: int,
    variance_pct: float = 0.0,
    po_reference: str | None = None,
    external_number: str | None = None,
) -> Dict[str, Any]:
    invoice_currency = rng.choice(_CURRENCIES)
    adjusted_amount_in_inr = round(po_amount_in_inr * (1.0 + variance_pct), 2)
    return {
        "invoice_id": _make_invoice_id(rng, profile["invoice_prefix"], sequence),
        "vendor": rng.choice(profile["aliases"]),
        "amount": _money(_inr_to_currency(adjusted_amount_in_inr, invoice_currency)),
        "currency": invoice_currency,
        "po_reference": po_reference if po_reference is not None else po["po_reference"],
        "external_invoice_number": external_number or _random_external_number(rng, profile["invoice_prefix"]),
    }


def _build_hard_task(episode_index: int) -> Dict[str, Any]:
    rng = _seeded_rng("hard", episode_index)
    selected_profiles = rng.sample(_VENDOR_PROFILES, k=6)

    purchase_orders: List[Dict[str, Any]] = []
    po_blueprints: List[Dict[str, Any]] = []
    for index, profile in enumerate(selected_profiles, start=1):
        purchase_order, blueprint = _make_purchase_order(rng, profile, index)
        purchase_orders.append(purchase_order)
        po_blueprints.append(blueprint)

    matches: Dict[str, str] = {}
    unmatched_invoices: List[str] = []
    discrepancies: Dict[str, Dict[str, Any]] = {}
    duplicate_invoices: List[str] = []
    invoices: List[Dict[str, Any]] = []

    exact_invoice = _make_valid_invoice(
        rng,
        selected_profiles[0],
        purchase_orders[0],
        po_blueprints[0]["amount_in_inr"],
        sequence=1,
    )
    invoices.append(exact_invoice)
    matches[exact_invoice["invoice_id"]] = purchase_orders[0]["po_id"]

    fuzzy_variance = rng.uniform(-0.018, 0.018)
    fuzzy_invoice = _make_valid_invoice(
        rng,
        selected_profiles[1],
        purchase_orders[1],
        po_blueprints[1]["amount_in_inr"],
        sequence=2,
        variance_pct=fuzzy_variance,
    )
    invoices.append(fuzzy_invoice)
    matches[fuzzy_invoice["invoice_id"]] = purchase_orders[1]["po_id"]

    vendor_fallback_invoice = _make_valid_invoice(
        rng,
        selected_profiles[2],
        purchase_orders[2],
        po_blueprints[2]["amount_in_inr"],
        sequence=3,
        po_reference=f"WRONG-{selected_profiles[2]['po_prefix']}",
    )
    invoices.append(vendor_fallback_invoice)
    matches[vendor_fallback_invoice["invoice_id"]] = purchase_orders[2]["po_id"]

    mismatch_invoice = _make_valid_invoice(
        rng,
        selected_profiles[3],
        purchase_orders[3],
        po_blueprints[3]["amount_in_inr"],
        sequence=4,
        variance_pct=rng.uniform(0.05, 0.16),
    )
    invoices.append(mismatch_invoice)
    discrepancies[mismatch_invoice["invoice_id"]] = {
        "issue_code": "amount_mismatch",
        "base_currency_delta": _money(
            abs(_currency_to_inr(float(mismatch_invoice["amount"]), mismatch_invoice["currency"]) - po_blueprints[3]["amount_in_inr"])
        ),
    }

    split_vendor = selected_profiles[4]
    split_invoice_currency = rng.choice(_CURRENCIES)
    split_invoice_amount_in_inr = round(po_blueprints[4]["amount_in_inr"] * rng.uniform(0.85, 1.10), 2)
    split_invoice = {
        "invoice_id": _make_invoice_id(rng, split_vendor["invoice_prefix"], 5),
        "vendor": rng.choice(split_vendor["aliases"]),
        "amount": _money(_inr_to_currency(split_invoice_amount_in_inr, split_invoice_currency)),
        "currency": split_invoice_currency,
        "po_reference": f"{purchase_orders[4]['po_reference']}/{purchase_orders[5]['po_reference']}",
        "po_references": [purchase_orders[4]["po_reference"], purchase_orders[5]["po_reference"]],
        "external_invoice_number": _random_external_number(rng, split_vendor["invoice_prefix"]),
    }
    invoices.append(split_invoice)
    discrepancies[split_invoice["invoice_id"]] = {
        "issue_code": "split_po",
        "po_references": list(split_invoice["po_references"]),
    }

    orphan_invoice_currency = rng.choice(_CURRENCIES)
    orphan_invoice = {
        "invoice_id": _make_invoice_id(rng, "ORP", 6),
        "vendor": "Orbit Stationery Hub",
        "amount": _money(_inr_to_currency(rng.uniform(1800.0, 6200.0), orphan_invoice_currency)),
        "currency": orphan_invoice_currency,
        "po_reference": "",
        "external_invoice_number": _random_external_number(rng, "ORB"),
    }
    invoices.append(orphan_invoice)
    unmatched_invoices.append(orphan_invoice["invoice_id"])

    primary_duplicate_source = _make_valid_invoice(
        rng,
        selected_profiles[5],
        purchase_orders[5],
        po_blueprints[5]["amount_in_inr"],
        sequence=7,
    )
    invoices.append(primary_duplicate_source)
    matches[primary_duplicate_source["invoice_id"]] = purchase_orders[5]["po_id"]

    duplicate_invoice = {
        "invoice_id": _make_invoice_id(rng, selected_profiles[5]["invoice_prefix"], 8),
        "vendor": primary_duplicate_source["vendor"],
        "amount": primary_duplicate_source["amount"],
        "currency": primary_duplicate_source["currency"],
        "po_reference": primary_duplicate_source["po_reference"],
        "external_invoice_number": primary_duplicate_source["external_invoice_number"],
    }
    invoices.append(duplicate_invoice)
    duplicate_invoices.append(duplicate_invoice["invoice_id"])

    return {
        "task_id": "po_reconcile_hard",
        "variant_id": f"hard_seeded_{episode_index:04d}",
        "difficulty": "hard",
        "type": "reconciliation",
        "max_steps": 18,
        "instructions": (
            "Reconcile invoices to purchase orders. Normalize vendor aliases, use FX-adjusted amounts, allow "
            "valid matches within +/-2%, flag unmatched invoices, flag amount mismatches, detect duplicate "
            "invoices, and flag invoices that reference multiple purchase orders."
        ),
        "document": {
            "purchase_orders": purchase_orders,
            "invoices": invoices,
            "required_outputs": [
                "matches",
                "flag_unmatched_invoice",
                "flag_amount_discrepancy",
                "flag_duplicate_invoice",
                "flag_split_po",
            ],
            "known_issue_catalog": [
                "amount_mismatch",
                "duplicate_invoice",
                "split_po",
            ],
            "matching_policy": {
                "amount_tolerance_percent": 2.0,
                "base_currency": "INR",
                "fx_rates_to_inr": dict(_FX_TO_INR),
                "notes": [
                    "Use PO references when present and trustworthy.",
                    "When invoice and PO currencies differ, compare after FX conversion to INR.",
                    "An invoice that references multiple PO references should be flagged as split_po.",
                ],
            },
            "vendor_directory": [
                {
                    "canonical_vendor": po["vendor"],
                    "aliases": list(po.get("aliases", [])),
                }
                for po in purchase_orders
            ],
        },
        "ground_truth": {
            "matches": matches,
            "unmatched_invoices": unmatched_invoices,
            "discrepancies": discrepancies,
            "duplicate_invoices": duplicate_invoices,
        },
    }


_TASK_BUILDERS = {
    "easy": _build_easy_task,
    "medium": _build_medium_task,
    "hard": _build_hard_task,
}


def generate_task(task_key: str, episode_index: int = 0) -> Dict[str, Any]:
    if task_key not in _TASK_BUILDERS:
        raise ValueError(f"Unknown difficulty '{task_key}'. Expected one of {list(_TASK_BUILDERS)}")

    task = _TASK_BUILDERS[task_key](episode_index)
    task["episode_index"] = episode_index
    task["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return task
