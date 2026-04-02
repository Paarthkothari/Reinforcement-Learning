from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

TASK_KEYS = ("easy", "medium", "hard")


def _easy_variant(
    *,
    variant_id: str,
    vendor_name: str,
    vendor_alias: str,
    invoice_number: str,
    invoice_date: str,
    currency: str,
    subtotal: str,
    tax: str,
    total_amount: str,
    bill_to: str,
    labels: Dict[str, str],
    noise_hints: List[str],
) -> Dict[str, Any]:
    invoice_text = (
        f"{labels['vendor']}: {vendor_alias}\n"
        f"{labels['invoice_number']}: {invoice_number}\n"
        f"{labels['invoice_date']}: {invoice_date}\n"
        f"{labels['currency']}: {currency}\n"
        f"{labels['subtotal']}: {subtotal}\n"
        f"{labels['tax']}: {tax}\n"
        f"{labels['total_amount']}: {total_amount}\n"
        f"{labels['bill_to']}: {bill_to}\n"
    )
    return {
        "task_id": "invoice_extract_easy",
        "variant_id": variant_id,
        "difficulty": "easy",
        "type": "extraction",
        "max_steps": 8,
        "instructions": (
            "Extract the required fields from the invoice text. Documents may contain label aliases "
            "or mild formatting noise similar to OCR cleanup."
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
                "vendor_name": [labels["vendor"], "Vendor", "Supplier", "Vendor Name"],
                "invoice_number": [labels["invoice_number"], "Invoice Number", "Invoice No", "Inv No"],
                "invoice_date": [labels["invoice_date"], "Invoice Date", "Date"],
                "currency": [labels["currency"], "Currency", "Curr."],
                "total_amount": [labels["total_amount"], "Total Amount", "Grand Total", "Total Due"],
            },
            "normalization_hints": {
                "vendor_aliases": {
                    vendor_alias: vendor_name,
                }
            },
            "noise_hints": noise_hints,
        },
        "ground_truth": {
            "vendor_name": vendor_name,
            "invoice_number": invoice_number,
            "invoice_date": invoice_date,
            "currency": currency,
            "total_amount": total_amount,
        },
    }


def _medium_variant(
    *,
    variant_id: str,
    vendor_name: str,
    invoice_number: str,
    invoice_date: str,
    currency: str,
    lines: List[str],
    subtotal: str,
    tax: str,
    total_amount: str,
    gstin: str,
    issues: List[str],
    review_notes: List[str],
) -> Dict[str, Any]:
    invoice_text = (
        f"Vendor: {vendor_name}\n"
        f"Invoice Number: {invoice_number}\n"
        f"Invoice Date: {invoice_date}\n"
        f"Currency: {currency}\n"
        "Line Items:\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n"
        f"Subtotal: {subtotal}\n"
        f"Tax: {tax}\n"
        f"Total Amount: {total_amount}\n"
        f"GSTIN: {gstin}\n"
    )
    return {
        "task_id": "invoice_validate_medium",
        "variant_id": variant_id,
        "difficulty": "medium",
        "type": "validation",
        "max_steps": 10,
        "instructions": (
            "Review the invoice and flag every genuine anomaly. False positives reduce score."
        ),
        "document": {
            "invoice_text": invoice_text,
            "known_issue_catalog": [
                "invalid_invoice_date",
                "duplicate_line_item",
                "subtotal_mismatch",
                "missing_gstin",
            ],
            "review_notes": review_notes,
        },
        "ground_truth": {
            "issues": issues,
        },
    }


def _hard_variant(
    *,
    variant_id: str,
    purchase_orders: List[Dict[str, Any]],
    invoices: List[Dict[str, Any]],
    matches: Dict[str, str],
    unmatched_invoices: List[str],
    discrepancies: Dict[str, str],
    duplicate_invoices: List[str],
    instructions: str,
) -> Dict[str, Any]:
    return {
        "task_id": "po_reconcile_hard",
        "variant_id": variant_id,
        "difficulty": "hard",
        "type": "reconciliation",
        "max_steps": 18,
        "instructions": instructions,
        "document": {
            "purchase_orders": purchase_orders,
            "invoices": invoices,
            "required_outputs": [
                "matches",
                "flag_unmatched_invoice",
                "flag_amount_discrepancy",
            ],
            "known_issue_catalog": [
                "amount_mismatch",
                "duplicate_invoice",
            ],
            "matching_policy": {
                "amount_tolerance": "50.00",
                "notes": [
                    "Use PO references when present and trustworthy.",
                    "Normalize vendor aliases before flagging unmatched invoices.",
                    "A single PO may be fulfilled by multiple invoices.",
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


TASK_VARIANTS = {
    "easy": [
        _easy_variant(
            variant_id="easy_clean_alias",
            vendor_name="Sunrise Office Supplies Pvt Ltd",
            vendor_alias="Sunrise Office Supplies Pvt. Ltd.",
            invoice_number="INV-2026-0142",
            invoice_date="2026-03-14",
            currency="INR",
            subtotal="11850.00",
            tax="2133.00",
            total_amount="13983.00",
            bill_to="Artha Retail LLP",
            labels={
                "vendor": "Supplier",
                "invoice_number": "Invoice No",
                "invoice_date": "Invoice Date",
                "currency": "Currency",
                "subtotal": "Subtotal",
                "tax": "Tax",
                "total_amount": "Grand Total",
                "bill_to": "Bill To",
            },
            noise_hints=["Vendor label uses a supplier synonym.", "Total amount label uses Grand Total."],
        ),
        _easy_variant(
            variant_id="easy_typography_noise",
            vendor_name="North Ridge Industrial Services LLP",
            vendor_alias="North Ridge Industrial Services LLP",
            invoice_number="NR-8831-A",
            invoice_date="2026-02-18",
            currency="USD",
            subtotal="4800.00",
            tax="864.00",
            total_amount="5664.00",
            bill_to="Helios Manufacturing Inc",
            labels={
                "vendor": "Vendor Name",
                "invoice_number": "Inv No",
                "invoice_date": "Date",
                "currency": "Curr.",
                "subtotal": "Subtotal",
                "tax": "Tax",
                "total_amount": "Total Due",
                "bill_to": "Bill To",
            },
            noise_hints=["Invoice number label is abbreviated.", "Currency label appears as Curr."],
        ),
        _easy_variant(
            variant_id="easy_shared_services",
            vendor_name="BluePeak Shared Services Private Limited",
            vendor_alias="BluePeak Shared Services Pvt Ltd",
            invoice_number="BP-1147",
            invoice_date="2026-01-29",
            currency="EUR",
            subtotal="7250.00",
            tax="1305.00",
            total_amount="8555.00",
            bill_to="Cedar & Oak Commerce GmbH",
            labels={
                "vendor": "Vendor",
                "invoice_number": "Invoice Number",
                "invoice_date": "Invoice Date",
                "currency": "Currency",
                "subtotal": "Subtotal",
                "tax": "Tax",
                "total_amount": "Total Amount",
                "bill_to": "Billed To",
            },
            noise_hints=["Vendor display name differs slightly from the legal entity name."],
        ),
    ],
    "medium": [
        _medium_variant(
            variant_id="medium_duplicate_and_math",
            vendor_name="Northwind Industrial Services",
            invoice_number="NIS-8831",
            invoice_date="31/02/2026",
            currency="INR",
            lines=[
                "Machine Parts A x2 @ 4500.00 = 9000.00",
                "Machine Parts A x2 @ 4500.00 = 9000.00",
                "Site Visit Fee x1 @ 2500.00 = 2500.00",
            ],
            subtotal="21500.00",
            tax="3690.00",
            total_amount="23000.00",
            gstin="MISSING",
            issues=[
                "invalid_invoice_date",
                "duplicate_line_item",
                "subtotal_mismatch",
                "missing_gstin",
            ],
            review_notes=["Baseline anomaly case with duplicate line items and bad totals."],
        ),
        _medium_variant(
            variant_id="medium_ocr_date_noise",
            vendor_name="Crescent Field Logistics",
            invoice_number="CFL-1180",
            invoice_date="2026/14/03",
            currency="USD",
            lines=[
                "Warehouse Transfer x3 @ 1200.00 = 3600.00",
                "Fuel Surcharge x1 @ 450.00 = 450.00",
                "Fuel Surcharge x1 @ 450.00 = 450.00",
            ],
            subtotal="5000.00",
            tax="400.00",
            total_amount="5400.00",
            gstin="MISSING",
            issues=[
                "invalid_invoice_date",
                "duplicate_line_item",
                "subtotal_mismatch",
                "missing_gstin",
            ],
            review_notes=["Date has OCR-style delimiter noise and invalid ordering."],
        ),
        _medium_variant(
            variant_id="medium_compliance_gap",
            vendor_name="Helio Maintenance Co.",
            invoice_number="HM-4402",
            invoice_date="29-13-2026",
            currency="EUR",
            lines=[
                "Preventive Maintenance x1 @ 1800.00 = 1800.00",
                "Calibration Charge x1 @ 700.00 = 700.00",
                "Calibration Charge x1 @ 700.00 = 700.00",
            ],
            subtotal="3500.00",
            tax="630.00",
            total_amount="4130.00",
            gstin="MISSING",
            issues=[
                "invalid_invoice_date",
                "duplicate_line_item",
                "subtotal_mismatch",
                "missing_gstin",
            ],
            review_notes=["Invalid month plus hidden duplicate service fee."],
        ),
    ],
    "hard": [
        _hard_variant(
            variant_id="hard_vendor_aliases",
            purchase_orders=[
                {"po_id": "PO-9001", "vendor": "Apex Components", "aliases": ["Apex Components Pvt Ltd", "APEX Components"], "amount": "12500.00", "po_reference": "APX-01"},
                {"po_id": "PO-9002", "vendor": "Metro Cables", "aliases": ["Metro Cables Ltd", "Metro Cable Works"], "amount": "8300.00", "po_reference": "MET-77"},
                {"po_id": "PO-9003", "vendor": "Zenith Packaging", "aliases": ["Zenith Packaging Co", "Zenith Pack"], "amount": "15400.00", "po_reference": "ZEN-20"},
                {"po_id": "PO-9004", "vendor": "North Harbor Freight", "aliases": ["North Harbour Freight", "NH Freight"], "amount": "6400.00", "po_reference": "NHF-10"},
                {"po_id": "PO-9005", "vendor": "BluePeak Shared Services", "aliases": ["BluePeak Shared Services Pvt Ltd", "BluePeak Services"], "amount": "9200.00", "po_reference": "BPS-55"},
            ],
            invoices=[
                {"invoice_id": "INV-A1", "vendor": "Apex Components Pvt Ltd", "amount": "12500.00", "po_reference": "APX-01"},
                {"invoice_id": "INV-M2", "vendor": "Metro Cable Works", "amount": "8225.00", "po_reference": "MET-77"},
                {"invoice_id": "INV-Z3", "vendor": "Zenith Pack", "amount": "15400.00", "po_reference": "ZEN-20"},
                {"invoice_id": "INV-N4", "vendor": "North Harbour Freight", "amount": "6400.00", "po_reference": ""},
                {"invoice_id": "INV-B5", "vendor": "BluePeak Services", "amount": "4600.00", "po_reference": "BPS-55"},
                {"invoice_id": "INV-B6", "vendor": "BluePeak Shared Services Pvt Ltd", "amount": "4600.00", "po_reference": "BPS-55"},
                {"invoice_id": "INV-X9", "vendor": "Orbit Stationery", "amount": "2100.00", "po_reference": ""},
            ],
            matches={
                "INV-A1": "PO-9001",
                "INV-M2": "PO-9002",
                "INV-Z3": "PO-9003",
                "INV-N4": "PO-9004",
                "INV-B5": "PO-9005",
                "INV-B6": "PO-9005",
            },
            unmatched_invoices=["INV-X9"],
            discrepancies={"INV-M2": "-75.00"},
            duplicate_invoices=[],
            instructions=(
                "Reconcile all invoices against the purchase orders. Use PO references when available, "
                "normalize vendor aliases, allow split invoices against the same PO, flag any unmatched "
                "invoice, and flag amount mismatches that exceed the stated tolerance."
            ),
        ),
        _hard_variant(
            variant_id="hard_conflicting_signals",
            purchase_orders=[
                {"po_id": "PO-8110", "vendor": "Helios Fabrication", "aliases": ["Helios Fabrication Ltd", "Helios Fab"], "amount": "11000.00", "po_reference": "HEL-10"},
                {"po_id": "PO-8111", "vendor": "Helios Fabricators", "aliases": ["Helios Fabricators Pvt Ltd", "Helios Fab Works"], "amount": "10850.00", "po_reference": "HEL-11"},
                {"po_id": "PO-8112", "vendor": "Cedar Logistics", "aliases": ["Cedar Logistics LLC", "Cedar Logix"], "amount": "5900.00", "po_reference": "CED-31"},
                {"po_id": "PO-8113", "vendor": "Atlas Controls", "aliases": ["Atlas Controls Inc", "Atlas Control Systems"], "amount": "7600.00", "po_reference": "ATL-12"},
                {"po_id": "PO-8114", "vendor": "Riverfront Packaging", "aliases": ["Riverfront Packaging Co", "Riverfront Pack"], "amount": "4400.00", "po_reference": "RIV-09"},
            ],
            invoices=[
                {"invoice_id": "INV-H1", "vendor": "Helios Fab", "amount": "11000.00", "po_reference": "HEL-10"},
                {"invoice_id": "INV-H2", "vendor": "Helios Fab Works", "amount": "10940.00", "po_reference": "HEL-11"},
                {"invoice_id": "INV-C3", "vendor": "Cedar Logix", "amount": "5900.00", "po_reference": ""},
                {"invoice_id": "INV-A4", "vendor": "Atlas Control Systems", "amount": "7600.00", "po_reference": "ATL-12"},
                {"invoice_id": "INV-R5", "vendor": "Riverfront Pack", "amount": "2200.00", "po_reference": "RIV-09"},
                {"invoice_id": "INV-R6", "vendor": "Riverfront Packaging Co", "amount": "2200.00", "po_reference": "RIV-09"},
                {"invoice_id": "INV-U7", "vendor": "Urban Office Mart", "amount": "1800.00", "po_reference": ""},
            ],
            matches={
                "INV-H1": "PO-8110",
                "INV-H2": "PO-8111",
                "INV-C3": "PO-8112",
                "INV-A4": "PO-8113",
                "INV-R5": "PO-8114",
                "INV-R6": "PO-8114",
            },
            unmatched_invoices=["INV-U7"],
            discrepancies={"INV-H2": "90.00"},
            duplicate_invoices=[],
            instructions=(
                "Reconcile invoices where vendor names are deliberately similar. Prefer exact PO references, "
                "fall back to vendor alias normalization when the reference is absent, and identify invoices "
                "whose amounts fall outside the allowed tolerance."
            ),
        ),
        _hard_variant(
            variant_id="hard_partial_payments_duplicates",
            purchase_orders=[
                {"po_id": "PO-7210", "vendor": "Maple Industrial Equipments", "aliases": ["Maple Industrial Equipment", "Maple Ind Equip"], "amount": "15000.00", "po_reference": "MAP-10"},
                {"po_id": "PO-7211", "vendor": "Oakline Maintenance", "aliases": ["Oakline Maint.", "Oakline Services"], "amount": "6400.00", "po_reference": "OAK-11"},
                {"po_id": "PO-7212", "vendor": "Silverline Packaging", "aliases": ["Silverline Pack", "Silverline Packaging Co"], "amount": "5200.00", "po_reference": "SIL-12"},
                {"po_id": "PO-7213", "vendor": "Vertex Cooling Systems", "aliases": ["Vertex Cooling", "Vertex Cool Systems"], "amount": "8800.00", "po_reference": "VER-13"},
            ],
            invoices=[
                {"invoice_id": "INV-MA1", "vendor": "Maple Ind Equip", "amount": "7500.00", "po_reference": "MAP-10", "external_invoice_number": "MAP-4431"},
                {"invoice_id": "INV-MA2", "vendor": "Maple Industrial Equipment", "amount": "7500.00", "po_reference": "MAP-10", "external_invoice_number": "MAP-4432"},
                {"invoice_id": "INV-OA3", "vendor": "Oakline Services", "amount": "6400.00", "po_reference": "WRONG-OAK", "external_invoice_number": "OAK-2001"},
                {"invoice_id": "INV-SI4", "vendor": "Silverline Pack", "amount": "5200.00", "po_reference": "SIL-12", "external_invoice_number": "SIL-9001"},
                {"invoice_id": "INV-SI5", "vendor": "Silverline Packaging Co", "amount": "5200.00", "po_reference": "SIL-12", "external_invoice_number": "SIL-9001"},
                {"invoice_id": "INV-VE6", "vendor": "Vertex Cooling", "amount": "9055.00", "po_reference": "VER-13", "external_invoice_number": "VER-3300"},
                {"invoice_id": "INV-UX7", "vendor": "Urban Spare Supplies", "amount": "1900.00", "po_reference": "", "external_invoice_number": "URB-118"},
            ],
            matches={
                "INV-MA1": "PO-7210",
                "INV-MA2": "PO-7210",
                "INV-OA3": "PO-7211",
                "INV-SI4": "PO-7212",
                "INV-VE6": "PO-7213",
            },
            unmatched_invoices=["INV-UX7"],
            discrepancies={"INV-VE6": "255.00"},
            duplicate_invoices=["INV-SI5"],
            instructions=(
                "Reconcile invoices with partial payments, one incorrect PO reference, and one duplicate "
                "invoice submission. Use vendor normalization to recover from bad references, avoid matching "
                "duplicates, and only flag amount mismatches when the invoiced total exceeds tolerance."
            ),
        ),
    ],
}


def generate_task(task_key: str, episode_index: int = 0) -> Dict[str, Any]:
    if task_key not in TASK_VARIANTS:
        raise ValueError(f"Unknown difficulty '{task_key}'. Expected one of {list(TASK_VARIANTS)}")

    task = deepcopy(TASK_VARIANTS[task_key][episode_index % len(TASK_VARIANTS[task_key])])
    task["episode_index"] = episode_index
    task["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return task
