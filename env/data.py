from __future__ import annotations

TASKS = {
    "easy": {
        "task_id": "invoice_extract_easy",
        "difficulty": "easy",
        "type": "extraction",
        "max_steps": 8,
        "instructions": (
            "Extract the required fields from the invoice text, then submit when you are confident."
        ),
        "document": {
            "invoice_text": (
                "Vendor: Sunrise Office Supplies Pvt Ltd\n"
                "Invoice Number: INV-2026-0142\n"
                "Invoice Date: 2026-03-14\n"
                "Currency: INR\n"
                "Subtotal: 11850.00\n"
                "Tax: 2133.00\n"
                "Total Amount: 13983.00\n"
                "Bill To: Artha Retail LLP\n"
            ),
            "required_fields": [
                "vendor_name",
                "invoice_number",
                "invoice_date",
                "currency",
                "total_amount",
            ],
        },
        "ground_truth": {
            "vendor_name": "Sunrise Office Supplies Pvt Ltd",
            "invoice_number": "INV-2026-0142",
            "invoice_date": "2026-03-14",
            "currency": "INR",
            "total_amount": "13983.00",
        },
    },
    "medium": {
        "task_id": "invoice_validate_medium",
        "difficulty": "medium",
        "type": "validation",
        "max_steps": 10,
        "instructions": (
            "Review the invoice and flag every genuine anomaly. False positives reduce score."
        ),
        "document": {
            "invoice_text": (
                "Vendor: Northwind Industrial Services\n"
                "Invoice Number: NIS-8831\n"
                "Invoice Date: 31/02/2026\n"
                "Currency: INR\n"
                "Line Items:\n"
                "- Machine Parts A x2 @ 4500.00 = 9000.00\n"
                "- Machine Parts A x2 @ 4500.00 = 9000.00\n"
                "- Site Visit Fee x1 @ 2500.00 = 2500.00\n"
                "Subtotal: 21500.00\n"
                "Tax: 3690.00\n"
                "Total Amount: 23000.00\n"
                "GSTIN: MISSING\n"
            ),
            "known_issue_catalog": [
                "invalid_invoice_date",
                "duplicate_line_item",
                "subtotal_mismatch",
                "missing_gstin",
            ],
        },
        "ground_truth": {
            "issues": [
                "invalid_invoice_date",
                "duplicate_line_item",
                "subtotal_mismatch",
                "missing_gstin",
            ]
        },
    },
    "hard": {
        "task_id": "po_reconcile_hard",
        "difficulty": "hard",
        "type": "reconciliation",
        "max_steps": 14,
        "instructions": (
            "Match invoices to purchase orders, identify any unmatched invoice, then submit."
        ),
        "document": {
            "purchase_orders": [
                {"po_id": "PO-9001", "vendor": "Apex Components", "amount": "12500.00"},
                {"po_id": "PO-9002", "vendor": "Metro Cables", "amount": "8300.00"},
                {"po_id": "PO-9003", "vendor": "Zenith Packaging", "amount": "15400.00"},
            ],
            "invoices": [
                {"invoice_id": "INV-A1", "vendor": "Apex Components", "amount": "12500.00"},
                {"invoice_id": "INV-M2", "vendor": "Metro Cables", "amount": "8200.00"},
                {"invoice_id": "INV-Z3", "vendor": "Zenith Packaging", "amount": "15400.00"},
                {"invoice_id": "INV-X9", "vendor": "Orbit Stationery", "amount": "2100.00"},
            ],
            "required_outputs": [
                "matches",
                "flag_unmatched_invoice",
                "flag_amount_discrepancy",
            ],
            "known_issue_catalog": [
                "amount_mismatch",
            ],
        },
        "ground_truth": {
            "matches": {
                "INV-A1": "PO-9001",
                "INV-M2": "PO-9002",
                "INV-Z3": "PO-9003",
            },
            "unmatched_invoices": ["INV-X9"],
            "discrepancies": {
                "INV-M2": "-100.00",
            },
        },
    },
}
