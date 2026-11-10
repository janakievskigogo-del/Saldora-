"""Tests for JSON persistence and Excel import/export."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from data_handler import (
    InvoiceDataError,
    InvoiceStore,
    export_invoices_to_excel,
    load_invoices_from_excel,
    load_overdue_invoices,
)
from legacy_models import Invoice


def test_json_store_roundtrip(tmp_path: Path, invoice: Invoice) -> None:
    store = InvoiceStore(tmp_path / "invoices.json")
    store.ensure_initialized()
    assert store.path.exists()
    assert store.load_all() == []

    store.upsert(invoice)
    loaded = store.load_all()
    assert len(loaded) == 1
    stored = loaded[0]
    assert stored.invoice_id == invoice.invoice_id
    assert stored.client_name == invoice.client_name
    assert stored.client_tax_id == invoice.client_tax_id
    assert stored.net_amount == Decimal("1000.00")
    assert stored.vat_amount == Decimal("180.00")
    assert stored.total_amount == Decimal("1180.00")
    assert stored.due_date == date(2026, 10, 16)
    assert stored.language.value == "EN"
    assert stored.channel.value == "Email"

    assert store.next_invoice_number(prefix="TST", year=2026) == "TST-2026-0002"


def test_json_store_mark_paid_and_overdue(tmp_path: Path, invoice: Invoice) -> None:
    store = InvoiceStore(tmp_path / "invoices.json")
    store.upsert(invoice)

    assert [item.invoice_id for item in store.overdue(date(2026, 10, 20))] == [
        invoice.invoice_id
    ]
    paid = store.mark_paid(invoice.invoice_id)
    assert paid.status == "PAID"
    assert store.overdue(date(2026, 10, 20)) == []


def test_json_store_mark_reminder_sent_persists_status(tmp_path: Path, invoice: Invoice) -> None:
    store = InvoiceStore(tmp_path / "invoices.json")
    store.upsert(invoice)
    sent_at = datetime(2026, 10, 30, 12, 0)

    updated = store.mark_reminder_sent(invoice.invoice_id, sent_at=sent_at)

    assert updated.status == "REMINDER_SENT"
    assert updated.reminder_count == 1
    assert updated.last_reminded_at == sent_at
    assert store.overdue(date(2026, 10, 30)) == [updated]

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    stored = payload["invoices"][0]
    assert stored["status"] == "REMINDER_SENT"
    assert stored["last_reminded_at"] == sent_at.isoformat()
    assert stored["reminder_count"] == 1


def test_json_store_rejects_duplicate_ids(tmp_path: Path, invoice: Invoice) -> None:
    store = InvoiceStore(tmp_path / "invoices.json")
    store.save_all([invoice, invoice])
    with pytest.raises(InvoiceDataError, match="Duplicate Invoice ID"):
        store.load_all()


def test_json_store_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "invoices.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(InvoiceDataError, match="not valid JSON"):
        InvoiceStore(path).load_all()


def test_json_store_supports_legacy_list_shape(tmp_path: Path, invoice: Invoice) -> None:
    path = tmp_path / "invoices.json"
    path.write_text(json.dumps([invoice.to_dict()], ensure_ascii=False), encoding="utf-8")
    loaded = InvoiceStore(path).load_all()
    assert len(loaded) == 1
    assert loaded[0].invoice_id == invoice.invoice_id


def test_excel_roundtrip(tmp_path: Path, invoice: Invoice) -> None:
    invoice = replace(
        invoice,
        ujp_status="MOCK_SUBMITTED",
        ujp_reference="MOCK-UJP-TEST",
        ujp_xml_path="data/ujp_xml/TST-2026-0001.xml",
        reminder_count=2,
        last_reminded_at=datetime(2026, 10, 20, 9, 30),
    )
    path = tmp_path / "invoices.xlsx"
    export_invoices_to_excel([invoice], path)
    assert path.exists()

    result = load_invoices_from_excel(path)
    assert result.warnings == []
    assert len(result.invoices) == 1
    imported = result.invoices[0]
    assert imported.invoice_id == invoice.invoice_id
    assert imported.client_name == invoice.client_name
    assert imported.client_tax_id == invoice.client_tax_id
    assert imported.total_amount == Decimal("1180.00")
    assert imported.vat_amount == Decimal("180.00")
    assert imported.vat_rate == Decimal("18.00")
    assert imported.due_date == date(2026, 10, 16)
    assert imported.status == "UNPAID"
    assert imported.language.value == "EN"
    assert imported.channel.value == "Email"
    assert imported.ujp_status == "MOCK_SUBMITTED"
    assert imported.ujp_reference == "MOCK-UJP-TEST"
    assert imported.ujp_xml_path == "data/ujp_xml/TST-2026-0001.xml"
    assert imported.reminder_count == 2
    assert imported.last_reminded_at == datetime(2026, 10, 20, 9, 30)


def test_legacy_excel_with_amount_only(tmp_path: Path) -> None:
    path = tmp_path / "legacy.xlsx"
    frame = pd.DataFrame(
        [
            {
                "Invoice Number": "OLD-001",
                "Client": "Legacy Client",
                "Amount": 500,
                "Due Date": "2026-01-15",
                "Status": "UNPAID",
            }
        ]
    )
    frame.to_excel(path, index=False, engine="openpyxl")

    result = load_invoices_from_excel(path)
    assert len(result.invoices) == 1
    imported = result.invoices[0]
    assert imported.invoice_id == "OLD-001"
    assert imported.total_amount == Decimal("500.00")
    assert imported.net_amount == Decimal("500.00")
    assert imported.vat_amount == Decimal("0.00")
    assert imported.vat_rate == Decimal("0.00")


def test_excel_import_reports_and_skips_invalid_rows(tmp_path: Path) -> None:
    path = tmp_path / "mixed.xlsx"
    frame = pd.DataFrame(
        [
            {
                "Invoice ID": "OK-001",
                "Client Name": "Good Client",
                "Total Amount": 100,
                "Due Date": "2026-01-15",
                "Status": "UNPAID",
            },
            {
                "Invoice ID": "BAD-001",
                "Client Name": "Bad Client",
                "Total Amount": 200,
                "Due Date": "not-a-date",
                "Status": "UNPAID",
            },
        ]
    )
    frame.to_excel(path, index=False, engine="openpyxl")

    result = load_invoices_from_excel(path)
    assert len(result.invoices) == 1
    assert result.invoices[0].invoice_id == "OK-001"
    assert any("invalid due date" in warning for warning in result.warnings)


def test_excel_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.xlsx"
    pd.DataFrame([{"Invoice ID": "X-1"}]).to_excel(path, index=False, engine="openpyxl")
    with pytest.raises(InvoiceDataError, match="missing required column"):
        load_invoices_from_excel(path)


def test_store_import_merge(tmp_path: Path, invoice: Invoice) -> None:
    store = InvoiceStore(tmp_path / "invoices.json")
    store.upsert(invoice)

    workbook = tmp_path / "extra.xlsx"
    pd.DataFrame(
        [
            {
                "Invoice ID": "TST-2026-0002",
                "Client Name": "Second Client",
                "Total Amount": 250,
                "Due Date": "2026-02-01",
                "Status": "UNPAID",
            }
        ]
    ).to_excel(workbook, index=False, engine="openpyxl")

    result = store.import_excel(workbook)
    assert len(result.invoices) == 1
    assert len(store.load_all()) == 2


def test_load_overdue_invoices_returns_legacy_context(tmp_path: Path, invoice: Invoice) -> None:
    path = tmp_path / "overdue.xlsx"
    export_invoices_to_excel([invoice], path)

    rows = load_overdue_invoices(path, as_of=date(2026, 10, 20))
    assert len(rows) == 1
    row = rows[0]
    assert row["invoice_id"] == invoice.invoice_id
    assert row["client_name"] == invoice.client_name
    assert row["amount"] == "1180.00"
    assert row["days_late"] == 4
    assert row["preferred_language"] == "EN"
    assert row["channel"] == "Email"
    assert row["tone_tier"] == "friendly"


def test_export_creates_parent_directory(tmp_path: Path, invoice: Invoice) -> None:
    path = tmp_path / "nested" / "out" / "invoices.xlsx"
    exported = export_invoices_to_excel([invoice], path)
    assert exported == path
    assert path.exists()
