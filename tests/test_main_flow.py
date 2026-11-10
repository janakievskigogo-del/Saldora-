"""Tests for CLI orchestration and the human-in-the-loop collections gate."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

import main
from config import Settings
from data_handler import InvoiceStore
from legacy_models import Invoice


def test_create_flow_persists_invoice_and_returns_created_flag(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InvoiceStore(settings.invoice_store_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": "Y")
    prompt = (
        "Create an invoice for client Flow Client LLC, tax ID MK4030000000000, "
        "amount 1000 EUR plus VAT 18%, due 2026-11-15, language English, "
        "email billing@flow.example"
    )

    code, created = main._create_invoice_flow(
        settings,
        store,
        prompt_text=prompt,
    )

    assert code == 0
    assert created is True
    invoices = store.load_all()
    assert len(invoices) == 1
    assert invoices[0].client_name == "Flow Client LLC"
    assert invoices[0].ujp_status == "MOCK_SUBMITTED"
    assert invoices[0].ujp_reference.startswith("MOCK-UJP-")
    assert invoices[0].ujp_xml_path is not None
    assert Path(invoices[0].ujp_xml_path).parent == settings.ubl_output_dir


def test_create_flow_cancel_does_not_persist(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InvoiceStore(settings.invoice_store_path)
    monkeypatch.setattr("builtins.input", lambda prompt="": "S")
    prompt = (
        "Create an invoice for client Cancel Client, tax ID MK4030000000000, "
        "amount 100 EUR, due 2026-11-15."
    )

    code, created = main._create_invoice_flow(settings, store, prompt_text=prompt)

    assert code == 0
    assert created is False
    assert store.load_all() == []


def test_collections_require_explicit_approval(
    settings: Settings,
    invoice: Invoice,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_of = date(2026, 10, 30)
    store = InvoiceStore(settings.invoice_store_path)
    store.upsert(replace(invoice, due_date=as_of - timedelta(days=5)))
    settings = replace(settings, groq_api_key="test-key")

    def fake_draft(self: object, client_data: object, as_of: date | None = None) -> str:
        return "Dear client, this is a polite reminder about invoice TST-2026-0001."

    monkeypatch.setattr(main.ToneAgent, "generate_draft", fake_draft)
    monkeypatch.setattr("builtins.input", lambda prompt="": "Y")

    code = main._run_collections(settings, store, as_of=as_of)

    assert code == 0
    updated = store.get("TST-2026-0001")
    assert updated is not None
    assert updated.reminder_count == 1
    assert updated.last_reminded_at is not None
    assert updated.status == "REMINDER_SENT"
    output = capsys.readouterr().out
    assert "DRAFT 1/1 - HUMAN APPROVAL REQUIRED" in output
    assert "EMAIL MESSAGE SENT TO billing@acme.com" in output
    assert "Sent (mock): 1" in output


def test_collections_skip_does_not_increment_reminder(
    settings: Settings,
    invoice: Invoice,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_of = date(2026, 10, 30)
    store = InvoiceStore(settings.invoice_store_path)
    store.upsert(replace(invoice, due_date=as_of - timedelta(days=5)))
    settings = replace(settings, groq_api_key="test-key")

    monkeypatch.setattr(
        main.ToneAgent,
        "generate_draft",
        lambda self, client_data, as_of=None: (
            "Dear client, this is a polite reminder about the invoice."
        ),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "S")

    code = main._run_collections(settings, store, as_of=as_of)

    assert code == 0
    updated = store.get("TST-2026-0001")
    assert updated is not None
    assert updated.reminder_count == 0
    output = capsys.readouterr().out
    assert "Skipped:     1" in output


def test_collections_without_key_fails_before_prompting(
    settings: Settings,
    invoice: Invoice,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    as_of = date(2026, 10, 30)
    store = InvoiceStore(settings.invoice_store_path)
    store.upsert(replace(invoice, due_date=as_of - timedelta(days=5)))

    code = main._run_collections(settings, store, as_of=as_of)

    assert code == 1
    captured = capsys.readouterr()
    assert "GROQ_API_KEY" in captured.err


def test_ubl_export_saves_validated_xml(
    settings: Settings,
    invoice: Invoice,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = InvoiceStore(settings.invoice_store_path)
    store.upsert(invoice)
    output_dir = tmp_path / "output"

    code = main._run_ubl_export(
        settings,
        store,
        invoice.invoice_id,
        output_dir=str(output_dir),
        show_xml=True,
    )

    assert code == 0
    xml_path = output_dir / f"{invoice.invoice_id}.xml"
    assert xml_path.exists()
    content = xml_path.read_text(encoding="utf-8")
    assert "<cbc:UBLVersionID>2.1</cbc:UBLVersionID>" in content
    output = capsys.readouterr().out
    assert "UBL 2.1 INVOICE XML" in output
    assert "Validation:  OK" in output
    assert "<cbc:Invoice" in output
