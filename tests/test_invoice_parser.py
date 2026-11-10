"""Tests for multilingual invoice intake extraction."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from config import Settings
from invoice_parser import InvoiceParsingError, parse_invoice_prompt


def test_macedonian_prompt(settings: Settings) -> None:
    prompt = (
        "Направи фактура за клиент Алфа ДООЕЛ со ЕДБ 4030000000000, "
        "износ 10.000,00 денари без ДДВ, рок до 15.11.2026, "
        "јазик македонски, email info@alfa.mk"
    )
    parsed = parse_invoice_prompt(prompt, settings)

    assert parsed.client_name == "Алфа ДООЕЛ"
    assert parsed.client_tax_id == "4030000000000"
    assert parsed.amount == Decimal("10000.00")
    assert parsed.currency == "MKD"
    assert parsed.vat_rate == Decimal("18.00")
    assert parsed.amount_includes_vat is False
    assert parsed.issue_date == settings.today()
    assert parsed.due_date == date(2026, 11, 15)
    assert parsed.language.value == "MK"
    assert parsed.channel.value == "Email"
    assert parsed.client_email == "info@alfa.mk"
    assert parsed.missing_fields == []


def test_albanian_prompt(settings: Settings) -> None:
    prompt = (
        "Krijo faturë për klientin Alba sh.p.k., NIPT L12345678A, "
        "shuma 50.000,00 LEK, afati 20.12.2026, në shqip, email info@alba.al"
    )
    parsed = parse_invoice_prompt(prompt, settings)

    assert parsed.client_name == "Alba sh.p.k."
    assert parsed.client_tax_id == "L12345678A"
    assert parsed.amount == Decimal("50000.00")
    assert parsed.currency == "ALL"
    assert parsed.due_date == date(2026, 12, 20)
    assert parsed.language.value == "AL"
    assert parsed.client_email == "info@alba.al"


def test_english_prompt_with_gross_amount_and_viber(settings: Settings) -> None:
    prompt = (
        "Create an invoice for Acme LLC, amount 1,500.00 EUR including VAT 18%, "
        "due 2026-11-15, language English, viber +38970123456"
    )
    parsed = parse_invoice_prompt(prompt, settings)

    assert parsed.client_name == "Acme LLC"
    assert parsed.amount == Decimal("1500.00")
    assert parsed.currency == "EUR"
    assert parsed.amount_includes_vat is True
    assert parsed.vat_rate == Decimal("18.00")
    assert parsed.due_date == date(2026, 11, 15)
    assert parsed.language.value == "EN"
    assert parsed.channel.value == "Viber"
    assert parsed.client_phone == "+38970123456"
    assert parsed.client_tax_id is None
    assert "client_tax_id" in parsed.missing_fields


def test_issue_date_is_never_reused_as_due_date(settings: Settings) -> None:
    prompt = (
        "Issue date 2026-10-01, client Beta DOO, tax ID 4030000000000, "
        "amount 1000 EUR. Language English."
    )
    parsed = parse_invoice_prompt(prompt, settings)

    assert parsed.issue_date == date(2026, 10, 1)
    assert parsed.due_date == date(2026, 10, 1) + timedelta(
        days=settings.default_payment_terms_days
    )
    assert parsed.due_date != parsed.issue_date
    assert any("payment term" in warning for warning in parsed.warnings)


def test_relative_due_date(settings: Settings) -> None:
    prompt = "Create an invoice for Gamma DOO, amount 500 EUR, due in 30 days."
    parsed = parse_invoice_prompt(prompt, settings)
    assert parsed.due_date == settings.today() + timedelta(days=30)


def test_vat_rate_and_exemption(settings: Settings) -> None:
    reverse_rate = parse_invoice_prompt(
        "Invoice for Delta DOO, amount 1000 EUR, 18% VAT, due 2026-11-15.",
        settings,
    )
    assert reverse_rate.vat_rate == Decimal("18.00")

    exempt = parse_invoice_prompt(
        "Invoice for Epsilon DOO, amount 1000 EUR, VAT exempt, due 2026-11-15.",
        settings,
    )
    assert exempt.vat_rate == Decimal("0.00")
    assert any("exemption" in warning for warning in exempt.warnings)


def test_breakdown(settings: Settings) -> None:
    parsed = parse_invoice_prompt(
        "Invoice for Acme LLC, amount 1180 EUR including VAT 18%, due 2026-11-15.",
        settings,
    )
    assert parsed.breakdown() == (
        Decimal("1000.00"),
        Decimal("180.00"),
        Decimal("1180.00"),
    )


def test_empty_prompt_raises(settings: Settings) -> None:
    with pytest.raises(InvoiceParsingError):
        parse_invoice_prompt("   ", settings)


def test_missing_fields_and_warnings(settings: Settings) -> None:
    parsed = parse_invoice_prompt("Create an invoice for Zeta LLC.", settings)
    assert parsed.client_name == "Zeta LLC"
    assert parsed.client_tax_id is None
    assert parsed.amount is None
    assert parsed.due_date == settings.today() + timedelta(
        days=settings.default_payment_terms_days
    )
    assert "client_tax_id" in parsed.missing_fields
    assert "amount" in parsed.missing_fields
    assert parsed.warnings
