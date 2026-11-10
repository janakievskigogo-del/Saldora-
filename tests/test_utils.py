"""Tests for shared normalization, money, date, and VAT helpers."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from utils import (
    calculate_vat_breakdown,
    extract_response_text,
    format_money,
    is_valid_email,
    is_valid_tax_id,
    normalize_channel,
    normalize_language_code,
    normalize_phone,
    normalize_status,
    normalize_tax_id,
    normalize_whitespace,
    now_utc,
    parse_date,
    parse_decimal,
    safe_filename,
    strip_code_fences,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.234,56", Decimal("1234.56")),
        ("1,234.56", Decimal("1234.56")),
        ("15.000", Decimal("15000.00")),
        ("15,000", Decimal("15000.00")),
        ("1 234,50", Decimal("1234.50")),
        ("MKD 1.200,00", Decimal("1200.00")),
        ("-99.90", Decimal("-99.90")),
        ("0,5", Decimal("0.50")),
        ("18%", Decimal("18.00")),
        (1234, Decimal("1234.00")),
        (1234.567, Decimal("1234.57")),
        (Decimal("1234.567"), Decimal("1234.57")),
    ],
)
def test_parse_decimal_handles_local_and_international_formats(
    raw: object,
    expected: Decimal,
) -> None:
    assert parse_decimal(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "abc", None, True, False])
def test_parse_decimal_rejects_non_numeric_values(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_decimal(raw)


def test_parse_decimal_rejects_nan() -> None:
    with pytest.raises(ValueError):
        parse_decimal(float("nan"))


def test_format_money() -> None:
    assert format_money(Decimal("1234.5"), "MKD") == "1,234.50 MKD"
    assert format_money("1234.5") == "1,234.50"


def test_vat_breakdown_net_input() -> None:
    net, vat, total = calculate_vat_breakdown(Decimal(1000), Decimal(18), False)
    assert (net, vat, total) == (Decimal("1000.00"), Decimal("180.00"), Decimal("1180.00"))


def test_vat_breakdown_gross_input() -> None:
    net, vat, total = calculate_vat_breakdown(Decimal(1180), Decimal(18), True)
    assert (net, vat, total) == (Decimal("1000.00"), Decimal("180.00"), Decimal("1180.00"))
    assert net + vat == total


def test_vat_breakdown_zero_rate() -> None:
    assert calculate_vat_breakdown(Decimal(100), Decimal(0), False) == (
        Decimal("100.00"),
        Decimal("0.00"),
        Decimal("100.00"),
    )


@pytest.mark.parametrize("rate", [Decimal(-1), Decimal(101)])
def test_vat_breakdown_rejects_invalid_rates(rate: Decimal) -> None:
    with pytest.raises(ValueError):
        calculate_vat_breakdown(Decimal(100), rate)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-10-30", date(2026, 10, 30)),
        ("30.10.2026", date(2026, 10, 30)),
        ("30/10/2026", date(2026, 10, 30)),
        ("2026/10/30", date(2026, 10, 30)),
        ("2026-10-30T12:15:00", date(2026, 10, 30)),
        (datetime(2026, 10, 30, 12, 0), date(2026, 10, 30)),
        (date(2026, 10, 30), date(2026, 10, 30)),
        (46325, date(2026, 10, 30)),
        ("46325", date(2026, 10, 30)),
    ],
)
def test_parse_date_formats(raw: object, expected: date) -> None:
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "not-a-date", "2026-99-99", None])
def test_parse_date_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(ValueError):
        parse_date(raw)


def test_normalize_tax_id_and_validation() -> None:
    assert normalize_tax_id("mk 403-000-000-0000") == "MK4030000000000"
    assert is_valid_tax_id("MK4030000000000")
    assert is_valid_tax_id("4030000000000")
    assert is_valid_tax_id("L12345678A")
    assert not is_valid_tax_id("ABC")
    assert not is_valid_tax_id("")
    assert not is_valid_tax_id("ABCDEFGHIJ")


def test_normalize_phone() -> None:
    assert normalize_phone("+389 (70) 123-456") == "+38970123456"
    assert normalize_phone("00389 70 123 456") == "+38970123456"
    assert normalize_phone("") == ""
    assert normalize_phone("++38970123456") == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("billing@example.com", True),
        ("first.last+tag@example.co.uk", True),
        ("not-an-email", False),
        ("a@b", False),
        ("", False),
        (None, False),
    ],
)
def test_is_valid_email(value: object, expected: bool) -> None:
    assert is_valid_email(value) is expected


def test_normalize_language_channel_and_status() -> None:
    assert normalize_language_code("македонски") == "MK"
    assert normalize_language_code("shqip") == "AL"
    assert normalize_language_code("english") == "EN"
    assert normalize_language_code("unknown") == "EN"

    assert normalize_channel("e-mail") == "Email"
    assert normalize_channel("Viber") == "Viber"
    assert normalize_channel("whats app") == "WhatsApp"

    assert normalize_status("платено") == "PAID"
    assert normalize_status("paguar") == "PAID"
    assert normalize_status("неплатено") == "UNPAID"
    assert normalize_status("reminder_sent") == "REMINDER_SENT"
    assert normalize_status("not a status") == "NOT_A_STATUS"


def test_normalize_whitespace_and_safe_filename() -> None:
    assert normalize_whitespace("  Acme \n  LLC\t") == "Acme LLC"
    assert safe_filename("NPL/2026 0001") == "NPL_2026_0001"
    assert safe_filename("////") == "document"


def test_now_utc_is_aware() -> None:
    assert now_utc().tzinfo is not None


def test_strip_code_fences_and_extract_response_text() -> None:
    assert strip_code_fences("```text\nHello\n```") == "Hello"
    response = SimpleNamespace(text="  Hello from Gemini  ")
    assert extract_response_text(response) == "Hello from Gemini"
    openai_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="  Hello from Groq  "))
        ]
    )
    assert extract_response_text(openai_response) == "Hello from Groq"
    blocked = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="Fallback")])
            )
        ],
    )
    assert extract_response_text(blocked) == "Fallback"

    class BrokenTextResponse:
        @property
        def text(self) -> str:
            raise ValueError("response was blocked")

        candidates = blocked.candidates

    assert extract_response_text(BrokenTextResponse()) == "Fallback"
