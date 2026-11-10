"""Tests for the Groq self-correcting invoice parser (AI path, fully offline)."""

from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

import invoice_parser
from invoice_parser import GeminiParsedInvoice, clean_json_response, parse_invoice_with_retry

VALID_PAYLOAD = {
    "client_name": "TEHNO MAK DOOEL",
    "tax_id": "4030123456789",
    "net_amount": "1000.00",
    "vat_rate": "18.00",
    "vat_amount": "180.00",
    "total_amount": "1180.00",
}


def test_clean_json_response_strips_fences() -> None:
    assert clean_json_response('  ```json\n{"a": 1}\n```  ') == '{"a": 1}'
    assert clean_json_response('```\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_json_response(' {"a": 1} ') == '{"a": 1}'
    assert clean_json_response("") == ""


def test_request_invoice_json_uses_groq_openai_client(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"ok": true}')
                    )
                ]
            )

    class FakeClient:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    result = invoice_parser._request_invoice_json("extract this invoice")

    assert result == '{"ok": true}'
    assert captured["api_key"] == "gsk_test-key"
    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["model"] == "llama-3.3-70b-versatile"
    assert captured["response_format"] == {"type": "json_object"}
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "extract this invoice"}


def test_request_invoice_json_requires_groq_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        invoice_parser._request_invoice_json("some invoice text")


def test_retry_repairs_bad_tax_id_then_succeeds(monkeypatch) -> None:
    prompts: list[str] = []

    def fake_request(prompt: str) -> str:
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                '```json\n'
                '{"client_name": "X", "tax_id": "123", "net_amount": "1.00", '
                '"vat_rate": "18.00", "vat_amount": "0.18", "total_amount": "1.18"}\n'
                '```'
            )
        return json.dumps(VALID_PAYLOAD)

    monkeypatch.setattr(invoice_parser, "_request_invoice_json", fake_request)

    result = parse_invoice_with_retry("Faktura TEHNO MAK 1000 MKD", max_retries=3)

    assert isinstance(result, GeminiParsedInvoice)
    assert result.client_name == "TEHNO MAK DOOEL"
    assert result.tax_id == "4030123456789"
    assert result.net_amount == Decimal("1000.00")
    assert result.total_amount == Decimal("1180.00")
    assert len(prompts) == 2
    assert "CRITICAL FIX REQUIRED: Previous attempt failed validation with error: " in prompts[1]
    assert "Return ONLY corrected valid JSON matching the exact schema." in prompts[1]
    assert "13 numeric digits" in prompts[1]
    assert "Faktura TEHNO MAK 1000 MKD" in prompts[0]


def test_all_retries_exhausted_raises_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(
        invoice_parser,
        "_request_invoice_json",
        lambda prompt: '```json\n{"client_name": "X"}\n```',
    )

    with pytest.raises(
        RuntimeError,
        match=r"AI parsing failed validation after 3 attempts\. Last error: ",
    ):
        parse_invoice_with_retry("some invoice text", max_retries=3)


def test_math_discrepancy_is_caught_and_fed_back(monkeypatch) -> None:
    prompts: list[str] = []

    def fake_request(prompt: str) -> str:
        prompts.append(prompt)
        # total_amount does not match net + VAT -> InvoiceMathError -> wrapped
        # into ValidationError by Pydantic -> caught -> fed back.
        return json.dumps({**VALID_PAYLOAD, "total_amount": "1180.01"})

    monkeypatch.setattr(invoice_parser, "_request_invoice_json", fake_request)

    with pytest.raises(RuntimeError, match="AI parsing failed validation"):
        parse_invoice_with_retry("invoice", max_retries=2)

    # Attempt 1's math failure must surface verbatim in attempt 2's prompt.
    assert "CRITICAL FIX REQUIRED: Previous attempt failed validation with error: " in prompts[1]
    assert "Math discrepancy: parsed total 1180.01" in prompts[1]


def test_empty_input_rejected() -> None:
    with pytest.raises(ValueError):
        parse_invoice_with_retry("   ")
    with pytest.raises(ValueError):
        parse_invoice_with_retry(123)  # type: ignore[arg-type]
