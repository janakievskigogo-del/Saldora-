"""Tests for the mock outbound dispatcher adapters."""

from __future__ import annotations

import pytest

from dispatcher import (
    DispatchError,
    dispatch_message,
    send_email,
    send_viber,
    send_whatsapp,
)
from legacy_models import Channel


def test_email_dispatch_renders_console_block(capsys: pytest.CaptureFixture[str]) -> None:
    result = dispatch_message(
        Channel.EMAIL,
        "billing@acme.com",
        "Dear client, please find your reminder.",
        invoice_id="TST-2026-0001",
        language="EN",
        subject="Payment reminder",
    )

    assert result.success
    assert result.channel == Channel.EMAIL
    assert result.recipient == "billing@acme.com"
    assert result.message_id.startswith("MOCK-EMAIL-")
    output = capsys.readouterr().out
    assert "EMAIL MESSAGE SENT TO billing@acme.com" in output
    assert "Subject: Payment reminder" in output
    assert "TST-2026-0001" in output


def test_viber_and_whatsapp_normalize_phone(capsys: pytest.CaptureFixture[str]) -> None:
    viber = send_viber(
        "+389 (70) 123-456",
        "Dear client, a short Viber reminder.",
        invoice_id="V-1",
        language="MK",
    )
    assert viber.success
    assert viber.recipient == "+38970123456"

    whatsapp = send_whatsapp(
        "00389 70 123 456",
        "Dear client, a short WhatsApp reminder.",
        invoice_id="W-1",
        language="AL",
    )
    assert whatsapp.success
    assert whatsapp.recipient == "+38970123456"
    output = capsys.readouterr().out
    assert "VIBER MESSAGE SENT" in output
    assert "WHATSAPP MESSAGE SENT" in output


def test_invalid_email_is_rejected() -> None:
    with pytest.raises(DispatchError, match="Invalid email"):
        send_email("not-an-email", "Hello", invoice_id="X-1", language="EN")


def test_invalid_phone_is_rejected() -> None:
    with pytest.raises(DispatchError, match="Invalid Viber"):
        send_viber("123", "Hello", invoice_id="X-1", language="MK")
    with pytest.raises(DispatchError, match="Invalid WhatsApp"):
        send_whatsapp("", "Hello", invoice_id="X-1", language="AL")


def test_empty_message_is_rejected() -> None:
    with pytest.raises(DispatchError, match="empty email"):
        send_email("billing@acme.com", "  ", invoice_id="X-1", language="EN")
    with pytest.raises(DispatchError, match="empty Viber"):
        send_viber("+38970123456", "", invoice_id="X-1", language="MK")
    with pytest.raises(DispatchError, match="empty WhatsApp"):
        send_whatsapp("+38970123456", "  ", invoice_id="X-1", language="AL")


def test_missing_recipient_is_rejected() -> None:
    with pytest.raises(DispatchError, match="No recipient"):
        dispatch_message(
            Channel.EMAIL,
            None,
            "Hello",
            invoice_id="X-1",
            language="EN",
        )


def test_unknown_channel_falls_back_to_email() -> None:
    result = dispatch_message(
        "not-a-real-channel",
        "billing@acme.com",
        "Hello, this is a fallback channel test.",
        invoice_id="X-1",
        language="EN",
    )
    assert result.channel == Channel.EMAIL
