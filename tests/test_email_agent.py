"""Unit and integration tests for email_agent.py."""

from __future__ import annotations

from decimal import Decimal
from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from email_agent import (
    GmailReplyAgent,
    InvoiceContext,
    _extract_invoice_codes,
    _extract_subject,
    _subject_matches_saldora,
    find_invoice_by_code,
    get_invoice_context,
    resolve_invoice_context,
)
from models import Base, EmailDraft, Invoice, InvoiceState


@pytest.fixture
def test_db():
    """Create an in-memory SQLite database seeded with test invoices."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()

    inv1 = Invoice(
        id="NPL-2026-0001",
        code="NPL-2026-0001",
        client_name="КамМаркет",
        amount=Decimal("32000.00"),
        total_amount=Decimal("32000.00"),
        status="UNPAID",
        state=InvoiceState.UNPAID,
    )
    inv2 = Invoice(
        id="INV-2026-002",
        code="INV-2026-002",
        client_name="Техно-Мак",
        amount=Decimal("15000.00"),
        total_amount=Decimal("15000.00"),
        status="UNPAID",
        state=InvoiceState.UNPAID,
    )
    session.add_all([inv1, inv2])
    session.commit()

    yield session

    session.close()


# ---------------------------------------------------------------------------
# 1. Flexible Subject Matching Tests
# ---------------------------------------------------------------------------


def test_subject_matches_saldora_various_positions_and_casings():
    """Subject matching should find 'Saldora' anywhere regardless of case or position."""
    # Word at beginning
    assert _subject_matches_saldora("Saldora - прашање за плаќање")
    assert _subject_matches_saldora("saldora: invoice inquiry")
    assert _subject_matches_saldora("SALDORA IMPORTANT")

    # Word in middle
    assert _subject_matches_saldora("Re: Saldora Reminder - Invoice")
    assert _subject_matches_saldora("Fwd: Regarding Saldora debt negotiation")
    assert _subject_matches_saldora("Порака од клиент за saldora фактура")

    # Word at end
    assert _subject_matches_saldora("Прашање за плаќање кон Saldora")
    assert _subject_matches_saldora("Urgent payment issue - saldora")

    # Cyrillic "Салдора"
    assert _subject_matches_saldora("Салдора потсетник")
    assert _subject_matches_saldora("Одговор за салдора фактура")

    # No exact NPL code required
    assert _subject_matches_saldora("Saldora договор")


def test_subject_matches_saldora_rejects_unrelated():
    """Subjects without Saldora should be rejected."""
    assert not _subject_matches_saldora("")
    assert not _subject_matches_saldora("Random email subject")
    assert not _subject_matches_saldora("Invoice NPL-2026-0001")  # No Saldora keyword
    assert not _subject_matches_saldora("Payment inquiry without keyword")


def test_extract_subject_mime_decoding():
    """MIME-encoded subjects should be properly decoded."""
    msg = EmailMessage()
    # RFC 2047 encoded subject
    msg["Subject"] = "=?UTF-8?B?U2FsZG9yYSDQn9GA0LDRiNCw0ZrQtQ==?="  # Saldora Прашање
    subject = _extract_subject(msg)
    assert "Saldora" in subject
    assert "Прашање" in subject
    assert _subject_matches_saldora(subject)


# ---------------------------------------------------------------------------
# 2. Flexible Invoice Code Extraction & Database Lookup Tests
# ---------------------------------------------------------------------------


def test_extract_invoice_codes_patterns():
    """Search subject and body for any pattern resembling an invoice code."""
    # Prefixed codes
    codes = _extract_invoice_codes("Re: Saldora NPL-2026-0001", "")
    assert "NPL-2026-0001" in codes

    codes = _extract_invoice_codes("Saldora", "Here is payment for INV-2026-001")
    assert "INV-2026-001" in codes

    codes = _extract_invoice_codes("Saldora", "Short code NPL-6 please check")
    assert "NPL-6" in codes

    # Labeled codes
    codes = _extract_invoice_codes("Saldora", "Фактура број 12345 за плаќање")
    assert "12345" in codes

    # Plain numbers
    codes = _extract_invoice_codes("Saldora 0001", "")
    assert "0001" in codes

    # Standalone 4-digit years should not be extracted as invoice codes
    codes = _extract_invoice_codes("Saldora 2026", "Message on 2026-11-04")
    assert "2026" not in codes


def test_find_invoice_by_code(test_db):
    """Database lookup matches exact codes, variants, compact codes, and numbers."""
    # Exact match
    inv = find_invoice_by_code(test_db, "NPL-2026-0001")
    assert inv is not None
    assert inv.id == "NPL-2026-0001"

    # Case-insensitive
    inv = find_invoice_by_code(test_db, "npl-2026-0001")
    assert inv is not None

    # Compact code without separators
    inv = find_invoice_by_code(test_db, "NPL20260001")
    assert inv is not None

    # Variant with underscores
    inv = find_invoice_by_code(test_db, "NPL_2026_0001")
    assert inv is not None

    # Plain number match
    inv = find_invoice_by_code(test_db, "0001")
    assert inv is not None
    assert inv.id == "NPL-2026-0001"

    # Second invoice
    inv2 = find_invoice_by_code(test_db, "INV-2026-002")
    assert inv2 is not None
    assert inv2.id == "INV-2026-002"


def test_resolve_invoice_context_fallback_when_no_code(test_db):
    """When no invoice code is found, default to most recent active UNPAID invoice."""
    with patch("email_agent.SessionLocal", return_value=test_db):
        context = resolve_invoice_context([], "client@example.com")
        assert context is not None
        # Should default to most recent unpaid invoice
        assert context.invoice_id in ("NPL-2026-0001", "INV-2026-002")
        assert context.client_email == "client@example.com"
        assert context.amount > 0


def test_resolve_invoice_context_fallback_when_code_not_found(test_db):
    """When an unrecognized code is provided, default to most recent unpaid invoice."""
    with patch("email_agent.SessionLocal", return_value=test_db):
        context = resolve_invoice_context(["INV-99999"], "client@example.com")
        assert context is not None
        # Does not drop reply! Defaults to active unpaid invoice
        assert context.client_email == "client@example.com"
        assert context.invoice_id in ("NPL-2026-0001", "INV-2026-002")


def test_get_invoice_context_routing():
    """Ensure context routes directly back to sender_email and extracts amount properly."""
    inv = Invoice(
        id="NPL-2026-0001",
        code="NPL-2026-0001",
        client_name="Тест ДОО",
        total_amount=Decimal("5000.00"),
        status="UNPAID",
    )
    context = get_invoice_context(inv, "sender@external.com")
    assert context.client_email == "sender@external.com"
    assert context.client_name == "Тест ДОО"
    assert context.amount == Decimal("5000.00")
    assert context.invoice_code == "NPL-2026-0001"


# ---------------------------------------------------------------------------
# 3. Immediate \Seen and Reply Routing Tests
# ---------------------------------------------------------------------------


def test_check_once_marks_seen_immediately_and_dispatches(test_db):
    """Verify that email is marked \\Seen immediately after fetching and replied to."""
    agent = GmailReplyAgent(username="test@gmail.com", password="password")

    # Mock IMAP connection
    mock_conn = MagicMock()

    def mock_uid(cmd, *args):
        if cmd == "search":
            if "Салдора" in args:
                return ("OK", [b""])
            return ("OK", [b"101"])
        elif cmd == "fetch":
            return (
                "OK",
                [
                    (
                        b"101 (BODY.PEEK[])",
                        b"From: client@test.com\r\nSubject: Saldora negotiation\r\n\r\nMoze li na 2 rati?",
                    )
                ],
            )
        elif cmd == "store":
            return ("OK", [b"Flags set"])
        return ("OK", [])

    mock_conn.uid.side_effect = mock_uid
    agent._connection = mock_conn

    with (
        patch.object(agent, "_ensure_connection", return_value=mock_conn),
        patch("email_agent.SessionLocal", return_value=test_db),
        patch(
            "email_agent.generate_ai_reply",
            return_value="Почитувани, прифаќаме плаќање на најмногу 2 рати.",
        ) as mock_ai,
        patch("email_agent.send_email_native", return_value=(True, "Success")),
    ):
        agent.check_once()

        # Check STORE \Seen was called on UID 101
        store_calls = [
            c
            for c in mock_conn.uid.call_args_list
            if c[0][0] == "store" and c[0][1] == "101" and "\\Seen" in str(c[0][3])
        ]
        assert len(store_calls) >= 1

        # Check AI model was called
        mock_ai.assert_called_once()
        args, kwargs = mock_ai.call_args
        assert isinstance(args[0], InvoiceContext)
        assert "Moze li na 2 rati?" in args[1]

        # Check EmailDraft was created for human-in-the-loop approval
        draft = test_db.query(EmailDraft).filter(EmailDraft.client_email == "client@test.com").first()
        assert draft is not None
        assert draft.status == "PENDING_APPROVAL"
        assert "Saldora" in draft.incoming_subject
        assert "Почитувани" in draft.ai_draft_response


def test_check_once_skips_when_subject_lacks_saldora():
    """If subject lacks Saldora, message is marked Seen but not sent to AI."""
    agent = GmailReplyAgent(username="test@gmail.com", password="password")
    mock_conn = MagicMock()

    def mock_uid(cmd, *args):
        if cmd == "search":
            if "Салдора" in args:
                return ("OK", [b""])
            return ("OK", [b"202"])
        elif cmd == "fetch":
            return (
                "OK",
                [
                    (
                        b"202 (BODY.PEEK[])",
                        b"From: spam@example.com\r\nSubject: Random Newsletter\r\n\r\nBuy now",
                    )
                ],
            )
        elif cmd == "store":
            return ("OK", [b"Flags set"])
        return ("OK", [])

    mock_conn.uid.side_effect = mock_uid

    with (
        patch.object(agent, "_ensure_connection", return_value=mock_conn),
        patch("email_agent.generate_ai_reply") as mock_ai,
        patch("email_agent.send_email_native") as mock_send,
    ):
        agent.check_once()

        # \Seen should be marked immediately
        store_calls = [
            c
            for c in mock_conn.uid.call_args_list
            if c[0][0] == "store" and c[0][1] == "202" and "\\Seen" in str(c[0][3])
        ]
        assert len(store_calls) >= 1

        # AI and send should not be called
        mock_ai.assert_not_called()
        mock_send.assert_not_called()


def test_empty_body_uses_subject_as_client_message(test_db):
    """When email body is empty, client_message falls back to subject so context is not lost."""
    agent = GmailReplyAgent(username="test@gmail.com", password="password")
    mock_conn = MagicMock()

    def mock_uid(cmd, *args):
        if cmd == "search":
            if "Салдора" in args:
                return ("OK", [b""])
            return ("OK", [b"303"])
        elif cmd == "fetch":
            return (
                "OK",
                [
                    (
                        b"303 (BODY.PEEK[])",
                        b"From: buyer@test.com\r\nSubject: Saldora NPL-2026-0001 baram odlozuvanje\r\n\r\n",
                    )
                ],
            )
        elif cmd == "store":
            return ("OK", [b"Flags set"])
        return ("OK", [])

    mock_conn.uid.side_effect = mock_uid

    with (
        patch.object(agent, "_ensure_connection", return_value=mock_conn),
        patch("email_agent.SessionLocal", return_value=test_db),
        patch(
            "email_agent.generate_ai_reply",
            return_value="Почитувани, во врска со фактурата NPL-2026-0001...",
        ) as mock_ai,
        patch("email_agent.send_email_native", return_value=(True, "Success")),
    ):
        agent.check_once()

        mock_ai.assert_called_once()
        passed_text = mock_ai.call_args[0][1]
        assert "Saldora NPL-2026-0001 baram odlozuvanje" in passed_text
        draft = test_db.query(EmailDraft).filter(EmailDraft.client_email == "buyer@test.com").first()
        assert draft is not None
        assert draft.status == "PENDING_APPROVAL"


def test_invoice_code_extraction_inv_and_npl_anywhere():
    """Verify INV-2026-001 and NPL-2026-0001 are extracted from anywhere in subject or body."""
    # INV-2026-001 in middle of body
    codes = _extract_invoice_codes(
        "Saldora Прашање",
        "Здраво,\nВе молам за информација за фактурата INV-2026-001.\nПоздрав",
    )
    assert "INV-2026-001" in codes

    # NPL-2026-0001 at start of subject
    codes = _extract_invoice_codes(
        "NPL-2026-0001 - Saldora потсетник",
        "Ќе уплатиме утре.",
    )
    assert "NPL-2026-0001" in codes

    # Numbers in subject
    codes = _extract_invoice_codes("Saldora 0001", "Платено")
    assert "0001" in codes

