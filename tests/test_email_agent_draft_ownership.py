"""Regression: inbound email drafts must inherit the invoice owner's user_id.

A draft created by the IMAP agent for invoice NPL-XXXX owned by user A must
land in user A's queue — not user_id=None (invisible, later mis-assigned to
the demo account at startup).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from models import Invoice, User


def _seed_invoice_with_owner(db: Session) -> tuple[Invoice, User]:
    unique = datetime.utcnow().strftime("%H%M%S%f")
    user = User(
        email=f"owner_{unique}@saldora.ai",
        hashed_password="x",
        company_name="Тест",
    )
    db.add(user)
    db.flush()
    invoice = Invoice(
        code=f"NPL-2026-{unique[-8:]}",
        client_name="Тест Клиент",
        total_amount=1180,
        status="UNPAID",
        user_id=user.id,
    )
    db.add(invoice)
    db.commit()
    return invoice, user


def test_draft_creation_uses_invoice_owner(tmp_path, monkeypatch):
    """email_agent must stamp invoice.user_id onto the created EmailDraft."""
    from database import SessionLocal
    from email_agent import _create_pending_draft
    from models import EmailDraft

    with SessionLocal() as db:
        invoice, user = _seed_invoice_with_owner(db)
        invoice_code = invoice.code
        owner_id = user.id

    draft_id = _create_pending_draft(
        invoice_code=invoice_code,
        sender="client@example.com",
        subject="Saldora - Барaње",
        message="Може ли рати?",
        reply="Можеме да понудиме 2 рати.",
    )
    assert draft_id is not None

    with SessionLocal() as db:
        draft = db.query(EmailDraft).filter(EmailDraft.id == draft_id).first()
        assert draft is not None
        assert draft.invoice_id == invoice_code
        # The draft must be visible to the invoice owner's queue query
        assert draft.user_id == owner_id
        draft_qs = (
            db.query(EmailDraft)
            .filter(EmailDraft.user_id == owner_id, EmailDraft.id == draft_id)
            .first()
        )
        assert draft_qs is not None

        # Cleanup
        db.delete(draft)
        db.delete(db.query(Invoice).filter(Invoice.code == invoice_code).first())
        db.delete(db.query(User).filter(User.id == user.id).first())
        db.commit()
