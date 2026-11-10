"""Tests for the invoice lifecycle state machine (state_machine.py)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from models import Base, Client, Invoice, InvoiceState
from state_machine import InvalidStateTransitionError, transition_invoice_state


@pytest.fixture
def db_session():
    """An isolated in-memory SQLite session (never the project's data file)."""

    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture
def invoice(db_session) -> Invoice:
    """A persisted DRAFT invoice ready for transition testing."""

    record = Invoice(
        id="TST-2026-0500",
        client_tax_id="4030000000000",
        net_amount=Decimal("1000.00"),
        vat_amount=Decimal("180.00"),
        total_amount=Decimal("1180.00"),
    )
    db_session.add(
        Client(tax_id="4030000000000", name="Acme LLC", email="billing@acme.com", phone="0701")
    )
    db_session.add(record)
    db_session.commit()
    return record


def test_draft_approval_requires_human_approval(invoice) -> None:
    with pytest.raises(
        InvalidStateTransitionError,
        match="Human approval required before approving invoice",
    ):
        transition_invoice_state(invoice, InvoiceState.APPROVED)
    assert invoice.state is InvoiceState.DRAFT


def test_approve_override_stamps_approval_and_persists(db_session, invoice) -> None:
    result = transition_invoice_state(invoice, InvoiceState.APPROVED, approve_override=True)

    assert result.state is InvoiceState.APPROVED
    assert result.is_approved is True
    persisted = db_session.scalar(select(Invoice).where(Invoice.id == "TST-2026-0500"))
    assert persisted.state is InvoiceState.APPROVED
    assert persisted.is_approved is True


def test_pre_approved_invoice_needs_no_override(invoice) -> None:
    invoice.is_approved = True
    result = transition_invoice_state(invoice, InvoiceState.APPROVED)
    assert result.state is InvoiceState.APPROVED
    assert result.is_approved is True


def test_string_and_enum_states_normalize(db_session, invoice) -> None:
    transition_invoice_state(invoice, "APPROVED", db_session, approve_override=True)
    assert invoice.state is InvoiceState.APPROVED

    transition_invoice_state(invoice, InvoiceState.UBL_GENERATED, db_session)
    transition_invoice_state(invoice, "SENT", db_session)
    persisted = db_session.scalar(select(Invoice).where(Invoice.id == "TST-2026-0500"))
    assert persisted.state is InvoiceState.SENT


def test_failed_reset_and_full_pipeline(db_session, invoice) -> None:
    transition_invoice_state(invoice, InvoiceState.FAILED, db_session)
    assert invoice.state is InvoiceState.FAILED

    transition_invoice_state(invoice, InvoiceState.DRAFT, db_session)
    assert invoice.state is InvoiceState.DRAFT

    transition_invoice_state(invoice, InvoiceState.APPROVED, db_session, approve_override=True)
    transition_invoice_state(invoice, InvoiceState.UBL_GENERATED, db_session)
    transition_invoice_state(invoice, InvoiceState.SENT, db_session)
    assert invoice.state is InvoiceState.SENT


@pytest.mark.parametrize(
    ("start", "target"),
    [
        (InvoiceState.DRAFT, InvoiceState.UBL_GENERATED),
        (InvoiceState.DRAFT, InvoiceState.SENT),
        (InvoiceState.APPROVED, InvoiceState.SENT),
        (InvoiceState.UBL_GENERATED, InvoiceState.APPROVED),
        (InvoiceState.SENT, InvoiceState.APPROVED),
        (InvoiceState.DRAFT, InvoiceState.DRAFT),
    ],
)
def test_invalid_transitions_raise(db_session, invoice, start, target) -> None:
    invoice.state = start
    with pytest.raises(
        InvalidStateTransitionError,
        match=f"Cannot transition from {start.value} to {target.value}",
    ):
        transition_invoice_state(invoice, target)


def test_db_rollback_on_failure(invoice, monkeypatch) -> None:
    class ExplodingSession:
        def __init__(self) -> None:
            self.rolled_back = False

        def add(self, obj) -> None:
            pass

        def commit(self) -> None:
            raise RuntimeError("disk full")

        def rollback(self) -> None:
            self.rolled_back = True

    session = ExplodingSession()
    with pytest.raises(RuntimeError, match="disk full"):
        transition_invoice_state(invoice, InvoiceState.FAILED, db_session=session)
    assert session.rolled_back is True
