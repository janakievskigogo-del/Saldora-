"""E2E integration tests: full invoice lifecycle over the SMTP dispatcher.

Real components in play (nothing stubbed except the transport socket):
* ``database.init_db`` against a tmp-file SQLite database (one shared file so
  pooled sessions hit the same instance).
* ``schemas.ClientSchema`` / ``schemas.InvoiceCreate`` validation feeding the
  ORM rows (money stays cent-exact Decimal end to end).
* ``state_machine.transition_invoice_state`` for every legal hop, including
  the hitl approval override and the ``FAILED -> DRAFT`` recovery path.
* ``smtplib.SMTP`` monkeypatched per test (success + ``SMTPException`` failure).
* A capture handler proving every record passing the PII logging path arrives
  masked (e-mail / tax ID / phone) on the happy path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import dispatcher
import database
from dispatcher import SMTPDispatcher
from logger import PIIFilter, logger as pii_logger
from models import Base, Client, Invoice, InvoiceState
from schemas import ClientSchema, InvoiceCreate
from state_machine import (
    InvalidStateTransitionError,
    transition_invoice_state,
)

_TAX_ID = "4030000000000"
_CLIENT_EMAIL = "billing@acme.com"
_CLIENT_PHONE = "077123456"
_CLIENT_NAME = "Acme LLC"


class _RecordingSMTP:
    """``smtplib.SMTP`` stand-in that records the transaction, always succeeds."""

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host = host
        self.port = port
        self.sent: list[object] = []

    def __enter__(self) -> "_RecordingSMTP":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self, context: object = None) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        return None

    def send_message(self, message: object) -> None:
        self.sent.append(message)


class _OutageSMTP(_RecordingSMTP):
    """Same stand-in, but every ``send_message`` raises ``SMTPException``."""

    def send_message(self, message: object) -> None:
        raise dispatcher.smtplib.SMTPException("Connection timed out")


class _MaskedCapture(logging.Handler):
    """Collects ``getMessage()`` output exactly as downstream handlers see it."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


@pytest.fixture
def e2e_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Session, Engine]]:
    """Tmp-file SQLite wired through the real ``database.init_db`` entry point."""

    engine = create_engine(f"sqlite:///{tmp_path / 'test_e2e.db'}")
    # Same file for the whole test: session pooling connects to one instance.
    monkeypatch.setattr(database, "engine", engine)
    database.init_db()  # step 1: tables live in the tmp DB
    with Session(engine) as session:
        yield session, engine
    engine.dispose()


@pytest.fixture
def smtp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in (
        ("SMTP_SERVER", "smtp.test.mk"),
        ("SMTP_PORT", "587"),
        ("SMTP_USER", "billing@test.mk"),
        ("SMTP_PASS", "secret"),
    ):
        monkeypatch.setenv(key, value)


def _seed_validated_rows(db: Session) -> str:
    """Create validated client + DRAFT invoice rows; return the invoice id."""

    client = ClientSchema(
        name=_CLIENT_NAME, tax_id=_TAX_ID, email=_CLIENT_EMAIL, phone=_CLIENT_PHONE
    )
    create = InvoiceCreate(
        client_tax_id=client.tax_id,
        net_amount=Decimal("1000.00"),
        vat_rate=Decimal("18.00"),
    )
    invoice_id = "E2E-2026-0001"
    db.add(
        Client(
            tax_id=client.tax_id,
            name=client.name,
            email=client.email,
            phone=client.phone,
        )
    )
    db.add(
        Invoice(
            id=invoice_id,
            client_tax_id=create.client_tax_id,
            net_amount=create.net_amount,
            vat_rate=create.vat_rate,
            vat_amount=create.vat_amount,
            total_amount=create.total_amount,
            state=InvoiceState.DRAFT,
        )
    )
    db.commit()
    return invoice_id


def _state_of(engine: Engine, invoice_id: str) -> InvoiceState:
    with Session(engine) as session:
        row = session.scalar(select(Invoice).where(Invoice.id == invoice_id))
        assert row is not None
        return InvoiceState(row.state)


def test_smtp_credentials_fall_back_to_legacy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``SMTP_*`` key falls back to its ``EMAIL_SMTP_*`` legacy alias."""

    for key in ("SMTP_SERVER", "SMTP_PORT", "SMTP_USER", "SMTP_PASS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "legacy.test.mk")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "465")
    monkeypatch.setenv("EMAIL_SMTP_USER", "legacy@test.mk")
    monkeypatch.setenv("EMAIL_SMTP_PASS", "legacy-secret")

    config = SMTPDispatcher._smtp_config()

    assert config == {
        "SMTP_SERVER": "legacy.test.mk",
        "SMTP_PORT": "465",
        "SMTP_USER": "legacy@test.mk",
        "SMTP_PASS": "legacy-secret",
    }

    monkeypatch.delenv("EMAIL_SMTP_SERVER", raising=False)
    with pytest.raises(
        ValueError,
        match="SMTP configuration missing: set SMTP_SERVER or EMAIL_SMTP_SERVER",
    ):
        SMTPDispatcher._smtp_config()


def test_full_invoice_lifecycle_happy_path(
    e2e_db, smtp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, engine = e2e_db
    invoice_id = _seed_validated_rows(db)
    dispatcher_obj = SMTPDispatcher()

    # Step 3: a DRAFT invoice must be refused before any socket activity.
    with pytest.raises(
        InvalidStateTransitionError,
        match="Invoice must be APPROVED before sending reminder",
    ):
        dispatcher_obj.send_reminder(invoice_id, db)
    assert _state_of(engine, invoice_id) is InvoiceState.DRAFT

    # Step 4: human-in-the-loop approval (stamps APPROVED).
    invoice_row = db.query(Invoice).filter_by(id=invoice_id).first()
    assert invoice_row is not None
    transition_invoice_state(
        invoice_row, InvoiceState.APPROVED, db, approve_override=True
    )
    assert _state_of(engine, invoice_id) is InvoiceState.APPROVED

    # Step 5: artifact preparation hop (UBL e-invoice XML generated upstream).
    transition_invoice_state(invoice_row, InvoiceState.UBL_GENERATED, db)
    assert _state_of(engine, invoice_id) is InvoiceState.UBL_GENERATED

    # Step 6: a successful SMTP transaction is simulated.
    monkeypatch.setattr(dispatcher.smtplib, "SMTP", _RecordingSMTP)

    accepted = dispatcher_obj.send_reminder(invoice_id, db)

    # Step 7: dispatcher walked UBL_GENERATED->SENT and committed.
    assert accepted is True
    assert _state_of(engine, invoice_id) is InvoiceState.SENT

    # Step 8: the PII filter must have masked every record it processed --
    # prove it with an intentionally PII-laden log line in the same run.
    captured = _MaskedCapture()
    pii_logger.addHandler(captured)
    try:
        pii_logger.warning(
            "Probe: %s / %s / %s", _CLIENT_EMAIL, _TAX_ID, _CLIENT_PHONE
        )
    finally:
        pii_logger.removeHandler(captured)

    pii_lines = [line for line in captured.lines if "Probe:" in line]
    assert pii_lines, "the PII probe record was never captured"
    combined = "\n".join(pii_lines)
    for raw_value in (_CLIENT_EMAIL, _TAX_ID, _CLIENT_PHONE):
        assert raw_value not in combined
    assert PIIFilter.EMAIL_MASK in combined            # e-mail masked
    assert PIIFilter.TAX_ID_MASK in combined           # tax ID masked
    # Phone keeps the configured prefix/star pattern instead of the raw number.
    assert "07****222" in combined


def test_invoice_lifecycle_smtp_failure_recovery(
    e2e_db, smtp_env, monkeypatch: pytest.MonkeyPatch
) -> None:
    db, engine = e2e_db
    invoice_id = _seed_validated_rows(db)

    invoice_row = db.query(Invoice).filter_by(id=invoice_id).first()
    assert invoice_row is not None
    transition_invoice_state(
        invoice_row, InvoiceState.APPROVED, db, approve_override=True
    )
    transition_invoice_state(invoice_row, InvoiceState.UBL_GENERATED, db)

    monkeypatch.setattr(dispatcher.smtplib, "SMTP", _OutageSMTP)
    accepted = SMTPDispatcher().send_reminder(invoice_id, db)

    # Step 4: the exception was caught safely and FAILED was committed.
    assert accepted is False
    assert _state_of(engine, invoice_id) is InvoiceState.FAILED

    # Step 5: the state machine permits the recovery hop back to DRAFT.
    persisted = db.query(Invoice).filter_by(id=invoice_id).first()
    assert persisted is not None
    transition_invoice_state(persisted, InvoiceState.DRAFT, db)
    assert _state_of(engine, invoice_id) is InvoiceState.DRAFT
