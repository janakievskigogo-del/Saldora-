"""Regression tests for the dispatcher ↔ SQL state machine integration.

Every test swaps the real ``database.engine`` for an isolated SQLite engine
(tmp file), so the suite never touches ``data/saldora.db``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

import dispatcher
from dispatcher import DispatchError, dispatch_message
from legacy_models import Channel
from models import Base, Client, Invoice, InvoiceState

_INVOICE_ID = "TST-2026-0600"


@contextmanager
def _session(engine: Engine) -> Iterator[Session]:
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def sql_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{tmp_path / 'test-saldora.db'}")
    yield engine
    engine.dispose()


@pytest.fixture
def dispatcher_db(sql_engine: Engine, monkeypatch: pytest.MonkeyPatch):
    """Isolated initialized SQLite engine swapped into ``dispatcher.get_db``."""

    Base.metadata.create_all(bind=sql_engine)

    @contextmanager
    def patched_get_db() -> Iterator[Session]:
        with _session(sql_engine) as session:
            yield session

    monkeypatch.setattr(dispatcher, "get_db", patched_get_db)
    return sql_engine


def _seed_invoice(engine: Engine, invoice_id: str, state: InvoiceState) -> None:
    with _session(engine) as session:
        session.add(
            Client(
                tax_id="4030000000000",
                name="Acme LLC",
                email="billing@acme.com",
                phone="+38970123456",
            )
        )
        session.add(
            Invoice(
                id=invoice_id,
                client_tax_id="4030000000000",
                net_amount=Decimal("1000.00"),
                vat_amount=Decimal("180.00"),
                total_amount=Decimal("1180.00"),
                state=state,
            )
        )
        session.commit()


def _state_of(engine: Engine, invoice_id: str) -> InvoiceState:
    with _session(engine) as session:
        row = session.scalar(select(Invoice).where(Invoice.id == invoice_id))
        assert row is not None
        return InvoiceState(row.state)


def test_dispatch_persists_sent_transition(
    dispatcher_db: Engine, capsys: pytest.CaptureFixture[str]
) -> None:
    """A legal UBL_GENERATED invoice reaches SENT and the mock send fires."""
    _seed_invoice(dispatcher_db, _INVOICE_ID, InvoiceState.UBL_GENERATED)

    result = dispatch_message(
        Channel.EMAIL,
        "billing@acme.com",
        "Dear client, a mock reminder body.",
        invoice_id=_INVOICE_ID,
        language="EN",
        subject="Payment reminder",
    )

    assert result.success
    assert result.message_id.startswith("MOCK-EMAIL-")
    assert _state_of(dispatcher_db, _INVOICE_ID) is InvoiceState.SENT
    assert "EMAIL MESSAGE SENT TO billing@acme.com" in capsys.readouterr().out


@pytest.mark.parametrize(
    "start_state",
    [
        InvoiceState.DRAFT,
        InvoiceState.APPROVED,
        InvoiceState.FAILED,
    ],
)
def test_dispatch_blocks_illegal_transition_before_sending(
    dispatcher_db: Engine, start_state: InvoiceState, capsys: pytest.CaptureFixture[str]
) -> None:
    """HITL gate: illegal lifecycle transitions raise and never dispatch."""
    _seed_invoice(dispatcher_db, _INVOICE_ID, start_state)

    with pytest.raises(
        DispatchError,
        match=f"State transition blocked: Cannot transition from {start_state.value} to SENT",
    ):
        dispatch_message(
            Channel.EMAIL,
            "billing@acme.com",
            "Dear client, a mock reminder body.",
            invoice_id=_INVOICE_ID,
            language="EN",
        )

    assert _state_of(dispatcher_db, _INVOICE_ID) is start_state
    output = capsys.readouterr().out
    assert "MESSAGE SENT" not in output


def test_dispatch_proceeds_when_invoice_missing(
    dispatcher_db: Engine,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An initialized-but-empty invoice table only warns; the send still fires."""
    with _session(dispatcher_db) as session:
        assert session.scalar(select(Client.tax_id).limit(1)) is None

    with caplog.at_level(logging.WARNING, "dispatcher"):
        result = dispatch_message(
            Channel.EMAIL,
            "billing@acme.com",
            "Dear client, a mock reminder body.",
            invoice_id=_INVOICE_ID,
            language="EN",
        )

    assert result.success
    assert "not found in DB prior to dispatch" in caplog.text
    assert result.message_id.startswith("MOCK-EMAIL-")
    assert "EMAIL MESSAGE SENT" in capsys.readouterr().out


def test_dispatch_degrades_gracefully_when_db_unavailable(
    tmp_path: Path,
    dispatcher_db: Engine,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A database file without the app tables warns; dispatch still succeeds."""
    broken = create_engine(f"sqlite:///{tmp_path / 'uninitialized.db'}")
    try:
        with Session(broken) as session:
            session.execute(text("CREATE TABLE unrelated (x INT)"))  # no app tables
            session.commit()

        @contextmanager
        def broken_get_db() -> Iterator[Session]:
            with _session(broken) as session:
                yield session

        monkeypatch.setattr(dispatcher, "get_db", broken_get_db)

        with caplog.at_level(logging.WARNING, "dispatcher"):
            result = dispatch_message(
                Channel.EMAIL,
                "billing@acme.com",
                "Dear client, a mock reminder body.",
                invoice_id=_INVOICE_ID,
                language="EN",
            )
    finally:
        broken.dispose()

    assert result.success
    assert "SQL state transition unavailable" in caplog.text
    assert result.message_id.startswith("MOCK-EMAIL-")
    assert "EMAIL MESSAGE SENT" in capsys.readouterr().out
