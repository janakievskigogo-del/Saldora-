"""Regression tests for field-level PII encryption (models.EncryptedString).

The tests read rows back through **raw SQL** to prove ciphertext sits at rest,
and through the **ORM** to prove plaintext comes back — including for legacy
rows that were written in plain text before encryption existed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

import models
from models import Base, Client

_RAW_CLIENT = "INSERT INTO clients (tax_id, name, email, phone) VALUES (:t, :n, :e, :p)"


@pytest.fixture
def engine(tmp_path) -> Iterator[Engine]:
    engine = create_engine(f"sqlite:///{tmp_path / 'enctest.db'}")
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def fresh_cipher(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every crypto test from a cold singleton with no env override."""

    monkeypatch.delenv("DB_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(models, "_FERNET", None)


def test_client_pii_is_encrypted_at_rest_but_plaintext_via_orm(
    engine: Engine, fresh_cipher
) -> None:
    with Session(engine) as session:
        session.add(
            Client(
                tax_id="4030000000000",
                name="Acme LLC",
                email="billing@acme.com",
                phone="+38970123456",
            )
        )
        session.commit()

        with engine.connect() as conn:
            raw_email, raw_phone = conn.execute(
                text("SELECT email, phone FROM clients WHERE tax_id = :t"),
                {"t": "4030000000000"},
            ).fetchone()

    # The at-rest values are Fernet tokens -- ciphertext, never plaintext.
    for raw in (raw_email, raw_phone):
        assert raw.startswith("gAAA")
        assert "billing@acme.com" not in raw
        assert "+38970123456" not in raw

    persisted = Session(engine).scalar(
        select(Client).where(Client.tax_id == "4030000000000")
    )
    assert persisted.email == "billing@acme.com"
    assert persisted.phone == "+38970123456"
    assert persisted.name == "Acme LLC"


def test_legacy_plaintext_rows_pass_through(engine: Engine, fresh_cipher) -> None:
    """Rows written before this feature must read back verbatim, never crash."""

    with engine.begin() as conn:
        conn.execute(
            text(_RAW_CLIENT),
            {
                "t": "4030000000000",
                "n": "Legacy LLC",
                "e": "billing@acme.com",
                "p": "+38970123456",
            },
        )

    with Session(engine) as session:
        row = session.scalar(select(Client).where(Client.tax_id == "4030000000000"))
        assert row.name == "Legacy LLC"
        assert row.email == "billing@acme.com"
        assert row.phone == "+38970123456"


def test_invalid_env_key_fails_loud_on_write(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, fresh_cipher
) -> None:
    monkeypatch.setenv("DB_ENCRYPTION_KEY", "not-a-fernet-key")

    with pytest.raises(StatementError, match="not a valid Fernet key"):
        with Session(engine) as session:
            session.add(
                Client(
                    tax_id="4030000000000",
                    name="Acme LLC",
                    email="billing@acme.com",
                    phone="+38970123456",
                )
            )
            session.commit()
