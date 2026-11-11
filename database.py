"""SQLAlchemy engine and session layer for Saldora."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from models import Base

__all__ = ["DATABASE_URL", "engine", "SessionLocal", "get_db", "init_db"]

# ``DATABASE_URL`` is read at import time, which can happen before api.py's
# load_dotenv() call — so resolve it from .env here too. override=False keeps
# an explicitly exported environment variable authoritative.
load_dotenv(override=False)


def _normalise_postgres_url(url_string: str) -> str:
    """Convert ``postgres://`` to ``postgresql://`` for SQLAlchemy compatibility.

    Supabase and most PaaS providers hand out ``postgres://`` connection strings;
    SQLAlchemy 2.x only accepts the ``postgresql://`` scheme.
    """

    if url_string.startswith("postgres://"):
        return "postgresql://" + url_string[len("postgres://") :]
    return url_string


DATABASE_URL = _normalise_postgres_url(
    os.environ.get("DATABASE_URL", "sqlite:///saldora.db")
)


def _ensure_sqlite_directory(url_string: str) -> None:
    url = make_url(url_string)
    if url.get_backend_name() != "sqlite":
        return
    database = url.database
    if not database or database == ":memory:":
        return
    directory = os.path.dirname(database)
    if directory:
        os.makedirs(directory, exist_ok=True)


_ensure_sqlite_directory(DATABASE_URL)


def _connect_args(url_string: str) -> dict[str, object]:
    url = make_url(url_string)
    if url.get_backend_name() == "sqlite":
        return {"check_same_thread": False}
    return {}


_engine_backend = make_url(DATABASE_URL).get_backend_name()

if _engine_backend == "sqlite":
    engine: Engine = create_engine(
        DATABASE_URL,
        echo=False,
        connect_args=_connect_args(DATABASE_URL),
    )
else:
    # PostgreSQL / Supabase: production connection pooling.
    # - ``pool_pre_ping``  rebuilds connections dropped by Supabase's pooler or
    #   idle disconnects instead of surfacing them as "server closed" 500s.
    # - ``pool_recycle``   proactively renews connections before server-side
    #   timeouts can kill them (Vercel/Supabase idle limits).
    # - ``pool_size``/``max_overflow``  10 steady + up to 20 burst connections,
    #   sized for serverless/API concurrency without exhausting Supabase's
    #   direct-connection cap.
    engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=300,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Create all tables defined in models.py if they don't exist and ensure schema migrations."""
    Base.metadata.create_all(bind=engine)
    if make_url(engine.url).get_backend_name() != "sqlite":
        # The additive migrations below are SQLite-only (PRAGMA introspection).
        # On PostgreSQL the schema is fully covered by ``create_all`` —
        # models.py already declares every migrated column — plus external
        # migrations managed in Supabase.
        return
    with engine.begin() as conn:
        for table in ["clients", "invoices", "delivery_notes", "email_drafts"]:
            result = conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
            existing_cols = [r[1] for r in result]
            if "user_id" not in existing_cols:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN user_id INTEGER REFERENCES users(id)")

        # Progressive-registration profile field (ЕДБ). ``create_all`` does not
        # alter existing tables, so add the column to already-shipped databases.
        user_cols = [
            row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
        ]
        if "tax_id" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN tax_id VARCHAR(13)")
        if "is_verified" not in user_cols:
            # Pre-OTP accounts (incl. the seeded demo profile) are trusted and
            # stay usable; the otp_codes table itself is created by create_all.
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN is_verified BOOLEAN NOT NULL DEFAULT 0"
            )
            conn.exec_driver_sql("UPDATE users SET is_verified = 1")

        # Terms of Service consent (registration wizard). Existing accounts keep
        # working because the column defaults to false; new wizard signups must
        # explicitly accept before their row is created.
        if "terms_accepted" not in user_cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN terms_accepted BOOLEAN NOT NULL DEFAULT 0"
            )
        if "terms_accepted_at" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN terms_accepted_at DATETIME")
        if "terms_version" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN terms_version VARCHAR(20)")

        # Per-account Documenso e-signature configuration (Settings view).
        if "documenso_api_key" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN documenso_api_key VARCHAR(500)")
        if "documenso_base_url" not in user_cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN documenso_base_url VARCHAR(500)")

        # Documenso document lifecycle attached to each e-Ispratnica. Existing
        # notes are backfilled as NOT_SENT so the status badge stays truthful.
        note_cols = [
            row[1]
            for row in conn.exec_driver_sql("PRAGMA table_info(delivery_notes)").fetchall()
        ]
        note_migrations = {
            "documenso_document_id": "VARCHAR(100)",
            "documenso_status": "VARCHAR(30) NOT NULL DEFAULT 'NOT_SENT'",
            "documenso_signing_url": "TEXT",
            "documenso_recipient_email": "VARCHAR(255)",
            "documenso_signed_at": "DATETIME",
            "documenso_signature_log": "TEXT",
            "documenso_verification_hash": "VARCHAR(128)",
        }
        for column, ddl in note_migrations.items():
            if column not in note_cols:
                conn.exec_driver_sql(
                    f"ALTER TABLE delivery_notes ADD COLUMN {column} {ddl}"
                )