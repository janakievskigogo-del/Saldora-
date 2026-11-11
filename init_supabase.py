#!/usr/bin/env python3
"""Bootstrap the Supabase PostgreSQL schema for Saldora.

DBA one-shot script:

1. Reads ``DATABASE_URL`` from the environment or ``.env`` (never hardcoded).
2. Standardises the legacy ``postgres://`` scheme to ``postgresql://`` for
   SQLAlchemy 2.x compatibility (Supabase hands out ``postgres://``).
3. Executes ``models.Base.metadata.create_all(bind=engine)`` — idempotent:
   existing tables, columns and enum types are left untouched.
4. Verifies the result: connects, prints the server version, lists every table
   in the database with its row count, and cross-checks against every table
   declared in ``models.Base.metadata``.

Usage:
    python init_supabase.py            # create tables + verify
    python init_supabase.py --check    # verify only — no DDL is executed

Exit codes: 0 = all expected tables exist, 1 = configuration/connection/
verification failure.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

from models import Base

# Only quoted identifiers built from our own metadata / the inspector are
# interpolated into the COUNT(*) queries below; this is a second guard line.
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

EXAMPLE_URL = "DATABASE_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres"


def _normalise_postgres_url(url_string: str) -> str:
    """Convert ``postgres://`` to ``postgresql://`` (SQLAlchemy 2 requirement)."""

    if url_string.startswith("postgres://"):
        return "postgresql://" + url_string[len("postgres://") :]
    return url_string


def _masked(url_string: str) -> str:
    """Render the connection target without leaking the password."""

    url = make_url(url_string)
    return f"{url.drivername}://{url.username}:***@{url.host}:{url.port}/{url.database}"


def _build_engine(url_string: str) -> Engine:
    """Production-grade engine: Supabase pooler-friendly pooling + TLS enforced.

    Supabase endpoints terminate TLS and reject plaintext connections, so
    ``sslmode=require`` is appended when the caller did not specify one.
    """

    parsed = make_url(url_string)
    if "sslmode" not in parsed.query:
        url_string += ("&" if "?" in url_string else "?") + "sslmode=require"

    return create_engine(
        url_string,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_recycle=300,
        connect_args={"connect_timeout": 15},
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create and verify the Saldora schema on Supabase PostgreSQL."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify only — connect and report, no DDL is executed",
    )
    args = parser.parse_args()

    # override=False keeps an explicitly exported environment variable
    # authoritative over a .env value, mirroring database.py.
    load_dotenv(override=False)
    raw_url = os.environ.get("DATABASE_URL", "").strip()
    if not raw_url:
        print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
        print("Set it in .env or export it, for example:", file=sys.stderr)
        print(f"  {EXAMPLE_URL}", file=sys.stderr)
        return 1

    url = _normalise_postgres_url(raw_url)
    if make_url(url).get_backend_name() != "postgresql":
        backend = make_url(url).get_backend_name()
        print(
            f"ERROR: this script targets PostgreSQL (Supabase); got backend {backend!r}.",
            file=sys.stderr,
        )
        print("Point DATABASE_URL at your Supabase Postgres instance.", file=sys.stderr)
        return 1

    engine = _build_engine(url)
    print(f"Target: {_masked(url)}")

    # ------------------------------------------------------------------ #
    # Connectivity check (fails fast with the server's own error message) #
    # ------------------------------------------------------------------ #
    try:
        with engine.connect() as conn:
            server_version = conn.execute(text("SELECT version()")).scalar_one()
        print(f"Server: {server_version}")
    except Exception as exc:  # noqa: BLE001 - every driver error is reportable
        print(f"ERROR: cannot connect to the database: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------ #
    # DDL: create every table declared in models.py (idempotent)          #
    # ------------------------------------------------------------------ #
    if args.check:
        print("Check-only mode: skipping DDL.")
    else:
        Base.metadata.create_all(bind=engine)
        print(
            f"create_all: OK — {len(Base.metadata.tables)} tables defined in models.py "
            "(existing tables untouched)."
        )

    # ------------------------------------------------------------------ #
    # Verification: list every table actually present in the database     #
    # ------------------------------------------------------------------ #
    inspector = inspect(engine)
    existing = sorted(set(inspector.get_table_names()))
    expected = sorted(Base.metadata.tables)
    missing = [table for table in expected if table not in existing]
    extra = [table for table in existing if table not in expected]

    print(f"\nTables in database ({len(existing)}):")
    for table in existing:
        marker = "OK" if table in expected else "EXTRA"
        print(f"  [{marker:5}] {table}")

    if missing:
        print("\nMissing expected tables:")
        for table in missing:
            print(f"  [MISSING] {table}")

    # Row counts give the DBA a quick data-integrity sanity check.
    if not missing:
        print("\nRow counts:")
        try:
            with engine.connect() as conn:
                for table in expected:
                    if not _SAFE_IDENTIFIER.match(table):
                        print(f"  {table}: skipped (unusual identifier)")
                        continue
                    count = conn.execute(
                        text(f'SELECT COUNT(*) FROM "{table}"')
                    ).scalar_one()
                    print(f"  {table}: {count}")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: row-count verification failed: {exc}", file=sys.stderr)

    engine.dispose()

    if missing:
        print(f"\nRESULT: FAILED — {len(missing)} of {len(expected)} expected tables missing.")
        return 1

    print(f"\nRESULT: OK — all {len(expected)} expected tables exist in the database.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
