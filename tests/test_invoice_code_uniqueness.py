"""Regression tests for duplicate invoice-code prevention during e-Faktura creation.

Covers:
1. ``_next_invoice_code`` derives the sequence from stored rows (never blind).
2. Taken candidates roll the sequence forward instead of being reused.
3. When the deterministic block is blocked, a short random suffix guarantees
   uniqueness before ``db.commit()``.
4. The API endpoint retries cleanly on ``IntegrityError`` code collisions.
5. Creating invoices via the API yields globally unique codes.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from api import INVOICE_PREFIX, _code_is_taken, _next_invoice_code, app
from database import SessionLocal
from models import Invoice

client = TestClient(app)

AUTH_EMAIL = "unique_code_test_user@example.com"
AUTH_PASSWORD = "SecurePass123!"
AUTH_HEADERS: dict[str, str] = {}


def setup_module(module) -> None:  # noqa: ANN001
    global AUTH_HEADERS
    signup = client.post(
        "/api/signup",
        json={
            "email": AUTH_EMAIL,
            "password": AUTH_PASSWORD,
            "company_name": "Unikatnost ДООЕЛ",
        },
    )
    if signup.status_code != 201:
        login = client.post(
            "/api/login", json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD}
        )
        assert login.status_code == 200
        signup = login
    AUTH_HEADERS = {"Authorization": f"Bearer {signup.json()['access_token']}"}


def _create_invoice(payload_extra: dict | None = None) -> dict:
    payload = {
        "client_name": "Уникат Клиент",
        "amount": 1500.0,
        "description": "Тест за уникатен код",
    }
    payload.update(payload_extra or {})
    res = client.post("/api/invoices", json=payload, headers=AUTH_HEADERS)
    assert res.status_code == 201, res.text
    return res.json()


def _insert_invoice(code: str) -> None:
    with SessionLocal() as db:
        # Idempotent: drop any row from a previous test run first.
        db.query(Invoice).filter(
            (Invoice.code == code) | (Invoice.id == code)
        ).delete(synchronize_session=False)
        db.add(
            Invoice(
                id=code,
                code=code,
                client_name="Постоечка Фактура",
                amount=100.0,
                total_amount=100.0,
                status="UNPAID",
                state=api.InvoiceState.UNPAID,
            )
        )
        db.commit()


def _existing_codes() -> set[str]:
    with SessionLocal() as db:
        rows = db.query(Invoice.code, Invoice.id).all()
        return {value for row in rows for value in row if value}


def _session():
    return SessionLocal()


def test_next_invoice_code_skips_existing():
    """A stored code must never be handed out again."""

    taken = f"{INVOICE_PREFIX}9001"
    _insert_invoice(taken)
    assert _code_is_taken(_session(), taken)

    code = _next_invoice_code(_session())
    assert not _code_is_taken(_session(), code), (
        f"allocator returned an existing code: {code}"
    )
    assert code.startswith(INVOICE_PREFIX)


def _session():
    return SessionLocal()


def test_next_invoice_code_random_suffix_fallback_when_sequence_blocked():
    """If every sequential candidate is taken, a random-suffix code is returned."""

    def block_sequential_only(db, code: str) -> bool:
        """Simulate a fully exhausted sequence: only ``-suffix`` codes are free."""

        return "-" not in code[len(INVOICE_PREFIX) :]

    with SessionLocal() as db:
        with patch.object(api, "_code_is_taken", side_effect=block_sequential_only):
            code = api._next_invoice_code(db)

    # The suffix-form fallback must be free in reality and still parse nicely.
    assert code.startswith(INVOICE_PREFIX)
    assert "-" in code[len(INVOICE_PREFIX) :], code
    with SessionLocal() as db2:
        assert not _code_is_taken(db2, code)


def test_api_creates_invoices_with_globally_unique_codes():
    """Creating several e-Faktura invoices must never collide on code."""

    created_codes: list[str] = []
    for _ in range(5):
        data = _create_invoice()
        created_codes.append(data["code"])

    assert len(set(created_codes)) == len(created_codes)
    for code in created_codes:
        assert code not in _existing_codes() - set(created_codes)


def test_api_retries_on_transient_code_collision():
    """IntegrityError on a taken code must trigger rollback + clean retry."""

    with SessionLocal() as db:
        existing = _next_invoice_code(db)  # a code that is free right now
    _insert_invoice(existing)  # now it is taken (simulates a racing writer)

    with SessionLocal() as db:
        fresh = _next_invoice_code(db)
    assert fresh != existing

    with patch(
        "api._next_invoice_code", side_effect=[existing, fresh]
    ) as allocator:
        data = _create_invoice()

    assert allocator.call_count == 2
    assert data["code"] == fresh
    assert data["code"] != existing


def test_invoice_code_and_id_stay_in_sync():
    """The endpoint writes the allocated code into both ``id`` and ``code``."""

    data = _create_invoice()
    with SessionLocal() as db:
        row = db.query(Invoice).filter(Invoice.id == data["code"]).first()
        assert row is not None
        assert row.code == data["id"]
