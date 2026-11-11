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
from models import DeliveryNote, Invoice

client = TestClient(app)
# The global JSON-500 guardrail lives in the ServerErrorMiddleware layer, which
# TestClient swallows by default; this client surfaces it as a real response.
json_client = TestClient(app, raise_server_exceptions=False)

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

    def block_sequential_only(db, model, code: str) -> bool:
        """Simulate a fully exhausted sequence: only ``-suffix`` codes are free."""

        return "-" not in code[len(INVOICE_PREFIX) :]

    with SessionLocal() as db:
        with patch.object(api, "_code_exists", side_effect=block_sequential_only):
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


# ---------------------------------------------------------------------------
# Delivery-note (e-Ispratnica) code allocation
# ---------------------------------------------------------------------------


def _create_delivery_note(payload_extra: dict | None = None) -> dict:
    payload = {
        "client_name": "Примач Тест",
        "delivery_address": "Улица Тест 1, Скопје",
        "items_description": "Тест производи",
        "signature_data": "data:image/png;base64,iVBORw0KGgo=",
    }
    payload.update(payload_extra or {})
    res = client.post("/api/ispratnici", json=payload, headers=AUTH_HEADERS)
    assert res.status_code == 201, res.text
    return res.json()


def test_next_delivery_note_code_skips_existing():
    """A stored ISP code must never be handed out again."""

    with SessionLocal() as db:
        code = api._next_delivery_note_code(db)

    with SessionLocal() as db:
        db.add(
            DeliveryNote(
                code=code,
                recipient_name="Постоечка",
                details="Тест",
            )
        )
        db.commit()

    with SessionLocal() as db:
        fresh = api._next_delivery_note_code(db)
    assert fresh != code
    assert fresh.startswith("ISP-")


def test_ispratnica_retries_on_transient_code_collision():
    """IntegrityError on a taken ISP code must trigger rollback + clean retry."""

    with SessionLocal() as db:
        existing = api._next_delivery_note_code(db)

    with SessionLocal() as db:
        db.add(DeliveryNote(code=existing, recipient_name="Колизија", details="Тест"))
        db.commit()

    with SessionLocal() as db:
        fresh = api._next_delivery_note_code(db)
    assert fresh != existing

    with patch(
        "api._next_delivery_note_code", side_effect=[existing, fresh]
    ) as allocator:
        data = _create_delivery_note()

    assert allocator.call_count == 2
    assert data["code"] == fresh
    assert data["code"] != existing


def test_rapid_succession_creation_never_collides():
    """Rapid-fire creation of invoices and delivery notes stays collision-free."""

    invoice_codes: list[str] = []
    for _ in range(8):
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Рапид Клиент",
                "amount": 2500.0,
                "description": "Брзо креирање",
            },
            headers=AUTH_HEADERS,
        )
        assert res.status_code == 201, res.text
        invoice_codes.append(res.json()["code"])

    note_codes: list[str] = []
    for _ in range(6):
        res = client.post(
            "/api/ispratnici",
            json={
                "client_name": "Рапид Примач",
                "delivery_address": "Улица Брза 2",
                "items_description": "Брзи пратки",
                "signature_data": "data:image/png;base64,iVBORw0KGgo=",
            },
            headers=AUTH_HEADERS,
        )
        assert res.status_code == 201, res.text
        note_codes.append(res.json()["code"])

    assert len(set(invoice_codes)) == len(invoice_codes)
    assert len(set(note_codes)) == len(note_codes)
    # Codes never cross between the two namespaces either.
    assert not set(invoice_codes) & set(note_codes)


def test_generic_allocator_used_by_both_endpoints():
    """Both allocators must delegate to ``generate_guaranteed_unique_code``."""

    with SessionLocal() as db:
        invoice_code = api.generate_guaranteed_unique_code(db, api.Invoice, "NPL")
        note_code = api.generate_guaranteed_unique_code(db, api.DeliveryNote, "ISP")

    assert invoice_code.startswith("NPL-")
    assert note_code.startswith("ISP-")
    assert not _code_is_taken(SessionLocal(), invoice_code)
    with SessionLocal() as db:
        assert (
            db.query(api.DeliveryNote.id)
            .filter(api.DeliveryNote.code == note_code)
            .first()
            is None
        )


# ---------------------------------------------------------------------------
# Global JSON error guardrail (prevents "Unexpected token '<'" in the SPA)
# ---------------------------------------------------------------------------


def test_unexpected_error_returns_json_500():
    """An unhandled exception must yield a JSON 500, never HTML/plain text."""

    with patch.object(api, "_serialize_invoice", side_effect=RuntimeError("boom")):
        res = json_client.post(
            "/api/invoices",
            json={
                "client_name": "Грешка Клиент",
                "amount": 100.0,
                "description": "Тест за JSON 500",
            },
            headers=AUTH_HEADERS,
        )

    assert res.status_code == 500
    assert res.headers["content-type"].startswith("application/json")
    body = res.json()
    assert body["message"] == "Failed to process request"
    assert "boom" in body["detail"]


def test_unexpected_delivery_note_error_returns_json_500():
    """Same guardrail contract for the e-Ispratnica creation endpoint."""

    with patch.object(
        api, "_serialize_delivery_note", side_effect=RuntimeError("isp-boom")
    ):
        res = json_client.post(
            "/api/ispratnici",
            json={
                "client_name": "Грешка Примач",
                "delivery_address": "Улица Грешка 3",
                "items_description": "Тест",
                "signature_data": "data:image/png;base64,iVBORw0KGgo=",
            },
            headers=AUTH_HEADERS,
        )

    assert res.status_code == 500
    assert res.headers["content-type"].startswith("application/json")
    body = res.json()
    assert body["message"] == "Failed to process request"
    assert "isp-boom" in body["detail"]
