"""Regression tests for fail-safe, timestamped invoice/delivery-note codes.

Code format: ``{INV|ISP}-{YYYYMMDD}-{HEX6}`` (e.g. ``INV-20261111-8A3F9E``).
Covers:
1. Format + timestamp bucket correctness.
2. DB verification: a candidate already stored is never handed out (the
   generator regenerates a fresh hex instead).
3. Bounded exhaustion: a broken uniqueness oracle raises RuntimeError, never
   hangs.
4. The API endpoint ``IntegrityError -> rollback -> regenerate -> recommit``
   retry loop for both invoices and delivery notes.
5. Rapid-fire creation stays collision-free.
6. The global JSON error guardrail (never an HTML/plain-text 500).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from api import _code_is_taken, app
from database import SessionLocal
from models import DeliveryNote, Invoice

client = TestClient(app)
# The global JSON-500 guardrail lives in the ServerErrorMiddleware layer, which
# TestClient swallows by default; this client surfaces it as a real response.
json_client = TestClient(app, raise_server_exceptions=False)

INVOICE_CODE_RE = re.compile(r"^INV-\d{8}-[0-9A-F]{6}$")
DELIVERY_NOTE_CODE_RE = re.compile(r"^ISP-\d{8}-[0-9A-F]{6}$")

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
        signup = client.post(
            "/api/login", json={"email": AUTH_EMAIL, "password": AUTH_PASSWORD}
        )
        assert signup.status_code == 200
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


# ---------------------------------------------------------------------------
# Generator format + uniqueness semantics
# ---------------------------------------------------------------------------


def test_generated_invoice_code_is_timestamped_hex():
    with SessionLocal() as db:
        code = api._next_invoice_code(db)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert INVOICE_CODE_RE.match(code), code
    assert code.startswith(f"INV-{stamp}-"), code
    assert not _code_is_taken(SessionLocal(), code)


def test_generated_delivery_note_code_is_timestamped_hex():
    with SessionLocal() as db:
        code = api._next_delivery_note_code(db)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert DELIVERY_NOTE_CODE_RE.match(code), code
    assert code.startswith(f"ISP-{stamp}-"), code


def test_allocator_verifies_uniqueness_against_db():
    """A candidate that already exists is rejected and a fresh hex is used.

    Pins the RNG to a known value so the first candidate equals the seeded
    row; the allocator must NOT hand it back (spec: ``filter_by(code=...)``
    verification) and must return the second candidate instead.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    taken = f"INV-{stamp}-DEADBE"
    _insert_invoice(taken)

    with SessionLocal() as db:
        with patch.object(api.secrets, "token_hex", side_effect=["deadbe", "cafe12"]):
            code = api._next_invoice_code(db)

    assert code == f"INV-{stamp}-CAFE12"
    assert code != taken


def test_allocator_gives_up_after_bounded_attempts():
    """A permanently-colliding oracle must raise, never hang."""

    with SessionLocal() as db:
        with (
            patch.object(api.secrets, "token_hex", return_value="deadbe"),
            patch.object(api, "_code_exists", return_value=True),
        ):
            try:
                api._next_invoice_code(db)
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError after bounded attempts")


def test_generic_allocator_used_by_both_models():
    with SessionLocal() as db:
        invoice_code = api.generate_guaranteed_unique_code(db, api.Invoice, "INV")
        note_code = api.generate_guaranteed_unique_code(db, api.DeliveryNote, "ISP")

    assert INVOICE_CODE_RE.match(invoice_code)
    assert DELIVERY_NOTE_CODE_RE.match(note_code)
    assert not _code_is_taken(SessionLocal(), invoice_code)
    with SessionLocal() as db:
        assert (
            db.query(api.DeliveryNote.id)
            .filter(api.DeliveryNote.code == note_code)
            .first()
            is None
        )


# ---------------------------------------------------------------------------
# Endpoint-level retry & rollback backstop
# ---------------------------------------------------------------------------


def test_api_retries_on_transient_code_collision():
    """IntegrityError on a taken code must trigger rollback + clean retry."""

    with SessionLocal() as db:
        existing = api._next_invoice_code(db)  # a code that is free right now
    _insert_invoice(existing)  # now it is taken (simulates a racing writer)

    with SessionLocal() as db:
        fresh = api._next_invoice_code(db)
    assert fresh != existing

    with patch(
        "api._next_invoice_code", side_effect=[existing, fresh]
    ) as allocator:
        data = _create_invoice()

    assert allocator.call_count == 2
    assert data["code"] == fresh
    assert data["code"] != existing


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

    assert all(INVOICE_CODE_RE.match(c) for c in invoice_codes)
    assert all(DELIVERY_NOTE_CODE_RE.match(c) for c in note_codes)
    assert len(set(invoice_codes)) == len(invoice_codes)
    assert len(set(note_codes)) == len(note_codes)
    # Codes never cross between the two namespaces either.
    assert not set(invoice_codes) & set(note_codes)


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
