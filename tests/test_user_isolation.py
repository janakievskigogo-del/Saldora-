"""Row-level user isolation tests: no cross-tenant reads or writes.

A user must never see another user's records (404 on read, existence hidden)
and must receive 403 Forbidden when attempting to modify an entity owned by
another account (write authorization, per spec).
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _signup(email: str) -> dict[str, str]:
    res = client.post(
        "/api/signup",
        json={
            "email": email,
            "password": "SecurePass123!",
            "company_name": "Изолациона Тест Фирма",
            "tax_id": "4030001234567",
        },
    )
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _make_invoice(headers: dict[str, str], client_name: str) -> str:
    res = client.post(
        "/api/invoices",
        headers=headers,
        json={
            "client_name": client_name,
            "amount": 1180.0,
            "description": "Изолационен тест",
        },
    )
    assert res.status_code == 201, res.text
    return res.json()["code"]


def test_cross_tenant_list_and_read_isolated():
    """User B's lists and reads never contain user A's records."""
    a = _signup(f"iso_a_{uuid.uuid4().hex[:8]}@saldora.ai")
    b = _signup(f"iso_b_{uuid.uuid4().hex[:8]}@saldora.ai")
    code_a = _make_invoice(a, "Клиент А")

    listings = client.get("/api/invoices", headers=b).json()
    assert all(inv["code"] != code_a for inv in listings)

    res = client.get(f"/api/invoices/{code_a}", headers=b)
    assert res.status_code == 404
    assert "Клиент А" not in res.text

    # Stats expose no cross-tenant aggregates
    stats_b = client.get("/api/stats", headers=b).json()
    assert stats_b["unpaid_amount"] == 0
    assert stats_b["unpaid_count"] == 0


def test_cross_tenant_write_returns_403():
    """User B modifying user A's invoice must get 403 Forbidden, not 404."""
    a = _signup(f"iso_c_{uuid.uuid4().hex[:8]}@saldora.ai")
    b = _signup(f"iso_d_{uuid.uuid4().hex[:8]}@saldora.ai")
    code_a = _make_invoice(a, "Клиент А")

    res = client.post(
        f"/api/invoices/{code_a}/approve",
        headers=b,
        json={"approved_by": "Напаѓач Б"},
    )
    assert res.status_code == 403

    res = client.post(f"/api/invoices/{code_a}/storno", headers=b, json={})
    assert res.status_code == 403

    res = client.post(
        f"/api/invoices/{code_a}/remind",
        headers=b,
        json={"recipient_email": "attacker@example.com"},
    )
    assert res.status_code == 403

    # The invoice must remain untouched by B's attempts
    after = client.get("/api/invoices", headers=a).json()
    mine = next(inv for inv in after if inv["code"] == code_a)
    assert mine["is_approved"] is False
    assert mine["status"] != "STORNO"


def test_cross_tenant_ispratnica_write_returns_403():
    """Signing another user's delivery note must return 403."""
    a = _signup(f"iso_e_{uuid.uuid4().hex[:8]}@saldora.ai")
    b = _signup(f"iso_f_{uuid.uuid4().hex[:8]}@saldora.ai")
    code_a = _make_invoice(a, "Клиент А")

    created = client.post(
        "/api/ispratnici",
        headers=a,
        json={
            "client_name": "Клиент А",
            "delivery_address": "ул. Тест 1",
            "items_description": "2x Тест артикл",
            "signature_data": "data:image/png;base64,iVBORw0KGgo=",
            "invoice_id": code_a,
        },
    )
    assert created.status_code == 201, created.text
    note_code = created.json()["code"]

    # B cannot even read A's note…
    assert client.get(f"/api/ispratnici/{note_code}", headers=b).status_code == 404
    # …and certainly cannot sign it
    res = client.post(
        f"/api/ispratnica/{note_code}/sign",
        headers=b,
        json={"signature_data": "data:image/png;base64,forged="},
    )
    assert res.status_code == 403


def test_email_draft_endpoints_scoped():
    """Email drafts: list scoping plus 403 on cross-tenant approve/reject."""
    from database import SessionLocal
    from models import EmailDraft, User

    a = _signup(f"iso_g_{uuid.uuid4().hex[:8]}@saldora.ai")
    b = _signup(f"iso_h_{uuid.uuid4().hex[:8]}@saldora.ai")

    # Seed a draft owned by A directly in the DB (no inbound email required)
    with SessionLocal() as db:
        a_email = client.get("/api/users/me", headers=a).json()["email"]
        owner = db.query(User).filter(User.email == a_email).first()
        draft = EmailDraft(
            user_id=owner.id,
            client_email="client@example.com",
            incoming_subject="Saldora - Барaње за места",
            incoming_body="Треба повеќе време за плаќање.",
            ai_draft_response="Можеме да понудиме 2 рати.",
        )
        db.add(draft)
        db.commit()
        draft_id = draft.id

    assert all(d["id"] != draft_id for d in client.get("/api/email-drafts", headers=b).json())

    res = client.post(f"/api/email-drafts/{draft_id}/approve", headers=b)
    assert res.status_code == 403

    res = client.post(f"/api/email-drafts/{draft_id}/reject", headers=b)
    assert res.status_code == 403
