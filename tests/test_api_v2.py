"""Tests for the upgraded institutional Saldora API endpoints with JWT Auth & Data Isolation."""

import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
from database import SessionLocal
from models import EmailDraft

client = TestClient(app)


def get_authenticated_headers(email: str | None = None, password: str = "SecurePass123!") -> dict[str, str]:
    """Helper to register/login a user and return the Authorization header."""
    user_email = email or f"user_{uuid.uuid4().hex[:8]}@example.com"
    # Attempt signup first
    signup_res = client.post(
        "/api/signup",
        json={"email": user_email, "password": password, "company_name": "Тест Корпорација ДООЕЛ"},
    )
    if signup_res.status_code == 201:
        token = signup_res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    # If already exists, log in
    login_res = client.post("/api/login", json={"email": user_email, "password": password})
    assert login_res.status_code == 200
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_auth_signup_login_and_me():
    """Verify signup, login, password verification, and /api/users/me."""
    unique_email = f"auth_test_{uuid.uuid4().hex[:8]}@saldora.ai"
    password = "securePassword123!"

    # 1. Signup
    signup_res = client.post(
        "/api/signup",
        json={
            "email": unique_email,
            "password": password,
            "company_name": "Автентична Фирма ДООЕЛ",
        },
    )
    assert signup_res.status_code == 201
    signup_data = signup_res.json()
    assert "access_token" in signup_data
    assert signup_data["token_type"] == "bearer"
    assert signup_data["user"]["email"] == unique_email
    assert signup_data["user"]["company_name"] == "Автентична Фирма ДООЕЛ"

    # Duplicate signup should fail
    dup_res = client.post(
        "/api/signup",
        json={
            "email": unique_email,
            "password": password,
            "company_name": "Дупликат",
        },
    )
    assert dup_res.status_code == 400

    # 2. Login with wrong password should fail
    bad_login = client.post(
        "/api/login",
        json={"email": unique_email, "password": "wrongPassword!"},
    )
    assert bad_login.status_code == 401

    # 3. Login with correct password
    good_login = client.post(
        "/api/login",
        json={"email": unique_email, "password": password},
    )
    assert good_login.status_code == 200
    token = good_login.json()["access_token"]

    # 4. /api/users/me without token should be 401
    unauth_me = client.get("/api/users/me")
    assert unauth_me.status_code == 401

    # 5. /api/users/me with Bearer token
    auth_me = client.get("/api/users/me", headers={"Authorization": f"Bearer {token}"})
    assert auth_me.status_code == 200
    me_data = auth_me.json()
    assert me_data["email"] == unique_email
    assert me_data["company_name"] == "Автентична Фирма ДООЕЛ"


def test_data_isolation_between_users():
    """Ensure User A cannot view, modify, or download User B's invoices or delivery notes."""
    headers_a = get_authenticated_headers()
    headers_b = get_authenticated_headers()

    # User A creates an invoice
    inv_payload = {
        "client_name": "Таен Клиент А",
        "amount": 50000.0,
        "description": "Доверлив проект А",
        "issue_date": "2026-11-04",
        "due_date": "2026-11-20",
    }
    create_res = client.post("/api/invoices", json=inv_payload, headers=headers_a)
    assert create_res.status_code == 201
    code = create_res.json()["code"]

    # User A sees the invoice in their list
    list_a = client.get("/api/invoices", headers=headers_a)
    assert list_a.status_code == 200
    codes_a = [inv["code"] for inv in list_a.json()]
    assert code in codes_a

    # User B CANNOT see User A's invoice in their list
    list_b = client.get("/api/invoices", headers=headers_b)
    assert list_b.status_code == 200
    codes_b = [inv["code"] for inv in list_b.json()]
    assert code not in codes_b

    # User B cannot download User A's XML (404)
    xml_b = client.get(f"/api/invoices/{code}/xml", headers=headers_b)
    assert xml_b.status_code == 404

    # User B cannot approve User A's invoice (403 Forbidden — explicit write denial)
    approve_b = client.post(
        f"/api/invoices/{code}/approve",
        json={"approved_by": "Хакер"},
        headers=headers_b,
    )
    assert approve_b.status_code == 403


def test_create_invoice_with_ubl_fields_and_xml_generation():
    headers = get_authenticated_headers()
    payload = {
        "client_name": "Тест Фирма ДООЕЛ",
        "amount": 11800.0,
        "description": "Испорака на софтверски лиценци",
        "client_tax_id": "4030009876543",
        "client_address": "ул. Рузвелтова бр. 5, Скопје",
        "issue_date": "2026-11-04",
        "due_date": "2026-11-20",
        "vat_rate": 18.0,
    }
    res = client.post("/api/invoices", json=payload, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["client_name"] == "Тест Фирма ДООЕЛ"
    assert data["net_amount"] == 10000.0
    assert data["vat_amount"] == 1800.0
    assert data["total_amount"] == 11800.0
    assert data["is_approved"] is False
    assert data["xml_file_path"] is not None

    code = data["code"]

    # Verify XML download
    xml_res = client.get(f"/api/invoices/{code}/xml", headers=headers)
    assert xml_res.status_code == 200
    assert "application/xml" in xml_res.headers.get("content-type", "")
    assert f"<cbc:ID>{code}</cbc:ID>" in xml_res.text

    # Verify manual approval
    app_res = client.post(
        f"/api/invoices/{code}/approve",
        json={"approved_by": "Иван Сметководител"},
        headers=headers,
    )
    assert app_res.status_code == 200
    app_data = app_res.json()
    assert app_data["is_approved"] is True
    assert app_data["approved_by"] == "Иван Сметководител"

    # Verify storniraj route
    storno_res = client.post(
        f"/api/invoices/{code}/storniraj",
        json={"reason": "Тест сторнирање"},
        headers=headers,
    )
    assert storno_res.status_code == 200
    storno_data = storno_res.json()
    assert storno_data["status"] == "STORNO"


def test_create_and_list_ispratnici():
    headers = get_authenticated_headers()
    isp_payload = {
        "client_name": "Маркет Експрес",
        "delivery_address": "Индустриска зона Визбегово",
        "items_description": "20x Пакети со производи",
        "issue_date": "2026-11-04",
        "signature_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    }
    res = client.post("/api/ispratnici", json=isp_payload, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["client_name"] == "Маркет Експрес"
    assert data["is_signed"] is True
    assert data["ujp_status"] == "Sent to UJP"

    # List delivery notes
    list_res = client.get("/api/ispratnici", headers=headers)
    assert list_res.status_code == 200
    notes = list_res.json()
    assert any(n["code"] == data["code"] for n in notes)

    # Full document detail by numeric id (used by the viewer modal)
    detail_res = client.get(f"/api/ispratnici/{data['id']}", headers=headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["code"] == data["code"]
    assert detail["client_name"] == isp_payload["client_name"]
    assert detail["delivery_address"] == isp_payload["delivery_address"]
    assert detail["items_description"] == isp_payload["items_description"]
    assert detail["issue_date"] == isp_payload["issue_date"]
    assert detail["ujp_status"] == "Sent to UJP"
    assert detail["signature_data"] == isp_payload["signature_data"]

    # Re-signing persists the new Base64 payload for the document viewer.
    updated_signature = "data:image/png;base64,U0lHTkVELVVQREFURUQ="
    sign_res = client.post(
        f"/api/ispratnica/{data['code']}/sign",
        json={"signature_data": updated_signature},
        headers=headers,
    )
    assert sign_res.status_code == 200
    resigned = client.get(f"/api/ispratnici/{data['id']}", headers=headers).json()
    assert resigned["signature_data"] == updated_signature

    # Code-based lookup resolves the same record.
    code_res = client.get(f"/api/ispratnici/{data['code']}", headers=headers)
    assert code_res.status_code == 200
    assert code_res.json()["id"] == data["id"]

    # Another tenant cannot read the document.
    other_headers = get_authenticated_headers()
    other_res = client.get(f"/api/ispratnici/{data['id']}", headers=other_headers)
    assert other_res.status_code == 404


def test_remind_attaches_xml_file():
    """Verify that outbound payment reminder emails attach the invoice's UBL XML file."""
    headers = get_authenticated_headers()
    # Create invoice first
    inv_payload = {
        "client_name": "Клиент за Потсетник",
        "amount": 20000.0,
        "description": "Тест потсетник за наплата",
    }
    create_res = client.post("/api/invoices", json=inv_payload, headers=headers)
    assert create_res.status_code == 201
    code = create_res.json()["code"]
    xml_path = create_res.json()["xml_file_path"]

    with patch("api.send_email_native", return_value=(True, "Success")) as mock_send:
        remind_res = client.post(
            f"/api/invoices/{code}/remind",
            json={"recipient_email": "klient@firma.mk"},
            headers=headers,
        )
        assert remind_res.status_code == 200
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        # args: recipient, subject, body, extra_headers, attachment_path
        assert args[0] == "klient@firma.mk"
        assert code in args[1]
        assert args[4] == xml_path


def test_email_drafts_workflow_approve_and_reject():
    """Verify human-in-the-loop workflow: list drafts, edit+approve, and reject."""
    headers = get_authenticated_headers()
    # Fetch current user id
    me_res = client.get("/api/users/me", headers=headers)
    user_id = me_res.json()["id"]

    with SessionLocal() as db:
        # Create draft 1 (to be approved)
        draft1 = EmailDraft(
            user_id=user_id,
            invoice_id="NPL-2026-0001",
            client_email="partner@kompanija.mk",
            incoming_subject="Saldora molba za plakanje na rati",
            incoming_body="Pocituvani, sakame da uplatime na 2 rati.",
            ai_draft_response="Почитувани, се согласуваме со уплата на 2 рати.",
            status="PENDING_APPROVAL",
        )
        # Create draft 2 (to be rejected)
        draft2 = EmailDraft(
            user_id=user_id,
            invoice_id="NPL-2026-0001",
            client_email="spam@domain.com",
            incoming_subject="Saldora spam",
            incoming_body="Spam message",
            ai_draft_response="Nepotrebna poraka",
            status="PENDING_APPROVAL",
        )
        db.add_all([draft1, draft2])
        db.commit()
        db.refresh(draft1)
        db.refresh(draft2)
        d1_id = draft1.id
        d2_id = draft2.id

    # 1. Check stats endpoint includes pending drafts
    stats_res = client.get("/api/stats", headers=headers)
    assert stats_res.status_code == 200
    assert stats_res.json()["pending_email_drafts"] >= 2

    # 2. Check list drafts
    list_res = client.get("/api/email-drafts?status=PENDING_APPROVAL", headers=headers)
    assert list_res.status_code == 200
    draft_ids = [d["id"] for d in list_res.json()]
    assert d1_id in draft_ids
    assert d2_id in draft_ids

    # 3. Approve draft 1 with custom edited text
    with patch("api.send_email_native", return_value=(True, "Success")) as mock_send:
        app_res = client.post(
            f"/api/email-drafts/{d1_id}/approve",
            json={"edited_body": "Почитувани, одобрено за плаќање на 2 рати. Со почит, Директор."},
            headers=headers,
        )
        assert app_res.status_code == 200
        mock_send.assert_called_once()
        send_args = mock_send.call_args[0]
        assert send_args[0] == "partner@kompanija.mk"
        assert "Директор" in send_args[2]

    # Verify status changed to SENT in DB
    with SessionLocal() as db:
        d1 = db.query(EmailDraft).filter(EmailDraft.id == d1_id).first()
        assert d1.status == "SENT"

    # 4. Reject draft 2
    rej_res = client.post(f"/api/email-drafts/{d2_id}/reject", headers=headers)
    assert rej_res.status_code == 200

    # Verify status changed to REJECTED in DB
    with SessionLocal() as db:
        d2 = db.query(EmailDraft).filter(EmailDraft.id == d2_id).first()
        assert d2.status == "REJECTED"
