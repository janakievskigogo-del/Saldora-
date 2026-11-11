"""End-to-end tests for the OTP email-verification signup flow.

SMTP dispatch is mocked so no real emails leave the machine during tests;
the verification-code email template itself is covered by a live manual E2E.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
from api import app
from database import SessionLocal
from models import OTPCode, User

client = TestClient(app)

STRONG_PASSWORD = "SecurePass123!"


def _unique_email() -> str:
    return f"otp_{uuid.uuid4().hex[:10]}@saldora.ai"


def _latest_otp_code(email: str) -> str:
    with SessionLocal() as db:
        otp = (
            db.query(OTPCode)
            .filter(OTPCode.email == email)
            .order_by(OTPCode.created_at.desc(), OTPCode.id.desc())
            .first()
        )
        assert otp is not None
        return otp.code


def _request_otp(email: str, sent_log: list[dict]):
    """request-otp with SMTP captured into ``sent_log``."""

    def fake_send(recipient, subject, body_text, *args, **kwargs):
        sent_log.append(
            {"recipient": recipient, "subject": subject, "body": body_text, "html": kwargs.get("html_body")}
        )
        return True, "Success"

    with patch.object(api, "send_email_native", side_effect=fake_send):
        return client.post(
            "/api/signup/request-otp",
            json={
                "email": email,
                "password": STRONG_PASSWORD,
                "password_confirm": STRONG_PASSWORD,
            },
        )


def test_request_otp_dispatches_and_persists():
    email = _unique_email()
    sent_log: list[dict] = []

    res = _request_otp(email, sent_log)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "success"
    assert data["message"] == "Verification code sent"
    assert data["expires_in_seconds"] == 300

    # Email dispatched exactly once, with the persisted code present
    assert len(sent_log) == 1
    assert sent_log[0]["recipient"] == email
    code = _latest_otp_code(email)
    assert len(code) == 6 and code.isdigit()
    assert code in sent_log[0]["body"]
    # Dark-mode HTML renders each digit as its own tile
    assert sent_log[0]["html"] and all(
        f">{digit}</td>" in sent_log[0]["html"] for digit in code
    )


def test_request_otp_rejects_verified_account():
    existing_email = _unique_email()
    res = client.post(
        "/api/signup",
        json={
            "email": existing_email,
            "password": STRONG_PASSWORD,
            "company_name": "Тест",
        },
    )
    assert res.status_code == 201

    res = client.post("/api/signup/request-otp", json={"email": existing_email})
    assert res.status_code == 400
    assert "Сметката веќе постои" in res.json()["detail"]


def test_verify_otp_valid_and_invalid():
    email = _unique_email()
    _request_otp(email, [])
    code = _latest_otp_code(email)

    ok = client.post("/api/signup/verify-otp", json={"email": email, "otp_code": code})
    assert ok.status_code == 200
    assert ok.json()["status"] == "valid"

    bad = client.post("/api/signup/verify-otp", json={"email": email, "otp_code": "000000"})
    assert bad.status_code == 400
    assert "Невалиден или истечен верификационен код" in bad.json()["detail"]


def test_complete_signup_full_flow():
    email = _unique_email()
    _request_otp(email, [])
    code = _latest_otp_code(email)

    res = client.post(
        "/api/signup/complete",
        json={
            "email": email,
            "password": STRONG_PASSWORD,
            "password_confirm": STRONG_PASSWORD,
            "company_name": "OTP Корпорација ДООЕЛ",
            "tax_id": "4030007777777",
            "otp_code": code,
            "terms_accepted": True,
        },
    )
    assert res.status_code == 201, res.text
    data = res.json()
    assert data["access_token"] and data["token_type"] == "bearer"
    assert data["user"]["email"] == email

    # Account is activated, consent is auditable, and login works immediately
    with SessionLocal() as db:
        user = db.query(User).filter(User.email == email).first()
        assert user is not None and user.is_verified is True
        assert user.terms_accepted is True
        assert user.terms_accepted_at is not None
        assert user.terms_version

    login = client.post("/api/login", json={"email": email, "password": STRONG_PASSWORD})
    assert login.status_code == 200

    # The OTP is consumed: replaying it must fail
    replay = client.post(
        "/api/signup/complete",
        json={
            "email": _unique_email(),
            "password": STRONG_PASSWORD,
            "password_confirm": STRONG_PASSWORD,
            "company_name": "Друга",
            "otp_code": code,
            "terms_accepted": True,
        },
    )
    assert replay.status_code == 400


def test_complete_signup_rejects_wrong_code():
    email = _unique_email()
    _request_otp(email, [])

    res = client.post(
        "/api/signup/complete",
        json={
            "email": email,
            "password": STRONG_PASSWORD,
            "password_confirm": STRONG_PASSWORD,
            "company_name": "OTP Корпорација",
            "otp_code": "999999",
            "terms_accepted": True,
        },
    )
    assert res.status_code == 400
    assert "Невалиден или истечен верификационен код" in res.json()["detail"]


def test_request_otp_rejects_weak_password():
    email = _unique_email()
    res = client.post(
        "/api/signup/request-otp",
        json={
            "email": email,
            "password": "pass",
            "password_confirm": "pass",
        },
    )
    assert res.status_code == 400
    assert "7 карактери" in res.json()["detail"]


def test_request_otp_rejects_password_mismatch():
    email = _unique_email()
    res = client.post(
        "/api/signup/request-otp",
        json={
            "email": email,
            "password": STRONG_PASSWORD,
            "password_confirm": "SecurePass123",
        },
    )
    assert res.status_code == 400
    assert "не се совпаѓаат" in res.json()["detail"]


def test_complete_signup_enforces_policy_and_consent():
    email = _unique_email()
    _request_otp(email, [])
    code = _latest_otp_code(email)

    weak = client.post(
        "/api/signup/complete",
        json={
            "email": email,
            "password": "password123",
            "company_name": "Корпорација",
            "otp_code": code,
            "terms_accepted": True,
        },
    )
    assert weak.status_code == 400
    assert "специјален знак" in weak.json()["detail"]

    no_consent = client.post(
        "/api/signup/complete",
        json={
            "email": email,
            "password": STRONG_PASSWORD,
            "password_confirm": STRONG_PASSWORD,
            "company_name": "Корпорација",
            "otp_code": code,
        },
    )
    assert no_consent.status_code == 400
    assert "Условите за користење" in no_consent.json()["detail"]


def test_request_otp_invalidates_previous_codes():
    """A second request must invalidate the first code (exactly one active)."""
    email = _unique_email()
    sent_log: list[dict] = []
    _request_otp(email, sent_log)
    first_code = _latest_otp_code(email)
    _request_otp(email, sent_log)
    second_code = _latest_otp_code(email)
    assert first_code != second_code

    stale = client.post("/api/signup/verify-otp", json={"email": email, "otp_code": first_code})
    assert stale.status_code == 400

    fresh = client.post("/api/signup/verify-otp", json={"email": email, "otp_code": second_code})
    assert fresh.status_code == 200
