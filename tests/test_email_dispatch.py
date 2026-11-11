"""Contract tests for outbound email dispatch and honest HTTP outcomes.

Covers the transport priority (Resend -> SMTP), port/TLS handling, the strict
timeout, the RFC-safe ``From`` header, and the two endpoints that must never
report a delivery that did not happen: the signup OTP and the payment reminder.
"""

from __future__ import annotations

import smtplib
import uuid
from email.header import decode_header, make_header
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api
import bot_service
from api import app
from database import SessionLocal
from models import Client, OTPCode

client = TestClient(app)

STRONG_PASSWORD = "SecurePass123!"


class _RecordingSMTP:
    """Minimal ``smtplib.SMTP`` / ``SMTP_SSL`` stand-in for offline tests."""

    instances: list[_RecordingSMTP] = []
    fail_with: Exception | None = None

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_credentials: tuple[str, str] | None = None
        self.message = None
        type(self).instances.append(self)

    def __enter__(self) -> _RecordingSMTP:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def starttls(self) -> None:
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        self.login_credentials = (username, password)

    def send_message(self, message) -> None:
        failure = type(self).fail_with
        if failure is not None:
            raise failure
        self.message = message


class _FakeResendEmails:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.error: Exception | None = None

    def send(self, params: dict) -> dict:
        if self.error is not None:
            raise self.error
        self.calls.append(params)
        return {"id": "test-email-id"}


class _FakeResend:
    def __init__(self) -> None:
        self.api_key: str | None = None
        self.Emails = _FakeResendEmails()


@pytest.fixture(autouse=True)
def _reset_transport_recorders():
    _RecordingSMTP.instances = []
    _RecordingSMTP.fail_with = None
    yield
    _RecordingSMTP.instances = []
    _RecordingSMTP.fail_with = None


@pytest.fixture
def smtp_env(monkeypatch):
    """Deterministic SMTP-only configuration with a recording transport."""

    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("BUSINESS_NAME", "Saldora")
    monkeypatch.setenv("SMTP_SERVER", "smtp.test.mk")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USER", "billing@test.mk")
    monkeypatch.setenv("SMTP_PASSWORD", "  app-password  ")
    monkeypatch.setenv("SENDER_EMAIL", "billing@test.mk")
    monkeypatch.delenv("SMTP_USERNAME", raising=False)
    monkeypatch.delenv("SMTP_PASS", raising=False)
    monkeypatch.setattr(bot_service.smtplib, "SMTP", _RecordingSMTP)
    monkeypatch.setattr(bot_service.smtplib, "SMTP_SSL", _RecordingSMTP)
    return monkeypatch


# ---------------------------------------------------------------------------
# 1. Transport engine (bot_service.send_email_native)
# ---------------------------------------------------------------------------


def test_smtp_587_uses_starttls_timeout_and_stripped_password(smtp_env):
    ok, error = bot_service.send_email_native("klient@firma.mk", "Тема", "Текст")

    assert ok is True
    assert error is None
    assert len(_RecordingSMTP.instances) == 1

    smtp = _RecordingSMTP.instances[0]
    assert (smtp.host, smtp.port, smtp.timeout) == ("smtp.test.mk", 587, 12)
    assert smtp.started_tls is True
    assert smtp.login_credentials == ("billing@test.mk", "app-password")

    from_header = str(make_header(decode_header(smtp.message["From"])))
    assert "Saldora" in from_header
    assert "billing@test.mk" in from_header


def test_smtp_port_465_uses_implicit_tls(smtp_env):
    smtp_env.setenv("SMTP_PORT", "465")

    ok, error = bot_service.send_email_native("klient@firma.mk", "Тема", "Текст")

    assert ok is True
    assert error is None
    smtp = _RecordingSMTP.instances[0]
    assert smtp.port == 465
    assert smtp.started_tls is False


def test_resend_is_primary_when_api_key_present(smtp_env):
    fake_resend = _FakeResend()
    smtp_env.setenv("RESEND_API_KEY", "re_test_key")
    smtp_env.setenv("RESEND_FROM_EMAIL", "billing@saldora.ai")
    smtp_env.setattr(bot_service, "resend", fake_resend)

    ok, error = bot_service.send_email_native("klient@firma.mk", "Тема", "Текст")

    assert ok is True
    assert error is None
    assert fake_resend.api_key == "re_test_key"
    assert len(fake_resend.Emails.calls) == 1
    params = fake_resend.Emails.calls[0]
    assert params["to"] == "klient@firma.mk"
    assert params["from"] == "Saldora <billing@saldora.ai>"
    # SMTP is never touched when the primary provider succeeds.
    assert _RecordingSMTP.instances == []


def test_resend_failure_degrades_to_smtp(smtp_env):
    fake_resend = _FakeResend()
    fake_resend.Emails.error = RuntimeError("resend 500")
    smtp_env.setenv("RESEND_API_KEY", "re_test_key")
    smtp_env.setattr(bot_service, "resend", fake_resend)

    ok, error = bot_service.send_email_native("klient@firma.mk", "Тема", "Текст")

    assert ok is True
    assert error is None
    assert len(_RecordingSMTP.instances) == 1


def test_transport_failure_returns_explicit_error_and_logs_stderr(smtp_env, capsys):
    _RecordingSMTP.fail_with = smtplib.SMTPAuthenticationError(535, b"bad credentials")

    ok, error = bot_service.send_email_native("klient@firma.mk", "Тема", "Текст")

    assert ok is False
    assert error and "SMTP delivery failed" in error
    assert "bad credentials" in capsys.readouterr().err


def test_missing_smtp_config_is_reported_not_swallowed(smtp_env):
    smtp_env.delenv("SMTP_USER", raising=False)
    smtp_env.delenv("SMTP_PASSWORD", raising=False)
    smtp_env.delenv("SMTP_USERNAME", raising=False)
    smtp_env.delenv("BUSINESS_EMAIL", raising=False)
    smtp_env.delenv("SENDER_EMAIL", raising=False)
    smtp_env.setattr(bot_service, "SMTP_USERNAME", "")
    smtp_env.setattr(bot_service, "SMTP_PASSWORD", "")
    smtp_env.setattr(bot_service, "SENDER_EMAIL", "")

    ok, error = bot_service.send_email_native("klient@firma.mk", "Тема", "Текст")

    assert ok is False
    assert error == "Missing SMTP config in .env"
    assert _RecordingSMTP.instances == []


# ---------------------------------------------------------------------------
# 2. Login verification code endpoint
# ---------------------------------------------------------------------------


def _unique_email() -> str:
    return f"dispatch_{uuid.uuid4().hex[:10]}@saldora.ai"


def test_otp_transport_failure_returns_json_502_and_keeps_code_persisted():
    email = _unique_email()

    with patch.object(api, "send_email_native", return_value=(False, "smtp timeout")):
        res = client.post("/api/signup/request-otp", json={"email": email})

    assert res.status_code == 502, res.text
    assert res.headers["content-type"].startswith("application/json")
    assert res.json()["detail"] == "Email delivery failed: smtp timeout"

    # The OTP was committed BEFORE the dispatch attempt, so the failure did not
    # roll it back and the wizard can retry with a fresh code.
    with SessionLocal() as db:
        otp = (
            db.query(OTPCode)
            .filter(OTPCode.email == email, OTPCode.is_used.is_(False))
            .first()
        )
        assert otp is not None
        assert len(otp.code) == 6 and otp.code.isdigit()


def test_otp_success_returns_explicit_success_payload():
    email = _unique_email()

    with patch.object(
        api, "send_email_native", return_value=(True, None)
    ) as mock_send:
        res = client.post("/api/signup/request-otp", json={"email": email})

    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "success"
    assert data["message"] == "Verification code sent"
    assert data["expires_in_seconds"] == 300
    assert mock_send.call_count == 1
    assert mock_send.call_args.kwargs.get("html_body")


# ---------------------------------------------------------------------------
# 3. Invoice payment reminder endpoint
# ---------------------------------------------------------------------------


def _auth_headers() -> dict[str, str]:
    email = f"reminder_{uuid.uuid4().hex[:8]}@saldora.ai"
    signup = client.post(
        "/api/signup",
        json={
            "email": email,
            "password": STRONG_PASSWORD,
            "company_name": "Потсетник Тест ДООЕЛ",
        },
    )
    assert signup.status_code == 201, signup.text
    token = signup.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _create_invoice(
    headers: dict[str, str], *, client_tax_id: str | None = None
) -> dict:
    payload: dict[str, object] = {
        "client_name": "Клиент за Потсетник",
        "amount": 5000.0,
        "description": "Консултантски услуги",
        "due_date": "2026-12-31",
    }
    if client_tax_id:
        payload["client_tax_id"] = client_tax_id
    res = client.post("/api/invoices", json=payload, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()


def test_reminder_without_any_client_email_returns_400():
    headers = _auth_headers()
    invoice = _create_invoice(headers)

    with patch.object(api, "send_email_native") as mock_send:
        res = client.post(
            f"/api/invoices/{invoice['code']}/remind", json={}, headers=headers
        )

    assert res.status_code == 400, res.text
    assert res.json()["detail"] == "Клиентот нема внесено е-пошта за потсетник."
    assert mock_send.call_count == 0


def test_reminder_dispatches_cleanly_to_explicit_recipient():
    headers = _auth_headers()
    invoice = _create_invoice(headers)

    with patch.object(
        api, "send_email_native", return_value=(True, None)
    ) as mock_send:
        res = client.post(
            f"/api/invoices/{invoice['code']}/remind",
            json={"recipient_email": "klient@firma.mk"},
            headers=headers,
        )

    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    assert body["status"] == "success"
    assert body["message"] == "Reminder sent successfully"
    assert mock_send.call_count == 1

    args = mock_send.call_args[0]
    assert args[0] == "klient@firma.mk"
    assert invoice["code"] in args[1]
    # Macedonian reminder: invoice code, MKD total, due date and instructions.
    assert invoice["code"] in args[2]
    assert "5,000.00 МКД" in args[2]
    assert "31.12.2026" in args[2]
    assert "Детали за плаќање" in args[2]


def test_reminder_uses_invoice_client_email_when_override_omitted():
    headers = _auth_headers()
    tax_id = f"4{uuid.uuid4().int % 10**12:012d}"
    invoice = _create_invoice(headers, client_tax_id=tax_id)

    with SessionLocal() as db:
        customer = db.query(Client).filter(Client.tax_id == tax_id).first()
        assert customer is not None
        customer.email = "customer@firma.mk"
        db.commit()

    with patch.object(
        api, "send_email_native", return_value=(True, None)
    ) as mock_send:
        res = client.post(
            f"/api/invoices/{invoice['code']}/remind", json={}, headers=headers
        )

    assert res.status_code == 200, res.text
    assert mock_send.call_args[0][0] == "customer@firma.mk"


def test_reminder_transport_failure_returns_json_502():
    headers = _auth_headers()
    invoice = _create_invoice(headers)

    with patch.object(
        api, "send_email_native", return_value=(False, "535 authentication failed")
    ):
        res = client.post(
            f"/api/invoices/{invoice['code']}/remind",
            json={"recipient_email": "klient@firma.mk"},
            headers=headers,
        )

    assert res.status_code == 502, res.text
    assert res.headers["content-type"].startswith("application/json")
    assert res.json()["detail"] == "Email delivery failed: 535 authentication failed"
