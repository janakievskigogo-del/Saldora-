"""Regression guards for the Vercel/Supabase runtime crash vectors.

Covers the defensive patches applied by the reliability audit:

1. Vercel read-only filesystem guardrail (config.py, api.py, models.py,
   database.py) — artifact directories resolve to /tmp and directory
   creation can never raise ``OSError`` through to the caller.
2. Transaction isolation + non-blocking post-commit email dispatch in
   ``POST /api/invoices`` and ``POST /api/ispratnici`` — mail failure never
   downgrades the HTTP 201, it only sets ``email_sent=false`` + ``warning``.
3. Demo auth 401 contract and the Documenso→canvas-signature fallback.
4. Global JSON exception shield message contract.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import config
from api import app
from database import SessionLocal

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    """Register (or log in) a fresh isolated user and return Bearer headers."""

    email = f"guard_{uuid.uuid4().hex[:8]}@saldora.ai"
    password = "GuardPass123!"
    res = client.post(
        "/api/signup",
        json={
            "email": email,
            "password": password,
            "company_name": "Guard Firma ДООЕЛ",
        },
    )
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _create_note(headers: dict[str, str]) -> dict:
    """Create one e-Ispratnica and return its serialized payload."""

    res = client.post(
        "/api/ispratnici",
        json={
            "client_name": "Примач Тест",
            "delivery_address": "Улица Тест 1",
            "items_description": "Тест палети",
            "signature_data": "data:image/png;base64,iVBORw0KGgo=",
        },
        headers=headers,
    )
    assert res.status_code == 201, res.text
    return res.json()


# ---------------------------------------------------------------------------
# 1. Vercel-aware filesystem guardrail
# ---------------------------------------------------------------------------


def test_vercel_runtime_detects_marker(monkeypatch) -> None:
    monkeypatch.setenv("VERCEL", "1")
    assert config.is_vercel_runtime() is True
    monkeypatch.delenv("VERCEL", raising=False)
    assert config.is_vercel_runtime() is False


def test_vercel_output_dir_resolves_to_tmp(monkeypatch, tmp_path) -> None:
    """On Vercel the UBL audit dir must live under the OS temp dir."""

    monkeypatch.setenv("VERCEL", "1")
    settings = config.Settings(data_dir=tmp_path / "data")
    resolved = settings.ubl_output_dir
    assert str(resolved).startswith(os.path.realpath(config.tempfile.gettempdir()))
    assert resolved.name == "saldora-output"
    monkeypatch.delenv("VERCEL", raising=False)
    settings = config.Settings(data_dir=tmp_path / "data")
    assert settings.ubl_output_dir == (tmp_path / "data").parent / "output"


def test_ensure_directories_never_raises_on_read_only_fs(monkeypatch, tmp_path) -> None:
    """``ensure_directories`` must swallow OSError (read-only filesystem)."""

    settings = config.Settings(data_dir=tmp_path / "data")

    def read_only_mkdir(*args, **kwargs):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(Path, "mkdir", read_only_mkdir)
    settings.ensure_directories()  # must not raise


def test_ensure_writable_dir_degrades_gracefully(tmp_path) -> None:
    """``_ensure_writable_dir`` returns None instead of raising OSError."""

    unwritable = tmp_path / "blocked"
    with patch.object(Path, "mkdir", side_effect=OSError(30, "Read-only file system")):
        assert api._ensure_writable_dir(unwritable) is None


def test_xml_dir_resolver_local_vs_vercel(monkeypatch, tmp_path) -> None:
    """The api.XML_DIR resolver relocates artifacts on Vercel only."""

    default = tmp_path / "data" / "xmls"
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("VERCEL_ENV", raising=False)
    assert api.config.resolve_writable_dir(default) == default
    monkeypatch.setenv("VERCEL_ENV", "production")
    resolved = api.config.resolve_writable_dir(default)
    assert str(resolved).startswith(os.path.realpath(config.tempfile.gettempdir()))


def test_invoice_creation_survives_unwritable_output_dir(
    monkeypatch, tmp_path
) -> None:
    """Invoice creation must succeed even when the XML dir is unwritable."""

    headers = _auth_headers()
    with patch.object(
        api, "XML_DIR", tmp_path / "definitely-unwritable"
    ), patch.object(Path, "mkdir", side_effect=OSError(30, "Read-only file system")):
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Клиент RO",
                "amount": 250.0,
                "description": "Read-only FS симулација",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["code"].startswith("INV-")
    # No XML audit copy is available, but the invoice itself is durable.
    assert body["total_amount"] == 250.0


# ---------------------------------------------------------------------------
# 2. Transaction isolation & non-blocking post-commit email dispatch
# ---------------------------------------------------------------------------


def test_invoice_email_dispatch_success_and_payload_flags() -> None:
    headers = _auth_headers()
    with patch(
        "api.send_email_native", return_value=(True, "Success")
    ) as mock_send:
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Поштенски Клиент",
                "amount": 118.0,
                "description": "Диспечер тест",
                "client_email": "billing@client.mk",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["email_sent"] is True
    assert body["warning"] is None
    assert mock_send.call_count == 1
    dispatched_recipient = mock_send.call_args[0][0]
    assert dispatched_recipient == "billing@client.mk"


def test_invoice_email_dispatch_failure_keeps_http_201() -> None:
    """Mail failure must only set the warning flag, never the status code."""

    headers = _auth_headers()
    with patch(
        "api.send_email_native", return_value=(False, "Missing SMTP config in .env")
    ):
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Неуспешна Пошта",
                "amount": 90.0,
                "description": "Диспечер fallback тест",
                "client_email": "billing@client.mk",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["email_sent"] is False
    assert "Missing SMTP config" in body["warning"]

    # The invoice is still durably stored despite the mail failure.
    list_res = client.get("/api/invoices", headers=headers)
    assert list_res.status_code == 200
    assert any(
        invoice["client_name"] == "Неуспешна Пошта" for invoice in list_res.json()
    )


def test_invoice_email_dispatch_exception_is_isolated() -> None:
    """An exception inside the mail path must never fail the document."""

    headers = _auth_headers()
    with patch(
        "api.send_email_native", side_effect=Exception("smtp connection reset")
    ):
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Изловен Сокет",
                "amount": 60.0,
                "description": "Изолиран диспечер",
                "client_email": "billing@client.mk",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["email_sent"] is False
    assert "smtp connection reset" in body["warning"]


def test_delivery_note_email_dispatch_failure_keeps_http_201() -> None:
    headers = _auth_headers()
    with patch(
        "api.send_email_native", return_value=(False, "Missing SMTP config in .env")
    ):
        res = client.post(
            "/api/ispratnici",
            json={
                "client_name": "Примач Пошта",
                "delivery_address": "Улица Пошта 5",
                "items_description": "Палети",
                "signature_data": "data:image/png;base64,iVBORw0KGgo=",
                "client_email": "logistics@client.mk",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["email_sent"] is False
    assert "Missing SMTP config" in body["warning"]


def test_invoice_email_dispatch_timeout_releases_request(monkeypatch) -> None:
    """A wedged mail provider must hit the bounded dispatcher deadline."""

    monkeypatch.setattr(api, "EMAIL_DISPATCH_TIMEOUT_SECONDS", 0.2)
    headers = _auth_headers()

    def hung_sender(*args, **kwargs):
        time.sleep(3)
        return True, "Success"

    with patch("api.send_email_native", side_effect=hung_sender):
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Бавен Сервер",
                "amount": 45.0,
                "description": "Deadline тест",
                "client_email": "billing@client.mk",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["email_sent"] is False
    assert "timed out" in body["warning"]


def test_invoice_without_client_email_skips_dispatch() -> None:
    """Legacy payloads (no client_email) keep working with no mail attempt."""

    headers = _auth_headers()
    with patch("api.send_email_native") as mock_send:
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Тих Клиент",
                "amount": 33.0,
                "description": "Без пошта",
            },
            headers=headers,
        )
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["email_sent"] is False
    assert body["warning"] is None
    assert mock_send.call_count == 0


def test_create_rejects_header_injection_in_client_email() -> None:
    headers = _auth_headers()
    res = client.post(
        "/api/invoices",
        json={
            "client_name": "Злонамерен Клиент",
            "amount": 10.0,
            "description": "Header injection",
            "client_email": "victim@client.mk\r\nBcc: attacker@evil.mk",
        },
        headers=headers,
    )
    assert res.status_code == 422
    assert "line breaks" in res.json()["detail"][0]["msg"]


# ---------------------------------------------------------------------------
# 3. Demo auth contract & Documenso→canvas fallback
# ---------------------------------------------------------------------------


def test_login_invalid_credentials_return_401_json() -> None:
    res = client.post(
        "/api/login",
        json={"email": f"ghost_{uuid.uuid4().hex[:6]}@saldora.ai", "password": "nope"},
    )
    assert res.status_code == 401
    assert res.headers["content-type"].startswith("application/json")
    assert "Невалидна" in res.json()["detail"]


def test_login_malformed_json_returns_clean_4xx_never_500() -> None:
    res = client.post(
        "/api/login",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code in (400, 401, 422)
    assert res.headers["content-type"].startswith("application/json")


def test_login_blank_credentials_return_400_json() -> None:
    res = client.post("/api/login", json={"email": "", "password": ""})
    assert res.status_code == 400
    assert res.headers["content-type"].startswith("application/json")


def test_documenso_unconfigured_saves_canvas_signature() -> None:
    """Without any Documenso key the canvas PNG is stored in Supabase."""

    headers = _auth_headers()
    note = _create_note(headers)

    # Guarantee the environment key is absent for this account.
    with patch.dict(os.environ, {"DOCUMENSO_API_KEY": ""}, clear=False):
        res = client.post(
            f"/api/ispratnici/{note['code']}/sign-documenso",
            json={
                "signature_data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
            },
            headers=headers,
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "SAVED_LOCALLY"
    assert body["signing_url"] == ""
    assert body["doc_id"] == ""

    detail = client.get(f"/api/ispratnici/{note['code']}", headers=headers)
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["is_signed"] is True
    assert payload["signature_data"] == "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
    assert payload["documenso_status"] == "NOT_SENT"


def test_documenso_unconfigured_without_signature_returns_structured_503() -> None:
    headers = _auth_headers()
    note = _create_note(headers)
    with patch.dict(os.environ, {"DOCUMENSO_API_KEY": ""}, clear=False):
        res = client.post(
            f"/api/ispratnici/{note['code']}/sign-documenso",
            json={},
            headers=headers,
        )
    assert res.status_code == 503
    assert res.headers["content-type"].startswith("application/json")
    assert "Documenso" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 3b. Supabase foreign-key crash vectors (SQLite tests had FK enforcement off)
# ---------------------------------------------------------------------------


def test_invoice_with_new_client_tax_id_auto_registers_client() -> None:
    """A first invoice for an unseen client EDB must not FK-fail on Supabase.

    PostgreSQL enforces invoices.client_tax_id -> clients.tax_id; the
    endpoint registers the client record before the invoice is written.
    """

    headers = _auth_headers()
    fresh_edb = f"40{uuid.uuid4().int % 10**11:011d}"[:13]
    assert len(fresh_edb) == 13 and fresh_edb.isdigit()

    res = client.post(
        "/api/invoices",
        json={
            "client_name": "Нов ЕДБ Клиент",
            "amount": 130.0,
            "description": "FK regression",
            "client_tax_id": fresh_edb,
        },
        headers=headers,
    )
    assert res.status_code == 201, res.text
    assert res.json()["client_tax_id"] == fresh_edb

    # The client record now exists so the foreign key resolves.
    from models import Client as ClientModel

    with SessionLocal() as session:
        assert (
            session.query(ClientModel).filter(ClientModel.tax_id == fresh_edb).first()
            is not None
        )


def test_delivery_note_with_dangling_invoice_id_returns_clean_404() -> None:
    """A nonexistent invoice reference answers 404 JSON, not a burned 409."""

    headers = _auth_headers()
    res = client.post(
        "/api/ispratnici",
        json={
            "client_name": "Примач Без Фактура",
            "delivery_address": "Улица Тест 7",
            "items_description": "Тест",
            "signature_data": "data:image/png;base64,iVBORw0KGgo=",
            "invoice_id": "INV-00000000-DOESNOTEXIST",
        },
        headers=headers,
    )
    assert res.status_code == 404
    assert res.headers["content-type"].startswith("application/json")
    assert "INV-00000000-DOESNOTEXIST" in res.json()["detail"]


# ---------------------------------------------------------------------------
# 4. Global JSON exception shield
# ---------------------------------------------------------------------------


def test_global_shield_message_contract() -> None:
    """The unhandled-exception shield returns the canonical JSON 500 shape."""

    headers = _auth_headers()
    with patch.object(api, "_serialize_invoice", side_effect=RuntimeError("shield-boom")):
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Штит Клиент",
                "amount": 70.0,
                "description": "Штит тест",
            },
            headers=headers,
        )
    assert res.status_code == 500
    assert res.headers["content-type"].startswith("application/json")
    body = res.json()
    assert body["message"] == "Internal server error"
    assert "shield-boom" in body["detail"]
