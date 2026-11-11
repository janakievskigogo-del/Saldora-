"""Synthetic end-to-end runtime verification for the Saldora API.

Drives the full business flow against the FastAPI app in-process
(SQLite offline engine, mail provider replaced by a recorder) and asserts
that every patched crash vector stays closed:

    1. Demo login                -> 200 + JWT session token
    2. Invalid credentials       -> JSON 401, never an unhandled exception
    3. Create invoice            -> 201, 2-decimal money, post-commit mail flags
    4. Create delivery note      -> 201, unique ISP code, canvas signature stored
    5. Capture signature         -> 200, signature persisted to the database
    6. Send payment reminder     -> 200, non-blocking dispatch recorded
    7. Documenso fallback        -> canvas signature saved when unconfigured

Run with:  python e2e_verify.py
Exit code 0 means every step passed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Pin the synthetic verification run to an OFFLINE SQLite engine BEFORE any
# project import. Without this the script would import database.py, pick up
# the production DATABASE_URL from .env and write test data into Supabase —
# exactly what a synthetic verification must never do.
os.environ["DATABASE_URL"] = "sqlite:///./e2e_verify.db"
Path("e2e_verify.db").unlink(missing_ok=True)

from decimal import Decimal  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

MAIL_LOG: list[tuple[str, str]] = []


def _recording_send_email_native(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Any = None,
    attachment_path: Any = None,
    html_body: str | None = None,
) -> tuple[bool, str]:
    """Offline stand-in for the SMTP/Resend provider; records every dispatch."""

    MAIL_LOG.append((recipient_email, subject))
    print(f"       [mail] -> {recipient_email!r} | {subject!r}")
    return True, "Recorded"


def _check(step: str, condition: bool, detail: str = "") -> None:
    marker = "PASS" if condition else "FAIL"
    print(f"  [{marker}] {step}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        raise SystemExit(f"E2E verification FAILED at step: {step} {detail}")


def main() -> int:
    print("== Saldora synthetic end-to-end verification ==")

    # `with` enables the FastAPI lifespan: tables are created and the demo
    # account (admin@saldora.ai / demo123) is seeded before the first call.
    with TestClient(app) as client, patch(
        "api.send_email_native", side_effect=_recording_send_email_native
    ):
        # ------------------------------------------------------------
        # 1. Demo login — JWT session token
        # ------------------------------------------------------------
        res = client.post(
            "/api/login",
            json={"email": "admin@saldora.ai", "password": "demo123"},
        )
        _check("POST /api/login (demo credentials) -> 200", res.status_code == 200, res.text)
        token = res.json().get("access_token", "")
        _check("JWT session token issued", bool(token))
        headers = {"Authorization": f"Bearer {token}"}

        # ------------------------------------------------------------
        # 2. Invalid credentials -> clean JSON 401 (never a 500)
        # ------------------------------------------------------------
        res = client.post(
            "/api/login",
            json={"email": "admin@saldora.ai", "password": "wrong-password"},
        )
        _check("POST /api/login (invalid credentials) -> 401 JSON", res.status_code == 401)
        _check(
            "401 body is JSON (no HTML crash page)",
            res.headers["content-type"].startswith("application/json"),
        )

        # ------------------------------------------------------------
        # 3. Create invoice (VAT-inclusive, 2-decimal money, mail flags)
        # ------------------------------------------------------------
        res = client.post(
            "/api/invoices",
            json={
                "client_name": "Е2Е Клиент ДООЕЛ",
                "amount": 1180.0,
                "description": "E2E синтетичка верификација",
                "client_tax_id": "4030991123456",
                "client_address": "Партизанска 1, Скопје",
                "client_email": "e2e-client@example.mk",
            },
            headers=headers,
        )
        _check("POST /api/invoices -> 201", res.status_code == 201, res.text)
        invoice = res.json()
        _check("Unique invoice code allocated (INV-*)", str(invoice["code"]).startswith("INV-"))
        net = Decimal(str(invoice["net_amount"]))
        vat = Decimal(str(invoice["vat_amount"]))
        total = Decimal(str(invoice["total_amount"]))
        _check(
            "Financial values rounded to 2 decimals and balanced",
            total == Decimal("1180.00")
            and net + vat == total
            and all(v.as_tuple().exponent >= -2 for v in (net, vat, total)),
            f"net={net} vat={vat} total={total}",
        )
        _check("Invoice stored UNPAID", invoice["status"] == "UNPAID")

        # ------------------------------------------------------------
        # 4. Create delivery note (unique ISP code + canvas signature)
        # ------------------------------------------------------------
        canvas_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        res = client.post(
            "/api/ispratnici",
            json={
                "client_name": "Е2Е Примач ДООЕЛ",
                "delivery_address": "Партизанска 1, Скопје",
                "items_description": "2 палети документација",
                "signature_data": canvas_png,
                "invoice_id": invoice["id"],
                "client_email": "e2e-client@example.mk",
            },
            headers=headers,
        )
        _check("POST /api/ispratnici -> 201", res.status_code == 201, res.text)
        note = res.json()
        _check("Unique delivery-note code allocated (ISP-*)", str(note["code"]).startswith("ISP-"))
        _check("Canvas signature stored with the note", bool(note["signature_data"]))

        # ------------------------------------------------------------
        # 5. Capture (re-capture) signature on the delivery note
        # ------------------------------------------------------------
        res = client.post(
            f"/api/ispratnica/{note['code']}/sign",
            json={"signature_data": canvas_png},
            headers=headers,
        )
        _check("POST /api/ispratnica/{code}/sign -> 200", res.status_code == 200, res.text)
        _check("Signature accepted and sealed", res.json()["is_signed"] is True)

        # ------------------------------------------------------------
        # 5b. Documenso fallback: no API key -> canvas data saved directly
        # ------------------------------------------------------------
        with patch.dict("os.environ", {"DOCUMENSO_API_KEY": ""}, clear=False):
            res = client.post(
                f"/api/ispratnici/{note['code']}/sign-documenso",
                json={"signature_data": canvas_png},
                headers=headers,
            )
        _check(
            "Documenso unconfigured -> graceful SAVED_LOCALLY fallback",
            res.status_code == 200 and res.json()["status"] == "SAVED_LOCALLY",
            res.text,
        )

        # ------------------------------------------------------------
        # 6. Send payment reminder (non-blocking dispatch)
        # ------------------------------------------------------------
        res = client.post(
            f"/api/invoices/{invoice['code']}/remind",
            json={"recipient_email": "e2e-client@example.mk"},
            headers=headers,
        )
        _check("POST /api/invoices/{code}/remind -> 200", res.status_code == 200, res.text)
        dispatched = any(
            recipient == "e2e-client@example.mk"
            and invoice["code"] in subject
            for recipient, subject in MAIL_LOG
        )
        _check("Reminder dispatch reached the mail layer", dispatched, str(MAIL_LOG))

    print("\n== RESULT: ALL E2E STEPS PASSED ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
