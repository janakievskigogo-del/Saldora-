"""One-shot migration of the legacy JSON audit store into SQLite.

Reads ``data/invoices.json`` (the ``InvoiceStore``/dataclass-era file) and seeds
the SQLAlchemy tables exactly once:

1. Skips silently when the file is missing **or** when the ``invoices`` table is
   already populated (no double-seeding).
2. Loads amounts with ``json.loads(..., parse_float=Decimal)`` so no money value
   ever degrades to a ``float``.
3. Reconstructs ``Client`` rows.  Legacy files may carry a ``clients`` array or
   attach client details directly to each invoice record; both shapes are
   merged by ``tax_id``, and tax IDs already present in
   ``clients``/duplicated inside the file are skipped to avoid
   unique-constraint failures.
4. Maps legacy statuses (``UNPAID``/``REMINDER_SENT``/...) onto
   :class:`InvoiceState` and stamps ``is_approved`` for anything past
   draft state.
5. Commits under a single transaction; any failure triggers a rollback and the
   exception propagates for the caller to handle.

Runs against the caller's ``Session`` (e.g. the one yielded by
``database.get_db``).  All logging goes through the PII-safe "saldora" logger,
because migration messages can contain tax IDs and e-mail addresses.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from logger import logger
from models import Client, Invoice, InvoiceState

__all__ = ["LEGACY_JSON_PATH", "seed_legacy_json_data"]

LEGACY_JSON_PATH = os.path.join("data", "invoices.json")

_DEFAULT_VAT_RATE = Decimal("18.00")
_CENT = Decimal("0.01")

# Legacy JSON-store statuses -> InvoiceState.  Anything unknown degrades to DRAFT.
_LEGACY_STATUS_TO_STATE: dict[str, InvoiceState] = {
    "draft": InvoiceState.DRAFT,
    "unpaid": InvoiceState.DRAFT,
    "approved": InvoiceState.APPROVED,
    "paid": InvoiceState.APPROVED,
    "ubl_generated": InvoiceState.UBL_GENERATED,
    "submitted": InvoiceState.UBL_GENERATED,
    "ujp_submitted": InvoiceState.UBL_GENERATED,
    "mock_submitted": InvoiceState.UBL_GENERATED,
    "sent": InvoiceState.SENT,
    "reminder_sent": InvoiceState.SENT,
    "delivered": InvoiceState.SENT,
    "failed": InvoiceState.FAILED,
    "validation_failed": InvoiceState.FAILED,
}


def _clean(raw: Any) -> str:
    return str(raw).strip() if raw is not None else ""


def _to_decimal(raw: Any) -> Decimal:
    """Coerce a legacy amount to ``Decimal``; floats never cross this boundary."""

    if raw is None:
        raise ValueError("amount missing")
    if isinstance(raw, Decimal):
        return raw
    return Decimal(str(raw).strip())


def _parse_naive_utc(raw: Any) -> datetime | None:
    """Parse an ISO timestamp; store naive UTC so SQLite keeps one format."""

    if raw is None or not _clean(raw):
        return None
    parsed = datetime.fromisoformat(str(raw).strip())
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _map_state(raw_status: Any) -> InvoiceState:
    return _LEGACY_STATUS_TO_STATE.get(_clean(raw_status).casefold(), InvoiceState.DRAFT)


def _merge_client(
    merged: dict[str, dict[str, str]],
    tax_id: str,
    *,
    name: str = "",
    email: str = "",
    phone: str = "",
) -> None:
    """Aggregate repeated client sightings, filling blank fields as they appear."""

    if not tax_id:
        return
    bucket = merged.setdefault(tax_id, {"tax_id": tax_id, "name": "", "email": "", "phone": ""})
    if name and not bucket["name"]:
        bucket["name"] = name
    if email and not bucket["email"]:
        bucket["email"] = email
    if phone and not bucket["phone"]:
        bucket["phone"] = phone


def _client_row(merged: dict[str, dict[str, str]], tax_id: str) -> dict[str, str]:
    row = merged[tax_id]
    # NOT NULL columns; legacy records frequently ship without phone/contact.
    return {
        "tax_id": tax_id,
        "name": row["name"] or "Unnamed client",
        "email": row["email"] or "",
        "phone": row["phone"] or "",
    }


def _invoice_row(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Build one insertable invoice dict, or ``None`` if it cannot be honored."""

    def pick(*keys: str) -> Any:
        for key in keys:
            value = entry.get(key)
            if value not in (None, ""):
                return value
        return None

    invoice_id = _clean(pick("invoice_id", "id", "invoice_number"))
    tax_id = _clean(pick("client_tax_id", "tax_id", "edb", "embg"))
    if not invoice_id:
        logger.warning("Legacy invoice %s has no invoice_id; skipped.", pick("invoice_number"))
        return None
    if not tax_id:
        logger.warning("Legacy invoice %s has no client_tax_id; skipped.", invoice_id)
        return None

    net: Decimal | None
    vat: Decimal | None
    total: Decimal | None
    raw_net, raw_vat, raw_total = pick("net_amount", "net", "subtotal"), pick(
        "vat_amount", "vat", "ddv"
    ), pick("total_amount", "amount", "amount_due", "gross_amount")
    raw_rate = pick("vat_rate", "ddv_rate", "vat_percent")
    try:
        net = _to_decimal(raw_net) if raw_net is not None else None
        vat = _to_decimal(raw_vat) if raw_vat is not None else None
        total = _to_decimal(raw_total) if raw_total is not None else None
        rate = _to_decimal(raw_rate) if raw_rate is not None else _DEFAULT_VAT_RATE

        # Mirror the legacy Excel loader's fallbacks, but in Decimal arithmetic.
        if net is None and total is not None and vat is not None:
            net = total - vat
        if net is None and total is not None:
            amount_only = total
            net, vat, rate = amount_only, Decimal("0.00"), Decimal("0.00")
        elif total is None and net is not None:
            if vat is None:
                vat = (net * (rate / Decimal("100"))).quantize(_CENT, rounding=ROUND_HALF_UP)
            total = net + vat
        if net is None or total is None:
            logger.warning("Legacy invoice %s has no usable amounts; skipped.", invoice_id)
            return None
        if vat is None:
            vat = Decimal("0.00")
        vat = vat.quantize(_CENT, rounding=ROUND_HALF_UP)
        net = net.quantize(_CENT, rounding=ROUND_HALF_UP)
        total = total.quantize(_CENT, rounding=ROUND_HALF_UP)
    except (ValueError, ArithmeticError) as exc:
        logger.warning("Legacy invoice %s has invalid amounts (%s); skipped.", invoice_id, exc)
        return None

    state = _map_state(pick("status", "payment_status"))
    created_at = None
    try:
        created_at = _parse_naive_utc(pick("created_at"))
    except ValueError as exc:
        logger.warning("Legacy invoice %s has invalid created_at (%s); using now().", invoice_id, exc)

    row = {
        "id": invoice_id,
        "client_tax_id": tax_id,
        "net_amount": net,
        "vat_rate": rate.quantize(_CENT, rounding=ROUND_HALF_UP),
        "vat_amount": vat,
        "total_amount": total,
        "state": state,
        "is_approved": state is not InvoiceState.DRAFT and state is not InvoiceState.FAILED,
    }
    if created_at is not None:
        # Omitted otherwise, so the created_at column default (naive UTC now) applies.
        row["created_at"] = created_at
    return row


def seed_legacy_json_data(db_session: Session) -> None:
    """Import ``data/invoices.json`` into the SQLite tables exactly once."""

    if not os.path.exists(LEGACY_JSON_PATH):
        logger.info("Legacy JSON %s not found; nothing to migrate.", LEGACY_JSON_PATH)
        return

    existing_invoices = db_session.scalar(select(func.count()).select_from(Invoice))
    if existing_invoices:
        logger.info(
            "Invoice table already holds %s row(s); skipping legacy JSON migration.",
            existing_invoices,
        )
        return

    logger.info("Migrating legacy invoice data from %s ...", LEGACY_JSON_PATH)

    with open(LEGACY_JSON_PATH, "r", encoding="utf-8") as handle:
        file_content = handle.read()
    # parse_float=Decimal keeps decimal-precision JSON money exact (no float).
    payload = json.loads(file_content, parse_float=Decimal)

    if isinstance(payload, dict):
        clients_raw = payload.get("clients") or []
        invoices_raw = payload.get("invoices") or []
    elif isinstance(payload, list):
        clients_raw, invoices_raw = [], payload
    else:
        raise ValueError(f"Legacy JSON {LEGACY_JSON_PATH} has an unsupported top-level shape.")

    if not isinstance(clients_raw, list) or not isinstance(invoices_raw, list):
        raise ValueError(f"Legacy JSON {LEGACY_JSON_PATH} has an unsupported section shape.")

    tax_ids_in_db = set(db_session.scalars(select(Client.tax_id)).all())

    merged_clients: dict[str, dict[str, str]] = {}
    for entry in clients_raw:
        if not isinstance(entry, dict):
            continue
        tax_id = _clean(entry.get("tax_id"))
        if not tax_id:
            continue
        _merge_client(
            merged_clients,
            tax_id,
            name=_clean(entry.get("name")),
            email=_clean(entry.get("email")),
            phone=_clean(entry.get("phone")),
        )

    for entry in invoices_raw:
        if not isinstance(entry, dict):
            continue
        tax_id = _clean(
            entry.get("client_tax_id") or entry.get("tax_id") or entry.get("edb") or entry.get("embg")
        )
        if not tax_id:
            continue
        _merge_client(
            merged_clients,
            tax_id,
            name=_clean(entry.get("client_name") or entry.get("client") or entry.get("customer")),
            email=_clean(entry.get("client_email") or entry.get("email")),
            phone=_clean(entry.get("client_phone") or entry.get("phone")),
        )

    client_rows = [
        _client_row(merged_clients, tax_id)
        for tax_id in merged_clients
        if tax_id not in tax_ids_in_db
    ]

    invoice_ids_in_db = set(db_session.scalars(select(Invoice.id)).all())
    invoice_rows: list[dict[str, Any]] = []
    seen_invoice_ids: set[str] = set()
    for entry in invoices_raw:
        if not isinstance(entry, dict):
            continue
        row = _invoice_row(entry)
        if row is None:
            continue
        if row["id"] in invoice_ids_in_db or row["id"] in seen_invoice_ids:
            logger.warning("Legacy invoice %s duplicates an existing ID; skipped.", row["id"])
            continue
        seen_invoice_ids.add(row["id"])
        invoice_rows.append(row)

    try:
        if client_rows:
            db_session.execute(insert(Client), client_rows)
        if invoice_rows:
            db_session.execute(insert(Invoice), invoice_rows)
        db_session.commit()
    except Exception as exc:
        db_session.rollback()
        logger.error("Legacy JSON migration failed and was rolled back: %s", exc)
        raise

    logger.info(
        "Legacy migration complete: %s client(s), %s invoice(s) inserted.",
        len(client_rows),
        len(invoice_rows),
    )
