"""Email delivery service for Saldora.

Supports both real sending (via SMTP/Gmail or Resend) and the existing mock
logger, plus the SQL-lifecycle ``SMTPDispatcher`` transport used by the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import os
import smtplib
import ssl
import logging
from decimal import Decimal
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape as html_escape
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import resend
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from config import Settings
from database import get_db
from legacy_models import Channel, DispatchResult
from models import Invoice, InvoiceState
from state_machine import InvalidStateTransitionError, transition_invoice_state
from utils import is_valid_email, normalize_phone, safe_filename
import models
import schemas
from logger import logger as pii_logger


class DispatchError(Exception):
    """Raised when a message cannot be dispatched to a valid destination."""

    pass


_SEPARATOR = "=" * 82
_SUBSEPARATOR = "-" * 82

logger = logging.getLogger(__name__)


def _message_id(prefix: str) -> str:
    return f"MOCK-{prefix}-{uuid4().hex[:12].upper()}"


#: Attribute / mapping keys that can carry a billing e-mail address across the
#: SQL model, the legacy JSON model, and ad-hoc dicts.
_RECIPIENT_KEYS = ("client_email", "email", "email_address", "contact_email")
#: Nested owner objects that may wrap the address (e.g. ``invoice.client.email``).
_RECIPIENT_NESTED_KEYS = ("client", "customer", "recipient")


def _coerce_email(source: object, _depth: int = 0) -> str | None:
    """Return the first valid e-mail address reachable from ``source``.

    The dispatch pipeline receives clients in several shapes: SQLAlchemy
    ``models.Client`` rows (``email``), legacy ``legacy_models.Invoice`` objects
    (``client_email``), plain dictionaries, and nested owner objects
    (``invoice.client.email``).  Checking one attribute only produced false
    "no valid e-mail address" failures when the address lived under another
    key, so every known shape is inspected here.
    """

    if source is None or _depth > 3:
        return None

    if isinstance(source, str):
        candidate = source.strip()
        return candidate if is_valid_email(candidate) else None

    if isinstance(source, Mapping):
        for key in _RECIPIENT_KEYS:
            value = source.get(key)
            if value is None:
                continue
            resolved = _coerce_email(value, _depth + 1)
            if resolved:
                return resolved
        for key in _RECIPIENT_NESTED_KEYS:
            if key in source:
                resolved = _coerce_email(source.get(key), _depth + 1)
                if resolved:
                    return resolved
        return None

    for attribute in _RECIPIENT_KEYS:
        try:
            value = getattr(source, attribute, None)
        except Exception:  # noqa: BLE001 - a broken property must not block dispatch
            continue
        resolved = _coerce_email(value, _depth + 1)
        if resolved:
            return resolved

    for attribute in _RECIPIENT_NESTED_KEYS:
        try:
            nested = getattr(source, attribute, None)
        except Exception:  # noqa: BLE001 - a broken relationship must not block dispatch
            continue
        resolved = _coerce_email(nested, _depth + 1)
        if resolved:
            return resolved

    return None


def _legacy_store_email(invoice_id: str) -> str | None:
    """Fall back to the JSON invoice store when the SQL ledger lacks an address.

    Mirrors are intentionally best-effort: an invoice can exist in
    ``data/invoices.json`` with ``client_email`` populated while the SQL client
    row is empty (or references a shared tax ID whose address was never filled
    in).  Imports stay local so the dispatcher does not depend on pandas unless
    this fallback is actually needed.
    """

    try:
        from config import load_settings
        from data_handler import InvoiceStore

        legacy_invoice = InvoiceStore(load_settings().invoice_store_path).get(invoice_id)
    except Exception:  # noqa: BLE001 - fallback sources must never break sending
        return None
    return _coerce_email(legacy_invoice)


def _render_mock(
    label: str,
    recipient: str,
    message: str,
    *,
    invoice_id: str,
    language: str,
    channel: str,
    subject: str | None = None,
    attachments: Optional[List[Path]] = None,
) -> None:
    """Render a mock email to the console."""
    print(_SEPARATOR)
    print(f"{label} MESSAGE SENT TO {recipient}")
    print(f"Invoice: {invoice_id} | Language: {language} | Channel: {channel}")
    if subject:
        print(f"Subject: {subject}")
    if attachments:
        print(f"Attachments: {', '.join(str(a) for a in attachments)}")
    print(_SUBSEPARATOR)
    print(message.strip())
    print(_SEPARATOR, flush=True)


def _send_via_resend(
    to_email: str,
    message: str,
    subject: str,
    *,
    invoice_id: str,
    attachments: Optional[List[Path]] = None,
) -> None:
    """Send email via Resend API."""
    if "RESEND_API_KEY" not in os.environ:
        raise ValueError("RESEND_API_KEY is missing in .env")

    resend.api_key = os.environ["RESEND_API_KEY"]

    email_params = {
        "from": os.environ.get("BUSINESS_EMAIL", "billing@yourcompany.mk"),
        "to": to_email,
        "subject": subject,
        "html": message,
    }

    if attachments:
        email_params["attachments"] = [
            {
                "content": list(path.read_bytes()),
                "filename": path.name,
                "content_type": "application/xml",
            }
            for path in attachments
        ]

    try:
        resend.Emails.send(params=email_params)
    except Exception as e:
        raise RuntimeError(f"Resend API error: {str(e)}") from e


def _send_via_smtp(
    to_email: str,
    message: str,
    subject: str,
    *,
    invoice_id: str,
    attachments: Optional[List[Path]] = None,
) -> None:
    """Send email via SMTP."""
    required_keys = (
        "EMAIL_SMTP_SERVER",
        "EMAIL_SMTP_PORT",
        "EMAIL_SMTP_USERNAME",
        "EMAIL_SMTP_PASSWORD",
    )
    missing = [key for key in required_keys if key not in os.environ]
    if missing:
        raise ValueError(f"Missing SMTP configuration in .env: {', '.join(missing)}")

    msg = MIMEMultipart()
    msg["From"] = os.environ.get("BUSINESS_EMAIL", "billing@yourcompany.mk")
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(message, "html"))

    if attachments:
        for path in attachments:
            with open(path, "rb") as f:
                part = MIMEApplication(f.read(), Name=path.name)
            part["Content-Disposition"] = f'attachment; filename="{path.name}"'
            msg.attach(part)

    try:
        with smtplib.SMTP(
            os.environ["EMAIL_SMTP_SERVER"],
            int(os.environ["EMAIL_SMTP_PORT"]),
        ) as server:
            if os.getenv("EMAIL_SMTP_USE_TLS", "false").lower() == "true":
                server.starttls()
            server.login(
                os.environ["EMAIL_SMTP_USERNAME"],
                os.environ["EMAIL_SMTP_PASSWORD"],
            )
            server.send_message(msg)
    except Exception as e:
        raise RuntimeError(f"SMTP error: {str(e)}") from e


def _find_ubl_attachment(invoice_id: str) -> Optional[Path]:
    """Find corresponding UBL 2.1 XML file in output directory."""
    output_dir = Path("output")
    xml_path = output_dir / f"{safe_filename(invoice_id)}.xml"
    return xml_path if xml_path.exists() else None


def send_email(
    to_email: str,
    message: str,
    subject: str | None = None,
    *,
    invoice_id: str = "N/A",
    language: str = "N/A",
) -> DispatchResult:
    """Send an email with fallback to mock delivery."""
    recipient = (to_email or "").strip()
    if not is_valid_email(recipient):
        raise DispatchError(f"Invalid email recipient: {recipient!r}.")
    if not message or not message.strip():
        raise DispatchError("Cannot send an empty email message.")

    final_subject = (subject or "Payment reminder").strip()
    ubl_attachment = _find_ubl_attachment(invoice_id) if invoice_id != "N/A" else None
    attachments = [ubl_attachment] if ubl_attachment else None

    # Try Resend first if configured
    if "RESEND_API_KEY" in os.environ:
        try:
            _send_via_resend(
                to_email=recipient,
                message=message,
                subject=final_subject,
                invoice_id=invoice_id,
                attachments=attachments,
            )
            return DispatchResult(
                success=True,
                channel=Channel.EMAIL,
                recipient=recipient,
                invoice_id=invoice_id,
                message_id=_message_id("RESEND"),
                subject=final_subject,
            )
        except Exception as e:
            logger.warning(f"Resend delivery failed, falling back to mock: {e}")

    # Try SMTP if configured
    if all(
        key in os.environ
        for key in (
            "EMAIL_SMTP_SERVER",
            "EMAIL_SMTP_PORT",
            "EMAIL_SMTP_USERNAME",
            "EMAIL_SMTP_PASSWORD",
        )
    ):
        try:
            _send_via_smtp(
                to_email=recipient,
                message=message,
                subject=final_subject,
                invoice_id=invoice_id,
                attachments=attachments,
            )
            return DispatchResult(
                success=True,
                channel=Channel.EMAIL,
                recipient=recipient,
                invoice_id=invoice_id,
                message_id=_message_id("SMTP"),
                subject=final_subject,
            )
        except Exception as e:
            logger.warning(f"SMTP delivery failed, falling back to mock: {e}")

    # Fallback to mock delivery
    _render_mock(
        label="EMAIL",
        recipient=recipient,
        message=message,
        invoice_id=invoice_id,
        language=language,
        channel=Channel.EMAIL.value,
        subject=final_subject,
        attachments=attachments,
    )

    return DispatchResult(
        success=True,
        channel=Channel.EMAIL,
        recipient=recipient,
        invoice_id=invoice_id,
        message_id=_message_id("EMAIL"),
        subject=final_subject,
    )


def send_viber(
    phone: str,
    message: str,
    *,
    invoice_id: str = "N/A",
    language: str = "N/A",
) -> DispatchResult:
    """Mock a Viber send with a clear console audit block."""
    recipient = normalize_phone(phone)
    if not recipient or len(recipient.lstrip("+")) < 7:
        raise DispatchError(f"Invalid Viber phone number: {phone!r}.")
    if not message or not message.strip():
        raise DispatchError("Cannot send an empty Viber message.")

    _render_mock(
        label="VIBER",
        recipient=recipient,
        message=message,
        invoice_id=invoice_id,
        language=language,
        channel=Channel.VIBER.value,
    )

    return DispatchResult(
        success=True,
        channel=Channel.VIBER,
        recipient=recipient,
        invoice_id=invoice_id,
        message_id=_message_id("VIBER"),
    )


def send_whatsapp(
    phone: str,
    message: str,
    *,
    invoice_id: str = "N/A",
    language: str = "N/A",
) -> DispatchResult:
    """Mock a WhatsApp send; included for the future omnichannel adapter."""
    recipient = normalize_phone(phone)
    if not recipient or len(recipient.lstrip("+")) < 7:
        raise DispatchError(f"Invalid WhatsApp phone number: {phone!r}.")
    if not message or not message.strip():
        raise DispatchError("Cannot send an empty WhatsApp message.")

    _render_mock(
        label="WHATSAPP",
        recipient=recipient,
        message=message,
        invoice_id=invoice_id,
        language=language,
        channel=Channel.WHATSAPP.value,
    )

    return DispatchResult(
        success=True,
        channel=Channel.WHATSAPP,
        recipient=recipient,
        invoice_id=invoice_id,
        message_id=_message_id("WHATSAPP"),
    )


def dispatch_message(
    channel: Channel | str,
    recipient: str | None,
    message: str,
    *,
    invoice_id: str,
    language: str,
    subject: str | None = None,
) -> DispatchResult:
    """Route an approved message to the correct channel adapter.

    A non-mock ``invoice_id`` is reconciled with the SQL lifecycle first: the
    invoice row must be legally allowed to reach ``SENT`` (the human-approval
    gate lives in ``state_machine.transition_invoice_state``).  An illegal
    lifecycle transition blocks the send; an unavailable SQL layer (legacy
    store not yet migrated, engine unreachable) only logs a warning so the
    legacy JSON audit trail keeps the dispatch flow working.
    """
    parsed_channel = (
        channel if isinstance(channel, Channel) else Channel.from_value(channel)
    )
    if not recipient or not recipient.strip():
        raise DispatchError(
            f"No recipient is available for {parsed_channel.value}; "
            "add an email or phone number before sending."
        )

    if invoice_id != "N/A":
        try:
            with get_db() as db:
                invoice = db.query(Invoice).filter_by(id=invoice_id).first()
                if invoice:
                    transition_invoice_state(invoice, "SENT", db_session=db)
                    logger.info(f"Invoice {invoice_id} successfully transitioned to SENT.")
                else:
                    logger.warning(f"Invoice {invoice_id} not found in DB prior to dispatch.")
        except InvalidStateTransitionError as exc:
            logger.error(f"Failed to transition state for invoice {invoice_id}: {exc}")
            raise DispatchError(f"State transition blocked: {exc}") from exc
        except SQLAlchemyError as exc:
            logger.warning(
                f"SQL state transition unavailable for invoice {invoice_id}: {exc}"
            )

    if parsed_channel == Channel.EMAIL:
        return send_email(
            recipient,
            message,
            subject=subject,
            invoice_id=invoice_id,
            language=language,
        )
    if parsed_channel == Channel.VIBER:
        return send_viber(
            recipient,
            message,
            invoice_id=invoice_id,
            language=language,
        )
    if parsed_channel == Channel.WHATSAPP:
        return send_whatsapp(
            recipient,
            message,
            invoice_id=invoice_id,
            language=language,
        )
    raise DispatchError(f"Unsupported channel: {parsed_channel!r}.")


# ---------------------------------------------------------------------------
# SQL-lifecycle transport adapters (human-approved reminders only)
# ---------------------------------------------------------------------------


class BaseDispatcher(ABC):
    """Contract for every outbound reminder transport adapter.

    Implementations receive the **caller-owned** SQLAlchemy session, so the
    lifecycle stamps (``SENT`` / ``FAILED``) commit atomically with the caller's
    transaction and sessions always close in :func:`database.get_db` blocks.
    """

    @abstractmethod
    def send_reminder(self, invoice_id: str, db_session: Session) -> bool:
        """Deliver one reminder for ``invoice_id``; True on accepted delivery."""


class SMTPDispatcher(BaseDispatcher):
    """Sends an approved invoice's reminder as HTML e-mail via SMTP.

    Contract (aligned with the invoice lifecycle):

    * Missing invoice/client rows raise :class:`ValueError`.
    * Only an ``APPROVED`` invoice may be queued for delivery; anything else
      raises :class:`state_machine.InvalidStateTransitionError` **before** any
      socket or logging activity, keeping the HITL gate airtight.
    * A successful send stamps ``SENT``; a delivery failure stamps ``FAILED``
      (allowed from every state, so the invoice always remains re-workable).
    * SMTP envelope credentials come from ``.env``:
      ``SMTP_SERVER`` / ``SMTP_PORT`` / ``SMTP_USER`` / ``SMTP_PASS``
      (``SMTP_USE_TLS`` optional, default enabled).
    * All file logging runs through the PII-safe logger, so e-mail addresses
      and phone numbers never reach collectors in clear text.
    """

    #: E-mail body notice emitted when UBL compliance XML is attached.
    _UBL_FOOTER = "The e-invoice XML attached to this message is the official document."

    @staticmethod
    def _smtp_config() -> dict[str, str]:
        """Read required SMTP credentials, legacy or new keys, before any socket.

        Resolution order per key (spec'd backward compatibility):

        * Server / User / Password: ``SMTP_*`` first, ``EMAIL_SMTP_*`` alias
          second (the keys this project shipped historically).
        * Port: ``SMTP_PORT``, then ``EMAIL_SMTP_PORT``, then default ``587``.

        Fail-fast rules (raised **before** any connection or lifecycle write):

        * Server/User missing -> the spec-hinted configuration error naming
          both key spellings for the server.
        * Password missing -> its own actionable error; failing late would
          leave the invoice stamped FAILED purely for half-configured env.
        """

        server = os.getenv("SMTP_SERVER") or os.getenv("EMAIL_SMTP_SERVER")
        user = os.getenv("SMTP_USER") or os.getenv("EMAIL_SMTP_USER")
        if not (server or "").strip() or not (user or "").strip():
            raise ValueError(
                "SMTP configuration missing: set SMTP_SERVER or "
                "EMAIL_SMTP_SERVER in .env"
            )
        password = os.getenv("SMTP_PASS") or os.getenv("EMAIL_SMTP_PASS")
        if not (password or "").strip():
            raise ValueError(
                "SMTP configuration missing: set SMTP_PASS or "
                "EMAIL_SMTP_PASS in .env"
            )

        port_raw = os.getenv("SMTP_PORT") or os.getenv("EMAIL_SMTP_PORT", "587")
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(
                f"SMTP port must be an integer, got {port_raw!r}."
            ) from exc

        return {
            "SMTP_SERVER": server.strip(),
            "SMTP_PORT": str(port),
            "SMTP_USER": user.strip(),
            "SMTP_PASS": password.strip(),
        }

    @staticmethod
    def _fetch_invoice(invoice_id: str, db_session: Session) -> tuple[models.Invoice, models.Client]:
        """Load the invoice and its owning client or raise ``ValueError``."""

        invoice = (
            db_session.query(models.Invoice).filter_by(id=invoice_id).first()
        )
        if invoice is None:
            raise ValueError("Invoice not found")
        client = (
            db_session.query(models.Client)
            .filter_by(tax_id=invoice.client_tax_id)
            .first()
        )
        if client is None:
            # The FK target vanished: treat as the same data-integrity miss.
            raise ValueError("Invoice not found")
        return invoice, client

    @staticmethod
    def _assert_sendable_state(invoice: models.Invoice) -> None:
        """HITL gate: only APPROVED invoices may enter the dispatch pipeline.

        ``SENT`` itself is only reachable from ``UBL_GENERATED``; callers pass
        ``APPROVED`` (per the CLI contract) and :meth:`send_reminder` walks the
        one legal intermediate step before delivery.
        """

        if invoice.state not in (InvoiceState.APPROVED, InvoiceState.UBL_GENERATED):
            raise InvalidStateTransitionError(
                "Invoice must be APPROVED before sending reminder"
            )

    @staticmethod
    def _validated_totals(invoice: models.Invoice) -> tuple[Decimal, Decimal]:
        """Re-validate amounts through the Pydantic fintech layer."""

        payload = schemas.InvoiceCreate(
            client_tax_id=invoice.client_tax_id,
            net_amount=Decimal(str(invoice.net_amount)),
            vat_rate=Decimal(str(invoice.vat_rate)),
        )
        return payload.vat_amount, payload.total_amount

    @staticmethod
    def _stamp(invoice: models.Invoice, db_session: Session, state: InvoiceState) -> None:
        """Persist a lifecycle stamp through the state machine."""

        transition_invoice_state(invoice, state, db_session=db_session)

    def _deliver(
        self,
        smtp_config: dict[str, str],
        recipient: str,
        subject: str,
        body_html: str,
        body_text: str,
        invoice_id: str,
        attachments: Optional[List[Path]],
    ) -> None:
        """One SMTP transaction; overridden nowhere, swapped in tests."""

        message = MIMEMultipart("alternative")
        if attachments:
            message = MIMEMultipart("mixed")
            alternative = MIMEMultipart("alternative")
            alternative.attach(MIMEText(body_text, "plain", "utf-8"))
            alternative.attach(MIMEText(body_html, "html", "utf-8"))
            message.attach(alternative)
            for path in attachments:
                with open(path, "rb") as handle:
                    part = MIMEApplication(handle.read(), Name=path.name)
                part["Content-Disposition"] = f'attachment; filename="{path.name}"'
                message.attach(part)
        else:
            message.attach(MIMEText(body_text, "plain", "utf-8"))
            message.attach(MIMEText(body_html, "html", "utf-8"))

        message["From"] = smtp_config["SMTP_USER"]
        message["To"] = recipient
        message["Subject"] = subject

        try:
            with smtplib.SMTP(
                smtp_config["SMTP_SERVER"],
                int(smtp_config["SMTP_PORT"]),
                timeout=30,
            ) as server:
                if os.getenv("SMTP_USE_TLS", "true").strip().casefold() in {
                    "1",
                    "true",
                    "yes",
                    "y",
                    "on",
                }:
                    server.starttls(context=ssl.create_default_context())
                server.login(smtp_config["SMTP_USER"], smtp_config["SMTP_PASS"])
                server.send_message(message)
        except Exception as exc:  # noqa: BLE001 - network errors must not type-leak
            raise RuntimeError(f"SMTP delivery failed: {exc}") from exc

    def send_reminder(self, invoice_id: str, db_session: Session) -> bool:
        """Deliver one APPROVED invoice reminder; stamp the lifecycle atomically."""

        smtp_config = self._smtp_config()
        invoice, client = self._fetch_invoice(invoice_id, db_session)
        self._assert_sendable_state(invoice)

        vat_amount, total_amount = self._validated_totals(invoice)

        # Resolve the recipient from every shape the data can take.  The SQL
        # client row is authoritative, but mirrors and legacy payloads also
        # carry ``client_email`` / nested ``client.email`` values; ignoring them
        # caused false "no valid e-mail address" errors for sendable invoices.
        recipient = (
            _coerce_email(client)
            or _coerce_email(invoice)
            or _legacy_store_email(invoice.id)
        )
        if not recipient:
            # No PII in the message: the raw address stays out of every log.
            raise ValueError(
                f"Client for invoice {invoice.id} has no valid e-mail address on file."
            )

        # Heal the SQL row when the fallback found a usable address, so the next
        # reminder (and other consumers of the ledger) see it directly.
        stored_email = (getattr(client, "email", "") or "").strip()
        if not is_valid_email(stored_email):
            try:
                client.email = recipient
                db_session.add(client)
                db_session.commit()
                pii_logger.info(
                    "Backfilled the client e-mail for invoice %s from a fallback source.",
                    invoice.id,
                )
            except Exception:  # noqa: BLE001 - persistence must not block delivery
                db_session.rollback()
                pii_logger.warning(
                    "Could not persist the recovered client e-mail for invoice %s.",
                    invoice.id,
                )

        subject = f"Payment reminder - invoice {invoice.id}"
        body_text = self._render_body_text(client.name, invoice, vat_amount, total_amount)
        body_html = self._render_body_html(client.name, invoice, vat_amount, total_amount)
        attachments = [path] if (path := _find_ubl_attachment(invoice.id)) else None

        if invoice.state != InvoiceState.UBL_GENERATED:
            # Walk the one legal step APPROVED -> UBL_GENERATED first: the SQL
            # ledger records that the dispatch artifact has been prepared (the
            # UBL file itself stays optional in mock/MVP mode).  ``SENT`` is
            # only ever legal from there, so skipping the step is impossible.
            self._stamp(invoice, db_session, InvoiceState.UBL_GENERATED)
            pii_logger.info(
                "Invoice %s prepared for dispatch; state=UBL_GENERATED.",
                invoice.id,
            )

        try:
            self._deliver(
                smtp_config=smtp_config,
                recipient=recipient,
                subject=subject,
                body_html=body_html,
                body_text=body_text,
                invoice_id=invoice.id,
                attachments=attachments,
            )
        except Exception:  # noqa: BLE001 - convert delivery failures into FAILED
            try:
                self._stamp(invoice, db_session, InvoiceState.FAILED)
                pii_logger.error(
                    "Reminder delivery failed for invoice %s; state=FAILED.",
                    invoice.id,
                )
            except Exception:  # noqa: BLE001 - never mask the SMTP root cause
                pii_logger.error(
                    "Reminder delivery failed for invoice %s and the FAILED "
                    "lifecycle stamp failed as well.",
                    invoice.id,
                )
            return False

        self._stamp(invoice, db_session, InvoiceState.SENT)
        pii_logger.info("Reminder %s delivered; state=SENT.", invoice.id)
        return True

    @staticmethod
    def _render_body_text(
        client_name: str,
        invoice: models.Invoice,
        vat_amount: Decimal,
        total_amount: Decimal,
    ) -> str:
        """Deterministic plain-text reminder body (sanitized template)."""

        currency = "MKD"
        lines = [
            f"Dear {client_name},",
            "",
            f"This is a friendly reminder that invoice {invoice.id} is awaiting payment.",
            f"Amount due: {total_amount} {currency} (including VAT of {vat_amount} {currency}).",
            "",
            "If payment has already been made, please ignore this message.",
            "",
            "Kind regards,",
            "Accounts receivable",
        ]
        return "\n".join(lines)

    @staticmethod
    def _render_body_html(
        client_name: str,
        invoice: models.Invoice,
        vat_amount: Decimal,
        total_amount: Decimal,
    ) -> str:
        """HTML mirror of the plain-text template with escaped interpolations."""

        currency = "MKD"
        safe_name = html_escape(client_name, quote=False)
        safe_id = html_escape(invoice.id, quote=False)
        return (
            "<html><body>"
            f"<p>Dear {safe_name},</p>"
            f"<p>This is a friendly reminder that invoice <b>{safe_id}</b> is "
            f"awaiting payment. Amount due: <b>{total_amount} {currency}</b> "
            f"(including VAT of {vat_amount} {currency}).</p>"
            "<p>If payment has already been made, please ignore this message.</p>"
            f"<p>{SMTPDispatcher._UBL_FOOTER}</p>"
            "<p>Kind regards,<br>Accounts receivable</p>"
            "</body></html>"
        )