"""SQLAlchemy 2.0 persistence models for Saldora.

Declarative 2.0 style using `DeclarativeBase` + `Mapped` / `mapped_column` annotations.
All monetary values are stored as `Numeric` (Decimal) for precision.
PII fields (name, email, phone, recipient names) use `EncryptedString` (Fernet at rest).
"""

from __future__ import annotations

import enum
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Numeric,
    String,
    Text,
    TypeDecorator,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "CancellationRequest",
    "Client",
    "DeliveryNote",
    "EmailDraft",
    "EmailDraftStatus",
    "EncryptedString",
    "Invoice",
    "InvoiceState",
    "OTPCode",
    "User",
]

_DB_KEY_ENV = "DB_ENCRYPTION_KEY"
_DB_KEY_PATH = Path("data") / "db_encryption.key"
_FERNET: Fernet | None = None


def _resolve_cipher() -> Fernet:
    """Build the process-wide Fernet singleton from env > key file > generate."""
    global _FERNET
    if _FERNET is not None:
        return _FERNET

    env_key = os.environ.get(_DB_KEY_ENV, "").strip()
    if env_key:
        key_material = env_key.encode("utf-8")
    elif _DB_KEY_PATH.exists():
        key_material = _DB_KEY_PATH.read_bytes().strip()
        if not key_material:
            raise RuntimeError(
                f"{_DB_KEY_PATH} exists but is empty; delete it or set {_DB_KEY_ENV}."
            )
    else:
        key_material = Fernet.generate_key()
        _DB_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _DB_KEY_PATH.write_bytes(key_material + b"\n")

    try:
        _FERNET = Fernet(key_material)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            f"{_DB_KEY_ENV} is not a valid Fernet key. Generate one with: "
            'python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        ) from exc
    return _FERNET


class EncryptedString(TypeDecorator):
    """A String column that stores Fernet ciphertext at rest with plaintext fallback."""

    impl = String
    cache_ok = True

    def __init__(self, length: int | None = None, **kwargs: object) -> None:
        super().__init__(length, **kwargs)

    def process_bind_param(self, value: str | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return _resolve_cipher().encrypt(str(value).encode("utf-8")).decode("ascii")

    def process_result_value(self, value: str | None, dialect: Dialect) -> str | None:
        if value is None:
            return None
        try:
            return _resolve_cipher().decrypt(str(value).encode("utf-8")).decode("utf-8")
        except InvalidToken:
            return value


class Base(DeclarativeBase):
    """Declarative base for every Saldora ORM model."""


class InvoiceState(str, enum.Enum):
    """Lifecycle of an invoice from draft to outbound delivery."""

    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    UBL_GENERATED = "UBL_GENERATED"
    SENT = "SENT"
    FAILED = "FAILED"
    UNPAID = "UNPAID"
    PAID = "PAID"
    STORNO = "STORNO"


_UTC_DEFAULT = datetime.utcnow


class User(Base):
    """User account for Saldora with credentials and profile information."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    company_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # ЕДБ / tax number of the account's company. Nullable so existing rows and
    # older API clients that never sent it keep working.
    tax_id: Mapped[str | None] = mapped_column(String(13), nullable=True)
    # Email OTP verification state. Nullable-free for new rows; the startup
    # migration backfills pre-OTP accounts (and the demo profile) to True.
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # Mandatory Terms of Service / Privacy Policy acceptance captured by the
    # registration wizard. The timestamp and version make the consent auditable.
    terms_accepted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    terms_accepted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    terms_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Per-account Documenso credentials (Settings → e-Signature). The API key is
    # encrypted at rest; when unset the server falls back to DOCUMENSO_API_KEY.
    documenso_api_key: Mapped[str | None] = mapped_column(
        EncryptedString(500), nullable=True
    )
    documenso_base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_UTC_DEFAULT)

    clients: Mapped[list[Client]] = relationship("Client", back_populates="user")
    invoices: Mapped[list[Invoice]] = relationship("Invoice", back_populates="user")
    delivery_notes: Mapped[list[DeliveryNote]] = relationship("DeliveryNote", back_populates="user")
    email_drafts: Mapped[list[EmailDraft]] = relationship("EmailDraft", back_populates="user")

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, email={self.email!r}, company_name={self.company_name!r})"


class OTPCode(Base):
    """A short-lived 6-digit verification code dispatched over email."""

    __tablename__ = "otp_codes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Nullable for codes issued to email addresses that have no row in users
    # yet (pending signups).
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    code: Mapped[str] = mapped_column(String(6), nullable=False)
    purpose: Mapped[str] = mapped_column(String(50), default="signup_verification", nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_used: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_UTC_DEFAULT)

    def __repr__(self) -> str:
        return f"OTPCode(id={self.id!r}, email={self.email!r}, purpose={self.purpose!r}, is_used={self.is_used!r})"


class Client(Base):
    """A registered fiscal client (customers table)."""

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    tax_id: Mapped[str] = mapped_column(String(13), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(EncryptedString(255), nullable=False)
    email: Mapped[str] = mapped_column(EncryptedString(255), nullable=False)
    phone: Mapped[str] = mapped_column(EncryptedString(50), nullable=False)

    user: Mapped[User | None] = relationship("User", back_populates="clients")
    invoices: Mapped[list[Invoice]] = relationship("Invoice", back_populates="client")

    def __repr__(self) -> str:
        masked = f"****{self.tax_id[-4:]}" if len(self.tax_id) >= 4 else "****"
        return f"Client(id={self.id!r}, name={self.name!r}, tax_id={masked!r})"


class Invoice(Base):
    """An issued invoice ('NPL-2026-0001' style human-readable primary key)."""

    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    code: Mapped[str | None] = mapped_column(String(50), unique=True, index=True, nullable=True)
    client_tax_id: Mapped[str | None] = mapped_column(
        String(13), ForeignKey("clients.tax_id"), nullable=True
    )
    client_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    net_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), default=Decimal("0.00"), nullable=False
    )
    vat_rate: Mapped[Decimal] = mapped_column(
        Numeric(5, 2), default=Decimal("18.00"), nullable=False
    )
    vat_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), default=Decimal("0.00"), nullable=False
    )
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), default=Decimal("0.00"), nullable=False
    )
    amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)

    status: Mapped[str] = mapped_column(String(50), default="UNPAID", nullable=False)
    state: Mapped[InvoiceState] = mapped_column(
        Enum(InvoiceState, native_enum=False), default=InvoiceState.DRAFT, nullable=False
    )
    is_approved: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    due_days: Mapped[int] = mapped_column(default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_UTC_DEFAULT)

    issue_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    client_address: Mapped[str | None] = mapped_column(String(500), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    xml_file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    client: Mapped[Client | None] = relationship("Client", back_populates="invoices")
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    user: Mapped[User | None] = relationship("User", back_populates="invoices")
    delivery_notes: Mapped[list[DeliveryNote]] = relationship(
        "DeliveryNote", back_populates="invoice", cascade="all, delete-orphan"
    )

    def __init__(self, **kwargs: object) -> None:
        if "code" in kwargs and "id" not in kwargs:
            kwargs["id"] = kwargs["code"]
        elif "id" in kwargs and "code" not in kwargs:
            kwargs["code"] = str(kwargs["id"])

        if "amount" in kwargs and kwargs["amount"] is not None:
            amt_val = kwargs["amount"]
            dec_amt = Decimal(str(amt_val)) if not isinstance(amt_val, Decimal) else amt_val
            kwargs["amount"] = dec_amt
            if "total_amount" not in kwargs:
                kwargs["total_amount"] = dec_amt
            if "net_amount" not in kwargs:
                vat_r = Decimal(str(kwargs.get("vat_rate", "18.00")))
                net = round(dec_amt / (Decimal("1.00") + (vat_r / Decimal("100"))), 2)
                kwargs["net_amount"] = net
                kwargs["vat_amount"] = dec_amt - net

        super().__init__(**kwargs)

    def __repr__(self) -> str:
        return (
            f"Invoice(id={self.id!r}, client_name={self.client_name!r}, "
            f"amount={self.total_amount!r}, status={self.status!r})"
        )


class DeliveryNote(Base):
    """An e-Ispratnica delivery note attached to an invoice."""

    __tablename__ = "delivery_notes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    invoice_id: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("invoices.id"), nullable=True
    )
    recipient_name: Mapped[str] = mapped_column(EncryptedString(255), nullable=False)
    details: Mapped[str] = mapped_column(String(500), nullable=False)
    is_signed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    signed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    client_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    delivery_address: Mapped[str | None] = mapped_column(String(500), nullable=True)
    items_description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    issue_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    signature_data: Mapped[str | None] = mapped_column(Text, nullable=True)
    ujp_status: Mapped[str] = mapped_column(String(100), default="Sent to UJP", nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    # Documenso e-signature lifecycle. ``NOT_SENT`` is the default for notes
    # signed on the field device; the API sets ``SENT`` while an embedded
    # signing session is pending and the webhook flips it to ``SIGNED``.
    documenso_document_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    documenso_status: Mapped[str] = mapped_column(
        String(30), default="NOT_SENT", nullable=False
    )
    documenso_signing_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    documenso_recipient_email: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    documenso_signed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # SHA-256 sealed JSON audit trail assembled from the Documenso webhook.
    documenso_signature_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    documenso_verification_hash: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )

    invoice: Mapped[Invoice | None] = relationship("Invoice", back_populates="delivery_notes")
    user: Mapped[User | None] = relationship("User", back_populates="delivery_notes")

    def __repr__(self) -> str:
        return f"DeliveryNote(id={self.id!r}, code={self.code!r}, is_signed={self.is_signed!r})"


class CancellationRequest(Base):
    """Human-review record for invoice storniranje before UJP submission."""

    __tablename__ = "cancellation_requests"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(String(50), ForeignKey("invoices.id"), nullable=False)
    requested_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_reviewed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_UTC_DEFAULT)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    invoice: Mapped[Invoice] = relationship("Invoice")

    def __repr__(self) -> str:
        return (
            f"CancellationRequest(id={self.id!r}, invoice_id={self.invoice_id!r}, "
            f"is_reviewed={self.is_reviewed!r})"
        )


class EmailDraftStatus(str, enum.Enum):
    """Approval lifecycle for AI-generated negotiation replies."""

    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SENT = "SENT"


class EmailDraft(Base):
    """Human-in-the-loop draft for client email replies generated by AI."""

    __tablename__ = "email_drafts"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    invoice_id: Mapped[str | None] = mapped_column(
        String(50), ForeignKey("invoices.id"), nullable=True
    )
    client_email: Mapped[str] = mapped_column(String(255), nullable=False)
    incoming_subject: Mapped[str] = mapped_column(String(500), nullable=False)
    incoming_body: Mapped[str] = mapped_column(Text, nullable=False)
    ai_draft_response: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), default="PENDING_APPROVAL", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_UTC_DEFAULT)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    invoice: Mapped[Invoice | None] = relationship("Invoice")
    user: Mapped[User | None] = relationship("User", back_populates="email_drafts")

    def __repr__(self) -> str:
        return (
            f"EmailDraft(id={self.id!r}, invoice_id={self.invoice_id!r}, "
            f"client_email={self.client_email!r}, status={self.status!r})"
        )

