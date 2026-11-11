"""FastAPI backend for Saldora.

Run with::

    python api.py

The module reuses the project's existing SQLAlchemy layer (``database.py``)
and ORM models (``models.py``); it does not define a second database engine.

Defensive notes
---------------
* Every monetary value crossing the HTTP boundary is converted from
  :class:`decimal.Decimal` to ``float`` explicitly.  A custom JSON response
  class is also installed as a final safety net, so no response can ever fail
  with ``TypeError: Object of type Decimal is not JSON serializable``.
* Sessions come from the project's context-managed ``get_db()`` and are
  always closed in a ``finally`` block.
* Blocking SMTP work runs in a worker thread, not on the event loop.
"""

from __future__ import annotations

import hashlib
import html as html_module
import json
import os
import re
import secrets
import string
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import bcrypt
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from database import get_db as _get_db_context
from database import init_db
from documenso_client import (
    DEFAULT_BASE_URL,
    DocumensoAPIError,
    DocumensoClient,
    DocumensoConfigError,
    is_configured_api_key,
    normalize_base_url,
)
from document_pdf import build_delivery_note_pdf
from logger import logger
from models import (
    CancellationRequest,
    Client,
    DeliveryNote,
    EmailDraft,
    Invoice,
    InvoiceState,
    OTPCode,
    User,
)

# Load environment variables before any module-level os.getenv() constant below.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

XML_DIR = BASE_DIR / "data" / "xmls"

INVOICE_PREFIX = "INV-"
DEFAULT_VAT_RATE = Decimal("18.00")
CENT = Decimal("0.01")
ALLOWED_STATUSES = {"UNPAID", "PAID", "STORNO"}
DEFAULT_STORNO_REASON = "Сторнирано по барање на корисникот"
MAX_CODE_ALLOCATION_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# JSON safety net
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Fallback encoder for types FastAPI might not have converted yet."""

    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class DecimalSafeJSONResponse(JSONResponse):
    """JSON response that can never choke on Decimal/datetime/enum values."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")


# ---------------------------------------------------------------------------
# Session dependency
# ---------------------------------------------------------------------------


def get_db() -> Iterator[Session]:
    """Yield a session from ``database.get_db()``; its ``finally`` closes it."""

    with _get_db_context() as session:
        yield session


DbSession = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# JWT Authentication & Passwords
# ---------------------------------------------------------------------------

JWT_SECRET = os.getenv("JWT_SECRET", "saldora_fallback_secret_key_2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

DEMO_EMAIL = "admin@saldora.ai"
DEMO_PASSWORD = "demo123"
DEMO_COMPANY_NAME = "Saldora Enterprise ДООЕЛ"
DEMO_TAX_ID = "4030991123456"

# Shared password policy (kept in one place so the wizard, the OTP endpoints,
# and the direct signup endpoint can never drift apart).
PASSWORD_MIN_LENGTH = 7
PASSWORD_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
_PASSWORD_SPECIAL_RE = re.compile(rf"[{re.escape(PASSWORD_SPECIAL_CHARS)}]")
TERMS_VERSION = "2026-11-06"


def password_policy_error(password: str) -> str | None:
    """Return the first Macedonian password-policy violation, if any."""

    if len(password) < PASSWORD_MIN_LENGTH:
        return f"Лозинката мора да содржи најмалку {PASSWORD_MIN_LENGTH} карактери."
    if not re.search(r"\d", password):
        return "Лозинката мора да содржи најмалку една цифра (0-9)."
    if not _PASSWORD_SPECIAL_RE.search(password):
        return "Лозинката мора да содржи најмалку еден специјален знак (!@#$%^&*...)."
    return None


def _validate_password_or_400(password: str, password_confirm: str | None = None) -> None:
    """Raise a Macedonian 400 response when the password policy is violated."""

    error = password_policy_error(password)
    if error:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=error,
        )
    if password_confirm is not None and password_confirm != password:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Лозинките не се совпаѓаат.",
        )


# passlib is the single password-hashing entry point. PBKDF2-SHA256 is used for
# new hashes because passlib's bcrypt backend is incompatible with bcrypt>=5.
# Legacy ``$2b$`` bcrypt hashes already in the database are still verified.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password with passlib (PBKDF2-SHA256)."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a passlib hash, including legacy bcrypt hashes."""
    if not hashed_password:
        return False
    try:
        if hashed_password.startswith(("$2a$", "$2b$", "$2y$")):
            return bcrypt.checkpw(
                plain_password.encode("utf-8"), hashed_password.encode("utf-8")
            )
        return pwd_context.verify(plain_password, hashed_password)
    except (TypeError, ValueError):
        return False


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login", auto_error=False)


def get_current_user(
    db: DbSession,
    token: Annotated[str | None, Depends(oauth2_scheme)] = None,
) -> User:
    """Extract and validate the JWT Bearer token and return the User."""
    if not token:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Невалидна автентикација или недостасува токен.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        sub: str | None = payload.get("sub")
        if sub is None:
            raise HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Невалиден токен.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user_id = int(sub)
    except (JWTError, ValueError):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Невалиден или истечен токен.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Корисникот не е пронајден.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def _ensure_default_user(db: Session) -> User:
    """Seed the demo account and assign existing unassigned records.

    The demo profile button must work even on a database that was created by an
    older build, so an existing demo account is repaired (password + tax id)
    instead of being left with a stale credential.
    """
    demo_user = db.query(User).filter(User.email == DEMO_EMAIL).first()
    if not demo_user:
        demo_user = User(
            email=DEMO_EMAIL,
            hashed_password=hash_password(DEMO_PASSWORD),
            company_name=DEMO_COMPANY_NAME,
            tax_id=DEMO_TAX_ID,
        )
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
    else:
        changed = False
        if not verify_password(DEMO_PASSWORD, demo_user.hashed_password):
            demo_user.hashed_password = hash_password(DEMO_PASSWORD)
            changed = True
        if not demo_user.tax_id:
            demo_user.tax_id = DEMO_TAX_ID
            changed = True
        if changed:
            db.commit()
            db.refresh(demo_user)
    # Associate legacy unassigned records with demo user
    db.query(Invoice).filter(Invoice.user_id.is_(None)).update(
        {Invoice.user_id: demo_user.id}, synchronize_session=False
    )
    db.query(DeliveryNote).filter(DeliveryNote.user_id.is_(None)).update(
        {DeliveryNote.user_id: demo_user.id}, synchronize_session=False
    )
    db.query(Client).filter(Client.user_id.is_(None)).update(
        {Client.user_id: demo_user.id}, synchronize_session=False
    )
    db.query(EmailDraft).filter(EmailDraft.user_id.is_(None)).update(
        {EmailDraft.user_id: demo_user.id}, synchronize_session=False
    )
    db.commit()
    return demo_user


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


def _normalize_email(value: str) -> str:
    """Shared email normalization/validation for every auth schema."""

    cleaned = value.strip().lower()
    if "@" not in cleaned or "." not in cleaned:
        raise ValueError("Невалиден формат на е-пошта.")
    return cleaned


def _normalize_tax_id(value: str | None) -> str | None:
    """Strip whitespace and validate a Macedonian ЕДБ (9–13 digits)."""

    if value is None:
        return None
    cleaned = re.sub(r"\s+", "", str(value))
    if not cleaned:
        return None
    if not cleaned.isdigit() or not 9 <= len(cleaned) <= 13:
        raise ValueError("ЕДБ мора да содржи 9 до 13 цифри.")
    return cleaned


# Single-source field types: every auth schema shares these so the email/EDB
# rules can never drift between endpoints.
NormalizedEmail = Annotated[str, AfterValidator(_normalize_email)]
NormalizedTaxId = Annotated[str | None, BeforeValidator(_normalize_tax_id)]


class UserCreate(BaseModel):
    """Registration payload for the progressive sign-up wizard."""

    email: NormalizedEmail = Field(min_length=3, max_length=255)
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=255)
    company_name: str = Field(min_length=1, max_length=255)
    tax_id: NormalizedTaxId = Field(default=None)
    terms_accepted: bool = Field(default=False)

    @field_validator("password")
    @classmethod
    def _validate_password_complexity(cls, v: str) -> str:
        """Enforce: ≥7 chars, at least one digit, at least one special character."""
        error = password_policy_error(v)
        if error:
            raise ValueError(error)
        return v


# Backwards-compatible alias for code that imported the previous name.
UserSignupRequest = UserCreate


class SignupOTPRequest(BaseModel):
    """Step 1 of the OTP wizard: request a 6-digit code for this address.

    The wizard sends the chosen password as well so the server can reject a
    policy-violating password *before* an OTP is emailed. ``password`` remains
    optional for API clients that only need an email challenge; ``complete``
    always re-validates the password before an account is created.
    """

    email: NormalizedEmail = Field(min_length=3, max_length=255)
    password: str | None = Field(default=None, max_length=255)
    password_confirm: str | None = Field(default=None, max_length=255)


class SignupVerifyRequest(BaseModel):
    """Step 2 of the OTP wizard: non-consuming code check."""

    email: NormalizedEmail = Field(min_length=3, max_length=255)
    otp_code: str = Field(min_length=6, max_length=6)


class SignupCompleteRequest(BaseModel):
    """Final OTP wizard payload: profile fields, consent, and the verified code.

    Password/consent checks run inside the endpoint so violations return a
    Macedonian ``400 Bad Request`` instead of Pydantic's generic 422 body.
    """

    email: NormalizedEmail = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=255)
    password_confirm: str | None = Field(default=None, max_length=255)
    company_name: str = Field(min_length=1, max_length=255)
    tax_id: NormalizedTaxId = Field(default=None)
    otp_code: str = Field(min_length=6, max_length=6)
    terms_accepted: bool = False


class UserProfileResponse(BaseModel):
    id: int
    email: str
    company_name: str
    tax_id: str | None = None
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any] | None = None


class InvoiceCreateRequest(BaseModel):
    """Payload for creating a VAT-inclusive invoice (UBL-ready)."""

    client_name: str = Field(min_length=1, max_length=255)
    amount: float = Field(gt=0, allow_inf_nan=False, description="Gross total, VAT included.")
    description: str = Field(max_length=500)
    client_tax_id: str | None = Field(default=None, max_length=50)
    client_address: str | None = Field(default=None, max_length=500)
    issue_date: date | None = Field(default=None)
    due_date: date | None = Field(default=None)
    vat_rate: float = Field(default=18.0, ge=0, le=100)


class ApproveRequest(BaseModel):
    approved_by: str = Field(min_length=1, max_length=255)


class StornoRequest(BaseModel):
    """Optional reason attached to a storno correction receipt."""

    reason: str | None = Field(default=None, max_length=500)


class RemindRequest(BaseModel):
    """Destination address for an SMTP payment reminder."""

    recipient_email: str = Field(min_length=3, max_length=255)

    @field_validator("recipient_email")
    @classmethod
    def _reject_header_injection(cls, value: str) -> str:
        """Strip the address and refuse CR/LF header injection attempts."""

        cleaned = value.strip()
        if any(character in cleaned for character in "\r\n"):
            raise ValueError("recipient_email must not contain line breaks")
        return cleaned


class SignRequest(BaseModel):
    """Base64-encoded signature captured for a delivery note."""

    signature_data: str = Field(min_length=1)


class DeliveryNoteCreateRequest(BaseModel):
    client_name: str = Field(min_length=1, max_length=255)
    delivery_address: str = Field(min_length=1, max_length=500)
    items_description: str = Field(min_length=1, max_length=1000)
    issue_date: date | None = Field(default=None)
    signature_data: str = Field(min_length=1, description="Base64 PNG from canvas")
    invoice_id: str | None = Field(default=None, max_length=50)


class StatsResponse(BaseModel):
    total_revenue: float
    unpaid_amount: float
    unpaid_count: int
    pending_ispratnici: int
    pending_email_drafts: int = 0


class EmailDraftResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    invoice_id: str | None
    client_email: str
    incoming_subject: str
    incoming_body: str
    ai_draft_response: str
    status: str
    created_at: datetime | None


class EmailDraftApproveRequest(BaseModel):
    edited_body: str | None = Field(default=None, description="Optional edited body overriding AI draft")


class InvoiceResponse(BaseModel):
    """Full invoice representation with JSON-safe monetary fields."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    code: str | None
    client_tax_id: str | None
    client_name: str | None
    description: str | None
    net_amount: float
    vat_rate: float
    vat_amount: float
    total_amount: float
    amount: float | None
    status: str
    state: str
    is_approved: bool
    due_days: int
    created_at: datetime | None
    issue_date: date | None
    due_date: date | None
    client_address: str | None
    approved_by: str | None
    xml_file_path: str | None


class DeliveryNoteResponse(BaseModel):
    id: int
    code: str
    invoice_id: str | None
    recipient_name: str
    details: str
    is_signed: bool
    signed_at: datetime | None
    client_name: str | None
    delivery_address: str | None
    items_description: str | None
    issue_date: date | None
    signature_data: str | None
    ujp_status: str
    documenso_document_id: str | None = None
    documenso_status: str = "NOT_SENT"
    documenso_signing_url: str | None = None
    documenso_recipient_email: str | None = None
    documenso_signed_at: datetime | None = None
    documenso_verification_hash: str | None = None


class CorrectionReceipt(BaseModel):
    ubl_version: str
    document_type: str
    type_code: str
    id: str
    issue_date: date
    currency: str
    original_invoice_id: str
    corrected_amount: float
    reason: str


class StornoResponse(BaseModel):
    success: bool
    id: str
    code: str
    status: str
    reason: str
    ubl: dict[str, str]
    correction_receipt: CorrectionReceipt


class RemindResponse(BaseModel):
    success: bool
    message: str


class SignResponse(BaseModel):
    success: bool
    id: int
    code: str
    is_signed: bool
    signed_at: datetime


class DocumensoSignRequest(BaseModel):
    """Optional signer address; defaults to the authenticated account email."""

    recipient_email: NormalizedEmail | None = Field(default=None, max_length=255)

    @field_validator("recipient_email", mode="before")
    @classmethod
    def _empty_recipient_is_none(cls, value: Any) -> Any:
        # An empty or blank address means "sign with my own email", not a 422.
        return value.strip() or None if isinstance(value, str) else value


class DocumensoSignResponse(BaseModel):
    signing_url: str
    doc_id: str
    status: str = "SENT"


class DocumensoConfigResponse(BaseModel):
    api_key_configured: bool
    api_key_masked: str
    base_url: str
    webhook_url: str


class DocumensoConfigUpdateRequest(BaseModel):
    api_key: str | None = Field(default=None, max_length=500)
    base_url: str | None = Field(default=None, max_length=500)
    clear_api_key: bool = False


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_float(value: Decimal | float | int | None) -> float | None:
    """Convert a Decimal/None money value to a JSON-safe float."""

    if value is None:
        return None
    return round(float(value), 2)


def _serialize_invoice(invoice: Invoice) -> dict[str, Any]:
    """Build a full invoice dictionary without leaking Decimal objects."""

    state = invoice.state.value if isinstance(invoice.state, InvoiceState) else invoice.state
    return {
        "id": invoice.id,
        "code": invoice.code,
        "client_tax_id": invoice.client_tax_id,
        "client_name": invoice.client_name,
        "description": invoice.description,
        "net_amount": _to_float(invoice.net_amount),
        "vat_rate": _to_float(invoice.vat_rate),
        "vat_amount": _to_float(invoice.vat_amount),
        "total_amount": _to_float(invoice.total_amount),
        "amount": _to_float(invoice.amount),
        "status": invoice.status,
        "state": state,
        "is_approved": bool(invoice.is_approved),
        "due_days": int(invoice.due_days or 0),
        "created_at": invoice.created_at,
        "issue_date": invoice.issue_date,
        "due_date": invoice.due_date,
        "client_address": invoice.client_address,
        "approved_by": invoice.approved_by,
        "xml_file_path": invoice.xml_file_path,
    }


def _serialize_delivery_note(note: DeliveryNote) -> dict[str, Any]:
    return {
        "id": note.id,
        "code": note.code,
        "invoice_id": note.invoice_id,
        "recipient_name": note.recipient_name,
        "details": note.details,
        "is_signed": bool(note.is_signed),
        "signed_at": note.signed_at,
        "client_name": note.client_name,
        "delivery_address": note.delivery_address,
        "items_description": note.items_description,
        "issue_date": note.issue_date,
        "signature_data": note.signature_data,
        "ujp_status": note.ujp_status,
        "documenso_document_id": note.documenso_document_id,
        "documenso_status": note.documenso_status or "NOT_SENT",
        "documenso_signing_url": note.documenso_signing_url,
        "documenso_recipient_email": note.documenso_recipient_email,
        "documenso_signed_at": note.documenso_signed_at,
        "documenso_verification_hash": note.documenso_verification_hash,
    }


def _serialize_email_draft(draft: EmailDraft) -> dict[str, Any]:
    return {
        "id": draft.id,
        "invoice_id": draft.invoice_id,
        "client_email": draft.client_email,
        "incoming_subject": draft.incoming_subject,
        "incoming_body": draft.incoming_body,
        "ai_draft_response": draft.ai_draft_response,
        "status": draft.status,
        "created_at": draft.created_at,
    }


def _find_invoice(db: Session, code: str, user_id: int | None = None) -> Invoice | None:
    """Locate an invoice by its human-readable code or primary key, optionally scoped by user."""

    query = db.query(Invoice).filter(or_(Invoice.code == code, Invoice.id == code))
    if user_id is not None:
        query = query.filter(Invoice.user_id == user_id)
    return query.first()


def _require_owned_invoice(db: Session, code: str, user_id: int) -> Invoice:
    """Fetch an invoice for a write operation, distinguishing 403 from 404.

    404 hides non-existent codes; a foreign-owned entity yields 403 Forbidden
    so cross-tenant write attempts are explicitly denied.
    """

    invoice = _find_invoice(db, code)
    if invoice is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Invoice '{code}' was not found.",
        )
    if invoice.user_id != user_id:
        logger.warning("Cross-tenant write on invoice %s by user %s denied.", code, user_id)
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Немате дозвола да го менувате овој документ.",
        )
    return invoice


# ---------------------------------------------------------------------------
# Business helpers
# ---------------------------------------------------------------------------


def _code_exists(db: Session, model: type, code: str) -> bool:
    """True when ``code`` is already used by ``model``.

    ``Invoice`` mirrors its code into the string primary key ``id``, so both
    columns are checked; other models (e.g. ``DeliveryNote``) use an integer
    PK and only carry the unique ``code`` column.
    """

    if model is Invoice:
        return (
            db.query(Invoice.id)
            .filter(or_(Invoice.code == code, Invoice.id == code))
            .first()
            is not None
        )
    return db.query(model.id).filter(model.code == code).first() is not None


def _code_is_taken(db: Session, code: str) -> bool:
    """Backwards-compatible alias: invoices use ``_code_exists`` semantics."""

    return _code_exists(db, Invoice, code)


def generate_guaranteed_unique_code(
    db: Session,
    model: type,
    prefix: str,
    *,
    now: datetime | None = None,
) -> str:
    """Allocate a fail-safe, timestamped code: ``{prefix}-{YYYYMMDD}-{HEX6}``.

    Design goals on Supabase/Vercel (stateless, concurrent, no shared counters):

    1. **Timestamped** — the UTC date bucket (``YYYYMMDD``) keeps codes
       chronologically sortable and distributes the random space per day.
    2. **Random** — a 6-character uppercase hex body (``secrets.token_hex(3)``,
       16.7M combinations per day) makes coincidental collisions practically
       impossible without any cross-instance sequence state.
    3. **Verified** — every candidate is checked against the database
       (``model.code == candidate`` and, for invoices, the mirrored ``id``
       primary key) before it is handed out; on collision a fresh hex is
       generated immediately.
    4. **Bounded** — 50 regeneration attempts; a same-day collision odds-wise
       cannot exhaust that, so exceeding it means the uniqueness oracle itself
       is broken and must surface as a hard error, not a hang.

    The endpoint-level ``IntegrityError -> rollback -> regenerate -> recommit``
    retry loops remain as the final backstop against true race conditions.
    """

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d")
    for _ in range(50):
        hex6 = secrets.token_hex(3).upper()
        candidate = f"{prefix}-{stamp}-{hex6}"
        if not _code_exists(db, model, candidate):
            return candidate

    raise RuntimeError(
        f"Could not allocate a unique {prefix} code after "
        "50 timestamped-hex regeneration attempts."
    )


def _next_invoice_code(db: Session) -> str:
    """Allocate the next fail-safe e-Faktura (invoice) code, e.g. ``INV-20261111-8A3F9E``."""

    return generate_guaranteed_unique_code(db, Invoice, "INV")


DELIVERY_NOTE_PREFIX = "ISP-"


def _next_delivery_note_code(db: Session) -> str:
    """Allocate the next fail-safe e-Ispratnica code, e.g. ``ISP-20261111-9B2C4D``."""

    return generate_guaranteed_unique_code(db, DeliveryNote, "ISP")


def _find_delivery_note(
    db: Session, identifier: str, user_id: int
) -> DeliveryNote | None:
    """Resolve an e-Ispratnica by numeric id or ``ISP-...`` code."""

    query = db.query(DeliveryNote).filter(DeliveryNote.user_id == user_id)
    note: DeliveryNote | None = None
    if identifier.isdigit():
        note = query.filter(DeliveryNote.id == int(identifier)).first()
    if note is None:
        note = query.filter(DeliveryNote.code == identifier).first()
    return note


def _require_owned_delivery_note(
    db: Session, identifier: str, user_id: int
) -> DeliveryNote:
    """Fetch an e-Ispratnica for a write, distinguishing 403 from 404.

    404 hides non-existent identifiers; a foreign-owned note yields 403
    Forbidden so cross-tenant write attempts are explicitly denied.
    """

    note = _find_delivery_note(db, identifier, user_id)
    if note is None:
        # One unscoped re-check separates "does not exist" from "not yours".
        unscoped = _find_delivery_note_unscoped(db, identifier)
        if unscoped is not None:
            logger.warning(
                "Cross-tenant write on delivery note %s by user %s denied.",
                identifier,
                user_id,
            )
            raise HTTPException(
                status_code=http_status.HTTP_403_FORBIDDEN,
                detail="Немате дозвола да го менувате овој документ.",
            )
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Delivery note '{identifier}' was not found.",
        )
    return note


def _find_delivery_note_unscoped(db: Session, identifier: str) -> DeliveryNote | None:
    """Existence probe by numeric id or code, ignoring ownership."""

    query = db.query(DeliveryNote)
    if identifier.isdigit():
        note = query.filter(DeliveryNote.id == int(identifier)).first()
        if note is not None:
            return note
    return query.filter(DeliveryNote.code == identifier).first()


def _mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "••••••••"
    return f"{value[:4]}••••••••{value[-4:]}"


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _build_documenso_audit(
    event: dict[str, Any], payload: dict[str, Any]
) -> tuple[str, str]:
    """Build the sealed JSON audit trail and its SHA-256 verification hash."""

    recipients = payload.get("recipients") or payload.get("Recipient") or []
    if not isinstance(recipients, list):
        recipients = []

    entries: list[dict[str, Any]] = []
    for recipient in recipients:
        if not isinstance(recipient, dict):
            continue
        token = str(recipient.get("token") or "")
        entries.append(
            {
                "recipient_id": recipient.get("id"),
                "email": recipient.get("email"),
                "name": recipient.get("name"),
                "role": recipient.get("role"),
                "signed_at": recipient.get("signedAt"),
                "signing_status": recipient.get("signingStatus"),
                "token_sha256": (
                    hashlib.sha256(token.encode("utf-8")).hexdigest()
                    if token
                    else None
                ),
            }
        )

    record = {
        "event": event.get("event"),
        "document_id": payload.get("id") or payload.get("documentId"),
        "envelope_id": payload.get("envelopeId"),
        "title": payload.get("title"),
        "status": payload.get("status"),
        "completed_at": payload.get("completedAt"),
        "created_at": event.get("createdAt"),
        "webhook_endpoint": event.get("webhookEndpoint"),
        "recipients": entries,
    }
    canonical = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _effective_documenso_base_url(user: User) -> str:
    try:
        return normalize_base_url(user.documenso_base_url)
    except DocumensoConfigError:
        return DEFAULT_BASE_URL


def _vat_breakdown(total: Decimal, vat_rate: Decimal = DEFAULT_VAT_RATE) -> tuple[Decimal, Decimal]:
    """Split a VAT-inclusive gross amount into net and VAT, cent-exact."""

    divisor = Decimal("1") + (vat_rate / Decimal("100"))
    net = (total / divisor).quantize(CENT, rounding=ROUND_HALF_UP)
    vat = (total - net).quantize(CENT, rounding=ROUND_HALF_UP)
    return net, vat


def _generate_ubl_xml(invoice: Invoice) -> str:
    """Generate a minimal UBL 2.1 Invoice XML string."""
    code = invoice.code or invoice.id
    issue = (invoice.issue_date or datetime.utcnow().date()).isoformat()
    due = (invoice.due_date or datetime.utcnow().date()).isoformat()
    net = float(_to_float(invoice.net_amount) or 0.0)
    vat = float(_to_float(invoice.vat_amount) or 0.0)
    total = float(_to_float(invoice.total_amount) or 0.0)
    supplier = "Saldora B2B"
    customer = invoice.client_name or "Unknown"
    tax_id = invoice.client_tax_id or ""
    address = invoice.client_address or ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Invoice xmlns="urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"
         xmlns:cbc="urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"
         xmlns:cac="urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2">
  <cbc:UBLVersionID>2.1</cbc:UBLVersionID>
  <cbc:ID>{code}</cbc:ID>
  <cbc:IssueDate>{issue}</cbc:IssueDate>
  <cbc:DueDate>{due}</cbc:DueDate>
  <cbc:DocumentCurrencyCode>MKD</cbc:DocumentCurrencyCode>
  <cac:AccountingSupplierParty>
    <cac:Party>
      <cac:PartyName><cbc:Name>{supplier}</cbc:Name></cac:PartyName>
    </cac:Party>
  </cac:AccountingSupplierParty>
  <cac:AccountingCustomerParty>
    <cac:Party>
      <cac:PartyName><cbc:Name>{customer}</cbc:Name></cac:PartyName>
      <cac:PartyTaxScheme>
        <cbc:CompanyID>{tax_id}</cbc:CompanyID>
        <cac:TaxScheme><cbc:ID>VAT</cbc:ID></cac:TaxScheme>
      </cac:PartyTaxScheme>
      <cac:PostalAddress><cbc:StreetName>{address}</cbc:StreetName></cac:PostalAddress>
    </cac:Party>
  </cac:AccountingCustomerParty>
  <cac:TaxTotal>
    <cbc:TaxAmount currencyID="MKD">{vat:.2f}</cbc:TaxAmount>
  </cac:TaxTotal>
  <cac:LegalMonetaryTotal>
    <cbc:LineExtensionAmount currencyID="MKD">{net:.2f}</cbc:LineExtensionAmount>
    <cbc:TaxExclusiveAmount currencyID="MKD">{net:.2f}</cbc:TaxExclusiveAmount>
    <cbc:TaxInclusiveAmount currencyID="MKD">{total:.2f}</cbc:TaxInclusiveAmount>
    <cbc:PayableAmount currencyID="MKD">{total:.2f}</cbc:PayableAmount>
  </cac:LegalMonetaryTotal>
</Invoice>"""


def send_email_native(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Mapping[str, str] | None = None,
    attachment_path: str | Path | None = None,
    html_body: str | None = None,
) -> tuple[bool, str]:
    """Lazy proxy to ``bot_service.send_email_native``.

    The proxy keeps ``api.py`` importable even when Telegram/SMTP optional
    dependencies are absent, while still using the real SMTP sender at call
    time.  Tests can monkeypatch either ``api.send_email_native`` or
    ``bot_service.send_email_native``.
    """

    from bot_service import send_email_native as _send_email_native

    return _send_email_native(
        recipient_email, subject, body_text, extra_headers, attachment_path, html_body
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Create missing tables and ensure default demo user exists before serving."""

    init_db()
    with _get_db_context() as session:
        _ensure_default_user(session)
    logger.info("Saldora API ready; database tables initialised.")
    yield


app = FastAPI(
    title="Saldora B2B API",
    description="Invoice, storno, payment reminder and e-Ispratnica backend.",
    version="1.0.0",
    default_response_class=DecimalSafeJSONResponse,
    lifespan=lifespan,
)

# Required by the deployment checklist: the dashboard may be served from a
# different origin during development.
# Explicit production origins can be configured with ``CORS_ORIGINS``, a
# comma-separated list (e.g. the Render dashboard domain). Without it the API
# stays open for any external deployment domain; when the wildcard is used we
# disable cookie credentials, as browsers reject ``Allow-Origin: *`` with
# credentials. Bearer-token auth does not need cookies.
_raw_cors_origins = os.getenv("CORS_ORIGINS", "").strip()
if _raw_cors_origins:
    CORS_ALLOW_ORIGINS: list[str] = [
        origin.strip() for origin in _raw_cors_origins.split(",") if origin.strip()
    ]
else:
    external_origin = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    CORS_ALLOW_ORIGINS = [external_origin] if external_origin else ["*"]
CORS_ALLOW_CREDENTIALS = CORS_ALLOW_ORIGINS != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Global JSON error guardrail
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Last-resort guardrail: unexpected server errors must never surface as
    raw HTML/plain-text pages that crash the SPA with ``Unexpected token '<'``.

    Every unhandled exception is logged with its stack trace and returned as a
    JSON 500 with a stable shape (``detail`` + ``message``). Request-scoped
    database sessions are closed (and thus their transactions released) by the
    ``get_db`` dependency teardown; endpoints that hold a session wrap their
    own bodies with ``db.rollback()`` for Supabase connection-lock hygiene.
    """

    logger.exception(
        "Unhandled server error on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc), "message": "Failed to process request"},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# System Health & IDE Background Probe Endpoints (prevents 404 console spam)
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/v1/models")
def dummy_v1_models():
    return {"object": "list", "data": []}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the bundled dashboard."""

    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Frontend not found. Add static/index.html.",
        )
    return FileResponse(index_file)


# ---------------------------------------------------------------------------
# Authentication Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/signup",
    response_model=TokenResponse,
    status_code=http_status.HTTP_201_CREATED,
)
def signup(payload: UserCreate, db: DbSession) -> dict[str, Any]:
    """Register a new user account and return a JWT access token."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Корисничка сметка со оваа е-пошта веќе постои.",
        )
    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        company_name=payload.company_name.strip(),
        tax_id=payload.tax_id,
        terms_accepted=payload.terms_accepted,
        terms_accepted_at=datetime.utcnow() if payload.terms_accepted else None,
        terms_version=TERMS_VERSION if payload.terms_accepted else None,
        # Trusted API-direct path (tests, integrations): account is active
        # immediately. The UI wizard uses the OTP flow instead.
        is_verified=True,
    )
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to create user")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Грешка при креирање на корисникот.",
        ) from exc

    token = create_access_token({"sub": str(new_user.id), "email": new_user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "email": new_user.email,
            "company_name": new_user.company_name,
            "tax_id": new_user.tax_id,
        },
    }


# ---------------------------------------------------------------------------
# OTP Email-Verification Signup (wizard flow)
# ---------------------------------------------------------------------------

OTP_PURPOSE_SIGNUP = "signup_verification"
OTP_TTL_SECONDS = 300
OTP_CODE_LENGTH = 6


def _generate_otp_code() -> str:
    """Cryptographically random 6-digit code (leading zeros allowed)."""

    return "".join(secrets.choice(string.digits) for _ in range(OTP_CODE_LENGTH))


def _find_active_otp(db: Session, email: str, code: str) -> OTPCode | None:
    """Latest unused, unexpired signup code for the address that matches ``code``."""

    otp = (
        db.query(OTPCode)
        .filter(
            OTPCode.email == email,
            OTPCode.purpose == OTP_PURPOSE_SIGNUP,
            OTPCode.is_used.is_(False),
            OTPCode.expires_at > datetime.utcnow(),
        )
        .order_by(OTPCode.created_at.desc(), OTPCode.id.desc())
        .first()
    )
    if otp is None or not secrets.compare_digest(otp.code, code.strip()):
        return None
    return otp


def _otp_email_html(email: str, code: str, minutes: int) -> str:
    """Executive dark-mode verification email."""

    safe_email = html_module.escape(email)
    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background-color:#121212;font-family:'Segoe UI',Arial,sans-serif;">
  <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#121212;padding:32px 16px;">
    <tr><td align="center">
      <table role="presentation" width="100%" style="max-width:520px;background-color:#2D2E2B;border:1px solid #50605A;border-radius:16px;padding:40px 36px;">
        <tr><td style="padding-bottom:8px;">
          <span style="font-size:20px;font-weight:700;letter-spacing:0.08em;color:#B9CABE;text-transform:uppercase;">SALDORA</span>
          <span style="font-size:11px;color:#81938A;display:block;padding-top:2px;">Executive Financial Vault</span>
        </td></tr>
        <tr><td style="padding:18px 0 6px;border-top:1px solid #50605A;">
          <span style="font-size:17px;font-weight:600;color:#10B981;">Верификација на регистрација</span>
        </td></tr>
        <tr><td style="font-size:14px;line-height:1.6;color:#B9CABE;padding-bottom:24px;">
          Вашата службена е-пошта <strong style="color:#F0FDF9;">{safe_email}</strong> беше
          искористена за барање на сметка. Внесете го следниов 6-цифрен код за
          да ја активирате сметката:
        </td></tr>
        <tr><td align="center" style="padding-bottom:24px;">
          <table role="presentation" cellpadding="0" cellspacing="0"><tr>
            {''.join(
                f'<td align="center" style="width:52px;height:60px;background-color:#1c1d1a;border:1px solid #10B981;border-radius:10px;'
                f'font-size:26px;font-weight:700;color:#10B981;font-family:Consolas,monospace;">{digit}</td>'
                f'<td style="width:8px;">&nbsp;</td>' for digit in code)}
          </tr></table>
        </td></tr>
        <tr><td style="font-size:12px;line-height:1.6;color:#81938A;padding-bottom:6px;">
          Кодот е валиден <strong style="color:#B9CABE;">{minutes} минути</strong> и може да се искористи само еднаш.
          Ако вие не побаравте сметка, слободно игнорирајте ја пораката.
        </td></tr>
        <tr><td style="padding-top:18px;border-top:1px solid #50605A;font-size:11px;color:#81938A;">
          © Saldora — Институционална платформа за е-Фактури · Защитено со AES-256
        </td></tr>
      </table>
    </td></tr>
  </table>
</body></html>"""


def _send_signup_otp_email(recipient: str, subject: str, body_text: str, html_body: str) -> None:
    """Background-task worker for signup OTP delivery.

    ``send_email_native`` never raises, so a provider outage only logs here —
    the client already received an instant response.
    """

    ok, message = send_email_native(recipient, subject, body_text, html_body=html_body)
    if not ok:
        logger.error("OTP email dispatch to %s failed: %s", recipient, message)
    else:
        logger.info("Signup OTP dispatched to %s.", recipient)


@app.post("/api/signup/request-otp")
def request_signup_otp(
    payload: SignupOTPRequest,
    background_tasks: BackgroundTasks,
    db: DbSession,
) -> dict[str, Any]:
    """Validate the address and password policy, then dispatch a 6-digit code.

    The wizard includes ``password`` and ``password_confirm`` so an invalid
    password is rejected before an OTP email is sent. API clients that omit the
    password still receive a code; the final ``complete`` call always enforces
    the policy again. The email itself is dispatched via ``BackgroundTasks`` so
    the endpoint responds instantly without blocking on the mail provider.
    """

    if payload.password is not None:
        _validate_password_or_400(payload.password, payload.password_confirm)

    email = payload.email
    existing = db.query(User).filter(User.email == email).first()
    if existing and existing.is_verified:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Сметката веќе постои.",
        )

    # Invalidate any previous pending codes so exactly one is active.
    db.query(OTPCode).filter(OTPCode.email == email, OTPCode.is_used.is_(False)).update(
        {OTPCode.is_used: True}, synchronize_session=False
    )
    code = _generate_otp_code()
    otp = OTPCode(
        email=email,
        code=code,
        purpose=OTP_PURPOSE_SIGNUP,
        expires_at=datetime.utcnow() + timedelta(seconds=OTP_TTL_SECONDS),
    )
    db.add(otp)
    db.commit()

    subject = "Верификационен код за Saldora регистрација"
    body_text = (
        f"Вашиот 6-цифрен верификационен код е: {code}\n\n"
        f"Кодот е валиден 5 минути и може да се искористи само еднаш.\n"
        f"Ако вие не побаравте сметка, игнорирајте ја пораката.\n\nSaldora"
    )
    background_tasks.add_task(
        _send_signup_otp_email,
        email,
        subject,
        body_text,
        _otp_email_html(email, code, OTP_TTL_SECONDS // 60),
    )
    return {"status": "sent", "expires_in_seconds": OTP_TTL_SECONDS}


@app.post("/api/signup/verify-otp")
def verify_signup_otp(payload: SignupVerifyRequest, db: DbSession) -> dict[str, str]:
    """Non-consuming validity check so the UI can auto-advance to the last step."""

    if _find_active_otp(db, payload.email, payload.otp_code) is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Невалиден или истечен верификационен код.",
        )
    return {"status": "valid"}


@app.post(
    "/api/signup/complete",
    response_model=TokenResponse,
    status_code=http_status.HTTP_201_CREATED,
)
def complete_signup(payload: SignupCompleteRequest, db: DbSession) -> dict[str, Any]:
    """Verify the OTP, activate the account, and issue a JWT access token."""

    # Policy and consent are checked before OTP lookup so the caller receives
    # the actionable Macedonian validation message.
    _validate_password_or_400(payload.password, payload.password_confirm)
    if not payload.terms_accepted:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=(
                "Мора да се согласите со Условите за користење и "
                "Политиката за приватност."
            ),
        )

    otp = _find_active_otp(db, payload.email, payload.otp_code)
    if otp is None:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Невалиден или истечен верификационен код.",
        )
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Сметката веќе постои.",
        )

    otp.is_used = True
    new_user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        company_name=payload.company_name.strip(),
        tax_id=payload.tax_id,
        terms_accepted=True,
        terms_accepted_at=datetime.utcnow(),
        terms_version=TERMS_VERSION,
        is_verified=True,
    )
    db.add(new_user)
    try:
        db.commit()
        db.refresh(new_user)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to complete OTP signup for %s", payload.email)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Грешка при креирање на корисникот.",
        ) from exc

    token = create_access_token({"sub": str(new_user.id), "email": new_user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": new_user.id,
            "email": new_user.email,
            "company_name": new_user.company_name,
            "tax_id": new_user.tax_id,
        },
    }


async def _oauth2_form_or_none(request: Request) -> OAuth2PasswordRequestForm | None:
    """Parse OAuth2 form credentials when the request is not JSON.

    ``/api/login`` accepts both the JSON payload used by the bundled frontend
    and the standard ``application/x-www-form-urlencoded`` OAuth2 form.
    """
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        return None
    form = await request.form()
    return OAuth2PasswordRequestForm(
        grant_type=form.get("grant_type"),
        username=str(form.get("username") or form.get("email") or ""),
        password=str(form.get("password") or ""),
        scope=str(form.get("scope") or ""),
        client_id=form.get("client_id"),
        client_secret=form.get("client_secret"),
    )


def _authenticate(db: Session, email: str, password: str) -> User:
    """Resolve credentials to a User or raise the canonical 401."""
    normalized_email = str(email).strip().lower()
    if not normalized_email or not password:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Задолжителни се е-пошта и лозинка.",
        )

    user = db.query(User).filter(User.email == normalized_email).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Невалидна е-пошта или лозинка.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def _token_payload(user: User) -> dict[str, Any]:
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "email": user.email,
            "company_name": user.company_name,
            "tax_id": user.tax_id,
        },
    }


@app.post("/api/login", response_model=TokenResponse)
async def login(
    request: Request,
    db: DbSession,
    form_data: Annotated[OAuth2PasswordRequestForm | None, Depends(_oauth2_form_or_none)] = None,
) -> dict[str, Any]:
    """Authenticate via JSON or OAuth2 form data and return a JWT access token."""
    if form_data is not None:
        email = form_data.username
        password = form_data.password
    else:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        if not isinstance(body, dict):
            body = {}
        email = body.get("email") or body.get("username") or ""
        password = body.get("password") or ""

    user = _authenticate(db, email, password)
    return _token_payload(user)


@app.post("/api/token", response_model=TokenResponse)
def login_oauth2(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: DbSession,
) -> dict[str, Any]:
    """Standard OAuth2 password flow used by Swagger's Authorize dialog."""
    user = _authenticate(db, form_data.username, form_data.password)
    return _token_payload(user)


@app.get("/api/users/me", response_model=UserProfileResponse)
def get_me(current_user: CurrentUser) -> dict[str, Any]:
    """Return profile data of the authenticated user."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "company_name": current_user.company_name,
        "tax_id": current_user.tax_id,
        "created_at": current_user.created_at,
    }


# ---------------------------------------------------------------------------
# Documenso e-Signature Configuration
# ---------------------------------------------------------------------------


def _documenso_config_payload(
    request: Request, user: User
) -> dict[str, Any]:
    env_key = os.environ.get("DOCUMENSO_API_KEY", "").strip()
    key = (user.documenso_api_key or env_key).strip()
    configured = is_configured_api_key(key)
    webhook_url = f"{str(request.base_url).rstrip('/')}/api/webhooks/documenso"
    return {
        "api_key_configured": configured,
        "api_key_masked": _mask_secret(key) if configured else "",
        "base_url": _effective_documenso_base_url(user),
        "webhook_url": webhook_url,
    }


@app.get("/api/settings/documenso", response_model=DocumensoConfigResponse)
def get_documenso_settings(
    request: Request,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Return masked Documenso configuration for the Settings card."""

    return _documenso_config_payload(request, current_user)


@app.put("/api/settings/documenso", response_model=DocumensoConfigResponse)
def update_documenso_settings(
    payload: DocumensoConfigUpdateRequest,
    request: Request,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Persist per-account Documenso credentials (API key encrypted at rest)."""

    if payload.clear_api_key:
        current_user.documenso_api_key = None

    if payload.api_key is not None:
        cleaned = payload.api_key.strip()
        if cleaned:
            if cleaned.casefold() in {"changeme", "replace_me"}:
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail="Внесете валиден Documenso API клуч.",
                )
            current_user.documenso_api_key = cleaned

    if payload.base_url is not None:
        cleaned_base = payload.base_url.strip()
        if cleaned_base:
            try:
                current_user.documenso_base_url = normalize_base_url(cleaned_base)
            except DocumensoConfigError as exc:
                raise HTTPException(
                    status_code=http_status.HTTP_400_BAD_REQUEST,
                    detail="Внесете валиден Documenso URL (http или https).",
                ) from exc
        else:
            current_user.documenso_base_url = None

    try:
        db.commit()
        db.refresh(current_user)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Documenso settings update failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не успеаја да се зачуваат Documenso поставките.",
        ) from exc

    return _documenso_config_payload(request, current_user)


# ---------------------------------------------------------------------------
# Business Endpoints (Protected & Scoped by CurrentUser)
# ---------------------------------------------------------------------------


@app.get("/api/stats", response_model=StatsResponse)
def get_stats(db: DbSession, current_user: CurrentUser) -> dict[str, Any]:
    """Return the dashboard KPI overview for the authenticated user."""

    try:
        total_revenue = (
            db.query(func.coalesce(func.sum(Invoice.total_amount), 0))
            .filter(Invoice.status == "PAID", Invoice.user_id == current_user.id)
            .scalar()
        )
        unpaid_amount = (
            db.query(func.coalesce(func.sum(Invoice.total_amount), 0))
            .filter(Invoice.status == "UNPAID", Invoice.user_id == current_user.id)
            .scalar()
        )
        unpaid_count = (
            db.query(func.count(Invoice.id))
            .filter(Invoice.status == "UNPAID", Invoice.user_id == current_user.id)
            .scalar()
        )
        pending_notes = (
            db.query(func.count(DeliveryNote.id))
            .filter(DeliveryNote.is_signed.is_(False), DeliveryNote.user_id == current_user.id)
            .scalar()
        )
        pending_drafts = (
            db.query(func.count(EmailDraft.id))
            .filter(
                EmailDraft.status == "PENDING_APPROVAL",
                EmailDraft.user_id == current_user.id,
            )
            .scalar()
        )
    except SQLAlchemyError as exc:
        logger.exception("Stats query failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read dashboard statistics.",
        ) from exc

    return {
        "total_revenue": float(_to_float(total_revenue) or 0.0),
        "unpaid_amount": float(_to_float(unpaid_amount) or 0.0),
        "unpaid_count": int(unpaid_count or 0),
        "pending_ispratnici": int(pending_notes or 0),
        "pending_email_drafts": int(pending_drafts or 0),
    }


@app.get("/api/invoices", response_model=list[InvoiceResponse])
def list_invoices(
    db: DbSession,
    current_user: CurrentUser,
    status_filter: str | None = Query(
        default=None,
        alias="status",
        description="Optional status filter: UNPAID, PAID or STORNO.",
    ),
) -> list[dict[str, Any]]:
    """List invoices newest-first for current user, optionally filtered by status."""

    query = db.query(Invoice).filter(Invoice.user_id == current_user.id)

    if status_filter is not None:
        normalized = status_filter.strip().upper()
        if normalized not in ALLOWED_STATUSES:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{status_filter}'. Use UNPAID, PAID or STORNO.",
            )
        query = query.filter(Invoice.status == normalized)

    try:
        invoices = query.order_by(Invoice.created_at.desc(), Invoice.id.desc()).all()
    except SQLAlchemyError as exc:
        logger.exception("Invoice list query failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read invoices.",
        ) from exc

    return [_serialize_invoice(invoice) for invoice in invoices]


@app.post(
    "/api/invoices",
    response_model=InvoiceResponse,
    status_code=http_status.HTTP_201_CREATED,
)
def create_invoice(
    payload: InvoiceCreateRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any] | JSONResponse:
    """JSON-guardrail wrapper: any unexpected failure rolls the session back
    (releasing Supabase connection locks) and returns a JSON 500 instead of an
    HTML/plain-text error page that would crash the frontend's ``res.json()``.
    """

    try:
        return _create_invoice_impl(payload, db, current_user)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a JSON 500
        db.rollback()
        logger.exception("Invoice creation failed unexpectedly")
        return JSONResponse(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "message": "Failed to process request"},
        )


def _create_invoice_impl(
    payload: InvoiceCreateRequest,
    db: Session,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Create a VAT-inclusive invoice for current user and allocate its code."""

    vat_rate_decimal = Decimal(str(payload.vat_rate)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    total = Decimal(str(payload.amount)).quantize(CENT, rounding=ROUND_HALF_UP)
    net, vat = _vat_breakdown(total, vat_rate_decimal)
    client_name = payload.client_name.strip()
    description = payload.description.strip()
    client_tax_id = (payload.client_tax_id or "").strip() or None
    client_address = (payload.client_address or "").strip() or None

    if total <= 0:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Amount must be greater than zero.",
        )

    if client_tax_id is not None:
        registered = (
            db.query(Client)
            .filter(Client.tax_id == client_tax_id, Client.user_id == current_user.id)
            .first()
        )
        if registered is None:
            logger.info("Invoice will reference a newly seen client tax id.")

    invoice: Invoice | None = None

    for attempt in range(1, MAX_CODE_ALLOCATION_ATTEMPTS + 1):
        code = _next_invoice_code(db)
        invoice = Invoice(
            id=code,
            code=code,
            user_id=current_user.id,
            client_tax_id=client_tax_id,
            client_name=client_name,
            description=description,
            amount=total,
            net_amount=net,
            vat_rate=vat_rate_decimal,
            vat_amount=vat,
            total_amount=total,
            status="UNPAID",
            state=InvoiceState.UNPAID,
            is_approved=False,
            due_days=0,
            client_address=client_address,
            issue_date=payload.issue_date,
            due_date=payload.due_date,
        )
        db.add(invoice)

        try:
            db.commit()
            db.refresh(invoice)
        except IntegrityError as exc:
            db.rollback()
            message = str(exc.orig).lower()
            is_code_collision = "unique" in message or "primary key" in message
            if not is_code_collision or attempt == MAX_CODE_ALLOCATION_ATTEMPTS:
                logger.exception("Could not persist invoice after %d attempt(s)", attempt)
                raise HTTPException(
                    status_code=http_status.HTTP_409_CONFLICT,
                    detail="Invoice could not be stored; the generated code is already in use.",
                ) from exc
            logger.warning("Invoice code collision on attempt %d; retrying.", attempt)
            continue
        except SQLAlchemyError as exc:
            db.rollback()
            logger.exception("Invoice insert failed")
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create invoice.",
            ) from exc

        # Generate and persist UBL XML
        try:
            XML_DIR.mkdir(parents=True, exist_ok=True)
            xml_content = _generate_ubl_xml(invoice)
            xml_path = XML_DIR / f"{code}.xml"
            xml_path.write_text(xml_content, encoding="utf-8")
            invoice.xml_file_path = str(xml_path)
            db.commit()
            db.refresh(invoice)
        except Exception:  # noqa: BLE001
            logger.exception("UBL XML generation failed for %s; continuing without XML.", code)

        return _serialize_invoice(invoice)

    # The loop either returns or raises; this is only a static-analysis guard.
    raise HTTPException(
        status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Could not allocate a unique invoice code.",
    )


@app.post("/api/invoices/{code}/approve", response_model=InvoiceResponse)
def approve_invoice(
    code: str,
    payload: ApproveRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Manually approve an invoice (human-in-the-loop sign-off) for current user."""
    invoice = _require_owned_invoice(db, code, current_user.id)
    if invoice.is_approved:
        raise HTTPException(status_code=409, detail="Invoice is already approved.")
    invoice.is_approved = True
    invoice.approved_by = payload.approved_by.strip()
    invoice.state = InvoiceState.APPROVED
    try:
        db.commit()
        db.refresh(invoice)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Approval failed for %s", code)
        raise HTTPException(status_code=500, detail="Could not approve invoice.") from exc
    return _serialize_invoice(invoice)


@app.post("/api/invoices/{code}/storno", response_model=StornoResponse)
@app.post("/api/invoices/{code}/storniraj", response_model=StornoResponse)
def storno_invoice(
    code: str,
    db: DbSession,
    current_user: CurrentUser,
    payload: StornoRequest | None = None,
) -> dict[str, Any]:
    """Mark an invoice as storno and return UBL 2.1 credit-note details."""

    invoice = _require_owned_invoice(db, code, current_user.id)

    reason = (payload.reason if payload and payload.reason else "").strip()
    if not reason:
        reason = DEFAULT_STORNO_REASON

    # Log the cancellation request for audit
    cr = CancellationRequest(
        invoice_id=invoice.id,
        reason=reason,
        is_reviewed=True,  # Treated as reviewed since operator submitted it
        reviewed_by="operator",
    )
    db.add(cr)

    invoice.status = "STORNO"
    invoice.state = InvoiceState.STORNO

    try:
        db.commit()
        db.refresh(invoice)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Storno update failed for %s", code)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not storno the invoice.",
        ) from exc

    invoice_id = invoice.id
    receipt = {
        "ubl_version": "2.1",
        "document_type": "CreditNote",
        "type_code": "381",
        "id": f"STORNO-{invoice_id}",
        "issue_date": datetime.utcnow().date(),
        "currency": "MKD",
        "original_invoice_id": invoice_id,
        "corrected_amount": float(_to_float(invoice.total_amount) or 0.0),
        "reason": reason,
    }

    return {
        "success": True,
        "id": invoice_id,
        "code": invoice.code or invoice.id,
        "status": invoice.status,
        "reason": reason,
        "ubl": {
            "version": "2.1",
            "document_type": "CreditNote",
            "profile": "e-Faktura MK",
        },
        "correction_receipt": receipt,
    }


@app.post("/api/invoices/{code}/remind", response_model=RemindResponse)
async def remind_invoice(
    code: str,
    payload: RemindRequest,
    background_tasks: BackgroundTasks,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any] | JSONResponse:
    """Send a real payment reminder for an invoice (non-blocking dispatch)."""

    invoice = _require_owned_invoice(db, code, current_user.id)

    recipient = payload.recipient_email.strip()
    invoice_code = invoice.code or invoice.id
    client_name = invoice.client_name or "Клиент"
    amount = _to_float(invoice.total_amount or invoice.amount) or 0.0

    subject = f"Потсетник за неплатена фактура {invoice_code} - Saldora"
    body = (
        f"Почитувани {client_name},\n\n"
        f"Ве известуваме дека фактурата {invoice_code} на износ "
        f"{amount:,.2f} МКД е со задоцнета наплата.\n"
        f"Ве молиме извршете ја уплатата во најкраток можен рок.\n\n"
        f"Со почит,\nSaldora B2B Систем"
    )

    # Determine XML attachment if available
    attachment_file: str | None = None
    if invoice.xml_file_path and Path(invoice.xml_file_path).is_file():
        attachment_file = invoice.xml_file_path
    elif (XML_DIR / f"{invoice_code}.xml").is_file():
        attachment_file = str(XML_DIR / f"{invoice_code}.xml")

    # End the read transaction before the background delivery task runs, so a
    # slow mail server cannot hold a SQLite lock and stall invoice writes.
    db.rollback()

    # Deliver via a FastAPI background task (Starlette runs the sync sender in
    # the threadpool): the client gets an instant response, and delivery
    # failures are logged instead of blocking the request.
    background_tasks.add_task(
        send_email_native,
        recipient,
        subject,
        body,
        None,
        attachment_file,
    )

    return {"success": True, "message": f"Email sent to {recipient}"}


@app.get("/api/invoices/{code}/xml")
def download_invoice_xml(
    code: str,
    db: DbSession,
    current_user: CurrentUser,
) -> FileResponse:
    """Download the generated UBL 2.1 XML for an invoice."""
    invoice = _find_invoice(db, code, user_id=current_user.id)
    if invoice is None:
        raise HTTPException(status_code=404, detail=f"Invoice '{code}' not found.")
    xml_path = invoice.xml_file_path
    if not xml_path or not Path(xml_path).is_file():
        raise HTTPException(status_code=404, detail="XML file not generated yet.")
    return FileResponse(xml_path, media_type="application/xml", filename=f"{code}.xml")


@app.get("/api/ispratnica", response_model=list[DeliveryNoteResponse])
def list_delivery_notes(
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    """List every e-Ispratnica delivery note for current user."""

    try:
        notes = (
            db.query(DeliveryNote)
            .filter(DeliveryNote.user_id == current_user.id)
            .order_by(DeliveryNote.id.asc())
            .all()
        )
    except SQLAlchemyError as exc:
        logger.exception("Delivery note list query failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read delivery notes.",
        ) from exc

    return [_serialize_delivery_note(note) for note in notes]


@app.get("/api/ispratnici", response_model=list[DeliveryNoteResponse])
def list_delivery_notes_plural(
    db: DbSession,
    current_user: CurrentUser,
) -> list[dict[str, Any]]:
    """Alias for /api/ispratnica — list all e-Ispratnici for current user."""
    return list_delivery_notes(db, current_user)


@app.get("/api/ispratnici/{identifier}", response_model=DeliveryNoteResponse)
def get_delivery_note(
    identifier: str,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Return one complete e-Ispratnica record, including its Base64 signature.

    ``identifier`` accepts either the numeric primary key or the document code
    (for example ``ISP-2026-0007``), so the document viewer can address a note
    without an extra lookup round-trip.
    """

    try:
        note = _find_delivery_note(db, identifier, current_user.id)
    except SQLAlchemyError as exc:
        logger.exception("Delivery note detail query failed for %s", identifier)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the delivery note.",
        ) from exc

    if note is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Delivery note '{identifier}' was not found.",
        )

    return _serialize_delivery_note(note)


@app.post("/api/ispratnici", response_model=DeliveryNoteResponse, status_code=201)
def create_delivery_note(
    payload: DeliveryNoteCreateRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any] | JSONResponse:
    """JSON-guardrail wrapper for e-Ispratnica creation (same contract as
    ``create_invoice``: rollback on failure, JSON 500, never raw HTML)."""

    try:
        return _create_delivery_note_impl(payload, db, current_user)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:  # noqa: BLE001 - converted to a JSON 500
        db.rollback()
        logger.exception("Delivery-note creation failed unexpectedly")
        return JSONResponse(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "message": "Failed to process request"},
        )


def _create_delivery_note_impl(
    payload: DeliveryNoteCreateRequest,
    db: Session,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Create a new e-Ispratnica with digital signature and file to UJP.

    The code is allocated with ``generate_guaranteed_unique_code`` immediately
    before the model is constructed, and the insert is retried with a freshly
    regenerated code when the database reports a unique-constraint collision
    (e.g. a concurrent writer won the same sequence number).
    """

    issue = payload.issue_date or datetime.utcnow().date()
    collision: IntegrityError | None = None

    for attempt in range(1, MAX_CODE_ALLOCATION_ATTEMPTS + 1):
        note_code = _next_delivery_note_code(db)
        note = DeliveryNote(
            code=note_code,
            user_id=current_user.id,
            invoice_id=payload.invoice_id,
            recipient_name=payload.client_name,
            client_name=payload.client_name,
            delivery_address=payload.delivery_address,
            items_description=payload.items_description,
            issue_date=issue,
            details=payload.items_description,
            signature_data=payload.signature_data,
            is_signed=True,
            signed_at=datetime.utcnow(),
            ujp_status="Sent to UJP",
        )
        db.add(note)
        try:
            db.commit()
            db.refresh(note)
        except IntegrityError as exc:
            db.rollback()
            collision = exc
            logger.warning(
                "Delivery-note code collision on attempt %d; regenerating.", attempt
            )
            continue
        except SQLAlchemyError as exc:
            db.rollback()
            logger.exception("Delivery note creation failed")
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create delivery note.",
            ) from exc

        logger.info("e-Ispratnica %s filed to UJP (simulated).", note_code)
        return _serialize_delivery_note(note)

    logger.exception(
        "Could not persist delivery note after %d attempt(s)",
        MAX_CODE_ALLOCATION_ATTEMPTS,
    )
    raise HTTPException(
        status_code=http_status.HTTP_409_CONFLICT,
        detail="Delivery note could not be stored; the generated code is already in use.",
    ) from collision


@app.post("/api/ispratnica/{code}/sign", response_model=SignResponse)
def sign_delivery_note(
    code: str,
    payload: SignRequest,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Record the e-signature acceptance flag for a delivery note."""

    note = _require_owned_delivery_note(db, code, current_user.id)

    # Persist the Base64 PNG so the e-Ispratnica document viewer can render the
    # signature captured on the field device, and update the legal state/time.
    note.signature_data = payload.signature_data
    note.is_signed = True
    note.signed_at = datetime.utcnow()

    try:
        db.commit()
        db.refresh(note)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Delivery note signing failed for %s", code)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not sign the delivery note.",
        ) from exc

    return {
        "success": True,
        "id": note.id,
        "code": note.code,
        "is_signed": bool(note.is_signed),
        "signed_at": note.signed_at,
    }


def _load_documenso_api_key(db: Session, user: User) -> str:
    """Return the freshest Documenso API key stored for ``user``.

    The key is read from SQLite on every call so a value saved in Settings is
    picked up immediately, even if the in-memory ``current_user`` object is
    stale. When the database value is empty, fall back to the environment.
    """

    stored_key = db.execute(
        select(User.documenso_api_key).where(User.id == user.id)
    ).scalar_one_or_none()
    key = (stored_key or "").strip()
    if not key:
        key = (os.environ.get("DOCUMENSO_API_KEY") or "").strip()
    return key


@app.post(
    "/api/ispratnici/{identifier}/sign-documenso",
    response_model=DocumensoSignResponse,
)
def sign_delivery_note_documenso(
    identifier: str,
    db: DbSession,
    current_user: CurrentUser,
    payload: DocumensoSignRequest | None = None,
) -> dict[str, Any]:
    """Create a Documenso document and return the embedded signing URL."""

    note = _require_owned_delivery_note(db, identifier, current_user.id)

    if note.documenso_status in {"SIGNED", "COMPLETED"}:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail="Овој документ е веќе дигитално потпишан преку Documenso.",
        )

    # Re-use an active signing session instead of creating duplicate documents.
    if note.documenso_status == "SENT" and note.documenso_signing_url:
        return {
            "signing_url": note.documenso_signing_url,
            "doc_id": note.documenso_document_id or "",
            "status": "SENT",
        }

    recipient_email = (
        payload.recipient_email if payload and payload.recipient_email else current_user.email
    )
    title = f"е-Испратница {note.code} — {note.client_name or note.recipient_name}"

    try:
        pdf_bytes = build_delivery_note_pdf(note)
    except Exception as exc:  # pragma: no cover - depends on local font stack
        logger.exception("Failed to render e-Ispratnica PDF for Documenso")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не успеа генерирањето на PDF документот.",
        ) from exc

    client: DocumensoClient | None = None
    try:
        api_key = _load_documenso_api_key(db, current_user)
        client = DocumensoClient(
            api_key=api_key,
            base_url=current_user.documenso_base_url,
        )
        logger.info("Using Documenso key: %s...", client.api_key[:8])
        result = client.create_document(
            pdf_bytes,
            title=title,
            recipient_email=recipient_email,
            recipient_name=note.client_name or note.recipient_name,
            external_id=note.code,
        )
    except ValueError as exc:
        logger.error("Documenso returned an invalid document identifier: %s", exc)
        raise HTTPException(
            status_code=http_status.HTTP_502_BAD_GATEWAY,
            detail="Documenso не врати валиден идентификатор на документот.",
        ) from exc
    except DocumensoConfigError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except DocumensoAPIError as exc:
        logger.warning(
            "Documenso document creation failed for %s (status=%s): %s",
            note.code,
            exc.status_code,
            exc,
        )
        raise HTTPException(
            status_code=http_status.HTTP_502_BAD_GATEWAY,
            detail="Documenso не успеа да го подготви документот за потпишување.",
        ) from exc
    finally:
        if client is not None:
            client.close()

    note.documenso_document_id = result["document_id"]
    note.documenso_status = "SENT"
    note.documenso_signing_url = result["signing_url"]
    note.documenso_recipient_email = recipient_email

    try:
        db.commit()
        db.refresh(note)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Failed to persist Documenso document for %s", note.code)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Не успеа зачувувањето на Documenso сесијата.",
        ) from exc

    logger.info(
        "Documenso document %s created for e-Ispratnica %s.",
        result["document_id"],
        note.code,
    )
    return {
        "signing_url": result["signing_url"],
        "doc_id": result["document_id"],
        "status": "SENT",
    }


@app.post("/api/webhooks/documenso")
async def documenso_webhook(request: Request, db: DbSession) -> dict[str, Any]:
    """Receive Documenso lifecycle events and seal the signature audit trail."""

    received_secret = request.headers.get("x-documenso-secret")
    if not DocumensoClient.verify_webhook_secret(received_secret):
        logger.warning("Rejected Documenso webhook with an invalid secret.")
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Documenso webhook secret.",
        )

    raw_body = await request.body()
    if len(raw_body) > 1_000_000:
        raise HTTPException(
            status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Webhook payload is too large.",
        )

    try:
        event = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid Documenso webhook payload.",
        ) from exc

    if not isinstance(event, dict):
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="Invalid Documenso webhook payload.",
        )

    event_type = event.get("event") or event.get("type")
    payload = event.get("payload")
    if not isinstance(payload, dict):
        payload = event

    if event_type != "DOCUMENT_COMPLETED":
        return {"received": True, "matched": False, "status": "IGNORED"}

    document_id = payload.get("id") or payload.get("documentId")
    external_id = payload.get("externalId")

    note: DeliveryNote | None = None
    try:
        if document_id is not None:
            note = (
                db.query(DeliveryNote)
                .filter(DeliveryNote.documenso_document_id == str(document_id))
                .first()
            )
        if note is None and external_id:
            note = (
                db.query(DeliveryNote)
                .filter(DeliveryNote.code == str(external_id))
                .first()
            )
    except SQLAlchemyError as exc:
        logger.exception("Documenso webhook lookup failed")
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the Documenso webhook.",
        ) from exc

    if note is None:
        logger.warning(
            "Documenso webhook for unknown document %s (%s).",
            document_id,
            external_id,
        )
        return {"received": True, "matched": False, "status": "NOT_FOUND"}

    if note.documenso_status in {"SIGNED", "COMPLETED"}:
        return {"received": True, "matched": True, "status": "SIGNED"}

    canonical_log, verification_hash = _build_documenso_audit(event, payload)
    signed_at = (
        _parse_iso_datetime(payload.get("completedAt"))
        or _parse_iso_datetime(event.get("createdAt"))
        or datetime.utcnow()
    )

    note.is_signed = True
    note.signed_at = signed_at
    note.documenso_status = "SIGNED"
    note.documenso_signed_at = signed_at
    note.documenso_signature_log = canonical_log
    note.documenso_verification_hash = verification_hash
    if document_id is not None:
        note.documenso_document_id = str(document_id)

    try:
        db.commit()
        db.refresh(note)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Documenso webhook persistence failed for %s", note.code)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not persist the Documenso signature.",
        ) from exc

    logger.info(
        "Documenso signature sealed for e-Ispratnica %s (hash=%s).",
        note.code,
        verification_hash[:16],
    )
    return {"received": True, "matched": True, "status": "SIGNED"}


# ---------------------------------------------------------------------------
# Human-in-the-Loop AI Email Draft Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/email-drafts", response_model=list[EmailDraftResponse])
def list_email_drafts(
    db: DbSession,
    current_user: CurrentUser,
    status: str | None = Query(default=None, description="Optional status filter"),
) -> list[dict[str, Any]]:
    """List incoming client email drafts and their AI responses for current user."""
    query = db.query(EmailDraft).filter(EmailDraft.user_id == current_user.id)
    if status is not None:
        query = query.filter(EmailDraft.status == status.strip().upper())
    drafts = query.order_by(
        (EmailDraft.status == "PENDING_APPROVAL").desc(),
        EmailDraft.created_at.desc(),
        EmailDraft.id.desc(),
    ).all()
    return [_serialize_email_draft(d) for d in drafts]


def _require_owned_draft(db: Session, draft_id: int, user_id: int) -> EmailDraft:
    """Fetch an email draft for a write operation, distinguishing 403 from 404."""

    draft = db.query(EmailDraft).filter(EmailDraft.id == draft_id).first()
    if draft is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Email draft #{draft_id} not found.",
        )
    if draft.user_id != user_id:
        logger.warning("Cross-tenant write on email draft #%d by user %s denied.", draft_id, user_id)
        raise HTTPException(
            status_code=http_status.HTTP_403_FORBIDDEN,
            detail="Немате дозвола да го менувате овој документ.",
        )
    return draft


@app.post("/api/email-drafts/{draft_id}/approve")
async def approve_email_draft(
    draft_id: int,
    background_tasks: BackgroundTasks,
    db: DbSession,
    current_user: CurrentUser,
    payload: EmailDraftApproveRequest | None = None,
) -> dict[str, Any]:
    """Approve an AI email draft (optionally edited) and dispatch it in the background."""
    draft = _require_owned_draft(db, draft_id, current_user.id)

    body_to_send = (
        payload.edited_body.strip()
        if payload and payload.edited_body and payload.edited_body.strip()
        else draft.ai_draft_response.strip()
    )

    # Attach invoice XML if linked invoice has one
    attachment_path: str | None = None
    if draft.invoice_id:
        inv = _find_invoice(db, draft.invoice_id, user_id=current_user.id)
        if inv and inv.xml_file_path and Path(inv.xml_file_path).is_file():
            attachment_path = inv.xml_file_path
        elif (XML_DIR / f"{draft.invoice_id}.xml").is_file():
            attachment_path = str(XML_DIR / f"{draft.invoice_id}.xml")

    subject = draft.incoming_subject
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}"

    draft.status = "SENT"
    draft.ai_draft_response = body_to_send
    try:
        db.commit()
        db.refresh(draft)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Could not commit draft #%d status", draft_id)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update draft state.",
        ) from exc

    # Deliver via a FastAPI background task (Starlette runs the sync sender in
    # the threadpool): the approval result returns instantly and the Resend /
    # SMTP round-trip happens off-request. ``send_email_native`` never raises,
    # so delivery failures surface only in the log.
    background_tasks.add_task(
        send_email_native,
        draft.client_email,
        subject,
        body_to_send,
        None,
        attachment_path,
    )

    return {
        "success": True,
        "message": f"Email successfully dispatched to {draft.client_email}",
        "draft": _serialize_email_draft(draft),
    }


@app.post("/api/email-drafts/{draft_id}/reject")
def reject_email_draft(
    draft_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Discard an AI email draft without sending."""
    draft = _require_owned_draft(db, draft_id, current_user.id)
    draft.status = "REJECTED"
    try:
        db.commit()
        db.refresh(draft)
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Could not reject draft #%d", draft_id)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update draft state.",
        ) from exc

    return {
        "success": True,
        "message": f"Draft #{draft_id} rejected.",
        "draft": _serialize_email_draft(draft),
    }


if __name__ == "__main__":
    import uvicorn

    # Render (and most container platforms) inject the public port via $PORT.
    # Bind to all interfaces and honour it; reload stays dev-only.
    try:
        port = int(os.getenv("PORT", "8000"))
    except ValueError:
        logger.warning("Invalid $PORT value %r; falling back to 8000.", os.getenv("PORT"))
        port = 8000
    uvicorn.run("api:app", host="0.0.0.0", port=port)
