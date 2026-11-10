"""Legacy domain models shared by FakturaBot.mk and Saldora (bridge module).

This module preserves the ORIGINAL dataclass-based contract from before the
SQLAlchemy 2.0 persistence layer landed, so every pre-existing consumer
(``config.py``, ``data_handler.py``, ``main.py``, ``tone_agent.py``,
``ujp_service.py``, ``dispatcher.py``, ``reply_parser.py`` and the test suite)
keeps working while the storage backend migrates to the new
``models.py`` / ``database.py`` / ``migration.py`` stack.

The legacy invoice dataclass is exported as ``LegacyInvoice``; ``Invoice = LegacyInvoice``
is kept as a backward-compatibility alias inside this module so existing import
statements continue to resolve it, and new code can import ``Invoice`` from
``models.py`` (the SQLAlchemy ORM model) without name collisions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from utils import (
    calculate_vat_breakdown,
    normalize_channel,
    normalize_language_code,
    normalize_phone,
    normalize_status,
    normalize_tax_id,
    now_utc,
    parse_date,
    parse_decimal,
)

__all__ = [
    "Channel",
    "DispatchResult",
    "Invoice",
    "Language",
    "LegacyInvoice",
    "ParsedInvoice",
    "ReplyAnalysis",
    "ReplyCategory",
    "ToneTier",
    "UJPSubmissionResult",
    "UJPValidationResult",
]


class Language(str, Enum):
    """Supported client communication languages."""

    MK = "MK"
    AL = "AL"
    EN = "EN"

    @classmethod
    def from_value(cls, value: Any, default: Language | None = None) -> Language:
        fallback = default or cls.EN
        return cls(normalize_language_code(value, fallback.value))


class Channel(str, Enum):
    """Supported outbound reminder channels."""

    EMAIL = "Email"
    VIBER = "Viber"
    WHATSAPP = "WhatsApp"

    @classmethod
    def from_value(cls, value: Any, default: Channel | None = None) -> Channel:
        fallback = default or cls.EMAIL
        return cls(normalize_channel(value, fallback.value))


class ToneTier(str, Enum):
    """Relationship-safe escalation tiers for collections."""

    FRIENDLY = "friendly"
    STANDARD = "standard"
    FIRM = "firm"

    @classmethod
    def from_days_late(cls, days_late: int) -> ToneTier:
        """Map overdue days to the exact product tiers.

        1-7   -> Friendly
        8-20  -> Standard
        21+   -> Firm
        """

        if days_late <= 7:
            return cls.FRIENDLY
        if days_late <= 20:
            return cls.STANDARD
        return cls.FIRM

    @property
    def label(self) -> str:
        return {
            ToneTier.FRIENDLY: "Friendly nudge",
            ToneTier.STANDARD: "Standard follow-up",
            ToneTier.FIRM: "Firm collection",
        }[self]


class ReplyCategory(str, Enum):
    """Normalized categories for inbound client replies."""

    PAID = "paid"
    EXTENSION_REQUEST = "extension_request"
    DISPUTE = "dispute"
    UNCLEAR = "unclear"


@dataclass(slots=True)
class LegacyInvoice:
    """An invoice persisted in the legacy local JSON audit store."""

    invoice_id: str
    client_name: str
    client_tax_id: str
    net_amount: Decimal
    vat_rate: Decimal
    vat_amount: Decimal
    total_amount: Decimal
    currency: str
    issue_date: date
    due_date: date
    status: str = "UNPAID"
    language: Language = Language.EN
    channel: Channel = Channel.EMAIL
    client_email: str | None = None
    client_phone: str | None = None
    client_address: str | None = None
    client_city: str | None = None
    client_country: str = "MK"
    description: str = "Professional services"
    ujp_status: str = "NOT_SUBMITTED"
    ujp_reference: str | None = None
    ujp_xml_path: str | None = None
    reminder_count: int = 0
    last_reminded_at: datetime | None = None
    created_at: datetime = field(default_factory=now_utc)
    updated_at: datetime = field(default_factory=now_utc)

    def __post_init__(self) -> None:
        self.invoice_id = str(self.invoice_id).strip()
        if not self.invoice_id:
            raise ValueError("invoice_id is required.")

        self.client_name = str(self.client_name).strip()
        self.client_tax_id = normalize_tax_id(self.client_tax_id)
        self.currency = (str(self.currency).strip() or "MKD").upper()
        self.status = normalize_status(self.status)
        self.language = (
            self.language if isinstance(self.language, Language) else Language.from_value(self.language)
        )
        self.channel = (
            self.channel if isinstance(self.channel, Channel) else Channel.from_value(self.channel)
        )

        self.net_amount = parse_decimal(self.net_amount)
        self.vat_rate = parse_decimal(self.vat_rate)
        self.vat_amount = parse_decimal(self.vat_amount)
        self.total_amount = parse_decimal(self.total_amount)
        self.description = str(self.description).strip() or "Professional services"
        self.reminder_count = max(0, int(self.reminder_count or 0))

        for field_name in ("client_email", "client_phone", "client_address", "client_city"):
            value = getattr(self, field_name)
            if isinstance(value, str) and not value.strip():
                setattr(self, field_name, None)

        if self.client_phone:
            normalized_phone = normalize_phone(self.client_phone)
            self.client_phone = normalized_phone or None

        if not isinstance(self.created_at, datetime):
            self.created_at = datetime.fromisoformat(str(self.created_at))
        if not isinstance(self.updated_at, datetime):
            self.updated_at = datetime.fromisoformat(str(self.updated_at))
        if self.last_reminded_at is not None and not isinstance(self.last_reminded_at, datetime):
            self.last_reminded_at = datetime.fromisoformat(str(self.last_reminded_at))

    # ------------------------------------------------------------------
    # Derived collection properties
    # ------------------------------------------------------------------

    def days_late(self, as_of: date | None = None) -> int:
        """Return the number of calendar days past the due date."""

        current = as_of or date.today()
        return max(0, (current - self.due_date).days)

    def is_overdue(self, as_of: date | None = None) -> bool:
        """Follow the MVP rule: anything not PAID and past due is collectible."""

        current = as_of or date.today()
        return self.status.strip().upper() != "PAID" and self.due_date < current

    def tone_tier(self, as_of: date | None = None) -> ToneTier:
        return ToneTier.from_days_late(self.days_late(as_of))

    def recipient_for_channel(self) -> str | None:
        """Return the destination associated with the preferred channel."""

        if self.channel == Channel.EMAIL:
            return self.client_email
        return self.client_phone

    def collection_context(self, as_of: date | None = None) -> dict[str, Any]:
        """Build the legacy, serializer-friendly context used by tone agents."""

        current = as_of or date.today()
        return {
            "invoice_id": self.invoice_id,
            "client_name": self.client_name,
            "amount": f"{self.total_amount:.2f}",
            "total_amount": f"{self.total_amount:.2f}",
            "currency": self.currency,
            "days_late": self.days_late(current),
            "preferred_language": self.language.value,
            "language": self.language.value,
            "channel": self.channel.value,
            "email": self.client_email,
            "phone": self.client_phone,
            "due_date": self.due_date.isoformat(),
            "issue_date": self.issue_date.isoformat(),
            "description": self.description,
            "tone_tier": self.tone_tier(current).value,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "invoice_id": self.invoice_id,
            "client_name": self.client_name,
            "client_tax_id": self.client_tax_id,
            "net_amount": f"{self.net_amount:.2f}",
            "vat_rate": f"{self.vat_rate:.2f}",
            "vat_amount": f"{self.vat_amount:.2f}",
            "total_amount": f"{self.total_amount:.2f}",
            "currency": self.currency,
            "issue_date": self.issue_date.isoformat(),
            "due_date": self.due_date.isoformat(),
            "status": self.status,
            "language": self.language.value,
            "channel": self.channel.value,
            "client_email": self.client_email,
            "client_phone": self.client_phone,
            "client_address": self.client_address,
            "client_city": self.client_city,
            "client_country": self.client_country,
            "description": self.description,
            "ujp_status": self.ujp_status,
            "ujp_reference": self.ujp_reference,
            "ujp_xml_path": self.ujp_xml_path,
            "reminder_count": self.reminder_count,
            "last_reminded_at": self.last_reminded_at.isoformat() if self.last_reminded_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LegacyInvoice:
        """Rehydrate an invoice, including legacy Excel-style field names."""

        def pick(*keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in payload and payload[key] not in (None, ""):
                    return payload[key]
            return default

        total_raw = pick("total_amount", "amount", "amount_due", "gross_amount")
        net_raw = pick("net_amount", "subtotal", "net")
        vat_raw = pick("vat_amount", "ddv", "vat")
        rate_raw = pick("vat_rate", "ddv_rate", "vat_percent")

        if total_raw is None and net_raw is None:
            raise ValueError("Invoice requires an amount or total_amount.")

        rate = parse_decimal(rate_raw) if rate_raw is not None else None
        if total_raw is None:
            net = parse_decimal(net_raw)
            if vat_raw is not None:
                vat = parse_decimal(vat_raw)
            elif rate is not None:
                vat = parse_decimal(
                    (net * rate / Decimal(100)).quantize(Decimal("0.01"))
                )
            else:
                vat = Decimal("0.00")
            total = parse_decimal(net + vat)
        else:
            total = parse_decimal(total_raw)
            if net_raw is not None:
                net = parse_decimal(net_raw)
            elif vat_raw is not None:
                net = parse_decimal(total - parse_decimal(vat_raw))
            else:
                net = total

            if vat_raw is not None:
                vat = parse_decimal(vat_raw)
            else:
                vat = parse_decimal(max(Decimal("0.00"), total - net))

        if rate is None:
            rate = parse_decimal(
                (vat / net * Decimal(100)).quantize(Decimal("0.01"))
                if net > 0 and vat > 0
                else Decimal("0.00")
            )

        due_raw = pick("due_date", "due", "payment_due_date")
        if due_raw is None:
            raise ValueError("Invoice requires a due_date.")
        due = parse_date(due_raw)
        issue_raw = pick("issue_date", "date", default=due)

        reminder_raw = pick("reminder_count", default=0)
        last_raw = pick("last_reminded_at")
        created_raw = pick("created_at")
        updated_raw = pick("updated_at")

        def parse_dt(raw: Any) -> datetime | None:
            if raw is None:
                return None
            if isinstance(raw, datetime):
                return raw
            return datetime.fromisoformat(str(raw))

        return cls(
            invoice_id=str(pick("invoice_id", "invoice_number", "id", default="")).strip(),
            client_name=str(pick("client_name", "customer_name", "client", "customer", default="")).strip(),
            client_tax_id=str(
                pick("client_tax_id", "tax_id", "edb", "embg", "tax_number", default="")
            ),
            net_amount=net,
            vat_rate=rate,
            vat_amount=vat,
            total_amount=total,
            currency=str(pick("currency", "currency_code", default="MKD")),
            issue_date=parse_date(issue_raw),
            due_date=due,
            status=str(pick("status", "payment_status", default="UNPAID")),
            language=Language.from_value(pick("language", "preferred_language", default="EN")),
            channel=Channel.from_value(pick("channel", "preferred_channel", default="Email")),
            client_email=pick("client_email", "email"),
            client_phone=pick("client_phone", "phone", "mobile"),
            client_address=pick("client_address", "address"),
            client_city=pick("client_city", "city"),
            client_country=str(pick("client_country", "country", default="MK")),
            description=str(pick("description", "service", "service_description", default="Professional services")),
            ujp_status=str(pick("ujp_status", default="NOT_SUBMITTED")),
            ujp_reference=pick("ujp_reference"),
            ujp_xml_path=pick("ujp_xml_path"),
            reminder_count=int(reminder_raw or 0),
            last_reminded_at=parse_dt(last_raw),
            created_at=parse_dt(created_raw) or now_utc(),
            updated_at=parse_dt(updated_raw) or now_utc(),
        )


# Backward-compatibility alias: legacy consumers import `Invoice` from here.
Invoice = LegacyInvoice


@dataclass(slots=True)
class ParsedInvoice:
    """A partially validated invoice draft extracted from a text/voice prompt."""

    raw_prompt: str
    client_name: str | None = None
    client_tax_id: str | None = None
    amount: Decimal | None = None
    amount_includes_vat: bool = False
    vat_rate: Decimal = Decimal("18.00")
    currency: str = "MKD"
    issue_date: date = field(default_factory=date.today)
    due_date: date | None = None
    language: Language = Language.MK
    channel: Channel = Channel.EMAIL
    client_email: str | None = None
    client_phone: str | None = None
    description: str = "Professional services"
    warnings: list[str] = field(default_factory=list)

    @property
    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.client_name:
            missing.append("client_name")
        if not self.client_tax_id:
            missing.append("client_tax_id")
        if self.amount is None:
            missing.append("amount")
        if self.due_date is None:
            missing.append("due_date")
        return missing

    def breakdown(self) -> tuple[Decimal, Decimal, Decimal]:
        """Return the net/VAT/gross breakdown implied by the prompt."""

        if self.amount is None:
            raise ValueError("Cannot calculate totals without an amount.")
        return calculate_vat_breakdown(
            amount=self.amount,
            vat_rate=self.vat_rate,
            amount_includes_vat=self.amount_includes_vat,
        )

    def to_invoice(self, invoice_id: str, description: str | None = None) -> LegacyInvoice:
        """Build a full Invoice after the human has filled any missing fields."""

        if self.amount is None:
            raise ValueError("Invoice amount is required.")
        if not self.client_name:
            raise ValueError("Client name is required.")
        if not self.client_tax_id:
            raise ValueError("Client tax ID is required.")
        if self.due_date is None:
            raise ValueError("Due date is required.")

        net, vat, total = self.breakdown()
        return LegacyInvoice(
            invoice_id=invoice_id,
            client_name=self.client_name,
            client_tax_id=self.client_tax_id,
            net_amount=net,
            vat_rate=self.vat_rate,
            vat_amount=vat,
            total_amount=total,
            currency=self.currency,
            issue_date=self.issue_date,
            due_date=self.due_date,
            status="UNPAID",
            language=self.language,
            channel=self.channel,
            client_email=self.client_email,
            client_phone=self.client_phone,
            description=description or self.description,
            ujp_status="NOT_SUBMITTED",
        )


@dataclass(slots=True)
class UJPValidationResult:
    """Result of the local mock UJP/UBL validation service."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)


@dataclass(slots=True)
class UJPSubmissionResult:
    """Mock submission outcome returned by ``ujp_service.py``."""

    invoice_id: str
    success: bool
    validation: UJPValidationResult
    xml_path: str | None = None
    reference: str | None = None
    message: str = ""
    mock: bool = True


@dataclass(slots=True)
class ReplyAnalysis:
    """Structured classification of an inbound client reply."""

    category: ReplyCategory
    confidence: float
    matched_terms: list[str]
    suggested_action: str
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "confidence": round(self.confidence, 2),
            "matched_terms": self.matched_terms,
            "suggested_action": self.suggested_action,
            "language": self.language,
        }


@dataclass(slots=True)
class DispatchResult:
    """Mock dispatcher result, shaped like a future provider response."""

    success: bool
    channel: Channel
    recipient: str
    invoice_id: str
    message_id: str
    subject: str | None = None
