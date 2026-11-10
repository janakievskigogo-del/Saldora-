"""Shared parsing and normalization helpers.

This module intentionally has no dependency on the rest of the application so it can
be reused safely by the Excel loader, the natural-language invoice parser, the reply
classifier, and future FastAPI handlers.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core normalization
# ---------------------------------------------------------------------------


def normalize_whitespace(value: str) -> str:
    """Collapse all whitespace runs into single spaces and trim the result."""

    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def normalize_language_code(value: Any, default: str = "EN") -> str:
    """Return one of ``MK``, ``AL``, or ``EN``.

    Both ISO-ish values and commonly used local names are accepted.  An
    unsupported value falls back to ``default``; callers that need strict
    behavior can compare the original value to the result.
    """

    if value is None or (isinstance(value, str) and not value.strip()):
        return default

    normalized = normalize_whitespace(str(value)).casefold()
    aliases = {
        "mk": "MK",
        "mkd": "MK",
        "mac": "MK",
        "macedonian": "MK",
        "македонски": "MK",
        "македонски јазик": "MK",
        "al": "AL",
        "sq": "AL",
        "sqi": "AL",
        "alb": "AL",
        "albanian": "AL",
        "shqip": "AL",
        "shqiptar": "AL",
        "en": "EN",
        "eng": "EN",
        "english": "EN",
        "англиски": "EN",
    }
    return aliases.get(normalized, default)


def normalize_channel(value: Any, default: str = "Email") -> str:
    """Return one of ``Email``, ``Viber``, or ``WhatsApp``."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return default

    normalized = (
        normalize_whitespace(str(value))
        .casefold()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    aliases = {
        "email": "Email",
        "emailmessage": "Email",
        "mail": "Email",
        "viber": "Viber",
        "vibber": "Viber",
        "whatsapp": "WhatsApp",
        "whatsap": "WhatsApp",
        "wa": "WhatsApp",
        "sms": "WhatsApp",
    }
    return aliases.get(normalized, default)


def normalize_status(value: Any, default: str = "UNPAID") -> str:
    """Normalize payment states while preserving unknown status labels."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return default

    normalized = normalize_whitespace(str(value)).casefold()
    aliases = {
        "paid": "PAID",
        "платено": "PAID",
        "paguar": "PAID",
        "unpaid": "UNPAID",
        "неплатено": "UNPAID",
        "не е платено": "UNPAID",
        "papaguar": "UNPAID",
        "pending": "PENDING",
        "во очекување": "PENDING",
        "në pritje": "PENDING",
        "ne pritje": "PENDING",
        "draft": "DRAFT",
        "cancelled": "CANCELLED",
        "canceled": "CANCELLED",
        "откажано": "CANCELLED",
        "anuluar": "CANCELLED",
        "overdue": "OVERDUE",
        "задоцнето": "OVERDUE",
        "reminder_sent": "REMINDER_SENT",
        "reminder sent": "REMINDER_SENT",
        "remindersent": "REMINDER_SENT",
        "потсетник испратен": "REMINDER_SENT",
        "kujtesë e dërguar": "REMINDER_SENT",
    }
    return aliases.get(normalized, normalize_whitespace(str(value)).upper().replace(" ", "_"))


# ---------------------------------------------------------------------------
# Money parsing
# ---------------------------------------------------------------------------


def parse_decimal(value: Any, decimal_places: int | None = 2) -> Decimal:
    """Parse a human/Excel money value into :class:`~decimal.Decimal`.

    Supports both common European (``1.234,56``) and US (``1,234.56``)
    formatting.  Currency names/symbols are stripped.  Ambiguous single
    separators followed by three digits are treated as thousands separators,
    which is the common convention in invoices for this region.
    """

    if isinstance(value, bool) or value is None:
        raise ValueError("A numeric value is required.")

    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, int):
        parsed = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Amount must be finite.")
        parsed = Decimal(str(value))
    elif isinstance(value, str):
        raw = unicodedata.normalize("NFKC", value).strip()
        raw = raw.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
        if not raw:
            raise ValueError("A numeric value is required.")

        # Keep only digits, signs and decimal separators.  This removes MKD,
        # EUR, ден., currency symbols, and stray words without losing the value.
        raw = re.sub(r"[^0-9,.\-+]", "", raw)
        if raw in {"", "-", "+", ".", ","}:
            raise ValueError(f"Could not parse amount from {value!r}.")

        sign = ""
        if raw[0] in "+-":
            sign, raw = raw[0], raw[1:]
        if not raw:
            raise ValueError(f"Could not parse amount from {value!r}.")

        has_comma = "," in raw
        has_dot = "." in raw

        if has_comma and has_dot:
            # The right-most separator is the decimal separator.
            if raw.rfind(",") > raw.rfind("."):
                raw = raw.replace(".", "").replace(",", ".")
            else:
                raw = raw.replace(",", "")
        elif has_comma or has_dot:
            separator = "," if has_comma else "."
            parts = raw.split(separator)
            if len(parts) > 2:
                # Multiple groups -> almost certainly thousands separators.
                raw = "".join(parts)
            else:
                left, right = parts
                if not right:
                    raw = left
                elif len(right) in {1, 2}:
                    raw = f"{left}.{right}"
                elif len(right) == 3 and left not in {"", "0"}:
                    # 15.000 / 15,000 -> fifteen thousand.
                    raw = left + right
                else:
                    raw = f"{left}.{right}"

        parsed = Decimal(sign + raw)
    else:
        parsed = Decimal(str(value))

    if decimal_places is not None:
        quantum = Decimal(1).scaleb(-decimal_places)
        parsed = parsed.quantize(quantum, rounding=ROUND_HALF_UP)
    return parsed


def format_money(amount: Decimal | float | str, currency: str | None = None) -> str:
    """Format a monetary value for console output or LLM context."""

    parsed = parse_decimal(amount)
    formatted = f"{parsed:,.2f}"
    return f"{formatted} {currency}".strip() if currency else formatted


def calculate_vat_breakdown(
    amount: Decimal,
    vat_rate: Decimal,
    amount_includes_vat: bool = False,
) -> tuple[Decimal, Decimal, Decimal]:
    """Return ``(net, vat, total)`` using two-decimal invoice arithmetic.

    The resulting values always satisfy ``net + vat == total`` after rounding.
    """

    amount = parse_decimal(amount)
    vat_rate = parse_decimal(vat_rate)

    if vat_rate < 0 or vat_rate > 100:
        raise ValueError("VAT rate must be between 0 and 100 percent.")

    if amount_includes_vat:
        total = amount
        net = (total / (Decimal(1) + (vat_rate / Decimal(100)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        vat = (total - net).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    else:
        net = amount
        vat = (net * vat_rate / Decimal(100)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        total = (net + vat).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return net, vat, total


# ---------------------------------------------------------------------------
# Dates, identifiers, and contact fields
# ---------------------------------------------------------------------------

_EXCEL_EPOCH = datetime(1899, 12, 30)
_DATE_FORMATS = (
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%y",
    "%d/%m/%y",
    "%Y.%m.%d",
)


def parse_date(value: Any) -> date:
    """Parse common Excel/JSON/local date values into :class:`datetime.date`."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    # ``pandas.Timestamp`` is a datetime subclass, so the check above covers it.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if 20_000 <= float(value) <= 80_000:
            return (_EXCEL_EPOCH + timedelta(days=float(value))).date()
        raise ValueError(f"Unsupported numeric date value: {value!r}")

    if value is None:
        raise ValueError("A due date is required.")

    text = normalize_whitespace(str(value))
    if not text:
        raise ValueError("A due date is required.")

    # Excel sometimes exports a serialized date as a numeric string.
    if re.fullmatch(r"\d{5}(?:\.0+)?", text):
        serial = int(float(text))
        if 20_000 <= serial <= 80_000:
            return (_EXCEL_EPOCH + timedelta(days=serial)).date()

    # Support ISO timestamps such as 2026-10-30T10:15:00.
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    raise ValueError(f"Unrecognized date format: {value!r}")


def normalize_tax_id(value: Any) -> str:
    """Clean a tax identifier while preserving an optional MK prefix."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).upper()
    text = re.sub(r"[\s\-–—./]", "", text)
    return text.strip()


def is_valid_tax_id(value: Any) -> bool:
    """Lenient structural check for ЕДБ / ЕМБГ / NIPT values.

    This is deliberately not a legal validation service.  It catches obvious
    input errors while allowing the varying lengths used in the region.
    """

    text = normalize_tax_id(value)
    return bool(re.fullmatch(r"(?:MK)?[0-9A-Z]{8,14}", text)) and any(
        character.isdigit() for character in text
    )


def normalize_phone(value: Any) -> str:
    """Normalize a phone number for a mock dispatcher or future provider."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"[^\d+]", "", text)
    if text.startswith("00"):
        text = "+" + text[2:]
    if text.count("+") > 1 or ("+" in text and not text.startswith("+")):
        return ""
    return text


def is_valid_email(value: Any) -> bool:
    """Provide a conservative structural email check for pre-send validation."""

    if not value:
        return False
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", str(value).strip()))


def safe_filename(value: str, fallback: str = "document") -> str:
    """Convert an identifier into a safe cross-platform filename stem."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return cleaned or fallback


def now_utc() -> datetime:
    """Return an aware UTC timestamp for audit fields."""

    return datetime.now(timezone.utc)


def strip_code_fences(text: str) -> str:
    """Remove accidental Markdown code fences returned by an LLM."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_response_text(response: Any) -> str:
    """Best-effort text extraction from an LLM response.

    Supports OpenAI-compatible chat completions (``choices[].message.content``,
    used by Groq) as well as the legacy google-genai shapes.  Provider
    properties can raise when a response is blocked or empty, so every access
    is defensive.
    """

    try:
        text = response.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not read response.text directly: %s", exc)

    try:
        for choice in list(getattr(response, "choices", None) or []):
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                content_parts: list[str] = []
                for part in content:
                    part_text = (
                        part.get("text")
                        if isinstance(part, dict)
                        else getattr(part, "text", None)
                    )
                    if isinstance(part_text, str) and part_text.strip():
                        content_parts.append(part_text.strip())
                if content_parts:
                    return "\n".join(content_parts)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not extract text from response choices: %s", exc)

    try:
        parts: list[str] = []
        for candidate in list(getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in list(getattr(content, "parts", None) or []):
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    parts.append(part_text)
        return "\n".join(parts).strip()
    except (AttributeError, TypeError, ValueError) as exc:
        logger.debug("Could not extract text from response candidates: %s", exc)
        return ""
