"""FakturaBot.mk intake parser.

Turns a short natural-language instruction (typed or transcribed from voice) into
a structured invoice draft.  The parser is deliberately conservative: it extracts
what it can prove from the text, records warnings, and leaves missing values for the
human-in-the-loop CLI to complete.  This avoids silently inventing financial data.

A second, AI-driven path (Step 3) lives here as well: ``parse_invoice_with_retry``
sends raw invoice documents to Groq with a JSON response contract and feeds
validation errors back into the prompt until the output passes the strict
``GeminiParsedInvoice`` arithmetic/tax-ID validation or the retry budget is spent.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from config import GROQ_BASE_URL, Settings
from legacy_models import Channel, Language, ParsedInvoice
from logger import logger
from schemas import GeminiParsedInvoice, InvoiceMathError, TaxIDValidationError
from utils import (
    extract_response_text,
    is_valid_tax_id,
    normalize_phone,
    normalize_tax_id,
    normalize_whitespace,
    parse_date,
    parse_decimal,
)

# ---------------------------------------------------------------------------
# Regex building blocks
# ---------------------------------------------------------------------------

_AMOUNT_TOKEN = (
    r"(?:\d{1,3}(?:[.,\s\u00a0]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
)
_CURRENCY_TOKEN = (
    r"(?:MKD|MKD\.?|EUR|EURO|€|ALL|LEK[ËE]?|RSD|DIN(?:ARI)?|USD|\$|"
    r"ДЕН(?:АРИ)?|ДИНАРИ)"
)
_DATE_TOKEN = r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})"

_AMOUNT_LABEL_RE = re.compile(
    rf"(?P<label>вкупн[ао]|вкупно|износ(?:от)?|сума(?:та)?|вредност(?:та)?|цена(?:та)?|"
    rf"amount|total(?:\s+amount)?|value|price|sum|shuma|vlera|çmimi)\s*"
    rf"(?:(?:е|is|изнесува|of|на|од)\s*)?[:= -]*\s*"
    rf"(?P<amount>{_AMOUNT_TOKEN})(?:\s*(?P<currency>{_CURRENCY_TOKEN}))?",
    re.IGNORECASE | re.UNICODE,
)

_CURRENCY_AMOUNT_RE = re.compile(
    rf"(?P<amount>{_AMOUNT_TOKEN})\s*(?P<currency>{_CURRENCY_TOKEN})",
    re.IGNORECASE | re.UNICODE,
)

_CURRENCY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:€|\bEUR\b|\bEURO\b)", re.IGNORECASE), "EUR"),
    (re.compile(r"(?:\bMKD\b|\bДЕН(?:АРИ)?\b|\bDEN(?:ARI)?\b)", re.IGNORECASE), "MKD"),
    (re.compile(r"(?:\bALL\b|\bLEK[ËE]?\b)", re.IGNORECASE), "ALL"),
    (re.compile(r"(?:\bRSD\b|\bDIN(?:ARI)?\b|\bДИНАРИ\b)", re.IGNORECASE), "RSD"),
    (re.compile(r"(?:\bUSD\b|\$)", re.IGNORECASE), "USD"),
)

_TAX_ID_RE = re.compile(
    r"(?:е\.?д\.?б\.?|едб|е\.?м\.?б\.?г\.?|ембг|дан(?:очен)?\s*(?:број|br\.?)|"
    r"edb|embg|tax\s*(?:id|number)|vat\s*(?:id|number)|nipt|nui)\s*"
    r"(?:број|br\.?|number)?\s*[:=#\-]?\s*"
    r"(?P<tax>(?:[A-Z]{1,3}[\s-]*)?[0-9][0-9A-Z-]{6,15})",
    re.IGNORECASE | re.UNICODE,
)

_CLIENT_LABELED_RE = re.compile(
    r"(?:клиент(?:от|ин|и)?|komitent(?:i)?|klient(?:ët|et|in|it|i)?|комитент(?:от)?|"
    r"купувач(?:от)?|customer|client|bill\s+to)\s*"
    r"(?:(?:е|is)\s*)?[:= -]*\s*"
    r"(?P<name>[^\s,.;:][^,;:\n]{1,79}?)"
    r"(?=\s+(?:со|with|и|and|за|for|për|shërbim(?:i)?|едб|ембг|дан|tax|износ|amount|"
    r"сума|вкупно|total|vlera|рок|due|afat|фактура|invoice|fatur[ëe]|тел|phone|email|"
    r"viber|whatsapp)\b|[,;:\n]|\.(?:\s|$)|$)",
    re.IGNORECASE | re.UNICODE,
)

_INVOICE_FOR_RE = re.compile(
    r"(?:фактура|invoice|fatur[ëe])\s+(?:за|for|për)\s+"
    r"(?P<name>[^\s,.;:][^,;:\n]{1,79}?)"
    r"(?=\s+(?:со|with|и|and|за|for|për|shërbim(?:i)?|едб|ембг|дан|tax|износ|amount|"
    r"сума|вкупно|total|vlera|рок|due|afat|тел|phone|email|viber|whatsapp)\b|[,;:\n]|\.(?:\s|$)|$)",
    re.IGNORECASE | re.UNICODE,
)

_CLIENT_FOR_COMPANY_RE = re.compile(
    r"(?:за\s+фирма|for\s+company|për\s+kompaninë)\s*[:=]?\s*"
    r"(?P<name>[^\s,.;:][^,;:\n]{1,79}?)"
    r"(?=\s+(?:со|with|и|and|за|for|për|едб|ембг|дан|tax|износ|amount|сума|вкупно|"
    r"total|vlera|рок|due|afat|тел|phone|email|viber|whatsapp)\b|[,;:\n]|\.(?:\s|$)|$)",
    re.IGNORECASE | re.UNICODE,
)

_DESCRIPTION_RE = re.compile(
    r"(?:за\s+услуг[аи]|услуга(?:та)?|for\s+(?:services?|work)|për\s+shërbim(?:in)?|"
    r"shërbim(?:i)?|опис|description)\s*"
    r"(?:на|of|:|=|-)?\s*"
    r"(?P<description>[^,.;\n]{3,120}?)"
    r"(?=\s+(?:во\s+износ|amount|сума|вкупно|total|со\s+ддв|без\s+ддв|with\s+vat|"
    r"without\s+vat|рок|due|afat|едб|ембг|tax|фактура|invoice)\b|[,.;\n]|$)",
    re.IGNORECASE | re.UNICODE,
)

_ISSUE_DATE_RE = re.compile(
    rf"(?:датум\s+на\s+издавање|issue\s+date|date\s+of\s+issue|data\s+e\s+lëshimit)"
    rf"\s*[:=]?\s*(?P<date>{_DATE_TOKEN})",
    re.IGNORECASE | re.UNICODE,
)

_DUE_DATE_RE = re.compile(
    rf"(?:рок(?:\s+до|\s+за\s+плаќање)?|доспева(?:ње)?(?:\s+на)?|due(?:\s+date)?|"
    rf"payment\s+due|плати\s+до|deri\s+(?:më|me)?|afat(?:i)?(?:\s+deri)?)\s*"
    rf"[:=]?\s*(?P<date>{_DATE_TOKEN})",
    re.IGNORECASE | re.UNICODE,
)

_RELATIVE_DUE_RE = re.compile(
    r"(?:рок(?:\s+од)?|payment\s+term(?:s)?|net|due\s+in|in|afat(?:i)?|brenda)\s*"
    r"(?P<days>\d{1,3})\s*(?:days?|дена|дени|dit[ëe](?:sh)?|dane|dan)\b",
    re.IGNORECASE | re.UNICODE,
)

_GENERIC_DATE_RE = re.compile(_DATE_TOKEN, re.UNICODE)

_VAT_RATE_RE = re.compile(
    r"(?:ддв|ddv|vat|tvsh)\D{0,22}?(?P<rate>\d{1,2}(?:[.,]\d+)?)\s*%",
    re.IGNORECASE | re.UNICODE,
)
_VAT_RATE_REVERSE_RE = re.compile(
    r"(?P<rate>\d{1,2}(?:[.,]\d+)?)\s*%\s*(?:ддв|ddv|vat|tvsh)",
    re.IGNORECASE | re.UNICODE,
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"(?:тел\.?|телефон|phone|mobile|mob\.?|mobil|contact)\s*[:=]?\s*"
    r"(?P<phone>\+?[0-9][0-9\s\-()/]{6,18}[0-9])",
    re.IGNORECASE | re.UNICODE,
)
_GENERIC_MK_PHONE_RE = re.compile(r"(?P<phone>\+389[\d\s\-()/]{6,15})", re.IGNORECASE)

_GROSS_VAT_PATTERNS = (
    re.compile(r"\bсо\s+(?:вклучен(?:о)?\s+)?ддв\b", re.IGNORECASE),
    re.compile(r"\bвклучен(?:о)?\s+ддв\b", re.IGNORECASE),
    re.compile(r"\bддв\s+вклучен(?:о)?\b", re.IGNORECASE),
    re.compile(r"\bwith\s+(?:included\s+)?vat\b", re.IGNORECASE),
    re.compile(r"\bvat\s+included\b", re.IGNORECASE),
    re.compile(r"\bincluding\s+vat\b", re.IGNORECASE),
    re.compile(r"\bme\s+tvsh\b", re.IGNORECASE),
    re.compile(r"\btvsh\s+e\s+përfshirë\b", re.IGNORECASE),
    re.compile(r"\bgross\b", re.IGNORECASE),
)

_NET_VAT_PATTERNS = (
    re.compile(r"\bбез\s+ддв\b", re.IGNORECASE),
    re.compile(r"\bпред\s+ддв\b", re.IGNORECASE),
    re.compile(r"\bплус\s+ддв\b", re.IGNORECASE),
    re.compile(r"\+\s*ддв\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+vat\b", re.IGNORECASE),
    re.compile(r"\bexcluding\s+vat\b", re.IGNORECASE),
    re.compile(r"\bnet\s+of\s+vat\b", re.IGNORECASE),
    re.compile(r"\bplus\s+vat\b", re.IGNORECASE),
    re.compile(r"\bpa\s+tvsh\b", re.IGNORECASE),
    re.compile(r"\bplus\s+tvsh\b", re.IGNORECASE),
)

_VAT_EXEMPT_PATTERNS = (
    re.compile(r"\bослободен(?:а|о)?\s+од\s+ддв\b", re.IGNORECASE),
    re.compile(r"\bне\s+е\s+обврзник\s+за\s+ддв\b", re.IGNORECASE),
    re.compile(r"\bvat\s+exempt\b", re.IGNORECASE),
    re.compile(r"\bnot\s+vat\s+registered\b", re.IGNORECASE),
    re.compile(r"\bpërjashtuar\s+nga\s+tvsh\b", re.IGNORECASE),
)

_LANGUAGE_EXPLICIT_PATTERNS: tuple[tuple[re.Pattern[str], Language], ...] = (
    (re.compile(r"\b(?:in|на|në)\s+english\b|\bанглиски\b|\banglezisht\b", re.IGNORECASE), Language.EN),
    (re.compile(r"\b(?:in|на|në)\s+albanian\b|\bалбански\b|\bshqip\b", re.IGNORECASE), Language.AL),
    (re.compile(r"\b(?:in|на|në)\s+macedonian\b|\bмакедонски\b|\bmaqedonisht\b", re.IGNORECASE), Language.MK),
)

_LANGUAGE_KEYWORDS = {
    Language.MK: (
        "фактура",
        "клиент",
        "комитент",
        "износ",
        "сума",
        "вкупно",
        "рок",
        "ддв",
        "услуга",
        "денари",
        "ден",
        "плаќање",
        "датум",
        "издај",
        "направи",
        "без",
        "со",
        "за",
        "на",
        "и",
    ),
    Language.AL: (
        "faturë",
        "fatura",
        "klient",
        "klienti",
        "klientin",
        "shërbim",
        "shërbimi",
        "vlera",
        "shuma",
        "tvsh",
        "afat",
        "afati",
        "ditë",
        "deri",
        "pagesë",
        "paguaj",
        "me",
        "për",
        "dhe",
        "ju",
        "lutem",
        "data",
        "lëshimit",
    ),
    Language.EN: (
        "invoice",
        "client",
        "customer",
        "service",
        "amount",
        "total",
        "due",
        "vat",
        "payment",
        "days",
        "please",
        "for",
        "issue",
        "tax",
        "including",
    ),
}

_INVALID_CLIENT_NAMES = {
    "клиент",
    "клиентот",
    "komitent",
    "klient",
    "klienti",
    "klientin",
    "комитент",
    "купувач",
    "client",
    "customer",
    "услуга",
    "услуги",
    "service",
    "services",
    "shërbim",
    "shërbimi",
    "фактура",
    "invoice",
    "fatura",
}


class InvoiceParsingError(ValueError):
    """Raised when the prompt cannot be parsed at all."""


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def _find_amount(
    text: str,
) -> tuple[Decimal | None, str | None, bool | None, tuple[int, int] | None]:
    """Find the most credible amount and return the currency/inclusion hints.

    A number is only accepted when it is attached to an amount label or a
    currency.  This deliberately avoids treating dates, tax IDs, or VAT rates as
    invoice totals.
    """

    candidates: list[tuple[Decimal, str | None, bool | None, tuple[int, int]]] = []

    for match in _AMOUNT_LABEL_RE.finditer(text):
        try:
            value = parse_decimal(match.group("amount"))
        except ValueError:
            continue
        label = (match.group("label") or "").casefold()
        label_suggests_gross: bool | None = None
        if any(token in label for token in ("вкупно", "total", "gross")):
            label_suggests_gross = True
        elif any(token in label for token in ("net", "без ддв", "without vat")):
            label_suggests_gross = False
        candidates.append(
            (
                value,
                (match.group("currency") or "").strip() or None,
                label_suggests_gross,
                match.span("amount"),
            )
        )

    for match in _CURRENCY_AMOUNT_RE.finditer(text):
        try:
            value = parse_decimal(match.group("amount"))
        except ValueError:
            continue
        candidates.append(
            (
                value,
                (match.group("currency") or "").strip() or None,
                None,
                match.span("amount"),
            )
        )

    if not candidates:
        return None, None, None, None

    # Prefer the first labeled amount, otherwise the first currency-tagged amount.
    value, currency, gross_hint, span = candidates[0]
    return value, currency, gross_hint, span


def _detect_currency(text: str, default: str) -> tuple[str, bool]:
    matches: list[tuple[int, str]] = []
    for pattern, code in _CURRENCY_PATTERNS:
        match = pattern.search(text)
        if match:
            matches.append((match.start(), code))
    if not matches:
        return default, False
    matches.sort(key=lambda item: item[0])
    return matches[0][1], True


def _detect_vat_rate(text: str, default: Decimal) -> tuple[Decimal, bool]:
    for pattern in (_VAT_RATE_RE, _VAT_RATE_REVERSE_RE):
        match = pattern.search(text)
        if match:
            try:
                rate = parse_decimal(match.group("rate"))
                if Decimal(0) <= rate <= Decimal(100):
                    return rate, True
            except ValueError:
                continue
    return default, False


def _nearest_vat_signal(text: str, amount_span: tuple[int, int]) -> bool | None:
    """Decide whether the extracted amount includes VAT based on proximity."""

    amount_center = (amount_span[0] + amount_span[1]) / 2
    signals: list[tuple[float, bool]] = []

    for pattern in _GROSS_VAT_PATTERNS:
        for match in pattern.finditer(text):
            center = (match.start() + match.end()) / 2
            signals.append((abs(center - amount_center), True))
    for pattern in _NET_VAT_PATTERNS:
        for match in pattern.finditer(text):
            center = (match.start() + match.end()) / 2
            signals.append((abs(center - amount_center), False))

    if not signals:
        return None
    signals.sort(key=lambda item: item[0])
    return signals[0][1]


def _extract_tax_id(text: str) -> str | None:
    match = _TAX_ID_RE.search(text)
    if not match:
        return None
    tax_id = normalize_tax_id(match.group("tax"))
    return tax_id or None


def _clean_client_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip().strip("«»\"'“”„ ")
    name = re.sub(r"\s+(?:со|with|и|and)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+", " ", name).strip(" -:,")
    if len(name) < 2 or name.casefold() in _INVALID_CLIENT_NAMES:
        return None
    return name


def _extract_client_name(text: str, tax_match_span: tuple[int, int] | None = None) -> str | None:
    searchable = text
    if tax_match_span:
        # Remove the tax identifier from the candidate window.  The label is kept
        # as a stop boundary, but the digits must never become a company name.
        start, end = tax_match_span
        searchable = text[:start] + " " + text[end:]

    for pattern in (_CLIENT_LABELED_RE, _INVOICE_FOR_RE, _CLIENT_FOR_COMPANY_RE):
        for match in pattern.finditer(searchable):
            candidate = _clean_client_name(match.group("name"))
            if candidate:
                return candidate
    return None


def _extract_description(text: str, default: str) -> str:
    match = _DESCRIPTION_RE.search(text)
    if not match:
        return default
    description = normalize_whitespace(match.group("description")).strip(" -:,")
    if len(description) < 3:
        return default
    return description[:120]


def _extract_due_date(
    text: str,
    issue_date: date,
    default_terms_days: int,
    warnings: list[str],
    issue_date_span: tuple[int, int] | None = None,
) -> tuple[date, bool]:
    explicit = _DUE_DATE_RE.search(text)
    if explicit:
        try:
            return parse_date(explicit.group("date")), True
        except ValueError:
            warnings.append("The stated due date could not be parsed; using the configured payment term.")

    relative = _RELATIVE_DUE_RE.search(text)
    if relative:
        days = int(relative.group("days"))
        if 0 < days <= 3650:
            return issue_date + timedelta(days=days), True

    # If the prompt has exactly one date that is not the explicit issue date,
    # it is usually the intended payment date.  Never turn the issue date into
    # the due date just because it is the only date in the prompt.
    all_dates = [
        match
        for match in _GENERIC_DATE_RE.finditer(text)
        if not (
            issue_date_span
            and match.start() < issue_date_span[1]
            and match.end() > issue_date_span[0]
        )
    ]
    if len(all_dates) == 1:
        try:
            return parse_date(all_dates[0].group(0)), True
        except ValueError:
            pass

    warnings.append(
        f"No due date was found; using the configured payment term of {default_terms_days} days."
    )
    return issue_date + timedelta(days=default_terms_days), False


def _detect_language(text: str, default: Language) -> Language:
    for pattern, language in _LANGUAGE_EXPLICIT_PATTERNS:
        if pattern.search(text):
            return language

    lowered = text.casefold()

    def score(language: Language) -> int:
        return sum(
            1
            for keyword in _LANGUAGE_KEYWORDS[language]
            if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lowered)
        )

    scores = {language: score(language) for language in Language}
    best_score = max(scores.values())
    if best_score == 0:
        return default

    tied = [language for language, value in scores.items() if value == best_score]
    if len(tied) > 1:
        return default
    return tied[0]


def _detect_channel(text: str, default: Channel) -> Channel:
    lowered = text.casefold()
    if "whatsapp" in lowered or "whats app" in lowered:
        return Channel.WHATSAPP
    if "viber" in lowered:
        return Channel.VIBER
    if re.search(r"\b(?:email|e-mail|mail|пошта)\b", lowered):
        return Channel.EMAIL
    return default


def _extract_email(text: str) -> str | None:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def _extract_phone(text: str) -> str | None:
    match = _PHONE_RE.search(text) or _GENERIC_MK_PHONE_RE.search(text)
    if not match:
        return None
    phone = normalize_phone(match.group("phone"))
    return phone or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_invoice_prompt(prompt: str, settings: Settings | None = None) -> ParsedInvoice:
    """Extract a structured invoice draft from text or a voice transcript.

    Parameters
    ----------
    prompt:
        The transcription/content of the business owner's request.
    settings:
        Injected configuration.  When omitted, values are loaded from ``.env``.

    Returns
    -------
    ParsedInvoice
        The parser never invents missing financial data.  Any missing fields are
        exposed through ``ParsedInvoice.missing_fields`` for HITL completion.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise InvoiceParsingError("The invoice prompt is empty.")

    settings = settings or Settings.from_env()
    text = normalize_whitespace(prompt)
    warnings: list[str] = []

    # Issue date is normally today.  Allow an explicit issue date in the prompt.
    issue_date = settings.today()
    issue_match = _ISSUE_DATE_RE.search(text)
    if issue_match:
        try:
            issue_date = parse_date(issue_match.group("date"))
        except ValueError:
            warnings.append("Issue date could not be parsed; using today's date.")

    tax_match = _TAX_ID_RE.search(text)
    tax_id = _extract_tax_id(text)
    if tax_match and not is_valid_tax_id(tax_id):
        warnings.append("The tax ID was found but does not look structurally valid; please verify it.")

    client_name = _extract_client_name(
        text,
        tax_match_span=tax_match.span() if tax_match else None,
    )

    amount, amount_currency, gross_hint, amount_span = _find_amount(text)
    currency, explicit_currency = _detect_currency(text, settings.default_currency)
    if amount_currency:
        currency = _normalize_currency(amount_currency)
        explicit_currency = True
    elif amount is not None and not explicit_currency:
        warnings.append(f"No currency was specified; using {currency}.")

    vat_exempt = any(pattern.search(text) for pattern in _VAT_EXEMPT_PATTERNS)
    if vat_exempt:
        vat_rate = Decimal("0.00")
        explicit_vat = True
        warnings.append("VAT exemption detected; VAT rate set to 0%.")
    else:
        vat_rate, explicit_vat = _detect_vat_rate(text, settings.default_vat_rate)
        if amount is not None and not explicit_vat:
            warnings.append(
                f"No VAT rate was specified; using the configured {vat_rate:.2f}% rate."
            )

    if amount is not None:
        inferred_gross = (
            _nearest_vat_signal(text, amount_span) if amount_span is not None else None
        )
        if gross_hint is not None:
            amount_includes_vat = gross_hint
        elif inferred_gross is not None:
            amount_includes_vat = inferred_gross
        else:
            amount_includes_vat = settings.default_amount_includes_vat
            if vat_rate > 0:
                assumed = "includes" if amount_includes_vat else "excludes"
                warnings.append(
                    f"VAT treatment was not explicit; assumed the amount {assumed} VAT."
                )
        if amount <= 0:
            warnings.append("The extracted amount is not positive; please correct it.")
    else:
        amount_includes_vat = settings.default_amount_includes_vat
        if amount is None:
            warnings.append("No invoice amount was found; please provide it.")

    due_date, explicit_due = _extract_due_date(
        text,
        issue_date=issue_date,
        default_terms_days=settings.default_payment_terms_days,
        warnings=warnings,
        issue_date_span=issue_match.span("date") if issue_match else None,
    )
    if explicit_due and due_date < issue_date:
        warnings.append("The due date is before the issue date; please verify it.")

    description = _extract_description(text, settings.default_service_description)
    language = _detect_language(text, settings.default_language)
    channel = _detect_channel(text, settings.default_channel)
    email = _extract_email(text)
    phone = _extract_phone(text)

    if not client_name:
        warnings.append("Client name was not found; please provide it.")
    if not tax_id:
        warnings.append("Client tax ID (ЕДБ/ЕМБГ) was not found; please provide it.")

    # De-duplicate warnings while preserving order.
    warnings = list(dict.fromkeys(warnings))

    return ParsedInvoice(
        raw_prompt=prompt,
        client_name=client_name,
        client_tax_id=tax_id,
        amount=amount,
        amount_includes_vat=amount_includes_vat,
        vat_rate=vat_rate,
        currency=currency,
        issue_date=issue_date,
        due_date=due_date,
        language=language,
        channel=channel,
        client_email=email,
        client_phone=phone,
        description=description,
        warnings=warnings,
    )


def _normalize_currency(value: str) -> str:
    text = value.strip().upper()
    if "€" in text or "EUR" in text or "EURO" in text:
        return "EUR"
    if "ДЕН" in text or "DEN" in text or "MKD" in text:
        return "MKD"
    if "LEK" in text or text == "ALL":
        return "ALL"
    if "ДИН" in text or "DIN" in text or "RSD" in text:
        return "RSD"
    if "$" in text or "USD" in text:
        return "USD"
    return text or "MKD"


def parse_voice_prompt(audio_path: str, settings: Settings | None = None) -> ParsedInvoice:
    """Transcribe a voice note with Groq Whisper, then parse the transcript.

    Imported lazily so the text-only path never requires an API key.
    """

    from voice_transcriber import transcribe_audio

    transcript = transcribe_audio(audio_path, settings=settings)
    return parse_invoice_prompt(transcript, settings=settings)


def parsing_summary(parsed: ParsedInvoice) -> dict[str, Any]:
    """Return a JSON-friendly summary for logs or future API responses."""

    try:
        net, vat, total = parsed.breakdown() if parsed.amount is not None else (None, None, None)
    except ValueError:
        net, vat, total = None, None, None

    return {
        "client_name": parsed.client_name,
        "client_tax_id": parsed.client_tax_id,
        "amount_given": str(parsed.amount) if parsed.amount is not None else None,
        "amount_includes_vat": parsed.amount_includes_vat,
        "vat_rate": str(parsed.vat_rate),
        "net_amount": str(net) if net is not None else None,
        "vat_amount": str(vat) if vat is not None else None,
        "total_amount": str(total) if total is not None else None,
        "currency": parsed.currency,
        "issue_date": parsed.issue_date.isoformat(),
        "due_date": parsed.due_date.isoformat() if parsed.due_date else None,
        "language": parsed.language.value,
        "channel": parsed.channel.value,
        "email": parsed.client_email,
        "phone": parsed.client_phone,
        "description": parsed.description,
        "missing_fields": parsed.missing_fields,
        "warnings": parsed.warnings,
    }


# ---------------------------------------------------------------------------
# Groq AI self-correcting invoice extraction (Step 3)
# ---------------------------------------------------------------------------

# Spec contract: the model entry point for structured invoice extraction.  The
# GROQ_MODEL environment variable (used by config.py) may override it.
GROQ_PARSE_MODEL = "llama-3.3-70b-versatile"

_INVOICE_EXTRACTION_PROMPT = (
    "You are a careful accounting clerk for a Macedonian e-invoicing platform.\n"
    "Extract the following fields from the invoice document below and return ONLY "
    "a single JSON object with exactly these keys:\n"
    '  {"client_name": string, "tax_id": string of exactly 13 digits, '
    '"net_amount": decimal string, "vat_rate": percentage as decimal string, '
    '"vat_amount": decimal string, "total_amount": decimal string}\n'
    "Rules: money values are strings with two decimals (e.g. \"15000.00\"); "
    "vat_amount must equal round(net_amount * vat_rate / 100, 2); "
    "tax_id is the Macedonian ЕДБ/ЕМБГ exactly 13 digits, digits only. "
    "Never invent values that are not present in the document."
)

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)


def clean_json_response(text: str) -> str:
    """Strip Markdown code fences and surrounding whitespace from an LLM reply."""

    cleaned = (text or "").strip()
    match = _CODE_FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1)
    return cleaned.strip()


def _get_groq_client():
    """Create the OpenAI-compatible Groq client from ``GROQ_API_KEY``."""

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set; provide it in the environment or .env file."
        )
    try:
        from openai import OpenAI  # lazy: the regex-only path needs no SDK
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for AI invoice extraction."
        ) from exc
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _request_invoice_json(prompt: str) -> str:
    """Call Groq's chat API and return the raw JSON completion text."""

    client = _get_groq_client()
    model = os.environ.get("GROQ_MODEL", GROQ_PARSE_MODEL).strip() or GROQ_PARSE_MODEL
    messages = [
        {
            "role": "system",
            "content": (
                "You extract structured invoice data and return only one valid "
                "JSON object. Never wrap the JSON in Markdown or add commentary."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except TypeError:
        # Older OpenAI-compatible builds may not expose ``response_format``.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
    response_text = extract_response_text(response)
    if not response_text.strip():
        raise ValueError("Groq returned an empty response.")
    return response_text


def _build_parse_prompt(raw_text: str, error_feedback: str | None) -> str:
    """Assemble the extraction prompt, always including the original text."""

    prompt = (
        f"{_INVOICE_EXTRACTION_PROMPT}\n\n"
        "Invoice document:\n"
        f"\"\"\"\n{raw_text}\n\"\"\""
    )
    if error_feedback:
        prompt += (
            "\n\nCRITICAL FIX REQUIRED: Previous attempt failed validation with error: "
            + error_feedback
            + "\nReturn ONLY corrected valid JSON matching the exact schema."
        )
    return prompt


def parse_invoice_with_retry(raw_text: str, max_retries: int = 3) -> GeminiParsedInvoice:
    """Extract a validated invoice from ``raw_text`` with Groq self-correction.

    Up to ``max_retries`` Groq calls are made.  Each failure to produce output
    that passes the strict Pydantic/Decimal validation is converted into
    pinpointed ``error_feedback`` that is appended to the next prompt, so the
    model fixes its own schema/tax-ID/arithmetic mistakes.

    Raises
    ------
    RuntimeError
        When every attempt fails; the last validation error is embedded.
    """

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("raw_text must contain invoice text to parse.")

    error_feedback = ""
    for attempt in range(1, max_retries + 1):
        prompt = _build_parse_prompt(raw_text, error_feedback or None)
        try:
            response_text = _request_invoice_json(prompt)
            clean_text = clean_json_response(response_text)
            parsed = GeminiParsedInvoice.model_validate_json(clean_text)
            logger.info(
                "Groq invoice extraction validated on attempt %s/%s.", attempt, max_retries
            )
            return parsed
        except (
            ValidationError,
            InvoiceMathError,
            TaxIDValidationError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            error_feedback = str(exc)
            # PII-safe logger: feeback can echo tax IDs / amounts from the document.
            logger.warning(
                "Groq parse attempt %s/%s failed validation: %s",
                attempt,
                max_retries,
                error_feedback,
            )

    raise RuntimeError(
        f"AI parsing failed validation after {max_retries} attempts. Last error: {error_feedback}"
    )
