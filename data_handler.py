"""Persistence and intake for FakturaBot.mk / Saldora.

Two storage paths are intentionally supported:

* JSON is the local source of truth / audit store used by the CLI.
* Excel remains a convenient import and export format for small businesses.

The original collections contract (``load_overdue_invoices``) is preserved and
returns dictionaries for easy future JSON/API consumption.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from legacy_models import Invoice
from utils import (
    normalize_channel,
    normalize_language_code,
    normalize_phone,
    normalize_status,
    now_utc,
    parse_date,
    parse_decimal,
    safe_filename,
)

logger = logging.getLogger(__name__)


class InvoiceDataError(RuntimeError):
    """Raised when an invoice source cannot be read or is structurally invalid."""


@dataclass(slots=True)
class ExcelImportResult:
    invoices: list[Invoice] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Header normalization / aliases
# ---------------------------------------------------------------------------

_HEADER_ALIASES: dict[str, set[str]] = {
    "invoice_id": {
        "invoice_id",
        "invoice_number",
        "invoice_no",
        "invoice",
        "number",
        "id",
        "faktura",
        "faktura_broj",
        "broj_na_faktura",
        "број_на_фактура",
        "фактура",
        "фактура_број",
        "fatura",
        "numri_i_fatures",
    },
    "client_name": {
        "client_name",
        "customer_name",
        "client",
        "customer",
        "komitent",
        "klient",
        "klienti",
        "клиент",
        "комитент",
        "купувач",
    },
    "client_tax_id": {
        "client_tax_id",
        "tax_id",
        "tax_number",
        "tax_no",
        "edb",
        "embg",
        "edb_embg",
        "edb_embg_broj",
        "danocen_broj",
        "даночен_број",
        "едб",
        "ембг",
        "едб_ембг",
        "nipt",
    },
    "net_amount": {
        "net_amount",
        "net",
        "subtotal",
        "neto",
        "osnovica",
        "основа",
        "основна_сума",
        "vlera_pa_tvsh",
    },
    "vat_rate": {
        "vat_rate",
        "vat_percent",
        "ddv_rate",
        "ddv_percent",
        "vat_percentage",
        "ddv_percentage",
        "ддв_стапка",
    },
    "vat_amount": {
        "vat_amount",
        "vat",
        "ddv",
        "ддв",
        "tvsh",
        "iznos_ddv",
        "износ_ддв",
    },
    "total_amount": {
        "total_amount",
        "amount",
        "amount_due",
        "total",
        "grand_total",
        "total_due",
        "gross",
        "gross_amount",
        "вкупно",
        "сума",
        "износ",
        "вкупен_износ",
        "vlera",
        "vlera_totale",
        "fatura_total",
    },
    "currency": {
        "currency",
        "currency_code",
        "валута",
        "valuta",
    },
    "issue_date": {
        "issue_date",
        "invoice_date",
        "date",
        "датум",
        "датум_на_издавање",
        "data_e_leshuar",
    },
    "due_date": {
        "due_date",
        "due",
        "due_on",
        "payment_due_date",
        "rok",
        "датум_на_доспевање",
        "доспева",
        "afati",
        "data_e_skadimit",
    },
    "status": {
        "status",
        "payment_status",
        "статус",
        "состојба",
        "statusi",
    },
    "preferred_language": {
        "language",
        "preferred_language",
        "lang",
        "јазик",
        "gjuha",
    },
    "channel": {
        "channel",
        "preferred_channel",
        "канал",
        "kanali",
    },
    "email": {
        "email",
        "e_mail",
        "client_email",
        "email_address",
    },
    "phone": {
        "phone",
        "client_phone",
        "phone_number",
        "mobile",
        "tel",
        "telephone",
        "телефон",
        "mobil",
        "numri_i_telefonit",
    },
    "description": {
        "description",
        "service",
        "service_description",
        "опис",
        "услуга",
        "shërbimi",
        "pershkrimi",
    },
    "ujp_status": {"ujp_status"},
    "ujp_reference": {"ujp_reference"},
    "ujp_xml_path": {"ujp_xml_path"},
    "reminder_count": {"reminder_count"},
    "last_reminded_at": {"last_reminded_at"},
}

_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in _HEADER_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical


def _normalize_header(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE)
    return re.sub(r"_+", "_", text).strip("_")


def _normalize_columns(columns: Iterable[Any]) -> dict[str, Any]:
    """Map workbook headers to canonical names and reject ambiguity."""

    mapped: dict[str, Any] = {}
    for raw_column in columns:
        normalized = _normalize_header(raw_column)
        canonical = _ALIAS_TO_CANONICAL.get(normalized)
        if not canonical:
            continue
        if canonical in mapped:
            raise InvoiceDataError(
                f"Ambiguous workbook columns: {mapped[canonical]!r} and {raw_column!r} "
                f"both map to {canonical!r}."
            )
        mapped[canonical] = raw_column
    return mapped


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------


def _is_blank(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def _as_text(value: Any) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _optional_amount(value: Any) -> Any:
    if _is_blank(value):
        return None
    return value


def _build_invoice_from_row(
    row: Mapping[str, Any],
    row_number: int,
) -> tuple[Invoice | None, list[str]]:
    warnings: list[str] = []

    def raw(key: str) -> Any:
        return row.get(key)

    invoice_id = _as_text(raw("invoice_id"))
    client_name = _as_text(raw("client_name"))
    if not invoice_id:
        return None, [f"Row {row_number}: missing Invoice ID; skipped."]
    if not client_name:
        return None, [f"Row {row_number}: missing Client Name; skipped."]

    due_raw = raw("due_date")
    try:
        due_date = parse_date(due_raw)
    except ValueError as exc:
        return None, [f"Row {row_number}: invalid due date ({exc}); skipped."]

    issue_raw = _optional_amount(raw("issue_date"))
    try:
        issue_date = parse_date(issue_raw) if issue_raw is not None else due_date
    except ValueError:
        issue_date = due_date
        warnings.append(f"Row {row_number}: issue date could not be parsed; using due date.")

    total_raw = _optional_amount(raw("total_amount"))
    net_raw = _optional_amount(raw("net_amount"))
    vat_raw = _optional_amount(raw("vat_amount"))
    rate_raw = _optional_amount(raw("vat_rate"))

    try:
        total_value = parse_decimal(total_raw) if total_raw is not None else None
        net_value = parse_decimal(net_raw) if net_raw is not None else None
        vat_value = parse_decimal(vat_raw) if vat_raw is not None else None
        rate_value = parse_decimal(rate_raw) if rate_raw is not None else None
    except ValueError as exc:
        return None, [f"Row {row_number}: invalid amount ({exc}); skipped."]

    if total_value is None and net_value is None:
        return None, [f"Row {row_number}: missing amount; skipped."]

    if total_value is None:
        assert net_value is not None
        if rate_value is not None:
            vat_value = (net_value * rate_value / 100).quantize(parse_decimal("0.01"))
        if vat_value is None:
            vat_value = parse_decimal("0.00")
        total_value = net_value + vat_value
        if rate_value is None:
            rate_value = (
                (vat_value / net_value * 100).quantize(parse_decimal("0.01"))
                if net_value > 0 and vat_value
                else parse_decimal("0.00")
            )
    elif net_value is None:
        if vat_value is not None:
            net_value = max(parse_decimal("0.00"), total_value - vat_value)
        else:
            # Legacy workbooks often contain only one "Amount" column.  Treat
            # it as the payable total rather than guessing a VAT split.
            net_value = total_value
            vat_value = parse_decimal("0.00")
            if rate_value is None:
                rate_value = parse_decimal("0.00")
        if rate_value is None:
            rate_value = (
                (vat_value / net_value * 100).quantize(parse_decimal("0.01"))
                if net_value > 0 and vat_value
                else parse_decimal("0.00")
            )
        if vat_value is None:
            vat_value = max(parse_decimal("0.00"), total_value - net_value)
    else:
        if vat_value is None:
            vat_value = max(parse_decimal("0.00"), total_value - net_value)
        if rate_value is None:
            rate_value = (
                (vat_value / net_value * 100).quantize(parse_decimal("0.01"))
                if net_value > 0 and vat_value
                else parse_decimal("0.00")
            )

    if total_value <= 0:
        return None, [f"Row {row_number}: total amount must be positive; skipped."]

    language_raw = _as_text(raw("preferred_language"))
    language = normalize_language_code(language_raw, "EN")
    if language_raw and language == "EN" and language_raw.strip().upper() not in {"EN", "ENG", "ENGLISH"}:
        warnings.append(
            f"Row {row_number}: unsupported language {language_raw!r}; defaulted to EN."
        )

    channel_raw = _as_text(raw("channel"))
    channel = normalize_channel(channel_raw, "Email")
    supported_channel_values = {
        "email",
        "e-mail",
        "mail",
        "viber",
        "whatsapp",
        "whats app",
        "wa",
    }
    if channel_raw and channel_raw.strip().casefold() not in supported_channel_values:
        warnings.append(
            f"Row {row_number}: unsupported channel {channel_raw!r}; defaulted to Email."
        )

    status = normalize_status(_as_text(raw("status")), "UNPAID")
    currency = (_as_text(raw("currency")) or "MKD").upper()
    email = _as_text(raw("email")) or None
    phone = normalize_phone(_as_text(raw("phone"))) or None
    description = _as_text(raw("description")) or "Imported invoice"

    ujp_status = _as_text(raw("ujp_status")) or "NOT_SUBMITTED"
    ujp_reference = _as_text(raw("ujp_reference")) or None
    ujp_xml_path = _as_text(raw("ujp_xml_path")) or None

    reminder_raw = _as_text(raw("reminder_count"))
    try:
        reminder_count = int(reminder_raw) if reminder_raw else 0
    except ValueError:
        reminder_count = 0
        warnings.append(
            f"Row {row_number}: invalid reminder count {reminder_raw!r}; using 0."
        )

    last_reminded_raw = _as_text(raw("last_reminded_at"))
    last_reminded_at: datetime | None = None
    if last_reminded_raw:
        try:
            last_reminded_at = datetime.fromisoformat(last_reminded_raw)
        except ValueError:
            warnings.append(
                f"Row {row_number}: invalid last reminded timestamp; using empty."
            )

    try:
        invoice = Invoice(
            invoice_id=invoice_id,
            client_name=client_name,
            client_tax_id=_as_text(raw("client_tax_id")),
            net_amount=net_value,
            vat_rate=rate_value if rate_value is not None else parse_decimal("0.00"),
            vat_amount=vat_value if vat_value is not None else parse_decimal("0.00"),
            total_amount=total_value,
            currency=currency,
            issue_date=issue_date,
            due_date=due_date,
            status=status,
            language=language,
            channel=channel,
            client_email=email,
            client_phone=phone,
            description=description,
            ujp_status=ujp_status,
            ujp_reference=ujp_reference,
            ujp_xml_path=ujp_xml_path,
            reminder_count=reminder_count,
            last_reminded_at=last_reminded_at,
        )
    except (TypeError, ValueError) as exc:
        return None, [f"Row {row_number}: could not build invoice ({exc}); skipped."]

    return invoice, warnings


def load_invoices_from_excel(
    path: Path | str,
    *,
    sheet_name: str | int = 0,
) -> ExcelImportResult:
    """Load all invoice rows from an Excel workbook.

    Required source columns after alias normalization:
    ``invoice_id``, ``client_name``, ``total_amount`` (or explicit net/VAT
    columns), ``due_date``, and ``status``.

    Rows with missing/invalid critical data are skipped and reported in
    ``ExcelImportResult.warnings`` rather than aborting the whole batch.
    """

    workbook_path = Path(path)
    if not workbook_path.exists():
        raise FileNotFoundError(
            f"Invoice workbook not found: {workbook_path}. "
            "Create invoices.xlsx or pass --excel PATH."
        )

    try:
        frame = pd.read_excel(workbook_path, sheet_name=sheet_name, engine="openpyxl")
    except Exception as exc:
        raise InvoiceDataError(f"Could not read Excel workbook {workbook_path}: {exc}") from exc

    if frame.empty:
        return ExcelImportResult()

    columns = _normalize_columns(frame.columns)
    missing: list[str] = []
    if "invoice_id" not in columns:
        missing.append("Invoice ID")
    if "client_name" not in columns:
        missing.append("Client Name")
    if "due_date" not in columns:
        missing.append("Due Date")
    if "status" not in columns:
        missing.append("Status")
    if "total_amount" not in columns and "net_amount" not in columns:
        missing.append("Amount / Total Amount")

    if missing:
        raise InvoiceDataError(
            "Workbook is missing required column(s): "
            + ", ".join(missing)
            + ". Expected headers include Invoice ID, Client Name, Amount, Due Date, "
            "Status, Preferred Language, and Channel."
        )

    result = ExcelImportResult()
    seen_ids: set[str] = set()

    # Keep source-column lookup separate from canonical row values.
    for index, source_row in frame.iterrows():
        row_number = int(index) + 2  # header is row 1
        row = {
            canonical: source_row[source_column]
            for canonical, source_column in columns.items()
        }
        invoice, warnings = _build_invoice_from_row(row, row_number)
        result.warnings.extend(warnings)
        if invoice is None:
            continue
        key = invoice.invoice_id.casefold()
        if key in seen_ids:
            result.warnings.append(
                f"Row {row_number}: duplicate Invoice ID {invoice.invoice_id!r}; skipped."
            )
            continue
        seen_ids.add(key)
        result.invoices.append(invoice)

    return result


def load_overdue_invoices(
    path: Path | str = "invoices.xlsx",
    as_of: date | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible collections intake.

    Reads an Excel file, filters rows that are not ``PAID`` and whose due date is
    strictly before ``as_of`` (today by default), and returns plain dictionaries
    with the original six fields plus contact/channel context.
    """

    result = load_invoices_from_excel(path)
    for warning in result.warnings:
        logger.warning(warning)

    current = as_of or date.today()
    return [
        invoice.collection_context(current)
        for invoice in result.invoices
        if invoice.is_overdue(current)
    ]


def export_invoices_to_excel(invoices: Iterable[Invoice], path: Path | str) -> Path:
    """Export invoices to an Excel workbook using canonical column names."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "Invoice ID": invoice.invoice_id,
            "Client Name": invoice.client_name,
            "Tax ID": invoice.client_tax_id,
            "Issue Date": invoice.issue_date.isoformat(),
            "Due Date": invoice.due_date.isoformat(),
            "Net Amount": float(invoice.net_amount),
            "VAT Rate": float(invoice.vat_rate),
            "VAT Amount": float(invoice.vat_amount),
            "Total Amount": float(invoice.total_amount),
            "Currency": invoice.currency,
            "Status": invoice.status,
            "Preferred Language": invoice.language.value,
            "Channel": invoice.channel.value,
            "Email": invoice.client_email or "",
            "Phone": invoice.client_phone or "",
            "Description": invoice.description,
            "UJP Status": invoice.ujp_status,
            "UJP Reference": invoice.ujp_reference or "",
            "UJP XML Path": invoice.ujp_xml_path or "",
            "Reminder Count": invoice.reminder_count,
            "Last Reminded At": invoice.last_reminded_at.isoformat()
            if invoice.last_reminded_at
            else "",
        }
        for invoice in invoices
    ]

    frame = pd.DataFrame(
        rows,
        columns=[
            "Invoice ID",
            "Client Name",
            "Tax ID",
            "Issue Date",
            "Due Date",
            "Net Amount",
            "VAT Rate",
            "VAT Amount",
            "Total Amount",
            "Currency",
            "Status",
            "Preferred Language",
            "Channel",
            "Email",
            "Phone",
            "Description",
            "UJP Status",
            "UJP Reference",
            "UJP XML Path",
            "Reminder Count",
            "Last Reminded At",
        ],
    )
    try:
        frame.to_excel(output_path, index=False, engine="openpyxl")
    except Exception as exc:
        raise InvoiceDataError(f"Could not write Excel workbook {output_path}: {exc}") from exc
    return output_path


# ---------------------------------------------------------------------------
# JSON store
# ---------------------------------------------------------------------------


class InvoiceStore:
    """Atomic local JSON store for invoices.

    The envelope is versioned to leave room for migrations.  A plain list is also
    accepted when reading, which keeps old MVP data files usable.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _read_payload(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InvoiceDataError(
                f"Invoice store {self.path} is not valid JSON: {exc}. "
                "Fix or move the file before retrying."
            ) from exc
        except OSError as exc:
            raise InvoiceDataError(f"Could not read invoice store {self.path}: {exc}") from exc

        if isinstance(payload, list):
            raw_invoices = payload
        elif isinstance(payload, dict):
            raw_invoices = payload.get("invoices", [])
        else:
            raise InvoiceDataError(f"Invoice store {self.path} has an unsupported JSON shape.")

        if not isinstance(raw_invoices, list):
            raise InvoiceDataError(f"Invoice store {self.path} has a non-list 'invoices' value.")
        for index, item in enumerate(raw_invoices):
            if not isinstance(item, dict):
                raise InvoiceDataError(
                    f"Invoice store {self.path} has a non-object invoice at index {index}."
                )
        return raw_invoices

    def load_all(self) -> list[Invoice]:
        """Load and validate all invoices, rejecting duplicate IDs."""

        invoices: list[Invoice] = []
        seen: set[str] = set()
        for index, item in enumerate(self._read_payload()):
            try:
                invoice = Invoice.from_dict(item)
            except (TypeError, ValueError) as exc:
                raise InvoiceDataError(
                    f"Invalid invoice at index {index} in {self.path}: {exc}"
                ) from exc
            key = invoice.invoice_id.casefold()
            if key in seen:
                raise InvoiceDataError(
                    f"Duplicate Invoice ID {invoice.invoice_id!r} in {self.path}. "
                    "Resolve duplicates before sending reminders."
                )
            seen.add(key)
            invoices.append(invoice)
        return invoices

    def save_all(self, invoices: Iterable[Invoice]) -> None:
        """Persist the complete store atomically."""

        invoice_list = list(invoices)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "updated_at": now_utc().isoformat(),
            "invoices": [invoice.to_dict() for invoice in invoice_list],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise InvoiceDataError(f"Could not save invoice store {self.path}: {exc}") from exc

    def upsert(self, invoice: Invoice) -> Invoice:
        invoices = self.load_all()
        invoice.updated_at = now_utc()
        for index, existing in enumerate(invoices):
            if existing.invoice_id.casefold() == invoice.invoice_id.casefold():
                invoices[index] = invoice
                break
        else:
            invoices.append(invoice)
        self.save_all(invoices)
        return invoice

    def get(self, invoice_id: str) -> Invoice | None:
        target = invoice_id.strip().casefold()
        for invoice in self.load_all():
            if invoice.invoice_id.casefold() == target:
                return invoice
        return None

    def mark_paid(self, invoice_id: str) -> Invoice:
        target = invoice_id.strip().casefold()
        invoices = self.load_all()
        for invoice in invoices:
            if invoice.invoice_id.casefold() == target:
                invoice.status = "PAID"
                invoice.updated_at = now_utc()
                self.save_all(invoices)
                return invoice
        raise InvoiceDataError(f"Invoice {invoice_id!r} was not found.")

    def mark_reminder_sent(
        self,
        invoice_id: str,
        sent_at: datetime | None = None,
    ) -> Invoice:
        """Record an approved reminder and persist the audit status.

        The payment status becomes ``REMINDER_SENT`` (the invoice remains
        collectible until it is explicitly marked ``PAID``), the reminder
        counter is incremented, and ``last_reminded_at`` is stamped.
        """

        target = invoice_id.strip().casefold()
        timestamp = sent_at or now_utc()
        invoices = self.load_all()
        for invoice in invoices:
            if invoice.invoice_id.casefold() == target:
                invoice.status = "REMINDER_SENT"
                invoice.reminder_count += 1
                invoice.last_reminded_at = timestamp
                invoice.updated_at = timestamp
                self.save_all(invoices)
                return invoice
        raise InvoiceDataError(f"Invoice {invoice_id!r} was not found.")

    def overdue(self, as_of: date | None = None) -> list[Invoice]:
        current = as_of or date.today()
        return [invoice for invoice in self.load_all() if invoice.is_overdue(current)]

    def next_invoice_number(self, prefix: str = "NPL", year: int | None = None) -> str:
        """Generate the next sequential human-friendly invoice number."""

        prefix = re.sub(r"[^A-Za-z0-9-]", "", prefix.upper()) or "NPL"
        current_year = year or date.today().year
        pattern = re.compile(
            rf"^{re.escape(prefix)}-{current_year}-(\d+)$",
            re.IGNORECASE,
        )
        highest = 0
        for invoice in self.load_all():
            match = pattern.match(invoice.invoice_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"{prefix}-{current_year}-{highest + 1:04d}"

    def import_excel(self, path: Path | str) -> ExcelImportResult:
        """Import an Excel workbook and merge rows by invoice ID."""

        result = load_invoices_from_excel(path)
        existing = {invoice.invoice_id.casefold(): invoice for invoice in self.load_all()}
        for invoice in result.invoices:
            existing[invoice.invoice_id.casefold()] = invoice
        self.save_all(existing.values())
        return result

    def export_excel(self, path: Path | str) -> Path:
        return export_invoices_to_excel(self.load_all(), path)

    def ensure_initialized(self) -> None:
        """Create an empty store if this is the first run."""

        if not self.path.exists():
            self.save_all([])


def safe_export_name(value: str) -> str:
    """Compatibility helper for CLI exports."""

    return safe_filename(value)
