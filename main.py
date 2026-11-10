"""Saldora + FakturaBot.mk command-line orchestrator.

This is the human-in-the-loop surface for the MVP:

1. Create an invoice from a text/voice request.
2. Validate and generate a mock UJP e-Faktura UBL document.
3. Persist the invoice locally.
4. Generate polite, localized collection reminders with Gemini model fallback.
5. Require explicit human approval before each mock send.
6. Persist ``REMINDER_SENT`` plus the reminder timestamp after approval.
7. Classify inbound client replies for the business owner.

The CLI deliberately contains no business logic that cannot be reused by a future
FastAPI endpoint.  Services receive configuration and data explicitly.
"""

from __future__ import annotations

# NOTE: the process environment is populated lazily by ``Settings.from_env()``
# inside ``main()`` (config.py does the dotenv loading there).  A module-level
# ``load_dotenv()`` would leak real SMTP/Resend credentials into every pytest
# run, because collection imports all test modules -- including ``main`` --
import argparse
import logging
import sys
from collections.abc import Callable, Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy import insert

from config import ConfigError, Settings
from data_handler import InvoiceDataError, InvoiceStore
from database import get_db, init_db
from dispatcher import BaseDispatcher, DispatchError, SMTPDispatcher, dispatch_message
from invoice_parser import InvoiceParsingError, parse_invoice_prompt
from legacy_models import Channel, Invoice, Language, ParsedInvoice, ReplyAnalysis
from logger import PIIFilter
from migration import seed_legacy_json_data
from models import Client as ClientRow
from models import Invoice as InvoiceRow
from models import InvoiceState
from reply_parser import ReplyParser
from schemas import ClientSchema, InvoiceCreate
from state_machine import InvalidStateTransitionError, transition_invoice_state
from tone_agent import (
    ToneAgent,
    ToneAgentError,
    ToneConfigurationError,
    build_email_subject,
)
from ujp_service import UJPService
from utils import (
    format_money,
    is_valid_email,
    is_valid_tax_id,
    normalize_channel,
    normalize_language_code,
    normalize_phone,
    now_utc,
    parse_date,
    parse_decimal,
)
from voice_transcriber import VoiceTranscriptionError, transcribe_audio

_SEPARATOR = "=" * 82
_SUBSEPARATOR = "-" * 82

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------


def _configure_console_encoding() -> None:
    """Avoid crashes when Cyrillic/Albanian text is printed on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


def _section(title: str) -> None:
    print()
    print(_SEPARATOR)
    print(title)
    print(_SEPARATOR)


def _error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def _warning(message: str) -> None:
    print(f"[WARNING] {message}")


def _info(message: str) -> None:
    print(f"[INFO] {message}")


def _read_multiline(instruction: str, *, keep_hint: str = "Type END on its own line to finish.") -> str:
    """Read a multi-line message without an external editor dependency."""

    print(instruction)
    print(keep_hint)
    lines: list[str] = []
    while True:
        line = input("| ")
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _ask_yes_no(question: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"{question} {suffix} ").strip().casefold()
    if not answer:
        return default
    return answer in {"y", "yes", "д", "да", "po"}


def _parse_as_of(raw: str | None, settings: Settings) -> date:
    if not raw:
        return settings.today()
    try:
        return parse_date(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid --as-of date {raw!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Invoice creation / UJP validation
# ---------------------------------------------------------------------------


def _print_parsed_invoice(parsed: ParsedInvoice) -> None:
    _section("FAKTURABOT.MK - PARSED INVOICE DRAFT")
    print(f"Client:          {parsed.client_name or '(missing)'}")
    print(f"Tax ID:          {parsed.client_tax_id or '(missing)'}")
    if parsed.amount is None:
        print("Amount entered:  (missing)")
    else:
        treatment = "VAT included" if parsed.amount_includes_vat else "VAT excluded"
        print(
            f"Amount entered:  {format_money(parsed.amount, parsed.currency)} "
            f"({treatment}, VAT {parsed.vat_rate:.2f}%)"
        )
        try:
            net, vat, total = parsed.breakdown()
            print(f"Net / VAT / Total: {format_money(net, parsed.currency)} / "
                  f"{format_money(vat, parsed.currency)} / "
                  f"{format_money(total, parsed.currency)}")
        except ValueError as exc:
            print(f"Totals:          [cannot calculate: {exc}]")
    print(f"Issue date:      {parsed.issue_date.isoformat()}")
    print(f"Due date:        {parsed.due_date.isoformat() if parsed.due_date else '(missing)'}")
    print(f"Language:        {parsed.language.value}")
    print(f"Channel:         {parsed.channel.value}")
    print(f"Email:           {parsed.client_email or '(not set)'}")
    print(f"Phone:           {parsed.client_phone or '(not set)'}")
    print(f"Description:     {parsed.description}")

    if parsed.missing_fields:
        print(f"Missing fields:  {', '.join(parsed.missing_fields)}")
    if parsed.warnings:
        print(_SUBSEPARATOR)
        print("Parser notes:")
        for warning in parsed.warnings:
            print(f"  - {warning}")


def _prompt_amount(parsed: ParsedInvoice) -> None:
    while True:
        raw = input("Invoice amount (e.g. 15000 or 15.000,00): ").strip()
        if not raw:
            return
        try:
            parsed.amount = parse_decimal(raw)
        except ValueError as exc:
            print(f"  Invalid amount: {exc}")
            continue
        if parsed.amount <= 0:
            print("  Amount must be greater than zero.")
            continue
        included = input(
            "Does this amount include VAT? "
            f"[{'Y/n' if parsed.amount_includes_vat else 'y/N'}] "
        ).strip().casefold()
        if included in {"y", "yes"}:
            parsed.amount_includes_vat = True
        elif included in {"n", "no"}:
            parsed.amount_includes_vat = False
        return


def _complete_missing_fields(parsed: ParsedInvoice, settings: Settings) -> None:
    """Ask only for information the deterministic parser could not extract."""

    if not parsed.client_name:
        while True:
            value = input("Client name (required): ").strip()
            if value:
                parsed.client_name = value
                break
            print("  Client name is required.")

    if not parsed.client_tax_id:
        while True:
            value = input("Client tax ID / ЕДБ / ЕМБГ (required): ").strip()
            if value:
                parsed.client_tax_id = value
                if not is_valid_tax_id(value):
                    _warning("Tax ID is structurally unusual; UJP validation will warn you.")
                break
            print("  Client tax ID is required for a UJP e-Faktura.")

    if parsed.amount is None:
        _prompt_amount(parsed)

    if parsed.due_date is None:
        while True:
            raw = input(
                f"Due date [YYYY-MM-DD, default today + {settings.default_payment_terms_days} days]: "
            ).strip()
            if not raw:
                from datetime import timedelta

                parsed.due_date = settings.today() + timedelta(
                    days=settings.default_payment_terms_days
                )
                break
            try:
                parsed.due_date = parse_date(raw)
                break
            except ValueError as exc:
                print(f"  Invalid date: {exc}")


def _edit_parsed_fields(parsed: ParsedInvoice, settings: Settings) -> None:
    """Small field editor used before creating the invoice."""

    while True:
        print()
        print("Edit fields:")
        print("  1. Client name       2. Client tax ID     3. Amount")
        print("  4. VAT rate          5. VAT treatment    6. Issue date")
        print("  7. Due date          8. Language         9. Channel")
        print(" 10. Email            11. Phone           12. Description")
        print("  0. Done")
        choice = input("Field to edit [0-12]: ").strip()

        if choice in {"0", ""}:
            return

        if choice == "1":
            value = input(f"Client name [{parsed.client_name or ''}]: ").strip()
            if value:
                parsed.client_name = value
        elif choice == "2":
            value = input(f"Client tax ID [{parsed.client_tax_id or ''}]: ").strip()
            if value:
                parsed.client_tax_id = value
        elif choice == "3":
            _prompt_amount(parsed)
        elif choice == "4":
            raw = input(f"VAT rate % [{parsed.vat_rate:.2f}]: ").strip()
            if raw:
                try:
                    rate = parse_decimal(raw)
                except ValueError as exc:
                    print(f"  Invalid VAT rate: {exc}")
                else:
                    if Decimal(0) <= rate <= Decimal(100):
                        parsed.vat_rate = rate
                    else:
                        print("  VAT rate must be between 0 and 100.")
        elif choice == "5":
            parsed.amount_includes_vat = _ask_yes_no(
                "Does the entered amount include VAT?",
                default=parsed.amount_includes_vat,
            )
        elif choice == "6":
            raw = input(f"Issue date [{parsed.issue_date.isoformat()}]: ").strip()
            if raw:
                try:
                    parsed.issue_date = parse_date(raw)
                except ValueError as exc:
                    print(f"  Invalid date: {exc}")
        elif choice == "7":
            raw = input(
                f"Due date [{parsed.due_date.isoformat() if parsed.due_date else ''}]: "
            ).strip()
            if raw:
                try:
                    parsed.due_date = parse_date(raw)
                except ValueError as exc:
                    print(f"  Invalid date: {exc}")
        elif choice == "8":
            raw = input(f"Language MK/AL/EN [{parsed.language.value}]: ").strip()
            if raw:
                language = normalize_language_code(raw, parsed.language.value)
                if language == parsed.language.value and raw.strip().upper() not in {
                    parsed.language.value,
                    "MACEDONIAN",
                    "ALBANIAN",
                    "ENGLISH",
                }:
                    print("  Unsupported language; use MK, AL, or EN.")
                else:
                    parsed.language = Language.from_value(language)
        elif choice == "9":
            raw = input(f"Channel Email/Viber/WhatsApp [{parsed.channel.value}]: ").strip()
            if raw:
                channel = normalize_channel(raw, parsed.channel.value)
                parsed.channel = Channel.from_value(channel)
        elif choice == "10":
            raw = input(f"Client email [{parsed.client_email or ''}]: ").strip()
            if raw:
                if not is_valid_email(raw):
                    _warning("Email is structurally invalid; it can still be stored for now.")
                parsed.client_email = raw
        elif choice == "11":
            raw = input(f"Client phone [{parsed.client_phone or ''}]: ").strip()
            if raw:
                parsed.client_phone = normalize_phone(raw) or raw
        elif choice == "12":
            raw = input(f"Description [{parsed.description}]: ").strip()
            if raw:
                parsed.description = raw
        else:
            print("  Unknown field number.")


def _create_invoice_flow(
    settings: Settings,
    store: InvoiceStore,
    *,
    prompt_text: str | None = None,
    voice_path: str | None = None,
    offer_collection: bool = False,
    on_created: Callable[[Invoice], None] | None = None,
) -> tuple[int, bool]:
    """Create one invoice and return ``(exit_code, invoice_created)``.

    ``on_created`` (optional) receives the persisted legacy ``Invoice`` right
    after the JSON-store upsert; the interactive menu uses it to mirror every
    newly created invoice into the SQL ledger as ``DRAFT`` without changing
    the legacy create contract for direct callers.
    """

    _section("NEW INVOICE")

    if not settings.is_business_configured:
        _warning(
            "BUSINESS_NAME and/or BUSINESS_TAX_ID are not configured. "
            "The invoice can be parsed, but UJP validation will reject it until .env is completed."
        )
    if not settings.has_groq_api_key and voice_path:
        _error(
            "Voice transcription requires GROQ_API_KEY. "
            "Set it in .env or provide a text prompt instead."
        )
        return 1, False

    try:
        if voice_path:
            _info(f"Transcribing voice prompt: {voice_path}")
            transcript = transcribe_audio(voice_path, settings=settings)
            print(f"Transcript: {transcript}")
            parsed = parse_invoice_prompt(transcript, settings=settings)
        else:
            if not prompt_text:
                prompt_text = input(
                    "Describe the invoice (client, tax ID, amount, VAT, due date, language): "
                ).strip()
            if not prompt_text:
                _info("No prompt entered. Cancelled.")
                return 0, False
            parsed = parse_invoice_prompt(prompt_text, settings=settings)
    except (VoiceTranscriptionError, InvoiceParsingError) as exc:
        _error(str(exc))
        return 1, False

    if parsed.missing_fields:
        _print_parsed_invoice(parsed)
    _complete_missing_fields(parsed, settings)

    while True:
        _print_parsed_invoice(parsed)
        choice = input(
            "\n[Y] Create and validate invoice   [E] Edit fields   [S] Cancel: "
        ).strip().casefold()
        if choice in {"y", "yes", "д", "да"}:
            break
        if choice in {"e", "edit"}:
            _edit_parsed_fields(parsed, settings)
            continue
        if choice in {"s", "skip", "no", "n", "откажи"}:
            _info("Invoice creation cancelled.")
            return 0, False
        print("Please enter Y, E, or S.")

    if parsed.amount is None or not parsed.client_name or not parsed.client_tax_id:
        _error("Required invoice fields are still missing.")
        return 1, False

    invoice_id = store.next_invoice_number(
        prefix=settings.invoice_prefix,
        year=settings.today().year,
    )
    try:
        invoice = parsed.to_invoice(invoice_id)
    except ValueError as exc:
        _error(f"Could not build invoice: {exc}")
        return 1, False

    submission = UJPService(settings).generate_and_submit(
        invoice,
        output_dir=settings.ubl_output_dir,
    )
    for warning in submission.validation.warnings:
        _warning(f"UJP: {warning}")
    for error in submission.validation.errors:
        _error(f"UJP: {error}")

    if not submission.success:
        print()
        _error("Invoice was not saved because mock UJP validation failed.")
        print(
            "Complete BUSINESS_NAME, BUSINESS_TAX_ID, and the client tax ID in .env, "
            "then retry."
        )
        return 1, False

    invoice.ujp_status = "MOCK_SUBMITTED"
    invoice.ujp_reference = submission.reference
    invoice.ujp_xml_path = submission.xml_path
    invoice.updated_at = now_utc()
    store.upsert(invoice)

    if on_created is not None:
        try:
            on_created(invoice)
        except Exception as exc:  # noqa: BLE001 - SQL ledger must not lose the JSON flow
            _warning(f"SQL ledger mirror skipped: {exc}")

    _section("INVOICE CREATED")
    print(f"Invoice ID:      {invoice.invoice_id}")
    print(f"Client:          {invoice.client_name}")
    print(f"Total:           {format_money(invoice.total_amount, invoice.currency)}")
    print(f"Due date:        {invoice.due_date.isoformat()}")
    print(f"UJP mock ref:    {invoice.ujp_reference}")
    print(f"UBL XML:         {invoice.ujp_xml_path}")
    print(f"Stored in:       {store.path}")
    _info(submission.message)

    if offer_collection and _ask_yes_no("Run collections now?", default=False):
        return _run_collections(settings=settings, store=store), True
    return 0, True


# ---------------------------------------------------------------------------
# Collections HITL
# ---------------------------------------------------------------------------


def _ensure_recipient(invoice: Invoice, store: InvoiceStore) -> str | None:
    """Ensure the invoice has a valid destination before spending an AI call."""

    channel = invoice.channel
    current = invoice.recipient_for_channel()

    if channel.value == "Email":
        if current and is_valid_email(current):
            return current
        if current:
            _warning(f"Stored email for {invoice.client_name} is invalid: {current!r}")
        while True:
            value = input(
                f"Email address for {invoice.client_name} (Enter to skip): "
            ).strip()
            if not value:
                return None
            if not is_valid_email(value):
                print("  Invalid email address. Try again or press Enter to skip.")
                continue
            invoice.client_email = value
            invoice.updated_at = now_utc()
            store.upsert(invoice)
            return value

    if current and normalize_phone(current):
        return normalize_phone(current)
    if current:
        _warning(f"Stored phone number for {invoice.client_name} is invalid: {current!r}")
    while True:
        value = input(
            f"{channel.value} phone for {invoice.client_name} (Enter to skip): "
        ).strip()
        if not value:
            return None
        normalized = normalize_phone(value)
        if not normalized or len(normalized.lstrip("+")) < 7:
            print("  Invalid phone number. Try again or press Enter to skip.")
            continue
        invoice.client_phone = normalized
        invoice.updated_at = now_utc()
        store.upsert(invoice)
        return normalized


def _print_collection_draft(
    index: int,
    total: int,
    invoice: Invoice,
    draft: str,
    as_of: date,
) -> None:
    tier = invoice.tone_tier(as_of)
    _section(f"DRAFT {index}/{total} - HUMAN APPROVAL REQUIRED")
    print(f"Client:      {invoice.client_name}")
    print(f"Invoice:     {invoice.invoice_id}")
    print(f"Amount:      {format_money(invoice.total_amount, invoice.currency)}")
    print(f"Due date:    {invoice.due_date.isoformat()} ({invoice.days_late(as_of)} days late)")
    print(f"Tone tier:   {tier.label} ({tier.value})")
    print(f"Language:    {invoice.language.value} | Channel: {invoice.channel.value}")
    print(f"Reminders:   {invoice.reminder_count}")
    print(_SUBSEPARATOR)
    print(draft.strip())
    print(_SUBSEPARATOR)


def _run_collections(
    settings: Settings,
    store: InvoiceStore,
    *,
    excel_path: str | None = None,
    as_of: date | None = None,
    limit: int | None = None,
) -> int:
    _section("SALDORA - COLLECTIONS")

    if excel_path:
        try:
            imported = store.import_excel(excel_path)
        except (FileNotFoundError, InvoiceDataError) as exc:
            _error(str(exc))
            return 1
        for warning in imported.warnings:
            _warning(warning)
        _info(f"Imported {len(imported.invoices)} invoice(s) from {excel_path}.")

    if not settings.has_groq_api_key:
        _error(
            "GROQ_API_KEY is missing or still a placeholder. "
            "Copy .env.example to .env and add a valid Groq API key to generate reminders."
        )
        return 1

    current = as_of or settings.today()
    try:
        invoices = store.overdue(current)
    except InvoiceDataError as exc:
        _error(str(exc))
        return 1

    if limit is not None:
        if limit <= 0:
            _error("--limit must be greater than zero.")
            return 1
        invoices = invoices[:limit]

    if not invoices:
        _info(f"No overdue, unpaid invoices as of {current.isoformat()}. Nothing to send.")
        return 0

    _info(
        f"{len(invoices)} overdue invoice(s) as of {current.isoformat()}. "
        "No message will be sent without your explicit approval."
    )

    agent = ToneAgent(settings)
    sent = 0
    skipped = 0
    failed = 0
    quit_requested = False

    for index, invoice in enumerate(invoices, start=1):
        print()
        print(
            f"[{index}/{len(invoices)}] {invoice.client_name} | "
            f"{invoice.invoice_id} | due {invoice.due_date.isoformat()}"
        )

        recipient = _ensure_recipient(invoice, store)
        if not recipient:
            _warning("No destination on file; skipped.")
            skipped += 1
            continue

        try:
            draft = agent.generate_draft(invoice, as_of=current)
        except ToneConfigurationError as exc:
            _error(str(exc))
            return 1
        except ToneAgentError as exc:
            _error(f"Draft generation failed for {invoice.invoice_id}: {exc}")
            failed += 1
            continue

        while True:
            _print_collection_draft(index, len(invoices), invoice, draft, current)
            choice = input(
                "[Y] Send exactly as is   [E] Edit message   [S] Skip client   [Q] Quit: "
            ).strip().casefold()

            if choice in {"y", "yes", "send"}:
                subject = (
                    build_email_subject(invoice.invoice_id, invoice.language)
                    if invoice.channel.value == "Email"
                    else None
                )
                try:
                    result = dispatch_message(
                        channel=invoice.channel,
                        recipient=recipient,
                        message=draft,
                        invoice_id=invoice.invoice_id,
                        language=invoice.language.value,
                        subject=subject,
                    )
                except DispatchError as exc:
                    _error(str(exc))
                    failed += 1
                    break

                try:
                    invoice = store.mark_reminder_sent(invoice.invoice_id)
                except InvoiceDataError as exc:
                    _error(f"Message was mocked but the audit update failed: {exc}")
                else:
                    _info(
                        f"Mock send complete: {result.message_id} | "
                        f"status={invoice.status} | last_reminded_at="
                        f"{invoice.last_reminded_at.isoformat() if invoice.last_reminded_at else 'n/a'}"
                    )
                sent += 1
                break

            if choice in {"e", "edit"}:
                revised = _read_multiline(
                    "Paste/edit the message. Existing message is kept if you finish without text."
                )
                if revised:
                    draft = revised
                    _info("Message updated.")
                continue

            if choice in {"s", "skip", "no", "n"}:
                _info(f"Skipped {invoice.client_name}.")
                skipped += 1
                break

            if choice in {"q", "quit"}:
                quit_requested = True
                _info("Stopping collections. Remaining invoices were not contacted.")
                break

            print("Please enter Y, E, S, or Q.")

        if quit_requested:
            break

    _section("COLLECTION SUMMARY")
    print(f"Sent (mock): {sent}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {failed}")
    print(f"Remaining:   {max(0, len(invoices) - sent - skipped - failed)}")
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def _print_reply_analysis(analysis: ReplyAnalysis) -> None:
    _section("REPLY CLASSIFICATION")
    print(f"Category:        {analysis.category.value}")
    print(f"Confidence:      {analysis.confidence:.0%}")
    print(f"Matched phrases: {', '.join(analysis.matched_terms) if analysis.matched_terms else '(none)'}")
    print(f"Language:        {analysis.language or '(not specified)'}")
    print(_SUBSEPARATOR)
    print(f"Suggested action: {analysis.suggested_action}")
    _warning("Reply classification is advisory. Verify payments and disputes manually.")


def _run_reply_flow(text: str | None = None) -> int:
    if not text:
        text = _read_multiline(
            "Paste the client's reply:",
            keep_hint="Type END on its own line to classify it.",
        )
    if not text.strip():
        _info("No reply supplied.")
        return 0

    analysis = ReplyParser().classify(text)
    _print_reply_analysis(analysis)
    return 0


# ---------------------------------------------------------------------------
# Data operations
# ---------------------------------------------------------------------------


def _run_import_excel(store: InvoiceStore, path: str) -> int:
    try:
        imported = store.import_excel(path)
    except (FileNotFoundError, InvoiceDataError) as exc:
        _error(str(exc))
        return 1

    for warning in imported.warnings:
        _warning(warning)
    _info(f"Imported {len(imported.invoices)} invoice(s) into {store.path}.")
    return 0


def _run_export_excel(store: InvoiceStore, path: str) -> int:
    try:
        output = store.export_excel(path)
    except (InvoiceDataError, OSError) as exc:
        _error(str(exc))
        return 1
    _info(f"Exported {len(store.load_all())} invoice(s) to {output}.")
    return 0


def _run_ubl_export(
    settings: Settings,
    store: InvoiceStore,
    invoice_id: str,
    *,
    output_dir: str | None = None,
    show_xml: bool = False,
) -> int:
    """Validate an existing invoice and save its UBL 2.1 XML audit copy."""

    try:
        invoice = store.get(invoice_id)
    except InvoiceDataError as exc:
        _error(str(exc))
        return 1
    if invoice is None:
        _error(f"Invoice {invoice_id!r} was not found in {store.path}.")
        return 1

    service = UJPService(settings)
    validation = service.validate_invoice(invoice)
    for warning in validation.warnings:
        _warning(f"UBL: {warning}")
    for error in validation.errors:
        _error(f"UBL: {error}")
    if not validation.is_valid:
        return 1

    try:
        xml_path = service.save_ubl_xml(invoice, output_dir=output_dir)
    except (OSError, ValueError) as exc:
        _error(f"Could not save UBL XML for {invoice.invoice_id}: {exc}")
        return 1

    _section("UBL 2.1 INVOICE XML")
    print(f"Invoice:     {invoice.invoice_id}")
    print(f"Client:      {invoice.client_name}")
    print(f"XML file:    {xml_path}")
    print("Validation:  OK (local UBL structure and totals)")
    print(f"UJP status:  {invoice.ujp_status}")
    if show_xml:
        print(_SUBSEPARATOR)
        try:
            print(xml_path.read_text(encoding="utf-8"))
        except OSError as exc:
            _error(f"Could not read back generated XML: {exc}")
            return 1
    return 0


def _print_invoice_list(
    store: InvoiceStore,
    *,
    as_of: date,
    overdue_only: bool = False,
    limit: int | None = None,
) -> int:
    try:
        invoices = store.load_all()
    except InvoiceDataError as exc:
        _error(str(exc))
        return 1

    if overdue_only:
        invoices = [invoice for invoice in invoices if invoice.is_overdue(as_of)]
    if limit is not None:
        invoices = invoices[:limit]

    _section("INVOICES")
    if not invoices:
        _info("No invoices found.")
        return 0

    print(f"{'ID':<18} {'Client':<26} {'Total':>15} {'Due':<12} {'Days':>4} {'Status':<12} {'Lang':<4} {'Channel'}")
    print("-" * 112)
    for invoice in invoices:
        days = invoice.days_late(as_of) if invoice.is_overdue(as_of) else 0
        print(
            f"{invoice.invoice_id[:17]:<18} "
            f"{invoice.client_name[:25]:<26} "
            f"{format_money(invoice.total_amount, invoice.currency):>15} "
            f"{invoice.due_date.isoformat():<12} "
            f"{days:>4} "
            f"{invoice.status[:11]:<12} "
            f"{invoice.language.value:<4} "
            f"{invoice.channel.value}"
        )
    print()
    _info(f"{len(invoices)} invoice(s).")
    return 0


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------


def _interactive_menu(settings: Settings, store: InvoiceStore) -> int:
    while True:
        _section("SALDORA + FAKTURABOT.MK")
        ai_status = "configured" if settings.has_groq_api_key else "MISSING GROQ_API_KEY"
        ujp_status = "configured" if settings.is_business_configured else "business details missing"
        print(f"Groq: {ai_status} | UJP mock: {ujp_status}")
        print(f"Invoice store: {store.path}")
        print()
        print("  1. Create invoice from text")
        print("  2. Create invoice from voice note")
        print("  3. Run collection reminders")
        print("  4. Classify a client reply")
        print("  5. Import Excel workbook")
        print("  6. Export Excel workbook")
        print("  7. List invoices")
        print("  8. Generate UBL 2.1 XML for an invoice")
        print("  9. Approve invoice (SQL ledger)")
        print(" 10. Send reminder via SMTP (SQL ledger)")
        print("  0. Exit")

        choice = input("\nChoose [0-10]: ").strip()
        try:
            if choice == "0":
                return 0
            if choice == "1":
                _create_invoice_flow(
                    settings,
                    store,
                    offer_collection=True,
                    on_created=_sync_legacy_invoice_to_sql,
                )
            elif choice == "2":
                voice = input("Path to voice note: ").strip()
                if not voice:
                    continue
                _create_invoice_flow(
                    settings,
                    store,
                    voice_path=voice,
                    offer_collection=True,
                )
            elif choice == "3":
                _run_collections(settings=settings, store=store)
            elif choice == "4":
                _run_reply_flow()
            elif choice == "5":
                path = input("Excel workbook path: ").strip()
                if path:
                    _run_import_excel(store, path)
            elif choice == "6":
                path = input("Output Excel path [data/invoices_export.xlsx]: ").strip()
                _run_export_excel(
                    store,
                    path or str(settings.data_dir / "invoices_export.xlsx"),
                )
            elif choice == "7":
                _print_invoice_list(store, as_of=settings.today())
            elif choice == "8":
                invoice_id = input("Invoice ID: ").strip()
                if invoice_id:
                    _run_ubl_export(settings, store, invoice_id)
            elif choice == "9":
                _run_approve_invoice_flow(settings)
            elif choice == "10":
                _run_send_reminder_flow(settings)
            else:
                print("Unknown option.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
        except EOFError:
            print("\nInput closed. Returning to menu.")


# ---------------------------------------------------------------------------
# SQL ledger: bootstrap, mirroring, and DB-backed CLI handlers
# ---------------------------------------------------------------------------


def _install_pii_filter() -> None:
    """Attach :class:`PIIFilter` to every root-logging handler, idempotently."""

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root.addHandler(handler)
    for handler in root.handlers:
        if not any(isinstance(f, PIIFilter) for f in handler.filters):
            handler.addFilter(PIIFilter())


def _bootstrap_sql_ledger() -> None:
    """Create the SQL tables, seed legacy JSON data once, and mask logging."""

    init_db()
    with get_db() as session:
        seed_legacy_json_data(session)
    _install_pii_filter()


def _sync_legacy_invoice_to_sql(invoice: Invoice) -> None:
    """Mirror one persisted legacy invoice into the SQL ledger as ``DRAFT``.

    Runs inside a single ``get_db()`` block so the session always closes --
    a leaked SQLite connection holds file locks.  Client identity is validated
    by :class:`ClientSchema`; amounts by :class:`InvoiceCreate`, whose computed
    VAT/total (ROUND_HALF_UP, cent-exact) are the only money values written.
    """

    client_payload = ClientSchema(
        name=invoice.client_name,
        tax_id=invoice.client_tax_id,
        email=invoice.client_email or "",
        phone=invoice.client_phone or "",
    )
    vat_rate = parse_decimal(invoice.vat_rate)
    amount_create = InvoiceCreate(
        client_tax_id=invoice.client_tax_id,
        net_amount=invoice.net_amount,
        vat_rate=vat_rate,
    )

    with get_db() as db:
        existing_client = (
            db.query(ClientRow).filter_by(tax_id=invoice.client_tax_id).first()
        )
        if existing_client is None:
            db.add(
                ClientRow(
                    tax_id=client_payload.tax_id,
                    name=client_payload.name,
                    email=client_payload.email,
                    phone=client_payload.phone,
                )
            )
        else:
            # A tax ID may already be registered with incomplete contact data.
            # Fill missing (or unusable) fields instead of silently keeping an
            # empty e-mail that later blocks reminder dispatch.
            if client_payload.email and not is_valid_email(
                (existing_client.email or "").strip()
            ):
                existing_client.email = client_payload.email
            if client_payload.phone and not (existing_client.phone or "").strip():
                existing_client.phone = client_payload.phone
            if client_payload.name and not (existing_client.name or "").strip():
                existing_client.name = client_payload.name
        is_new = (
            db.query(InvoiceRow).filter_by(id=invoice.invoice_id).first() is None
        )
        if not is_new:
            _info(f"SQL ledger already holds {invoice.invoice_id}; mirror skipped.")
            return
        db.execute(
            insert(InvoiceRow).values(
                id=invoice.invoice_id,
                client_tax_id=amount_create.client_tax_id,
                net_amount=amount_create.net_amount,
                vat_rate=amount_create.vat_rate,
                vat_amount=amount_create.vat_amount,
                total_amount=amount_create.total_amount,
                state=InvoiceState.DRAFT,
            )
        )
        db.commit()
    _info(
        f"SQL ledger: {invoice.invoice_id} inserted as DRAFT "
        f"(total {amount_create.total_amount})."
    )


def _run_approve_invoice_flow(settings: Settings, invoice_id: str | None = None) -> int:
    """Human-gated approval: ``DRAFT -> APPROVED`` through the state machine."""

    _section("APPROVE INVOICE")
    if not invoice_id:
        invoice_id = input("Invoice ID to approve: ").strip()
    if not invoice_id:
        _info("No invoice ID entered. Cancelled.")
        return 0

    with get_db() as db:
        invoice = db.query(InvoiceRow).filter_by(id=invoice_id).first()
        if invoice is None:
            _error(f"Invoice {invoice_id} not found in the SQL ledger.")
            return 1
        if invoice.state != InvoiceState.DRAFT:
            _error(
                f"Invoice {invoice_id} is {invoice.state.value}, not DRAFT; "
                "only DRAFT invoices can be approved."
            )
            return 1
        try:
            transition_invoice_state(
                invoice,
                InvoiceState.APPROVED,
                db,
                approve_override=True,
            )
        except InvalidStateTransitionError as exc:
            _error(f"Approval blocked: {exc}")
            return 1
        _info(f"Invoice {invoice_id} is now {invoice.state.value} (approved).")
    return 0


def _run_send_reminder_flow(settings: Settings, invoice_id: str | None = None) -> int:
    """Send one SMTP reminder for an APPROVED invoice via ``SMTPDispatcher``."""

    _section("SEND REMINDER (SQL ledger)")
    if not invoice_id:
        invoice_id = input("Invoice ID to send: ").strip()
    if not invoice_id:
        _info("No invoice ID entered. Cancelled.")
        return 0

    dispatcher = SMTPDispatcher()
    if not isinstance(dispatcher, BaseDispatcher):  # defensive contract check
        _error("SMTPDispatcher does not honour the BaseDispatcher contract.")
        return 1

    with get_db() as db:
        try:
            accepted = dispatcher.send_reminder(invoice_id, db)
        except InvalidStateTransitionError as exc:
            _error(f"Reminder blocked: {exc}")
            return 1
        except ValueError as exc:
            _error(f"Reminder failed: {exc}")
            return 1

    if accepted:
        _info(f"Reminder for {invoice_id} delivered; invoice marked SENT.")
        return 0
    _error(
        f"Reminder for {invoice_id} could not be delivered; "
        "the SQL ledger marked the invoice FAILED for reprocessing."
    )
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "FakturaBot.mk invoice creation + Saldora polite collections "
            "(human-in-the-loop MVP)."
        )
    )
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for unexpected errors.")
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser("create", help="Create and validate an invoice.")
    create_parser.add_argument("prompt", nargs="?", help="Invoice instruction in text.")
    create_parser.add_argument("--voice", help="Path to a voice note (Gemini transcription).")
    create_parser.add_argument(
        "--collect",
        action="store_true",
        help="Run collection reminders after the invoice is saved.",
    )

    collect_parser = subparsers.add_parser("collect", help="Generate and approve reminders.")
    collect_parser.add_argument("--excel", help="Import this Excel workbook before collecting.")
    collect_parser.add_argument("--as-of", help="Evaluate overdue status on YYYY-MM-DD.")
    collect_parser.add_argument("--limit", type=int, help="Maximum invoices this run.")

    reply_parser = subparsers.add_parser("reply", help="Classify an inbound client reply.")
    reply_parser.add_argument("text", nargs="?", help="Reply text; prompts if omitted.")

    import_parser = subparsers.add_parser("import-excel", help="Import invoices from Excel.")
    import_parser.add_argument("path", help="Workbook path.")

    export_parser = subparsers.add_parser("export-excel", help="Export invoices to Excel.")
    export_parser.add_argument(
        "path",
        nargs="?",
        default=str(Path("data") / "invoices_export.xlsx"),
        help="Output workbook path.",
    )

    ubl_parser = subparsers.add_parser(
        "ubl",
        help="Generate, validate, and save UBL 2.1 XML for a stored invoice.",
    )
    ubl_parser.add_argument("invoice_id", help="Stored invoice ID.")
    ubl_parser.add_argument(
        "--output-dir",
        help="Destination directory (default: output/).",
    )
    ubl_parser.add_argument(
        "--print",
        dest="print_xml",
        action="store_true",
        help="Print the generated XML after saving it.",
    )

    list_parser = subparsers.add_parser("list", help="List stored invoices.")
    list_parser.add_argument("--overdue", action="store_true", help="Only overdue invoices.")
    list_parser.add_argument("--as-of", help="Evaluation date YYYY-MM-DD.")
    list_parser.add_argument("--limit", type=int)

    approve_parser = subparsers.add_parser(
        "approve", help="Approve an invoice in the SQL ledger (HITL gate)."
    )
    approve_parser.add_argument("invoice_id", nargs="?", help="Invoice ID; prompts if omitted.")

    send_parser = subparsers.add_parser(
        "send", help="Send an SMTP reminder for an APPROVED invoice."
    )
    send_parser.add_argument("invoice_id", nargs="?", help="Invoice ID; prompts if omitted.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_console_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        _error(str(exc))
        return 1

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    try:
        _bootstrap_sql_ledger()
    except Exception as exc:  # noqa: BLE001 - the ledger degrades, the app must not
        _warning(f"SQL ledger bootstrap skipped: {exc}")
        _install_pii_filter()

    try:
        settings.ensure_directories()
    except OSError as exc:
        _error(f"Could not create data directories under {settings.data_dir}: {exc}")
        return 1

    store = InvoiceStore(settings.invoice_store_path)

    try:
        if args.command is None:
            if not sys.stdin.isatty():
                parser.print_help()
                return 2
            return _interactive_menu(settings, store)

        if args.command == "create":
            if args.voice and args.prompt:
                _error("Use either a text prompt or --voice, not both.")
                return 1
            if not settings.is_business_configured:
                _warning(
                    "BUSINESS_NAME/BUSINESS_TAX_ID are incomplete. "
                    "Set them in .env before UJP validation."
                )
            result, created = _create_invoice_flow(
                settings,
                store,
                prompt_text=args.prompt,
                voice_path=args.voice,
                on_created=_sync_legacy_invoice_to_sql,
            )
            if result == 0 and created and args.collect:
                return _run_collections(settings=settings, store=store)
            return result

        if args.command == "collect":
            as_of = _parse_as_of(args.as_of, settings)
            return _run_collections(
                settings=settings,
                store=store,
                excel_path=args.excel,
                as_of=as_of,
                limit=args.limit,
            )

        if args.command == "reply":
            return _run_reply_flow(args.text)

        if args.command == "import-excel":
            return _run_import_excel(store, args.path)

        if args.command == "export-excel":
            return _run_export_excel(store, args.path)

        if args.command == "ubl":
            return _run_ubl_export(
                settings,
                store,
                args.invoice_id,
                output_dir=args.output_dir,
                show_xml=args.print_xml,
            )

        if args.command == "list":
            as_of = _parse_as_of(args.as_of, settings)
            return _print_invoice_list(
                store,
                as_of=as_of,
                overdue_only=args.overdue,
                limit=args.limit,
            )

        if args.command == "approve":
            return _run_approve_invoice_flow(settings, args.invoice_id)

        if args.command == "send":
            return _run_send_reminder_flow(settings, args.invoice_id)

        parser.print_help()
        return 2

    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130
    except EOFError:
        print("\nInput closed.")
        return 130
    except (ConfigError, InvoiceDataError, FileNotFoundError, ValueError, OSError) as exc:
        _error(str(exc))
        return 1
    except Exception as exc:  # Keep the CLI professional while allowing --debug.
        if args.debug:
            logger.exception("Unexpected error")
        _error(f"Unexpected error: {exc}. Re-run with --debug for details.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
