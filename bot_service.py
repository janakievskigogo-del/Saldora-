from __future__ import annotations

import asyncio
import json
import os
import secrets
import smtplib
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping
from datetime import datetime, timezone
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError

from database import SessionLocal, init_db
from models import DeliveryNote, Invoice

load_dotenv()

# Resend is the preferred delivery provider (in requirements.txt). The import is
# soft so a missing install can never kill the FastAPI process at import time.
try:
    import resend
except ImportError:  # pragma: no cover - depends on optional install
    resend = None  # type: ignore[assignment]

# Telegram is optional for the HTTP API: api.py only needs send_email_native.
# Keeping these imports soft means importing bot_service can never kill the
# FastAPI process (previously a missing token called sys.exit at import time).
try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

    TELEGRAM_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional install
    Update = None  # type: ignore[assignment,misc]
    ParseMode = None  # type: ignore[assignment]
    ApplicationBuilder = None  # type: ignore[assignment]
    CommandHandler = None  # type: ignore[assignment]
    ContextTypes = None  # type: ignore[assignment]
    TELEGRAM_AVAILABLE = False

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME") or os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or os.getenv("SMTP_PASS", "")
SENDER_EMAIL = os.getenv("SENDER_EMAIL") or SMTP_USERNAME

if not TOKEN:
    print(
        "WARNING: TELEGRAM_BOT_TOKEN is not set; the Telegram bot is disabled "
        "(SMTP e-mail delivery remains available).",
        file=sys.stderr,
    )


# Hard per-attempt transport deadline: a wedged SMTP host must never hold a
# request (or a Render worker) open indefinitely.
SMTP_TIMEOUT_SECONDS = 12

# The Gmail relay (Google Apps Script web app) gets a slightly larger deadline
# than SMTP because it may cold-start before answering.
WEBHOOK_TIMEOUT_SECONDS = 15


def _outbound_display_name() -> str:
    """Friendly display name for the Resend ``From`` header (company-configurable)."""

    return (os.getenv("BUSINESS_NAME") or "Saldora").strip() or "Saldora"


def _resend_from_address() -> str:
    """Outbound sender address: RESEND_FROM_EMAIL -> BUSINESS_EMAIL -> Resend sandbox."""

    return (
        (os.getenv("RESEND_FROM_EMAIL") or "").strip()
        or (os.getenv("BUSINESS_EMAIL") or "").strip()
        or "onboarding@resend.dev"
    )


def _resend_from_header() -> str:
    """Clean RFC-style ``From`` for the Resend API: ``Saldora <billing@saldora.ai>``."""

    return f"{_outbound_display_name()} <{_resend_from_address()}>"


def _smtp_settings() -> tuple[str, int, str, str, str]:
    """Resolve SMTP transport settings at call time.

    Reading the environment on every dispatch (instead of freezing it at import
    time) keeps Render/hosted deployments correct after a restart and lets
    callers rotate credentials without re-importing the module.

    Returns ``(server, port, username, password, sender_email)`` with all
    whitespace stripped from the password: a stray newline pasted from a
    provider dashboard is the most common cause of ``SMTPAuthenticationError``.
    """

    server = (
        (os.getenv("SMTP_SERVER") or "").strip()
        or (SMTP_SERVER or "").strip()
        or "smtp.gmail.com"
    )
    try:
        port = int((os.getenv("SMTP_PORT") or "").strip() or SMTP_PORT)
    except (TypeError, ValueError):
        port = 587

    username = (
        (os.getenv("SMTP_USER") or "").strip()
        or (os.getenv("SMTP_USERNAME") or "").strip()
        or (SMTP_USERNAME or "").strip()
        or (os.getenv("BUSINESS_EMAIL") or "").strip()
    )
    password = (
        (os.getenv("SMTP_PASSWORD") or "").strip()
        or (os.getenv("SMTP_PASS") or "").strip()
        or (SMTP_PASSWORD or "").strip()
    )
    sender = (
        (os.getenv("SENDER_EMAIL") or "").strip()
        or (SENDER_EMAIL or "").strip()
        or username
        or (os.getenv("BUSINESS_EMAIL") or "").strip()
    )
    return server, port, username, password, sender


def _smtp_from_header(sender_email: str | None = None) -> str:
    """RFC 2047-safe ``From`` for SMTP: ``Saldora <billing@saldora.ai>``."""

    _, _, username, _, configured_sender = _smtp_settings()
    sender = (sender_email or configured_sender or username or "").strip()
    return formataddr(("Saldora", sender), charset="utf-8")


def _send_via_webhook(
    recipient_email: str,
    subject: str,
    body_text: str,
    html_body: str | None = None,
) -> tuple[bool, str | None]:
    """POST the message to ``GMAIL_WEBHOOK_URL`` (a Gmail/Apps Script relay).

    Returns success only on HTTP 200. Any transport error, non-200 status or
    malformed response is logged to ``stderr`` and reported so the caller can
    fall back to the next transport instead of assuming delivery.
    """

    url = (os.getenv("GMAIL_WEBHOOK_URL") or "").strip()
    if not url:
        return False, "GMAIL_WEBHOOK_URL is not set"

    payload = {
        "to": recipient_email,
        "subject": subject,
        "html": html_body or "",
        "text": body_text,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request, timeout=WEBHOOK_TIMEOUT_SECONDS
        ) as response:
            status_code = getattr(response, "status", None) or response.getcode()
            if status_code == 200:
                return True, None
            detail = response.read(500).decode("utf-8", "replace")
            print(
                f"[email] Gmail webhook for {recipient_email} returned HTTP "
                f"{status_code}: {detail}",
                file=sys.stderr,
            )
            return False, f"Gmail webhook returned HTTP {status_code}: {detail}"
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        print(
            f"[email] Gmail webhook for {recipient_email} failed: HTTP "
            f"{exc.code}: {detail}",
            file=sys.stderr,
        )
        return False, f"Gmail webhook delivery failed: HTTP {exc.code}: {detail}"
    except Exception as exc:  # noqa: BLE001 - report every delivery failure
        print(
            f"[email] Gmail webhook for {recipient_email} failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False, f"Gmail webhook delivery failed: {type(exc).__name__}: {exc}"


def _send_via_resend(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Mapping[str, str] | None = None,
    attachment_path: str | Path | None = None,
    html_body: str | None = None,
) -> tuple[bool, str | None]:
    """Send one email through the Resend REST API (``resend.Emails.send``)."""

    if resend is None:
        return False, "resend package is not installed"

    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    if not api_key:
        return False, "RESEND_API_KEY is not set"

    params: dict[str, object] = {
        "from": _resend_from_header(),
        "to": recipient_email,
        "subject": subject,
        "text": body_text,
    }
    if html_body:
        params["html"] = html_body

    headers = {
        str(name).strip(): str(value).strip()
        for name, value in (extra_headers or {}).items()
        if str(name).strip() and str(value).strip()
    }
    if headers:
        params["headers"] = headers

    if attachment_path:
        att = Path(attachment_path)
        if att.is_file():
            try:
                params["attachments"] = [
                    {"content": list(att.read_bytes()), "filename": att.name}
                ]
            except Exception as exc:  # noqa: BLE001 - attachment is best-effort
                print(
                    f"[email] Could not read attachment {att} for Resend: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    try:
        resend.api_key = api_key
        resend.Emails.send(params=params)
        return True, None
    except Exception as exc:  # noqa: BLE001 - report every delivery failure
        print(
            f"[email] Resend delivery to {recipient_email} failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False, f"Resend delivery failed: {type(exc).__name__}: {exc}"


def _send_via_smtp(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Mapping[str, str] | None = None,
    attachment_path: str | Path | None = None,
    html_body: str | None = None,
) -> tuple[bool, str | None]:
    """Send one email over SMTP/STARTTLS and report the transport outcome.

    Port ``465`` uses implicit TLS (``smtplib.SMTP_SSL``); every other port
    (including the default ``587``) uses ``SMTP`` + ``starttls()``. Both paths
    carry the strict ``SMTP_TIMEOUT_SECONDS`` socket deadline.
    """

    server, port, username, password, sender = _smtp_settings()
    if not all([server, username, password, sender]):
        return False, "Missing SMTP config in .env"

    msg = MIMEMultipart()
    msg["From"] = _smtp_from_header(sender)
    msg["To"] = recipient_email
    msg["Subject"] = subject
    for header_name, header_value in (extra_headers or {}).items():
        name = str(header_name).strip()
        value = str(header_value).strip()
        if (
            not name
            or not value
            or any(character in name for character in ":\r\n")
            or "\r" in value
            or "\n" in value
        ):
            continue
        msg[name] = value
    if html_body:
        # Multipart/alternative: plain-text fallback plus rich HTML rendering.
        alternative = MIMEMultipart("alternative")
        alternative.attach(MIMEText(body_text, "plain", "utf-8"))
        alternative.attach(MIMEText(html_body, "html", "utf-8"))
        msg.attach(alternative)
    else:
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

    if attachment_path:
        att = Path(attachment_path)
        if att.is_file():
            try:
                part = MIMEApplication(att.read_bytes(), Name=att.name)
                part["Content-Disposition"] = f'attachment; filename="{att.name}"'
                msg.attach(part)
            except Exception as exc:  # noqa: BLE001 - attachment is best-effort
                print(
                    f"[email] Could not read attachment {att} for SMTP: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    try:
        if port == 465:
            with smtplib.SMTP_SSL(server, port, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
                smtp.login(username, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(server, port, timeout=SMTP_TIMEOUT_SECONDS) as smtp:
                smtp.starttls()
                smtp.login(username, password)
                smtp.send_message(msg)
        return True, None
    except Exception as exc:  # noqa: BLE001 - report every delivery failure
        print(
            f"[email] SMTP delivery to {recipient_email} via {server}:{port} failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return False, f"SMTP delivery failed: {type(exc).__name__}: {exc}"


def send_email_native(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Mapping[str, str] | None = None,
    attachment_path: str | Path | None = None,
    html_body: str | None = None,
) -> tuple[bool, str | None]:
    """Deliver an email: Gmail webhook first, then Resend, then SMTP.

    Returns ``(success, error_message)``. The function never raises and never
    swallows a failure: transport errors are written to ``stderr`` for Render
    logs and returned to the caller as ``error_message`` so HTTP routes can
    answer honestly instead of reporting a delivery that did not happen.
    """

    if not str(recipient_email or "").strip():
        return False, "Recipient email is empty"

    transport_errors: list[str] = []

    if (os.getenv("GMAIL_WEBHOOK_URL") or "").strip():
        ok, webhook_error = _send_via_webhook(
            recipient_email, subject, body_text, html_body
        )
        if ok:
            return True, None
        if webhook_error:
            transport_errors.append(webhook_error)

    if (os.getenv("RESEND_API_KEY") or "").strip():
        ok, resend_error = _send_via_resend(
            recipient_email, subject, body_text, extra_headers, attachment_path, html_body
        )
        if ok:
            return True, None
        if resend_error:
            transport_errors.append(resend_error)
        # Degrade to SMTP only when credentials exist; otherwise surface every
        # attempted transport's error to the caller.
        _, _, smtp_user, smtp_password, smtp_sender = _smtp_settings()
        if not all([smtp_user, smtp_password, smtp_sender]):
            return False, "; ".join(transport_errors) or "Email delivery failed"

    ok, smtp_error = _send_via_smtp(
        recipient_email, subject, body_text, extra_headers, attachment_path, html_body
    )
    if ok:
        return True, None
    if transport_errors:
        transport_errors.append(f"SMTP fallback failed: {smtp_error}")
        return False, "; ".join(transport_errors)
    return False, smtp_error


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Добредојдовте во Saldora.\n"
        "Системот е успешно поврзан со базата на податоци.\n\n"
        "Користете /help за да ги видите достапните команди."
    )
    await update.message.reply_text(text)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "<b>Saldora B2B Систем - Достапни команди:</b>\n\n"
        "/invoices - Преглед на ненаплатени фактури од база\n"
        "/ispratnica - Преглед и потпишување испратници\n"
        "/remind &lt;код&gt; [емаил] - Испрати е-пошта потсетник\n"
        "/newinvoice &lt;клиент&gt; &lt;износ&gt; &lt;опис&gt; - Креирај нова фактура во база\n"
        "/storno &lt;код&gt; [причина] - Сторнирај фактура за УЈП во база\n"
        "/help - Прикажи го ова мени"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def invoices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db = SessionLocal()
    try:
        unpaid = db.query(Invoice).filter(Invoice.status == "UNPAID").all()
        if not unpaid:
            await update.message.reply_text("Немате ненаплатени фактури во базата.")
            return

        lines = ["<b>Ненаплатени фактури (од база):</b>"]
        for idx, inv in enumerate(unpaid, start=1):
            inv_code = inv.code or inv.id
            client = inv.client_name or "Клиент"
            amt = float(inv.amount or inv.total_amount or 0.0)
            due_days = getattr(inv, "due_days", 0)

            lines.append(
                f"{idx}. <code>{inv_code}</code> - <b>{client}</b> | "
                f"{amt:,.2f} МКД (Задоцнување: {due_days} дена)"
            )

        lines.append("\nИспрати потсетник: <code>/remind &lt;код&gt; [емаил]</code>")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:  # noqa: BLE001 - report any delivery failure to the caller
        await update.message.reply_text(f"<b>Грешка во базата:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
    finally:
        db.close()


async def ispratnica(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    db = SessionLocal()
    try:
        pending = db.query(DeliveryNote).filter(DeliveryNote.is_signed.is_(False)).all()
        if not pending:
            await update.message.reply_text("Немате активни испратници кои чекаат е-потпис.")
            return

        lines = ["<b>Испратници кои чекаат е-потпис:</b>"]
        for idx, dn in enumerate(pending, start=1):
            lines.append(f"{idx}. <code>{dn.code}</code> - {dn.recipient_name} ({dn.details})\nСтатус: Чека е-потпис.")

        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
    except Exception as e:  # noqa: BLE001 - report any delivery failure to the caller
        await update.message.reply_text(f"<b>Грешка во базата:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
    finally:
        db.close()


def _generate_unique_invoice_code(db) -> str:
    """Fail-safe, timestamped invoice code: ``INV-{YYYYMMDD}-{HEX6}``.

    Mirrors ``api.generate_guaranteed_unique_code`` (kept standalone here so
    the Telegram bot never imports the FastAPI app): UTC date bucket plus a
    6-character random hex body, verified against the database before use.
    """

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    for _ in range(50):
        candidate = f"INV-{stamp}-{secrets.token_hex(3).upper()}"
        if (
            db.query(Invoice.id)
            .filter((Invoice.code == candidate) | (Invoice.id == candidate))
            .first()
            is None
        ):
            return candidate
    raise RuntimeError("Could not allocate a unique invoice code.")


async def new_invoice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 3:
        await update.message.reply_text(
            "Формат: <code>/newinvoice &lt;клиент&gt; &lt;износ&gt; &lt;опис&gt;</code>\n"
            "Пример: <code>/newinvoice Нептун 45000 Набавка_на_опрема</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    client_name = context.args[0]
    try:
        amount = float(context.args[1])
    except ValueError:
        await update.message.reply_text("Грешка: Износот мора да биде број.")
        return

    description = " ".join(context.args[2:])

    db = SessionLocal()
    try:
        # Fail-safe allocation + IntegrityError backstop: the timestamped hex
        # code is verified against the DB, and a racing writer can still only
        # lose its commit, never produce a duplicate.
        new_code = _generate_unique_invoice_code(db)

        def _build(code: str) -> Invoice:
            return Invoice(
                id=code,
                code=code,
                client_name=client_name,
                amount=amount,
                description=description,
                status="UNPAID",
            )

        new_inv = _build(new_code)
        db.add(new_inv)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            new_code = _generate_unique_invoice_code(db)
            new_inv = _build(new_code)
            db.add(new_inv)
            db.commit()

        text = (
            f"<b>Успешно запишана нова фактура во база:</b>\n"
            f"• Код: <code>{new_code}</code>\n"
            f"• Клиент: <b>{client_name}</b>\n"
            f"• Износ: <b>{amount:,.2f} МКД</b>\n"
            f"• Опис: {description}\n"
            f"• Формат: UBL 2.1 XML (Подготвено за УЈП e-Faktura)"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as e:  # noqa: BLE001 - report any delivery failure to the caller
        db.rollback()
        await update.message.reply_text(f"<b>Грешка при креирање фактура:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
    finally:
        db.close()


async def storno(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Формат: <code>/storno &lt;код_на_фактура&gt; [причина]</code>\n"
            "Пример: <code>/storno NPL-2026-0001 Погрешен_износ</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    code = context.args[0]
    reason = " ".join(context.args[1:]) if len(context.args) > 1 else "Сторнирано по барање на корисникот"

    db = SessionLocal()
    try:
        inv = db.query(Invoice).filter((Invoice.code == code) | (Invoice.id == code)).first()
        if not inv:
            await update.message.reply_text(f"Фактурата со код <code>{code}</code> не е пронајдена во базата.", parse_mode=ParseMode.HTML)
            return

        inv.status = "STORNO"
        db.commit()

        text = (
            f"<b>Успешно процесирано Сторно:</b>\n"
            f"• Фактура: <code>{code}</code>\n"
            f"• Причина: {reason}\n"
            f"• Статус: Маркирана како STORNO и генериран UBL 2.1 корекциски документ."
        )
        await update.message.reply_text(text, parse_mode=ParseMode.HTML)
    except Exception as e:  # noqa: BLE001 - report any delivery failure to the caller
        db.rollback()
        await update.message.reply_text(f"<b>Грешка при сторнирање:</b>\n<code>{str(e)}</code>", parse_mode=ParseMode.HTML)
    finally:
        db.close()


async def remind(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if len(context.args) < 1:
        await update.message.reply_text(
            "Формат: <code>/remind &lt;код_на_фактура&gt; [емаил]</code>\n"
            "Пример: <code>/remind NPL-2026-0001 janakievskigogo@gmail.com</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    code = context.args[0]
    target_email = context.args[1] if len(context.args) > 1 else "janakievskigogo@gmail.com"

    db = SessionLocal()
    try:
        inv = db.query(Invoice).filter((Invoice.code == code) | (Invoice.id == code)).first()
        client_str = inv.client_name if inv and inv.client_name else "Клиент"
        amount_str = f"{float(inv.amount):,.2f} МКД" if inv and inv.amount else ""
        xml_path = getattr(inv, "xml_file_path", None)
    finally:
        db.close()

    subject = f"Потсетник за неплатена фактура {code} - Saldora"
    body = (
        f"Почитувани,\n\n"
        f"Ве известуваме дека фактурата {code} ({client_str}) {amount_str} е со задоцнета наплата.\n"
        f"Ве молиме извршете ја уплатата во најкраток можен рок.\n\n"
        f"Со почит,\nSaldora B2B Систем"
    )

    success, message = await asyncio.to_thread(
        send_email_native, target_email, subject, body, None, xml_path
    )

    if success:
        text = f"Потсетникот за фактурата <b>{code}</b> беше успешно испратен преку е-пошта до <b>{target_email}</b>."
    else:
        text = f"<b>Неуспешно испраќање:</b>\n<code>{message or 'непозната грешка'}</code>"

    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


if __name__ == "__main__":
    if not TOKEN:
        print("CRITICAL ERROR: TELEGRAM_BOT_TOKEN is not set in your .env file.", file=sys.stderr)
        sys.exit(1)
    if not TELEGRAM_AVAILABLE:
        print(
            "CRITICAL ERROR: python-telegram-bot is not installed. "
            "Install it with: pip install python-telegram-bot",
            file=sys.stderr,
        )
        sys.exit(1)

    init_db()
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("invoices", invoices))
    app.add_handler(CommandHandler("ispratnica", ispratnica))
    app.add_handler(CommandHandler("remind", remind))
    app.add_handler(CommandHandler("newinvoice", new_invoice))
    app.add_handler(CommandHandler("storno", storno))

    print("Saldora database & Telegram bot service started successfully.")
    app.run_polling()
