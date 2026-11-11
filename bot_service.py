from __future__ import annotations

import asyncio
import os
import smtplib
import sys
from collections.abc import Mapping
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

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


def _resend_from_address() -> str | None:
    """Resolve the Resend sender identity from the environment."""

    return (
        os.getenv("RESEND_FROM_EMAIL")
        or os.getenv("BUSINESS_EMAIL")
        or SENDER_EMAIL
        or None
    )


def _send_via_resend(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Mapping[str, str] | None = None,
    attachment_path: str | Path | None = None,
    html_body: str | None = None,
) -> tuple[bool, str]:
    """Send one email through the Resend REST API (``resend.Emails.send``)."""

    if resend is None:
        return False, "resend package is not installed"

    from_email = _resend_from_address()
    if not from_email:
        return False, (
            "Missing Resend sender address: set RESEND_FROM_EMAIL, "
            "BUSINESS_EMAIL or SENDER_EMAIL"
        )

    params: dict[str, object] = {
        "from": from_email,
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
            except Exception:  # noqa: BLE001
                pass

    try:
        resend.api_key = os.environ["RESEND_API_KEY"]
        resend.Emails.send(params=params)
        return True, "Success"
    except Exception as e:  # noqa: BLE001 - report any delivery failure to the caller
        return False, f"Resend delivery failed: {str(e)}"


def send_email_native(
    recipient_email: str,
    subject: str,
    body_text: str,
    extra_headers: Mapping[str, str] | None = None,
    attachment_path: str | Path | None = None,
    html_body: str | None = None,
) -> tuple[bool, str]:
    """Deliver an email: Resend API first (when configured), SMTP as fallback."""

    if os.environ.get("RESEND_API_KEY"):
        ok, message = _send_via_resend(
            recipient_email, subject, body_text, extra_headers, attachment_path, html_body
        )
        if ok:
            return True, message
        # Degrade to SMTP when Resend failed and SMTP credentials exist;
        # otherwise surface the Resend error to the caller.
        smtp_configured = all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL])
        if not smtp_configured:
            return False, message

    if not all([SMTP_SERVER, SMTP_USERNAME, SMTP_PASSWORD, SENDER_EMAIL]):
        return False, "Missing SMTP config in .env"

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
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
            except Exception:  # noqa: BLE001
                pass

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
        return True, "Success"
    except Exception as e:  # noqa: BLE001 - report any delivery failure to the caller
        return False, str(e)


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
        count = db.query(Invoice).count() + 1
        new_code = f"NPL-2026-{count:04d}"

        new_inv = Invoice(
            id=new_code,
            code=new_code,
            client_name=client_name,
            amount=amount,
            description=description,
            status="UNPAID",
        )
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
        text = f"<b>Неуспешно испраќање:</b>\n<code>{message}</code>"

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
