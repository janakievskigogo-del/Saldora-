"""PDF rendering for e-Ispratnica documents sent to Documenso.

The generated PDF is a clean, A4 delivery note containing the same data shown
in the in-app viewer. A Unicode font is used when one is available so the
Macedonian Cyrillic text is preserved; otherwise a Latin transliteration keeps
the document readable on systems without an embeddable TrueType font.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    # Linux containers (Render/Docker): DejaVu ships with most base images.
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    # macOS development fallbacks.
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
)


def _ensure_windows_fonts() -> None:
    """Add Windows font fallbacks derived from %WINDIR% (dev only)."""

    global _FONT_CANDIDATES
    windir = os.environ.get("WINDIR", "").strip()
    if not windir:
        return
    fonts = Path(windir) / "Fonts"
    _FONT_CANDIDATES = _FONT_CANDIDATES + (
        (str(fonts / "DejaVuSans.ttf"), str(fonts / "DejaVuSans-Bold.ttf")),
        (str(fonts / "arial.ttf"), str(fonts / "arialbd.ttf")),
    )


_ensure_windows_fonts()


_MK_TRANSLITERATION = str.maketrans(
    {
        "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Ѓ": "Gj",
        "Е": "E", "Ж": "Zh", "З": "Z", "Ѕ": "Dz", "И": "I", "Ј": "J",
        "К": "K", "Л": "L", "Љ": "Lj", "М": "M", "Н": "N", "Њ": "Nj",
        "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "Ќ": "Kj",
        "У": "U", "Ф": "F", "Х": "H", "Ц": "C", "Ч": "Ch", "Џ": "Dzh",
        "Ш": "Sh",
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ѓ": "gj",
        "е": "e", "ж": "zh", "з": "z", "ѕ": "dz", "и": "i", "ј": "j",
        "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ќ": "kj",
        "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "џ": "dzh",
        "ш": "sh",
    }
)


@lru_cache(maxsize=1)
def _register_fonts() -> tuple[str, str, bool]:
    """Register a Cyrillic-capable font pair when one is available."""

    env_font = os.environ.get("DOCUMENSO_PDF_FONT", "").strip()
    if env_font and Path(env_font).is_file():
        pdfmetrics.registerFont(TTFont("SaldoraSans", env_font))
        bold_path = os.environ.get("DOCUMENSO_PDF_FONT_BOLD", "").strip()
        if bold_path and Path(bold_path).is_file():
            pdfmetrics.registerFont(TTFont("SaldoraSans-Bold", bold_path))
            return "SaldoraSans", "SaldoraSans-Bold", True
        return "SaldoraSans", "SaldoraSans", True

    for regular, bold in _FONT_CANDIDATES:
        if not Path(regular).is_file():
            continue
        pdfmetrics.registerFont(TTFont("SaldoraSans", regular))
        if Path(bold).is_file():
            pdfmetrics.registerFont(TTFont("SaldoraSans-Bold", bold))
            return "SaldoraSans", "SaldoraSans-Bold", True
        return "SaldoraSans", "SaldoraSans", True

    # No embeddable Unicode font: Helvetica plus transliteration below.
    return "Helvetica", "Helvetica-Bold", False


def _text(value: Any, *, unicode_font: bool) -> str:
    cleaned = "" if value is None else str(value)
    if not unicode_font:
        cleaned = cleaned.translate(_MK_TRANSLITERATION)
        cleaned = cleaned.encode("ascii", "replace").decode("ascii")
    return escape(cleaned)


def _signature_flowable(signature_data: str | None) -> Image | None:
    if not signature_data:
        return None

    payload = str(signature_data).strip()
    if "," in payload and payload.lower().startswith("data:image/"):
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        return None
    if not raw:
        return None

    try:
        image = Image(BytesIO(raw))
    except (OSError, TypeError, ValueError):
        return None

    width, height = image.imageWidth, image.imageHeight
    if not width or not height:
        return None
    target_width = 55 * mm
    target_height = target_width * (height / width)
    if target_height > 28 * mm:
        target_height = 28 * mm
        target_width = target_height * (width / height)
    image.drawWidth = target_width
    image.drawHeight = target_height
    return image


def build_delivery_note_pdf(note: Any) -> bytes:
    """Render ``note`` (a ``DeliveryNote`` ORM instance) as PDF bytes."""

    regular_font, bold_font, unicode_font = _register_fonts()

    def txt(value: Any) -> str:
        return _text(value, unicode_font=unicode_font)

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"е-Испратница {getattr(note, 'code', '')}",
        author="Saldora",
    )

    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SaldoraTitle",
        parent=base_styles["Title"],
        fontName=bold_font,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F5257"),
        alignment=0,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "SaldoraSubtitle",
        parent=base_styles["Normal"],
        fontName=regular_font,
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#50605A"),
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "SaldoraHeading",
        parent=base_styles["Heading3"],
        fontName=bold_font,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#50605A"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "SaldoraBody",
        parent=base_styles["Normal"],
        fontName=regular_font,
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#121212"),
    )
    small_style = ParagraphStyle(
        "SaldoraSmall",
        parent=base_styles["Normal"],
        fontName=regular_font,
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#50605A"),
    )
    right_style = ParagraphStyle(
        "SaldoraRight",
        parent=body_style,
        alignment=TA_RIGHT,
    )

    code = getattr(note, "code", "—") or "—"
    issue_date = getattr(note, "issue_date", None)
    client_name = (
        getattr(note, "client_name", None)
        or getattr(note, "recipient_name", None)
        or "—"
    )
    address = getattr(note, "delivery_address", None) or "—"
    invoice_id = getattr(note, "invoice_id", None)
    items = (
        getattr(note, "items_description", None)
        or getattr(note, "details", None)
        or ""
    )

    story: list[Any] = [
        Paragraph("Saldora", title_style),
        Paragraph("Официјална е-Испратница / Delivery Note", subtitle_style),
        Paragraph(f"Број: <b>{txt(code)}</b>", body_style),
        Paragraph(
            f"Датум на издавање: {txt(issue_date) if issue_date else '—'}",
            body_style,
        ),
    ]
    if invoice_id:
        story.append(Paragraph(f"Поврзана фактура: {txt(invoice_id)}", body_style))
    story.append(Spacer(1, 6 * mm))

    details_rows = [
        [
            Paragraph("Клиент / Примач", heading_style),
            Paragraph("Адреса на достава", heading_style),
        ],
        [Paragraph(txt(client_name), body_style), Paragraph(txt(address), body_style)],
    ]
    details = Table(details_rows, colWidths=[87 * mm, 87 * mm])
    details.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#50605A")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5DCD8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F6F5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([details, Spacer(1, 7 * mm)])

    story.append(Paragraph("Спецификација на артикли", heading_style))
    item_rows: list[list[Any]] = [
        [
            Paragraph("#", heading_style),
            Paragraph("Опис на артикли / услуги", heading_style),
            Paragraph("Количина", heading_style),
        ]
    ]

    lines = [
        line.strip()
        for line in str(items).replace(";", "\n").splitlines()
        if line.strip()
    ]
    for index, line in enumerate(lines, start=1):
        quantity = "—"
        description = line
        parts = line.split(None, 1)
        if len(parts) == 2:
            candidate = parts[0].replace(",", ".")
            try:
                float(candidate)
            except ValueError:
                pass
            else:
                quantity = parts[0]
                description = parts[1].strip()
        item_rows.append(
            [
                Paragraph(str(index), small_style),
                Paragraph(txt(description), body_style),
                Paragraph(txt(quantity), right_style),
            ]
        )

    if len(item_rows) == 1:
        item_rows.append(
            [Paragraph("—", small_style), Paragraph("Нема внесена спецификација.", body_style), Paragraph("—", right_style)]
        )

    items_table = Table(item_rows, colWidths=[10 * mm, 139 * mm, 25 * mm], repeatRows=1)
    items_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#50605A")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DCD8")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4F6F5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([items_table, Spacer(1, 8 * mm)])

    story.append(Paragraph("Дигитален потпис на примачот", heading_style))
    signature = _signature_flowable(getattr(note, "signature_data", None))
    if signature is not None:
        signature_box = Table([[signature]], colWidths=[174 * mm])
        signature_box.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#50605A")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(signature_box)
    else:
        story.append(
            Paragraph(
                "Нема достапен визуелен потпис. Потписот се собира преку Documenso.",
                body_style,
            )
        )

    signed_at = getattr(note, "signed_at", None)
    if signed_at:
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(f"Временска ознака: {txt(signed_at)}", small_style))

    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "Овој документ е генериран автоматски од Saldora Executive Financial "
            "Vault. Дигиталниот потпис и аудиторската трага се составен дел од "
            "е-Испратницата.",
            small_style,
        )
    )

    document.build(story)
    return buffer.getvalue()
