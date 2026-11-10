# Saldora + FakturaBot.mk

A local, human-in-the-loop MVP for Macedonian small businesses:

- **FakturaBot.mk** turns a text or voice request into a structured invoice and a locally generated mock UJP e-Faktura UBL 2.1 document.
- **Saldora** drafts polite payment reminders in Macedonian, Albanian, or English, escalates the tone by days overdue, and requires explicit human approval before every mock send.

The project is deliberately modular so the deterministic services can be reused by a future FastAPI backend and React frontend.

> **Important:** UJP submission and message delivery are **mock-only** in this MVP. No network calls are made to UJP, email, Viber, or WhatsApp. See [Mock boundaries](#mock-boundaries).

---

## Features

| Area | Implemented in this MVP |
|---|---|
| Invoice intake | Text prompts and Groq Whisper-transcribed voice notes |
| Extraction | Client name, ЕДБ/ЕМБГ/NIPT tax ID, amount, VAT, currency, issue/due dates, language, channel, contact details |
| Validation | Deterministic structural and arithmetic checks before persistence |
| UJP e-Faktura | Local UBL 2.1 XML generation, local structural/math validation, validated `output/` audit write, mock reference |
| Collections | Friendly (1–7 days), Standard (8–20), Firm (21+); complete drafts with Groq model fallback |
| Groq resilience | `llama-3.3-70b-versatile` → `llama-3.1-8b-instant` on 429/5xx/API errors |
| Languages | Macedonian, Albanian, English |
| Reply handling | `paid`, `extension_request`, `dispute`, `unclear` |
| Storage | Atomic JSON audit store; Excel import/export; legacy "Amount"-only workbooks |
| Human-in-the-loop | Mandatory `[Y]` approval, `[E]` edit, `[S]` skip, `[Q]` quit before a send |
| Delivery | Console-rendered mock email, Viber, and WhatsApp adapters |
| Safety | No fabricated financial facts, no threats, no auto-marking an invoice PAID from a client reply |

---

## Requirements

- Python **3.10+** (tested with Python 3.12)
- A Groq API key for reminder drafting, invoice extraction, and voice transcription
- Windows, macOS, or Linux

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the runtime dependencies plus `pytest` and `ruff`.

---

## Configuration

Copy the documented template and fill in the real values:

```powershell
Copy-Item .env.example .env
```

```dotenv
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile

BUSINESS_NAME=Your Company DOOEL
BUSINESS_TAX_ID=4030000000000
BUSINESS_ADDRESS=Your Street 1
BUSINESS_CITY=Skopje
BUSINESS_COUNTRY=MK
BUSINESS_EMAIL=billing@yourcompany.mk
BUSINESS_PHONE=+38970000000
BUSINESS_IBAN=MK07200000000000000
SENDER_NAME=Your Name

DEFAULT_CURRENCY=MKD
DEFAULT_VAT_RATE=18
DEFAULT_PAYMENT_TERMS_DAYS=15
DEFAULT_AMOUNT_INCLUDES_VAT=false
DEFAULT_LANGUAGE=MK
DEFAULT_CHANNEL=Email
DEFAULT_SERVICE_DESCRIPTION=Professional services
INVOICE_PREFIX=NPL

DATA_DIR=data
TIMEZONE=Europe/Skopje
LOG_LEVEL=INFO
```

`GROQ_API_KEY` is the only accepted key. Placeholder values such as `gsk_...` or `your_groq_api_key_here` are treated as missing.

`GROQ_MODELS` is an ordered, comma-separated rotation. Draft generation starts with the first model and automatically tries each next model when Groq returns `429 RATE_LIMIT`, a 5xx response, or another API error. When `GROQ_MODELS` is unset, `GROQ_MODEL` becomes the first model and `llama-3.1-8b-instant` stays available as the default fallback. Voice notes use `GROQ_TRANSCRIPTION_MODEL` (default `whisper-large-v3`).

The `.env` file is ignored by Git. Never commit it or paste a key into source control.

---

## Web dashboard (Saldora Vault)

The bundled FastAPI app serves both the JSON API and `static/index.html`:

```powershell
python api.py
# equivalent: uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. The demo profile button signs in with
`admin@saldora.ai` / `demo123`; the account is seeded automatically on startup.

---

## Quick start

Show the interactive menu:

```powershell
python main.py
```

Create an invoice from text:

```powershell
python main.py create "Create an invoice for client Alfa DOOEL, tax ID 4030000000000, amount 10.000,00 MKD without VAT, due 15.11.2026, language Macedonian, email info@alfa.mk"
```

The CLI extracts what it can, asks only for missing required fields, shows a draft, and then asks you to confirm:

```text
[Y] Create and validate invoice   [E] Edit fields   [S] Cancel
```

Create from a voice note (requires `GROQ_API_KEY`):

```powershell
python main.py create --voice .\voice-notes\invoice-mk.m4a
```

Run collections for overdue, unpaid invoices:

```powershell
python main.py collect --as-of 2026-11-20 --limit 5
```

Classify a client reply:

```powershell
python main.py reply "I have already paid the invoice yesterday."
```

Generate, validate, and save a UBL 2.1 XML file for a stored invoice:

```powershell
python main.py ubl NPL-2026-0001
python main.py ubl NPL-2026-0001 --output-dir .\output --print
```

The default destination is `output/NPL-2026-0001.xml`. The command validates the invoice data and the generated XML before writing the file, then logs the exact local path.

Import and export Excel workbooks:

```powershell
python main.py import-excel .\invoices.xlsx
python main.py export-excel .\data\invoices_export.xlsx
```

List invoices:

```powershell
python main.py list
python main.py list --overdue --as-of 2026-11-20
```

Use `python main.py --debug ...` to show a traceback for an unexpected failure.

---

## Example prompts

**Macedonian**

```text
Направи фактура за клиент Алфа ДООЕЛ со ЕДБ 4030000000000,
износ 10.000,00 денари без ДДВ, рок до 15.11.2026,
јазик македонски, email info@alfa.mk
```

**Albanian**

```text
Krijo faturë për klientin Alba sh.p.k., NIPT L12345678A,
shuma 50.000,00 LEK, afati 20.12.2026, në shqip,
email info@alba.al
```

**English**

```text
Create an invoice for Acme LLC, tax ID MK4030000000000,
amount 1,500.00 EUR including VAT 18%, due 2026-11-15,
language English, email billing@acme.com
```

The parser does not silently invent missing financial data. Anything it cannot prove is left for the human to complete.

---

## Collections workflow

The tier is decided in Python from the number of calendar days past the due date:

| Days late | Tier | Tone |
|---:|---|---|
| 1–7 | Friendly | Warm, assumes an oversight, asks for confirmation or a date |
| 8–20 | Standard | Professional and direct, asks for payment or a specific date |
| 21+ | Firm | Serious and firm, asks for immediate payment or a call; no threats |

For each overdue invoice the CLI:

1. Ensures there is a valid email address or phone number.
2. Asks Gemini for a localized draft. The AI never decides the tier or changes financial facts.
3. Shows the exact draft, recipient, tier, language, and channel.
4. Waits for a human decision:
   - `[Y]` send the mock message;
   - `[E]` paste/edit the message (finish with `END` on its own line);
   - `[S]` skip this client;
   - `[Q]` quit the run.
5. After `[Y]`, updates `data/invoices.json` with `status: "REMINDER_SENT"`, increments the reminder count, and stamps `last_reminded_at` in the same atomic write.

A missing API key stops the run before any draft is generated. A missing email/phone asks for one and can be skipped. Draft generation tries the configured Gemini models in order, so a transient 503 or 429 on the first model does not abort the run. Output is capped generously at 2,048 tokens, and a truncated (`MAX_TOKENS`) response is rejected and retried on the next model rather than being shown to the user.

---

## Reply categories

`main.py reply` classifies inbound text into:

- `paid` — the client claims payment. This is advisory only: **verify the bank account before marking an invoice PAID.**
- `extension_request` — the client asks for more time, a payment plan, or a later date. Review and confirm any new terms in writing.
- `dispute` — the client disputes the invoice or says it was not received. Pause reminders and review it manually.
- `unclear` — no safe deterministic match, including ambiguous "paid but wrong" replies.

Negated statements such as "I haven't paid yet" are not classified as `paid`. Conflicting payment and dispute claims are returned as `unclear` (or `dispute` when the dispute is stronger) so a human can act.

---

## JSON and Excel storage

### JSON audit store

By default invoices are stored at `data/invoices.json`:

```json
{
  "schema_version": 1,
  "updated_at": "2026-10-30T20:00:00+00:00",
  "invoices": [
    {
      "invoice_id": "NPL-2026-0001",
      "client_name": "Acme LLC",
      "client_tax_id": "MK4030000000000",
      "net_amount": "1000.00",
      "vat_rate": "18.00",
      "vat_amount": "180.00",
      "total_amount": "1180.00",
      "currency": "MKD",
      "issue_date": "2026-10-30",
      "due_date": "2026-11-15",
      "status": "REMINDER_SENT",
      "language": "EN",
      "channel": "Email",
      "client_email": "billing@acme.com",
      "ujp_status": "MOCK_SUBMITTED",
      "ujp_reference": "MOCK-UJP-...",
      "ujp_xml_path": "output/NPL-2026-0001.xml",
      "reminder_count": 1,
      "last_reminded_at": "2026-10-31T09:15:00+00:00"
    }
  ]
}
```

The store is written atomically (temporary file + replace). A plain JSON list is also accepted for backward compatibility.

### Excel import/export

Export columns are:

`Invoice ID`, `Client Name`, `Tax ID`, `Issue Date`, `Due Date`, `Net Amount`, `VAT Rate`, `VAT Amount`, `Total Amount`, `Currency`, `Status`, `Preferred Language`, `Channel`, `Email`, `Phone`, `Description`, `UJP Status`, `UJP Reference`, `UJP XML Path`, `Reminder Count`, `Last Reminded At`.

For import, the required fields are:

- Invoice ID
- Client Name
- Amount (or Net Amount plus VAT columns)
- Due Date
- Status

Common aliases are accepted, including Macedonian, Albanian, and legacy English headers. A workbook with a single `Amount` column is treated as the payable total with zero VAT. Invalid rows are skipped with a warning rather than aborting the whole batch.

---

## Mock boundaries

### UJP e-Faktura

`ujp_service.py`:

- validates supplier/customer identity, dates, currency, and VAT arithmetic;
- generates a UBL 2.1-shaped invoice XML locally, including country-prefixed VAT IDs (BR-CO-09), one matching VAT category per invoice line (BR-CO-04/UBL-SR-48/BR-S-01), and a payee account for credit transfers (BR-61);
- validates the XML structure, totals, VAT categories, and payment account locally;
- writes the validated XML to `output/` (or an explicit `--output-dir`) and logs the exact file path;
- returns a `MOCK-UJP-...` reference.

Set `BUSINESS_IBAN` to the seller's bank account/IBAN. When omitted, the adapter falls back to `MK07200000000000000` so the mock document remains structurally valid.

It does **not** use official UJP XSD/Schematron rules, certificates, credentials, or network endpoints. A real integration requires the official UJP specifications, onboarding, certificate handling, retries/idempotency, and legal review.

### Message delivery

`dispatcher.py` prints an audit block to the console and returns a mock message ID. It does not send email, Viber, or WhatsApp messages. Replace the adapter functions with real provider clients when moving beyond the MVP.

---

## Voice prompts

Voice notes are transcribed by Groq Whisper and the transcript is passed to the same deterministic parser used for text. Supported extensions: `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm` (up to 18 MB). The audio bytes are sent to Groq for transcription; review your data-processing obligations before using real client audio.

---

## Project structure

```text
Saldora/
├── main.py                 # CLI, interactive menu, HITL collections orchestration
├── config.py               # .env loading, immutable Settings, validation
├── models.py               # Invoice, ParsedInvoice, enums, results
├── invoice_parser.py       # Text/voice-transcript extraction and VAT/date logic
├── ujp_service.py          # Mock UJP validation and UBL 2.1 generation
├── data_handler.py         # JSON InvoiceStore, Excel import/export
├── tone_agent.py           # Gemini reminder drafting and tier boundaries
├── reply_parser.py         # Multilingual reply classification
├── voice_transcriber.py    # Gemini audio transcription
├── dispatcher.py           # Console mock email/Viber/WhatsApp adapters
├── utils.py                # Money/date/contact parsing and VAT math
├── .env.example            # Documented configuration template
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Runtime + pytest
├── pyproject.toml          # Project metadata and pytest configuration
├── tests/                  # Offline pytest suite
├── output/                 # Generated UBL 2.1 XML audit files (git-ignored)
└── data/                   # Local JSON and Excel data (git-ignored)
```

---

## Testing

Run the offline suite:

```powershell
python -m pytest -q
```

The tests cover:

- money and date parsing;
- multilingual invoice extraction;
- VAT breakdowns and issue-date/due-date separation;
- UJP validation, UBL generation, tampered totals, validated XML export, and mock submission;
- JSON and Excel round-trips, legacy workbooks, and invalid rows;
- reply categories and ambiguous replies;
- deterministic tone tiers, mocked Gemini calls, exact model fallback order, token ceiling, and truncation fallback;
- console dispatcher behavior;
- the HITL approval/skip paths and configuration errors.

No test calls Gemini, UJP, or a messaging provider.

Run the linter with the project configuration:

```powershell
python -m ruff check .
```

---

## Privacy and production notes

- `.env`, `data/invoices.json`, `output/` XML files, and Excel exports are git-ignored by default.
- Invoice context is sent to Google when generating a draft or transcribing audio. Do not include secrets or unnecessary personal data in prompts.
- Client names, tax IDs, emails, phone numbers, and payment claims are personal/business data. Apply the retention and access rules that apply to your business.
- The reply classifier is advisory and must not be used to automatically close an invoice.
- Before production: add authentication/authorization, a real database, encrypted secret management, real UJP certification, provider integrations, rate limits, monitoring, backups, and an audit-retention policy.
- The mock UJP service is not a substitute for professional legal, tax, or accounting advice.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GROQ_API_KEY is missing or still a placeholder` | Fill the key in `.env` or set the environment variable. |
| `BUSINESS_NAME`/`BUSINESS_TAX_ID` error during create | Complete the business identity in `.env`; the mock UJP validator requires it. |
| Invoice creation says UJP validation failed | Check the client tax ID, business tax ID, dates, currency, and VAT arithmetic in the preview. |
| Excel import says required columns are missing | Use the canonical export headers or one of the documented aliases. |
| No overdue invoices | Check `--as-of`, the invoice due date, and `Status` (`PAID` invoices are excluded; `REMINDER_SENT` invoices remain collectible). |
| All Groq models failed | Check the API key and quota. The run reports every attempted model and its error; transient 429/5xx failures already fall through automatically. |
| UBL XML file location | Default is `output/<invoice_id>.xml`; use `python main.py ubl <invoice_id> --output-dir <path>` to override it. |
| Cyrillic/Albanian output looks wrong in an old console | Use Windows Terminal or PowerShell 7; the CLI requests UTF-8 output and replaces unsupported characters instead of crashing. |
| Voice note fails | Confirm `GROQ_API_KEY`, file format, and the 18 MB limit. |
| Timezone error | Use a valid IANA name such as `Europe/Skopje`. |

---

## Next steps

- Replace the mock UJP adapter with certified e-Faktura submission.
- Add real email/Viber/WhatsApp providers behind the existing dispatcher contract.
- Expose the services through FastAPI and add a React dashboard.
- Add authentication, a database, scheduled collection runs, and an approval audit trail.
- Add reCAPTCHA/rate limiting and a privacy policy for the public product.
