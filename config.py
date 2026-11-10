"""Centralized configuration for FakturaBot.mk / Saldora.

All environment access is performed here so service modules receive an explicit
``Settings`` object.  This makes tests deterministic and allows a future FastAPI
deployment to use the same services with injected configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

from legacy_models import Channel, Language
from utils import normalize_channel, normalize_language_code, parse_decimal

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# Groq's OpenAI-compatible API endpoint.  Both the reminder agent and the
# invoice parser use the OpenAI SDK pointed at this base URL.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Active Groq chat models, in the exact rotation order used by the reminder
# agent.  A failed model (429/5xx/API error) falls through to the next.
GROQ_MODEL_SEQUENCE: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)
DEFAULT_GROQ_MODEL = GROQ_MODEL_SEQUENCE[0]
DEFAULT_GROQ_TRANSCRIPTION_MODEL = "whisper-large-v3"

# Fallback seller account used for UBL PaymentMeansCode 30 (credit transfer).
# Override with BUSINESS_IBAN in .env to emit the real bank account.
DEFAULT_BUSINESS_IBAN = "MK07200000000000000"

_PLACEHOLDER_KEYS = {
    "",
    "replace_me",
    "replace-me",
    "your_api_key_here",
    "your_groq_api_key_here",
    "your_groq_key_here",
    "replace_with_your_groq_api_key",
    "gsk_...",
    "gsk_your_groq_api_key",
    "gsk_your_key_here",
    "changeme",
    "change-me",
    "none",
    "null",
    "paste_your_key_here",
}


class ConfigError(RuntimeError):
    """Raised when environment-backed configuration is invalid or missing."""


def _clean(value: str | None, default: str = "") -> str:
    return (value or "").strip() or default


def _dedupe_models(models: tuple[str, ...]) -> tuple[str, ...]:
    """Keep model order while removing blank and duplicate entries."""

    seen: set[str] = set()
    ordered: list[str] = []
    for model in models:
        cleaned = model.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return tuple(ordered)


def _parse_model_sequence(value: str | None) -> tuple[str, ...]:
    if not value or not value.strip():
        return ()
    return _dedupe_models(tuple(value.split(",")))


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().casefold() in {"1", "true", "yes", "y", "on"}


def _as_int(value: str | None, default: int, name: str) -> int:
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}.") from exc


def _as_decimal(value: str | None, default: Decimal, name: str) -> Decimal:
    if value is None or not value.strip():
        return default
    try:
        return parse_decimal(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be numeric, got {value!r}.") from exc


def _as_path(value: str | None, default: Path) -> Path:
    if not value or not value.strip():
        return default
    candidate = Path(value.strip()).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _is_real_secret(value: str | None) -> bool:
    if value is None:
        return False
    normalized = value.strip().casefold()
    return bool(normalized) and normalized not in _PLACEHOLDER_KEYS


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application settings loaded from ``.env`` / process env."""

    groq_api_key: str | None = None
    groq_model: str = DEFAULT_GROQ_MODEL
    groq_fallback_models: tuple[str, ...] = GROQ_MODEL_SEQUENCE[1:]
    groq_transcription_model: str = DEFAULT_GROQ_TRANSCRIPTION_MODEL

    business_name: str = ""
    business_tax_id: str = ""
    business_address: str = ""
    business_city: str = "Skopje"
    business_country: str = "MK"
    business_email: str = ""
    business_phone: str = ""
    business_iban: str = DEFAULT_BUSINESS_IBAN
    sender_name: str = ""

    default_currency: str = "MKD"
    default_vat_rate: Decimal = Decimal("18.00")
    default_payment_terms_days: int = 15
    default_amount_includes_vat: bool = False
    default_language: Language = Language.MK
    default_channel: Channel = Channel.EMAIL
    default_service_description: str = "Professional services"

    invoice_prefix: str = "NPL"
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    timezone: str = "Europe/Skopje"
    log_level: str = "INFO"
    db_encryption_key: str = ""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, env_file: Path | str | None = None) -> Settings:
        """Load settings from a dotenv file plus the current environment.

        Existing process environment variables are not overwritten by the
        dotenv file, which keeps container/CI configuration authoritative.
        """

        dotenv_path = Path(env_file) if env_file else PROJECT_ROOT / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path, override=False)

        api_key = _clean(os.getenv("GROQ_API_KEY"))
        if not _is_real_secret(api_key):
            api_key = ""

        configured_models = _parse_model_sequence(os.getenv("GROQ_MODELS"))
        if not configured_models:
            # GROQ_MODEL selects the first model; the remaining active models
            # stay in the documented fallback rotation order.
            primary_model = _clean(os.getenv("GROQ_MODEL"), DEFAULT_GROQ_MODEL)
            configured_models = _dedupe_models(
                (primary_model, *GROQ_MODEL_SEQUENCE)
            )
        model = configured_models[0]
        fallback_models = configured_models[1:]
        default_language_raw = _clean(os.getenv("DEFAULT_LANGUAGE"), "MK")
        default_channel_raw = _clean(os.getenv("DEFAULT_CHANNEL"), "Email")

        vat_rate = _as_decimal(
            os.getenv("DEFAULT_VAT_RATE"),
            Decimal("18.00"),
            "DEFAULT_VAT_RATE",
        )
        if vat_rate < 0 or vat_rate > 100:
            raise ConfigError("DEFAULT_VAT_RATE must be between 0 and 100.")

        payment_days = _as_int(
            os.getenv("DEFAULT_PAYMENT_TERMS_DAYS"),
            15,
            "DEFAULT_PAYMENT_TERMS_DAYS",
        )
        if payment_days <= 0:
            raise ConfigError("DEFAULT_PAYMENT_TERMS_DAYS must be greater than zero.")

        timezone_name = _clean(os.getenv("TIMEZONE"), "Europe/Skopje")
        try:
            ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ConfigError(f"Unknown TIMEZONE: {timezone_name!r}.") from exc

        prefix = _clean(os.getenv("INVOICE_PREFIX"), "NPL").upper()
        if not prefix.replace("-", "").isalnum():
            raise ConfigError("INVOICE_PREFIX may contain only letters, numbers, and hyphens.")

        language = Language.from_value(
            normalize_language_code(default_language_raw, Language.MK.value),
            default=Language.MK,
        )
        channel = Channel.from_value(
            normalize_channel(default_channel_raw, Channel.EMAIL.value),
            default=Channel.EMAIL,
        )

        return cls(
            groq_api_key=api_key or None,
            groq_model=model,
            groq_fallback_models=fallback_models,
            groq_transcription_model=_clean(
                os.getenv("GROQ_TRANSCRIPTION_MODEL"),
                DEFAULT_GROQ_TRANSCRIPTION_MODEL,
            ),
            business_name=_clean(os.getenv("BUSINESS_NAME")),
            business_tax_id=_clean(os.getenv("BUSINESS_TAX_ID")),
            business_address=_clean(os.getenv("BUSINESS_ADDRESS")),
            business_city=_clean(os.getenv("BUSINESS_CITY"), "Skopje"),
            business_country=_clean(os.getenv("BUSINESS_COUNTRY"), "MK").upper(),
            business_email=_clean(os.getenv("BUSINESS_EMAIL")),
            business_phone=_clean(os.getenv("BUSINESS_PHONE")),
            business_iban=_clean(
                os.getenv("BUSINESS_IBAN"), DEFAULT_BUSINESS_IBAN
            ).replace(" ", "").upper(),
            sender_name=_clean(os.getenv("SENDER_NAME")),
            default_currency=_clean(os.getenv("DEFAULT_CURRENCY"), "MKD").upper(),
            default_vat_rate=vat_rate,
            default_payment_terms_days=payment_days,
            default_amount_includes_vat=_as_bool(
                os.getenv("DEFAULT_AMOUNT_INCLUDES_VAT"), False
            ),
            default_language=language,
            default_channel=channel,
            default_service_description=_clean(
                os.getenv("DEFAULT_SERVICE_DESCRIPTION"), "Professional services"
            ),
            invoice_prefix=prefix,
            data_dir=_as_path(os.getenv("DATA_DIR"), DEFAULT_DATA_DIR),
            timezone=timezone_name,
            log_level=_clean(os.getenv("LOG_LEVEL"), "INFO").upper(),
            db_encryption_key=_clean(os.getenv("DB_ENCRYPTION_KEY")),
        )

    # ------------------------------------------------------------------
    # Derived paths and helpers
    # ------------------------------------------------------------------

    @property
    def has_groq_api_key(self) -> bool:
        return _is_real_secret(self.groq_api_key)

    @property
    def groq_model_sequence(self) -> tuple[str, ...]:
        """Return the ordered, de-duplicated model rotation for fallback."""

        return _dedupe_models((self.groq_model, *self.groq_fallback_models))

    @property
    def groq_models(self) -> tuple[str, ...]:
        """Alias used by service code and dashboards."""

        return self.groq_model_sequence

    @property
    def invoice_store_path(self) -> Path:
        return self.data_dir / "invoices.json"

    @property
    def ubl_output_dir(self) -> Path:
        """Default audit location for generated UBL 2.1 XML files.

        With the standard ``DATA_DIR=data`` layout this resolves to the
        project-level ``output/`` directory, so a file is created at
        ``output/NPL-2026-0001.xml``.
        """

        return self.data_dir.parent / "output"

    @property
    def ujp_output_dir(self) -> Path:
        """Backward-compatible alias for the UBL audit directory."""

        return self.ubl_output_dir

    @property
    def is_business_configured(self) -> bool:
        return bool(self.business_name and self.business_tax_id)

    def require_groq_api_key(self) -> str:
        if not self.has_groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is missing or still a placeholder. "
                "Create .env from .env.example and set a valid Groq API key "
                "(https://console.groq.com/keys)."
            )
        return str(self.groq_api_key)

    def today(self) -> date:
        """Return today's date in the configured business timezone."""

        return datetime.now(ZoneInfo(self.timezone)).date()

    def ensure_directories(self) -> None:
        """Create the local data directories used by the MVP."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ubl_output_dir.mkdir(parents=True, exist_ok=True)


def load_settings(env_file: Path | str | None = None) -> Settings:
    """Convenience wrapper used by the CLI and module-level entry points."""

    return Settings.from_env(env_file=env_file)
