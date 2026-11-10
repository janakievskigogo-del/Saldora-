"""Tests for .env-backed settings and placeholders."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from config import (
    DEFAULT_BUSINESS_IBAN,
    DEFAULT_GROQ_MODEL,
    DEFAULT_GROQ_TRANSCRIPTION_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL_SEQUENCE,
    ConfigError,
    Settings,
)
from legacy_models import Channel, Language


def _use_missing_env_file(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist.env"


def test_default_groq_model_rotation_order() -> None:
    assert GROQ_MODEL_SEQUENCE == (
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    )
    assert DEFAULT_GROQ_MODEL == "llama-3.3-70b-versatile"
    assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"
    assert Settings().groq_model_sequence == GROQ_MODEL_SEQUENCE


def test_groq_models_can_be_configured_as_comma_separated_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "GROQ_MODELS", "llama-3.1-8b-instant, llama-3.3-70b-versatile"
    )
    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))
    assert settings.groq_model_sequence == (
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    )


def test_groq_model_env_selects_primary_and_keeps_default_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GROQ_MODELS", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))
    assert settings.groq_model_sequence == (
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
    )


def test_settings_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GROQ_MODELS", raising=False)
    values = {
        "GROQ_API_KEY": "gsk_test-api-key",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "GROQ_TRANSCRIPTION_MODEL": "whisper-large-v3-turbo",
        "BUSINESS_NAME": "Env Company DOOEL",
        "BUSINESS_TAX_ID": "4030000000000",
        "BUSINESS_ADDRESS": "Env Street 1",
        "BUSINESS_CITY": "Bitola",
        "BUSINESS_COUNTRY": "mk",
        "BUSINESS_EMAIL": "billing@env.mk",
        "BUSINESS_PHONE": "+38970000000",
        "BUSINESS_IBAN": " mk33 0000 0000 0000 0000 ",
        "SENDER_NAME": "Env Sender",
        "DEFAULT_CURRENCY": "eur",
        "DEFAULT_VAT_RATE": "18",
        "DEFAULT_PAYMENT_TERMS_DAYS": "20",
        "DEFAULT_AMOUNT_INCLUDES_VAT": "true",
        "DEFAULT_LANGUAGE": "AL",
        "DEFAULT_CHANNEL": "Viber",
        "DEFAULT_SERVICE_DESCRIPTION": "Env consulting",
        "INVOICE_PREFIX": "env",
        "DATA_DIR": str(tmp_path / "data"),
        "TIMEZONE": "Europe/Skopje",
        "LOG_LEVEL": "debug",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))

    assert settings.has_groq_api_key
    assert settings.require_groq_api_key() == "gsk_test-api-key"
    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert settings.groq_transcription_model == "whisper-large-v3-turbo"
    assert settings.business_name == "Env Company DOOEL"
    assert settings.business_country == "MK"
    assert settings.business_iban == "MK330000000000000000"
    assert settings.sender_name == "Env Sender"
    assert settings.default_currency == "EUR"
    assert settings.default_vat_rate == Decimal("18.00")
    assert settings.default_payment_terms_days == 20
    assert settings.default_amount_includes_vat is True
    assert settings.default_language == Language.AL
    assert settings.default_channel == Channel.VIBER
    assert settings.default_service_description == "Env consulting"
    assert settings.invoice_prefix == "ENV"
    assert settings.data_dir == tmp_path / "data"
    assert settings.log_level == "DEBUG"

    settings.ensure_directories()
    assert settings.data_dir.is_dir()
    assert settings.ubl_output_dir == tmp_path / "output"
    assert settings.ubl_output_dir.is_dir()
    assert settings.ujp_output_dir == settings.ubl_output_dir


def test_groq_transcription_model_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GROQ_TRANSCRIPTION_MODEL", raising=False)
    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))
    assert settings.groq_transcription_model == DEFAULT_GROQ_TRANSCRIPTION_MODEL


def test_business_iban_defaults_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("BUSINESS_IBAN", raising=False)
    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))
    assert settings.business_iban == DEFAULT_BUSINESS_IBAN


@pytest.mark.parametrize(
    "placeholder",
    ["gsk_...", "your_groq_api_key_here", "gsk_your_groq_api_key"],
)
def test_placeholder_api_key_is_not_treated_as_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    placeholder: str,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", placeholder)
    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))
    assert not settings.has_groq_api_key
    with pytest.raises(ConfigError):
        settings.require_groq_api_key()


def test_gemini_key_is_not_accepted_as_a_groq_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-google-key")
    settings = Settings.from_env(env_file=_use_missing_env_file(tmp_path))
    assert not settings.has_groq_api_key


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("DEFAULT_VAT_RATE", "101"),
        ("DEFAULT_PAYMENT_TERMS_DAYS", "0"),
        ("TIMEZONE", "Not/AZone"),
        ("INVOICE_PREFIX", "NPL/2026"),
    ],
)
def test_invalid_settings_raise_config_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    monkeypatch.setenv(key, value)
    with pytest.raises(ConfigError):
        Settings.from_env(env_file=_use_missing_env_file(tmp_path))


def test_today_uses_configured_timezone(settings: Settings) -> None:
    assert isinstance(settings.today(), date)
