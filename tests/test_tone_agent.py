"""Tests for deterministic escalation tiers and Groq boundary behavior."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from config import GROQ_BASE_URL, Settings
from legacy_models import Channel, Invoice, Language, ToneTier
from tone_agent import (
    MAX_OUTPUT_TOKENS,
    ToneAgent,
    ToneConfigurationError,
    ToneGenerationError,
    build_email_subject,
)


def _completion(text: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ]
    )


class FakeCompletions:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        return _completion(self.text)


class FakeClient:
    def __init__(self, text: str) -> None:
        self.completions = FakeCompletions(text)
        self.chat = SimpleNamespace(completions=self.completions)


class FallbackCompletions:
    """Return a successful draft only for models not listed in ``failures``."""

    def __init__(
        self,
        failures: dict[str, Exception],
        *,
        text: str = "Dear client, please find the complete payment reminder below.",
        truncated: set[str] | None = None,
    ) -> None:
        self.failures = failures
        self.text = text
        self.truncated = truncated or set()
        self.calls: list[str] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> SimpleNamespace:
        self.calls.append(model)
        if model in self.failures:
            raise self.failures[model]
        finish_reason = "length" if model in self.truncated else "stop"
        return _completion(self.text, finish_reason=finish_reason)


class FallbackClient:
    def __init__(self, completions: FallbackCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)


def _agent(settings: Settings, text: str) -> tuple[ToneAgent, FakeClient]:
    client = FakeClient(text)
    return ToneAgent(settings, client=client), client


def _user_content(call: dict[str, object]) -> str:
    messages = call["messages"]
    assert isinstance(messages, list)
    return next(
        message["content"]
        for message in messages
        if message["role"] == "user"
    )


def _system_content(call: dict[str, object]) -> str:
    messages = call["messages"]
    assert isinstance(messages, list)
    return next(
        message["content"]
        for message in messages
        if message["role"] == "system"
    )


@pytest.mark.parametrize(
    ("days_late", "expected"),
    [(5, "FRIENDLY"), (10, "STANDARD"), (25, "FIRM")],
)
def test_selected_tier_is_sent_to_groq(
    settings: Settings,
    invoice: Invoice,
    days_late: int,
    expected: str,
) -> None:
    as_of = date(2026, 10, 30)
    invoice = replace(invoice, due_date=as_of - timedelta(days=days_late))
    writer, client = _agent(
        settings,
        "Dear client, this is a reminder about your outstanding invoice.",
    )

    draft = writer.generate_draft(invoice, as_of=as_of)

    assert draft
    contents = _user_content(client.chat.completions.calls[0])
    assert f"SELECTED TONE STAGE: {expected}" in contents
    assert f'"tone_tier": "{expected.lower()}"' in contents


def test_language_and_channel_are_forwarded(
    settings: Settings,
    invoice: Invoice,
) -> None:
    invoice = replace(invoice, language=Language.MK, channel=Channel.VIBER)
    writer, client = _agent(settings, "Почитуван клиенте, ве молиме проверете ја фактурата.")
    writer.generate_draft(invoice, as_of=date(2026, 10, 30))

    contents = _user_content(client.chat.completions.calls[0])
    assert "OUTPUT LANGUAGE: MK" in contents
    assert "Macedonian" in contents
    assert "CHANNEL: Viber" in contents
    assert "chat message" in contents


def test_reminder_before_due_date_is_refused(
    settings: Settings,
    invoice: Invoice,
) -> None:
    as_of = date(2026, 10, 30)
    invoice = replace(invoice, due_date=as_of + timedelta(days=1))
    writer, _ = _agent(settings, "This should never be used for a future due date.")
    with pytest.raises(ToneGenerationError, match="before the due date"):
        writer.generate_draft(invoice, as_of=as_of)


def test_empty_or_short_groq_output_is_rejected(
    settings: Settings,
    invoice: Invoice,
) -> None:
    as_of = date(2026, 10, 30)
    empty_writer, _ = _agent(settings, "")
    with pytest.raises(ToneGenerationError):
        empty_writer.generate_draft(invoice, as_of=as_of)

    short_writer, _ = _agent(settings, "Too short.")
    with pytest.raises(ToneGenerationError, match="too short"):
        short_writer.generate_draft(invoice, as_of=as_of)


def test_missing_key_raises_configuration_error(invoice: Invoice) -> None:
    writer = ToneAgent(Settings())
    with pytest.raises(ToneConfigurationError):
        writer.generate_draft(invoice, as_of=date(2026, 10, 30))


def test_client_is_initialized_with_groq_base_url(
    monkeypatch: pytest.MonkeyPatch,
    invoice: Invoice,
) -> None:
    settings = Settings(groq_api_key="gsk_test-key")
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, *, api_key: str, base_url: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = SimpleNamespace(
                completions=FakeCompletions(
                    "Dear client, this is a complete reminder about your outstanding invoice."
                )
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    draft = ToneAgent(settings).generate_draft(invoice, as_of=date(2026, 10, 30))

    assert draft
    assert captured == {"api_key": "gsk_test-key", "base_url": GROQ_BASE_URL}


def test_mapping_contract_still_works(settings: Settings) -> None:
    writer, client = _agent(
        settings,
        "Hello, this is a professional reminder about the outstanding invoice.",
    )
    draft = writer.generate_draft(
        {
            "invoice_id": "MAP-001",
            "client_name": "Mapping Client",
            "amount": "1000.00",
            "currency": "EUR",
            "days_late": 12,
            "preferred_language": "EN",
            "channel": "Email",
        },
        as_of=date(2026, 10, 30),
    )
    assert draft
    contents = _user_content(client.chat.completions.calls[0])
    assert "SELECTED TONE STAGE: STANDARD" in contents
    assert '"invoice_id": "MAP-001"' in contents
    assert '"amount_due": "1000.00"' in contents


def test_email_subject_is_localized() -> None:
    assert build_email_subject("TST-1", Language.EN) == "Payment reminder – invoice TST-1"
    assert "фактура" in build_email_subject("TST-1", "MK")
    assert "fatura" in build_email_subject("TST-1", "AL")


@pytest.mark.parametrize(
    "transient_error",
    [RuntimeError("503 UNAVAILABLE"), RuntimeError("429 RATE_LIMIT")],
)
def test_groq_rotation_falls_back_on_transient_errors(
    settings: Settings,
    invoice: Invoice,
    transient_error: RuntimeError,
) -> None:
    models = FallbackCompletions({"llama-3.3-70b-versatile": transient_error})
    writer = ToneAgent(settings, client=FallbackClient(models))

    draft = writer.generate_draft(invoice, as_of=date(2026, 10, 30))

    assert draft
    assert models.calls == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    assert writer.last_model_used == "llama-3.1-8b-instant"


def test_groq_rotation_falls_back_when_output_is_truncated(
    settings: Settings,
    invoice: Invoice,
) -> None:
    models = FallbackCompletions({}, truncated={"llama-3.3-70b-versatile"})
    writer = ToneAgent(settings, client=FallbackClient(models))

    writer.generate_draft(invoice, as_of=date(2026, 10, 30))

    assert models.calls == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    assert writer.last_model_used == "llama-3.1-8b-instant"


def test_all_groq_models_failing_raises_handled_error(
    settings: Settings,
    invoice: Invoice,
) -> None:
    failures = {
        model: RuntimeError(f"503 UNAVAILABLE for {model}")
        for model in settings.groq_model_sequence
    }
    models = FallbackCompletions(failures)
    writer = ToneAgent(settings, client=FallbackClient(models))

    with pytest.raises(ToneGenerationError, match="any configured model"):
        writer.generate_draft(invoice, as_of=date(2026, 10, 30))

    assert models.calls == list(settings.groq_model_sequence)


def test_draft_generation_allows_full_length_output(
    settings: Settings,
    invoice: Invoice,
) -> None:
    writer, client = _agent(
        settings,
        "Dear client, this is a complete reminder about your outstanding invoice.",
    )
    writer.generate_draft(invoice, as_of=date(2026, 10, 30))

    call = client.chat.completions.calls[0]
    assert call["max_tokens"] == MAX_OUTPUT_TOKENS
    assert MAX_OUTPUT_TOKENS >= 2048
    instructions = _system_content(call)
    assert "Never stop mid-sentence" in instructions


@pytest.mark.parametrize(
    ("days", "tier"),
    [
        (1, ToneTier.FRIENDLY),
        (7, ToneTier.FRIENDLY),
        (8, ToneTier.STANDARD),
        (20, ToneTier.STANDARD),
        (21, ToneTier.FIRM),
        (90, ToneTier.FIRM),
    ],
)
def test_tier_boundaries(days: int, tier: ToneTier) -> None:
    assert ToneTier.from_days_late(days) == tier
