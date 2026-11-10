"""Tests for conservative multilingual reply classification."""

from __future__ import annotations

import pytest

from legacy_models import Language, ReplyCategory
from reply_parser import ReplyParser, parse_reply


@pytest.mark.parametrize(
    "text",
    [
        "I have already paid the invoice yesterday.",
        "Paid in full.",
        "Платив вчера.",
        "Фактурата е платена.",
        "Fatura është paguar.",
        "Kam paguar.",
    ],
)
def test_paid_claims(text: str) -> None:
    analysis = parse_reply(text)
    assert analysis.category == ReplyCategory.PAID
    assert "Verify" in analysis.suggested_action


@pytest.mark.parametrize(
    "text",
    [
        "I need more time to pay.",
        "Can I pay next week?",
        "Can we set up a payment plan?",
        "Не можам да платам сега, молам за рок.",
        "Ќе платам следната недела.",
        "Nuk mund të paguaj tani.",
        "Kërkoj afat, javën e ardhshme.",
    ],
)
def test_extension_requests(text: str) -> None:
    assert parse_reply(text).category == ReplyCategory.EXTENSION_REQUEST


@pytest.mark.parametrize(
    "text",
    [
        "The invoice is incorrect.",
        "We never received the service.",
        "You double-charged us.",
        "Фактурата е погрешна.",
        "Не е точна фактурата.",
        "Fatura është e gabuar.",
        "Nuk e kam marrë faturën.",
    ],
)
def test_disputes(text: str) -> None:
    assert parse_reply(text).category == ReplyCategory.DISPUTE


@pytest.mark.parametrize(
    "text",
    [
        "Please send the invoice again.",
        "Hello, can you call me?",
        "I haven't paid yet.",
        "Nuk është paguar.",
        "",
    ],
)
def test_unclear_or_negated_replies(text: str) -> None:
    assert parse_reply(text).category == ReplyCategory.UNCLEAR


def test_paid_and_dispute_conflict_is_not_auto_paid() -> None:
    analysis = parse_reply("I paid but the invoice is wrong.")
    assert analysis.category == ReplyCategory.UNCLEAR
    assert analysis.confidence == pytest.approx(0.35)
    assert any("paid" in term for term in analysis.matched_terms)
    assert any("wrong" in term for term in analysis.matched_terms)


def test_strong_dispute_wins_over_paid_claim() -> None:
    analysis = parse_reply(
        "I paid, but I never received the invoice and the amount is wrong."
    )
    assert analysis.category == ReplyCategory.DISPUTE


def test_language_metadata() -> None:
    analysis = parse_reply("Платив вчера.", language=Language.MK)
    assert analysis.language == "MK"
    assert analysis.to_dict()["language"] == "MK"
    assert analysis.to_dict()["category"] == "paid"


def test_empty_reply_has_zero_confidence() -> None:
    analysis = ReplyParser().classify("   ")
    assert analysis.category == ReplyCategory.UNCLEAR
    assert analysis.confidence == 0.0
