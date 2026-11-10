"""Multilingual inbound-reply categorization for Saldora.

The classifier is deterministic and conservative.  It is deliberately not allowed
to mark an invoice as PAID by itself: "paid" only means the client claims payment,
and the business owner must verify the bank account first.  Ambiguous or conflicting
replies return ``unclear`` so a human can follow up.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from re import Pattern

from legacy_models import Language, ReplyAnalysis, ReplyCategory
from utils import normalize_whitespace


@dataclass(frozen=True, slots=True)
class _Rule:
    category: ReplyCategory
    pattern: Pattern[str]
    weight: float


def _compile(patterns: Iterable[str], flags: int = re.IGNORECASE | re.UNICODE) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(pattern, flags) for pattern in patterns)


def _rule(category: ReplyCategory, pattern: str, weight: float) -> _Rule:
    """Compile a rule once at import time instead of on every reply."""

    return _Rule(category, re.compile(pattern, re.IGNORECASE | re.UNICODE), weight)


_PAID_RULES = [
    _rule(ReplyCategory.PAID, pattern, weight)
    for pattern, weight in (
        (r"\b(?:i\s+)?(?:have\s+|already\s+)?paid\b", 3.0),
        (r"\bpaid\s+(?:in\s+full|already|today|yesterday)\b", 3.0),
        (r"\bpayment\s+(?:has\s+been|was|is)\s+(?:made|sent|completed|transferred|processed)\b", 3.0),
        (r"\b(?:transferred|remitted)\s+(?:the\s+)?(?:funds|payment|amount|money)\b", 3.0),
        (r"\bsettled\s+(?:the\s+)?(?:invoice|account|balance)\b", 3.0),
        (r"\binvoice\s+(?:is\s+|was\s+)?(?:paid|settled)\b", 3.0),
        (r"\b(?:payment|transfer)\s+(?:was\s+)?sent\b", 2.5),
        (r"\b(?:веќе\s+)?(?:платив(?:ме)?|уплатив(?:ме)?)\b", 3.0),
        (r"\b(?:е\s+)?(?:платен(?:а|о|и)?|уплатен(?:а|о|и)?|наплатен(?:а|о|и)?)\b", 3.0),
        (r"\bизврш(?:ена|ено|ив)\s+уплата\b", 3.0),
        (r"\bлегна(?:а|о)?\s+уплата\b", 2.5),
        (r"\bпренесов(?:ме)?\b", 2.5),
        (r"\b(?:e\s+)?kam\s+paguar\b", 3.0),
        (r"\bpaguajta\b", 3.0),
        (r"\bpaguar\b", 3.0),
        (r"\bpagesa\s+(?:u|është)\s+(?:kryer|bërë|dërguar)\b", 3.0),
        (r"\btransferova\b", 2.5),
        (r"\btransferuar\b", 2.5),
        (r"\bshlyer\b", 2.5),
    )
]

_EXTENSION_RULES = [
    _rule(ReplyCategory.EXTENSION_REQUEST, pattern, weight)
    for pattern, weight in (
        (r"\bextension\b", 3.0),
        (r"\bextend\b", 2.0),
        (r"\bmore\s+time\b", 3.0),
        (r"\b(?:postpone|delay)\b", 2.5),
        (r"\bnext\s+(?:week|month)\b", 2.5),
        (r"\bpayment\s+plan\b", 3.0),
        (r"\binstallments?\b", 2.5),
        (r"\bpay\s+in\s+parts\b", 2.5),
        (r"\b(?:will|can\s+only)\s+pay\b", 2.0),
        (r"\b(?:can(?:not|'t)|can\s+not)\s+pay\s+(?:right\s+)?(?:now|yet|this\s+month)\b", 3.0),
        (r"\bneed\s+(?:an?\s+extension|more\s+time)\b", 3.0),
        (r"\b(?:soon|shortly|asap)\b", 1.0),
        (r"\bпродолжување\b", 3.0),
        (r"\bодложување\b", 2.5),
        (r"\bодложам\b", 2.5),
        (r"\bподоцна\b", 2.0),
        (r"\bследната\s+недела\b", 2.5),
        (r"\bследниот\s+месец\b", 2.5),
        (r"\bна\s+рати\b", 2.5),
        (r"\bплан\s+за\s+плаќање\b", 3.0),
        (r"\bќе\s+платам\b", 2.5),
        (r"\bне\s+можам\s+(?:да\s+)?платам\b", 3.0),
        (r"\bне\s+можам\s+(?:сега|во\s+моментот|веднаш)\b", 3.0),
        (r"\bмолам\s+за\s+(?:рок|време)\b", 3.0),
        (r"\b(?:краток|тесен)\s+рок\b", 2.5),
        (r"\bshtyrje\b", 3.0),
        (r"\bafat\b", 1.5),
        (r"\bmë\s+vonë\b", 2.0),
        (r"\bjavën\s+(?:e\s+ardhshme|tjetër)\b", 2.5),
        (r"\bmuajin\s+(?:e\s+ardhshëm|tjetër)\b", 2.5),
        (r"\bkëste\b", 2.5),
        (r"\bplan\s+pagese\b", 3.0),
        (r"\bdo\s+të\s+paguaj\b", 2.5),
        (r"\bnuk\s+mund\s+(?:të\s+|te\s+)?(?:ta\s+)?paguaj\b", 3.0),
        (r"\bnuk\s+mund\s+(?:tani|ta\s+paguaj)\b", 3.0),
        (r"\bkërkoj\s+afat\b", 3.0),
    )
]

_DISPUTE_RULES = [
    _rule(ReplyCategory.DISPUTE, pattern, weight)
    for pattern, weight in (
        (r"\bdisput(?:e|ed|ing)\b", 3.0),
        (r"\bincorrect\b", 3.0),
        (r"\bwrong\b", 2.5),
        (r"\bnot\s+(?:received|ours|mine|ordered)\b", 3.0),
        (r"\bnever\s+(?:received|ordered)\b", 3.0),
        (r"\bovercharg(?:e|ed|ing)\b", 3.0),
        (r"\bdouble[-\s]?charged\b", 3.0),
        (r"\bunrecognized\b", 2.5),
        (r"\bproblem\s+with\s+(?:the\s+)?invoice\b", 2.5),
        (r"\bnot\s+(?:my|our)\s+invoice\b", 3.0),
        (r"\bnever\s+ordered\b", 3.0),
        (r"\bспор(?:от|на)?\b", 3.0),
        (r"\bоспорувам\b", 3.0),
        (r"\bоспорен(?:а|о)?\b", 3.0),
        (r"\bпогрешн(?:а|о|и)\b", 2.5),
        (r"\bнеточн(?:а|о|и)\b", 2.5),
        (r"\bне\s+(?:ја\s+)?(?:добив|признавам)\b", 3.0),
        (r"\bне\s+е\s+точн(?:а|о|и)\b", 3.0),
        (r"\bне\s+е\s+моја\b", 3.0),
        (r"\bне\s+сум\s+(?:ја\s+)?нарачал(?:а)?\b", 3.0),
        (r"\bпрефактурирање\b", 2.5),
        (r"\bпроблем\s+со\s+фактурата\b", 2.5),
        (r"\bkontestoj\b", 3.0),
        (r"\bgabuar\b", 2.5),
        (r"\be\s+pasaktë\b", 3.0),
        (r"\bnuk\s+e\s+kam\s+marrë\b", 3.0),
        (r"\bnuk\s+është\s+e\s+imja\b", 3.0),
        (r"\bnuk\s+e\s+kam\s+porositur\b", 3.0),
        (r"\bfatur[ëe]\s+e\s+gabuar\b", 2.5),
        (r"\bproblem\s+me\s+fatur\b", 2.5),
        (r"\bnuk\s+e\s+pranoj\b", 2.5),
    )
]

_NEGATED_PAID_RULES = _compile(
    (
        r"\bnot\s+(?:yet\s+)?paid\b",
        r"\bnot\s+(?:yet\s+)?(?:settled|transferred)\b",
        r"\b(?:have|has)\s+not\s+paid\b",
        r"\bhaven'?t\s+paid\b",
        r"\bhasn'?t\s+paid\b",
        r"\bunpaid\b",
        r"\bpayment\s+not\s+(?:made|sent|received)\b",
        r"\bне\s+сум\s+(?:платил(?:а|е)?|уплатил(?:а|е)?)\b",
        r"\bне\s+(?:е\s+)?(?:платено|уплатено)\b",
        r"\bнеплатено\b",
        r"\bnuk\s+(?:kam\s+|e\s+kam\s+)?(?:paguar|transferuar|shlyer)\b",
        r"\bnuk\s+(?:është|eshte)\s+(?:paguar|transferuar|shlyer)\b",
        r"\be\s+papaguar\b",
    )
)

_ACTION_BY_CATEGORY = {
    ReplyCategory.PAID: (
        "Verify the receipt in the business bank account before marking the invoice PAID."
    ),
    ReplyCategory.EXTENSION_REQUEST: (
        "Review the requested extension/payment date, decide whether to approve it, "
        "and confirm any new terms in writing. Do not auto-approve."
    ),
    ReplyCategory.DISPUTE: (
        "Pause reminders and review the disputed invoice with the business owner "
        "before replying or changing the invoice."
    ),
    ReplyCategory.UNCLEAR: (
        "Ask one short clarifying question: was payment made, when will it be made, "
        "or what exactly is disputed?"
    ),
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("’", "'").replace("‘", "'")
    return normalize_whitespace(normalized.casefold())


def _collect_matches(text: str, rules: Iterable[_Rule]) -> tuple[float, list[str]]:
    score = 0.0
    matched: list[str] = []
    for rule in rules:
        match = rule.pattern.search(text)
        if match:
            score += rule.weight
            matched.append(match.group(0))
    return score, matched


class ReplyParser:
    """Classify MK/AL/EN client replies into actionable categories."""

    def classify(
        self,
        text: str,
        language: Language | str | None = None,
    ) -> ReplyAnalysis:
        if not text or not text.strip():
            return ReplyAnalysis(
                category=ReplyCategory.UNCLEAR,
                confidence=0.0,
                matched_terms=[],
                suggested_action=_ACTION_BY_CATEGORY[ReplyCategory.UNCLEAR],
                language=str(language) if language else None,
            )

        normalized = _normalize(text)
        negated_paid = any(pattern.search(normalized) for pattern in _NEGATED_PAID_RULES)

        scores: dict[ReplyCategory, float] = {}
        matches: dict[ReplyCategory, list[str]] = {}

        paid_score, paid_matches = _collect_matches(normalized, _PAID_RULES)
        if negated_paid:
            paid_score = 0.0
            paid_matches = []
        scores[ReplyCategory.PAID] = paid_score
        matches[ReplyCategory.PAID] = paid_matches

        for category, rules in (
            (ReplyCategory.EXTENSION_REQUEST, _EXTENSION_RULES),
            (ReplyCategory.DISPUTE, _DISPUTE_RULES),
        ):
            score, category_matches = _collect_matches(normalized, rules)
            scores[category] = score
            matches[category] = category_matches

        # Safety-first tie priority: disputes need a human, then extension terms,
        # then payment claims.  A single strong signal can still win on score.
        priority = {
            ReplyCategory.DISPUTE: 2,
            ReplyCategory.EXTENSION_REQUEST: 1,
            ReplyCategory.PAID: 0,
        }
        ranked = sorted(
            scores.items(),
            key=lambda item: (item[1], priority[item[0]]),
            reverse=True,
        )
        top_category, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_score <= 0:
            return ReplyAnalysis(
                category=ReplyCategory.UNCLEAR,
                confidence=0.0,
                matched_terms=[],
                suggested_action=_ACTION_BY_CATEGORY[ReplyCategory.UNCLEAR],
                language=self._language_value(language),
            )

        # A payment claim that is mixed with a dispute is never safe to treat as
        # "paid".  If the dispute is at least as strong as the payment claim, the
        # dispute wins; otherwise the reply needs human review.
        if scores[ReplyCategory.PAID] > 0 and scores[ReplyCategory.DISPUTE] > 0:
            top_category = (
                ReplyCategory.DISPUTE
                if scores[ReplyCategory.DISPUTE] >= scores[ReplyCategory.PAID]
                else ReplyCategory.UNCLEAR
            )
            top_score = max(
                scores[ReplyCategory.PAID],
                scores[ReplyCategory.DISPUTE],
            )

        # If two credible signals have the same weight, prefer human review rather
        # than guessing.  This protects the business from acting on a guess.
        if (
            second_score > 0
            and abs(top_score - second_score) < 0.01
            and ranked[0][0] != ranked[1][0]
        ):
            top_category = ReplyCategory.UNCLEAR

        confidence = min(
            0.98,
            0.5 + (top_score * 0.08) + (max(0.0, top_score - second_score) * 0.05),
        )
        if top_category == ReplyCategory.UNCLEAR:
            confidence = 0.35
            combined_matches = (
                matches.get(ReplyCategory.PAID, [])
                + matches.get(ReplyCategory.EXTENSION_REQUEST, [])
                + matches.get(ReplyCategory.DISPUTE, [])
            )
            matched_terms = self._dedupe(combined_matches)[:8]
        else:
            matched_terms = self._dedupe(matches.get(top_category, []))[:8]

        return ReplyAnalysis(
            category=top_category,
            confidence=round(confidence, 2),
            matched_terms=matched_terms,
            suggested_action=_ACTION_BY_CATEGORY[top_category],
            language=self._language_value(language),
        )

    @staticmethod
    def _language_value(language: Language | str | None) -> str | None:
        if language is None:
            return None
        if isinstance(language, Language):
            return language.value
        return str(language).strip().upper() or None

    @staticmethod
    def _dedupe(values: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                result.append(value)
        return result


def parse_reply(
    text: str,
    language: Language | str | None = None,
) -> ReplyAnalysis:
    """Convenience wrapper for CLI and future API callers."""

    return ReplyParser().classify(text, language=language)
