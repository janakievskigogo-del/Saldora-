"""Zero-Knowledge AI tokenization and prompt-injection firewall for Saldora.

Two defensive layers in front of every LLM call:

1. :class:`TokenManager` builds "zero-knowledge" prompts: real invoice entities
   (client names, phone numbers, amounts, ...) are swapped for opaque
   ``[CATEGORY_n]`` placeholders before the text is handed to Gemini, and the
   real values are only ever restored **locally** in the callee's response via
   strict :py:meth:`str.replace` (never regex, never the model).

2. :class:`PromptFirewall` screens inbound conversational messages (Viber or
   Telegram replies) for prompt-injection signatures and, on a hit,
   short-circuits the pipeline with a hardcoded safe fallback so the LLM is
   never called with hostile content.

Security properties
-------------------
* Non-deterministic entity tokenization is avoided: the same value in the same
  category always yields the same token for the lifetime of the manager, so a
  multi-turn session never mixes two entities that share a prefix or substring.
* All value replacement is longest-first with :py:func:`re.escape` and word
  boundaries, so ``Ivan`` can never eat the inside of ``Ivanov`` and values
  containing regex metacharacters (``+389...``, ``1200.00``) are inert.
* Detokenization uses literal :py:meth:`str.replace`, ordered by token length,
  so ``[CLIENT_10]`` is restored before ``[CLIENT_1]`` and no bracket-parsing
  or backreference trick can redirect the replacement.
* The firewall never logs raw user content, only a verdict and category.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["SAFE_FALLBACK_RESPONSE", "PromptFirewall", "TokenManager"]

# Sentinel returned when the firewall short-circuits the LLM pipeline.
SAFE_FALLBACK_RESPONSE = "Security Violation: Request blocked by prompt firewall."

_WORD_CHAR = re.compile(r"\w", re.UNICODE)
_NON_ALNUM_RUN = re.compile(r"[^a-z0-9]+")
# Zero-width / invisible formatting characters: never word separators.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u2060-\u206f\u2028\u2029\ufeff\xad\u034f\u180e]")
_VALID_CATEGORY = re.compile(r"^[A-Z][A-Z0-9_]{0,29}$")


class TokenManager:
    """Maps real entities to opaque ``[CATEGORY_n]`` tokens and back again.

    State is per-manager (one instance per conversation/session): every
    (category, value) pair receives its token exactly once and keeps it for the
    lifetime of the mapping, so ``[CLIENT_1]`` always refers to the same
    entity even across repeated :py:meth:`tokenize` calls.  The reverse
    dictionary (:attr:`reverse_map`, keyed by token) is the only thing capable
    of turning tokens back into real values, and it never leaves the process.
    """

    def __init__(self) -> None:
        #: token -> original value (longest-token-safe detokenization input).
        self.reverse_map: dict[str, str] = {}
        #: original -> token, for audit trails and stable re-tokenization.
        self.forward_map: dict[tuple[str, str], str] = {}
        self._counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Tokenization (real -> placeholder)
    # ------------------------------------------------------------------

    def _token_for(self, category: str, value: str) -> str:
        """Return the stable token for ``(category, value)``, creating it once."""

        label = category.upper()
        if not _VALID_CATEGORY.match(label):
            raise ValueError(
                f"Invalid entity category {category!r}: use A-Z, 0-9 and "
                "underscores, max 30 chars, starting with a letter."
            )
        token = self.forward_map.get((label, value))
        if token is None:
            # Counter bumps only when a brand-new entity is seen, so an
            # entity appearing repeatedly across turns keeps one token.
            self._counters[label] = self._counters.get(label, 0) + 1
            token = f"[{label}_{self._counters[label]}]"
            self.forward_map[(label, value)] = token
            self.reverse_map[token] = value
        return token

    @staticmethod
    def _guarded_pattern(value: str) -> re.Pattern[str]:
        """Compile an escaped, word-boundary-guarded matcher for ``value``.

        The boundary guards keep a short entity (``Ivan``) from matching inside
        a longer token's neighborhood (``Ivanov``, ``IVANERGO``).  Values that
        start or end in non-word characters (``+38970111222``) simply skip the
        guard on that side.
        """

        escaped = re.escape(value)
        if _WORD_CHAR.match(value[0]):
            escaped = r"\b" + escaped
        if _WORD_CHAR.match(value[-1]):
            escaped = escaped + r"\b"
        return re.compile(escaped)

    def tokenize(self, text: str, entity_mapping: dict[str, str]) -> str:
        """Replace real entities in ``text`` with opaque placeholder tokens.

        Replacement runs **longest-first** so overlapping entities of the same
        category (``Ivanov`` vs ``Ivan``, ``1200.00`` vs ``1200``) can never
        partially consume each other, and every key is regex-escaped plus
        boundary-guarded so values with ``+`` or ``.`` are literal.
        """

        if not entity_mapping:
            return text

        # Deterministic ordering: longest entity first; ties broken by value
        # so equal-length entities still replace in a stable, testable order.
        ordered = sorted(entity_mapping.items(), key=lambda kv: (-len(kv[0]), kv[0]))

        result = text
        for value, category in ordered:
            token = self._token_for(category, value)
            result = self._guarded_pattern(value).sub(token, result)
        return result

    # ------------------------------------------------------------------
    # Detokenization (placeholder -> real)
    # ------------------------------------------------------------------

    def detokenize(self, llm_response: str) -> str:
        """Restore real entities, replacing tokens **longest-first**.

        Strict :py:meth:`str.replace` is used on purpose: placeholders are
        fully controlled constants, so regex would add bracket-parsing and
        backreference risks with zero benefit.  Longest-token order stops
        ``[CLIENT_1]`` from corrupting ``[CLIENT_10]``.
        """

        result = llm_response
        for token in sorted(self.reverse_map, key=lambda t: -len(t)):
            result = result.replace(token, self.reverse_map[token])
        return result


class PromptFirewall:
    """Denylist scanner that short-circuits prompt-injection attempts.

    The screen normalizes hostile text (NFKD, case, zero-width characters,
    punctuation smuggling) before matching, so ``IGNORE previous instructions``,
    ``i\\u200bgnore previous`` and ``mark-as-paid`` all resolve to the same
    canonical form and block.  It is intentionally conservative: a hit returns
    :data:`SAFE_FALLBACK_RESPONSE` and the caller must then **not** call the
    LLM with that content.  Raw user text is never logged, never echoed in the
    verdict, and never forwarded for fuzzy analysis.

    Trade-off (accepted by spec): broad signatures such as ``discount`` also
    block legitimate phrasings -- that is by design for a collections bot, and
    the list grows centrally here rather than at every call site.
    """

    # Normalized-form denylist.  Keep entries lowercase and space-separated.
    _INJECTION_SIGNATURES: tuple[re.Pattern[str], ...] = tuple(
        re.compile(pattern)
        for pattern in (
            r"ignore previous",
            r"ignore instructions",
            r"mark as paid",
            r"discount",
            r"bypass",
            r"system prompt",
            r"override",
        )
    )

    _logger_names = "ai_security"

    def _normalized_candidates(self, user_input: str) -> tuple[str, str]:
        """Return both canonical forms of the payload for the denylist scan.

        An invisible/zero-width character is ambiguous: inside a word it is
        noise (``i\\u200bgnore`` -> ``ignore``), between words it is smuggled
        whitespace (``mark\\u200bas\\u200bpaid`` -> ``mark as paid``).  No single
        transform catches both, so two candidates are produced and the scan
        fires on whichever one matches:

        * ``delete`` candidate: invisibles removed  -> fuses zero-width letters.
        * ``spoken``  candidate: invisibles as gaps -> splits zero-width words.

        Shared pre-passes: NFKD decomposition, lowercasing, and deletion of
        combining diacritics (``igno\\u0301re`` -> ``ignore``).  Then every
        remaining non-alphanumeric run collapses to one space so ``MARK-AS-PAID``
        and ``mark.as.paid`` both resolve to ``mark as paid``.
        """

        decomposed = "".join(
            ch
            for ch in unicodedata.normalize("NFKD", user_input).lower()
            if unicodedata.combining(ch) == 0
        )
        deleted = _INVISIBLE.sub("", decomposed)
        spaced = _INVISIBLE.sub(" ", decomposed)
        return (
            _NON_ALNUM_RUN.sub(" ", deleted),
            _NON_ALNUM_RUN.sub(" ", spaced),
        )

    def is_injection(self, user_input: str) -> bool:
        """Return ``True`` when any normalized form hits any signature."""

        candidates = self._normalized_candidates(user_input)
        return any(
            pattern.search(candidate)
            for candidate in candidates
            for pattern in self._INJECTION_SIGNATURES
        )

    def filter_input(self, user_input: str) -> tuple[bool, str]:
        """Gate a user message for the LLM.

        Returns ``(True, user_input)`` when clean (original text preserved for
        the pipeline) or ``(False, SAFE_FALLBACK_RESPONSE)`` when the message
        trips the firewall -- the caller must short-circuit and send the
        fallback to the user **without** invoking the model.
        """

        if self.is_injection(user_input):
            return False, SAFE_FALLBACK_RESPONSE
        return True, user_input
