"""
PII-safe logging infrastructure for Saldora.

Logs travel far (stdout -> container collector -> SIEM), so no personal of
fiscal data may ever leave the process in clear text.  :class:`PIIFilter`
masks e-mail addresses, Macedonian mobile numbers and tax IDs **inside the
``logging.LogRecord`` before any formatter or downstream handler sees it**,
and never drops records (``filter`` always returns ``True``).

Masking rules, applied in order:

1. E-mail addresses            -> ``x***@domain.com``
2. Macedonian mobile numbers   -> keeps ``+389/0`` prefix and ``7``, hides the rest
   (``0712345678`` -> ``07****222``, ``+38971234567`` -> ``+3897****222``)
3. Tax IDs (13 digits)         -> keeps the last 4 digits only
   (``4030123456789`` -> ``*********5678``)
"""

import logging
import re
import sys

__all__ = ["PIIFilter", "logger"]

_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


class PIIFilter(logging.Filter):
    """In-place PII masking filter for every record emitted by Saldora.

    The record's lazily-formatted text (``record.getMessage()``, i.e. the
    message already interpolated with ``record.args``) is masked once, then
    written back as the full literal message with ``args`` cleared, so no
    downstream formatter can resurrect the raw arguments.

    Patterns are class-level compiled constants used on the hot logging path.
    """

    EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    PHONE_PATTERN = re.compile(r"(\+389|0)7\d{7}")
    TAX_ID_PATTERN = re.compile(r"\b\d{13}\b")

    EMAIL_MASK = "x***@domain.com"
    # Keeps the group-1 prefix (``+389`` or ``0``) plus the leading ``7``.
    # ``\g<1>`` is used instead of a bare ``\1`` so the trailing ``7`` cannot
    # be mis-parsed as a second-digit group reference (``\17`` -> group 17).
    PHONE_MASK = r"\g<1>7****222"
    TAX_ID_MASK = "*********5678"

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask PII in-place and unconditionally pass the record through."""
        msg = record.getMessage()

        msg = self.EMAIL_PATTERN.sub(self.EMAIL_MASK, msg)
        msg = self.PHONE_PATTERN.sub(self.PHONE_MASK, msg)
        msg = self.TAX_ID_PATTERN.sub(self.TAX_ID_MASK, msg)

        record.msg = msg
        record.args = ()
        return True


# ---------------------------------------------------------------------------
# Configured application logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("saldora")
logger.setLevel(logging.INFO)
# Records must not travel up to the root logger: a naive root handler (or the
# logging.lastResort fallback) would emit the record *unmasked* and lie about
# our PII hygiene.  All output goes through the masked handler below.
logger.propagate = False

if not logger.handlers:  # idempotent under hot reloads / import graph cycles
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    _handler.addFilter(PIIFilter())
    logger.addHandler(_handler)
