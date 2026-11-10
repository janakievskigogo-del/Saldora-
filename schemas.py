"""
Pydantic V2 validation layer for Saldora invoice data.

Fintech design rules
--------------------
* Every monetary value, rate and total is a ``Decimal`` -- ``float`` is never
  used for money, so invoice arithmetic is exact end to end.
* A Macedonian tax identification number is exactly 13 numeric digits.
* Invoices parsed by an LLM/OCR pipeline (``GeminiParsedInvoice``) are never
  trusted blindly: the VAT amount and the grand total are re-computed from the
  parsed net amount and VAT rate, and a discrepancy of a single centita is
  rejected before the record can enter the system.

Exception contract
------------------
Validators raise :class:`TaxIDValidationError` / :class:`InvoiceMathError`
(subclasses of ``ValueError``).  Under Pydantic V2, ``ValueError``-subclasses
raised inside validators are surfaced as ``pydantic.ValidationError`` (which is
itself a ``ValueError``) carrying the original message intact, so callers
should catch ``ValueError`` / ``ValidationError`` around ``model_validate`` /
``model_construct`` endpoints and route on the message.
"""

import re
from decimal import Decimal, ROUND_HALF_UP

from pydantic import BaseModel, Field, field_validator, model_validator

__all__ = [
    "ClientSchema",
    "GeminiParsedInvoice",
    "InvoiceCreate",
    "InvoiceMathError",
    "TaxIDValidationError",
]

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class InvoiceMathError(ValueError):
    """Raised when invoice arithmetic (VAT / total) is internally inconsistent."""


class TaxIDValidationError(ValueError):
    """Raised when a Macedonian tax ID is not exactly 13 numeric digits."""


# ---------------------------------------------------------------------------
# Shared validation logic
# ---------------------------------------------------------------------------

# Strict equivalent of ``^\d{13}$``; ``fullmatch`` also rejects inputs that a
# line-anchored dollar sign would silently accept (e.g. a trailing newline).
_TAX_ID_PATTERN = re.compile(r"\d{13}")

_TAX_ID_ERROR = "Macedonian Tax ID must be exactly 13 numeric digits"


class _MacedonianTaxIDMixin:
    """Shared ``tax_id`` validator for every model that carries the field.

    Deliberately a plain mixin (not a ``BaseModel``) so the validator is bound
    once and reused by :class:`ClientSchema` and :class:`GeminiParsedInvoice`
    without forcing a shared field declaration onto subclasses.
    """

    @field_validator("tax_id", mode="after")
    @classmethod
    def _validate_tax_id(cls, value: str) -> str:
        if not _TAX_ID_PATTERN.fullmatch(value):
            raise TaxIDValidationError(_TAX_ID_ERROR)
        return value


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ClientSchema(_MacedonianTaxIDMixin, BaseModel):
    """A registered client (fiscal subject) of the Saldora platform."""

    name: str = Field(description="Registered company / merchant name.")
    tax_id: str = Field(description="Macedonian tax identification number (13 digits).")
    email: str = Field(description="Primary billing contact e-mail address.")
    phone: str = Field(description="Primary contact phone number (+389 / 07 format).")


class InvoiceCreate(BaseModel):
    """Payload used to create a new invoice.

    The platform computes VAT and the grand total server-side from
    ``net_amount`` and ``vat_rate``; the caller supplies the source-of-truth
    inputs only, never a total that could drift from the arithmetic.
    """

    client_tax_id: str = Field(description="Macedonian tax ID of the client being invoiced.")
    net_amount: Decimal = Field(description="Net amount, VAT excluded, in MKD.")
    vat_rate: Decimal = Field(
        default=Decimal("18.00"),
        description="VAT rate as a percentage (default: Macedonian standard 18.00%).",
    )

    @property
    def vat_amount(self) -> Decimal:
        """Computed VAT: ``net_amount * vat_rate / 100``, cent-exact, half-up."""

        return (self.net_amount * (self.vat_rate / Decimal("100"))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @property
    def total_amount(self) -> Decimal:
        """Computed grand total: ``net_amount + vat_amount`` (both cent-exact)."""

        return (self.net_amount + self.vat_amount).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )


class GeminiParsedInvoice(_MacedonianTaxIDMixin, BaseModel):
    """Invoice fields extracted from a document by the Gemini parsing pipeline.

    Arithmetic integrity is re-verified on construction::

        calc_vat   = round(net_amount * vat_rate / 100, 2)   # ROUND_HALF_UP
        calc_total = net_amount + calc_vat

    ``total_amount`` must match ``calc_total`` cent-for-cent, otherwise an
    :class:`InvoiceMathError` is raised and the parsed invoice is rejected.
    """

    client_name: str = Field(description="Client name exactly as extracted from the document.")
    tax_id: str = Field(description="Macedonian tax ID (13 digits) extracted from the document.")
    net_amount: Decimal = Field(description="Net amount, VAT excluded, extracted from the document.")
    vat_rate: Decimal = Field(description="VAT rate as a percentage, extracted from the document.")
    vat_amount: Decimal = Field(description="VAT amount, extracted from the document.")
    total_amount: Decimal = Field(description="Gross total amount, extracted from the document.")

    @model_validator(mode="after")
    def verify_amount_math(self) -> "GeminiParsedInvoice":
        """Re-compute VAT and total; enforce cent-for-cent consistency."""
        calc_vat = (
            self.net_amount * (self.vat_rate / Decimal("100"))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        calc_total = self.net_amount + calc_vat

        if abs(self.total_amount - calc_total) != Decimal("0.00"):
            raise InvoiceMathError(
                f"Math discrepancy: parsed total {self.total_amount} != expected {calc_total}"
            )
        return self
