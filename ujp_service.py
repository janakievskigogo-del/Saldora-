"""Mock UJP e-Faktura / UBL 2.1 generation and validation.

The real UJP e-Faktura integration requires official schemas, certificates, and a
network endpoint.  This module intentionally implements the same boundary as a
production adapter: validate domain data, generate UBL 2.1 XML, validate the XML,
write an audit copy to ``output/``, log its absolute path, and return a mock
submission reference.

No network call is made.  The generated document is a standards-shaped invoice
that a future real UJP adapter can render/submit with the official rules.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from xml.etree import ElementTree as ET

from config import DEFAULT_BUSINESS_IBAN, Settings
from legacy_models import Invoice, UJPSubmissionResult, UJPValidationResult
from utils import (
    is_valid_email,
    is_valid_tax_id,
    normalize_tax_id,
    parse_date,
    parse_decimal,
    safe_filename,
)

UBL_NS = "urn:oasis:names:specification:ubl:schema:xsd:Invoice-2"
CAC_NS = "urn:oasis:names:specification:ubl:schema:xsd:CommonAggregateComponents-2"
CBC_NS = "urn:oasis:names:specification:ubl:schema:xsd:CommonBasicComponents-2"

NS = {"ubl": UBL_NS, "cac": CAC_NS, "cbc": CBC_NS}

# Register stable prefixes before serializing so the XML is readable and stable.
ET.register_namespace("", UBL_NS)
ET.register_namespace("cac", CAC_NS)
ET.register_namespace("cbc", CBC_NS)

logger = logging.getLogger(__name__)


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _add(
    parent: ET.Element,
    namespace: str,
    name: str,
    text: str | None = None,
    attributes: dict[str, str] | None = None,
) -> ET.Element:
    """Small ElementTree convenience wrapper with namespace support."""

    element = ET.SubElement(parent, _tag(namespace, name), attributes or {})
    if text is not None:
        element.text = str(text)
    return element


def _money(value: Decimal) -> str:
    return f"{parse_decimal(value):.2f}"


def _percent(value: Decimal) -> str:
    """Format a VAT rate without insignificant trailing zeros (18.00 -> 18)."""

    return format(parse_decimal(value).normalize(), "f")


def _country_prefixed_tax_id(tax_id: str, country: str) -> str:
    """Return a VAT identifier prefixed with its ISO 3166-1 alpha-2 code.

    EN 16931 BR-CO-09 requires seller and buyer VAT identifiers to start with
    the country code.  Existing prefixes are preserved rather than duplicated.
    """

    normalized = normalize_tax_id(tax_id)
    country_code = (country or "").strip().upper()
    if not normalized or not country_code:
        return normalized
    if normalized.startswith(country_code):
        return normalized
    return f"{country_code}{normalized}"


def _vat_category(vat_rate: Decimal) -> str:
    """Map the invoice VAT rate to the EN 16931 category used by this adapter."""

    return "S" if parse_decimal(vat_rate) > 0 else "Z"


class UJPService:
    """Generate, validate, and mock-submit UBL 2.1 invoices."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Domain validation
    # ------------------------------------------------------------------

    def validate_invoice(self, invoice: Invoice) -> UJPValidationResult:
        """Validate invoice data before XML generation."""

        result = UJPValidationResult()

        if not invoice.invoice_id.strip():
            result.add_error("Invoice ID is required.")
        if not invoice.client_name.strip():
            result.add_error("Client name is required.")
        if not invoice.client_tax_id.strip():
            result.add_error("Client tax ID (ЕДБ/ЕМБГ) is required for UJP e-Faktura.")
        elif not is_valid_tax_id(invoice.client_tax_id):
            result.add_warning(
                "Client tax ID does not match the expected structural pattern; verify it manually."
            )

        if not self.settings.business_name.strip():
            result.add_error("BUSINESS_NAME is missing in .env; add your legal business name.")
        if not self.settings.business_tax_id.strip():
            result.add_error("BUSINESS_TAX_ID is missing in .env; add your ЕДБ.")
        elif not is_valid_tax_id(self.settings.business_tax_id):
            result.add_warning(
                "BUSINESS_TAX_ID does not match the expected structural pattern; verify it manually."
            )

        if not invoice.currency or len(invoice.currency) != 3:
            result.add_error("Currency must be a three-letter ISO code (for example MKD or EUR).")

        if invoice.issue_date > invoice.due_date:
            result.add_error("Due date cannot be before the issue date.")

        if invoice.net_amount < 0 or invoice.vat_amount < 0 or invoice.total_amount <= 0:
            result.add_error("Invoice amounts must be positive (only the total may be positive).")
        else:
            if abs((invoice.net_amount + invoice.vat_amount) - invoice.total_amount) > Decimal("0.02"):
                result.add_error(
                    "Amounts are inconsistent: net + VAT must equal the total amount."
                )
            expected_vat = (invoice.net_amount * invoice.vat_rate / Decimal(100)).quantize(
                Decimal("0.01")
            )
            if invoice.net_amount > 0 and abs(expected_vat - invoice.vat_amount) > Decimal("0.05"):
                result.add_error(
                    "VAT amount does not match the stated VAT rate and net amount."
                )

        if invoice.vat_rate < 0 or invoice.vat_rate > 100:
            result.add_error("VAT rate must be between 0 and 100.")

        if not invoice.description.strip():
            result.add_warning("Line-item description is empty; a generic description will be used.")

        if invoice.client_email and not is_valid_email(invoice.client_email):
            result.add_warning("Client email is structurally invalid.")

        if not invoice.client_email and not invoice.client_phone:
            result.add_warning(
                "No client email or phone is stored; invoice creation can proceed, but "
                "reminders cannot be delivered until contact details are added."
            )

        return result

    # ------------------------------------------------------------------
    # UBL generation
    # ------------------------------------------------------------------

    def build_ubl(self, invoice: Invoice) -> str:
        """Build a UBL 2.1 Invoice XML document for the supplied invoice."""

        vat_category = _vat_category(invoice.vat_rate)
        root = ET.Element(_tag(UBL_NS, "Invoice"))
        _add(root, CBC_NS, "UBLVersionID", "2.1")
        _add(root, CBC_NS, "CustomizationID", "urn:cen.eu:en16931:2017")
        _add(root, CBC_NS, "ID", invoice.invoice_id)
        _add(root, CBC_NS, "IssueDate", invoice.issue_date.isoformat())
        _add(root, CBC_NS, "DueDate", invoice.due_date.isoformat())
        _add(root, CBC_NS, "InvoiceTypeCode", "380")
        _add(
            root,
            CBC_NS,
            "Note",
            f"Payment due by {invoice.due_date.isoformat()} "
            f"({invoice.description}).",
        )
        _add(
            root,
            CBC_NS,
            "DocumentCurrencyCode",
            invoice.currency,
        )

        self._append_party(
            parent=root,
            wrapper="AccountingSupplierParty",
            name=self.settings.business_name,
            tax_id=self.settings.business_tax_id,
            address=self.settings.business_address,
            city=self.settings.business_city,
            country=self.settings.business_country,
            email=self.settings.business_email,
        )
        self._append_party(
            parent=root,
            wrapper="AccountingCustomerParty",
            name=invoice.client_name,
            tax_id=invoice.client_tax_id,
            address=invoice.client_address or "",
            city=invoice.client_city or "",
            country=invoice.client_country or "MK",
            email=invoice.client_email or "",
        )

        payment_means = _add(root, CAC_NS, "PaymentMeans")
        _add(payment_means, CBC_NS, "PaymentMeansCode", "30")
        # BR-61: credit transfer (code 30) requires the payee account.
        payee_account = _add(payment_means, CAC_NS, "PayeeFinancialAccount")
        _add(
            payee_account,
            CBC_NS,
            "ID",
            self.settings.business_iban or DEFAULT_BUSINESS_IBAN,
        )

        payment_terms = _add(root, CAC_NS, "PaymentTerms")
        _add(
            payment_terms,
            CBC_NS,
            "Note",
            f"Payment due by {invoice.due_date.isoformat()}",
        )

        tax_total = _add(root, CAC_NS, "TaxTotal")
        _add(tax_total, CBC_NS, "TaxAmount", _money(invoice.vat_amount), {"currencyID": invoice.currency})
        tax_subtotal = _add(tax_total, CAC_NS, "TaxSubtotal")
        _add(
            tax_subtotal,
            CBC_NS,
            "TaxableAmount",
            _money(invoice.net_amount),
            {"currencyID": invoice.currency},
        )
        _add(
            tax_subtotal,
            CBC_NS,
            "TaxAmount",
            _money(invoice.vat_amount),
            {"currencyID": invoice.currency},
        )
        tax_category = _add(tax_subtotal, CAC_NS, "TaxCategory")
        _add(tax_category, CBC_NS, "ID", vat_category)
        _add(tax_category, CBC_NS, "Percent", _percent(invoice.vat_rate))
        tax_scheme = _add(tax_category, CAC_NS, "TaxScheme")
        _add(tax_scheme, CBC_NS, "ID", "VAT")

        monetary_total = _add(root, CAC_NS, "LegalMonetaryTotal")
        _add(
            monetary_total,
            CBC_NS,
            "LineExtensionAmount",
            _money(invoice.net_amount),
            {"currencyID": invoice.currency},
        )
        _add(
            monetary_total,
            CBC_NS,
            "TaxExclusiveAmount",
            _money(invoice.net_amount),
            {"currencyID": invoice.currency},
        )
        _add(
            monetary_total,
            CBC_NS,
            "TaxInclusiveAmount",
            _money(invoice.total_amount),
            {"currencyID": invoice.currency},
        )
        _add(
            monetary_total,
            CBC_NS,
            "PayableAmount",
            _money(invoice.total_amount),
            {"currencyID": invoice.currency},
        )

        line = _add(root, CAC_NS, "InvoiceLine")
        _add(line, CBC_NS, "ID", "1")
        _add(
            line,
            CBC_NS,
            "InvoicedQuantity",
            "1",
            {"unitCode": "H87"},
        )
        _add(
            line,
            CBC_NS,
            "LineExtensionAmount",
            _money(invoice.net_amount),
            {"currencyID": invoice.currency},
        )
        item = _add(line, CAC_NS, "Item")
        # UBL ItemType places Description, Name, then ClassifiedTaxCategory.
        _add(item, CBC_NS, "Description", invoice.description[:240])
        _add(item, CBC_NS, "Name", invoice.description[:120])
        # BR-CO-04 / UBL-SR-48 / BR-S-01: exactly one VAT category per line,
        # using the same code and percentage as the TaxSubtotal breakdown.
        classified_tax_category = _add(item, CAC_NS, "ClassifiedTaxCategory")
        _add(classified_tax_category, CBC_NS, "ID", vat_category)
        _add(classified_tax_category, CBC_NS, "Percent", _percent(invoice.vat_rate))
        line_tax_scheme = _add(classified_tax_category, CAC_NS, "TaxScheme")
        _add(line_tax_scheme, CBC_NS, "ID", "VAT")
        price = _add(line, CAC_NS, "Price")
        _add(
            price,
            CBC_NS,
            "PriceAmount",
            _money(invoice.net_amount),
            {"currencyID": invoice.currency},
        )

        ET.indent(root, space="  ")
        xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
        return xml_bytes.decode("utf-8")

    # ------------------------------------------------------------------
    # UBL persistence
    # ------------------------------------------------------------------

    def save_ubl_xml(
        self,
        invoice: Invoice,
        output_dir: Path | str | None = None,
        *,
        xml_content: str | None = None,
    ) -> Path:
        """Validate and save a UBL 2.1 XML audit copy.

        The default destination is ``output/<invoice_id>.xml`` (a sibling of
        the configured ``DATA_DIR``). With the standard project layout, for
        example, ``NPL-2026-0001`` is written to
        ``output/NPL-2026-0001.xml``.

        Passing ``output_dir`` lets callers use another directory such as a
        temporary folder in tests.

        Raises:
            ValueError: if the generated XML does not pass local structural and
                arithmetic validation.
            OSError: if the destination cannot be created or written.
        """

        content = xml_content if xml_content is not None else self.build_ubl(invoice)
        validation = self.validate_ubl(content)
        if not validation.is_valid:
            details = "; ".join(validation.errors) or "unknown validation error"
            raise ValueError(f"Generated UBL XML is invalid: {details}")

        target_dir = (
            Path(output_dir) if output_dir is not None else self.settings.ubl_output_dir
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        xml_path = (target_dir / f"{safe_filename(invoice.invoice_id)}.xml").resolve()
        xml_path.write_text(content, encoding="utf-8")
        logger.info("UBL 2.1 XML file created at %s", xml_path)
        return xml_path

    def _append_party(
        self,
        parent: ET.Element,
        wrapper: str,
        name: str,
        tax_id: str,
        address: str,
        city: str,
        country: str,
        email: str,
    ) -> None:
        """Append a UBL AccountingSupplierParty / AccountingCustomerParty."""

        wrapper_element = _add(parent, CAC_NS, wrapper)
        party = _add(wrapper_element, CAC_NS, "Party")

        if email:
            _add(party, CBC_NS, "EndpointID", email, {"schemeID": "EM"})

        party_name = _add(party, CAC_NS, "PartyName")
        _add(party_name, CBC_NS, "Name", name)

        if address or city or country:
            postal = _add(party, CAC_NS, "PostalAddress")
            if address:
                _add(postal, CBC_NS, "StreetName", address)
            if city:
                _add(postal, CBC_NS, "CityName", city)
            if country:
                country_element = _add(postal, CAC_NS, "Country")
                _add(
                    country_element,
                    CBC_NS,
                    "IdentificationCode",
                    country.upper(),
                    {"listID": "ISO3166-1:Alpha2"},
                )

        tax_scheme = _add(party, CAC_NS, "PartyTaxScheme")
        # BR-CO-09: VAT identifiers start with the party's ISO country code.
        _add(
            tax_scheme,
            CBC_NS,
            "CompanyID",
            _country_prefixed_tax_id(tax_id, country),
        )
        scheme = _add(tax_scheme, CAC_NS, "TaxScheme")
        _add(scheme, CBC_NS, "ID", "VAT")

        legal = _add(party, CAC_NS, "PartyLegalEntity")
        _add(legal, CBC_NS, "RegistrationName", name)

    # ------------------------------------------------------------------
    # UBL structural validation
    # ------------------------------------------------------------------

    def validate_ubl(self, xml_content: str) -> UJPValidationResult:
        """Validate generated XML structurally and arithmetically.

        This is intentionally a local mock validator, not an official UJP XSD or
        Schematron validation service.
        """

        result = UJPValidationResult()
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            result.add_error(f"Generated XML is not well-formed: {exc}")
            return result

        if root.tag != _tag(UBL_NS, "Invoice"):
            result.add_error("Root element must be a UBL Invoice in the Invoice-2 namespace.")

        def text(path: str) -> str:
            element = root.find(path, NS)
            return (element.text or "").strip() if element is not None else ""

        def child_text(element: ET.Element, path: str) -> str:
            child = element.find(path, NS)
            return (child.text or "").strip() if child is not None else ""

        if text(".//cbc:UBLVersionID") != "2.1":
            result.add_error("UBLVersionID must be 2.1.")
        if not text(".//cbc:ID"):
            result.add_error("Invoice ID is missing from the generated XML.")
        if not text(".//cbc:IssueDate"):
            result.add_error("IssueDate is missing from the generated XML.")
        if not text(".//cbc:DocumentCurrencyCode"):
            result.add_error("DocumentCurrencyCode is missing from the generated XML.")

        supplier_id = text(".//cac:AccountingSupplierParty//cbc:CompanyID")
        customer_id = text(".//cac:AccountingCustomerParty//cbc:CompanyID")
        if not supplier_id:
            result.add_error("Supplier tax ID is missing from the generated XML.")
        if not customer_id:
            result.add_error("Customer tax ID is missing from the generated XML.")

        # BR-CO-09: VAT identifiers must carry the party's ISO country prefix.
        for label, party_path in (
            ("Supplier", ".//cac:AccountingSupplierParty"),
            ("Customer", ".//cac:AccountingCustomerParty"),
        ):
            company_element = root.find(
                f"{party_path}/cac:Party/cac:PartyTaxScheme/cbc:CompanyID", NS
            )
            country_element = root.find(
                f"{party_path}/cac:Party/cac:PostalAddress/cac:Country/"
                "cbc:IdentificationCode",
                NS,
            )
            company_id = (
                (company_element.text or "").strip()
                if company_element is not None
                else ""
            )
            country_code = (
                (country_element.text or "").strip().upper()
                if country_element is not None
                else ""
            )
            if company_id and not country_code:
                result.add_error(f"{label} country code is missing from the generated XML.")
            elif company_id and not company_id.startswith(country_code):
                result.add_error(
                    f"{label} tax ID {company_id!r} must start with the ISO country "
                    f"code {country_code!r} (BR-CO-09)."
                )

        line_count = len(root.findall(".//cac:InvoiceLine", NS))
        if line_count < 1:
            result.add_error("At least one InvoiceLine is required.")

        subtotal_categories = root.findall(
            ".//cac:TaxTotal/cac:TaxSubtotal/cac:TaxCategory", NS
        )
        subtotal_category_id = ""
        subtotal_percent: Decimal | None = None
        if len(subtotal_categories) != 1:
            result.add_error("Exactly one TaxTotal/TaxSubtotal/TaxCategory is required.")
        else:
            subtotal_category = subtotal_categories[0]
            subtotal_category_id = child_text(subtotal_category, "cbc:ID")
            subtotal_percent_text = child_text(subtotal_category, "cbc:Percent")
            subtotal_scheme_id = child_text(
                subtotal_category, "cac:TaxScheme/cbc:ID"
            )
            if not subtotal_category_id:
                result.add_error("TaxSubtotal TaxCategory/ID is missing.")
            if subtotal_scheme_id != "VAT":
                result.add_error("TaxSubtotal TaxScheme/ID must be VAT.")
            try:
                subtotal_percent = parse_decimal(subtotal_percent_text)
            except ValueError:
                result.add_error(
                    "TaxSubtotal TaxCategory/Percent is missing or invalid."
                )

        if subtotal_percent is not None:
            expected_category = "S" if subtotal_percent > 0 else "Z"
            if subtotal_category_id != expected_category:
                result.add_error(
                    "TaxSubtotal TaxCategory/ID must be "
                    f"{expected_category!r} for a {subtotal_percent}% VAT rate."
                )

        # BR-CO-04 / UBL-SR-48 / BR-S-01 / BR-S-08: one line category that
        # exactly matches the subtotal category code and percentage.
        for index, line_element in enumerate(
            root.findall(".//cac:InvoiceLine", NS), start=1
        ):
            categories = line_element.findall(
                "cac:Item/cac:ClassifiedTaxCategory", NS
            )
            if len(categories) != 1:
                result.add_error(
                    f"InvoiceLine {index} must contain exactly one "
                    "Item/ClassifiedTaxCategory (BR-CO-04/UBL-SR-48)."
                )
                continue

            line_category = categories[0]
            line_category_id = child_text(line_category, "cbc:ID")
            line_percent_text = child_text(line_category, "cbc:Percent")
            line_scheme_id = child_text(line_category, "cac:TaxScheme/cbc:ID")
            if line_scheme_id != "VAT":
                result.add_error(
                    f"InvoiceLine {index} ClassifiedTaxCategory "
                    "TaxScheme/ID must be VAT."
                )
            if subtotal_category_id and line_category_id != subtotal_category_id:
                result.add_error(
                    f"InvoiceLine {index} VAT category {line_category_id!r} must "
                    f"match TaxSubtotal category {subtotal_category_id!r}."
                )
            try:
                line_percent = parse_decimal(line_percent_text)
            except ValueError:
                result.add_error(
                    f"InvoiceLine {index} ClassifiedTaxCategory/Percent is "
                    "missing or invalid."
                )
                continue
            if subtotal_percent is not None and line_percent != subtotal_percent:
                result.add_error(
                    f"InvoiceLine {index} VAT percentage {line_percent}% must "
                    f"match TaxSubtotal {subtotal_percent}% (BR-S-08)."
                )
            if line_percent > 0 and line_category_id != "S":
                result.add_error(
                    f"InvoiceLine {index} with positive VAT must use category "
                    "'S' (BR-S-01)."
                )
            if line_percent == 0 and line_category_id != "Z":
                result.add_error(
                    f"InvoiceLine {index} with zero VAT must use category 'Z'."
                )

        # BR-61: PaymentMeansCode 30 (credit transfer) needs a payee account.
        if text(".//cac:PaymentMeans/cbc:PaymentMeansCode") == "30":
            payee_account = text(
                ".//cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID"
            )
            if not payee_account:
                result.add_error(
                    "PaymentMeansCode 30 requires PayeeFinancialAccount/ID (BR-61)."
                )

        try:
            issue_date = parse_date(text(".//cbc:IssueDate"))
            due_date = parse_date(text(".//cbc:DueDate"))
            if due_date < issue_date:
                result.add_error("DueDate is before IssueDate in the generated XML.")
        except ValueError as exc:
            result.add_error(f"Invalid issue/due date in generated XML: {exc}")

        try:
            tax_exclusive = parse_decimal(text(".//cac:LegalMonetaryTotal/cbc:TaxExclusiveAmount"))
            tax_inclusive = parse_decimal(text(".//cac:LegalMonetaryTotal/cbc:TaxInclusiveAmount"))
            payable = parse_decimal(text(".//cac:LegalMonetaryTotal/cbc:PayableAmount"))
            tax_amount = parse_decimal(text(".//cac:TaxTotal/cbc:TaxAmount"))
            if payable <= 0:
                result.add_error("PayableAmount must be positive.")
            if abs((tax_exclusive + tax_amount) - payable) > Decimal("0.02"):
                result.add_error(
                    "XML totals are inconsistent: TaxExclusiveAmount + TaxAmount "
                    "must equal PayableAmount."
                )
            if abs(tax_inclusive - payable) > Decimal("0.02"):
                result.add_error("TaxInclusiveAmount and PayableAmount must match.")
        except ValueError as exc:
            result.add_error(f"XML monetary totals are invalid: {exc}")

        # Check that the invoice ID and currency are ISO-shaped.  This is not
        # business validation, just a quick regression guard for generated files.
        currency = text(".//cbc:DocumentCurrencyCode")
        if currency and (len(currency) != 3 or not currency.isalpha()):
            result.add_error(f"Invalid document currency code: {currency!r}.")

        return result

    # ------------------------------------------------------------------
    # Mock submission
    # ------------------------------------------------------------------

    def generate_and_submit(
        self,
        invoice: Invoice,
        output_dir: Path | str | None = None,
    ) -> UJPSubmissionResult:
        """Validate, generate XML, write it, and return a mock UJP reference."""

        validation = self.validate_invoice(invoice)
        if not validation.is_valid:
            return UJPSubmissionResult(
                invoice_id=invoice.invoice_id,
                success=False,
                validation=validation,
                message="Invoice did not pass validation; no UJP document was generated.",
            )

        xml_content = self.build_ubl(invoice)
        xml_validation = self.validate_ubl(xml_content)
        validation.errors.extend(xml_validation.errors)
        validation.warnings.extend(xml_validation.warnings)
        if not validation.is_valid:
            return UJPSubmissionResult(
                invoice_id=invoice.invoice_id,
                success=False,
                validation=validation,
                message="Generated UBL document did not pass structural validation.",
            )

        xml_path = self.save_ubl_xml(
            invoice,
            output_dir=output_dir,
            xml_content=xml_content,
        )

        reference = (
            f"MOCK-UJP-{datetime.now(timezone.utc):%Y%m%d%H%M%S}-"
            f"{uuid4().hex[:8].upper()}"
        )
        return UJPSubmissionResult(
            invoice_id=invoice.invoice_id,
            success=True,
            validation=validation,
            xml_path=str(xml_path),
            reference=reference,
            message=(
                "Mock UJP submission accepted. XML generated locally; "
                "no network request was made."
            ),
            mock=True,
        )
