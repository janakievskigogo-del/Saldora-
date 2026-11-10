"""Tests for the local mock UJP e-Faktura / UBL service."""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from config import Settings
from legacy_models import Invoice
from ujp_service import NS, UBL_NS, UJPService


def test_validate_invoice_success(settings: Settings, invoice: Invoice) -> None:
    result = UJPService(settings).validate_invoice(invoice)
    assert result.is_valid
    assert result.errors == []
    assert result.warnings == []


def test_build_and_validate_ubl(settings: Settings, invoice: Invoice) -> None:
    service = UJPService(settings)
    xml = service.build_ubl(invoice)
    assert "<cbc:UBLVersionID>2.1</cbc:UBLVersionID>" in xml
    assert "MK4030000000000" in xml
    assert "Test Company DOOEL" in xml

    root = ET.fromstring(xml)
    assert root.tag == f"{{{UBL_NS}}}Invoice"
    currency = root.find(".//cbc:DocumentCurrencyCode", NS)
    assert currency is not None and currency.text == "MKD"

    validation = service.validate_ubl(xml)
    assert validation.is_valid
    assert validation.errors == []


def test_ubl_emits_schema_ordered_elements(settings: Settings, invoice: Invoice) -> None:
    root = ET.fromstring(UJPService(settings).build_ubl(invoice))
    root_children = [child.tag.rsplit("}", 1)[-1] for child in root]

    assert root_children.index("Note") < root_children.index("DocumentCurrencyCode")
    assert root_children.index("AccountingSupplierParty") < root_children.index("TaxTotal")
    assert root_children.index("TaxTotal") < root_children.index("InvoiceLine")

    item = root.find(".//cac:InvoiceLine/cac:Item", NS)
    assert item is not None
    item_children = [child.tag.rsplit("}", 1)[-1] for child in item]
    assert item_children.index("Description") < item_children.index("Name")
    assert item_children.index("Name") < item_children.index("ClassifiedTaxCategory")


def test_party_tax_ids_have_country_prefix(settings: Settings, invoice: Invoice) -> None:
    root = ET.fromstring(UJPService(settings).build_ubl(invoice))

    supplier_id = root.findtext(
        ".//cac:AccountingSupplierParty//cbc:CompanyID", namespaces=NS
    )
    customer_id = root.findtext(
        ".//cac:AccountingCustomerParty//cbc:CompanyID", namespaces=NS
    )

    assert supplier_id == "MK4030000000000"
    assert customer_id == "MK4030000000000"


def test_line_vat_category_matches_tax_subtotal(
    settings: Settings, invoice: Invoice
) -> None:
    root = ET.fromstring(UJPService(settings).build_ubl(invoice))

    line_categories = root.findall(
        ".//cac:InvoiceLine/cac:Item/cac:ClassifiedTaxCategory", NS
    )
    assert len(line_categories) == 1
    line_category = line_categories[0]
    assert line_category.findtext("cbc:ID", namespaces=NS) == "S"
    assert line_category.findtext("cbc:Percent", namespaces=NS) == "18"
    assert line_category.findtext("cac:TaxScheme/cbc:ID", namespaces=NS) == "VAT"

    subtotal = root.find(".//cac:TaxTotal/cac:TaxSubtotal/cac:TaxCategory", NS)
    assert subtotal is not None
    assert subtotal.findtext("cbc:ID", namespaces=NS) == "S"
    assert subtotal.findtext("cbc:Percent", namespaces=NS) == "18"


def test_payment_means_30_includes_seller_iban(
    settings: Settings, invoice: Invoice
) -> None:
    root = ET.fromstring(UJPService(settings).build_ubl(invoice))

    assert (
        root.findtext(".//cac:PaymentMeans/cbc:PaymentMeansCode", namespaces=NS)
        == "30"
    )
    assert (
        root.findtext(
            ".//cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID",
            namespaces=NS,
        )
        == "MK07200000000000000"
    )


def test_configured_business_iban_is_used(settings: Settings, invoice: Invoice) -> None:
    settings = replace(settings, business_iban="MK330000000000000000")
    root = ET.fromstring(UJPService(settings).build_ubl(invoice))

    assert (
        root.findtext(
            ".//cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID",
            namespaces=NS,
        )
        == "MK330000000000000000"
    )


def test_generate_and_submit_writes_xml(
    settings: Settings,
    invoice: Invoice,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "ujp"
    submission = UJPService(settings).generate_and_submit(invoice, output_dir=output_dir)

    assert submission.success
    assert submission.mock is True
    assert submission.reference is not None
    assert submission.reference.startswith("MOCK-UJP-")
    assert submission.xml_path is not None
    xml_path = Path(submission.xml_path)
    assert xml_path.exists()
    assert invoice.invoice_id in xml_path.read_text(encoding="utf-8")


def test_save_ubl_xml_uses_output_directory_and_is_well_formed(
    settings: Settings,
    invoice: Invoice,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="ujp_service")
    service = UJPService(settings)

    xml_path = service.save_ubl_xml(invoice)

    assert xml_path == (settings.data_dir.parent / "output" / f"{invoice.invoice_id}.xml").resolve()
    assert xml_path.exists()
    content = xml_path.read_text(encoding="utf-8")
    assert "<cbc:UBLVersionID>2.1</cbc:UBLVersionID>" in content
    assert service.validate_ubl(content).is_valid
    assert f"UBL 2.1 XML file created at {xml_path}" in caplog.text


def test_save_ubl_xml_accepts_custom_output_directory(
    settings: Settings,
    invoice: Invoice,
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"

    xml_path = UJPService(settings).save_ubl_xml(invoice, output_dir=output)

    assert xml_path.parent == output
    assert xml_path.exists()


def test_save_ubl_xml_rejects_invalid_document(
    settings: Settings,
    invoice: Invoice,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="invalid"):
        UJPService(settings).save_ubl_xml(
            invoice,
            output_dir=tmp_path / "xml",
            xml_content="<not-xml",
        )


def test_inconsistent_amounts_are_rejected(settings: Settings, invoice: Invoice) -> None:
    broken = replace(invoice, vat_amount=Decimal("200.00"), total_amount=Decimal("1200.00"))
    result = UJPService(settings).validate_invoice(broken)
    assert not result.is_valid
    assert any("VAT amount" in error for error in result.errors)


def test_due_date_before_issue_date_is_rejected(settings: Settings, invoice: Invoice) -> None:
    broken = replace(invoice, due_date=invoice.issue_date - timedelta(days=1))
    result = UJPService(settings).validate_invoice(broken)
    assert not result.is_valid
    assert any("Due date" in error for error in result.errors)


def test_missing_business_identity_is_rejected(tmp_path: Path, invoice: Invoice) -> None:
    settings = Settings(data_dir=tmp_path)
    result = UJPService(settings).validate_invoice(invoice)
    assert not result.is_valid
    assert any("BUSINESS_NAME" in error for error in result.errors)
    assert any("BUSINESS_TAX_ID" in error for error in result.errors)


def test_invalid_submission_does_not_write_file(
    settings: Settings,
    invoice: Invoice,
    tmp_path: Path,
) -> None:
    broken = replace(invoice, due_date=invoice.issue_date - timedelta(days=1))
    submission = UJPService(settings).generate_and_submit(broken, output_dir=tmp_path / "xml")
    assert not submission.success
    assert submission.xml_path is None
    assert not (tmp_path / "xml").exists()


def test_malformed_xml_is_rejected(settings: Settings) -> None:
    validation = UJPService(settings).validate_ubl("<not-xml")
    assert not validation.is_valid
    assert any("well-formed" in error for error in validation.errors)


def test_tampered_totals_are_rejected(settings: Settings, invoice: Invoice) -> None:
    service = UJPService(settings)
    root = ET.fromstring(service.build_ubl(invoice))
    payable = root.find(".//cac:LegalMonetaryTotal/cbc:PayableAmount", NS)
    assert payable is not None
    payable.text = "9999.00"
    tampered = ET.tostring(root, encoding="unicode")

    validation = service.validate_ubl(tampered)
    assert not validation.is_valid
    assert any("totals are inconsistent" in error for error in validation.errors)


def test_missing_line_tax_category_is_rejected(
    settings: Settings, invoice: Invoice
) -> None:
    service = UJPService(settings)
    root = ET.fromstring(service.build_ubl(invoice))
    item = root.find(".//cac:InvoiceLine/cac:Item", NS)
    assert item is not None
    category = item.find("cac:ClassifiedTaxCategory", NS)
    assert category is not None
    item.remove(category)

    validation = service.validate_ubl(ET.tostring(root, encoding="unicode"))
    assert not validation.is_valid
    assert any("exactly one" in error for error in validation.errors)


def test_line_tax_percentage_must_match_subtotal(
    settings: Settings, invoice: Invoice
) -> None:
    service = UJPService(settings)
    root = ET.fromstring(service.build_ubl(invoice))
    percent = root.find(
        ".//cac:InvoiceLine/cac:Item/cac:ClassifiedTaxCategory/cbc:Percent", NS
    )
    assert percent is not None
    percent.text = "20.00"

    validation = service.validate_ubl(ET.tostring(root, encoding="unicode"))
    assert not validation.is_valid
    assert any("BR-S-08" in error for error in validation.errors)


def test_missing_tax_id_country_prefix_is_rejected(
    settings: Settings, invoice: Invoice
) -> None:
    service = UJPService(settings)
    root = ET.fromstring(service.build_ubl(invoice))
    company_id = root.find(
        ".//cac:AccountingSupplierParty//cbc:CompanyID", NS
    )
    assert company_id is not None
    company_id.text = "4030000000000"

    validation = service.validate_ubl(ET.tostring(root, encoding="unicode"))
    assert not validation.is_valid
    assert any("BR-CO-09" in error for error in validation.errors)


def test_missing_payee_account_is_rejected(settings: Settings, invoice: Invoice) -> None:
    service = UJPService(settings)
    root = ET.fromstring(service.build_ubl(invoice))
    payee_account = root.find(
        ".//cac:PaymentMeans/cac:PayeeFinancialAccount/cbc:ID", NS
    )
    assert payee_account is not None
    payee_account.text = ""

    validation = service.validate_ubl(ET.tostring(root, encoding="unicode"))
    assert not validation.is_valid
    assert any("BR-61" in error for error in validation.errors)


def test_structurally_unusual_client_tax_id_warns(settings: Settings, invoice: Invoice) -> None:
    invoice = replace(invoice, client_tax_id="BAD")
    result = UJPService(settings).validate_invoice(invoice)
    assert result.is_valid
    assert any("Client tax ID" in warning for warning in result.warnings)


def test_invalid_currency_is_rejected(settings: Settings, invoice: Invoice) -> None:
    broken = replace(invoice, currency="EURO")
    result = UJPService(settings).validate_invoice(broken)
    assert not result.is_valid
    assert any("Currency" in error for error in result.errors)


def test_zero_vat_invoice_is_valid(settings: Settings, invoice: Invoice) -> None:
    zero_vat = replace(
        invoice,
        net_amount=Decimal("1000.00"),
        vat_rate=Decimal("0.00"),
        vat_amount=Decimal("0.00"),
        total_amount=Decimal("1000.00"),
    )
    service = UJPService(settings)
    xml = service.build_ubl(zero_vat)
    assert service.validate_invoice(zero_vat).is_valid
    assert service.validate_ubl(xml).is_valid

    root = ET.fromstring(xml)
    line_category = root.find(
        ".//cac:InvoiceLine/cac:Item/cac:ClassifiedTaxCategory", NS
    )
    assert line_category is not None
    assert line_category.findtext("cbc:ID", namespaces=NS) == "Z"
    assert line_category.findtext("cbc:Percent", namespaces=NS) == "0"


def test_due_date_required_by_ubl(settings: Settings, invoice: Invoice) -> None:
    # The model always has a due date; this guards the XML boundary if the
    # generator is changed later.
    xml = UJPService(settings).build_ubl(invoice)
    assert f"<cbc:DueDate>{invoice.due_date.isoformat()}</cbc:DueDate>" in xml
    assert date(2026, 10, 16).isoformat() in xml
