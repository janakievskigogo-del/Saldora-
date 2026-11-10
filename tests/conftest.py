"""Shared pytest fixtures for the offline MVP test suite."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from config import Settings
from database import init_db
from legacy_models import Channel, Invoice, Language

# Plain TestClient usage skips the FastAPI lifespan, so apply the idempotent
# startup migration (missing columns / tables) before any endpoint test runs.
init_db()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Isolated settings with a valid mock business identity."""

    return Settings(
        business_name="Test Company DOOEL",
        business_tax_id="4030000000000",
        business_address="Test Street 1",
        business_city="Skopje",
        business_country="MK",
        business_email="billing@test.mk",
        default_currency="MKD",
        default_vat_rate=Decimal("18.00"),
        default_payment_terms_days=15,
        default_language=Language.MK,
        default_channel=Channel.EMAIL,
        invoice_prefix="TST",
        data_dir=tmp_path / "data",
        timezone="Europe/Skopje",
    )


@pytest.fixture
def invoice() -> Invoice:
    """A valid invoice that can be used by UJP, store, and tone tests."""

    return Invoice(
        invoice_id="TST-2026-0001",
        client_name="Acme LLC",
        client_tax_id="MK4030000000000",
        net_amount=Decimal("1000.00"),
        vat_rate=Decimal("18.00"),
        vat_amount=Decimal("180.00"),
        total_amount=Decimal("1180.00"),
        currency="MKD",
        issue_date=date(2026, 10, 1),
        due_date=date(2026, 10, 16),
        status="UNPAID",
        language=Language.EN,
        channel=Channel.EMAIL,
        client_email="billing@acme.com",
        client_phone="+38970123456",
        client_city="Skopje",
        client_country="MK",
        description="Consulting services",
    )
