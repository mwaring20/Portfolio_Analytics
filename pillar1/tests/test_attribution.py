"""
Tests for Phase 8 Attribution Analysis.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.attribution import AttributionEngine, AttributionResult, AttributionSummary
from pillar1.canonical_schema import BENCHMARK_PRICES_COLUMNS
from pillar1.security_master import AssetClass, SecurityMaster, SecurityType


def test_attribution_basic_case():
    """Test basic Brinson-Fachler attribution."""
    # Portfolio: 60% equity, 40% fixed income
    # Benchmark: 100% equity
    # Equity outperforms fixed income

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("6000.00"),
                "price": Decimal("600.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 1),
                "quantity": Decimal("20"),
                "market_value": Decimal("4000.00"),
                "price": Decimal("200.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("6300.00"),  # 5% gain
                "price": Decimal("630.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("20"),
                "market_value": Decimal("4040.00"),  # 1% gain
                "price": Decimal("202.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    # Benchmark: 100% equity, 5% return
    benchmark_prices = pd.DataFrame(
        [
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 1),
                "level": Decimal("100"),
                "total_return_level": Decimal("100"),
            },
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 31),
                "level": Decimal("105"),
                "total_return_level": Decimal("105"),  # 5% return
            },
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    security_master = {
        "VTI": SecurityMaster(
            security_id="VTI",
            cusip="922908363",
            ticker="VTI",
            name="Vanguard Total Stock Market ETF",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.EQUITY,
            sector=None,
            geography="US",
            expense_ratio=Decimal("0.0003"),
        ),
        "BND": SecurityMaster(
            security_id="BND",
            cusip="921937835",
            ticker="BND",
            name="Vanguard Total Bond Market ETF",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.FIXED_INCOME,
            sector=None,
            geography="US",
            expense_ratio=Decimal("0.0003"),
        ),
        "SP500": SecurityMaster(
            security_id="SP500",
            cusip=None,
            ticker="SP500",
            name="S&P 500",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.EQUITY,
            sector=None,
            geography="US",
            expense_ratio=None,
        ),
    }

    engine = AttributionEngine()
    attribution = engine.compute_attribution(
        daily_positions,
        benchmark_prices,
        security_master,
        date(2024, 1, 1),
        date(2024, 1, 31),
        dimension="asset_class",
    )

    # Portfolio return: (10340 - 10000) / 10000 = 3.4%
    # Benchmark return: 5%
    # Active return: -1.6%
    assert attribution.portfolio_return > Decimal("0.03")
    assert attribution.portfolio_return < Decimal("0.04")
    assert attribution.benchmark_return == Decimal("0.05")
    assert attribution.active_return < 0  # Underperformed

    # Should have category results
    assert len(attribution.by_category) >= 1


def test_attribution_empty_positions():
    """Test attribution with empty positions."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    benchmark_prices = pd.DataFrame(
        [
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 1),
                "level": Decimal("100"),
                "total_return_level": Decimal("100"),
            },
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 31),
                "level": Decimal("105"),
                "total_return_level": Decimal("105"),
            },
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    security_master = {}

    engine = AttributionEngine()
    attribution = engine.compute_attribution(
        daily_positions,
        benchmark_prices,
        security_master,
        date(2024, 1, 1),
        date(2024, 1, 31),
    )

    assert attribution.portfolio_return == Decimal("0")
    assert attribution.benchmark_return == Decimal("0.05")
    assert attribution.active_return == Decimal("-0.05")


def test_attribution_missing_dates():
    """Test attribution when start or end date is missing."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("6000.00"),
                "price": Decimal("600.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    benchmark_prices = pd.DataFrame(
        [
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 1),
                "level": Decimal("100"),
                "total_return_level": Decimal("100"),
            },
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    security_master = {
        "VTI": SecurityMaster(
            security_id="VTI",
            cusip="922908363",
            ticker="VTI",
            name="Vanguard Total Stock Market ETF",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.EQUITY,
            sector=None,
            geography="US",
            expense_ratio=Decimal("0.0003"),
        ),
    }

    engine = AttributionEngine()
    attribution = engine.compute_attribution(
        daily_positions,
        benchmark_prices,
        security_master,
        date(2024, 1, 1),
        date(2024, 1, 31),
    )

    # Should return zeros when data is missing
    assert attribution.portfolio_return == Decimal("0")


def test_attribution_sector_dimension():
    """Test attribution by sector dimension."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("2000.00"),
                "price": Decimal("200.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("2100.00"),  # 5% gain
                "price": Decimal("210.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    benchmark_prices = pd.DataFrame(
        [
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 1),
                "level": Decimal("100"),
                "total_return_level": Decimal("100"),
            },
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 31),
                "level": Decimal("105"),
                "total_return_level": Decimal("105"),
            },
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    security_master = {
        "AAPL": SecurityMaster(
            security_id="AAPL",
            cusip="037833100",
            ticker="AAPL",
            name="Apple Inc.",
            security_type=SecurityType.EQUITY,
            asset_class=AssetClass.EQUITY,
            sector="Technology",
            geography="US",
            expense_ratio=Decimal("0"),
        ),
        "SP500": SecurityMaster(
            security_id="SP500",
            cusip=None,
            ticker="SP500",
            name="S&P 500",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.EQUITY,
            sector=None,
            geography="US",
            expense_ratio=None,
        ),
    }

    engine = AttributionEngine()
    attribution = engine.compute_attribution(
        daily_positions,
        benchmark_prices,
        security_master,
        date(2024, 1, 1),
        date(2024, 1, 31),
        dimension="sector",
    )

    # Should have sector-level attribution
    assert len(attribution.by_category) >= 1
    # Technology sector should be present
    tech_categories = [c for c in attribution.by_category if c.category == "Technology"]
    assert len(tech_categories) >= 1


def test_attribution_effects_sum_correctly():
    """Test that allocation + selection + interaction = total effect."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("6000.00"),
                "price": Decimal("600.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 1),
                "quantity": Decimal("20"),
                "market_value": Decimal("4000.00"),
                "price": Decimal("200.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("6300.00"),
                "price": Decimal("630.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("20"),
                "market_value": Decimal("4040.00"),
                "price": Decimal("202.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    benchmark_prices = pd.DataFrame(
        [
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 1),
                "level": Decimal("100"),
                "total_return_level": Decimal("100"),
            },
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 31),
                "level": Decimal("105"),
                "total_return_level": Decimal("105"),
            },
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    security_master = {
        "VTI": SecurityMaster(
            security_id="VTI",
            cusip="922908363",
            ticker="VTI",
            name="Vanguard Total Stock Market ETF",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.EQUITY,
            sector=None,
            geography="US",
            expense_ratio=Decimal("0.0003"),
        ),
        "BND": SecurityMaster(
            security_id="BND",
            cusip="921937835",
            ticker="BND",
            name="Vanguard Total Bond Market ETF",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.FIXED_INCOME,
            sector=None,
            geography="US",
            expense_ratio=Decimal("0.0003"),
        ),
        "SP500": SecurityMaster(
            security_id="SP500",
            cusip=None,
            ticker="SP500",
            name="S&P 500",
            security_type=SecurityType.ETF,
            asset_class=AssetClass.EQUITY,
            sector=None,
            geography="US",
            expense_ratio=None,
        ),
    }

    engine = AttributionEngine()
    attribution = engine.compute_attribution(
        daily_positions,
        benchmark_prices,
        security_master,
        date(2024, 1, 1),
        date(2024, 1, 31),
    )

    # Check that total_active_return equals allocation + selection + interaction
    expected_total = (
        attribution.allocation_effect
        + attribution.selection_effect
        + attribution.interaction_effect
    )
    assert abs(attribution.total_active_return - expected_total) < Decimal("0.0001")


def test_attribution_result_structure():
    """Test that AttributionResult has all required fields."""
    result = AttributionResult(
        category="Equity",
        portfolio_weight=Decimal("0.6"),
        benchmark_weight=Decimal("1.0"),
        portfolio_return=Decimal("0.05"),
        benchmark_return=Decimal("0.05"),
        allocation_effect=Decimal("-0.02"),
        selection_effect=Decimal("0.0"),
        interaction_effect=Decimal("0.0"),
        total_effect=Decimal("-0.02"),
    )

    assert result.category == "Equity"
    assert result.portfolio_weight == Decimal("0.6")
    assert result.benchmark_weight == Decimal("1.0")
    assert result.allocation_effect == Decimal("-0.02")


def test_attribution_summary_structure():
    """Test that AttributionSummary has all required fields."""
    summary = AttributionSummary(
        portfolio_return=Decimal("0.034"),
        benchmark_return=Decimal("0.05"),
        active_return=Decimal("-0.016"),
        allocation_effect=Decimal("-0.02"),
        selection_effect=Decimal("0.0"),
        interaction_effect=Decimal("0.004"),
        total_active_return=Decimal("-0.016"),
        by_category=[],
    )

    assert summary.portfolio_return == Decimal("0.034")
    assert summary.benchmark_return == Decimal("0.05")
    assert summary.active_return == Decimal("-0.016")
    assert len(summary.by_category) == 0


def test_attribution_unknown_dimension():
    """Test that unknown dimension raises ValueError."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    benchmark_prices = pd.DataFrame(columns=BENCHMARK_PRICES_COLUMNS)

    security_master = {}

    engine = AttributionEngine()
    with pytest.raises(ValueError, match="Unknown dimension"):
        engine.compute_attribution(
            daily_positions,
            benchmark_prices,
            security_master,
            date(2024, 1, 1),
            date(2024, 1, 31),
            dimension="unknown",
        )


def test_compute_total_portfolio_return():
    """Test total portfolio return calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("10000.00"),
                "price": Decimal("1000.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("10500.00"),
                "price": Decimal("1050.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = AttributionEngine()
    total_return = engine._compute_total_portfolio_return(
        daily_positions, date(2024, 1, 1), date(2024, 1, 31)
    )

    # (10500 - 10000) / 10000 = 0.05 = 5%
    assert total_return == Decimal("0.05")


def test_compute_total_benchmark_return():
    """Test total benchmark return calculation."""
    benchmark_prices = pd.DataFrame(
        [
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 1),
                "level": Decimal("100"),
                "total_return_level": Decimal("100"),
            },
            {
                "benchmark_id": "SP500",
                "date": date(2024, 1, 31),
                "level": Decimal("105"),
                "total_return_level": Decimal("105"),
            },
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    engine = AttributionEngine()
    total_return = engine._compute_total_benchmark_return(
        benchmark_prices, date(2024, 1, 1), date(2024, 1, 31)
    )

    # (105 - 100) / 100 = 0.05 = 5%
    assert total_return == Decimal("0.05")
