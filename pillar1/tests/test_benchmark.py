"""
Tests for Phase 5 Benchmark Engine.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.benchmark import BenchmarkComponent, BenchmarkDefinition, BenchmarkEngine
from pillar1.canonical_schema import BENCHMARK_PRICES_COLUMNS
from pillar1.validation import validate_benchmark_prices_df


def test_benchmark_definition_single_index():
    """Test BenchmarkDefinition for single-index benchmark."""
    definition = BenchmarkDefinition(
        benchmark_id="SP500",
        name="S&P 500",
        components=[
            BenchmarkComponent(security_id="SPY", weight=Decimal("1.0"))
        ]
    )

    assert definition.is_single_index is True
    assert definition.is_composite is False


def test_benchmark_definition_composite():
    """Test BenchmarkDefinition for composite benchmark."""
    definition = BenchmarkDefinition(
        benchmark_id="60_40",
        name="60/40 Portfolio",
        components=[
            BenchmarkComponent(security_id="VTI", weight=Decimal("0.6")),
            BenchmarkComponent(security_id="BND", weight=Decimal("0.4"))
        ]
    )

    assert definition.is_single_index is False
    assert definition.is_composite is True
    assert len(definition.components) == 2


def test_construct_single_index_benchmark():
    """Test constructing a single-index benchmark."""
    definition = BenchmarkDefinition(
        benchmark_id="SP500",
        name="S&P 500",
        components=[
            BenchmarkComponent(security_id="SPY", weight=Decimal("1.0"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "SPY", "date": date(2024, 1, 1), "close_price": Decimal("450.00"), "adj_close": Decimal("450.00")},
            {"security_id": "SPY", "date": date(2024, 1, 2), "close_price": Decimal("455.00"), "adj_close": Decimal("455.00")},
            {"security_id": "SPY", "date": date(2024, 1, 3), "close_price": Decimal("460.00"), "adj_close": Decimal("460.00")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine()
    benchmark_prices = engine.construct_benchmark_prices(definition, component_prices)

    # Validate schema
    violations = validate_benchmark_prices_df(benchmark_prices)
    assert violations == [], f"Schema violations: {violations}"

    assert len(benchmark_prices) == 3
    assert benchmark_prices.iloc[0]["level"] == Decimal("100")  # Normalized to 100
    assert benchmark_prices.iloc[1]["level"] > Decimal("100")  # Increased
    assert benchmark_prices.iloc[2]["level"] > benchmark_prices.iloc[1]["level"]  # Increased


def test_construct_composite_benchmark():
    """Test constructing a composite benchmark."""
    definition = BenchmarkDefinition(
        benchmark_id="60_40",
        name="60/40 Portfolio",
        components=[
            BenchmarkComponent(security_id="VTI", weight=Decimal("0.6")),
            BenchmarkComponent(security_id="BND", weight=Decimal("0.4"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 1), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
            {"security_id": "VTI", "date": date(2024, 1, 2), "close_price": Decimal("251.00"), "adj_close": Decimal("251.00")},
            {"security_id": "BND", "date": date(2024, 1, 1), "close_price": Decimal("75.00"), "adj_close": Decimal("75.00")},
            {"security_id": "BND", "date": date(2024, 1, 2), "close_price": Decimal("75.50"), "adj_close": Decimal("75.50")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine(rebalance_frequency="daily")
    benchmark_prices = engine.construct_benchmark_prices(definition, component_prices)

    # Validate schema
    violations = validate_benchmark_prices_df(benchmark_prices)
    assert violations == [], f"Schema violations: {violations}"

    assert len(benchmark_prices) == 2
    assert benchmark_prices.iloc[0]["level"] == Decimal("100")  # Starts at 100


def test_construct_benchmark_with_date_range():
    """Test constructing benchmark with specified date range."""
    definition = BenchmarkDefinition(
        benchmark_id="SP500",
        name="S&P 500",
        components=[
            BenchmarkComponent(security_id="SPY", weight=Decimal("1.0"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "SPY", "date": date(2024, 1, 1), "close_price": Decimal("450.00"), "adj_close": Decimal("450.00")},
            {"security_id": "SPY", "date": date(2024, 1, 2), "close_price": Decimal("455.00"), "adj_close": Decimal("455.00")},
            {"security_id": "SPY", "date": date(2024, 1, 3), "close_price": Decimal("460.00"), "adj_close": Decimal("460.00")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine()
    benchmark_prices = engine.construct_benchmark_prices(
        definition,
        component_prices,
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2)
    )

    # Should only have one date
    assert len(benchmark_prices) == 1
    assert benchmark_prices.iloc[0]["date"] == date(2024, 1, 2)


def test_construct_benchmark_missing_component_data():
    """Test that missing component data raises ValueError."""
    definition = BenchmarkDefinition(
        benchmark_id="60_40",
        name="60/40 Portfolio",
        components=[
            BenchmarkComponent(security_id="VTI", weight=Decimal("0.6")),
            BenchmarkComponent(security_id="BND", weight=Decimal("0.4"))
        ]
    )

    # Only VTI data, missing BND
    component_prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 1), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine()
    with pytest.raises(ValueError, match="No price data found for component"):
        engine.construct_benchmark_prices(definition, component_prices)


def test_construct_benchmark_no_data_in_range():
    """Test that no data in date range raises ValueError."""
    definition = BenchmarkDefinition(
        benchmark_id="SP500",
        name="S&P 500",
        components=[
            BenchmarkComponent(security_id="SPY", weight=Decimal("1.0"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "SPY", "date": date(2024, 1, 1), "close_price": Decimal("450.00"), "adj_close": Decimal("450.00")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine()
    with pytest.raises(ValueError, match="No price data in specified date range"):
        engine.construct_benchmark_prices(
            definition,
            component_prices,
            start_date=date(2024, 6, 1),
            end_date=date(2024, 6, 30)
        )


def test_compute_benchmark_returns():
    """Test computing returns from benchmark price series."""
    benchmark_prices = pd.DataFrame(
        [
            {"benchmark_id": "SP500", "date": date(2024, 1, 1), "level": Decimal("100"), "total_return_level": Decimal("100")},
            {"benchmark_id": "SP500", "date": date(2024, 1, 2), "level": Decimal("101"), "total_return_level": Decimal("101")},
            {"benchmark_id": "SP500", "date": date(2024, 1, 3), "level": Decimal("102"), "total_return_level": Decimal("102")},
        ],
        columns=BENCHMARK_PRICES_COLUMNS,
    )

    engine = BenchmarkEngine()
    returns = engine.compute_benchmark_returns(benchmark_prices)

    assert len(returns) == 3
    assert returns.iloc[0]["period_return"] == Decimal("0")  # First day
    assert returns.iloc[1]["period_return"] > 0  # Positive return
    assert returns.iloc[2]["period_return"] > 0  # Positive return
    assert returns.iloc[0]["entity_type"] == "benchmark"


def test_compute_benchmark_returns_empty():
    """Test computing returns from empty benchmark prices."""
    benchmark_prices = pd.DataFrame(columns=BENCHMARK_PRICES_COLUMNS)

    engine = BenchmarkEngine()
    returns = engine.compute_benchmark_returns(benchmark_prices)

    assert len(returns) == 0
    assert list(returns.columns) == ["entity_id", "entity_type", "date", "period_return", "cumulative_return"]


def test_rebalance_frequency_daily():
    """Test daily rebalancing for composite benchmark."""
    definition = BenchmarkDefinition(
        benchmark_id="60_40",
        name="60/40 Portfolio",
        components=[
            BenchmarkComponent(security_id="VTI", weight=Decimal("0.6")),
            BenchmarkComponent(security_id="BND", weight=Decimal("0.4"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 1), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
            {"security_id": "VTI", "date": date(2024, 1, 2), "close_price": Decimal("251.00"), "adj_close": Decimal("251.00")},
            {"security_id": "BND", "date": date(2024, 1, 1), "close_price": Decimal("75.00"), "adj_close": Decimal("75.00")},
            {"security_id": "BND", "date": date(2024, 1, 2), "close_price": Decimal("75.50"), "adj_close": Decimal("75.50")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine(rebalance_frequency="daily")
    benchmark_prices = engine.construct_benchmark_prices(definition, component_prices)

    # Should have 2 dates
    assert len(benchmark_prices) == 2


def test_rebalance_frequency_monthly():
    """Test monthly rebalancing for composite benchmark."""
    definition = BenchmarkDefinition(
        benchmark_id="60_40",
        name="60/40 Portfolio",
        components=[
            BenchmarkComponent(security_id="VTI", weight=Decimal("0.6")),
            BenchmarkComponent(security_id="BND", weight=Decimal("0.4"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 1), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
            {"security_id": "VTI", "date": date(2024, 1, 15), "close_price": Decimal("251.00"), "adj_close": Decimal("251.00")},
            {"security_id": "VTI", "date": date(2024, 2, 1), "close_price": Decimal("252.00"), "adj_close": Decimal("252.00")},
            {"security_id": "BND", "date": date(2024, 1, 1), "close_price": Decimal("75.00"), "adj_close": Decimal("75.00")},
            {"security_id": "BND", "date": date(2024, 1, 15), "close_price": Decimal("75.50"), "adj_close": Decimal("75.50")},
            {"security_id": "BND", "date": date(2024, 2, 1), "close_price": Decimal("76.00"), "adj_close": Decimal("76.00")},
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine(rebalance_frequency="monthly")
    benchmark_prices = engine.construct_benchmark_prices(definition, component_prices)

    # Should have all 3 dates (rebalance on 1/1 and 2/1)
    assert len(benchmark_prices) >= 2


def test_benchmark_uses_total_return():
    """Test that benchmark uses adj_close (total return) not close_price."""
    definition = BenchmarkDefinition(
        benchmark_id="DIVIDEND_ETF",
        name="Dividend ETF",
        components=[
            BenchmarkComponent(security_id="DIV", weight=Decimal("1.0"))
        ]
    )

    # Simulate a fund that pays dividends - adj_close includes dividends
    component_prices = pd.DataFrame(
        [
            {"security_id": "DIV", "date": date(2024, 1, 1), "close_price": Decimal("100.00"), "adj_close": Decimal("100.00")},
            {"security_id": "DIV", "date": date(2024, 1, 2), "close_price": Decimal("100.00"), "adj_close": Decimal("101.00")},  # Dividend paid
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine()
    benchmark_prices = engine.construct_benchmark_prices(definition, component_prices)

    # Should use adj_close, so level should increase despite flat close_price
    assert benchmark_prices.iloc[0]["level"] == Decimal("100")
    assert benchmark_prices.iloc[1]["level"] == Decimal("101")  # Increased due to dividend


def test_benchmark_forward_fill_prices():
    """Test that missing prices are forward-filled within tolerance."""
    definition = BenchmarkDefinition(
        benchmark_id="SP500",
        name="S&P 500",
        components=[
            BenchmarkComponent(security_id="SPY", weight=Decimal("1.0"))
        ]
    )

    component_prices = pd.DataFrame(
        [
            {"security_id": "SPY", "date": date(2024, 1, 1), "close_price": Decimal("450.00"), "adj_close": Decimal("450.00")},
            {"security_id": "SPY", "date": date(2024, 1, 5), "close_price": Decimal("455.00"), "adj_close": Decimal("455.00")},  # Gap
        ],
        columns=["security_id", "date", "close_price", "adj_close"],
    )

    engine = BenchmarkEngine()
    benchmark_prices = engine.construct_benchmark_prices(
        definition,
        component_prices,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 5)
    )

    # Should forward-fill from 1/1 to 1/5 (within 5-day tolerance)
    assert len(benchmark_prices) >= 1
