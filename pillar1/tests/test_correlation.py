"""
Tests for Phase 10 Correlation Matrix.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.correlation import CorrelationEngine, LookbackWindow


def test_compute_correlation_matrix():
    """Test basic correlation matrix computation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("20"),
                "market_value": Decimal("1500.00"),
                "price": Decimal("75.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    # Create price data with some correlation
    prices = []
    base_date = date(2024, 1, 1)
    for i in range(30):  # 30 days of data
        d = base_date + pd.Timedelta(days=i)
        # VTI and BND with some correlation
        vti_price = Decimal("250") + Decimal(str(i * 0.5))
        bnd_price = Decimal("75") + Decimal(str(i * 0.2))
        prices.append({"security_id": "VTI", "date": d, "close_price": vti_price, "adj_close": vti_price})
        prices.append({"security_id": "BND", "date": d, "close_price": bnd_price, "adj_close": bnd_price})

    prices_df = pd.DataFrame(prices, columns=["security_id", "date", "close_price", "adj_close"])

    engine = CorrelationEngine()
    correlation_matrix = engine.compute_correlation_matrix(
        daily_positions, prices_df, date(2024, 1, 31), LookbackWindow.ONE_MONTH
    )

    assert not correlation_matrix.empty
    assert "VTI" in correlation_matrix.index
    assert "BND" in correlation_matrix.index
    # Diagonal should be 1
    assert correlation_matrix.loc["VTI", "VTI"] == 1.0


def test_compute_correlation_matrix_empty_holdings():
    """Test correlation matrix with empty holdings."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    prices_df = pd.DataFrame(columns=["security_id", "date", "close_price", "adj_close"])

    engine = CorrelationEngine()
    correlation_matrix = engine.compute_correlation_matrix(
        daily_positions, prices_df, date(2024, 1, 31), LookbackWindow.ONE_MONTH
    )

    assert correlation_matrix.empty


def test_compute_average_correlation():
    """Test average correlation calculation."""
    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.5, 0.3],
            "BND": [0.5, 1.0, 0.2],
            "AAPL": [0.3, 0.2, 1.0],
        },
        index=["VTI", "BND", "AAPL"],
    )

    engine = CorrelationEngine()
    avg_corr = engine.compute_average_correlation(correlation_matrix)

    assert avg_corr is not None
    # Average of 0.5, 0.3, 0.2 = 0.333
    assert abs(avg_corr - Decimal("0.333")) < Decimal("0.01")


def test_compute_average_correlation_empty():
    """Test average correlation with empty matrix."""
    correlation_matrix = pd.DataFrame()

    engine = CorrelationEngine()
    avg_corr = engine.compute_average_correlation(correlation_matrix)

    assert avg_corr is None


def test_compute_highest_correlation_pairs():
    """Test finding highest correlation pairs."""
    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.8, 0.3],
            "BND": [0.8, 1.0, 0.2],
            "AAPL": [0.3, 0.2, 1.0],
        },
        index=["VTI", "BND", "AAPL"],
    )

    engine = CorrelationEngine()
    pairs = engine.compute_highest_correlation_pairs(
        correlation_matrix, threshold=Decimal("0.5"), top_n=5
    )

    assert len(pairs) == 1
    assert pairs[0]["security_1"] == "VTI"
    assert pairs[0]["security_2"] == "BND"
    assert pairs[0]["correlation"] == Decimal("0.8")


def test_compute_highest_correlation_pairs_empty():
    """Test highest correlation pairs with empty matrix."""
    correlation_matrix = pd.DataFrame()

    engine = CorrelationEngine()
    pairs = engine.compute_highest_correlation_pairs(correlation_matrix)

    assert len(pairs) == 0


def test_compute_portfolio_correlation_with_benchmark():
    """Test portfolio-benchmark correlation."""
    portfolio_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.005")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.01")},
        ]
    )

    benchmark_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.012")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.018")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.003")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.022")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.012")},
        ]
    )

    engine = CorrelationEngine()
    correlation = engine.compute_portfolio_correlation_with_benchmark(
        portfolio_returns, benchmark_returns, date(2024, 1, 5), LookbackWindow.ONE_MONTH
    )

    assert correlation is not None
    # Should be high correlation since returns are similar
    assert correlation > Decimal("0.9")


def test_compute_portfolio_correlation_with_benchmark_insufficient_data():
    """Test portfolio-benchmark correlation with insufficient data."""
    portfolio_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}]
    )

    benchmark_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.012")}]
    )

    engine = CorrelationEngine()
    correlation = engine.compute_portfolio_correlation_with_benchmark(
        portfolio_returns, benchmark_returns, date(2024, 1, 1), LookbackWindow.ONE_MONTH
    )

    assert correlation is None


def test_lookback_window_to_days():
    """Test LookbackWindow to_days conversion."""
    assert LookbackWindow.ONE_MONTH.to_days() == 21
    assert LookbackWindow.THREE_MONTHS.to_days() == 63
    assert LookbackWindow.ONE_YEAR.to_days() == 252
    assert LookbackWindow.THREE_YEARS.to_days() == 756


def test_correlation_matrix_with_different_lookbacks():
    """Test correlation matrix with different lookback windows."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    # Create 100 days of price data
    prices = []
    base_date = date(2023, 10, 1)
    for i in range(100):
        d = base_date + pd.Timedelta(days=i)
        vti_price = Decimal("250") + Decimal(str(i * 0.5))
        prices.append({"security_id": "VTI", "date": d, "close_price": vti_price, "adj_close": vti_price})

    prices_df = pd.DataFrame(prices, columns=["security_id", "date", "close_price", "adj_close"])

    engine = CorrelationEngine()

    # Test different lookback windows
    for lookback in [LookbackWindow.ONE_MONTH, LookbackWindow.THREE_MONTHS, LookbackWindow.ONE_YEAR]:
        correlation_matrix = engine.compute_correlation_matrix(
            daily_positions, prices_df, date(2024, 1, 31), lookback
        )
        # With only one security, matrix should be 1x1
        assert not correlation_matrix.empty


def test_compute_sector_correlation():
    """Test sector-level correlation aggregation."""
    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.5, 0.3],
            "BND": [0.5, 1.0, 0.2],
            "AAPL": [0.3, 0.2, 1.0],
        },
        index=["VTI", "BND", "AAPL"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
        "BND": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Fixed Income"})})(),
        "AAPL": type("Security", (), {"sector": "Technology", "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = CorrelationEngine()
    sector_corr = engine.compute_sector_correlation(correlation_matrix, security_master, dimension="asset_class")

    # Should have Equity and Fixed Income sectors
    assert "Equity" in sector_corr.index or "Unknown" in sector_corr.index


def test_compute_sector_correlation_empty():
    """Test sector correlation with empty matrix."""
    correlation_matrix = pd.DataFrame()
    security_master = {}

    engine = CorrelationEngine()
    sector_corr = engine.compute_sector_correlation(correlation_matrix, security_master)

    assert sector_corr.empty


def test_trading_days_configuration():
    """Test that trading days per year is configurable."""
    engine = CorrelationEngine(trading_days_per_year=365)
    assert engine.trading_days_per_year == 365

    engine_default = CorrelationEngine()
    assert engine_default.trading_days_per_year == 252


def test_correlation_matrix_min_observations():
    """Test that minimum observations threshold is respected."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    # Only 5 days of data
    prices = []
    base_date = date(2024, 1, 26)
    for i in range(5):
        d = base_date + pd.Timedelta(days=i)
        vti_price = Decimal("250") + Decimal(str(i * 0.5))
        prices.append({"security_id": "VTI", "date": d, "close_price": vti_price, "adj_close": vti_price})

    prices_df = pd.DataFrame(prices, columns=["security_id", "date", "close_price", "adj_close"])

    engine = CorrelationEngine()
    correlation_matrix = engine.compute_correlation_matrix(
        daily_positions, prices_df, date(2024, 1, 31), LookbackWindow.ONE_MONTH, min_observations=20
    )

    # Should return empty due to insufficient observations
    assert correlation_matrix.empty


def test_correlation_matrix_single_security():
    """Test correlation matrix with single security."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    prices = []
    base_date = date(2024, 1, 1)
    for i in range(30):
        d = base_date + pd.Timedelta(days=i)
        vti_price = Decimal("250") + Decimal(str(i * 0.5))
        prices.append({"security_id": "VTI", "date": d, "close_price": vti_price, "adj_close": vti_price})

    prices_df = pd.DataFrame(prices, columns=["security_id", "date", "close_price", "adj_close"])

    engine = CorrelationEngine()
    correlation_matrix = engine.compute_correlation_matrix(
        daily_positions, prices_df, date(2024, 1, 31), LookbackWindow.ONE_MONTH
    )

    # Single security should have 1x1 matrix
    assert correlation_matrix.shape == (1, 1)
    assert correlation_matrix.iloc[0, 0] == 1.0
