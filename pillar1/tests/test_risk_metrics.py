"""
Tests for Phase 9 Risk Metrics Core.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.risk_metrics import RiskMetrics, RiskMetricsEngine


def test_compute_volatility():
    """Test volatility calculation."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("-0.005")},
            {"date": date(2024, 1, 3), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 4), "period_return": Decimal("-0.01")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.015")},
        ]
    )

    engine = RiskMetricsEngine()
    volatility = engine.compute_volatility(returns_df)

    assert volatility is not None
    assert volatility > 0


def test_compute_volatility_insufficient_data():
    """Test volatility with insufficient data returns None."""
    returns_df = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}]
    )

    engine = RiskMetricsEngine()
    volatility = engine.compute_volatility(returns_df)

    assert volatility is None


def test_compute_sharpe_ratio():
    """Test Sharpe ratio calculation."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 3), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.015")},
        ]
    )

    engine = RiskMetricsEngine()
    sharpe = engine.compute_sharpe_ratio(returns_df, risk_free_rate=Decimal("0.02"))

    assert sharpe is not None


def test_compute_sharpe_ratio_zero_volatility():
    """Test Sharpe ratio with zero volatility returns None."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 3), "period_return": Decimal("0.01")},
        ]
    )

    engine = RiskMetricsEngine()
    sharpe = engine.compute_sharpe_ratio(returns_df, risk_free_rate=Decimal("0.02"))

    # Zero volatility should return None
    assert sharpe is None


def test_compute_sortino_ratio():
    """Test Sortino ratio calculation."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("-0.02")},  # Downside
            {"date": date(2024, 1, 3), "period_return": Decimal("0.03")},
            {"date": date(2024, 1, 4), "period_return": Decimal("-0.01")},  # Downside
            {"date": date(2024, 1, 5), "period_return": Decimal("0.02")},
        ]
    )

    engine = RiskMetricsEngine()
    sortino = engine.compute_sortino_ratio(returns_df, risk_free_rate=Decimal("0.02"))

    assert sortino is not None


def test_compute_sortino_ratio_no_downside():
    """Test Sortino ratio with no downside returns returns None."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.05")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.06")},
            {"date": date(2024, 1, 3), "period_return": Decimal("0.07")},
        ]
    )

    engine = RiskMetricsEngine()
    sortino = engine.compute_sortino_ratio(returns_df, risk_free_rate=Decimal("0.02"))

    # No downside returns - should return None (infinite Sortino)
    assert sortino is None


def test_compute_max_drawdown():
    """Test max drawdown calculation."""
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
                "date": date(2024, 1, 2),
                "quantity": Decimal("10"),
                "market_value": Decimal("11000.00"),  # Peak
                "price": Decimal("1100.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 3),
                "quantity": Decimal("10"),
                "market_value": Decimal("9000.00"),  # Drawdown
                "price": Decimal("900.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 4),
                "quantity": Decimal("10"),
                "market_value": Decimal("9500.00"),  # Recovery
                "price": Decimal("950.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = RiskMetricsEngine()
    max_dd, duration = engine.compute_max_drawdown(daily_positions)

    assert max_dd is not None
    assert max_dd < 0  # Drawdown is negative
    assert duration is not None
    assert duration > 0


def test_compute_max_drawdown_insufficient_data():
    """Test max drawdown with insufficient data returns None."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("10000.00"),
                "price": Decimal("1000.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = RiskMetricsEngine()
    max_dd, duration = engine.compute_max_drawdown(daily_positions)

    assert max_dd is None
    assert duration is None


def test_compute_beta():
    """Test beta calculation via regression."""
    portfolio_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.01")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.025")},
        ]
    )

    benchmark_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.025")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.005")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.03")},
        ]
    )

    engine = RiskMetricsEngine()
    beta, alpha = engine.compute_beta(portfolio_returns, benchmark_returns)

    assert beta is not None
    assert alpha is not None


def test_compute_beta_insufficient_data():
    """Test beta with insufficient data returns None."""
    portfolio_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}]
    )

    benchmark_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.015")}]
    )

    engine = RiskMetricsEngine()
    beta, alpha = engine.compute_beta(portfolio_returns, benchmark_returns)

    assert beta is None
    assert alpha is None


def test_compute_tracking_error():
    """Test tracking error calculation."""
    portfolio_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.01")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.025")},
        ]
    )

    benchmark_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.025")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.005")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.03")},
        ]
    )

    engine = RiskMetricsEngine()
    tracking_error = engine.compute_tracking_error(portfolio_returns, benchmark_returns)

    assert tracking_error is not None
    assert tracking_error >= 0


def test_compute_tracking_error_insufficient_data():
    """Test tracking error with insufficient data returns None."""
    portfolio_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}]
    )

    benchmark_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.015")}]
    )

    engine = RiskMetricsEngine()
    tracking_error = engine.compute_tracking_error(portfolio_returns, benchmark_returns)

    assert tracking_error is None


def test_compute_information_ratio():
    """Test information ratio calculation."""
    portfolio_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.01")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.025")},
        ]
    )

    benchmark_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.025")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.005")},
            {"date": date(2024, 1, 4), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 5), "period_return": Decimal("0.03")},
        ]
    )

    engine = RiskMetricsEngine()
    ir = engine.compute_information_ratio(portfolio_returns, benchmark_returns)

    assert ir is not None


def test_compute_all_metrics():
    """Test computing all metrics at once."""
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
                "date": date(2024, 1, 2),
                "quantity": Decimal("10"),
                "market_value": Decimal("11000.00"),
                "price": Decimal("1100.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 3),
                "quantity": Decimal("10"),
                "market_value": Decimal("9000.00"),
                "price": Decimal("900.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    portfolio_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.02")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.01")},
        ]
    )

    benchmark_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.015")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.025")},
            {"date": date(2024, 1, 3), "period_return": Decimal("-0.005")},
        ]
    )

    engine = RiskMetricsEngine()
    metrics = engine.compute_all_metrics(
        daily_positions,
        portfolio_returns,
        benchmark_returns,
        risk_free_rate=Decimal("0.02"),
    )

    assert isinstance(metrics, RiskMetrics)
    assert metrics.volatility is not None
    assert metrics.max_drawdown is not None
    assert metrics.beta is not None
    assert metrics.alpha is not None
    assert metrics.tracking_error is not None


def test_compute_all_metrics_no_benchmark():
    """Test computing metrics without benchmark."""
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
                "date": date(2024, 1, 2),
                "quantity": Decimal("10"),
                "market_value": Decimal("11000.00"),
                "price": Decimal("1100.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    portfolio_returns = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.01")},
            {"date": date(2024, 1, 2), "period_return": Decimal("0.02")},
        ]
    )

    engine = RiskMetricsEngine()
    metrics = engine.compute_all_metrics(
        daily_positions,
        portfolio_returns,
        benchmark_returns=None,
        risk_free_rate=Decimal("0.02"),
    )

    assert isinstance(metrics, RiskMetrics)
    assert metrics.volatility is not None
    assert metrics.beta is None  # No benchmark
    assert metrics.alpha is None  # No benchmark
    assert metrics.tracking_error is None  # No benchmark


def test_trading_days_configuration():
    """Test that trading days per year is configurable."""
    engine = RiskMetricsEngine(trading_days_per_year=365)
    assert engine.trading_days_per_year == 365

    engine_default = RiskMetricsEngine()
    assert engine_default.trading_days_per_year == 252


def test_risk_metrics_dataclass():
    """Test RiskMetrics dataclass structure."""
    metrics = RiskMetrics(
        volatility=Decimal("0.15"),
        sharpe_ratio=Decimal("1.5"),
        sortino_ratio=Decimal("2.0"),
        max_drawdown=Decimal("-0.20"),
        max_drawdown_duration=30,
        beta=Decimal("1.1"),
        alpha=Decimal("0.01"),
        tracking_error=Decimal("0.05"),
        information_ratio=Decimal("0.8"),
    )

    assert metrics.volatility == Decimal("0.15")
    assert metrics.sharpe_ratio == Decimal("1.5")
    assert metrics.max_drawdown_duration == 30
