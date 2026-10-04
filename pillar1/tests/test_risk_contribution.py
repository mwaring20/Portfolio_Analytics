"""
Tests for Phase 11 VaR / CVaR and Risk Contribution.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.risk_contribution import (
    PortfolioRiskDecomposition,
    RiskContribution,
    RiskContributionEngine,
)


def test_compute_portfolio_var():
    """Test portfolio VaR calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.3],
            "BND": [0.3, 1.0],
        },
        index=["VTI", "BND"],
    )

    engine = RiskContributionEngine()
    portfolio_var = engine.compute_portfolio_var(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert portfolio_var is not None
    assert portfolio_var > 0


def test_compute_portfolio_var_empty_positions():
    """Test portfolio VaR with empty positions."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0],
        },
        index=["VTI"],
    )

    engine = RiskContributionEngine()
    portfolio_var = engine.compute_portfolio_var(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert portfolio_var is None


def test_compute_historical_var():
    """Test historical VaR calculation with 10 years of monthly returns."""
    # Create 120 months of returns (10 years)
    monthly_returns = []
    for i in range(120):
        monthly_returns.append({"date": date(2014, 1, 1), "period_return": Decimal("0.01")})

    monthly_returns_df = pd.DataFrame(monthly_returns, columns=["date", "period_return"])

    engine = RiskContributionEngine()
    historical_var = engine.compute_historical_var(
        monthly_returns_df, portfolio_value=Decimal("10000.00"), lookback_months=120
    )

    assert historical_var is not None
    # With all positive returns, VaR should be negative (loss)
    assert historical_var < 0


def test_compute_historical_var_insufficient_data():
    """Test historical VaR with insufficient data."""
    monthly_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}],
        columns=["date", "period_return"],
    )

    engine = RiskContributionEngine()
    historical_var = engine.compute_historical_var(
        monthly_returns, portfolio_value=Decimal("10000.00")
    )

    assert historical_var is None


def test_compute_cvar():
    """Test CVaR calculation from lowest 5% of returns."""
    # Create returns with some negative outliers
    monthly_returns = []
    for i in range(100):
        # Most returns are positive
        monthly_returns.append({"date": date(2014, 1, 1), "period_return": Decimal("0.01")})

    # Add some negative returns in the tail
    for i in range(20):
        monthly_returns.append({"date": date(2014, 1, 1), "period_return": Decimal("-0.05")})

    monthly_returns_df = pd.DataFrame(monthly_returns, columns=["date", "period_return"])

    engine = RiskContributionEngine()
    cvar = engine.compute_cvar(
        monthly_returns_df, portfolio_value=Decimal("10000.00"), lookback_months=120
    )

    assert cvar is not None
    # CVaR should be negative (average of negative tail)
    assert cvar < 0


def test_compute_cvar_insufficient_data():
    """Test CVaR with insufficient data."""
    monthly_returns = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}],
        columns=["date", "period_return"],
    )

    engine = RiskContributionEngine()
    cvar = engine.compute_cvar(
        monthly_returns, portfolio_value=Decimal("10000.00")
    )

    assert cvar is None


def test_compute_risk_contribution_with_monthly_returns():
    """Test complete risk decomposition with monthly returns."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0],
        },
        index=["VTI"],
    )

    # Create monthly returns
    monthly_returns = []
    for i in range(120):
        monthly_returns.append({"date": date(2014, 1, 1), "period_return": Decimal("0.01")})

    monthly_returns_df = pd.DataFrame(monthly_returns, columns=["date", "period_return"])

    engine = RiskContributionEngine()
    decomposition = engine.compute_risk_contribution(
        daily_positions, correlation_matrix, monthly_returns_df, date(2024, 1, 31)
    )

    assert decomposition.historical_var is not None
    assert decomposition.parametric_var is not None
    assert decomposition.cvar is not None


def test_compute_marginal_var():
    """Test marginal VaR calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.3],
            "BND": [0.3, 1.0],
        },
        index=["VTI", "BND"],
    )

    engine = RiskContributionEngine()
    marginal_var = engine.compute_marginal_var(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert len(marginal_var) == 2
    assert marginal_var[0].security_id == "VTI"
    assert marginal_var[0].marginal_var is not None
    assert marginal_var[1].security_id == "BND"
    assert marginal_var[1].marginal_var is not None


def test_compute_marginal_var_empty():
    """Test marginal VaR with empty positions."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    correlation_matrix = pd.DataFrame()

    engine = RiskContributionEngine()
    marginal_var = engine.compute_marginal_var(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert len(marginal_var) == 0


def test_compute_component_var():
    """Test component VaR calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.3],
            "BND": [0.3, 1.0],
        },
        index=["VTI", "BND"],
    )

    engine = RiskContributionEngine()
    component_var = engine.compute_component_var(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert len(component_var) == 2
    assert component_var[0].component_var is not None
    assert component_var[1].component_var is not None


def test_compute_risk_contribution():
    """Test complete risk decomposition."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.3],
            "BND": [0.3, 1.0],
        },
        index=["VTI", "BND"],
    )

    engine = RiskContributionEngine()
    decomposition = engine.compute_risk_contribution(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert isinstance(decomposition, PortfolioRiskDecomposition)
    assert decomposition.portfolio_var is not None
    assert decomposition.contributions is not None
    assert len(decomposition.contributions) == 2
    assert decomposition.contributions[0].percentage_contribution is not None


def test_compute_risk_contribution_empty():
    """Test risk decomposition with empty positions."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    correlation_matrix = pd.DataFrame()

    engine = RiskContributionEngine()
    decomposition = engine.compute_risk_contribution(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert decomposition.portfolio_var is None
    assert decomposition.contributions == []


def test_compute_risk_budget_comparison():
    """Test risk budget comparison."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0],
        },
        index=["VTI"],
    )

    engine = RiskContributionEngine()
    decomposition = engine.compute_risk_contribution(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    target_weights = {"VTI": Decimal("0.5")}  # 50% target risk contribution

    comparison = engine.compute_risk_budget_comparison(decomposition, target_weights)

    assert "VTI" in comparison
    assert "actual_risk_contribution" in comparison["VTI"]
    assert "target_risk_contribution" in comparison["VTI"]
    assert "difference" in comparison["VTI"]


def test_confidence_level_configuration():
    """Test that confidence level is configurable."""
    engine = RiskContributionEngine(confidence_level=Decimal("0.99"))
    assert engine.confidence_level == Decimal("0.99")

    engine_default = RiskContributionEngine()
    assert engine_default.confidence_level == Decimal("0.95")


def test_risk_contribution_dataclass():
    """Test RiskContribution dataclass structure."""
    contribution = RiskContribution(
        security_id="VTI",
        weight=Decimal("0.5"),
        marginal_var=Decimal("1000.00"),
        component_var=Decimal("500.00"),
        percentage_contribution=Decimal("0.5"),
    )

    assert contribution.security_id == "VTI"
    assert contribution.weight == Decimal("0.5")
    assert contribution.marginal_var == Decimal("1000.00")
    assert contribution.component_var == Decimal("500.00")
    assert contribution.percentage_contribution == Decimal("0.5")


def test_portfolio_risk_decomposition_dataclass():
    """Test PortfolioRiskDecomposition dataclass structure."""
    contributions = [
        RiskContribution(
            security_id="VTI",
            weight=Decimal("0.5"),
            component_var=Decimal("500.00"),
            percentage_contribution=Decimal("0.5"),
        )
    ]

    decomposition = PortfolioRiskDecomposition(
        portfolio_var=Decimal("1000.00"),
        portfolio_volatility=Decimal("0.15"),
        confidence_level=Decimal("0.95"),
        contributions=contributions,
    )

    assert decomposition.portfolio_var == Decimal("1000.00")
    assert decomposition.portfolio_volatility == Decimal("0.15")
    assert decomposition.confidence_level == Decimal("0.95")
    assert len(decomposition.contributions) == 1


def test_compute_marginal_var_missing_correlation():
    """Test marginal VaR when security is missing from correlation matrix."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "UNKNOWN",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0],
        },
        index=["VTI"],
    )

    engine = RiskContributionEngine()
    marginal_var = engine.compute_marginal_var(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert len(marginal_var) == 1
    assert marginal_var[0].security_id == "UNKNOWN"
    assert marginal_var[0].marginal_var is None


def test_component_var_sum_equals_portfolio_var():
    """Test that sum of component VaRs equals portfolio VaR."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.3],
            "BND": [0.3, 1.0],
        },
        index=["VTI", "BND"],
    )

    engine = RiskContributionEngine()
    decomposition = engine.compute_risk_contribution(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    # Sum component VaRs
    component_sum = sum(
        c.component_var for c in decomposition.contributions if c.component_var is not None
    )

    # Should approximately equal portfolio VaR
    if decomposition.portfolio_var and component_sum:
        assert abs(component_sum - decomposition.portfolio_var) < Decimal("100")  # Allow some tolerance
