"""
Tests for Phase 13 Fama-French Factor Exposure.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.fama_french import (
    FactorExposure,
    FactorRegressionResult,
    FamaFrenchEngine,
)


def test_run_regression_basic():
    """Test basic Fama-French regression."""
    # Create sample returns data
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.02")},
            {"date": date(2024, 2, 1), "period_return": Decimal("0.015")},
            {"date": date(2024, 3, 1), "period_return": Decimal("-0.01")},
            {"date": date(2024, 4, 1), "period_return": Decimal("0.03")},
            {"date": date(2024, 5, 1), "period_return": Decimal("0.025")},
        ]
    )

    # Create factor returns (Mkt-RF, SMB, HML)
    factor_returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002},
            {"date": date(2024, 2, 1), "Mkt-RF": 0.02, "SMB": 0.01, "HML": 0.001},
            {"date": date(2024, 3, 1), "Mkt-RF": -0.015, "SMB": -0.01, "HML": -0.005},
            {"date": date(2024, 4, 1), "Mkt-RF": 0.025, "SMB": 0.015, "HML": 0.003},
            {"date": date(2024, 5, 1), "Mkt-RF": 0.01, "SMB": 0.008, "HML": 0.002},
        ]
    )

    engine = FamaFrenchEngine()
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="ACC001",
        entity_type="account",
        lookback_months=36,
    )

    assert result.entity_id == "ACC001"
    assert result.entity_type == "account"
    assert result.r_squared is not None
    assert result.intercept is not None
    assert len(result.factor_exposures) == 3


def test_run_regression_insufficient_data():
    """Test regression with insufficient data."""
    returns_df = pd.DataFrame(
        [{"date": date(2024, 1, 1), "period_return": Decimal("0.01")}]
    )

    factor_returns_df = pd.DataFrame(
        [{"date": date(2024, 1, 1), "Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002}]
    )

    engine = FamaFrenchEngine()
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="ACC001",
        entity_type="account",
    )

    assert result.r_squared is None
    assert result.intercept is None
    assert len(result.factor_exposures) == 0


def test_run_regression_with_risk_free_rate():
    """Test regression with risk-free rate for excess returns."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.02")},
            {"date": date(2024, 2, 1), "period_return": Decimal("0.015")},
            {"date": date(2024, 3, 1), "period_return": Decimal("-0.01")},
        ]
    )

    factor_returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002},
            {"date": date(2024, 2, 1), "Mkt-RF": 0.02, "SMB": 0.01, "HML": 0.001},
            {"date": date(2024, 3, 1), "Mkt-RF": -0.015, "SMB": -0.01, "HML": -0.005},
        ]
    )

    risk_free_rate = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "rf": 0.001},
            {"date": date(2024, 2, 1), "rf": 0.001},
            {"date": date(2024, 3, 1), "rf": 0.001},
        ]
    )

    engine = FamaFrenchEngine()
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="ACC001",
        entity_type="account",
        risk_free_rate=risk_free_rate,
    )

    assert result.r_squared is not None
    assert result.intercept is not None


def test_run_regression_no_factor_data():
    """Test regression when factor returns don't match dates."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.02")},
            {"date": date(2024, 2, 1), "period_return": Decimal("0.015")},
        ]
    )

    factor_returns_df = pd.DataFrame(
        [{"date": date(2023, 1, 1), "Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002}]
    )

    engine = FamaFrenchEngine()
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="ACC001",
        entity_type="account",
    )

    # No matching dates - should return empty result
    assert result.r_squared is None
    assert len(result.factor_exposures) == 0


def test_generate_factor_report():
    """Test factor exposure report generation."""
    factor_exposures = [
        FactorExposure(
            factor_name="Mkt-RF",
            coefficient=Decimal("1.2"),
            t_statistic=Decimal("3.5"),
            p_value=Decimal("0.001"),
        ),
        FactorExposure(
            factor_name="SMB",
            coefficient=Decimal("0.3"),
            t_statistic=Decimal("1.2"),
            p_value=Decimal("0.25"),
        ),
        FactorExposure(
            factor_name="HML",
            coefficient=Decimal("-0.1"),
            t_statistic=Decimal("-0.5"),
            p_value=Decimal("0.6"),
        ),
    ]

    result = FactorRegressionResult(
        entity_id="ACC001",
        entity_type="account",
        lookback_months=36,
        r_squared=Decimal("0.85"),
        intercept=Decimal("0.005"),
        factor_exposures=factor_exposures,
    )

    engine = FamaFrenchEngine()
    report = engine.generate_factor_report(result)

    assert "Fama-French Factor Exposure Report" in report
    assert "ACC001" in report
    assert "R²: 0.8500" in report
    assert "Mkt-RF: 1.2000" in report
    assert "Intercept (Alpha): 0.0050" in report


def test_generate_factor_report_no_exposures():
    """Test report generation with no factor exposures."""
    result = FactorRegressionResult(
        entity_id="ACC001",
        entity_type="account",
        lookback_months=36,
        r_squared=None,
        intercept=None,
        factor_exposures=[],
    )

    engine = FamaFrenchEngine()
    report = engine.generate_factor_report(result)

    assert "No factor exposures calculated" in report


def test_factor_exposure_dataclass():
    """Test FactorExposure dataclass structure."""
    exposure = FactorExposure(
        factor_name="Mkt-RF",
        coefficient=Decimal("1.2"),
        t_statistic=Decimal("3.5"),
        p_value=Decimal("0.001"),
    )

    assert exposure.factor_name == "Mkt-RF"
    assert exposure.coefficient == Decimal("1.2")
    assert exposure.t_statistic == Decimal("3.5")
    assert exposure.p_value == Decimal("0.001")


def test_factor_regression_result_dataclass():
    """Test FactorRegressionResult dataclass structure."""
    factor_exposures = [
        FactorExposure(factor_name="Mkt-RF", coefficient=Decimal("1.2"))
    ]

    result = FactorRegressionResult(
        entity_id="ACC001",
        entity_type="account",
        lookback_months=36,
        r_squared=Decimal("0.85"),
        intercept=Decimal("0.005"),
        factor_exposures=factor_exposures,
    )

    assert result.entity_id == "ACC001"
    assert result.entity_type == "account"
    assert result.lookback_months == 36
    assert result.r_squared == Decimal("0.85")
    assert len(result.factor_exposures) == 1


def test_run_regression_lookback_filtering():
    """Test that lookback parameter filters data correctly."""
    # Create 60 months of returns
    returns_df = pd.DataFrame(
        [
            {"date": date(2020, 1, 1), "period_return": Decimal("0.01")}
            for _ in range(60)
        ]
    )

    factor_returns_df = pd.DataFrame(
        [
            {"date": date(2020, 1, 1), "Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002}
            for _ in range(60)
        ]
    )

    engine = FamaFrenchEngine()
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="ACC001",
        entity_type="account",
        lookback_months=24,  # Only use last 24 months
    )

    assert result.lookback_months == 24
    assert result.r_squared is not None


def test_run_regression_different_entity_types():
    """Test regression for different entity types."""
    returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "period_return": Decimal("0.02")},
            {"date": date(2024, 2, 1), "period_return": Decimal("0.015")},
        ]
    )

    factor_returns_df = pd.DataFrame(
        [
            {"date": date(2024, 1, 1), "Mkt-RF": 0.01, "SMB": 0.005, "HML": -0.002},
            {"date": date(2024, 2, 1), "Mkt-RF": 0.02, "SMB": 0.01, "HML": 0.001},
        ]
    )

    engine = FamaFrenchEngine()

    # Test security entity type
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="VTI",
        entity_type="security",
    )
    assert result.entity_type == "security"

    # Test composite entity type
    result = engine.run_regression(
        returns_df,
        factor_returns_df,
        entity_id="PORTFOLIO_001",
        entity_type="composite",
    )
    assert result.entity_type == "composite"
