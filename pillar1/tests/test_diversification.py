"""
Tests for Phase 11 Diversification Analysis.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.diversification import (
    DiversificationEngine,
    DiversificationMetrics,
)


def test_compute_effective_number_of_securities():
    """Test ENS calculation with equal weights."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = DiversificationEngine()
    ens = engine.compute_effective_number_of_securities(daily_positions, date(2024, 1, 31))

    # Equal weights: 3 securities, each 33.3%
    # ENS = 1 / (0.333^2 + 0.333^2 + 0.333^2) = 1 / (0.111 + 0.111 + 0.111) = 1 / 0.333 = 3
    assert ens is not None
    assert abs(ens - Decimal("3")) < Decimal("0.1")


def test_compute_effective_number_of_securities_concentrated():
    """Test ENS with concentrated portfolio."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("9000.00"),  # 90%
                "price": Decimal("900.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),  # 10%
                "price": Decimal("100.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = DiversificationEngine()
    ens = engine.compute_effective_number_of_securities(daily_positions, date(2024, 1, 31))

    # Concentrated: ENS should be close to 1
    assert ens is not None
    assert ens < Decimal("2")


def test_compute_effective_number_of_securities_empty():
    """Test ENS with empty portfolio."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    engine = DiversificationEngine()
    ens = engine.compute_effective_number_of_securities(daily_positions, date(2024, 1, 31))

    assert ens is None


def test_compute_hhi():
    """Test HHI calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = DiversificationEngine()
    hhi = engine.compute_hhi(daily_positions, date(2024, 1, 31))

    # Equal weights: 0.5^2 + 0.5^2 = 0.25 + 0.25 = 0.5
    assert hhi is not None
    assert abs(hhi - Decimal("0.5")) < Decimal("0.01")


def test_compute_hhi_concentrated():
    """Test HHI with concentrated portfolio."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("9000.00"),  # 90%
                "price": Decimal("900.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),  # 10%
                "price": Decimal("100.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = DiversificationEngine()
    hhi = engine.compute_hhi(daily_positions, date(2024, 1, 31))

    # Concentrated: 0.9^2 + 0.1^2 = 0.81 + 0.01 = 0.82
    assert hhi is not None
    assert abs(hhi - Decimal("0.82")) < Decimal("0.01")


def test_compute_concentration_ratio():
    """Test concentration ratio for top N positions."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("4000.00"),  # 40%
                "price": Decimal("400.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("3000.00"),  # 30%
                "price": Decimal("300.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("2000.00"),  # 20%
                "price": Decimal("200.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "MSFT",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),  # 10%
                "price": Decimal("100.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = DiversificationEngine()
    cr = engine.compute_concentration_ratio(daily_positions, date(2024, 1, 31), top_n=2)

    # Top 2: 40% + 30% = 70%
    assert cr is not None
    assert abs(cr - Decimal("0.7")) < Decimal("0.01")


def test_compute_concentration_ratio_top_5():
    """Test concentration ratio with top 5 when fewer than 5 positions."""
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

    engine = DiversificationEngine()
    cr = engine.compute_concentration_ratio(daily_positions, date(2024, 1, 31), top_n=5)

    # Top 5 but only 2 positions: should be 100%
    assert cr is not None
    assert abs(cr - Decimal("1.0")) < Decimal("0.01")


def test_compute_diversification_ratio():
    """Test diversification ratio using correlation matrix."""
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

    # Low correlation matrix
    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.2],
            "BND": [0.2, 1.0],
        },
        index=["VTI", "BND"],
    )

    engine = DiversificationEngine()
    div_ratio = engine.compute_diversification_ratio(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    # Low correlation should give higher diversification ratio
    assert div_ratio is not None
    assert div_ratio > 1


def test_compute_diversification_ratio_empty_correlation():
    """Test diversification ratio with empty correlation matrix."""
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

    correlation_matrix = pd.DataFrame()

    engine = DiversificationEngine()
    div_ratio = engine.compute_diversification_ratio(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert div_ratio is None


def test_compute_all_metrics():
    """Test computing all diversification metrics."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("4000.00"),
                "price": Decimal("400.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("3000.00"),
                "price": Decimal("300.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("3000.00"),
                "price": Decimal("300.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    correlation_matrix = pd.DataFrame(
        {
            "VTI": [1.0, 0.3, 0.4],
            "BND": [0.3, 1.0, 0.2],
            "AAPL": [0.4, 0.2, 1.0],
        },
        index=["VTI", "BND", "AAPL"],
    )

    engine = DiversificationEngine()
    metrics = engine.compute_all_metrics(
        daily_positions, correlation_matrix, date(2024, 1, 31)
    )

    assert isinstance(metrics, DiversificationMetrics)
    assert metrics.effective_number_of_securities is not None
    assert metrics.hhi is not None
    assert metrics.concentration_ratio is not None
    assert metrics.diversification_ratio is not None


def test_compute_all_metrics_no_correlation():
    """Test computing metrics without correlation matrix."""
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

    engine = DiversificationEngine()
    metrics = engine.compute_all_metrics(daily_positions, correlation_matrix=None)

    assert isinstance(metrics, DiversificationMetrics)
    assert metrics.effective_number_of_securities is not None
    assert metrics.diversification_ratio is None  # No correlation matrix


def test_assess_diversification_quality():
    """Test qualitative diversification assessment."""
    metrics = DiversificationMetrics(
        effective_number_of_securities=Decimal("25"),
        hhi=Decimal("0.04"),
        concentration_ratio=Decimal("0.35"),
    )

    engine = DiversificationEngine()
    assessments = engine.assess_diversification_quality(metrics)

    assert assessments["ens_quality"] == "Well Diversified"
    assert assessments["hhi_quality"] == "Low Concentration"
    assert assessments["concentration_quality"] == "Low Concentration"


def test_assess_diversification_quality_concentrated():
    """Test assessment for concentrated portfolio."""
    metrics = DiversificationMetrics(
        effective_number_of_securities=Decimal("3"),
        hhi=Decimal("0.5"),
        concentration_ratio=Decimal("0.85"),
    )

    engine = DiversificationEngine()
    assessments = engine.assess_diversification_quality(metrics)

    assert assessments["ens_quality"] == "Highly Concentrated"
    assert assessments["hhi_quality"] == "High Concentration"
    assert assessments["concentration_quality"] == "Very High Concentration"


def test_diversification_metrics_dataclass():
    """Test DiversificationMetrics dataclass structure."""
    metrics = DiversificationMetrics(
        effective_number_of_securities=Decimal("10"),
        hhi=Decimal("0.1"),
        diversification_ratio=Decimal("2.5"),
        concentration_ratio=Decimal("0.5"),
    )

    assert metrics.effective_number_of_securities == Decimal("10")
    assert metrics.hhi == Decimal("0.1")
    assert metrics.diversification_ratio == Decimal("2.5")
    assert metrics.concentration_ratio == Decimal("0.5")


def test_compute_all_metrics_auto_date():
    """Test compute_all_metrics with auto-detected date."""
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

    engine = DiversificationEngine()
    metrics = engine.compute_all_metrics(daily_positions)  # No date specified

    assert metrics.effective_number_of_securities is not None
