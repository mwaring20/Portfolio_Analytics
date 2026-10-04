"""
Tests for Phase 14 Monte Carlo Simulation.
"""

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.monte_carlo import (
    MonteCarloEngine,
    RetirementCashFlow,
    SimulationPath,
    SimulationResult,
)


def test_run_simulation_basic():
    """Test basic Monte Carlo simulation."""
    expected_returns = {"Equity": Decimal("0.08"), "Fixed Income": Decimal("0.04")}
    volatilities = {"Equity": Decimal("0.15"), "Fixed Income": Decimal("0.05")}
    weights = {"Equity": Decimal("0.6"), "Fixed Income": Decimal("0.4")}

    correlation_matrix = pd.DataFrame(
        {
            "Equity": [1.0, 0.3],
            "Fixed Income": [0.3, 1.0],
        },
        index=["Equity", "Fixed Income"],
    )

    engine = MonteCarloEngine()
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=10,
        num_paths=100,
    )

    assert result.initial_value == Decimal("100000")
    assert result.num_paths == 100
    assert result.num_periods > 0
    assert len(result.paths) == 100
    assert result.percentile_bands is not None
    assert result.final_value_distribution is not None


def test_run_simulation_empty_weights():
    """Test simulation with empty weights."""
    engine = MonteCarloEngine()
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns={},
        volatilities={},
        correlation_matrix=pd.DataFrame(),
        weights={},
        num_years=10,
        num_paths=100,
    )

    assert result.num_paths == 0
    assert result.num_periods == 0
    assert len(result.paths) == 0


def test_run_simulation_single_asset():
    """Test simulation with single asset class."""
    expected_returns = {"Equity": Decimal("0.08")}
    volatilities = {"Equity": Decimal("0.15")}
    weights = {"Equity": Decimal("1.0")}

    correlation_matrix = pd.DataFrame(
        {
            "Equity": [1.0],
        },
        index=["Equity"],
    )

    engine = MonteCarloEngine()
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=5,
        num_paths=50,
    )

    assert result.num_paths == 50
    assert len(result.paths) == 50


def test_calculate_percentile_bands():
    """Test percentile band calculation."""
    paths = [
        SimulationPath(path_id=0, values=[Decimal("100"), Decimal("110"), Decimal("120")]),
        SimulationPath(path_id=1, values=[Decimal("100"), Decimal("105"), Decimal("115")]),
        SimulationPath(path_id=2, values=[Decimal("100"), Decimal("115"), Decimal("125")]),
    ]

    engine = MonteCarloEngine()
    bands = engine._calculate_percentile_bands(paths, [50])

    assert "p50" in bands
    assert len(bands["p50"]) == 3


def test_run_retirement_simulation():
    """Test retirement simulation with cash flows."""
    expected_returns = {"Equity": Decimal("0.08"), "Fixed Income": Decimal("0.04")}
    volatilities = {"Equity": Decimal("0.15"), "Fixed Income": Decimal("0.05")}
    weights = {"Equity": Decimal("0.6"), "Fixed Income": Decimal("0.4")}

    correlation_matrix = pd.DataFrame(
        {
            "Equity": [1.0, 0.3],
            "Fixed Income": [0.3, 1.0],
        },
        index=["Equity", "Fixed Income"],
    )

    # Create cash flows (monthly contributions)
    cash_flows = []
    for i in range(12 * 30):  # 30 years of monthly contributions
        cf_date = date.today() + timedelta(days=30 * i)
        cash_flows.append(
            RetirementCashFlow(date=cf_date, amount=Decimal("1000"), flow_type="contribution")
        )

    engine = MonteCarloEngine()
    result = engine.run_retirement_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        cash_flows=cash_flows,
        target_value=Decimal("1000000"),
        num_years=30,
        num_paths=100,
    )

    assert result.num_paths == 100
    assert result.final_value_distribution is not None
    assert "probability_of_success" in result.final_value_distribution


def test_run_retirement_simulation_with_withdrawals():
    """Test retirement simulation with withdrawals."""
    expected_returns = {"Equity": Decimal("0.06"), "Fixed Income": Decimal("0.03")}
    volatilities = {"Equity": Decimal("0.12"), "Fixed Income": Decimal("0.04")}
    weights = {"Equity": Decimal("0.4"), "Fixed Income": Decimal("0.6")}

    correlation_matrix = pd.DataFrame(
        {
            "Equity": [1.0, 0.2],
            "Fixed Income": [0.2, 1.0],
        },
        index=["Equity", "Fixed Income"],
    )

    # Retirement withdrawals
    cash_flows = []
    for i in range(12 * 20):  # 20 years of monthly withdrawals
        cf_date = date.today() + timedelta(days=30 * i)
        cash_flows.append(
            RetirementCashFlow(date=cf_date, amount=Decimal("5000"), flow_type="withdrawal")
        )

    engine = MonteCarloEngine()
    result = engine.run_retirement_simulation(
        initial_value=Decimal("1000000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        cash_flows=cash_flows,
        num_years=20,
        num_paths=50,
    )

    assert result.num_paths == 50
    # With withdrawals, final values should generally be lower
    assert result.final_value_distribution["p50"] < Decimal("1000000")


def test_generate_simulation_report():
    """Test simulation report generation."""
    result = SimulationResult(
        initial_value=Decimal("100000"),
        num_paths=100,
        num_periods=120,
        period_length_years=Decimal("0.0833"),
        final_value_distribution={
            "p5": Decimal("80000"),
            "p25": Decimal("95000"),
            "p50": Decimal("110000"),
            "p75": Decimal("130000"),
            "p95": Decimal("160000"),
        },
    )

    engine = MonteCarloEngine()
    report = engine.generate_simulation_report(result)

    assert "Monte Carlo Simulation Report" in report
    assert "$100,000.00" in report
    assert "Number of Paths: 100" in report
    assert "p50: $110,000.00" in report


def test_simulation_path_dataclass():
    """Test SimulationPath dataclass structure."""
    path = SimulationPath(
        path_id=1,
        values=[Decimal("100"), Decimal("110"), Decimal("120")],
    )

    assert path.path_id == 1
    assert len(path.values) == 3
    assert path.values[0] == Decimal("100")


def test_simulation_result_dataclass():
    """Test SimulationResult dataclass structure."""
    result = SimulationResult(
        initial_value=Decimal("100000"),
        num_paths=100,
        num_periods=120,
        period_length_years=Decimal("0.0833"),
        paths=[],
        percentile_bands={},
        final_value_distribution={},
    )

    assert result.initial_value == Decimal("100000")
    assert result.num_paths == 100
    assert result.num_periods == 120


def test_retirement_cash_flow_dataclass():
    """Test RetirementCashFlow dataclass structure."""
    cf = RetirementCashFlow(
        date=date(2024, 1, 1),
        amount=Decimal("1000"),
        flow_type="contribution",
    )

    assert cf.date == date(2024, 1, 1)
    assert cf.amount == Decimal("1000")
    assert cf.flow_type == "contribution"


def test_run_simulation_different_num_paths():
    """Test simulation with different number of paths."""
    expected_returns = {"Equity": Decimal("0.08")}
    volatilities = {"Equity": Decimal("0.15")}
    weights = {"Equity": Decimal("1.0")}
    correlation_matrix = pd.DataFrame({"Equity": [1.0]}, index=["Equity"])

    engine = MonteCarloEngine()

    # Test with 10 paths
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=5,
        num_paths=10,
    )
    assert result.num_paths == 10

    # Test with 1000 paths
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=5,
        num_paths=1000,
    )
    assert result.num_paths == 1000


def test_run_simulation_different_horizons():
    """Test simulation with different time horizons."""
    expected_returns = {"Equity": Decimal("0.08")}
    volatilities = {"Equity": Decimal("0.15")}
    weights = {"Equity": Decimal("1.0")}
    correlation_matrix = pd.DataFrame({"Equity": [1.0]}, index=["Equity"])

    engine = MonteCarloEngine()

    # 1 year
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=1,
        num_paths=50,
    )
    assert result.num_periods == 12  # Monthly

    # 10 years
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=10,
        num_paths=50,
    )
    assert result.num_periods == 120  # Monthly


def test_cholesky_decomposition():
    """Test that Cholesky decomposition is used for correlation."""
    # This is implicitly tested by the simulation running without errors
    # with a non-identity correlation matrix
    expected_returns = {"Equity": Decimal("0.08"), "Fixed Income": Decimal("0.04")}
    volatilities = {"Equity": Decimal("0.15"), "Fixed Income": Decimal("0.05")}
    weights = {"Equity": Decimal("0.6"), "Fixed Income": Decimal("0.4")}

    # Non-identity correlation matrix
    correlation_matrix = pd.DataFrame(
        {
            "Equity": [1.0, 0.5],
            "Fixed Income": [0.5, 1.0],
        },
        index=["Equity", "Fixed Income"],
    )

    engine = MonteCarloEngine()
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=5,
        num_paths=50,
    )

    # Should succeed with correlated returns
    assert result.num_paths == 50


def test_final_value_distribution_percentiles():
    """Test that final value distribution includes all percentiles."""
    expected_returns = {"Equity": Decimal("0.08")}
    volatilities = {"Equity": Decimal("0.15")}
    weights = {"Equity": Decimal("1.0")}
    correlation_matrix = pd.DataFrame({"Equity": [1.0]}, index=["Equity"])

    engine = MonteCarloEngine()
    result = engine.run_simulation(
        initial_value=Decimal("100000"),
        expected_returns=expected_returns,
        volatilities=volatilities,
        correlation_matrix=correlation_matrix,
        weights=weights,
        num_years=5,
        num_paths=100,
    )

    assert "p5" in result.final_value_distribution
    assert "p25" in result.final_value_distribution
    assert "p50" in result.final_value_distribution
    assert "p75" in result.final_value_distribution
    assert "p95" in result.final_value_distribution
