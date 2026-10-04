"""
Phase 14 — Monte Carlo Simulation.

Per architecture doc:
  - Probability cone of outcomes and retirement readiness as two sub-modules
  - Use Cholesky decomposition of Phase 10 correlation matrix for correlated random draws
  - 1,000–10,000 paths is standard
  - Summary output: percentile bands, probability of success
  - Retirement readiness needs cash-flow modeling (contributions, withdrawals, timing)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class SimulationPath:
    """A single Monte Carlo simulation path."""
    path_id: int
    values: List[Decimal]


@dataclass
class SimulationResult:
    """Result of Monte Carlo simulation."""
    initial_value: Decimal
    num_paths: int
    num_periods: int
    period_length_years: Decimal
    paths: List[SimulationPath] = None
    percentile_bands: Dict[str, List[Decimal]] = None  # e.g., {"p5": [...], "p50": [...], "p95": [...]}
    final_value_distribution: Dict[str, Decimal] = None  # e.g., {"p5": X, "p50": Y, "p95": Z}


@dataclass
class RetirementCashFlow:
    """Cash flow for retirement planning."""
    date: date
    amount: Decimal
    flow_type: str  # "contribution" or "withdrawal"


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for portfolio projections.

    Uses Cholesky decomposition for correlated asset class returns.
    """

    def __init__(self):
        """Initialize the Monte Carlo engine."""
        pass

    def run_simulation(
        self,
        initial_value: Decimal,
        expected_returns: Dict[str, Decimal],
        volatilities: Dict[str, Decimal],
        correlation_matrix: pd.DataFrame,
        weights: Dict[str, Decimal],
        num_years: int,
        num_paths: int = 1000,
        time_step: Decimal = Decimal("0.0833"),  # Monthly (1/12 year)
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            initial_value: starting portfolio value
            expected_returns: expected annual return by asset class
            volatilities: annual volatility by asset class
            correlation_matrix: correlation matrix from Phase 10
            weights: portfolio weights by asset class
            num_years: number of years to simulate
            num_paths: number of simulation paths (default 1000)
            time_step: time step in years (default monthly)

        Returns:
            SimulationResult with all paths and percentile bands
        """
        asset_classes = list(weights.keys())
        num_assets = len(asset_classes)

        if num_assets == 0:
            return SimulationResult(
                initial_value=initial_value,
                num_paths=0,
                num_periods=0,
                period_length_years=time_step,
                paths=[],
                percentile_bands={},
                final_value_distribution={},
            )

        # Convert to numpy arrays
        mu = np.array([float(expected_returns.get(ac, 0)) for ac in asset_classes])
        sigma = np.array([float(volatilities.get(ac, 0.15)) for ac in asset_classes])
        w = np.array([float(weights.get(ac, 0)) for ac in asset_classes])

        # Cholesky decomposition of correlation matrix
        corr_matrix = correlation_matrix.values if not correlation_matrix.empty else np.eye(num_assets)
        L = np.linalg.cholesky(corr_matrix)

        # Number of periods
        num_periods = int(float(num_years) / float(time_step))

        # Generate paths
        paths: List[SimulationPath] = []
        for path_id in range(num_paths):
            values = [initial_value]

            current_value = float(initial_value)

            for period in range(num_periods):
                # Generate correlated random normal variables
                z = np.random.randn(num_assets)
                correlated_z = L @ z

                # Scale by volatilities
                scaled_returns = mu * float(time_step) + sigma * correlated_z * np.sqrt(float(time_step))

                # Portfolio return
                portfolio_return = np.dot(w, scaled_returns)

                # Update value
                current_value = current_value * (1 + portfolio_return)
                values.append(Decimal(str(current_value)))

            paths.append(SimulationPath(path_id=path_id, values=values))

        # Calculate percentile bands
        percentile_bands = self._calculate_percentile_bands(paths, [5, 25, 50, 75, 95])

        # Calculate final value distribution
        final_values = [path.values[-1] for path in paths]
        final_value_distribution = {
            "p5": np.percentile([float(v) for v in final_values], 5),
            "p25": np.percentile([float(v) for v in final_values], 25) if len(final_values) > 0 else 0,
            "p50": np.percentile([float(v) for v in final_values], 50) if len(final_values) > 0 else 0,
            "p75": np.percentile([float(v) for v in final_values], 75) if len(final_values) > 0 else 0,
            "p95": np.percentile([float(v) for v in final_values], 95) if len(final_values) > 0 else 0,
        }

        return SimulationResult(
            initial_value=initial_value,
            num_paths=num_paths,
            num_periods=num_periods,
            period_length_years=time_step,
            paths=paths,
            percentile_bands=percentile_bands,
            final_value_distribution={
                k: Decimal(str(v)) for k, v in final_value_distribution.items()
            },
        )

    def _calculate_percentile_bands(
        self,
        paths: List[SimulationPath],
        percentiles: List[int],
    ) -> Dict[str, List[Decimal]]:
        """Calculate percentile bands across all paths."""
        if not paths:
            return {}

        num_periods = len(paths[0].values)
        bands = {f"p{p}": [Decimal("0")] * num_periods for p in percentiles}

        for period in range(num_periods):
            period_values = [float(path.values[period]) for path in paths]
            for p in percentiles:
                percentile_value = np.percentile(period_values, p)
                bands[f"p{p}"][period] = Decimal(str(percentile_value))

        return bands

    def run_retirement_simulation(
        self,
        initial_value: Decimal,
        expected_returns: Dict[str, Decimal],
        volatilities: Dict[str, Decimal],
        correlation_matrix: pd.DataFrame,
        weights: Dict[str, Decimal],
        cash_flows: List[RetirementCashFlow],
        target_value: Optional[Decimal] = None,
        num_years: int = 30,
        num_paths: int = 1000,
    ) -> SimulationResult:
        """
        Run retirement readiness simulation with cash flows.

        Args:
            initial_value: starting portfolio value
            expected_returns: expected annual return by asset class
            volatilities: annual volatility by asset class
            correlation_matrix: correlation matrix from Phase 10
            weights: portfolio weights by asset class
            cash_flows: list of contributions and withdrawals
            target_value: optional target value at end of horizon
            num_years: number of years to simulate
            num_paths: number of simulation paths

        Returns:
            SimulationResult with retirement-specific metrics
        """
        asset_classes = list(weights.keys())
        num_assets = len(asset_classes)

        if num_assets == 0:
            return SimulationResult(
                initial_value=initial_value,
                num_paths=0,
                num_periods=0,
                period_length_years=Decimal("0.0833"),
                paths=[],
                percentile_bands={},
                final_value_distribution={},
            )

        # Convert to numpy arrays
        mu = np.array([float(expected_returns.get(ac, 0)) for ac in asset_classes])
        sigma = np.array([float(volatilities.get(ac, 0.15)) for ac in asset_classes])
        w = np.array([float(weights.get(ac, 0)) for ac in asset_classes])

        # Cholesky decomposition
        corr_matrix = correlation_matrix.values if not correlation_matrix.empty else np.eye(num_assets)
        L = np.linalg.cholesky(corr_matrix)

        # Monthly simulation
        num_periods = num_years * 12
        time_step = Decimal("0.0833")

        # Create cash flow schedule
        cash_flow_schedule = {}
        for cf in cash_flows:
            # Find the period index for this cash flow
            months_from_start = (cf.date - date.today()).days // 30
            if 0 <= months_from_start < num_periods:
                if months_from_start not in cash_flow_schedule:
                    cash_flow_schedule[months_from_start] = Decimal("0")
                if cf.flow_type == "contribution":
                    cash_flow_schedule[months_from_start] += cf.amount
                else:  # withdrawal
                    cash_flow_schedule[months_from_start] -= cf.amount

        # Generate paths
        paths: List[SimulationPath] = []
        success_count = 0

        for path_id in range(num_paths):
            values = [initial_value]
            current_value = float(initial_value)

            for period in range(num_periods):
                # Generate correlated returns
                z = np.random.randn(num_assets)
                correlated_z = L @ z
                scaled_returns = mu * float(time_step) + sigma * correlated_z * np.sqrt(float(time_step))
                portfolio_return = np.dot(w, scaled_returns)

                # Update value with return
                current_value = current_value * (1 + portfolio_return)

                # Apply cash flow for this period
                if period in cash_flow_schedule:
                    current_value += float(cash_flow_schedule[period])

                values.append(Decimal(str(current_value)))

            # Check if target value met
            if target_value is not None and values[-1] >= target_value:
                success_count += 1

            paths.append(SimulationPath(path_id=path_id, values=values))

        # Calculate percentile bands
        percentile_bands = self._calculate_percentile_bands(paths, [5, 25, 50, 75, 95])

        # Calculate final value distribution
        final_values = [path.values[-1] for path in paths]
        final_value_distribution = {
            "p5": np.percentile([float(v) for v in final_values], 5) if len(final_values) > 0 else 0,
            "p25": np.percentile([float(v) for v in final_values], 25) if len(final_values) > 0 else 0,
            "p50": np.percentile([float(v) for v in final_values], 50) if len(final_values) > 0 else 0,
            "p75": np.percentile([float(v) for v in final_values], 75) if len(final_values) > 0 else 0,
            "p95": np.percentile([float(v) for v in final_values], 95) if len(final_values) > 0 else 0,
        }

        # Add probability of success
        probability_of_success = Decimal(str(success_count / num_paths)) if num_paths > 0 else Decimal("0")
        final_value_distribution["probability_of_success"] = probability_of_success

        return SimulationResult(
            initial_value=initial_value,
            num_paths=num_paths,
            num_periods=num_periods,
            period_length_years=time_step,
            paths=paths,
            percentile_bands=percentile_bands,
            final_value_distribution={
                k: Decimal(str(v)) for k, v in final_value_distribution.items()
            },
        )

    def generate_simulation_report(
        self,
        result: SimulationResult,
    ) -> str:
        """
        Generate a human-readable simulation report.

        Args:
            result: SimulationResult from run_simulation

        Returns:
            Formatted report string
        """
        lines = [
            "Monte Carlo Simulation Report",
            f"Initial Value: ${result.initial_value:,.2f}",
            f"Number of Paths: {result.num_paths}",
            f"Number of Periods: {result.num_periods}",
            "",
            "Final Value Distribution:",
        ]

        if result.final_value_distribution:
            for percentile in ["p5", "p25", "p50", "p75", "p95"]:
                if percentile in result.final_value_distribution:
                    if percentile == "probability_of_success":
                        lines.append(f"  Probability of Success: {result.final_value_distribution[percentile]:.2%}")
                    else:
                        lines.append(f"  {percentile}: ${result.final_value_distribution[percentile]:,.2f}")

        return "\n".join(lines)
