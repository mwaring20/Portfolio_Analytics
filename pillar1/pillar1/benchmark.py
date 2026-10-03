"""
Phase 5 — Benchmark Engine.

Per architecture doc:
  - Composite benchmarks (e.g. 60/40) need their own rebalancing logic
  - Align benchmark and account valuation dates exactly
  - Use total return index data (dividends reinvested), not price-only
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import List, Optional

import pandas as pd

from pydantic import BaseModel, Field

from .canonical_schema import (
    BENCHMARK_PRICES_COLUMNS,
    BENCHMARK_PRICES_DECIMAL_COLUMNS,
    BENCHMARK_PRICES_REQUIRED_COLUMNS,
)


@dataclass
class BenchmarkComponent:
    """A single component of a composite benchmark."""
    security_id: str
    weight: Decimal  # Should sum to 1.0 across all components


class BenchmarkDefinition(BaseModel):
    """
    Reference entity for benchmark definitions.

    Supports both single-index benchmarks (e.g., S&P 500) and composite
    benchmarks (e.g., 60/40 portfolio).
    """
    benchmark_id: str
    name: str
    components: List[BenchmarkComponent] = Field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        """True if this benchmark has multiple components."""
        return len(self.components) > 1

    @property
    def is_single_index(self) -> bool:
        """True if this benchmark is a single index."""
        return len(self.components) == 1


class BenchmarkEngine:
    """
    Constructs benchmark return series from component price data.

    For single-index benchmarks: uses the component's total return directly.
    For composite benchmarks: constructs a blended return series with periodic
    rebalancing to maintain target weights.
    """

    def __init__(self, rebalance_frequency: str = "monthly"):
        """
        Initialize the benchmark engine.

        rebalance_frequency: how often to rebalance composite benchmarks
                             ("daily", "monthly", "quarterly", "yearly")
        """
        self.rebalance_frequency = rebalance_frequency

    def construct_benchmark_prices(
        self,
        definition: BenchmarkDefinition,
        component_prices_df: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Construct benchmark price series from component price data.

        Args:
            definition: BenchmarkDefinition for the benchmark
            component_prices_df: prices DataFrame with columns (security_id, date, close_price, adj_close)
            start_date: start of period (defaults to earliest date in component data)
            end_date: end of period (defaults to latest date in component data)

        Returns:
            benchmark_prices_df with columns (benchmark_id, date, level, total_return_level)
        """
        if definition.is_single_index:
            return self._construct_single_index_benchmark(
                definition, component_prices_df, start_date, end_date
            )
        else:
            return self._construct_composite_benchmark(
                definition, component_prices_df, start_date, end_date
            )

    def _construct_single_index_benchmark(
        self,
        definition: BenchmarkDefinition,
        component_prices_df: pd.DataFrame,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """Construct a single-index benchmark from one component."""
        component = definition.components[0]
        security_id = component.security_id

        # Get price data for this component
        prices = component_prices_df[component_prices_df["security_id"] == security_id].copy()

        if prices.empty:
            raise ValueError(f"No price data found for component {security_id}")

        # Filter to date range
        if start_date:
            prices = prices[prices["date"] >= start_date]
        if end_date:
            prices = prices[prices["date"] <= end_date]

        if prices.empty:
            raise ValueError(f"No price data in specified date range for {security_id}")

        # Use adj_close for total return (dividends reinvested)
        prices = prices.sort_values("date").reset_index(drop=True)

        # Normalize to start at 100
        first_level = prices.iloc[0]["adj_close"]
        prices["level"] = prices["adj_close"] / first_level * Decimal("100")

        # Build output
        benchmark_prices = pd.DataFrame(
            {
                "benchmark_id": definition.benchmark_id,
                "date": prices["date"],
                "level": prices["level"],
                "total_return_level": prices["level"],  # Same for total return index
            }
        )

        return benchmark_prices[BENCHMARK_PRICES_COLUMNS]

    def _construct_composite_benchmark(
        self,
        definition: BenchmarkDefinition,
        component_prices_df: pd.DataFrame,
        start_date: Optional[date],
        end_date: Optional[date],
    ) -> pd.DataFrame:
        """
        Construct a composite benchmark from multiple components.

        Simulates a portfolio that rebalances periodically to maintain
        target weights. Uses total return (adj_close) for each component.
        """
        # Get all component price data
        all_dates = set()
        component_data = {}

        for component in definition.components:
            security_id = component.security_id
            prices = component_prices_df[component_prices_df["security_id"] == security_id].copy()

            if prices.empty:
                raise ValueError(f"No price data found for component {security_id}")

            # Filter to date range
            if start_date:
                prices = prices[prices["date"] >= start_date]
            if end_date:
                prices = prices[prices["date"] <= end_date]

            if prices.empty:
                raise ValueError(f"No price data in specified date range for {security_id}")

            prices = prices.sort_values("date").reset_index(drop=True)
            component_data[security_id] = prices
            all_dates.update(prices["date"].tolist())

        if not all_dates:
            raise ValueError("No date range found for composite benchmark")

        # Get union of all dates
        all_dates = sorted(all_dates)

        # Determine rebalance dates
        rebalance_dates = self._get_rebalance_dates(all_dates)

        # Build composite series
        benchmark_rows = []

        # Initialize with target weights at first date
        current_weights = {c.security_id: c.weight for c in definition.components}
        portfolio_value = Decimal("100")  # Start at 100

        for i, d in enumerate(all_dates):
            # Check if this is a rebalance date
            if d in rebalance_dates:
                # Rebalance to target weights
                current_weights = {c.security_id: c.weight for c in definition.components}

            # Calculate component values
            component_values = {}
            total_value = Decimal("0")

            for component in definition.components:
                security_id = component.security_id
                weight = current_weights[security_id]

                # Get price for this date (forward-fill if missing)
                price = self._get_price_for_date(component_data[security_id], d)

                if price is None:
                    # Missing price - skip this date
                    continue

                # Component value = portfolio_value * weight
                component_value = portfolio_value * weight
                component_values[security_id] = component_value
                total_value += component_value

            if total_value == 0:
                continue

            # Calculate new portfolio value from component returns
            new_portfolio_value = Decimal("0")
            for component in definition.components:
                security_id = component.security_id
                component_value = component_values[security_id]

                # Get previous day's price to calculate return
                if i > 0:
                    prev_date = all_dates[i - 1]
                    prev_price = self._get_price_for_date(component_data[security_id], prev_date)
                    curr_price = self._get_price_for_date(component_data[security_id], d)

                    if prev_price and curr_price and prev_price != 0:
                        component_return = (curr_price - prev_price) / prev_price
                        new_component_value = component_value * (Decimal("1") + component_return)
                    else:
                        new_component_value = component_value
                else:
                    new_component_value = component_value

                new_portfolio_value += new_component_value

            portfolio_value = new_portfolio_value

            benchmark_rows.append(
                {
                    "benchmark_id": definition.benchmark_id,
                    "date": d,
                    "level": portfolio_value,
                    "total_return_level": portfolio_value,
                }
            )

        if not benchmark_rows:
            return pd.DataFrame(columns=BENCHMARK_PRICES_COLUMNS)

        benchmark_prices_df = pd.DataFrame(benchmark_rows, columns=BENCHMARK_PRICES_COLUMNS)
        return benchmark_prices_df

    def _get_rebalance_dates(self, all_dates: List[date]) -> set[date]:
        """Determine which dates are rebalance dates based on frequency."""
        if self.rebalance_frequency == "daily":
            return set(all_dates)

        rebalance_dates = set()

        for d in all_dates:
            if self.rebalance_frequency == "monthly":
                # First trading day of each month
                if d.day <= 3:  # Approximate first few days
                    rebalance_dates.add(d)
            elif self.rebalance_frequency == "quarterly":
                # First trading day of each quarter
                if d.day <= 3 and d.month in [1, 4, 7, 10]:
                    rebalance_dates.add(d)
            elif self.rebalance_frequency == "yearly":
                # First trading day of each year
                if d.day <= 3 and d.month == 1:
                    rebalance_dates.add(d)

        return rebalance_dates

    def _get_price_for_date(
        self, prices_df: pd.DataFrame, target_date: date
    ) -> Optional[Decimal]:
        """
        Get price for a specific date with forward-fill.

        Returns None if no price is available (even with forward-fill).
        """
        # Try exact match first
        exact_match = prices_df[prices_df["date"] == target_date]
        if not exact_match.empty:
            return exact_match.iloc[0]["adj_close"]

        # Forward-fill: find the most recent price before target_date
        prior_prices = prices_df[prices_df["date"] < target_date]
        if not prior_prices.empty:
            most_recent = prior_prices.iloc[-1]
            # Only forward-fill if within reasonable window (e.g., 5 days)
            if (target_date - most_recent["date"]).days <= 5:
                return most_recent["adj_close"]

        return None

    def compute_benchmark_returns(
        self,
        benchmark_prices_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute daily returns from benchmark price series.

        Args:
            benchmark_prices_df: benchmark prices from construct_benchmark_prices

        Returns:
            returns_df with columns (entity_id, entity_type, date, period_return, cumulative_return)
        """
        if benchmark_prices_df.empty:
            return pd.DataFrame(columns=[
                "entity_id", "entity_type", "date", "period_return", "cumulative_return"
            ])

        df = benchmark_prices_df.sort_values("date").reset_index(drop=True)

        returns_rows = []
        cumulative_return = Decimal("0")

        for i, row in df.iterrows():
            if i == 0:
                period_return = Decimal("0")
                cumulative_return = Decimal("0")
            else:
                prev_level = df.iloc[i - 1]["total_return_level"]
                curr_level = row["total_return_level"]

                if prev_level != 0:
                    period_return = (curr_level - prev_level) / prev_level
                else:
                    period_return = Decimal("0")

                # Cumulative return: (1 + r1) * (1 + r2) * ... - 1
                # For simplicity, we'll use the level change from inception
                inception_level = df.iloc[0]["total_return_level"]
                if inception_level != 0:
                    cumulative_return = (curr_level - inception_level) / inception_level
                else:
                    cumulative_return = Decimal("0")

            returns_rows.append(
                {
                    "entity_id": row["benchmark_id"],
                    "entity_type": "benchmark",
                    "date": row["date"],
                    "period_return": period_return,
                    "cumulative_return": cumulative_return,
                }
            )

        return pd.DataFrame(returns_rows)
