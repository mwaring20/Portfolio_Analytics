"""
Phase 8 — Attribution Analysis.

Per architecture doc:
  - Brinson-Fachler decomposition (allocation, selection, interaction effects)
  - Holding-level attribution needs per-security returns and weights
  - Single-period attribution per standard period is a reasonable MVP
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd

from .security_master import SecurityMaster


@dataclass
class AttributionResult:
    """Result of attribution analysis for a single category."""
    category: str  # e.g., asset class, sector
    portfolio_weight: Decimal
    benchmark_weight: Decimal
    portfolio_return: Decimal
    benchmark_return: Decimal
    allocation_effect: Decimal
    selection_effect: Decimal
    interaction_effect: Decimal
    total_effect: Decimal


@dataclass
class AttributionSummary:
    """Summary of attribution analysis."""
    portfolio_return: Decimal
    benchmark_return: Decimal
    active_return: Decimal  # portfolio - benchmark
    allocation_effect: Decimal
    selection_effect: Decimal
    interaction_effect: Decimal
    total_active_return: Decimal  # Should equal active return
    by_category: List[AttributionResult]


class AttributionEngine:
    """
    Brinson-Fachler attribution analysis.

    Decomposes portfolio return relative to benchmark into:
      - Allocation effect: overweighting/underweighting outperforming sectors
      - Selection effect: security selection within sectors
      - Interaction effect: interaction between allocation and selection

    Formula (Brinson-Fachler):
      Allocation = (Wp - Wb) * (Rb - Rb_total)
      Selection = Wb * (Rp - Rb)
      Interaction = (Wp - Wb) * (Rp - Rb)

    Where:
      Wp = portfolio weight in category
      Wb = benchmark weight in category
      Rp = portfolio return in category
      Rb = benchmark return in category
      Rb_total = total benchmark return
    """

    def __init__(self):
        """Initialize the attribution engine."""
        pass

    def compute_attribution(
        self,
        daily_positions_df: pd.DataFrame,
        benchmark_prices_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        start_date: date,
        end_date: date,
        dimension: str = "asset_class",
    ) -> AttributionSummary:
        """
        Compute Brinson-Fachler attribution analysis.

        Args:
            daily_positions_df: daily positions from valuation engine
            benchmark_prices_df: benchmark price series
            security_master: mapping of security_id to SecurityMaster
            start_date: start of attribution period
            end_date: end of attribution period
            dimension: attribution dimension ("asset_class", "sector")

        Returns:
            AttributionSummary with decomposition results
        """
        # Compute portfolio returns by category
        portfolio_returns = self._compute_portfolio_returns_by_category(
            daily_positions_df, security_master, start_date, end_date, dimension
        )

        # Compute benchmark returns by category
        benchmark_returns = self._compute_benchmark_returns_by_category(
            benchmark_prices_df, security_master, start_date, end_date, dimension
        )

        # Compute overall portfolio and benchmark returns
        portfolio_total_return = self._compute_total_portfolio_return(
            daily_positions_df, start_date, end_date
        )
        benchmark_total_return = self._compute_total_benchmark_return(
            benchmark_prices_df, start_date, end_date
        )

        # Compute attribution effects for each category
        category_results: List[AttributionResult] = []
        all_categories = set(portfolio_returns.keys()) | set(benchmark_returns.keys())

        total_allocation = Decimal("0")
        total_selection = Decimal("0")
        total_interaction = Decimal("0")

        for category in sorted(all_categories):
            wp = portfolio_returns.get(category, {}).get("weight", Decimal("0"))
            wb = benchmark_returns.get(category, {}).get("weight", Decimal("0"))
            rp = portfolio_returns.get(category, {}).get("return", Decimal("0"))
            rb = benchmark_returns.get(category, {}).get("return", Decimal("0"))

            # Brinson-Fachler formulas
            allocation_effect = (wp - wb) * (rb - benchmark_total_return)
            selection_effect = wb * (rp - rb)
            interaction_effect = (wp - wb) * (rp - rb)
            total_effect = allocation_effect + selection_effect + interaction_effect

            category_results.append(
                AttributionResult(
                    category=category,
                    portfolio_weight=wp,
                    benchmark_weight=wb,
                    portfolio_return=rp,
                    benchmark_return=rb,
                    allocation_effect=allocation_effect,
                    selection_effect=selection_effect,
                    interaction_effect=interaction_effect,
                    total_effect=total_effect,
                )
            )

            total_allocation += allocation_effect
            total_selection += selection_effect
            total_interaction += interaction_effect

        active_return = portfolio_total_return - benchmark_total_return
        total_active_return = total_allocation + total_selection + total_interaction

        return AttributionSummary(
            portfolio_return=portfolio_total_return,
            benchmark_return=benchmark_total_return,
            active_return=active_return,
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            total_active_return=total_active_return,
            by_category=category_results,
        )

    def _compute_portfolio_returns_by_category(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        start_date: date,
        end_date: date,
        dimension: str,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compute portfolio weights and returns by category.

        Returns dict: {category: {"weight": Decimal, "return": Decimal}}
        """
        # Filter to date range
        df = daily_positions_df[
            (daily_positions_df["date"] >= start_date)
            & (daily_positions_df["date"] <= end_date)
        ].copy()

        if df.empty:
            return {}

        # Get start and end values by category
        start_date_df = df[df["date"] == start_date]
        end_date_df = df[df["date"] == end_date]

        if start_date_df.empty or end_date_df.empty:
            return {}

        category_values_start: Dict[str, Decimal] = {}
        category_values_end: Dict[str, Decimal] = {}
        total_value_start = Decimal("0")
        total_value_end = Decimal("0")

        # Get start values by category
        for _, pos in start_date_df.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]
            total_value_start += market_value

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            if dimension == "asset_class":
                category = sec.asset_class.value
            elif dimension == "sector":
                category = sec.sector or "Unknown"
            else:
                raise ValueError(f"Unknown dimension: {dimension}")

            if category not in category_values_start:
                category_values_start[category] = Decimal("0")
            category_values_start[category] += market_value

        # Get end values by category
        for _, pos in end_date_df.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]
            total_value_end += market_value

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            if dimension == "asset_class":
                category = sec.asset_class.value
            elif dimension == "sector":
                category = sec.sector or "Unknown"
            else:
                raise ValueError(f"Unknown dimension: {dimension}")

            if category not in category_values_end:
                category_values_end[category] = Decimal("0")
            category_values_end[category] += market_value

        # Compute weights and returns
        result: Dict[str, Dict[str, Decimal]] = {}
        all_categories = set(category_values_start.keys()) | set(category_values_end.keys())

        for category in all_categories:
            start_val = category_values_start.get(category, Decimal("0"))
            end_val = category_values_end.get(category, Decimal("0"))

            weight = start_val / total_value_start if total_value_start > 0 else Decimal("0")

            if start_val > 0:
                category_return = (end_val - start_val) / start_val
            else:
                category_return = Decimal("0")

            result[category] = {
                "weight": weight,
                "return": category_return,
            }

        return result

    def _compute_benchmark_returns_by_category(
        self,
        benchmark_prices_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        start_date: date,
        end_date: date,
        dimension: str,
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compute benchmark weights and returns by category.

        For simplicity, this assumes the benchmark is a single index.
        For composite benchmarks, would need to decompose by component.
        """
        # Filter to date range
        df = benchmark_prices_df[
            (benchmark_prices_df["date"] >= start_date)
            & (benchmark_prices_df["date"] <= end_date)
        ].copy()

        if df.empty:
            return {}

        # Get start and end levels
        start_row = df[df["date"] == start_date]
        end_row = df[df["date"] == end_date]

        if start_row.empty or end_row.empty:
            return {}

        start_level = start_row.iloc[0]["total_return_level"]
        end_level = end_row.iloc[0]["total_return_level"]

        # For a single-index benchmark, return is simple
        if start_level > 0:
            benchmark_return = (end_level - start_level) / start_level
        else:
            benchmark_return = Decimal("0")

        # For single-index, weight is 100% in the single category
        # This is a simplification - real implementation would need
        # benchmark component mapping to categories
        benchmark_id = df.iloc[0]["benchmark_id"]

        # Try to map benchmark to a category
        if benchmark_id in security_master:
            sec = security_master[benchmark_id]
            if dimension == "asset_class":
                category = sec.asset_class.value
            elif dimension == "sector":
                category = sec.sector or "Unknown"
            else:
                raise ValueError(f"Unknown dimension: {dimension}")

            return {
                category: {
                    "weight": Decimal("1.0"),
                    "return": benchmark_return,
                }
            }

        # Fallback: assume equity for unknown benchmarks
        return {
            "Equity": {
                "weight": Decimal("1.0"),
                "return": benchmark_return,
            }
        }

    def _compute_total_portfolio_return(
        self,
        daily_positions_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> Decimal:
        """Compute total portfolio return over period."""
        df = daily_positions_df[
            (daily_positions_df["date"] >= start_date)
            & (daily_positions_df["date"] <= end_date)
        ].copy()

        if df.empty:
            return Decimal("0")

        start_date_df = df[df["date"] == start_date]
        end_date_df = df[df["date"] == end_date]

        if start_date_df.empty or end_date_df.empty:
            return Decimal("0")

        total_start = start_date_df["market_value"].sum()
        total_end = end_date_df["market_value"].sum()

        if total_start == 0:
            return Decimal("0")

        return (total_end - total_start) / total_start

    def _compute_total_benchmark_return(
        self,
        benchmark_prices_df: pd.DataFrame,
        start_date: date,
        end_date: date,
    ) -> Decimal:
        """Compute total benchmark return over period."""
        df = benchmark_prices_df[
            (benchmark_prices_df["date"] >= start_date)
            & (benchmark_prices_df["date"] <= end_date)
        ].copy()

        if df.empty:
            return Decimal("0")

        start_row = df[df["date"] == start_date]
        end_row = df[df["date"] == end_date]

        if start_row.empty or end_row.empty:
            return Decimal("0")

        start_level = start_row.iloc[0]["total_return_level"]
        end_level = end_row.iloc[0]["total_return_level"]

        if start_level == 0:
            return Decimal("0")

        return (end_level - start_level) / start_level
