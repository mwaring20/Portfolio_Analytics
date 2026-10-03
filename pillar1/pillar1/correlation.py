"""
Phase 10 — Correlation Matrix.

Per architecture doc:
  - Correlation across holdings (not just asset classes)
  - Lookback window selection: 1M, 3M, 1Y, 3Y
  - Used by diversification analysis (Phase 11), risk contribution (Phase 13), stress testing (Phase 14)
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class LookbackWindow(Enum):
    """Standard lookback windows for correlation analysis."""
    ONE_MONTH = "1M"
    THREE_MONTHS = "3M"
    ONE_YEAR = "1Y"
    THREE_YEARS = "3Y"

    def to_days(self) -> int:
        """Convert lookback window to approximate trading days."""
        mapping = {
            LookbackWindow.ONE_MONTH: 21,  # ~21 trading days
            LookbackWindow.THREE_MONTHS: 63,  # ~63 trading days
            LookbackWindow.ONE_YEAR: 252,  # ~252 trading days
            LookbackWindow.THREE_YEARS: 756,  # ~756 trading days
        }
        return mapping[self]


class CorrelationEngine:
    """
    Computes correlation matrices across portfolio holdings.

    Uses daily returns and configurable lookback windows.
    """

    def __init__(self, trading_days_per_year: int = 252):
        """
        Initialize the correlation engine.

        trading_days_per_year: number of trading days for window calculations
        """
        self.trading_days_per_year = trading_days_per_year

    def compute_correlation_matrix(
        self,
        daily_positions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        as_of_date: date,
        lookback: LookbackWindow = LookbackWindow.ONE_YEAR,
        min_observations: int = 20,
    ) -> pd.DataFrame:
        """
        Compute correlation matrix across holdings.

        Args:
            daily_positions_df: daily positions from valuation engine
            prices_df: price data with columns (security_id, date, close_price, adj_close)
            as_of_date: the date for which to compute correlation
            lookback: lookback window for correlation calculation
            min_observations: minimum number of observations required

        Returns:
            DataFrame with correlation matrix (securities as rows and columns)
        """
        # Get holdings as of as_of_date
        holdings = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if holdings.empty:
            return pd.DataFrame()

        # Get unique securities
        securities = holdings["security_id"].unique().tolist()

        # Determine start date based on lookback
        lookback_days = lookback.to_days()
        start_date = as_of_date - timedelta(days=lookback_days * 2)  # Buffer for non-trading days

        # Get price data for all securities in lookback window
        price_data = prices_df[
            (prices_df["security_id"].isin(securities))
            & (prices_df["date"] >= start_date)
            & (prices_df["date"] <= as_of_date)
        ].copy()

        if price_data.empty:
            return pd.DataFrame()

        # Pivot to get returns by security
        price_pivot = price_data.pivot(index="date", columns="security_id", values="adj_close")

        # Calculate daily returns
        returns = price_pivot.pct_change().dropna()

        # Filter to lookback window (trading days)
        if len(returns) > lookback_days:
            returns = returns.tail(lookback_days)

        # Check minimum observations
        if len(returns) < min_observations:
            return pd.DataFrame()

        # Compute correlation matrix
        correlation_matrix = returns.corr()

        # Fill NaN with 0 (no correlation for insufficient data)
        correlation_matrix = correlation_matrix.fillna(0)

        return correlation_matrix

    def compute_average_correlation(
        self,
        correlation_matrix: pd.DataFrame,
    ) -> Optional[Decimal]:
        """
        Compute average correlation across all pairs.

        Args:
            correlation_matrix: correlation matrix from compute_correlation_matrix

        Returns:
            Average correlation as Decimal, or None if matrix is empty
        """
        if correlation_matrix.empty:
            return None

        # Extract upper triangle (excluding diagonal)
        values = []
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                values.append(correlation_matrix.iloc[i, j])

        if not values:
            return None

        avg_correlation = np.mean(values)
        return Decimal(str(avg_correlation))

    def compute_highest_correlation_pairs(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: Decimal = Decimal("0.7"),
        top_n: int = 10,
    ) -> List[Dict[str, any]]:
        """
        Find pairs with highest correlation.

        Args:
            correlation_matrix: correlation matrix from compute_correlation_matrix
            threshold: minimum correlation to include
            top_n: maximum number of pairs to return

        Returns:
            List of dicts with security_1, security_2, correlation
        """
        if correlation_matrix.empty:
            return []

        pairs = []

        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                sec1 = correlation_matrix.index[i]
                sec2 = correlation_matrix.columns[j]
                corr = correlation_matrix.iloc[i, j]

                if corr >= float(threshold):
                    pairs.append(
                        {
                            "security_1": sec1,
                            "security_2": sec2,
                            "correlation": Decimal(str(corr)),
                        }
                    )

        # Sort by correlation descending
        pairs.sort(key=lambda x: x["correlation"], reverse=True)

        # Return top N
        return pairs[:top_n]

    def compute_portfolio_correlation_with_benchmark(
        self,
        portfolio_returns_df: pd.DataFrame,
        benchmark_returns_df: pd.DataFrame,
        as_of_date: date,
        lookback: LookbackWindow = LookbackWindow.ONE_YEAR,
    ) -> Optional[Decimal]:
        """
        Compute correlation between portfolio and benchmark.

        Args:
            portfolio_returns_df: portfolio returns with 'date' and 'period_return'
            benchmark_returns_df: benchmark returns with 'date' and 'period_return'
            as_of_date: the date for which to compute correlation
            lookback: lookback window

        Returns:
            Correlation as Decimal, or None if insufficient data
        """
        # Determine start date
        lookback_days = lookback.to_days()
        start_date = as_of_date - timedelta(days=lookback_days * 2)

        # Filter to lookback window
        portfolio_filtered = portfolio_returns_df[
            (portfolio_returns_df["date"] >= start_date)
            & (portfolio_returns_df["date"] <= as_of_date)
        ].copy()

        benchmark_filtered = benchmark_returns_df[
            (benchmark_returns_df["date"] >= start_date)
            & (benchmark_returns_df["date"] <= as_of_date)
        ].copy()

        if portfolio_filtered.empty or benchmark_filtered.empty:
            return None

        # Merge on date
        merged = pd.merge(
            portfolio_filtered[["date", "period_return"]],
            benchmark_filtered[["date", "period_return"]],
            on="date",
            suffixes=("_portfolio", "_benchmark"),
        )

        if len(merged) < 20:  # Minimum observations
            return None

        # Compute correlation
        correlation = merged["period_return_portfolio"].corr(
            merged["period_return_benchmark"]
        )

        if pd.isna(correlation):
            return None

        return Decimal(str(correlation))

    def compute_sector_correlation(
        self,
        correlation_matrix: pd.DataFrame,
        security_master: Dict[str, any],
        dimension: str = "sector",
    ) -> pd.DataFrame:
        """
        Aggregate correlation matrix by sector or asset class.

        Args:
            correlation_matrix: correlation matrix from compute_correlation_matrix
            security_master: mapping of security_id to SecurityMaster
            dimension: aggregation dimension ("sector" or "asset_class")

        Returns:
            DataFrame with correlation matrix by sector
        """
        if correlation_matrix.empty:
            return pd.DataFrame()

        # Map securities to sectors
        security_to_sector = {}
        for security_id in correlation_matrix.index:
            if security_id in security_master:
                sec = security_master[security_id]
                if dimension == "sector":
                    sector = sec.sector or "Unknown"
                elif dimension == "asset_class":
                    sector = sec.asset_class.value
                else:
                    sector = "Unknown"
                security_to_sector[security_id] = sector
            else:
                security_to_sector[security_id] = "Unknown"

        # Create sector-level correlation
        sectors = list(set(security_to_sector.values()))
        sector_correlation = pd.DataFrame(index=sectors, columns=sectors, data=0)

        # Aggregate correlations
        for sec1 in correlation_matrix.index:
            for sec2 in correlation_matrix.columns:
                sector1 = security_to_sector[sec1]
                sector2 = security_to_sector[sec2]

                if sector1 == sector2:
                    # Same sector - use average correlation within sector
                    continue

                # Add correlation to sector pair
                sector_correlation.loc[sector1, sector2] += correlation_matrix.loc[sec1, sec2]
                sector_correlation.loc[sector2, sector1] += correlation_matrix.loc[sec1, sec2]

        # Normalize by count
        for sector1 in sectors:
            for sector2 in sectors:
                if sector1 != sector2:
                    # Count number of security pairs
                    count = 0
                    for sec1 in correlation_matrix.index:
                        for sec2 in correlation_matrix.columns:
                            if (
                                security_to_sector[sec1] == sector1
                                and security_to_sector[sec2] == sector2
                            ):
                                count += 1

                    if count > 0:
                        sector_correlation.loc[sector1, sector2] /= count

        # Diagonal is 1 (perfect correlation with itself)
        for sector in sectors:
            sector_correlation.loc[sector, sector] = Decimal("1")

        return sector_correlation
