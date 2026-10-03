"""
Phase 9 — Risk Metrics Core.

Per architecture doc:
  - Volatility: annualized std dev of periodic returns
  - Sharpe/Sortino: need risk-free rate series (e.g., 3-month T-bill)
  - Beta: regression of account returns against benchmark returns
  - Max drawdown: capture drawdown duration, not just magnitude
  - Tracking error: annualized std dev of (account return - benchmark return)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    volatility: Optional[Decimal] = None  # Annualized volatility
    sharpe_ratio: Optional[Decimal] = None
    sortino_ratio: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    max_drawdown_duration: Optional[int] = None  # Days
    beta: Optional[Decimal] = None
    alpha: Optional[Decimal] = None  # Jensen's alpha
    tracking_error: Optional[Decimal] = None
    information_ratio: Optional[Decimal] = None


class RiskMetricsEngine:
    """
    Computes risk metrics for portfolio analysis.

    Uses daily returns and applies appropriate scaling factors for annualization.
    """

    def __init__(self, trading_days_per_year: int = 252):
        """
        Initialize the risk metrics engine.

        trading_days_per_year: number of trading days for annualization (default 252)
        """
        self.trading_days_per_year = trading_days_per_year

    def compute_volatility(
        self,
        returns_df: pd.DataFrame,
    ) -> Optional[Decimal]:
        """
        Compute annualized volatility from daily returns.

        Args:
            returns_df: DataFrame with 'period_return' column

        Returns:
            Annualized volatility as Decimal, or None if insufficient data
        """
        if returns_df.empty or len(returns_df) < 2:
            return None

        returns = returns_df["period_return"].dropna()

        if len(returns) < 2:
            return None

        # Convert to float for numpy
        returns_float = returns.astype(float)

        # Daily standard deviation
        daily_vol = np.std(returns_float, ddof=1)

        # Annualize: multiply by sqrt(trading_days)
        annualized_vol = daily_vol * np.sqrt(self.trading_days_per_year)

        return Decimal(str(annualized_vol))

    def compute_sharpe_ratio(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: Decimal,
    ) -> Optional[Decimal]:
        """
        Compute Sharpe ratio: (return - risk_free_rate) / volatility.

        Args:
            returns_df: DataFrame with 'period_return' column
            risk_free_rate: annual risk-free rate (e.g., 0.02 for 2%)

        Returns:
            Sharpe ratio as Decimal, or None if insufficient data
        """
        if returns_df.empty or len(returns_df) < 2:
            return None

        returns = returns_df["period_return"].dropna()

        if len(returns) < 2:
            return None

        volatility = self.compute_volatility(returns_df)

        if volatility is None or volatility == 0:
            return None

        # Annualized return
        returns_float = returns.astype(float)
        annual_return = np.mean(returns_float) * self.trading_days_per_year

        # Sharpe ratio
        excess_return = annual_return - float(risk_free_rate)
        sharpe = excess_return / float(volatility)

        return Decimal(str(sharpe))

    def compute_sortino_ratio(
        self,
        returns_df: pd.DataFrame,
        risk_free_rate: Decimal,
    ) -> Optional[Decimal]:
        """
        Compute Sortino ratio: (return - risk_free_rate) / downside_deviation.

        Only considers downside volatility (returns below risk-free rate).

        Args:
            returns_df: DataFrame with 'period_return' column
            risk_free_rate: annual risk-free rate

        Returns:
            Sortino ratio as Decimal, or None if insufficient data
        """
        if returns_df.empty or len(returns_df) < 2:
            return None

        returns = returns_df["period_return"].dropna()

        if len(returns) < 2:
            return None

        returns_float = returns.astype(float)

        # Daily risk-free rate
        daily_rf = float(risk_free_rate) / self.trading_days_per_year

        # Downside deviation: only returns below risk-free rate
        downside_returns = returns_float[returns_float < daily_rf]

        if len(downside_returns) < 2:
            # No downside returns - infinite Sortino
            return None

        downside_dev = np.std(downside_returns, ddof=1)
        annualized_downside_dev = downside_dev * np.sqrt(self.trading_days_per_year)

        if annualized_downside_dev == 0:
            return None

        # Annualized return
        annual_return = np.mean(returns_float) * self.trading_days_per_year

        # Sortino ratio
        excess_return = annual_return - float(risk_free_rate)
        sortino = excess_return / annualized_downside_dev

        return Decimal(str(sortino))

    def compute_max_drawdown(
        self,
        daily_positions_df: pd.DataFrame,
    ) -> tuple[Optional[Decimal], Optional[int]]:
        """
        Compute maximum drawdown and duration.

        Args:
            daily_positions_df: daily positions with market values

        Returns:
            Tuple of (max_drawdown_pct, duration_days)
        """
        if daily_positions_df.empty:
            return None, None

        # Compute daily portfolio values
        daily_values = daily_positions_df.groupby("date")["market_value"].sum().sort_index()

        if len(daily_values) < 2:
            return None, None

        # Convert to numpy array
        values = daily_values.values.astype(float)

        # Running maximum
        running_max = np.maximum.accumulate(values)

        # Drawdown at each point
        drawdown = (values - running_max) / running_max

        # Maximum drawdown
        max_dd = np.min(drawdown)

        # Find duration of max drawdown
        max_dd_idx = np.argmin(drawdown)
        peak_idx = np.argmax(values[:max_dd_idx + 1])
        duration = max_dd_idx - peak_idx

        return Decimal(str(max_dd)), duration

    def compute_beta(
        self,
        portfolio_returns_df: pd.DataFrame,
        benchmark_returns_df: pd.DataFrame,
    ) -> tuple[Optional[Decimal], Optional[Decimal]]:
        """
        Compute beta and alpha via linear regression.

        Args:
            portfolio_returns_df: portfolio returns with 'date' and 'period_return'
            benchmark_returns_df: benchmark returns with 'date' and 'period_return'

        Returns:
            Tuple of (beta, alpha)
        """
        if portfolio_returns_df.empty or benchmark_returns_df.empty:
            return None, None

        # Merge on date
        merged = pd.merge(
            portfolio_returns_df[["date", "period_return"]],
            benchmark_returns_df[["date", "period_return"]],
            on="date",
            suffixes=("_portfolio", "_benchmark"),
        )

        if len(merged) < 2:
            return None, None

        # Extract returns
        portfolio_returns = merged["period_return_portfolio"].astype(float).values
        benchmark_returns = merged["period_return_benchmark"].astype(float).values

        # Remove NaN values
        mask = ~np.isnan(portfolio_returns) & ~np.isnan(benchmark_returns)
        portfolio_returns = portfolio_returns[mask]
        benchmark_returns = benchmark_returns[mask]

        if len(portfolio_returns) < 2:
            return None, None

        # Linear regression: portfolio_return = alpha + beta * benchmark_return
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            benchmark_returns, portfolio_returns
        )

        beta = Decimal(str(slope))
        alpha = Decimal(str(intercept))

        return beta, alpha

    def compute_tracking_error(
        self,
        portfolio_returns_df: pd.DataFrame,
        benchmark_returns_df: pd.DataFrame,
    ) -> Optional[Decimal]:
        """
        Compute tracking error: annualized std dev of (portfolio_return - benchmark_return).

        Args:
            portfolio_returns_df: portfolio returns with 'date' and 'period_return'
            benchmark_returns_df: benchmark returns with 'date' and 'period_return'

        Returns:
            Tracking error as Decimal, or None if insufficient data
        """
        if portfolio_returns_df.empty or benchmark_returns_df.empty:
            return None

        # Merge on date
        merged = pd.merge(
            portfolio_returns_df[["date", "period_return"]],
            benchmark_returns_df[["date", "period_return"]],
            on="date",
            suffixes=("_portfolio", "_benchmark"),
        )

        if len(merged) < 2:
            return None

        # Calculate excess returns
        merged["excess_return"] = (
            merged["period_return_portfolio"] - merged["period_return_benchmark"]
        )

        excess_returns = merged["excess_return"].dropna()

        if len(excess_returns) < 2:
            return None

        # Daily standard deviation of excess returns
        daily_te = np.std(excess_returns.astype(float), ddof=1)

        # Annualize
        annualized_te = daily_te * np.sqrt(self.trading_days_per_year)

        return Decimal(str(annualized_te))

    def compute_information_ratio(
        self,
        portfolio_returns_df: pd.DataFrame,
        benchmark_returns_df: pd.DataFrame,
    ) -> Optional[Decimal]:
        """
        Compute information ratio: active_return / tracking_error.

        Args:
            portfolio_returns_df: portfolio returns with 'date' and 'period_return'
            benchmark_returns_df: benchmark returns with 'date' and 'period_return'

        Returns:
            Information ratio as Decimal, or None if insufficient data
        """
        if portfolio_returns_df.empty or benchmark_returns_df.empty:
            return None

        # Merge on date
        merged = pd.merge(
            portfolio_returns_df[["date", "period_return"]],
            benchmark_returns_df[["date", "period_return"]],
            on="date",
            suffixes=("_portfolio", "_benchmark"),
        )

        if len(merged) < 2:
            return None

        # Calculate excess returns
        merged["excess_return"] = (
            merged["period_return_portfolio"] - merged["period_return_benchmark"]
        )

        excess_returns = merged["excess_return"].dropna()

        if len(excess_returns) < 2:
            return None

        # Average excess return (annualized)
        avg_excess = np.mean(excess_returns.astype(float)) * self.trading_days_per_year

        # Tracking error
        tracking_error = self.compute_tracking_error(portfolio_returns_df, benchmark_returns_df)

        if tracking_error is None or tracking_error == 0:
            return None

        # Information ratio
        ir = avg_excess / float(tracking_error)

        return Decimal(str(ir))

    def compute_all_metrics(
        self,
        daily_positions_df: pd.DataFrame,
        portfolio_returns_df: pd.DataFrame,
        benchmark_returns_df: Optional[pd.DataFrame] = None,
        risk_free_rate: Decimal = Decimal("0.02"),
    ) -> RiskMetrics:
        """
        Compute all risk metrics in one call.

        Args:
            daily_positions_df: daily positions from valuation engine
            portfolio_returns_df: portfolio returns
            benchmark_returns_df: benchmark returns (optional)
            risk_free_rate: annual risk-free rate (default 2%)

        Returns:
            RiskMetrics object with all computed metrics
        """
        metrics = RiskMetrics()

        # Volatility
        metrics.volatility = self.compute_volatility(portfolio_returns_df)

        # Sharpe ratio
        metrics.sharpe_ratio = self.compute_sharpe_ratio(portfolio_returns_df, risk_free_rate)

        # Sortino ratio
        metrics.sortino_ratio = self.compute_sortino_ratio(portfolio_returns_df, risk_free_rate)

        # Max drawdown
        max_dd, duration = self.compute_max_drawdown(daily_positions_df)
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_duration = duration

        # Beta and alpha (if benchmark provided)
        if benchmark_returns_df is not None:
            beta, alpha = self.compute_beta(portfolio_returns_df, benchmark_returns_df)
            metrics.beta = beta
            metrics.alpha = alpha

            # Tracking error
            metrics.tracking_error = self.compute_tracking_error(
                portfolio_returns_df, benchmark_returns_df
            )

            # Information ratio
            metrics.information_ratio = self.compute_information_ratio(
                portfolio_returns_df, benchmark_returns_df
            )

        return metrics
