"""
Phase 11 — Diversification Analysis.

Per architecture doc:
  - Effective number of securities: 1 / sum of squared weights
  - Herfindahl-Hirschman Index (HHI): sum of squared weights
  - Diversification ratio: weighted average volatility / portfolio volatility
  - Uses correlation matrix from Phase 10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DiversificationMetrics:
    """Container for diversification metrics."""
    effective_number_of_securities: Optional[Decimal] = None
    hhi: Optional[Decimal] = None  # Herfindahl-Hirschman Index
    diversification_ratio: Optional[Decimal] = None
    concentration_ratio: Optional[Decimal] = None  # Top N concentration


class DiversificationEngine:
    """
    Analyzes portfolio diversification using concentration and correlation metrics.

    Combines weight-based concentration (ENS, HHI) with correlation-based diversification.
    """

    def __init__(self):
        """Initialize the diversification engine."""
        pass

    def compute_effective_number_of_securities(
        self,
        daily_positions_df: pd.DataFrame,
        as_of_date: date,
    ) -> Optional[Decimal]:
        """
        Compute effective number of securities (ENS).

        ENS = 1 / sum(w_i^2) where w_i is the weight of security i.

        A value of 10 means the portfolio is as diversified as 10 equal-weighted positions.

        Args:
            daily_positions_df: daily positions from valuation engine
            as_of_date: the date to analyze

        Returns:
            ENS as Decimal, or None if portfolio is empty
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return None

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return None

        # Calculate weights
        weights = positions["market_value"] / total_value

        # Sum of squared weights
        sum_squared_weights = (weights ** 2).sum()

        if sum_squared_weights == 0:
            return None

        # ENS = 1 / sum(w_i^2)
        ens = 1 / sum_squared_weights

        return Decimal(str(ens))

    def compute_hhi(
        self,
        daily_positions_df: pd.DataFrame,
        as_of_date: date,
    ) -> Optional[Decimal]:
        """
        Compute Herfindahl-Hirschman Index (HHI).

        HHI = sum(w_i^2) where w_i is the weight of security i.

        Range: 1/N (perfectly diversified) to 1 (single position).
        Often multiplied by 10,000 for regulatory use (0-10,000 scale).

        Args:
            daily_positions_df: daily positions from valuation engine
            as_of_date: the date to analyze

        Returns:
            HHI as Decimal (0-1 scale), or None if portfolio is empty
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return None

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return None

        # Calculate weights
        weights = positions["market_value"] / total_value

        # Sum of squared weights
        hhi = (weights ** 2).sum()

        return Decimal(str(hhi))

    def compute_concentration_ratio(
        self,
        daily_positions_df: pd.DataFrame,
        as_of_date: date,
        top_n: int = 5,
    ) -> Optional[Decimal]:
        """
        Compute concentration ratio: weight of top N positions.

        Args:
            daily_positions_df: daily positions from valuation engine
            as_of_date: the date to analyze
            top_n: number of top positions to consider

        Returns:
            Concentration ratio as Decimal, or None if portfolio is empty
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return None

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return None

        # Sort by market value descending
        positions_sorted = positions.sort_values("market_value", ascending=False)

        # Sum top N
        top_n_value = positions_sorted.head(top_n)["market_value"].sum()

        concentration_ratio = top_n_value / total_value

        return Decimal(str(concentration_ratio))

    def compute_diversification_ratio(
        self,
        daily_positions_df: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        as_of_date: date,
    ) -> Optional[Decimal]:
        """
        Compute diversification ratio using correlation matrix.

        Diversification Ratio = (sum(w_i * sigma_i)) / sigma_portfolio

        Where:
          - w_i = weight of security i
          - sigma_i = volatility of security i (simplified as sqrt of variance)
          - sigma_portfolio = portfolio volatility

        This is a simplified version. A full implementation would need individual
        security volatilities from price data.

        Args:
            daily_positions_df: daily positions from valuation engine
            correlation_matrix: correlation matrix from Phase 10
            as_of_date: the date to analyze

        Returns:
            Diversification ratio as Decimal, or None if insufficient data
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty or correlation_matrix.empty:
            return None

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return None

        # Calculate weights
        weights = positions["market_value"] / total_value

        # For simplification, assume equal volatility for all securities
        # A full implementation would use actual security volatilities
        # Diversification ratio = weighted average of 1 / sqrt(correlation)
        # This is a proxy: higher correlation = lower diversification

        # Get average correlation for each security
        avg_correlations = []
        for sec_id in positions["security_id"]:
            if sec_id in correlation_matrix.index:
                # Average correlation with other securities
                correlations = correlation_matrix.loc[sec_id].drop(sec_id)
                if not correlations.empty:
                    avg_corr = correlations.mean()
                    avg_correlations.append(avg_corr)
                else:
                    avg_correlations.append(0.0)
            else:
                avg_correlations.append(0.0)

        # Diversification ratio: higher when correlations are lower
        # Simplified: 1 / average correlation
        if avg_correlations:
            avg_correlation = np.mean(avg_correlations)
            if avg_correlation > 0:
                div_ratio = 1 / avg_correlation
                return Decimal(str(div_ratio))

        return None

    def compute_all_metrics(
        self,
        daily_positions_df: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None,
        as_of_date: date = None,
    ) -> DiversificationMetrics:
        """
        Compute all diversification metrics.

        Args:
            daily_positions_df: daily positions from valuation engine
            correlation_matrix: correlation matrix from Phase 10 (optional)
            as_of_date: the date to analyze

        Returns:
            DiversificationMetrics object with all computed metrics
        """
        if as_of_date is None and not daily_positions_df.empty:
            as_of_date = daily_positions_df["date"].max()

        if as_of_date is None:
            return DiversificationMetrics()

        metrics = DiversificationMetrics()

        # Effective number of securities
        metrics.effective_number_of_securities = self.compute_effective_number_of_securities(
            daily_positions_df, as_of_date
        )

        # HHI
        metrics.hhi = self.compute_hhi(daily_positions_df, as_of_date)

        # Concentration ratio (top 5)
        metrics.concentration_ratio = self.compute_concentration_ratio(
            daily_positions_df, as_of_date, top_n=5
        )

        # Diversification ratio (if correlation matrix provided)
        if correlation_matrix is not None:
            metrics.diversification_ratio = self.compute_diversification_ratio(
                daily_positions_df, correlation_matrix, as_of_date
            )

        return metrics

    def assess_diversification_quality(
        self,
        metrics: DiversificationMetrics,
    ) -> Dict[str, str]:
        """
        Assess diversification quality based on metrics.

        Returns qualitative assessment (e.g., "Well Diversified", "Concentrated").

        Args:
            metrics: DiversificationMetrics from compute_all_metrics

        Returns:
            Dict with qualitative assessments
        """
        assessments = {}

        # ENS assessment
        if metrics.effective_number_of_securities:
            ens = metrics.effective_number_of_securities
            if ens >= 20:
                assessments["ens_quality"] = "Well Diversified"
            elif ens >= 10:
                assessments["ens_quality"] = "Moderately Diversified"
            elif ens >= 5:
                assessments["ens_quality"] = "Somewhat Concentrated"
            else:
                assessments["ens_quality"] = "Highly Concentrated"

        # HHI assessment
        if metrics.hhi:
            hhi = metrics.hhi
            # On 0-1 scale
            if hhi <= 0.05:
                assessments["hhi_quality"] = "Low Concentration"
            elif hhi <= 0.15:
                assessments["hhi_quality"] = "Moderate Concentration"
            elif hhi <= 0.25:
                assessments["hhi_quality"] = "High Concentration"
            else:
                assessments["hhi_quality"] = "Very High Concentration"

        # Concentration ratio assessment
        if metrics.concentration_ratio:
            cr = metrics.concentration_ratio
            if cr <= 0.4:
                assessments["concentration_quality"] = "Low Concentration"
            elif cr <= 0.6:
                assessments["concentration_quality"] = "Moderate Concentration"
            elif cr <= 0.8:
                assessments["concentration_quality"] = "High Concentration"
            else:
                assessments["concentration_quality"] = "Very High Concentration"

        return assessments
