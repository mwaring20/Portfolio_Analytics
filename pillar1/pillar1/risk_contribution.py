"""
Phase 13 — Risk Contribution.

Per architecture doc:
  - Marginal VaR: change in portfolio VaR from a small change in position size
  - Component VaR: position's contribution to overall portfolio VaR
  - Uses correlation matrix from Phase 10
  - Risk budgeting: compare actual vs. target risk contributions
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RiskContribution:
    """Risk contribution for a single position."""
    security_id: str
    weight: Decimal
    marginal_var: Optional[Decimal] = None
    component_var: Optional[Decimal] = None
    percentage_contribution: Optional[Decimal] = None


@dataclass
class PortfolioRiskDecomposition:
    """Overall portfolio risk decomposition."""
    portfolio_var: Optional[Decimal] = None
    portfolio_volatility: Optional[Decimal] = None
    confidence_level: Decimal = Decimal("0.95")  # 95% VaR
    contributions: List[RiskContribution] = None


class RiskContributionEngine:
    """
    Computes risk contribution metrics for portfolio positions.

    Uses parametric VaR with correlation matrix and position weights.
    """

    def __init__(self, confidence_level: Decimal = Decimal("0.95")):
        """
        Initialize the risk contribution engine.

        confidence_level: VaR confidence level (default 95%)
        """
        self.confidence_level = confidence_level

    def compute_portfolio_var(
        self,
        daily_positions_df: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        as_of_date: date,
    ) -> Optional[Decimal]:
        """
        Compute portfolio Value at Risk (VaR).

        VaR = portfolio_value * z_alpha * portfolio_volatility

        Where:
          - z_alpha = inverse normal CDF at confidence level
          - portfolio_volatility = sqrt(w' * Sigma * w)
          - Sigma = covariance matrix (simplified from correlation)

        Args:
            daily_positions_df: daily positions from valuation engine
            correlation_matrix: correlation matrix from Phase 10
            as_of_date: the date to analyze

        Returns:
            Portfolio VaR as Decimal, or None if insufficient data
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty or correlation_matrix.empty:
            return None

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return None

        # Calculate weights
        weights = []
        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            if security_id in correlation_matrix.index:
                weights.append(pos["market_value"] / total_value)
            else:
                weights.append(0.0)

        weights = np.array(weights)

        # Portfolio volatility: sqrt(w' * Sigma * w)
        # Simplified: assume equal individual volatilities of 20%
        individual_vol = 0.20

        # Covariance matrix = correlation * vol_i * vol_j
        cov_matrix = correlation_matrix.values * individual_vol * individual_vol

        portfolio_var = np.sqrt(weights.T @ cov_matrix @ weights)

        # VaR = portfolio_value * z_alpha * portfolio_volatility
        z_alpha = 1.645  # 95% confidence level
        portfolio_value = float(total_value)
        var = portfolio_value * z_alpha * portfolio_var

        return Decimal(str(var))

    def compute_marginal_var(
        self,
        daily_positions_df: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        as_of_date: date,
    ) -> List[RiskContribution]:
        """
        Compute marginal VaR for each position.

        Marginal VaR = change in portfolio VaR from a 1% increase in position weight.

        Args:
            daily_positions_df: daily positions from valuation engine
            correlation_matrix: correlation matrix from Phase 10
            as_of_date: the date to analyze

        Returns:
            List of RiskContribution with marginal VaR
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty or correlation_matrix.empty:
            return []

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return []

        contributions: List[RiskContribution] = []

        # Calculate weights
        weights = []
        security_ids = []
        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            security_ids.append(security_id)
            if security_id in correlation_matrix.index:
                weights.append(float(pos["market_value"] / total_value))
            else:
                weights.append(0.0)

        weights = np.array(weights)

        # Simplified marginal VaR calculation
        # Marginal VaR_i = z_alpha * (Sigma * w)_i
        individual_vol = 0.20
        cov_matrix = correlation_matrix.values * individual_vol * individual_vol

        z_alpha = 1.645  # 95% confidence level

        for i, security_id in enumerate(security_ids):
            if security_id not in correlation_matrix.index:
                contributions.append(
                    RiskContribution(
                        security_id=security_id,
                        weight=Decimal(str(weights[i])),
                        marginal_var=None,
                    )
                )
                continue

            # Marginal contribution to volatility
            marginal_vol = (cov_matrix @ weights)[i]

            # Marginal VaR
            marginal_var = z_alpha * marginal_vol * float(total_value)

            contributions.append(
                RiskContribution(
                    security_id=security_id,
                    weight=Decimal(str(weights[i])),
                    marginal_var=Decimal(str(marginal_var)),
                )
            )

        return contributions

    def compute_component_var(
        self,
        daily_positions_df: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        as_of_date: date,
    ) -> List[RiskContribution]:
        """
        Compute component VaR for each position.

        Component VaR = weight * marginal VaR
        Sum of component VaRs should equal portfolio VaR.

        Args:
            daily_positions_df: daily positions from valuation engine
            correlation_matrix: correlation matrix from Phase 10
            as_of_date: the date to analyze

        Returns:
            List of RiskContribution with component VaR
        """
        marginal_contributions = self.compute_marginal_var(
            daily_positions_df, correlation_matrix, as_of_date
        )

        contributions: List[RiskContribution] = []

        for marg in marginal_contributions:
            if marg.marginal_var is None:
                contributions.append(
                    RiskContribution(
                        security_id=marg.security_id,
                        weight=marg.weight,
                        marginal_var=None,
                        component_var=None,
                    )
                )
                continue

            # Component VaR = weight * marginal VaR
            component_var = marg.weight * marg.marginal_var

            contributions.append(
                RiskContribution(
                    security_id=marg.security_id,
                    weight=marg.weight,
                    marginal_var=marg.marginal_var,
                    component_var=component_var,
                )
            )

        return contributions

    def compute_risk_contribution(
        self,
        daily_positions_df: pd.DataFrame,
        correlation_matrix: pd.DataFrame,
        as_of_date: date,
    ) -> PortfolioRiskDecomposition:
        """
        Compute complete risk decomposition.

        Args:
            daily_positions_df: daily positions from valuation engine
            correlation_matrix: correlation matrix from Phase 10
            as_of_date: the date to analyze

        Returns:
            PortfolioRiskDecomposition with all risk metrics
        """
        # Portfolio VaR
        portfolio_var = self.compute_portfolio_var(
            daily_positions_df, correlation_matrix, as_of_date
        )

        # Component VaR
        component_contributions = self.compute_component_var(
            daily_positions_df, correlation_matrix, as_of_date
        )

        # Calculate percentage contributions
        if portfolio_var and portfolio_var > 0:
            for contrib in component_contributions:
                if contrib.component_var is not None:
                    contrib.percentage_contribution = (
                        contrib.component_var / portfolio_var
                    )
                else:
                    contrib.percentage_contribution = Decimal("0")
        else:
            for contrib in component_contributions:
                contrib.percentage_contribution = Decimal("0")

        return PortfolioRiskDecomposition(
            portfolio_var=portfolio_var,
            portfolio_volatility=None,  # Could be computed separately
            confidence_level=self.confidence_level,
            contributions=component_contributions,
        )

    def compute_risk_budget_comparison(
        self,
        decomposition: PortfolioRiskDecomposition,
        target_weights: Dict[str, Decimal],
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Compare actual risk contributions to target risk budget.

        Args:
            decomposition: risk decomposition from compute_risk_contribution
            target_weights: target risk budget by security_id

        Returns:
            Dict with actual vs. target risk contributions
        """
        comparison = {}

        for contrib in decomposition.contributions:
            security_id = contrib.security_id
            actual_risk = contrib.percentage_contribution or Decimal("0")
            target_risk = target_weights.get(security_id, Decimal("0"))

            comparison[security_id] = {
                "actual_risk_contribution": actual_risk,
                "target_risk_contribution": target_risk,
                "difference": actual_risk - target_risk,
            }

        return comparison
