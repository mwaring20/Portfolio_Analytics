"""
Phase 13 — Fama-French Factor Exposure.

Per architecture doc:
  - Source factor returns from Ken French Data Library (market, SMB, HML, optionally momentum/quality)
  - Run multiple linear regression of excess returns against factor returns (statsmodels) over a chosen lookback window
  - Report R² alongside coefficients — a low R² means the model isn't explaining much
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class FactorExposure:
    """Factor exposure for a single factor."""
    factor_name: str
    coefficient: Optional[Decimal] = None
    t_statistic: Optional[Decimal] = None
    p_value: Optional[Decimal] = None


@dataclass
class FactorRegressionResult:
    """Result of Fama-French factor regression."""
    entity_id: str
    entity_type: str  # "account", "security", "composite"
    lookback_months: int
    r_squared: Optional[Decimal] = None
    intercept: Optional[Decimal] = None
    factor_exposures: List[FactorExposure] = None


class FamaFrenchEngine:
    """
    Computes Fama-French factor exposures via linear regression.

    Uses factor returns from Ken French Data Library format.
    """

    def __init__(self):
        """Initialize the Fama-French engine."""
        pass

    def run_regression(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
        entity_id: str,
        entity_type: str = "account",
        lookback_months: int = 36,
        risk_free_rate: Optional[pd.DataFrame] = None,
    ) -> FactorRegressionResult:
        """
        Run Fama-French factor regression.

        Args:
            returns_df: entity returns with 'date' and 'period_return' columns
            factor_returns_df: factor returns with columns for Mkt-RF, SMB, HML, etc.
            entity_id: identifier for the entity being analyzed
            entity_type: type of entity (account, security, composite)
            lookback_months: number of months to use in regression
            risk_free_rate: optional risk-free rate series for excess returns

        Returns:
            FactorRegressionResult with coefficients and R²
        """
        # Filter returns to lookback window
        if not returns_df.empty:
            returns_df = returns_df.tail(lookback_months).copy()

        if returns_df.empty or len(returns_df) < 12:
            return FactorRegressionResult(
                entity_id=entity_id,
                entity_type=entity_type,
                lookback_months=lookback_months,
                r_squared=None,
                intercept=None,
                factor_exposures=[],
            )

        # Merge with factor returns on date
        merged = pd.merge(
            returns_df[["date", "period_return"]],
            factor_returns_df,
            on="date",
            how="inner",
        )

        if merged.empty or len(merged) < 12:
            return FactorRegressionResult(
                entity_id=entity_id,
                entity_type=entity_type,
                lookback_months=lookback_months,
                r_squared=None,
                intercept=None,
                factor_exposures=[],
            )

        # Calculate excess returns if risk-free rate provided
        if risk_free_rate is not None:
            merged = pd.merge(
                merged,
                risk_free_rate,
                on="date",
                how="left",
            )
            merged["excess_return"] = merged["period_return"] - merged["rf"].fillna(0)
            y = merged["excess_return"].astype(float)
        else:
            y = merged["period_return"].astype(float)

        # Get factor columns (exclude date and return columns)
        factor_cols = [col for col in merged.columns if col not in ["date", "period_return", "excess_return", "rf"]]

        if not factor_cols:
            return FactorRegressionResult(
                entity_id=entity_id,
                entity_type=entity_type,
                lookback_months=lookback_months,
                r_squared=None,
                intercept=None,
                factor_exposures=[],
            )

        # Prepare X matrix (add intercept)
        X = merged[factor_cols].astype(float).values
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        # Run OLS regression
        try:
            coefficients, residuals, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

            # Calculate R-squared
            y_pred = X_with_intercept @ coefficients
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Calculate standard errors and t-statistics
            n = len(y)
            p = X_with_intercept.shape[1]
            mse = ss_res / (n - p) if n > p else 0
            var_cov = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept) if n > p else None

            factor_exposures: List[FactorExposure] = []

            # Intercept
            intercept = Decimal(str(coefficients[0]))

            # Factor exposures
            for i, factor_name in enumerate(factor_cols):
                coef = Decimal(str(coefficients[i + 1]))

                if var_cov is not None:
                    std_error = np.sqrt(var_cov[i + 1, i + 1])
                    t_stat = coefficients[i + 1] / std_error if std_error > 0 else 0
                    # Two-tailed p-value
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - p))
                else:
                    t_stat = None
                    p_val = None

                factor_exposures.append(
                    FactorExposure(
                        factor_name=factor_name,
                        coefficient=coef,
                        t_statistic=Decimal(str(t_stat)) if t_stat is not None else None,
                        p_value=Decimal(str(p_val)) if p_val is not None else None,
                    )
                )

            return FactorRegressionResult(
                entity_id=entity_id,
                entity_type=entity_type,
                lookback_months=lookback_months,
                r_squared=Decimal(str(r_squared)),
                intercept=intercept,
                factor_exposures=factor_exposures,
            )

        except np.linalg.LinAlgError:
            return FactorRegressionResult(
                entity_id=entity_id,
                entity_type=entity_type,
                lookback_months=lookback_months,
                r_squared=None,
                intercept=None,
                factor_exposures=[],
            )

    def generate_factor_report(
        self,
        result: FactorRegressionResult,
    ) -> str:
        """
        Generate a human-readable factor exposure report.

        Args:
            result: FactorRegressionResult from run_regression

        Returns:
            Formatted report string
        """
        lines = [
            f"Fama-French Factor Exposure Report",
            f"Entity: {result.entity_id} ({result.entity_type})",
            f"Lookback: {result.lookback_months} months",
            f"R²: {result.r_squared:.4f}" if result.r_squared else "R²: N/A",
            "",
            "Factor Exposures:",
        ]

        if result.factor_exposures:
            for exposure in result.factor_exposures:
                lines.append(
                    f"  {exposure.factor_name}: {exposure.coefficient:.4f} "
                    f"(t-stat: {exposure.t_statistic:.2f}, p-value: {exposure.p_value:.4f})"
                    if exposure.t_statistic is not None
                    else f"  {exposure.factor_name}: {exposure.coefficient:.4f}"
                )
        else:
            lines.append("  No factor exposures calculated.")

        if result.intercept is not None:
            lines.append(f"")
            lines.append(f"Intercept (Alpha): {result.intercept:.4f}")

        return "\n".join(lines)
