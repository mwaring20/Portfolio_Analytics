"""
Phase 12 — Suitability Flagging.

Per architecture doc:
  - Client risk tolerance vs. portfolio risk (volatility, drawdown)
  - Sector/asset class restrictions (ESG, prohibited sectors)
  - Liquidity constraints (minimum cash, max illiquid percentage)
  - Compliance recordkeeping: flag and document violations
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from pydantic import BaseModel, Field


class RiskToleranceLevel(Enum):
    """Client risk tolerance levels."""
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"
    SPECULATIVE = "Speculative"


class SuitabilityFlagType(Enum):
    """Types of suitability violations."""
    RISK_TOLERANCE = "Risk Tolerance"
    SECTOR_RESTRICTION = "Sector Restriction"
    ASSET_CLASS_RESTRICTION = "Asset Class Restriction"
    LIQUIDITY_CONSTRAINT = "Liquidity Constraint"
    CONCENTRATION_LIMIT = "Concentration Limit"


@dataclass
class SuitabilityFlag:
    """A suitability violation flag."""
    flag_type: SuitabilityFlagType
    description: str
    actual_value: Optional[Decimal] = None
    limit_value: Optional[Decimal] = None
    severity: str = "Warning"  # "Warning" or "Violation"


class ClientRiskProfile(BaseModel):
    """
    Client risk profile and constraints.

    Defines the client's risk tolerance and any investment restrictions.
    """
    client_id: str
    risk_tolerance: RiskToleranceLevel
    max_volatility: Optional[Decimal] = Field(
        default=None, description="Maximum acceptable portfolio volatility"
    )
    max_drawdown: Optional[Decimal] = Field(
        default=None, description="Maximum acceptable drawdown"
    )
    min_sharpe_ratio: Optional[Decimal] = Field(
        default=None, description="Minimum acceptable Sharpe ratio"
    )
    prohibited_sectors: List[str] = Field(
        default_factory=list, description="Sectors client will not invest in"
    )
    prohibited_asset_classes: List[str] = Field(
        default_factory=list, description="Asset classes client will not invest in"
    )
    max_single_position: Optional[Decimal] = Field(
        default=None, description="Maximum percentage for a single position"
    )
    min_cash_percentage: Optional[Decimal] = Field(
        default=None, description="Minimum cash percentage required"
    )
    max_illiquid_percentage: Optional[Decimal] = Field(
        default=None, description="Maximum percentage in illiquid assets"
    )


class SuitabilityEngine:
    """
    Checks portfolio suitability against client risk profile and constraints.

    Flags violations for compliance recordkeeping.
    """

    def __init__(self):
        """Initialize the suitability engine."""
        pass

    def check_suitability(
        self,
        client_profile: ClientRiskProfile,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        risk_metrics: Optional[Dict[str, Decimal]] = None,
        as_of_date: Optional[date] = None,
    ) -> List[SuitabilityFlag]:
        """
        Check portfolio suitability against client profile.

        Args:
            client_profile: Client risk profile and constraints
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            risk_metrics: optional risk metrics (volatility, drawdown, Sharpe)
            as_of_date: the date to analyze

        Returns:
            List of SuitabilityFlag objects for any violations
        """
        if as_of_date is None and not daily_positions_df.empty:
            as_of_date = daily_positions_df["date"].max()

        if as_of_date is None:
            return []

        flags: List[SuitabilityFlag] = []

        # Check risk tolerance
        if risk_metrics:
            flags.extend(self._check_risk_tolerance(client_profile, risk_metrics))

        # Check sector restrictions
        flags.extend(
            self._check_sector_restrictions(
                client_profile, daily_positions_df, security_master, as_of_date
            )
        )

        # Check asset class restrictions
        flags.extend(
            self._check_asset_class_restrictions(
                client_profile, daily_positions_df, security_master, as_of_date
            )
        )

        # Check concentration limits
        flags.extend(
            self._check_concentration_limits(
                client_profile, daily_positions_df, as_of_date
            )
        )

        # Check liquidity constraints
        flags.extend(
            self._check_liquidity_constraints(
                client_profile, daily_positions_df, security_master, as_of_date
            )
        )

        return flags

    def _check_risk_tolerance(
        self,
        client_profile: ClientRiskProfile,
        risk_metrics: Dict[str, Decimal],
    ) -> List[SuitabilityFlag]:
        """Check risk metrics against client tolerance."""
        flags: List[SuitabilityFlag] = []

        # Volatility check
        if client_profile.max_volatility is not None:
            volatility = risk_metrics.get("volatility")
            if volatility and volatility > client_profile.max_volatility:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.RISK_TOLERANCE,
                        description=f"Portfolio volatility ({volatility:.2%}) exceeds client maximum ({client_profile.max_volatility:.2%})",
                        actual_value=volatility,
                        limit_value=client_profile.max_volatility,
                        severity="Violation",
                    )
                )

        # Drawdown check
        if client_profile.max_drawdown is not None:
            max_dd = risk_metrics.get("max_drawdown")
            if max_dd and max_dd < -client_profile.max_drawdown:  # Drawdown is negative
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.RISK_TOLERANCE,
                        description=f"Portfolio drawdown ({max_dd:.2%}) exceeds client maximum ({-client_profile.max_drawdown:.2%})",
                        actual_value=max_dd,
                        limit_value=-client_profile.max_drawdown,
                        severity="Violation",
                    )
                )

        # Sharpe ratio check
        if client_profile.min_sharpe_ratio is not None:
            sharpe = risk_metrics.get("sharpe_ratio")
            if sharpe and sharpe < client_profile.min_sharpe_ratio:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.RISK_TOLERANCE,
                        description=f"Portfolio Sharpe ratio ({sharpe:.2f}) below client minimum ({client_profile.min_sharpe_ratio:.2f})",
                        actual_value=sharpe,
                        limit_value=client_profile.min_sharpe_ratio,
                        severity="Warning",
                    )
                )

        return flags

    def _check_sector_restrictions(
        self,
        client_profile: ClientRiskProfile,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        as_of_date: date,
    ) -> List[SuitabilityFlag]:
        """Check for prohibited sectors."""
        if not client_profile.prohibited_sectors:
            return []

        flags: List[SuitabilityFlag] = []
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        for _, pos in positions.iterrows():
            security_id = pos["security_id"]

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            sector = sec.sector

            if sector in client_profile.prohibited_sectors:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.SECTOR_RESTRICTION,
                        description=f"Holding {security_id} in prohibited sector: {sector}",
                        actual_value=pos["market_value"],
                        limit_value=Decimal("0"),
                        severity="Violation",
                    )
                )

        return flags

    def _check_asset_class_restrictions(
        self,
        client_profile: ClientRiskProfile,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        as_of_date: date,
    ) -> List[SuitabilityFlag]:
        """Check for prohibited asset classes."""
        if not client_profile.prohibited_asset_classes:
            return []

        flags: List[SuitabilityFlag] = []
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        for _, pos in positions.iterrows():
            security_id = pos["security_id"]

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            asset_class = sec.asset_class.value

            if asset_class in client_profile.prohibited_asset_classes:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.ASSET_CLASS_RESTRICTION,
                        description=f"Holding {security_id} in prohibited asset class: {asset_class}",
                        actual_value=pos["market_value"],
                        limit_value=Decimal("0"),
                        severity="Violation",
                    )
                )

        return flags

    def _check_concentration_limits(
        self,
        client_profile: ClientRiskProfile,
        daily_positions_df: pd.DataFrame,
        as_of_date: date,
    ) -> List[SuitabilityFlag]:
        """Check single position concentration limits."""
        if client_profile.max_single_position is None:
            return []

        flags: List[SuitabilityFlag] = []
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return []

        for _, pos in positions.iterrows():
            weight = pos["market_value"] / total_value

            if weight > client_profile.max_single_position:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.CONCENTRATION_LIMIT,
                        description=f"Position {pos['security_id']} concentration ({weight:.2%}) exceeds limit ({client_profile.max_single_position:.2%})",
                        actual_value=Decimal(str(weight)),
                        limit_value=client_profile.max_single_position,
                        severity="Warning",
                    )
                )

        return flags

    def _check_liquidity_constraints(
        self,
        client_profile: ClientRiskProfile,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        as_of_date: date,
    ) -> List[SuitabilityFlag]:
        """Check liquidity constraints (min cash, max illiquid)."""
        flags: List[SuitabilityFlag] = []
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        total_value = positions["market_value"].sum()

        if total_value == 0:
            return []

        # Check minimum cash
        if client_profile.min_cash_percentage is not None:
            cash_value = Decimal("0")
            for _, pos in positions.iterrows():
                security_id = pos["security_id"]
                if security_id not in security_master:
                    continue

                sec = security_master[security_id]
                # Assume cash-like if asset_class is Cash or similar
                if hasattr(sec, "asset_class") and sec.asset_class.value == "Cash":
                    cash_value += pos["market_value"]

            cash_percentage = cash_value / total_value

            if cash_percentage < client_profile.min_cash_percentage:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.LIQUIDITY_CONSTRAINT,
                        description=f"Cash percentage ({cash_percentage:.2%}) below minimum ({client_profile.min_cash_percentage:.2%})",
                        actual_value=cash_percentage,
                        limit_value=client_profile.min_cash_percentage,
                        severity="Warning",
                    )
                )

        # Check maximum illiquid percentage
        if client_profile.max_illiquid_percentage is not None:
            illiquid_value = Decimal("0")
            for _, pos in positions.iterrows():
                security_id = pos["security_id"]
                if security_id not in security_master:
                    continue

                sec = security_master[security_id]
                # Assume illiquid if asset_class is Private Equity, Real Estate, etc.
                if hasattr(sec, "asset_class") and sec.asset_class.value in [
                    "Private Equity",
                    "Real Estate",
                    "Alternatives",
                ]:
                    illiquid_value += pos["market_value"]

            illiquid_percentage = illiquid_value / total_value

            if illiquid_percentage > client_profile.max_illiquid_percentage:
                flags.append(
                    SuitabilityFlag(
                        flag_type=SuitabilityFlagType.LIQUIDITY_CONSTRAINT,
                        description=f"Illiquid percentage ({illiquid_percentage:.2%}) exceeds maximum ({client_profile.max_illiquid_percentage:.2%})",
                        actual_value=illiquid_percentage,
                        limit_value=client_profile.max_illiquid_percentage,
                        severity="Warning",
                    )
                )

        return flags

    def generate_suitability_report(
        self,
        flags: List[SuitabilityFlag],
    ) -> str:
        """
        Generate a human-readable suitability report.

        Args:
            flags: List of SuitabilityFlag objects

        Returns:
            Formatted report string
        """
        if not flags:
            return "No suitability violations detected."

        lines = ["Suitability Violations:"]
        lines.append("")

        # Group by severity
        violations = [f for f in flags if f.severity == "Violation"]
        warnings = [f for f in flags if f.severity == "Warning"]

        if violations:
            lines.append("VIOLATIONS:")
            for flag in violations:
                lines.append(f"  - {flag.description}")
            lines.append("")

        if warnings:
            lines.append("WARNINGS:")
            for flag in warnings:
                lines.append(f"  - {flag.description}")

        return "\n".join(lines)
