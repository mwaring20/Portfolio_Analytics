"""
Phase 7 — Portfolio Construction Analysis.

Per architecture doc:
  - Allocation breakdown: group daily_positions market value by asset_class/sector/geography
  - Concentration risk flags: configurable thresholds (e.g., single position >10%, sector >25%)
  - Fee drag: pull expense ratios from SecurityMaster and advisor fees from account config
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd

from .security_master import SecurityMaster


@dataclass
class ConcentrationFlag:
    """A concentration risk flag."""
    account_id: str
    as_of_date: date
    flag_type: str  # "single_position", "single_sector", "single_geography"
    identifier: str  # security_id, sector name, or geography name
    current_value: Decimal
    threshold_pct: Decimal
    actual_pct: Decimal


@dataclass
class AllocationBreakdown:
    """Allocation breakdown for a specific dimension."""
    dimension: str  # "asset_class", "sector", "geography"
    as_of_date: date
    allocations: Dict[str, Decimal]  # category -> percentage
    total_value: Decimal


@dataclass
class FeeDragAnalysis:
    """Fee drag analysis results."""
    account_id: str
    as_of_date: date
    total_expense_ratio: Decimal  # Weighted average of fund expense ratios
    annual_fee_dollar: Decimal  # Total annual fees in dollars
    annual_fee_pct: Decimal  # Annual fees as percentage of portfolio value
    advisor_fee_pct: Optional[Decimal] = None  # Advisor fee if configured
    total_fee_pct: Optional[Decimal] = None  # Total including advisor fees


class PortfolioAnalyzer:
    """
    Analyzes portfolio construction: allocation, concentration risk, and fee drag.

    Depends on daily_positions + SecurityMaster taxonomy (shared across pillars).
    """

    def __init__(
        self,
        single_position_threshold: Decimal = Decimal("0.10"),  # 10%
        single_sector_threshold: Decimal = Decimal("0.25"),  # 25%
        single_geography_threshold: Decimal = Decimal("0.50"),  # 50%
    ):
        """
        Initialize the portfolio analyzer.

        single_position_threshold: max allowed percentage for a single position
        single_sector_threshold: max allowed percentage for a single sector
        single_geography_threshold: max allowed percentage for a single geography
        """
        self.single_position_threshold = single_position_threshold
        self.single_sector_threshold = single_sector_threshold
        self.single_geography_threshold = single_geography_threshold

    def compute_allocation(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        as_of_date: date,
        dimension: str = "asset_class",
    ) -> AllocationBreakdown:
        """
        Compute allocation breakdown for a specific dimension.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            as_of_date: the date to analyze
            dimension: one of "asset_class", "sector", "geography"

        Returns:
            AllocationBreakdown with category percentages
        """
        # Filter to as_of_date
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return AllocationBreakdown(
                dimension=dimension,
                as_of_date=as_of_date,
                allocations={},
                total_value=Decimal("0"),
            )

        # Join with security master to get classification
        allocations: Dict[str, Decimal] = {}
        total_value = Decimal("0")

        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]
            total_value += market_value

            # Get classification from security master
            if security_id not in security_master:
                # Unclassified security - skip or mark as unknown
                continue

            sec = security_master[security_id]

            if dimension == "asset_class":
                category = sec.asset_class.value
            elif dimension == "sector":
                category = sec.sector or "Unknown"
            elif dimension == "geography":
                category = sec.geography or "Unknown"
            else:
                raise ValueError(f"Unknown dimension: {dimension}")

            if category not in allocations:
                allocations[category] = Decimal("0")
            allocations[category] += market_value

        # Convert to percentages
        if total_value > 0:
            allocations = {
                cat: (value / total_value) * Decimal("100")
                for cat, value in allocations.items()
            }

        return AllocationBreakdown(
            dimension=dimension,
            as_of_date=as_of_date,
            allocations=allocations,
            total_value=total_value,
        )

    def detect_concentration_risks(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        as_of_date: date,
    ) -> List[ConcentrationFlag]:
        """
        Detect concentration risks exceeding thresholds.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            as_of_date: the date to analyze

        Returns:
            List of ConcentrationFlag objects for risks exceeding thresholds
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return []

        total_value = positions["market_value"].sum()
        if total_value == 0:
            return []

        flags: List[ConcentrationFlag] = []

        # Check single position concentration
        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]
            actual_pct = (market_value / total_value) * Decimal("100")

            if actual_pct > self.single_position_threshold * Decimal("100"):
                flags.append(
                    ConcentrationFlag(
                        account_id=pos["account_id"],
                        as_of_date=as_of_date,
                        flag_type="single_position",
                        identifier=security_id,
                        current_value=market_value,
                        threshold_pct=self.single_position_threshold * Decimal("100"),
                        actual_pct=actual_pct,
                    )
                )

        # Check sector concentration
        sector_values: Dict[str, Decimal] = {}
        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            sector = sec.sector or "Unknown"

            if sector not in sector_values:
                sector_values[sector] = Decimal("0")
            sector_values[sector] += market_value

        for sector, value in sector_values.items():
            actual_pct = (value / total_value) * Decimal("100")
            if actual_pct > self.single_sector_threshold * Decimal("100"):
                flags.append(
                    ConcentrationFlag(
                        account_id=positions.iloc[0]["account_id"],
                        as_of_date=as_of_date,
                        flag_type="single_sector",
                        identifier=sector,
                        current_value=value,
                        threshold_pct=self.single_sector_threshold * Decimal("100"),
                        actual_pct=actual_pct,
                    )
                )

        # Check geography concentration
        geo_values: Dict[str, Decimal] = {}
        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            geography = sec.geography or "Unknown"

            if geography not in geo_values:
                geo_values[geography] = Decimal("0")
            geo_values[geography] += market_value

        for geography, value in geo_values.items():
            actual_pct = (value / total_value) * Decimal("100")
            if actual_pct > self.single_geography_threshold * Decimal("100"):
                flags.append(
                    ConcentrationFlag(
                        account_id=positions.iloc[0]["account_id"],
                        as_of_date=as_of_date,
                        flag_type="single_geography",
                        identifier=geography,
                        current_value=value,
                        threshold_pct=self.single_geography_threshold * Decimal("100"),
                        actual_pct=actual_pct,
                    )
                )

        return flags

    def analyze_fee_drag(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        as_of_date: date,
        advisor_fee_pct: Optional[Decimal] = None,
    ) -> FeeDragAnalysis:
        """
        Analyze fee drag from fund expense ratios and advisor fees.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            as_of_date: the date to analyze
            advisor_fee_pct: annual advisor fee as percentage (e.g., 1.0 for 1%)

        Returns:
            FeeDragAnalysis with expense ratio and fee calculations
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return FeeDragAnalysis(
                account_id="",
                as_of_date=as_of_date,
                total_expense_ratio=Decimal("0"),
                annual_fee_dollar=Decimal("0"),
                annual_fee_pct=Decimal("0"),
                advisor_fee_pct=advisor_fee_pct,
                total_fee_pct=advisor_fee_pct if advisor_fee_pct else Decimal("0"),
            )

        total_value = positions["market_value"].sum()
        if total_value == 0:
            return FeeDragAnalysis(
                account_id=positions.iloc[0]["account_id"],
                as_of_date=as_of_date,
                total_expense_ratio=Decimal("0"),
                annual_fee_dollar=Decimal("0"),
                annual_fee_pct=Decimal("0"),
                advisor_fee_pct=advisor_fee_pct,
                total_fee_pct=advisor_fee_pct if advisor_fee_pct else Decimal("0"),
            )

        # Calculate weighted average expense ratio
        weighted_expense_sum = Decimal("0")
        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            market_value = pos["market_value"]
            weight = market_value / total_value

            if security_id not in security_master:
                continue

            sec = security_master[security_id]
            expense_ratio = sec.expense_ratio or Decimal("0")

            weighted_expense_sum += weight * expense_ratio

        total_expense_ratio = weighted_expense_sum
        annual_fund_fees = total_value * total_expense_ratio

        # Add advisor fees if configured
        annual_advisor_fees = Decimal("0")
        if advisor_fee_pct:
            annual_advisor_fees = total_value * (advisor_fee_pct / Decimal("100"))

        total_annual_fees = annual_fund_fees + annual_advisor_fees
        annual_fee_pct = (total_annual_fees / total_value) * Decimal("100") if total_value > 0 else Decimal("0")

        total_fee_pct = total_expense_ratio * Decimal("100")
        if advisor_fee_pct:
            total_fee_pct += advisor_fee_pct

        return FeeDragAnalysis(
            account_id=positions.iloc[0]["account_id"],
            as_of_date=as_of_date,
            total_expense_ratio=total_expense_ratio,
            annual_fee_dollar=total_annual_fees,
            annual_fee_pct=annual_fee_pct,
            advisor_fee_pct=advisor_fee_pct,
            total_fee_pct=total_fee_pct,
        )

    def get_portfolio_summary(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, SecurityMaster],
        as_of_date: date,
        advisor_fee_pct: Optional[Decimal] = None,
    ) -> dict:
        """
        Get a comprehensive portfolio summary including allocation, concentration, and fees.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            as_of_date: the date to analyze
            advisor_fee_pct: annual advisor fee as percentage

        Returns:
            dict with all analysis results
        """
        asset_allocation = self.compute_allocation(
            daily_positions_df, security_master, as_of_date, "asset_class"
        )
        sector_allocation = self.compute_allocation(
            daily_positions_df, security_master, as_of_date, "sector"
        )
        geography_allocation = self.compute_allocation(
            daily_positions_df, security_master, as_of_date, "geography"
        )

        concentration_flags = self.detect_concentration_risks(
            daily_positions_df, security_master, as_of_date
        )

        fee_analysis = self.analyze_fee_drag(
            daily_positions_df, security_master, as_of_date, advisor_fee_pct
        )

        return {
            "as_of_date": as_of_date,
            "asset_allocation": asset_allocation,
            "sector_allocation": sector_allocation,
            "geography_allocation": geography_allocation,
            "concentration_flags": concentration_flags,
            "fee_analysis": fee_analysis,
            "total_positions": len(daily_positions_df[daily_positions_df["date"] == as_of_date]),
            "total_value": asset_allocation.total_value,
        }
