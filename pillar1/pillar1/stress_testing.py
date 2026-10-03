"""
Phase 14 — Stress Testing.

Per architecture doc:
  - Scenario analysis: what-if scenarios (market crash, rate shock, sector shock)
  - Historical shock simulation: apply historical market shocks to current portfolio
  - Uses correlation matrix from Phase 10 and risk contribution from Phase 13
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from pydantic import BaseModel, Field


class ShockType(Enum):
    """Types of market shocks."""
    MARKET_CRASH = "Market Crash"
    RATE_SHOCK = "Rate Shock"
    SECTOR_SHOCK = "Sector Shock"
    CURRENCY_SHOCK = "Currency Shock"
    COMMODITY_SHOCK = "Commodity Shock"


@dataclass
class ShockScenario:
    """A stress test shock scenario."""
    scenario_id: str
    name: str
    description: str
    shock_type: ShockType
    shock_magnitude: Decimal  # Percentage change (e.g., -0.30 for -30%)
    affected_sectors: Optional[List[str]] = None  # For sector-specific shocks
    affected_asset_classes: Optional[List[str]] = None  # For asset class-specific shocks


class StressTestResult(BaseModel):
    """Result of a stress test scenario."""
    scenario_id: str
    scenario_name: str
    portfolio_value_before: Decimal
    portfolio_value_after: Decimal
    portfolio_value_change: Decimal
    portfolio_value_change_pct: Decimal
    position_impacts: Dict[str, Decimal] = Field(
        default_factory=dict, description="Security ID to value change"
    )
    worst_performing_position: Optional[str] = None
    best_performing_position: Optional[str] = None


class StressTestEngine:
    """
    Performs stress testing on portfolios using scenario analysis.

    Applies hypothetical shocks to current positions to estimate impact.
    """

    def __init__(self):
        """Initialize the stress test engine."""
        # Predefined standard scenarios
        self.standard_scenarios = self._get_standard_scenarios()

    def _get_standard_scenarios(self) -> List[ShockScenario]:
        """Get standard stress test scenarios."""
        return [
            ShockScenario(
                scenario_id="2008_CRASH",
                name="2008 Financial Crisis",
                description="Equity markets decline 30%, bonds rally 10%",
                shock_type=ShockType.MARKET_CRASH,
                shock_magnitude=Decimal("-0.30"),
                affected_asset_classes=["Equity"],
            ),
            ShockScenario(
                scenario_id="RATE_HIKE",
                name="Rapid Rate Hike",
                description="Interest rates rise 2%, bonds decline 10%",
                shock_type=ShockType.RATE_SHOCK,
                shock_magnitude=Decimal("-0.10"),
                affected_asset_classes=["Fixed Income"],
            ),
            ShockScenario(
                scenario_id="TECH_CRASH",
                name="Tech Sector Crash",
                description="Technology sector declines 40%",
                shock_type=ShockType.SECTOR_SHOCK,
                shock_magnitude=Decimal("-0.40"),
                affected_sectors=["Technology"],
            ),
            ShockScenario(
                scenario_id="COVID_CRASH",
                name="COVID-19 Crash",
                description="Equity markets decline 35%, bonds rally 5%",
                shock_type=ShockType.MARKET_CRASH,
                shock_magnitude=Decimal("-0.35"),
                affected_asset_classes=["Equity"],
            ),
            ShockScenario(
                scenario_id="INFLATION_SPIKE",
                name="Inflation Spike",
                description="Real assets up 20%, bonds down 15%",
                shock_type=ShockType.COMMODITY_SHOCK,
                shock_magnitude=Decimal("0.20"),
                affected_asset_classes=["Real Estate", "Commodities"],
            ),
        ]

    def run_scenario(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        scenario: ShockScenario,
        as_of_date: date,
    ) -> StressTestResult:
        """
        Run a single stress test scenario.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            scenario: shock scenario to apply
            as_of_date: the date to analyze

        Returns:
            StressTestResult with scenario impact
        """
        positions = daily_positions_df[daily_positions_df["date"] == as_of_date].copy()

        if positions.empty:
            return StressTestResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                portfolio_value_before=Decimal("0"),
                portfolio_value_after=Decimal("0"),
                portfolio_value_change=Decimal("0"),
                portfolio_value_change_pct=Decimal("0"),
            )

        portfolio_value_before = positions["market_value"].sum()
        position_impacts: Dict[str, Decimal] = {}

        worst_change = Decimal("0")
        best_change = Decimal("0")
        worst_position = None
        best_position = None

        for _, pos in positions.iterrows():
            security_id = pos["security_id"]
            current_value = pos["market_value"]

            # Determine if this position is affected by the scenario
            shock_pct = self._get_shock_for_position(
                security_id, security_master, scenario
            )

            # Calculate new value
            value_change = current_value * shock_pct
            new_value = current_value + value_change

            position_impacts[security_id] = value_change

            # Track best/worst performers
            if value_change < worst_change:
                worst_change = value_change
                worst_position = security_id
            if value_change > best_change:
                best_change = value_change
                best_position = security_id

        portfolio_value_after = portfolio_value_before + sum(position_impacts.values())
        portfolio_value_change = portfolio_value_after - portfolio_value_before
        portfolio_value_change_pct = (
            portfolio_value_change / portfolio_value_before
            if portfolio_value_before > 0
            else Decimal("0")
        )

        return StressTestResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            portfolio_value_before=portfolio_value_before,
            portfolio_value_after=portfolio_value_after,
            portfolio_value_change=portfolio_value_change,
            portfolio_value_change_pct=portfolio_value_change_pct,
            position_impacts=position_impacts,
            worst_performing_position=worst_position,
            best_performing_position=best_position,
        )

    def _get_shock_for_position(
        self,
        security_id: str,
        security_master: Dict[str, any],
        scenario: ShockScenario,
    ) -> Decimal:
        """
        Determine shock percentage for a specific position.

        Args:
            security_id: security identifier
            security_master: mapping of security_id to SecurityMaster
            scenario: shock scenario

        Returns:
            Shock percentage to apply
        """
        if security_id not in security_master:
            # Unknown security - assume neutral
            return Decimal("0")

        sec = security_master[security_id]
        asset_class = sec.asset_class.value if hasattr(sec, "asset_class") else "Unknown"
        sector = sec.sector if hasattr(sec, "sector") else None

        # Check if affected by asset class
        if scenario.affected_asset_classes:
            if asset_class in scenario.affected_asset_classes:
                return scenario.shock_magnitude

        # Check if affected by sector
        if scenario.affected_sectors and sector:
            if sector in scenario.affected_sectors:
                return scenario.shock_magnitude

        # Not affected - assume neutral (or small market-wide effect)
        if scenario.shock_type == ShockType.MARKET_CRASH:
            # Market crash affects everything, but less for non-affected
            return scenario.shock_magnitude * Decimal("0.3")  # 30% of main shock
        elif scenario.shock_type == ShockType.RATE_SHOCK:
            # Rate shock affects bonds more, equities less
            if asset_class == "Fixed Income":
                return scenario.shock_magnitude
            else:
                return scenario.shock_magnitude * Decimal("0.2")

        return Decimal("0")

    def run_all_standard_scenarios(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        as_of_date: date,
    ) -> List[StressTestResult]:
        """
        Run all standard stress test scenarios.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            as_of_date: the date to analyze

        Returns:
            List of StressTestResult for each scenario
        """
        results = []

        for scenario in self.standard_scenarios:
            result = self.run_scenario(
                daily_positions_df, security_master, scenario, as_of_date
            )
            results.append(result)

        return results

    def run_custom_scenario(
        self,
        daily_positions_df: pd.DataFrame,
        security_master: Dict[str, any],
        scenario_id: str,
        scenario_name: str,
        shock_magnitude: Decimal,
        affected_sectors: Optional[List[str]] = None,
        affected_asset_classes: Optional[List[str]] = None,
        as_of_date: Optional[date] = None,
    ) -> StressTestResult:
        """
        Run a custom stress test scenario.

        Args:
            daily_positions_df: daily positions from valuation engine
            security_master: mapping of security_id to SecurityMaster
            scenario_id: custom scenario identifier
            scenario_name: custom scenario name
            shock_magnitude: shock percentage
            affected_sectors: sectors affected by shock
            affected_asset_classes: asset classes affected by shock
            as_of_date: the date to analyze

        Returns:
            StressTestResult with scenario impact
        """
        if as_of_date is None and not daily_positions_df.empty:
            as_of_date = daily_positions_df["date"].max()

        if as_of_date is None:
            as_of_date = date.today()

        scenario = ShockScenario(
            scenario_id=scenario_id,
            name=scenario_name,
            description="Custom scenario",
            shock_type=ShockType.MARKET_CRASH,  # Default
            shock_magnitude=shock_magnitude,
            affected_sectors=affected_sectors,
            affected_asset_classes=affected_asset_classes,
        )

        return self.run_scenario(daily_positions_df, security_master, scenario, as_of_date)

    def generate_stress_test_report(
        self,
        results: List[StressTestResult],
    ) -> str:
        """
        Generate a human-readable stress test report.

        Args:
            results: List of StressTestResult objects

        Returns:
            Formatted report string
        """
        if not results:
            return "No stress test results available."

        lines = ["Stress Test Results"]
        lines.append("=" * 50)
        lines.append("")

        for result in results:
            lines.append(f"Scenario: {result.scenario_name}")
            lines.append(f"Portfolio Value Before: ${result.portfolio_value_before:,.2f}")
            lines.append(f"Portfolio Value After: ${result.portfolio_value_after:,.2f}")
            lines.append(f"Change: ${result.portfolio_value_change:,.2f} ({result.portfolio_value_change_pct:.2%})")
            lines.append("")

            if result.worst_performing_position:
                lines.append(f"Worst Position: {result.worst_performing_position}")
            if result.best_performing_position:
                lines.append(f"Best Position: {result.best_performing_position}")
            lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines)
