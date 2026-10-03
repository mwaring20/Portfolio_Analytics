"""
Tests for Phase 14 Stress Testing.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.stress_testing import (
    ShockScenario,
    ShockType,
    StressTestEngine,
    StressTestResult,
)


def test_run_scenario_market_crash():
    """Test market crash scenario."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
        "BND": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Fixed Income"})})(),
    }

    scenario = ShockScenario(
        scenario_id="CRASH",
        name="Market Crash",
        description="30% market decline",
        shock_type=ShockType.MARKET_CRASH,
        shock_magnitude=Decimal("-0.30"),
        affected_asset_classes=["Equity"],
    )

    engine = StressTestEngine()
    result = engine.run_scenario(daily_positions, security_master, scenario, date(2024, 1, 31))

    assert result.scenario_id == "CRASH"
    assert result.portfolio_value_before == Decimal("10000.00")
    assert result.portfolio_value_after < result.portfolio_value_before
    assert result.portfolio_value_change < 0


def test_run_scenario_empty_positions():
    """Test scenario with empty positions."""
    daily_positions = pd.DataFrame(
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"]
    )

    security_master = {}

    scenario = ShockScenario(
        scenario_id="CRASH",
        name="Market Crash",
        description="30% market decline",
        shock_type=ShockType.MARKET_CRASH,
        shock_magnitude=Decimal("-0.30"),
    )

    engine = StressTestEngine()
    result = engine.run_scenario(daily_positions, security_master, scenario, date(2024, 1, 31))

    assert result.portfolio_value_before == Decimal("0")
    assert result.portfolio_value_after == Decimal("0")


def test_run_scenario_sector_shock():
    """Test sector-specific shock."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "AAPL": type("Security", (), {"sector": "Technology", "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
        "BND": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Fixed Income"})})(),
    }

    scenario = ShockScenario(
        scenario_id="TECH_CRASH",
        name="Tech Crash",
        description="40% tech decline",
        shock_type=ShockType.SECTOR_SHOCK,
        shock_magnitude=Decimal("-0.40"),
        affected_sectors=["Technology"],
    )

    engine = StressTestEngine()
    result = engine.run_scenario(daily_positions, security_master, scenario, date(2024, 1, 31))

    # AAPL should be affected, BND should not
    assert result.position_impacts["AAPL"] < 0
    assert result.position_impacts["BND"] == 0  # Not affected


def test_get_shock_for_position_affected_asset_class():
    """Test shock determination for affected asset class."""
    security_master = {
        "BND": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Fixed Income"})})(),
    }

    scenario = ShockScenario(
        scenario_id="RATE_HIKE",
        name="Rate Hike",
        description="Bond decline",
        shock_type=ShockType.RATE_SHOCK,
        shock_magnitude=Decimal("-0.10"),
        affected_asset_classes=["Fixed Income"],
    )

    engine = StressTestEngine()
    shock = engine._get_shock_for_position("BND", security_master, scenario)

    assert shock == Decimal("-0.10")


def test_get_shock_for_position_not_affected():
    """Test shock determination for non-affected position."""
    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    scenario = ShockScenario(
        scenario_id="RATE_HIKE",
        name="Rate Hike",
        description="Bond decline",
        shock_type=ShockType.RATE_SHOCK,
        shock_magnitude=Decimal("-0.10"),
        affected_asset_classes=["Fixed Income"],
    )

    engine = StressTestEngine()
    shock = engine._get_shock_for_position("VTI", security_master, scenario)

    # Equity not directly affected by rate shock in this simplified model
    assert shock == Decimal("0")


def test_get_shock_for_position_unknown_security():
    """Test shock determination for unknown security."""
    security_master = {}

    scenario = ShockScenario(
        scenario_id="CRASH",
        name="Crash",
        description="Market decline",
        shock_type=ShockType.MARKET_CRASH,
        shock_magnitude=Decimal("-0.30"),
    )

    engine = StressTestEngine()
    shock = engine._get_shock_for_position("UNKNOWN", security_master, scenario)

    assert shock == Decimal("0")


def test_run_all_standard_scenarios():
    """Test running all standard scenarios."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = StressTestEngine()
    results = engine.run_all_standard_scenarios(
        daily_positions, security_master, date(2024, 1, 31)
    )

    # Should have 5 standard scenarios
    assert len(results) == 5
    for result in results:
        assert isinstance(result, StressTestResult)


def test_run_custom_scenario():
    """Test running a custom scenario."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = StressTestEngine()
    result = engine.run_custom_scenario(
        daily_positions,
        security_master,
        scenario_id="CUSTOM",
        scenario_name="Custom Crash",
        shock_magnitude=Decimal("-0.50"),
        as_of_date=date(2024, 1, 31),
    )

    assert result.scenario_id == "CUSTOM"
    assert result.scenario_name == "Custom Crash"
    assert result.portfolio_value_after < result.portfolio_value_before


def test_run_custom_scenario_auto_date():
    """Test custom scenario with auto-detected date."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = StressTestEngine()
    result = engine.run_custom_scenario(
        daily_positions,
        security_master,
        scenario_id="CUSTOM",
        scenario_name="Custom",
        shock_magnitude=Decimal("-0.20"),
        # No as_of_date specified
    )

    assert result.portfolio_value_before == Decimal("5000.00")


def test_generate_stress_test_report():
    """Test stress test report generation."""
    results = [
        StressTestResult(
            scenario_id="CRASH",
            scenario_name="Market Crash",
            portfolio_value_before=Decimal("10000.00"),
            portfolio_value_after=Decimal("7000.00"),
            portfolio_value_change=Decimal("-3000.00"),
            portfolio_value_change_pct=Decimal("-0.30"),
            worst_performing_position="VTI",
            best_performing_position="BND",
        )
    ]

    engine = StressTestEngine()
    report = engine.generate_stress_test_report(results)

    assert "Stress Test Results" in report
    assert "Market Crash" in report
    assert "$10,000.00" in report
    assert "$7,000.00" in report


def test_generate_stress_test_report_empty():
    """Test report generation with no results."""
    engine = StressTestEngine()
    report = engine.generate_stress_test_report([])

    assert report == "No stress test results available."


def test_shock_scenario_dataclass():
    """Test ShockScenario dataclass."""
    scenario = ShockScenario(
        scenario_id="TEST",
        name="Test Scenario",
        description="Test",
        shock_type=ShockType.MARKET_CRASH,
        shock_magnitude=Decimal("-0.30"),
        affected_sectors=["Technology"],
        affected_asset_classes=["Equity"],
    )

    assert scenario.scenario_id == "TEST"
    assert scenario.name == "Test Scenario"
    assert scenario.shock_type == ShockType.MARKET_CRASH
    assert scenario.shock_magnitude == Decimal("-0.30")


def test_stress_test_result_model():
    """Test StressTestResult Pydantic model."""
    result = StressTestResult(
        scenario_id="TEST",
        scenario_name="Test",
        portfolio_value_before=Decimal("10000.00"),
        portfolio_value_after=Decimal("7000.00"),
        portfolio_value_change=Decimal("-3000.00"),
        portfolio_value_change_pct=Decimal("-0.30"),
    )

    assert result.scenario_id == "TEST"
    assert result.portfolio_value_before == Decimal("10000.00")
    assert result.portfolio_value_change_pct == Decimal("-0.30")


def test_shock_type_enum():
    """Test ShockType enum."""
    assert ShockType.MARKET_CRASH.value == "Market Crash"
    assert ShockType.RATE_SHOCK.value == "Rate Shock"
    assert ShockType.SECTOR_SHOCK.value == "Sector Shock"


def test_standard_scenarios_exist():
    """Test that standard scenarios are predefined."""
    engine = StressTestEngine()
    assert len(engine.standard_scenarios) == 5

    scenario_names = [s.name for s in engine.standard_scenarios]
    assert "2008 Financial Crisis" in scenario_names
    assert "Rapid Rate Hike" in scenario_names
    assert "Tech Sector Crash" in scenario_names
    assert "COVID-19 Crash" in scenario_names
    assert "Inflation Spike" in scenario_names


def test_position_impacts_tracking():
    """Test that position impacts are tracked correctly."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
        "BND": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Fixed Income"})})(),
    }

    scenario = ShockScenario(
        scenario_id="CRASH",
        name="Crash",
        description="Market decline",
        shock_type=ShockType.MARKET_CRASH,
        shock_magnitude=Decimal("-0.30"),
        affected_asset_classes=["Equity"],
    )

    engine = StressTestEngine()
    result = engine.run_scenario(daily_positions, security_master, scenario, date(2024, 1, 31))

    assert "VTI" in result.position_impacts
    assert "BND" in result.position_impacts
    assert result.position_impacts["VTI"] < 0  # VTI affected
