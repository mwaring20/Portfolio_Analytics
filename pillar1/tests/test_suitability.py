"""
Tests for Phase 12 Suitability Flagging.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.suitability import (
    ClientRiskProfile,
    RiskToleranceLevel,
    SuitabilityEngine,
    SuitabilityFlag,
    SuitabilityFlagType,
)


def test_check_risk_tolerance_volatility():
    """Test volatility check against client tolerance."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        max_volatility=Decimal("0.15"),  # 15%
    )

    risk_metrics = {"volatility": Decimal("0.20")}  # 20% - exceeds limit

    engine = SuitabilityEngine()
    flags = engine._check_risk_tolerance(client_profile, risk_metrics)

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.RISK_TOLERANCE
    assert flags[0].severity == "Violation"


def test_check_risk_tolerance_drawdown():
    """Test drawdown check against client tolerance."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.CONSERVATIVE,
        max_drawdown=Decimal("0.10"),  # 10%
    )

    risk_metrics = {"max_drawdown": Decimal("-0.15")}  # -15% - exceeds limit

    engine = SuitabilityEngine()
    flags = engine._check_risk_tolerance(client_profile, risk_metrics)

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.RISK_TOLERANCE
    assert flags[0].severity == "Violation"


def test_check_risk_tolerance_sharpe():
    """Test Sharpe ratio check against client tolerance."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.AGGRESSIVE,
        min_sharpe_ratio=Decimal("1.0"),
    )

    risk_metrics = {"sharpe_ratio": Decimal("0.5")}  # Below minimum

    engine = SuitabilityEngine()
    flags = engine._check_risk_tolerance(client_profile, risk_metrics)

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.RISK_TOLERANCE
    assert flags[0].severity == "Warning"


def test_check_risk_tolerance_no_violations():
    """Test that compliant portfolio has no risk tolerance flags."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        max_volatility=Decimal("0.15"),
        max_drawdown=Decimal("0.10"),
        min_sharpe_ratio=Decimal("0.5"),
    )

    risk_metrics = {
        "volatility": Decimal("0.12"),
        "max_drawdown": Decimal("-0.05"),
        "sharpe_ratio": Decimal("1.5"),
    }

    engine = SuitabilityEngine()
    flags = engine._check_risk_tolerance(client_profile, risk_metrics)

    assert len(flags) == 0


def test_check_sector_restrictions():
    """Test sector restriction checks."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        prohibited_sectors=["Tobacco", "Weapons"],
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "MO",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "MO": type("Security", (), {"sector": "Tobacco", "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = SuitabilityEngine()
    flags = engine._check_sector_restrictions(
        client_profile, daily_positions, security_master, date(2024, 1, 31)
    )

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.SECTOR_RESTRICTION
    assert flags[0].severity == "Violation"


def test_check_sector_restrictions_no_prohibitions():
    """Test that no prohibited sectors means no flags."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        prohibited_sectors=[],
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = SuitabilityEngine()
    flags = engine._check_sector_restrictions(
        client_profile, daily_positions, security_master, date(2024, 1, 31)
    )

    assert len(flags) == 0


def test_check_asset_class_restrictions():
    """Test asset class restriction checks."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.CONSERVATIVE,
        prohibited_asset_classes=["Cryptocurrency", "Alternatives"],
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "BTC",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("1000.00"),
                "price": Decimal("100.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "BTC": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Cryptocurrency"})})(),
    }

    engine = SuitabilityEngine()
    flags = engine._check_asset_class_restrictions(
        client_profile, daily_positions, security_master, date(2024, 1, 31)
    )

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.ASSET_CLASS_RESTRICTION


def test_check_concentration_limits():
    """Test single position concentration limits."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        max_single_position=Decimal("0.10"),  # 10%
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),  # 50%
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),  # 50%
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = SuitabilityEngine()
    flags = engine._check_concentration_limits(
        client_profile, daily_positions, date(2024, 1, 31)
    )

    assert len(flags) == 2  # Both positions exceed 10%
    assert flags[0].flag_type == SuitabilityFlagType.CONCENTRATION_LIMIT
    assert flags[0].severity == "Warning"


def test_check_concentration_limits_no_limit():
    """Test that no concentration limit means no flags."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.AGGRESSIVE,
        max_single_position=None,
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("10000.00"),
                "price": Decimal("1000.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    engine = SuitabilityEngine()
    flags = engine._check_concentration_limits(
        client_profile, daily_positions, date(2024, 1, 31)
    )

    assert len(flags) == 0


def test_check_liquidity_constraints_min_cash():
    """Test minimum cash constraint."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.CONSERVATIVE,
        min_cash_percentage=Decimal("0.05"),  # 5%
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("10000.00"),
                "price": Decimal("1000.00"),
            }
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = SuitabilityEngine()
    flags = engine._check_liquidity_constraints(
        client_profile, daily_positions, security_master, date(2024, 1, 31)
    )

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.LIQUIDITY_CONSTRAINT


def test_check_liquidity_constraints_max_illiquid():
    """Test maximum illiquid constraint."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        max_illiquid_percentage=Decimal("0.10"),  # 10%
    )

    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "REIT",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),  # 50%
                "price": Decimal("500.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 31),
                "quantity": Decimal("10"),
                "market_value": Decimal("5000.00"),  # 50%
                "price": Decimal("500.00"),
            },
        ],
        columns=["account_id", "security_id", "date", "quantity", "market_value", "price"],
    )

    security_master = {
        "REIT": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Real Estate"})})(),
        "VTI": type("Security", (), {"sector": None, "asset_class": type("AssetClass", (), {"value": "Equity"})})(),
    }

    engine = SuitabilityEngine()
    flags = engine._check_liquidity_constraints(
        client_profile, daily_positions, security_master, date(2024, 1, 31)
    )

    assert len(flags) == 1
    assert flags[0].flag_type == SuitabilityFlagType.LIQUIDITY_CONSTRAINT


def test_check_suitability_comprehensive():
    """Test comprehensive suitability check."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        max_volatility=Decimal("0.15"),
        prohibited_sectors=["Tobacco"],
        max_single_position=Decimal("0.20"),
    )

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

    risk_metrics = {"volatility": Decimal("0.12")}

    engine = SuitabilityEngine()
    flags = engine.check_suitability(
        client_profile, daily_positions, security_master, risk_metrics, date(2024, 1, 31)
    )

    # Should check all constraints
    assert isinstance(flags, list)


def test_generate_suitability_report_no_violations():
    """Test report generation with no violations."""
    engine = SuitabilityEngine()
    report = engine.generate_suitability_report([])

    assert report == "No suitability violations detected."


def test_generate_suitability_report_with_violations():
    """Test report generation with violations."""
    flags = [
        SuitabilityFlag(
            flag_type=SuitabilityFlagType.RISK_TOLERANCE,
            description="Volatility exceeds limit",
            severity="Violation",
        ),
        SuitabilityFlag(
            flag_type=SuitabilityFlagType.CONCENTRATION_LIMIT,
            description="Position concentration exceeds limit",
            severity="Warning",
        ),
    ]

    engine = SuitabilityEngine()
    report = engine.generate_suitability_report(flags)

    assert "VIOLATIONS:" in report
    assert "WARNINGS:" in report
    assert "Volatility exceeds limit" in report


def test_client_risk_profile_model():
    """Test ClientRiskProfile Pydantic model."""
    profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
        max_volatility=Decimal("0.15"),
        prohibited_sectors=["Tobacco"],
    )

    assert profile.client_id == "CLIENT001"
    assert profile.risk_tolerance == RiskToleranceLevel.MODERATE
    assert profile.max_volatility == Decimal("0.15")
    assert "Tobacco" in profile.prohibited_sectors


def test_suitability_flag_dataclass():
    """Test SuitabilityFlag dataclass."""
    flag = SuitabilityFlag(
        flag_type=SuitabilityFlagType.RISK_TOLERANCE,
        description="Test flag",
        actual_value=Decimal("0.20"),
        limit_value=Decimal("0.15"),
        severity="Violation",
    )

    assert flag.flag_type == SuitabilityFlagType.RISK_TOLERANCE
    assert flag.description == "Test flag"
    assert flag.actual_value == Decimal("0.20")
    assert flag.limit_value == Decimal("0.15")
    assert flag.severity == "Violation"


def test_risk_tolerance_level_enum():
    """Test RiskToleranceLevel enum."""
    assert RiskToleranceLevel.CONSERVATIVE.value == "Conservative"
    assert RiskToleranceLevel.MODERATE.value == "Moderate"
    assert RiskToleranceLevel.AGGRESSIVE.value == "Aggressive"
    assert RiskToleranceLevel.SPECULATIVE.value == "Speculative"


def test_check_suitability_auto_date():
    """Test suitability check with auto-detected date."""
    client_profile = ClientRiskProfile(
        client_id="CLIENT001",
        risk_tolerance=RiskToleranceLevel.MODERATE,
    )

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

    engine = SuitabilityEngine()
    flags = engine.check_suitability(
        client_profile, daily_positions, security_master, as_of_date=None
    )

    # Should use max date from positions
    assert isinstance(flags, list)
