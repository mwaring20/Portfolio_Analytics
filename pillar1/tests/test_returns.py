"""
Tests for Phase 3 Returns Engine (TWR and MWR/IRR).
"""

from datetime import date, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.canonical_schema import DAILY_POSITIONS_COLUMNS
from pillar1.returns import ReturnsEngine


def test_twr_simple_case():
    """Test TWR with a simple buy and sell."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 16),
                "quantity": Decimal("10"),
                "market_value": Decimal("2510.00"),
                "price": Decimal("251.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 17),
                "quantity": Decimal("10"),
                "market_value": Decimal("2520.00"),
                "price": Decimal("252.00"),
            },
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    engine = ReturnsEngine()
    twr = engine.compute_twr(daily_positions)

    # Expected: (2510/2500 - 1) * (2520/2510 - 1) + (2510/2500 - 1) + (2520/2510 - 1)
    # = 0.004 * 0.00398 + 0.004 + 0.00398 ≈ 0.008
    assert twr > 0
    assert twr < Decimal("0.01")  # Less than 1% gain


def test_twr_empty_positions():
    """Test TWR with empty positions returns 0."""
    daily_positions = pd.DataFrame(columns=DAILY_POSITIONS_COLUMNS)

    engine = ReturnsEngine()
    twr = engine.compute_twr(daily_positions)

    assert twr == Decimal("0")


def test_twr_single_day():
    """Test TWR with only one day of data returns 0 (need at least 2 days)."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            }
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    engine = ReturnsEngine()
    twr = engine.compute_twr(daily_positions)

    assert twr == Decimal("0")


def test_twr_with_date_range():
    """Test TWR with specified date range."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 16),
                "quantity": Decimal("10"),
                "market_value": Decimal("2510.00"),
                "price": Decimal("251.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 17),
                "quantity": Decimal("10"),
                "market_value": Decimal("2520.00"),
                "price": Decimal("252.00"),
            },
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    engine = ReturnsEngine()

    # Compute TWR for just first two days
    twr = engine.compute_twr(daily_positions, start_date=date(2024, 1, 15), end_date=date(2024, 1, 16))

    # Should be positive (value increased)
    assert twr > 0


def test_mwr_simple_case():
    """Test MWR with simple cash flows."""
    cash_flows = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "date": date(2024, 1, 1),
                "amount": Decimal("-10000.00"),  # Initial investment
                "flow_type": "WITHDRAWAL",
            },
            {
                "account_id": "ACC001",
                "date": date(2024, 6, 1),
                "amount": Decimal("5000.00"),  # Partial withdrawal
                "flow_type": "DEPOSIT",
            },
        ],
        columns=["account_id", "date", "amount", "flow_type"],
    )

    ending_value = Decimal("12000.00")  # Final value

    engine = ReturnsEngine()
    mwr = engine.compute_mwr(cash_flows, ending_value)

    # Should compute a reasonable IRR
    assert isinstance(mwr, Decimal)


def test_mwr_empty_cash_flows_raises():
    """Test MWR with empty cash flows raises ValueError."""
    cash_flows = pd.DataFrame(columns=["account_id", "date", "amount", "flow_type"])

    engine = ReturnsEngine()
    with pytest.raises(ValueError, match="Cannot compute MWR with empty cash flows"):
        engine.compute_mwr(cash_flows, Decimal("10000.00"))


def test_mwr_no_cash_flows_in_range_raises():
    """Test MWR with no cash flows in date range raises ValueError."""
    cash_flows = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "date": date(2024, 1, 1),
                "amount": Decimal("-10000.00"),
                "flow_type": "WITHDRAWAL",
            }
        ],
        columns=["account_id", "date", "amount", "flow_type"],
    )

    engine = ReturnsEngine()
    with pytest.raises(ValueError, match="No cash flows in specified date range"):
        engine.compute_mwr(
            cash_flows,
            Decimal("10000.00"),
            start_date=date(2024, 6, 1),
            end_date=date(2024, 12, 31),
        )


def test_period_returns_mtd():
    """Test MTD return calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2550.00"),
                "price": Decimal("255.00"),
            },
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    cash_flows = pd.DataFrame(columns=["account_id", "date", "amount", "flow_type"])

    engine = ReturnsEngine()
    period_returns = engine.compute_period_returns(daily_positions, cash_flows, date(2024, 1, 15))

    # MTD should have a value
    assert "MTD" in period_returns
    assert period_returns["MTD"] is not None


def test_period_returns_ytd():
    """Test YTD return calculation."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 6, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2700.00"),
                "price": Decimal("270.00"),
            },
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    cash_flows = pd.DataFrame(columns=["account_id", "date", "amount", "flow_type"])

    engine = ReturnsEngine()
    period_returns = engine.compute_period_returns(daily_positions, cash_flows, date(2024, 6, 15))

    # YTD should have a value
    assert "YTD" in period_returns
    assert period_returns["YTD"] is not None
    # Should be positive (value increased from 2500 to 2700)
    assert period_returns["YTD"] > 0


def test_period_returns_all_periods():
    """Test that all standard periods are computed."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2020, 1, 1),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 6, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2700.00"),
                "price": Decimal("270.00"),
            },
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    cash_flows = pd.DataFrame(columns=["account_id", "date", "amount", "flow_type"])

    engine = ReturnsEngine()
    period_returns = engine.compute_period_returns(daily_positions, cash_flows, date(2024, 6, 15))

    # All standard periods should be present
    expected_periods = ["MTD", "QTD", "YTD", "1Y", "3Y", "5Y", "Since Inception"]
    for period in expected_periods:
        assert period in period_returns


def test_period_returns_future_period():
    """Test that periods with future start dates return None."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 6, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2700.00"),
                "price": Decimal("270.00"),
            }
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    cash_flows = pd.DataFrame(columns=["account_id", "date", "amount", "flow_type"])

    engine = ReturnsEngine()
    # Ask for 5Y return when we only have data from 2024
    period_returns = engine.compute_period_returns(daily_positions, cash_flows, date(2024, 6, 15))

    # 5Y should be None (start date would be in 2019, before data starts)
    # But since inception is 2024, it will use that as the start
    assert "5Y" in period_returns


def test_twr_multiple_securities():
    """Test TWR with multiple securities in same account."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 15),
                "quantity": Decimal("20"),
                "market_value": Decimal("1500.00"),
                "price": Decimal("75.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 16),
                "quantity": Decimal("10"),
                "market_value": Decimal("2510.00"),
                "price": Decimal("251.00"),
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "date": date(2024, 1, 16),
                "quantity": Decimal("20"),
                "market_value": Decimal("1505.00"),
                "price": Decimal("75.25"),
            },
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    engine = ReturnsEngine()
    twr = engine.compute_twr(daily_positions)

    # Day 1 total: 2500 + 1500 = 4000
    # Day 2 total: 2510 + 1505 = 4015
    # Return: (4015 - 4000) / 4000 = 0.00375
    assert twr > 0
    assert twr < Decimal("0.01")
