"""
Tests for Phase 2 Position Reconciliation.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.canonical_schema import (
    CUSTODIAN_POSITIONS_COLUMNS,
    DAILY_POSITIONS_COLUMNS,
)
from pillar1.position_reconciliation import PositionReconciliation


def test_reconciliation_perfect_match():
    """Test reconciliation when daily and custodian positions match exactly."""
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

    custodian_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 1,
            }
        ],
        columns=CUSTODIAN_POSITIONS_COLUMNS,
    )

    reconciler = PositionReconciliation()
    report, unreconciled = reconciler.reconcile(daily_positions, custodian_positions)

    assert len(report) == 1
    assert report.iloc[0]["is_reconciled"] is True
    assert len(unreconciled) == 0

    summary = reconciler.get_reconciliation_summary(report)
    assert summary["total_positions"] == 1
    assert summary["reconciled_count"] == 1
    assert summary["unreconciled_count"] == 0
    assert summary["reconciliation_rate"] == Decimal("100")


def test_reconciliation_quantity_mismatch():
    """Test reconciliation when quantities differ beyond tolerance."""
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

    custodian_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 15),
                "quantity": Decimal("9"),  # 1 share difference
                "market_value": Decimal("2250.00"),
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 1,
            }
        ],
        columns=CUSTODIAN_POSITIONS_COLUMNS,
    )

    reconciler = PositionReconciliation(quantity_tolerance=Decimal("0.01"))
    report, unreconciled = reconciler.reconcile(daily_positions, custodian_positions)

    assert len(report) == 1
    assert report.iloc[0]["is_reconciled"] is False
    assert report.iloc[0]["quantity_diff"] == Decimal("1")
    assert len(unreconciled) == 1
    assert "Quantity diff" in unreconciled.iloc[0]["reconciliation_issue"]


def test_reconciliation_value_mismatch():
    """Test reconciliation when market values differ beyond tolerance."""
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

    custodian_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2400.00"),  # 4% difference
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 1,
            }
        ],
        columns=CUSTODIAN_POSITIONS_COLUMNS,
    )

    reconciler = PositionReconciliation(value_tolerance_pct=Decimal("0.01"))  # 1% tolerance
    report, unreconciled = reconciler.reconcile(daily_positions, custodian_positions)

    assert len(report) == 1
    assert report.iloc[0]["is_reconciled"] is False
    assert report.iloc[0]["value_diff_pct"] == Decimal("4.166666666666667")
    assert len(unreconciled) == 1
    assert "Value diff" in unreconciled.iloc[0]["reconciliation_issue"]


def test_reconciliation_missing_in_daily():
    """Test reconciliation when position exists in custodian but not daily."""
    daily_positions = pd.DataFrame(columns=DAILY_POSITIONS_COLUMNS)

    custodian_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 1,
            }
        ],
        columns=CUSTODIAN_POSITIONS_COLUMNS,
    )

    reconciler = PositionReconciliation()
    report, unreconciled = reconciler.reconcile(daily_positions, custodian_positions)

    assert len(report) == 0
    assert len(unreconciled) == 1
    assert "not in daily_positions" in unreconciled.iloc[0]["reconciliation_issue"]


def test_reconciliation_missing_in_custodian():
    """Test reconciliation when position exists in daily but not custodian."""
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

    custodian_positions = pd.DataFrame(columns=CUSTODIAN_POSITIONS_COLUMNS)

    reconciler = PositionReconciliation()
    report, unreconciled = reconciler.reconcile(daily_positions, custodian_positions)

    assert len(report) == 0
    assert len(unreconciled) == 1
    assert "not in custodian snapshot" in unreconciled.iloc[0]["reconciliation_issue"]


def test_reconciliation_within_tolerance():
    """Test that small differences within tolerance are reconciled."""
    daily_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "date": date(2024, 1, 15),
                "quantity": Decimal("10.005"),  # Small rounding difference
                "market_value": Decimal("2500.00"),
                "price": Decimal("250.00"),
            }
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    custodian_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 1,
            }
        ],
        columns=CUSTODIAN_POSITIONS_COLUMNS,
    )

    reconciler = PositionReconciliation(quantity_tolerance=Decimal("0.01"))
    report, unreconciled = reconciler.reconcile(daily_positions, custodian_positions)

    assert len(report) == 1
    assert report.iloc[0]["is_reconciled"] is True
    assert len(unreconciled) == 0


def test_reconciliation_specific_date():
    """Test reconciliation filtered to a specific date."""
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
        ],
        columns=DAILY_POSITIONS_COLUMNS,
    )

    custodian_positions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 15),
                "quantity": Decimal("10"),
                "market_value": Decimal("2500.00"),
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 1,
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "as_of_date": date(2024, 1, 16),
                "quantity": Decimal("10"),
                "market_value": Decimal("2510.00"),
                "source_custodian": "schwab",
                "source_file": "snapshot.csv",
                "source_row": 2,
            },
        ],
        columns=CUSTODIAN_POSITIONS_COLUMNS,
    )

    reconciler = PositionReconciliation()
    report, unreconciled = reconciler.reconcile(
        daily_positions, custodian_positions, as_of_date=date(2024, 1, 15)
    )

    # Should only reconcile the specified date
    assert len(report) == 1
    assert report.iloc[0]["date"] == date(2024, 1, 15)


def test_reconciliation_summary_empty():
    """Test reconciliation summary with empty report."""
    reconciler = PositionReconciliation()
    report = pd.DataFrame(columns=[
        "account_id", "security_id", "date", "daily_quantity",
        "custodian_quantity", "quantity_diff", "daily_market_value",
        "custodian_market_value", "value_diff_pct", "is_reconciled"
    ])

    summary = reconciler.get_reconciliation_summary(report)

    assert summary["total_positions"] == 0
    assert summary["reconciled_count"] == 0
    assert summary["unreconciled_count"] == 0
    assert summary["reconciliation_rate"] == Decimal("0")
