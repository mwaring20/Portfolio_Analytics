"""
Tests for Phase 2 Valuation Engine.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.canonical_schema import (
    DAILY_POSITIONS_COLUMNS,
    PRICES_COLUMNS,
)
from pillar1.validation import (
    validate_daily_positions_df,
    validate_prices_df,
)
from pillar1.valuation import ValuationEngine


def test_valuation_engine_basic_buy_sell():
    """Test basic buy/sell transactions roll forward correctly."""
    # Create simple transactions
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "BUY",
                "raw_type": "Buy",
                "quantity": Decimal("10"),
                "price": Decimal("250.00"),
                "net_amount": Decimal("-2500.00"),
                "fee_amount": Decimal("0"),
                "description": "Buy VTI",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": False,
                "review_reason": None,
            },
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "trade_date": date(2024, 2, 1),
                "settlement_date": date(2024, 2, 5),
                "txn_type": "SELL",
                "raw_type": "Sell",
                "quantity": Decimal("5"),
                "price": Decimal("255.00"),
                "net_amount": Decimal("1275.00"),
                "fee_amount": Decimal("0"),
                "description": "Sell VTI",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 2,
                "needs_review": False,
                "review_reason": None,
            },
        ],
        columns=[
            "account_id",
            "security_id",
            "trade_date",
            "settlement_date",
            "txn_type",
            "raw_type",
            "quantity",
            "price",
            "net_amount",
            "fee_amount",
            "description",
            "source_custodian",
            "source_file",
            "source_row",
            "needs_review",
            "review_reason",
        ],
    )

    # Create price data
    prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 15), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
            {"security_id": "VTI", "date": date(2024, 1, 16), "close_price": Decimal("251.00"), "adj_close": Decimal("251.00")},
            {"security_id": "VTI", "date": date(2024, 1, 17), "close_price": Decimal("252.00"), "adj_close": Decimal("252.00")},
            {"security_id": "VTI", "date": date(2024, 2, 1), "close_price": Decimal("255.00"), "adj_close": Decimal("255.00")},
        ],
        columns=PRICES_COLUMNS,
    )

    engine = ValuationEngine()
    daily_positions, price_gaps = engine.compute_daily_positions(transactions, prices)

    # Validate schema
    violations = validate_daily_positions_df(daily_positions)
    assert violations == [], f"Schema violations: {violations}"

    # Check that we have positions
    assert len(daily_positions) > 0

    # Check position after buy
    pos_after_buy = daily_positions[daily_positions["date"] == date(2024, 1, 15)]
    assert len(pos_after_buy) == 1
    assert pos_after_buy.iloc[0]["quantity"] == Decimal("10")
    assert pos_after_buy.iloc[0]["market_value"] == Decimal("2500.00")

    # Check position after sell
    pos_after_sell = daily_positions[daily_positions["date"] == date(2024, 2, 1)]
    assert len(pos_after_sell) == 1
    assert pos_after_sell.iloc[0]["quantity"] == Decimal("5")
    assert pos_after_sell.iloc[0]["market_value"] == Decimal("1275.00")


def test_valuation_engine_forward_fill_prices():
    """Test that missing prices are forward-filled within tolerance window."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "BUY",
                "raw_type": "Buy",
                "quantity": Decimal("10"),
                "price": Decimal("250.00"),
                "net_amount": Decimal("-2500.00"),
                "fee_amount": Decimal("0"),
                "description": "Buy VTI",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": False,
                "review_reason": None,
            },
        ],
        columns=[
            "account_id",
            "security_id",
            "trade_date",
            "settlement_date",
            "txn_type",
            "raw_type",
            "quantity",
            "price",
            "net_amount",
            "fee_amount",
            "description",
            "source_custodian",
            "source_file",
            "source_row",
            "needs_review",
            "review_reason",
        ],
    )

    # Prices with gaps
    prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 15), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
            {"security_id": "VTI", "date": date(2024, 1, 21), "close_price": Decimal("260.00"), "adj_close": Decimal("260.00")},
        ],
        columns=PRICES_COLUMNS,
    )

    engine = ValuationEngine(max_forward_fill_days=5)
    daily_positions, price_gaps = engine.compute_daily_positions(
        transactions, prices, end_date=date(2024, 1, 21)
    )

    # Should have positions on days 1/15-1/19 (forward-filled from 1/15)
    # Day 1/20 would be the 6th day, beyond tolerance
    assert len(daily_positions) >= 1
    assert len(price_gaps) > 0  # Should flag gaps beyond tolerance


def test_valuation_engine_multiple_securities():
    """Test valuation with multiple securities in the same account."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "BUY",
                "raw_type": "Buy",
                "quantity": Decimal("10"),
                "price": Decimal("250.00"),
                "net_amount": Decimal("-2500.00"),
                "fee_amount": Decimal("0"),
                "description": "Buy VTI",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": False,
                "review_reason": None,
            },
            {
                "account_id": "ACC001",
                "security_id": "BND",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "BUY",
                "raw_type": "Buy",
                "quantity": Decimal("20"),
                "price": Decimal("75.00"),
                "net_amount": Decimal("-1500.00"),
                "fee_amount": Decimal("0"),
                "description": "Buy BND",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 2,
                "needs_review": False,
                "review_reason": None,
            },
        ],
        columns=[
            "account_id",
            "security_id",
            "trade_date",
            "settlement_date",
            "txn_type",
            "raw_type",
            "quantity",
            "price",
            "net_amount",
            "fee_amount",
            "description",
            "source_custodian",
            "source_file",
            "source_row",
            "needs_review",
            "review_reason",
        ],
    )

    prices = pd.DataFrame(
        [
            {"security_id": "VTI", "date": date(2024, 1, 15), "close_price": Decimal("250.00"), "adj_close": Decimal("250.00")},
            {"security_id": "BND", "date": date(2024, 1, 15), "close_price": Decimal("75.00"), "adj_close": Decimal("75.00")},
        ],
        columns=PRICES_COLUMNS,
    )

    engine = ValuationEngine()
    daily_positions, price_gaps = engine.compute_daily_positions(transactions, prices)

    # Should have positions for both securities
    assert len(daily_positions) == 2
    securities = set(daily_positions["security_id"])
    assert securities == {"VTI", "BND"}


def test_valuation_engine_empty_transactions():
    """Test that empty transactions produce empty daily positions."""
    transactions = pd.DataFrame(
        columns=[
            "account_id",
            "security_id",
            "trade_date",
            "settlement_date",
            "txn_type",
            "raw_type",
            "quantity",
            "price",
            "net_amount",
            "fee_amount",
            "description",
            "source_custodian",
            "source_file",
            "source_row",
            "needs_review",
            "review_reason",
        ],
    )

    prices = pd.DataFrame(columns=PRICES_COLUMNS)

    engine = ValuationEngine()
    daily_positions, price_gaps = engine.compute_daily_positions(transactions, prices)

    assert len(daily_positions) == 0
    assert list(daily_positions.columns) == DAILY_POSITIONS_COLUMNS


def test_valuation_engine_missing_price_data():
    """Test that missing price data for a security is flagged."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "NO_PRICE",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "BUY",
                "raw_type": "Buy",
                "quantity": Decimal("10"),
                "price": Decimal("100.00"),
                "net_amount": Decimal("-1000.00"),
                "fee_amount": Decimal("0"),
                "description": "Buy NO_PRICE",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": False,
                "review_reason": None,
            },
        ],
        columns=[
            "account_id",
            "security_id",
            "trade_date",
            "settlement_date",
            "txn_type",
            "raw_type",
            "quantity",
            "price",
            "net_amount",
            "fee_amount",
            "description",
            "source_custodian",
            "source_file",
            "source_row",
            "needs_review",
            "review_reason",
        ],
    )

    # No price data for NO_PRICE
    prices = pd.DataFrame(columns=PRICES_COLUMNS)

    engine = ValuationEngine()
    daily_positions, price_gaps = engine.compute_daily_positions(transactions, prices)

    # Should have no positions (can't value without prices)
    assert len(daily_positions) == 0
    # Should flag gaps
    assert len(price_gaps) > 0
