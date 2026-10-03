"""
Tests for Phase 3 Cash Flow Extraction.
"""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from pillar1.canonical_schema import CASH_FLOWS_COLUMNS
from pillar1.cash_flows import CashFlowExtractor
from pillar1.validation import validate_cash_flows_df


def test_cash_flow_extractor_buy_sell():
    """Test that BUY and SELL transactions are extracted as cash flows."""
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

    extractor = CashFlowExtractor()
    cash_flows = extractor.extract_cash_flows(transactions)

    # Validate schema
    violations = validate_cash_flows_df(cash_flows)
    assert violations == [], f"Schema violations: {violations}"

    assert len(cash_flows) == 2

    # BUY should be WITHDRAWAL (cash out)
    buy_flow = cash_flows[cash_flows["date"] == date(2024, 1, 15)].iloc[0]
    assert buy_flow["flow_type"] == "WITHDRAWAL"
    assert buy_flow["amount"] == Decimal("-2500.00")

    # SELL should be DEPOSIT (cash in)
    sell_flow = cash_flows[cash_flows["date"] == date(2024, 2, 1)].iloc[0]
    assert sell_flow["flow_type"] == "DEPOSIT"
    assert sell_flow["amount"] == Decimal("1275.00")


def test_cash_flow_extractor_fees_excluded_by_default():
    """Test that fees are excluded from cash flows by default."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": None,
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 15),
                "txn_type": "FEE",
                "raw_type": "Service Fee",
                "quantity": None,
                "price": None,
                "net_amount": Decimal("-100.00"),
                "fee_amount": Decimal("100.00"),
                "description": "Quarterly advisory fee",
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

    extractor = CashFlowExtractor(include_fees=False)
    cash_flows = extractor.extract_cash_flows(transactions)

    assert len(cash_flows) == 0


def test_cash_flow_extractor_fees_included_when_configured():
    """Test that fees are included when configured."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": None,
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 15),
                "txn_type": "FEE",
                "raw_type": "Service Fee",
                "quantity": None,
                "price": None,
                "net_amount": Decimal("-100.00"),
                "fee_amount": Decimal("100.00"),
                "description": "Quarterly advisory fee",
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

    extractor = CashFlowExtractor(include_fees=True)
    cash_flows = extractor.extract_cash_flows(transactions)

    assert len(cash_flows) == 1
    assert cash_flows.iloc[0]["flow_type"] == "FEE"
    assert cash_flows.iloc[0]["amount"] == Decimal("-100.00")


def test_cash_flow_extractor_dividends():
    """Test dividend extraction."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "DIVIDEND",
                "raw_type": "Dividend",
                "quantity": None,
                "price": None,
                "net_amount": Decimal("50.00"),
                "fee_amount": Decimal("0"),
                "description": "Dividend payment",
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

    extractor = CashFlowExtractor(include_dividends=True)
    cash_flows = extractor.extract_cash_flows(transactions)

    assert len(cash_flows) == 1
    assert cash_flows.iloc[0]["flow_type"] == "DIVIDEND"
    # Dividends should be positive (cash inflow)
    assert cash_flows.iloc[0]["amount"] == Decimal("50.00")


def test_cash_flow_extractor_reinvest_excluded():
    """Test that reinvested dividends are NOT cash flows (shares, not cash)."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "VTI",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 17),
                "txn_type": "REINVEST",
                "raw_type": "Reinvest Dividend",
                "quantity": Decimal("0.5"),
                "price": Decimal("250.00"),
                "net_amount": Decimal("-125.00"),
                "fee_amount": Decimal("0"),
                "description": "Reinvest dividend",
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

    extractor = CashFlowExtractor()
    cash_flows = extractor.extract_cash_flows(transactions)

    # REINVEST should not appear in cash flows
    assert len(cash_flows) == 0


def test_cash_flow_extractor_stock_split_excluded():
    """Test that stock splits are NOT cash flows."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": "AAPL",
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 15),
                "txn_type": "STOCK_SPLIT",
                "raw_type": "Stock Split",
                "quantity": Decimal("40"),
                "price": None,
                "net_amount": Decimal("0"),
                "fee_amount": Decimal("0"),
                "description": "4-for-1 split",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": True,
                "review_reason": "Stock split",
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

    extractor = CashFlowExtractor()
    cash_flows = extractor.extract_cash_flows(transactions)

    assert len(cash_flows) == 0


def test_cash_flow_extractor_journal_account_level():
    """Test that journal entries are treated as external flows at account level."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": None,
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 15),
                "txn_type": "JOURNAL_CASH",
                "raw_type": "Journal",
                "quantity": None,
                "price": None,
                "net_amount": Decimal("5000.00"),
                "fee_amount": Decimal("0"),
                "description": "Transfer from ACC002",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": True,
                "review_reason": "Inter-account transfer",
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

    extractor = CashFlowExtractor(household_level=False)
    cash_flows = extractor.extract_cash_flows(transactions)

    assert len(cash_flows) == 1
    assert cash_flows.iloc[0]["flow_type"] == "TRANSFER_IN"
    assert cash_flows.iloc[0]["amount"] == Decimal("5000.00")


def test_cash_flow_extractor_journal_household_level():
    """Test that journal entries are excluded at household level."""
    transactions = pd.DataFrame(
        [
            {
                "account_id": "ACC001",
                "security_id": None,
                "trade_date": date(2024, 1, 15),
                "settlement_date": date(2024, 1, 15),
                "txn_type": "JOURNAL_CASH",
                "raw_type": "Journal",
                "quantity": None,
                "price": None,
                "net_amount": Decimal("5000.00"),
                "fee_amount": Decimal("0"),
                "description": "Transfer from ACC002",
                "source_custodian": "schwab",
                "source_file": "test.csv",
                "source_row": 1,
                "needs_review": True,
                "review_reason": "Inter-account transfer",
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

    extractor = CashFlowExtractor(household_level=True)
    cash_flows = extractor.extract_cash_flows(transactions)

    # At household level, internal transfers cancel out
    assert len(cash_flows) == 0


def test_get_external_cash_flows_only():
    """Test filtering to only external cash flows."""
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
                "trade_date": date(2024, 1, 20),
                "settlement_date": date(2024, 1, 22),
                "txn_type": "DIVIDEND",
                "raw_type": "Dividend",
                "quantity": None,
                "price": None,
                "net_amount": Decimal("50.00"),
                "fee_amount": Decimal("0"),
                "description": "Dividend",
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

    extractor = CashFlowExtractor(include_dividends=True)
    cash_flows = extractor.extract_cash_flows(transactions)

    # Should have both BUY (WITHDRAWAL) and DIVIDEND
    assert len(cash_flows) == 2

    # Filter to external only
    external_only = extractor.get_external_cash_flows_only(cash_flows)

    # Should only have the BUY (WITHDRAWAL)
    assert len(external_only) == 1
    assert external_only.iloc[0]["flow_type"] == "WITHDRAWAL"


def test_cash_flows_sorted_by_date():
    """Test that cash flows are sorted by date for XIRR."""
    transactions = pd.DataFrame(
        [
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

    extractor = CashFlowExtractor()
    cash_flows = extractor.extract_cash_flows(transactions)

    # Should be sorted by date
    assert cash_flows.iloc[0]["date"] == date(2024, 1, 15)
    assert cash_flows.iloc[1]["date"] == date(2024, 2, 1)
