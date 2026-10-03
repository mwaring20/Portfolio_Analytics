"""
Phase 3 — Cash Flow Extraction.

Extracts external cash flows from transactions for MWR/XIRR calculation.

Per architecture doc:
  - Fees should NOT appear as cash flows (already reflected in ending value)
  - Use actual effective date (trade_date), not batch-processing date
  - Decide account-level vs. household-level MWR up front
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import List, Optional

import pandas as pd

from .canonical_schema import (
    CASH_FLOWS_COLUMNS,
    CASH_FLOWS_DECIMAL_COLUMNS,
    CASH_FLOWS_REQUIRED_COLUMNS,
    FLOW_TYPES,
    TXN_TYPES,
)


class CashFlowExtractor:
    """
    Extracts cash flows from transactions for MWR/XIRR calculation.

    Key design decisions:
      - Fees excluded by default (already reflected in ending value)
      - Deposits: positive amounts (cash coming in)
      - Withdrawals: negative amounts (cash going out)
      - Dividends/interest: can be included or excluded based on configuration
      - Inter-account transfers: treated as external flows at account level,
        but may be internal at household level (configurable)
    """

    def __init__(
        self,
        include_fees: bool = False,
        include_dividends: bool = True,
        include_interest: bool = True,
        household_level: bool = False,
    ):
        """
        Initialize the cash flow extractor.

        include_fees: if True, include fee transactions as cash flows
        include_dividends: if True, include dividend payments
        include_interest: if True, include interest payments
        household_level: if True, treat inter-account transfers as internal
                         (excluded from cash flows); if False, treat as external
        """
        self.include_fees = include_fees
        self.include_dividends = include_dividends
        self.include_interest = include_interest
        self.household_level = household_level

    def extract_cash_flows(
        self,
        transactions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract cash flows from transactions.

        Args:
            transactions_df: canonical transactions DataFrame

        Returns:
            cash_flows_df with columns (account_id, date, amount, flow_type)
        """
        cash_flows: List[dict] = []

        for _, txn in transactions_df.iterrows():
            flow_type, amount = self._classify_and_sign_transaction(txn)

            if flow_type is None:
                # Transaction type not included in cash flows
                continue

            cash_flows.append(
                {
                    "account_id": txn["account_id"],
                    "date": txn["trade_date"],
                    "amount": amount,
                    "flow_type": flow_type,
                }
            )

        if cash_flows:
            cash_flows_df = pd.DataFrame(cash_flows, columns=CASH_FLOWS_COLUMNS)
            # Sort by date for XIRR solver
            cash_flows_df = cash_flows_df.sort_values("date").reset_index(drop=True)
        else:
            cash_flows_df = pd.DataFrame(columns=CASH_FLOWS_COLUMNS)

        return cash_flows_df

    def _classify_and_sign_transaction(
        self, txn: pd.Series
    ) -> tuple[Optional[str], Optional[Decimal]]:
        """
        Classify a transaction and determine its cash flow amount with sign.

        Returns (flow_type, amount) or (None, None) if not a cash flow.
        """
        txn_type = txn["txn_type"]
        net_amount = txn["net_amount"]

        # Fees - excluded by default per architecture doc
        if txn_type == "FEE":
            if self.include_fees:
                return "FEE", net_amount
            return None, None

        # Dividends
        if txn_type == "DIVIDEND":
            if self.include_dividends:
                # Dividends are positive cash inflows
                return "DIVIDEND", abs(net_amount or Decimal("0"))
            return None, None

        # Interest
        if txn_type == "INTEREST":
            if self.include_interest:
                # Interest is positive cash inflow
                return "INTEREST", abs(net_amount or Decimal("0"))
            return None, None

        # Reinvested dividends - these are NOT cash flows (shares, not cash)
        if txn_type == "REINVEST":
            return None, None

        # Stock splits - not cash flows
        if txn_type == "STOCK_SPLIT":
            return None, None

        # Inter-account transfers
        if txn_type == "JOURNAL_CASH":
            if self.household_level:
                # At household level, internal transfers cancel out
                return None, None
            else:
                # At account level, treat as external flow
                # Negative net_amount = cash out (transfer out)
                # Positive net_amount = cash in (transfer in)
                if net_amount and net_amount < 0:
                    return "TRANSFER_OUT", net_amount
                elif net_amount and net_amount > 0:
                    return "TRANSFER_IN", net_amount
                return None, None

        # In-kind transfers - no cash amount
        if txn_type == "TRANSFER_IN_KIND":
            return None, None

        # Buy transactions - cash outflow
        if txn_type == "BUY":
            if net_amount:
                return "WITHDRAWAL", net_amount  # Already negative
            return None, None

        # Sell transactions - cash inflow
        if txn_type == "SELL":
            if net_amount:
                return "DEPOSIT", net_amount  # Already positive
            return None, None

        # Unknown transaction type
        return None, None

    def get_external_cash_flows_only(
        self,
        cash_flows_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Filter to only external cash flows (deposits and withdrawals).

        Useful for pure MWR calculation where dividends/interest are
        considered internal to the portfolio's return.
        """
        external_types = {"DEPOSIT", "WITHDRAWAL"}
        return cash_flows_df[cash_flows_df["flow_type"].isin(external_types)].copy()
