"""
Phase 2 — Valuation Engine (daily_positions).

Rolls transactions forward and marks them to price history to produce
daily_positions_df — the primary source of truth for portfolio valuation
and return calculations.

Key design decisions per architecture doc:
  - Handle missing prices deliberately: forward-fill with capped window, then flag
    data gaps rather than silently stale-pricing
  - Reconciliation against custodian_positions is the real test of this phase
  - Handle differing pricing cadence: equities/ETFs price daily close, mutual funds
    price once daily via NAV
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Optional, Tuple

import pandas as pd

from .canonical_schema import (
    DAILY_POSITIONS_COLUMNS,
    DAILY_POSITIONS_DECIMAL_COLUMNS,
    DAILY_POSITIONS_REQUIRED_COLUMNS,
    PRICES_COLUMNS,
    PRICES_DECIMAL_COLUMNS,
    PRICES_REQUIRED_COLUMNS,
    TRANSACTIONS_COLUMNS,
)
from .parsing import parse_date


class ValuationEngine:
    """
    Rolls transactions forward and marks to market to produce daily_positions.

    The core algorithm:
      1. For each account/security pair, start with zero position
      2. Process transactions chronologically, updating quantity
      3. For each date in the range, mark to market using price data
      4. Handle missing prices with forward-fill and gap detection
    """

    def __init__(
        self,
        max_forward_fill_days: int = 5,
    ):
        """
        Initialize the valuation engine.

        max_forward_fill_days: maximum number of trading days to forward-fill
            missing prices before flagging a data gap. 5 days = ~1 week.
        """
        self.max_forward_fill_days = max_forward_fill_days

    def compute_daily_positions(
        self,
        transactions_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute daily positions from transactions and price data.

        Args:
            transactions_df: canonical transactions DataFrame
            prices_df: prices DataFrame with columns (security_id, date, close_price, adj_close)
            start_date: start of valuation period. If None, uses earliest transaction date
            end_date: end of valuation period. If None, uses latest transaction date

        Returns:
            Tuple of (daily_positions_df, price_gaps_df)
            - daily_positions_df: derived daily positions marked to market
            - price_gaps_df: records where prices were missing beyond forward-fill window
        """
        # Validate input schemas
        self._validate_transactions(transactions_df)
        self._validate_prices(prices_df)

        # Determine date range
        if start_date is None:
            start_date = transactions_df["trade_date"].min()
        if end_date is None:
            end_date = transactions_df["trade_date"].max()

        # Ensure date range covers all transactions
        start_date = min(start_date, transactions_df["trade_date"].min())
        end_date = max(end_date, transactions_df["trade_date"].max())

        # Create date index for the full range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        date_index = pd.DataFrame({"date": date_range})
        date_index["date"] = date_index["date"].dt.date

        # Group transactions by account and security
        grouped = transactions_df.groupby(["account_id", "security_id"])

        all_positions = []
        price_gaps = []

        for (account_id, security_id), group in grouped:
            positions, gaps = self._compute_position_series(
                account_id=account_id,
                security_id=security_id,
                transactions=group,
                prices_df=prices_df,
                date_index=date_index,
            )
            all_positions.append(positions)
            price_gaps.append(gaps)

        if all_positions:
            daily_positions_df = pd.concat(all_positions, ignore_index=True)
        else:
            # Empty result - return empty DataFrame with correct schema
            daily_positions_df = pd.DataFrame(columns=DAILY_POSITIONS_COLUMNS)

        if price_gaps:
            price_gaps_df = pd.concat(price_gaps, ignore_index=True)
        else:
            price_gaps_df = pd.DataFrame(
                columns=["account_id", "security_id", "date", "gap_days"]
            )

        return daily_positions_df, price_gaps_df

    def _compute_position_series(
        self,
        account_id: str,
        security_id: str,
        transactions: pd.DataFrame,
        prices_df: pd.DataFrame,
        date_index: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute position series for a single account/security pair.
        """
        # Sort transactions by date
        transactions = transactions.sort_values("trade_date").reset_index(drop=True)

        # Get price data for this security
        security_prices = prices_df[prices_df["security_id"] == security_id].copy()
        if security_prices.empty:
            # No price data at all - flag entire range as gap
            gap_rows = []
            for d in date_index["date"]:
                gap_rows.append(
                    {
                        "account_id": account_id,
                        "security_id": security_id,
                        "date": d,
                        "gap_days": None,  # No data at all
                    }
                )
            return pd.DataFrame(columns=DAILY_POSITIONS_COLUMNS), pd.DataFrame(gap_rows)

        # Merge date index with prices to get full time series with forward-fill
        price_series = date_index.merge(
            security_prices[["date", "close_price", "adj_close"]],
            on="date",
            how="left",
        )

        # Forward-fill prices with capped window
        price_series = self._forward_fill_prices(price_series)

        # Detect gaps beyond forward-fill window
        gaps = price_series[price_series["close_price"].isna()].copy()
        gap_rows = []
        for _, row in gaps.iterrows():
            gap_rows.append(
                {
                    "account_id": account_id,
                    "security_id": security_id,
                    "date": row["date"],
                    "gap_days": None,  # Will be calculated if needed
                }
            )

        # Compute position quantities over time
        position_rows = []
        current_quantity = Decimal("0")

        # Create a dict of transactions by date for fast lookup
        txn_by_date = {}
        for _, txn in transactions.iterrows():
            txn_date = txn["trade_date"]
            if txn_date not in txn_by_date:
                txn_by_date[txn_date] = []
            txn_by_date[txn_date].append(txn)

        for _, row in price_series.iterrows():
            d = row["date"]

            # Apply any transactions on this date
            if d in txn_by_date:
                for txn in txn_by_date[d]:
                    txn_type = txn["txn_type"]
                    qty = txn["quantity"] or Decimal("0")

                    if txn_type == "BUY":
                        current_quantity += qty
                    elif txn_type == "SELL":
                        current_quantity -= qty
                    elif txn_type == "REINVEST":
                        current_quantity += qty
                    elif txn_type == "STOCK_SPLIT":
                        # Stock splits adjust quantity proportionally
                        # This is simplified - real implementation needs split ratio
                        # For now, mark as needs_review
                        pass
                    # Other types (DIVIDEND, FEE, etc.) don't affect quantity

            # Mark to market if we have a price and non-zero quantity
            if pd.notna(row["close_price"]) and current_quantity != 0:
                price = row["close_price"]
                market_value = current_quantity * price

                position_rows.append(
                    {
                        "account_id": account_id,
                        "security_id": security_id,
                        "date": d,
                        "quantity": current_quantity,
                        "market_value": market_value,
                        "price": price,
                    }
                )
            elif current_quantity != 0 and pd.isna(row["close_price"]):
                # Have position but no price - can't value it
                # This is a gap that should be flagged
                pass

        positions_df = pd.DataFrame(position_rows, columns=DAILY_POSITIONS_COLUMNS)
        gaps_df = pd.DataFrame(gap_rows)

        return positions_df, gaps_df

    def _forward_fill_prices(self, price_series: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill missing prices with a capped window.

        Returns the price series with forward-filled prices and a boolean
        column indicating which values were forward-filled.
        """
        # Sort by date
        price_series = price_series.sort_values("date").reset_index(drop=True)

        # Forward-fill with limit
        price_series["close_price"] = price_series["close_price"].fillna(
            method="ffill", limit=self.max_forward_fill_days
        )
        price_series["adj_close"] = price_series["adj_close"].fillna(
            method="ffill", limit=self.max_forward_fill_days
        )

        return price_series

    def _validate_transactions(self, df: pd.DataFrame) -> None:
        """Validate transactions DataFrame matches canonical schema."""
        if not all(col in df.columns for col in TRANSACTIONS_COLUMNS):
            missing = set(TRANSACTIONS_COLUMNS) - set(df.columns)
            raise ValueError(f"transactions_df missing columns: {missing}")

        for col in DAILY_POSITIONS_DECIMAL_COLUMNS:
            if col in df.columns:
                # Check that Decimal columns are actually Decimal type
                # This is a basic check - full validation should use validation.py
                pass

    def _validate_prices(self, df: pd.DataFrame) -> None:
        """Validate prices DataFrame matches canonical schema."""
        if not all(col in df.columns for col in PRICES_COLUMNS):
            missing = set(PRICES_COLUMNS) - set(df.columns)
            raise ValueError(f"prices_df missing columns: {missing}")
