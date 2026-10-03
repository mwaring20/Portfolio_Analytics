"""
Phase 2 — Position Reconciliation.

Compares derived daily_positions against custodian_positions to flag
divergences beyond tolerance. This is the real test of the valuation
engine per the architecture doc.

Reconciliation flags are themselves audit-trail evidence for Pillar 3.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import List, Optional, Tuple

import pandas as pd

from .canonical_schema import (
    CUSTODIAN_POSITIONS_COLUMNS,
    DAILY_POSITIONS_COLUMNS,
)


class PositionReconciliation:
    """
    Reconciles derived daily_positions against custodian position snapshots.

    Flags divergences beyond configurable tolerance thresholds for:
      - Quantity differences
      - Market value differences
      - Missing positions in either dataset
    """

    def __init__(
        self,
        quantity_tolerance: Decimal = Decimal("0.01"),
        value_tolerance_pct: Decimal = Decimal("0.01"),  # 1%
    ):
        """
        Initialize the reconciler.

        quantity_tolerance: absolute quantity difference threshold (e.g., 0.01 shares)
        value_tolerance_pct: percentage difference threshold for market values
        """
        self.quantity_tolerance = quantity_tolerance
        self.value_tolerance_pct = value_tolerance_pct

    def reconcile(
        self,
        daily_positions_df: pd.DataFrame,
        custodian_positions_df: pd.DataFrame,
        as_of_date: Optional[date] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reconcile daily positions against custodian positions.

        Args:
            daily_positions_df: derived daily positions from valuation engine
            custodian_positions_df: custodian position snapshot
            as_of_date: if provided, reconcile only this date; otherwise reconcile all dates

        Returns:
            Tuple of (reconciliation_report_df, unreconciled_df)
            - reconciliation_report_df: detailed comparison for each matched position
            - unreconciled_df: positions that could not be matched or exceeded tolerance
        """
        # If as_of_date specified, filter both datasets
        if as_of_date:
            daily_positions_df = daily_positions_df[
                daily_positions_df["date"] == as_of_date
            ].copy()
            custodian_positions_df = custodian_positions_df[
                custodian_positions_df["as_of_date"] == as_of_date
            ].copy()

        # Create composite keys for matching
        daily_positions_df["_key"] = (
            daily_positions_df["account_id"].astype(str)
            + "|"
            + daily_positions_df["security_id"].astype(str)
            + "|"
            + daily_positions_df["date"].astype(str)
        )

        custodian_positions_df["_key"] = (
            custodian_positions_df["account_id"].astype(str)
            + "|"
            + custodian_positions_df["security_id"].astype(str)
            + "|"
            + custodian_positions_df["as_of_date"].astype(str)
        )

        # Find matches and mismatches
        daily_keys = set(daily_positions_df["_key"])
        custodian_keys = set(custodian_positions_df["_key"])

        matched_keys = daily_keys & custodian_keys
        only_in_daily = daily_keys - custodian_keys
        only_in_custodian = custodian_keys - daily_keys

        # Build reconciliation report for matched positions
        report_rows = []
        unreconciled_rows = []

        for key in matched_keys:
            daily_row = daily_positions_df[daily_positions_df["_key"] == key].iloc[0]
            custodian_row = custodian_positions_df[
                custodian_positions_df["_key"] == key
            ].iloc[0]

            qty_diff = abs(
                (daily_row["quantity"] or Decimal("0"))
                - (custodian_row["quantity"] or Decimal("0"))
            )
            value_diff_pct = self._calculate_value_diff_pct(
                daily_row["market_value"], custodian_row["market_value"]
            )

            is_reconciled = (
                qty_diff <= self.quantity_tolerance
                and value_diff_pct <= self.value_tolerance_pct
            )

            report_rows.append(
                {
                    "account_id": daily_row["account_id"],
                    "security_id": daily_row["security_id"],
                    "date": daily_row["date"],
                    "daily_quantity": daily_row["quantity"],
                    "custodian_quantity": custodian_row["quantity"],
                    "quantity_diff": qty_diff,
                    "daily_market_value": daily_row["market_value"],
                    "custodian_market_value": custodian_row["market_value"],
                    "value_diff_pct": value_diff_pct,
                    "is_reconciled": is_reconciled,
                }
            )

            if not is_reconciled:
                unreconciled_rows.append(
                    {
                        "account_id": daily_row["account_id"],
                        "security_id": daily_row["security_id"],
                        "date": daily_row["date"],
                        "reconciliation_issue": self._describe_issue(
                            qty_diff, value_diff_pct
                        ),
                    }
                )

        # Add unmatched positions to unreconciled
        for key in only_in_daily:
            row = daily_positions_df[daily_positions_df["_key"] == key].iloc[0]
            unreconciled_rows.append(
                {
                    "account_id": row["account_id"],
                    "security_id": row["security_id"],
                    "date": row["date"],
                    "reconciliation_issue": "Position in daily_positions but not in custodian snapshot",
                }
            )

        for key in only_in_custodian:
            row = custodian_positions_df[custodian_positions_df["_key"] == key].iloc[0]
            unreconciled_rows.append(
                {
                    "account_id": row["account_id"],
                    "security_id": row["security_id"],
                    "date": row["as_of_date"],
                    "reconciliation_issue": "Position in custodian snapshot but not in daily_positions",
                }
            )

        # Clean up temporary keys
        daily_positions_df = daily_positions_df.drop(columns=["_key"])
        custodian_positions_df = custodian_positions_df.drop(columns=["_key"])

        reconciliation_report_df = pd.DataFrame(report_rows)
        unreconciled_df = pd.DataFrame(unreconciled_rows)

        return reconciliation_report_df, unreconciled_df

    def _calculate_value_diff_pct(
        self, daily_value: Optional[Decimal], custodian_value: Optional[Decimal]
    ) -> Decimal:
        """
        Calculate percentage difference between two market values.

        Returns 0 if both are None or both are 0.
        Returns Decimal('Infinity') if denominator is 0 but numerator is not.
        """
        daily = daily_value or Decimal("0")
        custodian = custodian_value or Decimal("0")

        if daily == Decimal("0") and custodian == Decimal("0"):
            return Decimal("0")

        if custodian == Decimal("0"):
            return Decimal("Infinity")

        diff = abs(daily - custodian)
        return (diff / custodian) * Decimal("100")

    def _describe_issue(
        self, qty_diff: Decimal, value_diff_pct: Decimal
    ) -> str:
        """Generate a human-readable description of reconciliation issues."""
        issues = []

        if qty_diff > self.quantity_tolerance:
            issues.append(f"Quantity diff {qty_diff} exceeds tolerance {self.quantity_tolerance}")

        if value_diff_pct > self.value_tolerance_pct and value_diff_pct != Decimal("Infinity"):
            issues.append(
                f"Value diff {value_diff_pct:.2f}% exceeds tolerance {self.value_tolerance_pct:.2f}%"
            )
        elif value_diff_pct == Decimal("Infinity"):
            issues.append("Value diff is infinite (custodian value is zero)")

        return "; ".join(issues) if issues else "No issue"

    def get_reconciliation_summary(
        self, reconciliation_report_df: pd.DataFrame
    ) -> dict:
        """
        Generate summary statistics from a reconciliation report.

        Returns a dict with:
          - total_positions: total number of matched positions
          - reconciled_count: number of positions within tolerance
          - unreconciled_count: number of positions exceeding tolerance
          - reconciliation_rate: percentage of reconciled positions
        """
        if reconciliation_report_df.empty:
            return {
                "total_positions": 0,
                "reconciled_count": 0,
                "unreconciled_count": 0,
                "reconciliation_rate": Decimal("0"),
            }

        total = len(reconciliation_report_df)
        reconciled = reconciliation_report_df["is_reconciled"].sum()
        unreconciled = total - reconciled

        return {
            "total_positions": total,
            "reconciled_count": int(reconciled),
            "unreconciled_count": int(unreconciled),
            "reconciliation_rate": (
                (Decimal(reconciled) / Decimal(total)) * Decimal("100")
                if total > 0
                else Decimal("0")
            ),
        }
