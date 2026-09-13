"""
Fidelity per-account "History_for_Account_*.csv" export adapter.

Handles the documented quirks (see quirks_manifest.md):
  - 2 blank lines above the header        -> skiprows=2
  - Trailing legal-boilerplate block      -> detected and dropped
                                              structurally (see
                                              _split_data_and_boilerplate),
                                              logged, never passed downstream
  - "No Description" on cash-only rows    -> description falls back to the
                                              Action column, which holds the
                                              actual readable text
  - Comma-thousands dollar amounts        -> parsing.parse_money
  - Per-account files, not household-wide -> load_fidelity_household()
                                              stitches N files into one
                                              transactions_df

Deliberate/discovered edge case not in the manifest's callout list but
present in the actual data:
  - "TRANSFERRED FROM ACCT" rows (in-kind security transfer: a share
    quantity with no price and no dollar amount) -> included,
    txn_type=TRANSFER_IN_KIND, needs_review=True. Valuation of the
    transferred shares depends on Phase 2's price history, not this
    adapter.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import pandas as pd

from ..canonical_schema import (
    QUARANTINE_COLUMNS,
    STRUCTURAL_DROPPED_COLUMNS,
    TRANSACTIONS_COLUMNS,
)
from ..parsing import parse_date, parse_money, parse_quantity

SOURCE_CUSTODIAN = "fidelity"

HEADER = [
    "Run Date",
    "Account",
    "Action",
    "Symbol",
    "Security Description",
    "Security Type",
    "Quantity",
    "Price ($)",
    "Commission ($)",
    "Fees ($)",
    "Accrued Interest ($)",
    "Amount ($)",
    "Settlement Date",
]

# Prefix-matched against the free-text Action column. Order matters —
# checked top to bottom, first match wins.
FIDELITY_ACTION_PREFIX_MAP = [
    ("YOU BOUGHT", ("BUY", False, None)),
    ("YOU SOLD", ("SELL", False, None)),
    ("REINVESTMENT", ("REINVEST", False, None)),
    ("DIRECT DEBIT ADVISORY FEE", ("FEE", False, None)),
    ("DIVIDEND RECEIVED", ("DIVIDEND", False, None)),
    ("INTEREST EARNED", ("INTEREST", False, None)),
    (
        "TRANSFERRED FROM ACCT",
        (
            "TRANSFER_IN_KIND",
            True,
            "In-kind security transfer — no price/amount provided; valuation depends on Phase 2 price history",
        ),
    ),
]


def _map_action(action: str) -> Tuple[str, bool, Union[str, None]]:
    for prefix, mapping in FIDELITY_ACTION_PREFIX_MAP:
        if action.startswith(prefix):
            return mapping
    return "UNKNOWN", True, f"Unrecognized Fidelity action: {action!r}"


def _split_data_and_boilerplate(
    path: Path,
) -> Tuple[List[dict], List[dict]]:
    """
    Read the raw file (after the 2 blank preamble lines) and separate real
    transaction rows from the trailing legal-boilerplate block.

    Approach: a real data row (a) has exactly len(HEADER) fields and
    (b) starts with a value that parses as a valid MM/DD/YYYY date. The
    boilerplate paragraph fails both conditions (prose lines, wrapped
    across multiple physical lines, inconsistent field counts) so it's
    identified structurally rather than by a fragile fixed-row-count
    assumption — robust to the boilerplate block changing length across
    export vintages.
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        lines = list(csv.reader(f))

    # Skip the 2 blank preamble lines, then the header row itself.
    body = lines[2:]
    if not body or [c.strip() for c in body[0]] != HEADER:
        raise ValueError(f"{path.name}: expected Fidelity header at row 3, got {body[0] if body else None}")
    body = body[1:]

    data_rows: List[dict] = []
    dropped_rows: List[dict] = []

    for j, fields in enumerate(body):
        source_row = j + 4  # 2 blank + 1 header + 1-index
        is_data_row = len(fields) == len(HEADER)
        if is_data_row:
            try:
                parse_date(fields[0])
            except ValueError:
                is_data_row = False

        if is_data_row:
            data_rows.append(dict(zip(HEADER, fields)))
        else:
            dropped_rows.append(
                {
                    "source_custodian": SOURCE_CUSTODIAN,
                    "source_file": path.name,
                    "source_row": source_row,
                    "raw_line": ",".join(fields),
                    "drop_reason": "trailing legal-boilerplate block (non-data row: wrong field count or unparseable date)",
                }
            )

    return data_rows, dropped_rows


def load_fidelity_export(path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse a single Fidelity per-account History export.

    Returns (transactions_df, quarantine_df, structural_dropped_df).
    No custodian_positions_df here either — this is a transaction history
    export, not a position snapshot.
    """
    path = Path(path)
    data_rows, dropped_rows = _split_data_and_boilerplate(path)

    transactions: List[dict] = []
    quarantine: List[dict] = []

    for idx, row in enumerate(data_rows):
        source_row = idx + 4  # matches the offset used in _split_data_and_boilerplate

        trade_date = None
        try:
            trade_date = parse_date(row["Run Date"])
        except ValueError as e:
            quarantine.append(
                {
                    "source_custodian": SOURCE_CUSTODIAN,
                    "source_file": path.name,
                    "source_row": source_row,
                    "raw_row": row,
                    "quarantine_reason": f"unparseable trade date: {e}",
                }
            )
            continue

        action = row["Action"]
        txn_type, needs_review, review_reason = _map_action(action)

        raw_description = row["Security Description"]
        description = action if raw_description == "No Description" else (raw_description or None)

        transactions.append(
            {
                "account_id": row["Account"],
                "security_id": row["Symbol"] or None,
                "trade_date": trade_date,
                "settlement_date": parse_date(row["Settlement Date"]),
                "txn_type": txn_type,
                "raw_type": action,
                "quantity": parse_quantity(row["Quantity"]),
                "price": parse_money(row["Price ($)"]),
                "net_amount": parse_money(row["Amount ($)"]),
                "fee_amount": parse_money(row["Commission ($)"]) or parse_money(row["Fees ($)"]),
                "description": description,
                "source_custodian": SOURCE_CUSTODIAN,
                "source_file": path.name,
                "source_row": source_row,
                "needs_review": needs_review,
                "review_reason": review_reason,
            }
        )

    transactions_df = pd.DataFrame(transactions, columns=TRANSACTIONS_COLUMNS)
    quarantine_df = pd.DataFrame(quarantine, columns=QUARANTINE_COLUMNS)
    structural_dropped_df = pd.DataFrame(dropped_rows, columns=STRUCTURAL_DROPPED_COLUMNS)
    return transactions_df, quarantine_df, structural_dropped_df


def load_fidelity_household(
    paths: Iterable[Union[str, Path]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stitch multiple per-account Fidelity exports into one household-level
    result. Fidelity's real export is per-account, not household-wide —
    this is the seam where that gets reassembled, rather than assuming
    (incorrectly) that one file equals one client.
    """
    all_txns, all_quarantine, all_dropped = [], [], []
    for p in paths:
        txns, quarantine, dropped = load_fidelity_export(p)
        all_txns.append(txns)
        all_quarantine.append(quarantine)
        all_dropped.append(dropped)

    transactions_df = pd.concat(all_txns, ignore_index=True) if all_txns else pd.DataFrame(columns=TRANSACTIONS_COLUMNS)
    quarantine_df = pd.concat(all_quarantine, ignore_index=True) if all_quarantine else pd.DataFrame(columns=QUARANTINE_COLUMNS)
    dropped_df = pd.concat(all_dropped, ignore_index=True) if all_dropped else pd.DataFrame(columns=STRUCTURAL_DROPPED_COLUMNS)
    return transactions_df, quarantine_df, dropped_df
