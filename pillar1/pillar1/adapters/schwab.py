"""
Schwab advisor/master export adapter.

Handles the documented quirks (see quirks_manifest.md):
  - UTF-8 BOM                         -> encoding="utf-8-sig"
  - 3 preamble rows above the header  -> skiprows=3
  - Combined "Symbol/CUSIP" column    -> this file only ever populates it
                                         with tickers; renamed to security_id
                                         as-is (CUSIP-vs-ticker disambiguation
                                         is a Phase 1 SecurityMaster concern)
  - "$" + parentheses-negative dollar fields -> parsing.parse_money

Deliberate data-quality edge cases handled explicitly:
  - Blank Date (one VTI buy row)   -> QUARANTINED, not inferred from
                                       settlement date. Silently inferring
                                       a trade date would corrupt daily
                                       valuation math three phases from now
                                       for a save that costs nothing to do
                                       properly: a human confirms it once.
  - Stock Split (AAPL 4-for-1)     -> included, txn_type=STOCK_SPLIT,
                                       needs_review=True. Not a buy/sell;
                                       downstream share-quantity/cost-basis
                                       adjustment logic must treat it apart.
  - Same-day buy+sell (MSFT)       -> both rows included normally; nothing
                                       about same-day round-trips breaks the
                                       transactions_df contract itself.
  - Fee-only rows, no security     -> included, txn_type=FEE, security_id=None
                                       (explicitly allowed for FEE).
  - Journal (inter-account xfers)  -> included, txn_type=JOURNAL_CASH,
                                       needs_review=True (account- vs.
                                       household-level MWR treatment is a
                                       Phase 3 decision, not this adapter's).
  - Reinvest Shares (DRIP,         -> included, txn_type=REINVEST, quantity
    fractional quantities)            kept as Decimal to preserve precision.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from ..canonical_schema import QUARANTINE_COLUMNS, TRANSACTIONS_COLUMNS
from ..parsing import parse_date, parse_money, parse_quantity
from ..type_mapping import ExactTypeMap, load_schwab_type_map

SOURCE_CUSTODIAN = "schwab"

_DECLARED_COUNT_RE = re.compile(r"(\d+)\s+records exported", re.IGNORECASE)


def parse_declared_record_count(path: Union[str, Path]) -> Optional[int]:
    """
    Read the custodian-declared record count from Schwab's own preamble
    line (e.g. "100 records exported"), if present. Used by the
    reconciliation report to auto-detect the expected row count instead
    of requiring it to be passed manually and risking a stale/wrong
    number being trusted.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as f:
        first_line = f.readline()
    m = _DECLARED_COUNT_RE.search(first_line)
    return int(m.group(1)) if m else None


def _map_type(raw_type: str, type_map: ExactTypeMap) -> Tuple[str, bool, Union[str, None]]:
    return type_map.resolve(raw_type)


def load_schwab_export(
    path: Union[str, Path], type_map: Optional[ExactTypeMap] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a raw Schwab advisor/master export.

    Returns (transactions_df, quarantine_df). This file format is a
    transaction history, not a position snapshot, so no
    custodian_positions_df is produced here — see
    adapters/positions.py for that contract.

    type_map: override the config-driven Type-column mapping (see
    type_mapping.py). Defaults to config/schwab_type_map.yaml.
    """
    type_map = type_map or load_schwab_type_map()
    path = Path(path)
    raw_df = pd.read_csv(path, skiprows=3, encoding="utf-8-sig", dtype=str)
    raw_df = raw_df.where(pd.notna(raw_df), None)

    transactions: List[dict] = []
    quarantine: List[dict] = []

    for i, row in raw_df.iterrows():
        source_row = i + 5  # +3 preamble rows + 1 header row + 1-index
        raw_row_dict = row.to_dict()

        trade_date = None
        try:
            trade_date = parse_date(row["Date"])
        except ValueError as e:
            quarantine.append(
                {
                    "source_custodian": SOURCE_CUSTODIAN,
                    "source_file": path.name,
                    "source_row": source_row,
                    "raw_row": raw_row_dict,
                    "quarantine_reason": f"unparseable trade date: {e}",
                }
            )
            continue

        if trade_date is None:
            quarantine.append(
                {
                    "source_custodian": SOURCE_CUSTODIAN,
                    "source_file": path.name,
                    "source_row": source_row,
                    "raw_row": raw_row_dict,
                    "quarantine_reason": (
                        "missing trade date — not inferred from settlement date; "
                        f"settlement_date on file = {row.get('Settlement date')!r}. "
                        "Needs manual confirmation before inclusion."
                    ),
                }
            )
            continue

        txn_type, needs_review, review_reason = _map_type(row["Type"], type_map)

        transactions.append(
            {
                "account_id": row["Account"],
                "security_id": row["Symbol/CUSIP"] or None,
                "trade_date": trade_date,
                "settlement_date": parse_date(row["Settlement date"]),
                "txn_type": txn_type,
                "raw_type": row["Type"],
                "quantity": parse_quantity(row["Quantity"]),
                "price": parse_money(row["Price"]),
                "net_amount": parse_money(row["Amount"]),
                "fee_amount": parse_money(row["Account fee"]),
                "description": row["Description"] or None,
                "source_custodian": SOURCE_CUSTODIAN,
                "source_file": path.name,
                "source_row": source_row,
                "needs_review": needs_review,
                "review_reason": review_reason,
            }
        )

    transactions_df = pd.DataFrame(transactions, columns=TRANSACTIONS_COLUMNS)
    quarantine_df = pd.DataFrame(quarantine, columns=QUARANTINE_COLUMNS)
    return transactions_df, quarantine_df
