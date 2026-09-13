"""
Canonical schema contract for Phase 0 (Custodian Ingestion).

Per the Pillar 1 data architecture doc: transactions are a pandas DataFrame
with an *enforced column/dtype contract*, not a Pydantic model (Pydantic is
reserved for low-cardinality reference/config entities). This module is the
single place that contract is defined, so every adapter and every test
checks itself against the same rules.

Design note on "quarantine" vs "needs_review":
  - QUARANTINE: a row that cannot be safely included in transactions_df at
    all (e.g. missing trade date). Held out entirely, never silently
    dropped, always inspectable.
  - needs_review=True (in-band): the row IS included in transactions_df —
    it parses cleanly — but a downstream phase (returns engine, cash flow
    extraction, security master) needs to make an explicit business
    decision about it (stock splits, inter-account transfers, in-kind
    transfers with no valuation). Flagging in-band means Phase 7's
    concentration-risk-flag pattern and Phase 15's audit-trail-integration
    pass both have a natural place to pick these up later.
"""

from __future__ import annotations

from typing import Optional

# --- Canonical transaction types -------------------------------------------
# One shared vocabulary across custodians. Each adapter's raw type/action
# strings map onto this set — this is the ONLY place custodian-specific
# type strings should ever be compared against, per adapter, in a mapping
# table (see adapters/schwab.py, adapters/fidelity.py).
TXN_TYPES = {
    "BUY",
    "SELL",
    "DIVIDEND",
    "REINVEST",
    "FEE",
    "INTEREST",
    "STOCK_SPLIT",
    "JOURNAL_CASH",       # cash moved between accounts (Schwab "Journal")
    "TRANSFER_IN_KIND",   # securities moved between accounts, no cash amount
    "UNKNOWN",            # unmapped raw type — always needs_review=True
}

# Transaction types for which security_id is legitimately allowed to be
# null (fee/cash-only events with no associated security).
TYPES_WITHOUT_SECURITY_OK = {"FEE", "INTEREST", "JOURNAL_CASH"}

# Transaction types for which net_amount is legitimately allowed to be
# null (no cash value at all — e.g. an in-kind transfer priced later
# once Phase 2's valuation engine marks the received shares to market).
TYPES_WITHOUT_AMOUNT_OK = {"TRANSFER_IN_KIND"}

# The canonical column order for transactions_df. Every adapter must
# return a DataFrame with exactly these columns, in this order.
TRANSACTIONS_COLUMNS = [
    "account_id",
    "security_id",       # raw ticker/symbol as given by custodian; CUSIP
                          # fallback resolution against SecurityMaster is
                          # Phase 1's job, not Phase 0's.
    "trade_date",
    "settlement_date",
    "txn_type",
    "raw_type",           # original custodian type/action string, verbatim
    "quantity",
    "price",
    "net_amount",
    "fee_amount",
    "description",
    "source_custodian",
    "source_file",
    "source_row",          # 1-indexed row number in the raw file, for
                            # traceability back to the original export
    "needs_review",
    "review_reason",
]

# Columns that must be Decimal-or-None (never float, per design principle
# in the architecture doc — fractional-share DRIP + float error).
DECIMAL_COLUMNS = ["quantity", "price", "net_amount", "fee_amount"]

# Columns that must never be null, regardless of txn_type.
ALWAYS_REQUIRED_COLUMNS = [
    "account_id",
    "trade_date",
    "txn_type",
    "raw_type",
    "source_custodian",
    "source_file",
    "source_row",
    "needs_review",
]

QUARANTINE_COLUMNS = [
    "source_custodian",
    "source_file",
    "source_row",
    "raw_row",          # dict of the original raw field values, untouched
    "quarantine_reason",
]

STRUCTURAL_DROPPED_COLUMNS = [
    "source_custodian",
    "source_file",
    "source_row",
    "raw_line",
    "drop_reason",
]
