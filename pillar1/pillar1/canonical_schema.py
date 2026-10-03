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

# --- custodian_positions_df contract (Phase 0 stub) --------------------------
#
# Per the architecture doc: "Raw snapshot — reconciliation only, never
# used directly in return calcs." This is a DIFFERENT export type from
# the transaction-history files adapters/schwab.py and
# adapters/fidelity.py parse — a point-in-time holdings snapshot, not a
# transaction log. No real Schwab/Fidelity position-snapshot export has
# been seen yet (only transaction-history exports), so there is no
# concrete adapter — only the contract, defined now per the architecture
# doc's own principle that reference/config schemas should be settled up
# front so they don't change later. See adapters/positions.py.

CUSTODIAN_POSITIONS_COLUMNS = [
    "account_id",
    "as_of_date",
    "security_id",
    "quantity",
    "market_value",
    "source_custodian",
    "source_file",
    "source_row",
]

CUSTODIAN_POSITIONS_DECIMAL_COLUMNS = ["quantity", "market_value"]

# Every column is required for a position snapshot row — unlike
# transactions, there's no txn_type-dependent nullability here; a
# position either has a quantity/value as of a date, or it isn't a valid
# row.
CUSTODIAN_POSITIONS_REQUIRED_COLUMNS = list(CUSTODIAN_POSITIONS_COLUMNS)


# --- prices_df contract (Phase 2) ------------------------------------------
#
# Historical price data for securities. Used by the valuation engine to mark
# positions to market. Both close_price (price return) and adj_close (total
# return with dividends reinvested) are required per the architecture doc.

PRICES_COLUMNS = [
    "security_id",
    "date",
    "close_price",
    "adj_close",
]

PRICES_DECIMAL_COLUMNS = ["close_price", "adj_close"]

PRICES_REQUIRED_COLUMNS = list(PRICES_COLUMNS)


# --- daily_positions_df contract (Phase 2) ----------------------------------
#
# Derived: transactions rolled forward, marked to prices. This is the primary
# source of truth for portfolio valuation and return calculations.

DAILY_POSITIONS_COLUMNS = [
    "account_id",
    "security_id",
    "date",
    "quantity",
    "market_value",
    "price",
]

DAILY_POSITIONS_DECIMAL_COLUMNS = ["quantity", "market_value", "price"]

DAILY_POSITIONS_REQUIRED_COLUMNS = list(DAILY_POSITIONS_COLUMNS)


# --- cash_flows_df contract (Phase 3) ---------------------------------------
#
# Derived: external flows extracted from transactions. Feeds MWR/XIRR.
# Per architecture doc: fees should NOT appear as cash flows (already
# reflected in ending value); use actual effective date not batch date.

CASH_FLOWS_COLUMNS = [
    "account_id",
    "date",
    "amount",
    "flow_type",
]

CASH_FLOWS_DECIMAL_COLUMNS = ["amount"]

CASH_FLOWS_REQUIRED_COLUMNS = list(CASH_FLOWS_COLUMNS)

# Flow types for cash flow classification
FLOW_TYPES = {
    "DEPOSIT",      # External cash coming in
    "WITHDRAWAL",   # External cash going out
    "DIVIDEND",     # Dividend payments (may be reinvested or paid out)
    "INTEREST",     # Interest payments
    "FEE",          # Advisory/management fees (excluded from MWR by default)
    "TRANSFER_IN",  # Transfer from another account
    "TRANSFER_OUT", # Transfer to another account
}


# --- benchmark_prices_df contract (Phase 5) ---------------------------------
#
# Composite benchmark constructed from BenchmarkDefinition. Total return
# index data (dividends reinvested), not price-only.

BENCHMARK_PRICES_COLUMNS = [
    "benchmark_id",
    "date",
    "level",
    "total_return_level",
]

BENCHMARK_PRICES_DECIMAL_COLUMNS = ["level", "total_return_level"]

BENCHMARK_PRICES_REQUIRED_COLUMNS = list(BENCHMARK_PRICES_COLUMNS)


# --- returns_df contract (Phase 4, referenced in Phase 5) --------------------
#
# Long/tidy format — one table for account, composite, and benchmark returns.

RETURNS_COLUMNS = [
    "entity_id",
    "entity_type",  # "account", "composite", "benchmark"
    "date",
    "period_return",
    "cumulative_return",
]

RETURNS_DECIMAL_COLUMNS = ["period_return", "cumulative_return"]

RETURNS_REQUIRED_COLUMNS = list(RETURNS_COLUMNS)

