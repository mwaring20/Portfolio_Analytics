from decimal import Decimal
from pathlib import Path

import pandas as pd

from pillar1.adapters.schwab import load_schwab_export
from pillar1.validation import validate_transactions_df

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_export_phantom.csv"


def _load():
    return load_schwab_export(FIXTURE)


def test_zero_schema_violations():
    txns, _ = _load()
    violations = validate_transactions_df(txns)
    assert violations == [], f"Schema violations: {violations}"


def test_zero_rows_lost_silently():
    """100 records exported (per file preamble) must all be accounted for
    across transactions_df + quarantine_df — none may just vanish."""
    txns, quarantine = _load()
    assert len(txns) + len(quarantine) == 100


def test_blank_date_row_is_quarantined_not_inferred():
    txns, quarantine = _load()
    # The blank-date row is a VTI buy in account 7042-1193 — must not
    # appear anywhere in transactions_df.
    leaked = txns[(txns["account_id"] == "7042-1193") & (txns["security_id"] == "VTI") & (txns["price"] == Decimal("253.10"))]
    assert leaked.empty

    assert len(quarantine) == 1
    q = quarantine.iloc[0]
    assert q["source_custodian"] == "schwab"
    assert "missing trade date" in q["quarantine_reason"]
    assert q["raw_row"]["Symbol/CUSIP"] == "VTI"


def test_stock_split_flagged_not_converted_as_normal_transaction():
    txns, _ = _load()
    split = txns[txns["txn_type"] == "STOCK_SPLIT"]
    assert len(split) == 1
    row = split.iloc[0]
    assert row["security_id"] == "AAPL"
    assert bool(row["needs_review"]) is True
    assert "split" in row["review_reason"].lower()


def test_same_day_buy_and_sell_both_present():
    txns, _ = _load()
    msft_0314 = txns[
        (txns["account_id"] == "7042-1193")
        & (txns["security_id"] == "MSFT")
        & (txns["trade_date"].astype(str) == "2025-03-14")
    ]
    assert set(msft_0314["txn_type"]) == {"BUY", "SELL"}


def test_fee_only_row_has_no_security_id():
    txns, _ = _load()
    fees = txns[txns["txn_type"] == "FEE"]
    assert len(fees) >= 1
    assert fees["security_id"].isna().all()
    quarterly = fees[fees["raw_type"] == "Service Fee"]
    assert any("QUARTERLY ADVISORY FEE" in (d or "") for d in quarterly["description"])


def test_journal_rows_flagged_for_review():
    txns, _ = _load()
    journal = txns[txns["txn_type"] == "JOURNAL_CASH"]
    assert len(journal) == 5
    assert journal["needs_review"].all()
    assert journal["security_id"].isna().all()
    # Net amounts should be signed correctly (transfer out negative, in positive)
    assert (journal["net_amount"] < 0).any()
    assert (journal["net_amount"] > 0).any()


def test_reinvest_shares_preserve_fractional_decimal_quantity():
    txns, _ = _load()
    reinvest = txns[txns["txn_type"] == "REINVEST"]
    assert len(reinvest) == 10
    for q in reinvest["quantity"]:
        assert isinstance(q, Decimal)
    # Spot check one exact fractional quantity from the raw file.
    small = reinvest[reinvest["quantity"] == Decimal("0.0295")]
    assert not small.empty


def test_parenthesized_comma_amount_parses_correctly():
    txns, _ = _load()
    row = txns[(txns["account_id"] == "7042-1193") & (txns["security_id"] == "AAPL") & (txns["txn_type"] == "BUY") & (txns["trade_date"].astype(str) == "2024-10-17")]
    assert len(row) == 1
    assert row.iloc[0]["net_amount"] == Decimal("-3452.01")


def test_bom_and_preamble_handled():
    txns, _ = _load()
    # If BOM/preamble weren't handled, the first column name would be
    # mangled or the header row would be treated as data.
    assert set(txns["account_id"].unique()) == {"7042-1193", "7042-1194", "7042-2871"}
