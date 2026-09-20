from decimal import Decimal
from pathlib import Path

import pandas as pd

from pillar1.adapters.fidelity import load_fidelity_export, load_fidelity_household
from pillar1.validation import validate_transactions_df

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ACCT_1 = FIXTURE_DIR / "History_for_Account_Z12-345678.csv"
ACCT_2 = FIXTURE_DIR / "History_for_Account_Z98-765432.csv"


def test_zero_schema_violations_single_file():
    txns, _, _ = load_fidelity_export(ACCT_1)
    violations = validate_transactions_df(txns)
    assert violations == [], f"Schema violations: {violations}"


def test_trailing_boilerplate_dropped_not_passed_downstream():
    txns, quarantine, dropped = load_fidelity_export(ACCT_1)
    # None of the boilerplate prose should leak into transactions_df.
    assert not txns["account_id"].astype(str).str.contains("information contained", case=False).any()
    assert len(dropped) > 0
    assert all("boilerplate" in reason for reason in dropped["drop_reason"])
    # Every real transaction row should still be present.
    assert len(txns) + len(quarantine) == 33  # 33 real data rows in this fixture


def test_no_description_falls_back_to_action_text():
    txns, _, _ = load_fidelity_export(ACCT_1)
    cash_rows = txns[txns["security_id"].isna() & (txns["txn_type"].isin(["FEE", "INTEREST", "DIVIDEND"]))]
    assert not cash_rows.empty
    for desc in cash_rows["description"]:
        assert desc is not None
        assert desc != "No Description"


def test_transferred_from_acct_flagged_transfer_in_kind():
    txns, _, _ = load_fidelity_export(ACCT_1)
    transfers = txns[txns["txn_type"] == "TRANSFER_IN_KIND"]
    assert len(transfers) == 1
    row = transfers.iloc[0]
    assert row["security_id"] == "VTI"
    assert row["quantity"] == Decimal("12.0")
    assert pd.isna(row["net_amount"])  # no dollar amount provided by custodian
    assert bool(row["needs_review"]) is True


def test_comma_thousands_amount_parses_correctly():
    txns, _, _ = load_fidelity_export(ACCT_1)
    row = txns[(txns["security_id"] == "VOO") & (txns["txn_type"] == "BUY") & (txns["trade_date"].astype(str) == "2024-11-11")]
    assert len(row) == 1
    assert row.iloc[0]["net_amount"] == Decimal("-14372.45")

    # The >$1,000 comma case explicitly called out as previously-buggy.
    sold = txns[(txns["security_id"] == "VTI") & (txns["txn_type"] == "SELL") & (txns["trade_date"].astype(str) == "2025-12-25")]
    assert len(sold) == 1
    assert sold.iloc[0]["net_amount"] == Decimal("7453.11")


def test_household_stitching_combines_both_accounts():
    txns, quarantine, dropped = load_fidelity_household([ACCT_1, ACCT_2])
    assert set(txns["account_id"].unique()) == {"Z12-345678", "Z98-765432"}
    assert set(txns["source_file"].unique()) == {ACCT_1.name, ACCT_2.name}
    violations = validate_transactions_df(txns)
    assert violations == []
