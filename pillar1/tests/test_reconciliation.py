from pathlib import Path

from pillar1.adapters.fidelity import load_fidelity_household
from pillar1.adapters.schwab import load_schwab_export
from pillar1.reconciliation import build_reconciliation_report

FIXTURES = Path(__file__).parent / "fixtures"


def test_schwab_reconciles_against_declared_total():
    txns, quarantine = load_schwab_export(FIXTURES / "schwab_export_phantom.csv")
    report = build_reconciliation_report(
        source_label="schwab_export_phantom.csv",
        transactions_df=txns,
        quarantine_df=quarantine,
        total_input_rows=100,
        total_source="auto-detected from file's declared record count",
    )
    assert report.reconciles is True
    assert report.accounted_for == 100
    assert report.schema_violations == []
    assert report.txn_type_breakdown["JOURNAL_CASH"] == 5
    assert report.txn_type_breakdown["STOCK_SPLIT"] == 1
    assert "auto-detected" in report.as_text()


def test_schwab_flags_mismatch_when_total_is_wrong():
    txns, quarantine = load_schwab_export(FIXTURES / "schwab_export_phantom.csv")
    report = build_reconciliation_report(
        source_label="schwab_export_phantom.csv",
        transactions_df=txns,
        quarantine_df=quarantine,
        total_input_rows=99,  # deliberately wrong
    )
    assert report.reconciles is False
    assert "MISMATCH" in report.as_text()


def test_schwab_needs_review_breakdown_has_reasons():
    txns, quarantine = load_schwab_export(FIXTURES / "schwab_export_phantom.csv")
    report = build_reconciliation_report(
        source_label="schwab_export_phantom.csv",
        transactions_df=txns,
        quarantine_df=quarantine,
    )
    types_flagged = {r["txn_type"] for r in report.needs_review_breakdown}
    assert types_flagged == {"STOCK_SPLIT", "JOURNAL_CASH"}
    for r in report.needs_review_breakdown:
        assert r["reason"]  # never blank


def test_schwab_quarantine_detail_present():
    txns, quarantine = load_schwab_export(FIXTURES / "schwab_export_phantom.csv")
    report = build_reconciliation_report(
        source_label="schwab_export_phantom.csv",
        transactions_df=txns,
        quarantine_df=quarantine,
    )
    assert report.quarantine_count == 1
    assert "missing trade date" in report.quarantine_detail[0]["quarantine_reason"]
    assert "missing trade date" in report.as_text()


def test_fidelity_household_report_has_no_declared_total():
    txns, quarantine, dropped = load_fidelity_household(
        [FIXTURES / "History_for_Account_Z12-345678.csv", FIXTURES / "History_for_Account_Z98-765432.csv"]
    )
    report = build_reconciliation_report(
        source_label="household",
        transactions_df=txns,
        quarantine_df=quarantine,
        structural_dropped_df=dropped,
    )
    assert report.reconciles is None  # nothing to reconcile against
    assert report.structural_dropped_count == 20
    assert report.schema_violations == []
    text = report.as_text()
    assert "TRANSFER_IN_KIND" in text
