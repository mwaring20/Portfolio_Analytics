from pathlib import Path

from pillar1.ingest import ingest_fidelity_household, ingest_schwab
from pillar1.security_master import AssetClass
from pillar1.security_resolution import SecurityMasterRegistry

from .fakes import FakeProvider

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingest_schwab_resolves_every_security_in_the_fixture():
    registry = SecurityMasterRegistry(FakeProvider(), seed_table={})
    result = ingest_schwab(FIXTURES / "schwab_export_phantom.csv", registry)

    distinct_ids = set(result.transactions_df["security_id"].dropna().unique())
    # Every ticker in the Schwab fixture (AAPL, MSFT, VTI, VOO, BND, SCHD)
    # is covered by the fake provider's default profiles, so nothing
    # should be left unresolved.
    assert distinct_ids == {"VTI", "AAPL", "MSFT", "BND", "VOO", "SCHD"}
    assert result.unresolved_securities == []
    assert set(result.resolved_securities.keys()) == distinct_ids


def test_ingest_schwab_flags_etfs_for_review_but_still_classifies_them():
    registry = SecurityMasterRegistry(FakeProvider(), seed_table={})
    result = ingest_schwab(FIXTURES / "schwab_export_phantom.csv", registry)

    vti_entry = result.resolved_securities["VTI"]
    assert vti_entry.needs_review is True
    assert vti_entry.security.asset_class == AssetClass.EQUITY  # non-null, per the Phase 1 hard gate

    bnd_entry = result.resolved_securities["BND"]
    assert bnd_entry.security.asset_class == AssetClass.FIXED_INCOME

    aapl_entry = result.resolved_securities["AAPL"]
    assert aapl_entry.needs_review is False


def test_ingest_fidelity_household_shares_registry_cache_across_files():
    registry = SecurityMasterRegistry(FakeProvider(), seed_table={})
    result = ingest_fidelity_household(
        [FIXTURES / "History_for_Account_Z12-345678.csv", FIXTURES / "History_for_Account_Z98-765432.csv"],
        registry,
    )
    distinct_ids = set(result.transactions_df["security_id"].dropna().unique())
    assert distinct_ids == {"SCHD", "VOO", "AAPL", "BND", "MSFT", "VTI"}
    assert result.unresolved_securities == []
    # AAPL appears in both household accounts — should still be resolved
    # (and thus looked up) exactly once thanks to registry caching.
    fake = registry.provider
    assert fake.ticker_lookup_calls.count("AAPL") == 1


def test_ingest_reports_unresolved_security_when_provider_misses():
    provider = FakeProvider(profiles={})  # nothing resolves
    registry = SecurityMasterRegistry(provider, seed_table={})
    result = ingest_schwab(FIXTURES / "schwab_export_phantom.csv", registry)

    assert result.resolved_securities == {}
    assert len(result.unresolved_securities) == 6  # all 6 distinct tickers
    summary = result.security_master_summary()
    assert "unresolved" in summary.lower()
