from pillar1.security_master import AssetClass
from pillar1.security_resolution import SecurityMasterRegistry
from pillar1.security_seed import load_seed_table

from .fakes import FakeProvider


def test_default_seed_table_loads_and_validates_against_pydantic_model():
    table = load_seed_table()
    assert "VTI" in table
    vti = table["VTI"]
    assert vti.ticker == "VTI"
    assert vti.asset_class == AssetClass.EQUITY
    bnd = table["BND"]
    assert bnd.asset_class == AssetClass.FIXED_INCOME


def test_custom_seed_path_overrides_default(tmp_path):
    config = tmp_path / "custom_seed.yaml"
    config.write_text(
        """
FAKEETF:
  cusip: "111111111"
  name: "Fake Test ETF"
  security_type: ETF
  asset_class: Equity
"""
    )
    table = load_seed_table(config)
    assert set(table.keys()) == {"FAKEETF"}
    assert table["FAKEETF"].cusip == "111111111"


def test_registry_resolves_seeded_ticker_without_any_provider_call():
    """The actual point of seeding: zero API calls, zero needs_review
    flag, for a security a human has already verified."""
    provider = FakeProvider(profiles={})  # would fail every lookup if hit
    registry = SecurityMasterRegistry(provider)  # default (real) seed table

    entry = registry.resolve("VTI")
    assert entry is not None
    assert entry.needs_review is False
    assert entry.review_reason is None
    assert entry.security.asset_class == AssetClass.EQUITY
    assert provider.ticker_lookup_calls == []  # never touched the provider


def test_registry_resolves_seeded_cusip_without_any_provider_call():
    provider = FakeProvider(profiles={}, cusip_map={})
    registry = SecurityMasterRegistry(provider)

    entry = registry.resolve("922908363")  # VTI's seeded CUSIP
    assert entry is not None
    assert entry.security.ticker == "VTI"
    assert provider.cusip_lookup_calls == []


def test_seeded_bond_etf_has_correct_asset_class_with_no_heuristic_flag():
    """Contrast with the heuristic path (test_security_resolution.py),
    where an unseeded bond ETF is still flagged needs_review=True even
    when the name heuristic gets it right. A seeded entry never is."""
    provider = FakeProvider(profiles={})
    registry = SecurityMasterRegistry(provider)

    entry = registry.resolve("BND")
    assert entry.security.asset_class == AssetClass.FIXED_INCOME
    assert entry.needs_review is False


def test_unseeded_ticker_still_falls_through_to_provider():
    """Confirms seeding doesn't somehow short-circuit resolution for
    everything — only tickers actually in the table."""
    provider = FakeProvider()  # AAPL is in the fake's own default profiles
    registry = SecurityMasterRegistry(provider)  # real seed table (no AAPL in it)

    entry = registry.resolve("AAPL")
    assert entry is not None
    assert provider.ticker_lookup_calls == ["AAPL"]  # had to hit the provider
