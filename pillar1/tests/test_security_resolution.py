import pandas as pd
import pytest

from pillar1.providers.base import ProviderAuthError, ProviderLookupError, ProviderProfile
from pillar1.security_master import AssetClass, SecurityType
from pillar1.security_resolution import (
    SecurityMasterRegistry,
    classify_asset_class,
    looks_like_cusip,
    resolve_transactions_securities,
)

from .fakes import FakeProvider


class RaisingProvider:
    """Provider stub that raises a chosen exception for specific
    identifiers — used to test the registry's crash-proofing without any
    real HTTP layer involved."""

    def __init__(self, raise_on_ticker=None, raise_on_cusip=None):
        self.raise_on_ticker = raise_on_ticker or {}
        self.raise_on_cusip = raise_on_cusip or {}

    def lookup_by_ticker(self, ticker):
        if ticker in self.raise_on_ticker:
            raise self.raise_on_ticker[ticker]
        return None

    def resolve_cusip_to_ticker(self, cusip):
        if cusip in self.raise_on_cusip:
            raise self.raise_on_cusip[cusip]
        return None


def test_looks_like_cusip():
    assert looks_like_cusip("037833100") is True
    assert looks_like_cusip("AAPL") is False
    assert looks_like_cusip("922908363") is True


def test_classify_plain_equity_high_confidence():
    profile = ProviderProfile(
        symbol="AAPL", name="Apple Inc.", cusip="037833100", sector="Technology",
        industry="Consumer Electronics", country="US", is_etf=False, is_fund=False,
        exchange="NASDAQ", is_actively_trading=True,
    )
    sec_type, asset_class, sub, needs_review, reason = classify_asset_class(profile)
    assert sec_type == SecurityType.EQUITY
    assert asset_class == AssetClass.EQUITY
    assert needs_review is False
    assert reason is None


def test_classify_etf_flagged_needs_review():
    profile = ProviderProfile(
        symbol="VTI", name="Vanguard Total Stock Market ETF", cusip="922908363", sector=None,
        industry=None, country="US", is_etf=True, is_fund=False,
        exchange="NYSEARCA", is_actively_trading=True,
    )
    sec_type, asset_class, sub, needs_review, reason = classify_asset_class(profile)
    assert sec_type == SecurityType.ETF
    assert needs_review is True
    assert "heuristically" in reason.lower()


def test_classify_bond_etf_name_heuristic():
    profile = ProviderProfile(
        symbol="BND", name="Vanguard Total Bond Market ETF", cusip="921937835", sector=None,
        industry=None, country="US", is_etf=True, is_fund=False,
        exchange="NASDAQ", is_actively_trading=True,
    )
    sec_type, asset_class, sub, needs_review, reason = classify_asset_class(profile)
    assert asset_class == AssetClass.FIXED_INCOME
    assert needs_review is True  # still flagged even though the heuristic got it right


def test_classify_mutual_fund_always_flagged():
    profile = ProviderProfile(
        symbol="VTSAX", name="Vanguard Total Stock Market Index Fund", cusip=None, sector=None,
        industry=None, country="US", is_etf=False, is_fund=True,
        exchange=None, is_actively_trading=True,
    )
    sec_type, asset_class, sub, needs_review, reason = classify_asset_class(profile)
    assert sec_type == SecurityType.MUTUAL_FUND
    assert asset_class == AssetClass.PENDING_REVIEW  # honest sentinel, not a guessed real class
    assert needs_review is True
    assert "morningstar" in reason.lower()


def test_registry_resolves_by_ticker_and_caches():
    provider = FakeProvider()
    registry = SecurityMasterRegistry(provider, seed_table={})

    entry1 = registry.resolve("AAPL")
    entry2 = registry.resolve("AAPL")

    assert entry1 is not None
    assert entry1.security.security_id == "037833100"  # CUSIP-first security_id
    assert entry1.security.ticker == "AAPL"
    assert entry1.needs_review is False
    # Cached — provider only hit once.
    assert provider.ticker_lookup_calls == ["AAPL"]
    assert entry2.security.security_id == entry1.security.security_id


def test_registry_resolves_cusip_first_then_ticker():
    provider = FakeProvider()
    registry = SecurityMasterRegistry(provider, seed_table={})

    entry = registry.resolve("037833100")  # a CUSIP, not a ticker
    assert entry is not None
    assert entry.security.ticker == "AAPL"
    assert provider.cusip_lookup_calls == ["037833100"]
    assert provider.ticker_lookup_calls == ["AAPL"]


def test_registry_records_unresolved_when_no_profile_found():
    provider = FakeProvider()
    registry = SecurityMasterRegistry(provider, seed_table={})

    entry = registry.resolve("ZZZNOTREAL")
    assert entry is None
    assert len(registry.unresolved) == 1
    assert registry.unresolved[0].raw_identifier == "ZZZNOTREAL"


def test_registry_records_unresolved_when_cusip_has_no_ticker_match():
    provider = FakeProvider()
    registry = SecurityMasterRegistry(provider, seed_table={})

    entry = registry.resolve("999999999")  # cusip-shaped, not in cusip_map
    assert entry is None
    assert len(registry.unresolved) == 1
    assert "cusip" in registry.unresolved[0].reason.lower()


def test_resolve_transactions_securities_against_a_dataframe():
    provider = FakeProvider()
    registry = SecurityMasterRegistry(provider, seed_table={})
    txns = pd.DataFrame(
        {
            "security_id": ["AAPL", "MSFT", "AAPL", None, "VTI"],
        }
    )
    resolved, unresolved = resolve_transactions_securities(txns, registry)
    assert set(resolved.keys()) == {"AAPL", "MSFT", "VTI"}
    assert unresolved == []
    # AAPL only looked up once despite appearing twice in the DataFrame.
    assert provider.ticker_lookup_calls.count("AAPL") == 1


# --- hardening: crash-proofing (ProviderLookupError vs ProviderAuthError) ---


def test_registry_catches_lookup_error_from_ticker_search_as_unresolved():
    provider = RaisingProvider(raise_on_ticker={"AAPL": ProviderLookupError("rate limited after retries")})
    registry = SecurityMasterRegistry(provider, seed_table={})

    entry = registry.resolve("AAPL")
    assert entry is None
    assert len(registry.unresolved) == 1
    assert "rate limited after retries" in registry.unresolved[0].reason


def test_registry_catches_lookup_error_from_cusip_search_as_unresolved():
    provider = RaisingProvider(raise_on_cusip={"037833100": ProviderLookupError("transient network error")})
    registry = SecurityMasterRegistry(provider, seed_table={})

    entry = registry.resolve("037833100")  # cusip-shaped
    assert entry is None
    assert len(registry.unresolved) == 1
    assert "transient network error" in registry.unresolved[0].reason


def test_registry_propagates_auth_error_instead_of_swallowing_it():
    """A bad/missing API key affects every subsequent lookup — this must
    NOT be caught and quietly turned into 'unresolved', which would bury
    a systemic config problem under a pile of misleading per-security
    entries. It should stop the run."""
    provider = RaisingProvider(raise_on_ticker={"AAPL": ProviderAuthError("bad API key")})
    registry = SecurityMasterRegistry(provider, seed_table={})

    with pytest.raises(ProviderAuthError):
        registry.resolve("AAPL")
    assert registry.unresolved == []  # never silently recorded


def test_registry_propagates_auth_error_from_cusip_path_too():
    provider = RaisingProvider(raise_on_cusip={"037833100": ProviderAuthError("bad API key")})
    registry = SecurityMasterRegistry(provider, seed_table={})

    with pytest.raises(ProviderAuthError):
        registry.resolve("037833100")


def test_one_failed_security_does_not_stop_the_rest_of_the_batch():
    """The actual point of this hardening: one bad symbol shouldn't take
    down an entire household's resolution run."""
    provider = RaisingProvider(raise_on_ticker={"BADTICKER": ProviderLookupError("boom")})
    # Merge in the default good profiles so the rest resolve normally.
    provider.raise_on_ticker = {"BADTICKER": ProviderLookupError("boom")}
    fake_good = FakeProvider()

    class MixedProvider:
        def lookup_by_ticker(self, ticker):
            if ticker == "BADTICKER":
                raise ProviderLookupError("boom")
            return fake_good.lookup_by_ticker(ticker)

        def resolve_cusip_to_ticker(self, cusip):
            return fake_good.resolve_cusip_to_ticker(cusip)

    registry = SecurityMasterRegistry(MixedProvider(), seed_table={})
    txns = pd.DataFrame({"security_id": ["AAPL", "BADTICKER", "MSFT"]})
    resolved, unresolved = resolve_transactions_securities(txns, registry)

    assert set(resolved.keys()) == {"AAPL", "MSFT"}
    assert len(unresolved) == 1
    assert unresolved[0].raw_identifier == "BADTICKER"

