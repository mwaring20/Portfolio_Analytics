import json

from pillar1.providers.caching import CachingProvider

from .fakes import FakeProvider


def test_first_lookup_hits_wrapped_provider_and_caches(tmp_path):
    cache_path = tmp_path / "security_cache.json"
    fake = FakeProvider()
    provider = CachingProvider(fake, cache_path)

    profile = provider.lookup_by_ticker("AAPL")
    assert profile is not None
    assert fake.ticker_lookup_calls == ["AAPL"]
    assert cache_path.exists()


def test_second_lookup_within_same_instance_does_not_hit_wrapped_provider(tmp_path):
    cache_path = tmp_path / "security_cache.json"
    fake = FakeProvider()
    provider = CachingProvider(fake, cache_path)

    provider.lookup_by_ticker("AAPL")
    provider.lookup_by_ticker("AAPL")
    assert fake.ticker_lookup_calls == ["AAPL"]  # only once


def test_cache_persists_across_provider_instances(tmp_path):
    """The actual point: a fresh process/run reusing the same cache file
    should not re-spend API quota on a security already resolved."""
    cache_path = tmp_path / "security_cache.json"
    fake1 = FakeProvider()
    provider1 = CachingProvider(fake1, cache_path)
    provider1.lookup_by_ticker("AAPL")

    # Simulate a brand-new run: fresh provider instance, fresh underlying
    # fake (so if the cache didn't work, this fake would have to be hit).
    fake2 = FakeProvider()
    provider2 = CachingProvider(fake2, cache_path)
    profile = provider2.lookup_by_ticker("AAPL")

    assert profile is not None
    assert profile.symbol == "AAPL"
    assert fake2.ticker_lookup_calls == []  # never touched — served from disk cache


def test_cusip_resolution_also_cached_across_instances(tmp_path):
    cache_path = tmp_path / "security_cache.json"
    fake1 = FakeProvider()
    provider1 = CachingProvider(fake1, cache_path)
    provider1.resolve_cusip_to_ticker("037833100")

    fake2 = FakeProvider()
    provider2 = CachingProvider(fake2, cache_path)
    ticker = provider2.resolve_cusip_to_ticker("037833100")

    assert ticker == "AAPL"
    assert fake2.cusip_lookup_calls == []


def test_negative_result_is_never_cached(tmp_path):
    """A miss must always be retried next time, not permanently
    remembered as a miss — the security might get listed later, or last
    time's miss might have been a transient issue."""
    cache_path = tmp_path / "security_cache.json"
    fake = FakeProvider(profiles={})  # nothing resolves
    provider = CachingProvider(fake, cache_path)

    result1 = provider.lookup_by_ticker("NOTFOUND")
    result2 = provider.lookup_by_ticker("NOTFOUND")

    assert result1 is None
    assert result2 is None
    # Hit the wrapped provider both times — no cache entry for a miss.
    assert fake.ticker_lookup_calls == ["NOTFOUND", "NOTFOUND"]


def test_cache_file_is_readable_json(tmp_path):
    cache_path = tmp_path / "security_cache.json"
    provider = CachingProvider(FakeProvider(), cache_path)
    provider.lookup_by_ticker("AAPL")
    provider.resolve_cusip_to_ticker("037833100")

    with open(cache_path) as f:
        raw = json.load(f)
    assert "AAPL" in raw["profiles"]
    assert raw["cusip_map"]["037833100"] == "AAPL"


def test_cache_stats(tmp_path):
    provider = CachingProvider(FakeProvider(), tmp_path / "security_cache.json")
    provider.lookup_by_ticker("AAPL")
    provider.resolve_cusip_to_ticker("037833100")
    stats = provider.cache_stats()
    assert stats["cached_profiles"] == 1
    assert stats["cached_cusip_mappings"] == 1
