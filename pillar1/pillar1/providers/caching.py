"""
On-disk cache wrapping any SecurityLookupProvider.

Why this matters concretely: FMP's free tier is 250 calls/day. Without
caching, re-running ingestion against the same client's transaction
history every weekend re-spends quota on securities that were already
resolved last time and haven't changed. This wraps ANY provider
(implements the same SecurityLookupProvider protocol) so it's a drop-in
— `CachingProvider(FinancialModelingPrepProvider())` instead of the
raw provider, nothing else changes.

Deliberately only caches POSITIVE results (a found profile / a resolved
ticker). A "not found" is never cached: the security might get listed
later, or the miss might have been a transient issue that
ProviderLookupError didn't happen to catch — caching a negative result
would make that failure permanent until someone thinks to clear the
cache file, which is a worse failure mode than one extra API call next
time.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Union

from .base import ProviderProfile, SecurityLookupProvider


class CachingProvider:
    def __init__(self, wrapped: SecurityLookupProvider, cache_path: Union[str, Path]):
        self.wrapped = wrapped
        self.cache_path = Path(cache_path)
        self._profiles: Dict[str, ProviderProfile] = {}
        self._cusip_map: Dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with open(self.cache_path, "r") as f:
            raw = json.load(f)
        self._profiles = {
            ticker: ProviderProfile(**fields) for ticker, fields in raw.get("profiles", {}).items()
        }
        self._cusip_map = raw.get("cusip_map", {})

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "profiles": {ticker: asdict(profile) for ticker, profile in self._profiles.items()},
            "cusip_map": self._cusip_map,
        }
        # Write to a temp file then rename, so a crash mid-write never
        # corrupts a previously-good cache file.
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(raw, f, indent=2)
        tmp_path.replace(self.cache_path)

    def lookup_by_ticker(self, ticker: str) -> Optional[ProviderProfile]:
        if ticker in self._profiles:
            return self._profiles[ticker]
        profile = self.wrapped.lookup_by_ticker(ticker)
        if profile is not None:
            self._profiles[ticker] = profile
            self._save()
        return profile

    def resolve_cusip_to_ticker(self, cusip: str) -> Optional[str]:
        if cusip in self._cusip_map:
            return self._cusip_map[cusip]
        ticker = self.wrapped.resolve_cusip_to_ticker(cusip)
        if ticker is not None:
            self._cusip_map[cusip] = ticker
            self._save()
        return ticker

    def cache_stats(self) -> Dict[str, int]:
        return {"cached_profiles": len(self._profiles), "cached_cusip_mappings": len(self._cusip_map)}
