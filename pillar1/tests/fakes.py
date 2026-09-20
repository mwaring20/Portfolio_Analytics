"""
Fake SecurityLookupProvider for tests.

Deliberately illustrative data — these CUSIPs are plausible-looking test
fixtures, not verified against a live registry. Fine for exercising the
resolution pipeline; not to be treated as ground truth.
"""

from __future__ import annotations

from typing import Dict, Optional

from pillar1.providers.base import ProviderProfile


class FakeProvider:
    def __init__(self, profiles: Optional[Dict[str, ProviderProfile]] = None, cusip_map: Optional[Dict[str, str]] = None):
        self.profiles = profiles if profiles is not None else dict(DEFAULT_PROFILES)
        self.cusip_map = cusip_map if cusip_map is not None else dict(DEFAULT_CUSIP_MAP)
        self.ticker_lookup_calls = []
        self.cusip_lookup_calls = []

    def lookup_by_ticker(self, ticker: str) -> Optional[ProviderProfile]:
        self.ticker_lookup_calls.append(ticker)
        return self.profiles.get(ticker)

    def resolve_cusip_to_ticker(self, cusip: str) -> Optional[str]:
        self.cusip_lookup_calls.append(cusip)
        return self.cusip_map.get(cusip)


DEFAULT_PROFILES: Dict[str, ProviderProfile] = {
    "AAPL": ProviderProfile(
        symbol="AAPL", name="Apple Inc.", cusip="037833100", sector="Technology",
        industry="Consumer Electronics", country="US", is_etf=False, is_fund=False,
        exchange="NASDAQ", is_actively_trading=True,
    ),
    "MSFT": ProviderProfile(
        symbol="MSFT", name="Microsoft Corp", cusip="594918104", sector="Technology",
        industry="Software - Infrastructure", country="US", is_etf=False, is_fund=False,
        exchange="NASDAQ", is_actively_trading=True,
    ),
    "VTI": ProviderProfile(
        symbol="VTI", name="Vanguard Total Stock Market ETF", cusip="922908363", sector=None,
        industry=None, country="US", is_etf=True, is_fund=False,
        exchange="NYSEARCA", is_actively_trading=True,
    ),
    "VOO": ProviderProfile(
        symbol="VOO", name="Vanguard S&P 500 ETF", cusip="922908769", sector=None,
        industry=None, country="US", is_etf=True, is_fund=False,
        exchange="NYSEARCA", is_actively_trading=True,
    ),
    "BND": ProviderProfile(
        symbol="BND", name="Vanguard Total Bond Market ETF", cusip="921937835", sector=None,
        industry=None, country="US", is_etf=True, is_fund=False,
        exchange="NASDAQ", is_actively_trading=True,
    ),
    "SCHD": ProviderProfile(
        symbol="SCHD", name="Schwab US Dividend Equity ETF", cusip="808524797", sector=None,
        industry=None, country="US", is_etf=True, is_fund=False,
        exchange="NYSEARCA", is_actively_trading=True,
    ),
}

DEFAULT_CUSIP_MAP: Dict[str, str] = {
    "037833100": "AAPL",
}
