"""
Financial Modeling Prep (FMP) provider.

Free tier: 250 calls/day, no card required, documented REST API (as
opposed to Yahoo Finance's undocumented cookie+crumb-gated endpoint,
which breaks without notice and rate-limits aggressively — not a
foundation to build a compliance product's classification pipeline on).

Sign up for a free API key at https://financialmodelingprep.com and set
it as the FMP_API_KEY environment variable. This client raises clearly
at call time (not at import time) if the key is missing, so importing
this module in an offline/test context never requires a key.

Resilience: a single security's lookup failure must never crash an
ingestion run. Transient network errors and HTTP 429 (rate limit) are
retried with backoff; if they persist, they raise FMPRateLimitError /
FMPProviderError (both ProviderLookupError subclasses), which
SecurityMasterRegistry catches and turns into an unresolved-security
entry rather than an unhandled exception. An invalid/missing API key
(401) raises FMPAuthError (a ProviderAuthError subclass) instead, which
deliberately propagates uncaught — that's a systemic problem, not a
per-security one.

IMPORTANT — sandbox note: this client cannot be exercised against the
live API from within this build environment (network egress here is
restricted to package registries only). The endpoint shapes below are
built from FMP's current documented /stable/ API and a verified sample
response, and covered by tests using mocked HTTP responses — but treat
the first real run against your own FMP_API_KEY as the actual
integration test, and diff the live response shape against
ProviderProfile if anything looks off.

/stable/search-cusip's free-tier availability specifically should be
re-verified against FMP's current plan limits before depending on it in
production — legacy docs list it without a clear free/paid marker.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import requests

from .base import ProviderAuthError, ProviderLookupError, ProviderProfile

DEFAULT_BASE_URL = "https://financialmodelingprep.com/stable"
DEFAULT_TIMEOUT = 15
DEFAULT_MAX_RETRIES = 2  # total attempts = 1 + max_retries
DEFAULT_BACKOFF_SECONDS = 1.0


class FMPAuthError(ProviderAuthError):
    pass


class FMPRateLimitError(ProviderLookupError):
    pass


class FMPProviderError(ProviderLookupError):
    pass


class FinancialModelingPrepProvider:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        session: Optional[requests.Session] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        sleep_fn=time.sleep,
    ):
        # Resolved lazily-but-eagerly here (not deferred to call time) so
        # a missing key fails fast at construction, while import of this
        # module itself never touches the environment or the network.
        self._api_key = api_key or os.environ.get("FMP_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep_fn  # injectable so tests don't actually sleep

    def _require_key(self) -> str:
        if not self._api_key:
            raise FMPAuthError(
                "FMP_API_KEY is not set. Sign up for a free key at "
                "https://financialmodelingprep.com and set it as an "
                "environment variable, or pass api_key= explicitly."
            )
        return self._api_key

    def _get_with_retry(self, url: str, params: dict, what: str) -> dict:
        """
        Shared GET-with-retry logic for both endpoints. Raises FMPAuthError
        immediately on 401 (never retried — a bad key won't fix itself).
        Retries transient network errors and 429s up to max_retries times
        with linear backoff, then raises FMPRateLimitError / FMPProviderError.
        """
        last_network_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                last_network_error = e
                if attempt < self.max_retries:
                    self._sleep(self.backoff_seconds * (attempt + 1))
                    continue
                raise FMPProviderError(
                    f"Network error {what} after {self.max_retries + 1} attempt(s): {e}"
                ) from e

            if resp.status_code == 401:
                raise FMPAuthError(f"FMP rejected the API key (HTTP 401) {what}")

            if resp.status_code == 429:
                if attempt < self.max_retries:
                    self._sleep(self.backoff_seconds * (attempt + 1))
                    continue
                raise FMPRateLimitError(
                    f"FMP rate limit (429) persisted after {self.max_retries + 1} attempt(s) {what}"
                )

            if resp.status_code >= 400:
                raise FMPProviderError(f"FMP returned HTTP {resp.status_code} {what}")

            return resp.json()

        # Unreachable in practice (loop always returns or raises), but keeps
        # type checkers happy and fails loudly instead of returning None.
        raise FMPProviderError(f"Exhausted retries {what}: {last_network_error}")

    def lookup_by_ticker(self, ticker: str) -> Optional[ProviderProfile]:
        api_key = self._require_key()
        data = self._get_with_retry(
            f"{self.base_url}/profile",
            {"symbol": ticker, "apikey": api_key},
            what=f"looking up ticker {ticker!r}",
        )
        if not data:
            return None
        record = data[0]
        return ProviderProfile(
            symbol=record.get("symbol", ticker),
            name=record.get("companyName") or ticker,
            cusip=record.get("cusip") or None,
            sector=record.get("sector") or None,
            industry=record.get("industry") or None,
            country=record.get("country") or None,
            is_etf=bool(record.get("isEtf", False)),
            is_fund=bool(record.get("isFund", False)),
            exchange=record.get("exchange") or None,
            is_actively_trading=record.get("isActivelyTrading"),
        )

    def resolve_cusip_to_ticker(self, cusip: str) -> Optional[str]:
        api_key = self._require_key()
        data = self._get_with_retry(
            f"{self.base_url}/search-cusip",
            {"cusip": cusip, "apikey": api_key},
            what=f"looking up CUSIP {cusip!r}",
        )
        if not data:
            return None
        return data[0].get("symbol") or None
