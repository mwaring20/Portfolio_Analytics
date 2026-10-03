"""
Price data provider for Phase 2 valuation engine.

Follows the same pattern as SecurityLookupProvider: abstract base class
with concrete implementations for different data sources. The default
implementation uses Financial Modeling Prep (FMP) historical price data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from decimal import Decimal
from typing import List, Optional

import pandas as pd


class PriceDataError(Exception):
    """Base exception for price data provider errors."""
    pass


class ProviderAuthError(PriceDataError):
    """API key authentication or authorization failure."""
    pass


class PriceNotFoundError(PriceDataError):
    """Price data not available for the requested security/date range."""
    pass


class PriceProvider(ABC):
    """Abstract base class for price data providers."""

    @abstractmethod
    def get_historical_prices(
        self,
        security_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a security.

        Returns a DataFrame with columns:
            - date (date)
            - close_price (Decimal)
            - adj_close (Decimal)

        Raises:
            ProviderAuthError: if API key is invalid/missing
            PriceNotFoundError: if no data available for the security
            PriceDataError: for other provider-specific errors
        """
        pass


class FinancialModelingPrepPriceProvider(PriceProvider):
    """
    FMP historical price data provider.

    Uses the /historical-price-full endpoint to fetch daily OHLCV data
    with adjusted close for total return calculations.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FMP price provider.

        api_key: FMP API key. If None, reads from FMP_API_KEY environment variable.
        """
        import os

        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ProviderAuthError(
                "FMP API key not provided and FMP_API_KEY environment variable not set"
            )
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_historical_prices(
        self,
        security_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical prices from FMP.

        FMP returns all available history; we filter to the requested range.
        """
        import requests

        url = f"{self.base_url}/historical-price-full/{security_id}"
        params = {"apikey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            raise PriceDataError(f"Network error fetching prices for {security_id}: {e}")

        if response.status_code == 401:
            raise ProviderAuthError("Invalid FMP API key")
        if response.status_code == 404:
            raise PriceNotFoundError(f"No price data found for security {security_id}")
        if response.status_code != 200:
            raise PriceDataError(
                f"FMP API returned {response.status_code} for {security_id}: {response.text}"
            )

        data = response.json()
        if not data or "historical" not in data:
            raise PriceNotFoundError(f"No price data found for security {security_id}")

        # Parse historical price data
        price_rows = []
        for item in data["historical"]:
            price_date = pd.to_datetime(item["date"]).date()
            if start_date <= price_date <= end_date:
                price_rows.append(
                    {
                        "date": price_date,
                        "close_price": Decimal(str(item["close"])),
                        "adj_close": Decimal(str(item.get("adjClose", item["close"]))),
                    }
                )

        if not price_rows:
            raise PriceNotFoundError(
                f"No price data for {security_id} in range {start_date} to {end_date}"
            )

        df = pd.DataFrame(price_rows)
        df = df.sort_values("date").reset_index(drop=True)
        return df


class BenchmarkPriceProvider(PriceProvider):
    """
    Specialized price provider for benchmark indices.

    Uses the same FMP API but specifically for index/ETF data used as benchmarks.
    This is a thin wrapper around FinancialModelingPrepPriceProvider with
    benchmark-specific error handling.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the benchmark price provider."""
        self._delegate = FinancialModelingPrepPriceProvider(api_key)

    def get_historical_prices(
        self,
        security_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical prices for a benchmark index.

        Delegates to FMP provider but with benchmark-specific error messages.
        """
        try:
            return self._delegate.get_historical_prices(security_id, start_date, end_date)
        except PriceNotFoundError as e:
            raise PriceNotFoundError(
                f"Benchmark index data not available for {security_id}: {e}"
            ) from e


class CachingPriceProvider(PriceProvider):
    """
    Caching wrapper for price providers.

    Caches price data on disk to avoid repeated API calls for the same
    security and date range. Only successful results are cached; failures
    are always retried.
    """

    def __init__(
        self,
        provider: PriceProvider,
        cache_path: str = "price_cache.json",
    ):
        """
        Initialize the caching provider.

        provider: underlying PriceProvider to wrap
        cache_path: path to JSON cache file
        """
        self.provider = provider
        self.cache_path = cache_path
        self._cache: dict = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk if it exists."""
        import json
        from pathlib import Path

        cache_file = Path(self.cache_path)
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupt cache - start fresh
                return {}
        return {}

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        import json
        from pathlib import Path

        cache_file = Path(self.cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, "w") as f:
            json.dump(self._cache, f, indent=2, default=str)

    def _cache_key(self, security_id: str, start_date: date, end_date: date) -> str:
        """Generate a cache key for the request."""
        return f"{security_id}|{start_date}|{end_date}"

    def get_historical_prices(
        self,
        security_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get prices from cache if available, otherwise fetch from provider.

        Cached data is stored as a list of dicts and reconstructed into a DataFrame.
        """
        cache_key = self._cache_key(security_id, start_date, end_date)

        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            df = pd.DataFrame(cached_data)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["close_price"] = df["close_price"].apply(Decimal)
            df["adj_close"] = df["adj_close"].apply(Decimal)
            return df.sort_values("date").reset_index(drop=True)

        # Cache miss - fetch from provider
        df = self.provider.get_historical_prices(security_id, start_date, end_date)

        # Cache the result
        self._cache[cache_key] = df.to_dict(orient="records")
        self._save_cache()

        return df
