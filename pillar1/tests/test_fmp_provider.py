from unittest.mock import MagicMock

import pytest
import requests

from pillar1.providers.base import ProviderAuthError, ProviderLookupError
from pillar1.providers.fmp import (
    FinancialModelingPrepProvider,
    FMPAuthError,
    FMPProviderError,
    FMPRateLimitError,
)

# Shape verified against FMP's current documented /profile response
# (company name, sector, industry, country, cusip, isEtf, isFund, etc.)
MOCK_PROFILE_RESPONSE = [
    {
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "cusip": "037833100",
        "isin": "US0378331005",
        "exchange": "NASDAQ",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "US",
        "isEtf": False,
        "isFund": False,
        "isActivelyTrading": True,
    }
]


def _mock_resp(status_code, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    return resp


def _mock_session(json_body, status_code=200):
    session = MagicMock()
    session.get.return_value = _mock_resp(status_code, json_body)
    return session


def _sequence_session(items):
    """items: list of Exception instances or (status_code, json_body) tuples,
    consumed in order across successive session.get() calls."""
    session = MagicMock()
    side_effects = [
        item if isinstance(item, Exception) else _mock_resp(*item) for item in items
    ]
    session.get.side_effect = side_effects
    return session


# --- basic behavior (unchanged from before hardening) -----------------------

def test_missing_api_key_raises_at_call_time_not_import_time():
    provider = FinancialModelingPrepProvider(api_key=None, session=_mock_session(MOCK_PROFILE_RESPONSE))
    with pytest.raises(FMPAuthError):
        provider.lookup_by_ticker("AAPL")


def test_fmp_auth_error_is_a_provider_auth_error():
    """Confirms the registry-level ProviderAuthError propagation contract
    actually applies to FMP's own auth error, not just in the abstract."""
    assert issubclass(FMPAuthError, ProviderAuthError)


def test_fmp_rate_limit_and_provider_errors_are_lookup_errors():
    """Confirms these are the catchable-per-security kind, not the
    propagate-and-stop kind."""
    assert issubclass(FMPRateLimitError, ProviderLookupError)
    assert issubclass(FMPProviderError, ProviderLookupError)


def test_lookup_by_ticker_parses_real_response_shape():
    session = _mock_session(MOCK_PROFILE_RESPONSE)
    provider = FinancialModelingPrepProvider(api_key="fake-key", session=session)
    profile = provider.lookup_by_ticker("AAPL")

    assert profile is not None
    assert profile.symbol == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.cusip == "037833100"
    assert profile.sector == "Technology"
    assert profile.country == "US"
    assert profile.is_etf is False
    assert profile.is_fund is False

    session.get.assert_called_once()
    args, kwargs = session.get.call_args
    assert kwargs["params"]["symbol"] == "AAPL"
    assert kwargs["params"]["apikey"] == "fake-key"


def test_lookup_by_ticker_returns_none_on_empty_result():
    session = _mock_session([])
    provider = FinancialModelingPrepProvider(api_key="fake-key", session=session)
    assert provider.lookup_by_ticker("NOTREAL") is None


def test_resolve_cusip_to_ticker():
    session = _mock_session([{"symbol": "AAPL", "name": "Apple Inc."}])
    provider = FinancialModelingPrepProvider(api_key="fake-key", session=session)
    ticker = provider.resolve_cusip_to_ticker("037833100")
    assert ticker == "AAPL"


# --- hardening: retry/backoff and crash-proofing -----------------------------

def test_401_raises_immediately_without_retry():
    session = _sequence_session([(401, {})])
    provider = FinancialModelingPrepProvider(
        api_key="bad-key", session=session, max_retries=2, sleep_fn=lambda s: None
    )
    with pytest.raises(FMPAuthError):
        provider.lookup_by_ticker("AAPL")
    assert session.get.call_count == 1  # never retried — a bad key won't fix itself


def test_500_raises_provider_error_immediately_without_retry():
    """A plain server error isn't a rate limit and isn't a network
    exception — no reason to believe retrying helps, so it's not retried
    (only 429 and network errors are)."""
    session = _sequence_session([(500, {})])
    provider = FinancialModelingPrepProvider(
        api_key="fake-key", session=session, max_retries=2, sleep_fn=lambda s: None
    )
    with pytest.raises(FMPProviderError):
        provider.lookup_by_ticker("AAPL")
    assert session.get.call_count == 1


def test_429_is_retried_and_succeeds_on_second_attempt():
    sleeps = []
    session = _sequence_session([(429, {}), (200, MOCK_PROFILE_RESPONSE)])
    provider = FinancialModelingPrepProvider(
        api_key="fake-key", session=session, max_retries=2, backoff_seconds=0.01, sleep_fn=sleeps.append
    )
    profile = provider.lookup_by_ticker("AAPL")
    assert profile is not None
    assert profile.symbol == "AAPL"
    assert session.get.call_count == 2
    assert len(sleeps) == 1  # backed off exactly once before the retry that succeeded


def test_429_persisting_past_max_retries_raises_rate_limit_error():
    session = _sequence_session([(429, {}), (429, {}), (429, {})])
    provider = FinancialModelingPrepProvider(
        api_key="fake-key", session=session, max_retries=2, backoff_seconds=0.01, sleep_fn=lambda s: None
    )
    with pytest.raises(FMPRateLimitError):
        provider.lookup_by_ticker("AAPL")
    assert session.get.call_count == 3  # 1 initial + 2 retries


def test_network_error_is_retried_and_succeeds():
    session = _sequence_session([requests.exceptions.ConnectionError("boom"), (200, MOCK_PROFILE_RESPONSE)])
    provider = FinancialModelingPrepProvider(
        api_key="fake-key", session=session, max_retries=2, backoff_seconds=0.01, sleep_fn=lambda s: None
    )
    profile = provider.lookup_by_ticker("AAPL")
    assert profile is not None
    assert session.get.call_count == 2


def test_network_error_persisting_past_max_retries_raises_provider_error():
    session = _sequence_session(
        [requests.exceptions.ConnectionError("boom")] * 3
    )
    provider = FinancialModelingPrepProvider(
        api_key="fake-key", session=session, max_retries=2, backoff_seconds=0.01, sleep_fn=lambda s: None
    )
    with pytest.raises(FMPProviderError):
        provider.lookup_by_ticker("AAPL")
    assert session.get.call_count == 3


def test_retry_and_backoff_applies_identically_to_cusip_search():
    session = _sequence_session([(429, {}), (200, [{"symbol": "AAPL"}])])
    provider = FinancialModelingPrepProvider(
        api_key="fake-key", session=session, max_retries=2, backoff_seconds=0.01, sleep_fn=lambda s: None
    )
    ticker = provider.resolve_cusip_to_ticker("037833100")
    assert ticker == "AAPL"
    assert session.get.call_count == 2
