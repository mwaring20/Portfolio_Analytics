"""
Provider-agnostic interface for security metadata lookups.

security_resolution.py depends only on this interface, never on a
specific vendor's HTTP client directly. That means:
  - tests can inject a fake/mock provider with zero network dependency
  - swapping Financial Modeling Prep for a different vendor later (or
    adding a second provider as a fallback when the first misses) touches
    only providers/, not the resolution or classification logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


class ProviderAuthError(RuntimeError):
    """Raised when provider credentials are missing or invalid. This is a
    systemic configuration problem, not a per-security failure — every
    subsequent lookup will fail the same way. Callers (e.g.
    SecurityMasterRegistry) should let this propagate rather than
    silently marking securities unresolved one by one, which would bury
    the real problem under hundreds of misleading "not found" entries."""


class ProviderLookupError(RuntimeError):
    """Raised for a single-security lookup failure that should NOT crash
    a whole ingestion run: a transient network error, a persisted rate
    limit, or an unexpected response for one symbol. Callers should
    catch this, record the security as unresolved with the message as
    the reason, and continue with the rest of the batch."""


@dataclass
class ProviderProfile:
    """Vendor-neutral shape of a security profile lookup result. Every
    concrete provider maps its own raw response onto this before handing
    it back — so classify_asset_class() and the rest of the resolution
    pipeline never see a vendor-specific field name."""

    symbol: str
    name: str
    cusip: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    country: Optional[str]
    is_etf: bool
    is_fund: bool
    exchange: Optional[str]
    is_actively_trading: Optional[bool]


class SecurityLookupProvider(Protocol):
    def lookup_by_ticker(self, ticker: str) -> Optional[ProviderProfile]:
        """Return a profile for a ticker symbol, or None if not found."""
        ...

    def resolve_cusip_to_ticker(self, cusip: str) -> Optional[str]:
        """Resolve a CUSIP to a ticker symbol, or None if not found. Used
        for the CUSIP-first identifier fallback chain — CUSIP search
        typically returns a bare symbol/name match, not a full profile,
        so callers resolve the ticker and then call lookup_by_ticker."""
        ...
