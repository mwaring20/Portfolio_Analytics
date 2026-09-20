"""
Phase 1 — security resolution and classification.

Implements two architecture-doc requirements directly:
  - "Key by a stable identifier with a fallback chain — CUSIP first
    (tickers can change on corporate actions), ticker as fallback."
  - "Never let an unclassified security default silently to 'other' —
    route to a manual-review queue instead."

Confidence-aware classification: an equity's sector/geography comes
straight from the provider with high confidence. An ETF's asset_class
is *not* reliably available from a basic profile endpoint (see
providers/fmp.py's module docstring) — so it's filled with a best-guess
heuristic AND explicitly flagged for human confirmation, rather than
either leaving it null (violates the Pydantic non-null contract) or
asserting it with false confidence. A mutual fund is routed to manual
review outright, per the architecture doc's explicit instruction that
funds "often need Morningstar-style category mapping instead" of
automated lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .providers.base import ProviderLookupError, ProviderProfile, SecurityLookupProvider
from .security_master import AssetClass, SecurityMaster, SecurityType
from .security_seed import load_seed_table

# CUSIPs are 9 alphanumeric characters. Tickers in the fixtures/target
# custodians are short (<=5) uppercase-letter strings. This is a
# heuristic, not a guarantee — a false negative just falls through to
# the ticker path and fails lookup there, which is a safe failure mode
# (goes to unresolved, not misclassified).
_CUSIP_SHAPE = re.compile(r"^[0-9A-Z]{9}$")

# Name substrings suggesting a fixed-income ETF, used only as a
# low-confidence heuristic — every ETF classified this way is flagged
# needs_review=True regardless of which branch it takes.
_FIXED_INCOME_NAME_HINTS = ("BOND", "FIXED INCOME", "TREASURY", "AGGREGATE")


def looks_like_cusip(identifier: str) -> bool:
    return bool(_CUSIP_SHAPE.match(identifier.strip().upper()))


@dataclass
class SecurityMasterEntry:
    """Wraps a resolved SecurityMaster with the review metadata that the
    architecture doc's exact SecurityMaster schema doesn't itself carry
    (keeping that Pydantic model schema-pure, matching the doc field for
    field) — the review workflow lives at this registry layer instead."""

    security: SecurityMaster
    needs_review: bool
    review_reason: Optional[str]


@dataclass
class UnresolvedSecurity:
    """A security that could not be classified at all — never becomes a
    SecurityMaster; sits in the manual-review queue until a human
    resolves it."""

    raw_identifier: str
    reason: str


def classify_asset_class(
    profile: ProviderProfile,
) -> Tuple[SecurityType, AssetClass, Optional[str], bool, Optional[str]]:
    """
    Returns (security_type, asset_class, sub_asset_class, needs_review, review_reason).
    """
    if profile.is_fund:
        # Mutual funds: architecture doc explicitly says these need
        # Morningstar-style category mapping, not automated ticker
        # lookup. PENDING_REVIEW (not a guessed real asset class) makes
        # this honest — a downstream consumer that ignores needs_review
        # and reads asset_class directly gets an obviously-not-real
        # value instead of a plausible-looking wrong one.
        return (
            SecurityType.MUTUAL_FUND,
            AssetClass.PENDING_REVIEW,
            None,
            True,
            "Mutual fund — requires Morningstar-style category mapping, not auto-classified from a basic profile lookup",
        )

    if profile.is_etf:
        name_upper = (profile.name or "").upper()
        if any(hint in name_upper for hint in _FIXED_INCOME_NAME_HINTS):
            asset_class = AssetClass.FIXED_INCOME
        else:
            asset_class = AssetClass.EQUITY
        return (
            SecurityType.ETF,
            asset_class,
            profile.industry,
            True,
            "ETF asset_class inferred heuristically from fund name (no true asset-class field available "
            "from this provider's basic profile endpoint) — confirm against the fund's fact sheet",
        )

    # Plain equity — provider sector/geography trusted directly.
    return (SecurityType.EQUITY, AssetClass.EQUITY, profile.sector, False, None)


class SecurityMasterRegistry:
    """
    Caches resolved securities (by both CUSIP and ticker, per the
    architecture doc's fallback chain) and accumulates an unresolved
    list for anything that couldn't be classified.

    Pre-warms its cache from the manually-verified seed table (see
    security_seed.py) at construction time — a seed-table entry costs
    zero API calls and carries needs_review=False, since it represents a
    human-confirmed classification rather than an automated guess. This
    reuses the same cache-then-provider resolve() path rather than
    adding a separate lookup branch: a seeded ticker is simply already
    in the cache before the first .resolve() call.

    Pass seed_table={} to disable seeding entirely (e.g. in tests that
    want a clean slate); omit it to use the bundled default table.
    """

    def __init__(
        self,
        provider: SecurityLookupProvider,
        seed_table: Optional[Dict[str, SecurityMaster]] = None,
    ):
        self.provider = provider
        self._by_cusip: Dict[str, SecurityMasterEntry] = {}
        self._by_ticker: Dict[str, SecurityMasterEntry] = {}
        self.unresolved: List[UnresolvedSecurity] = []

        seed_table = seed_table if seed_table is not None else load_seed_table()
        for security in seed_table.values():
            self._register(SecurityMasterEntry(security=security, needs_review=False, review_reason=None))

    def _register(self, entry: SecurityMasterEntry) -> None:
        if entry.security.cusip:
            self._by_cusip[entry.security.cusip] = entry
        if entry.security.ticker:
            self._by_ticker[entry.security.ticker] = entry

    def _build_entry_from_profile(self, profile: ProviderProfile) -> SecurityMasterEntry:
        security_type, asset_class, sub_asset_class, needs_review, reason = classify_asset_class(profile)
        # CUSIP-first identifier: use it as the stable security_id when
        # the provider gave us one, falling back to ticker — exactly the
        # fallback chain the architecture doc calls for.
        security_id = profile.cusip or profile.symbol
        security = SecurityMaster(
            security_id=security_id,
            cusip=profile.cusip,
            ticker=profile.symbol,
            name=profile.name,
            security_type=security_type,
            asset_class=asset_class,
            sub_asset_class=sub_asset_class,
            sector=profile.sector,
            geography=profile.country,
            expense_ratio=None,  # not available from this provider's basic profile endpoint
        )
        return SecurityMasterEntry(security=security, needs_review=needs_review, review_reason=reason)

    def resolve(self, raw_identifier: str) -> Optional[SecurityMasterEntry]:
        """
        Resolve one raw identifier (as it appears in transactions_df's
        security_id column) to a SecurityMasterEntry, or None if it
        couldn't be classified (in which case it's recorded in
        self.unresolved). Cached — a given identifier only hits the
        provider once per registry instance.
        """
        raw_identifier = raw_identifier.strip()

        if raw_identifier in self._by_cusip:
            return self._by_cusip[raw_identifier]
        if raw_identifier in self._by_ticker:
            return self._by_ticker[raw_identifier]

        ticker = raw_identifier
        if looks_like_cusip(raw_identifier):
            try:
                resolved_ticker = self.provider.resolve_cusip_to_ticker(raw_identifier)
            except ProviderLookupError as e:
                # Per-security failure (rate limit, transient network
                # error, unexpected response) — record and move on.
                # ProviderAuthError is deliberately NOT caught here: a bad
                # API key affects every lookup, so it should propagate and
                # stop the run rather than quietly filling the unresolved
                # queue with hundreds of misleading entries.
                self.unresolved.append(
                    UnresolvedSecurity(
                        raw_identifier=raw_identifier,
                        reason=f"CUSIP lookup failed: {e}",
                    )
                )
                return None
            if resolved_ticker is None:
                self.unresolved.append(
                    UnresolvedSecurity(
                        raw_identifier=raw_identifier,
                        reason="Looked like a CUSIP but no matching ticker was found via CUSIP search",
                    )
                )
                return None
            ticker = resolved_ticker

        try:
            profile = self.provider.lookup_by_ticker(ticker)
        except ProviderLookupError as e:
            self.unresolved.append(
                UnresolvedSecurity(
                    raw_identifier=raw_identifier,
                    reason=f"Provider lookup failed: {e}",
                )
            )
            return None
        if profile is None:
            self.unresolved.append(
                UnresolvedSecurity(
                    raw_identifier=raw_identifier,
                    reason=f"No profile found for ticker {ticker!r}",
                )
            )
            return None

        entry = self._build_entry_from_profile(profile)
        self._register(entry)
        return entry


def resolve_transactions_securities(
    transactions_df: pd.DataFrame, registry: SecurityMasterRegistry
) -> Tuple[Dict[str, SecurityMasterEntry], List[UnresolvedSecurity]]:
    """
    Resolve every distinct non-null security_id appearing in a Phase 0
    transactions_df against the given registry.

    Deliberately does NOT mutate transactions_df — Phase 0's contract
    stays stable and independently testable; this returns a mapping
    callers can join against security_id when they need classification
    data, keeping each phase's DataFrame contract from Phase 0 intact
    for Phase 2+ to reuse as-is.
    """
    distinct_ids = sorted(transactions_df["security_id"].dropna().unique())
    resolved: Dict[str, SecurityMasterEntry] = {}
    for raw_id in distinct_ids:
        entry = registry.resolve(raw_id)
        if entry is not None:
            resolved[raw_id] = entry
    return resolved, registry.unresolved
