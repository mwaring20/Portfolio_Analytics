"""
Phase 1 — Security Master reference entity.

Low-cardinality, relationship-bearing, validated at creation time — per
the architecture doc's design principle, this is a Pydantic model, not a
DataFrame. The field set matches the architecture doc's SecurityMaster
definition exactly; nothing added or removed here, so this stays the one
place Phase 2+ can trust for "what does this schema actually promise."

Because asset_class is a required (non-Optional) field, Pydantic itself
enforces the Phase 1 done-bar: "every distinct security ... has a
non-null asset_class". A security that can't be classified simply cannot
become a SecurityMaster instance — see security_resolution.py, where an
unclassifiable security is routed to the unresolved/manual-review list
instead of being force-fit into this model with a placeholder value.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class SecurityType(str, Enum):
    EQUITY = "Equity"
    ETF = "ETF"
    MUTUAL_FUND = "Mutual Fund"
    BOND = "Bond"
    CASH = "Cash"
    OTHER = "Other"


class AssetClass(str, Enum):
    EQUITY = "Equity"
    FIXED_INCOME = "Fixed Income"
    CASH = "Cash"
    ALTERNATIVE = "Alternative"
    MULTI_ASSET = "Multi-Asset"
    PENDING_REVIEW = "Pending Review"
    """Explicit sentinel for "we do not actually know yet" — used instead
    of guessing a real asset class (e.g. defaulting a mutual fund to
    Equity) when the classification pipeline has no reliable basis for
    one. Satisfies the Pydantic non-null requirement honestly rather
    than shipping a plausible-looking wrong value dressed up with a
    review flag that downstream code might ignore. Any security carrying
    this value must not be used in allocation, attribution, or risk
    calculations until a human resolves it — this is functionally an
    "unresolved" state that still fits the schema, not a real class."""


class SecurityMaster(BaseModel):
    security_id: str
    cusip: Optional[str] = None
    ticker: Optional[str] = None
    name: str
    security_type: SecurityType
    asset_class: AssetClass
    sub_asset_class: Optional[str] = None
    sector: Optional[str] = None
    geography: Optional[str] = None
    expense_ratio: Optional[Decimal] = None
