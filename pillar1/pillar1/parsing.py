"""
Shared field-parsing helpers.

Custodians format the same underlying data differently:
  - Schwab negatives:   ($1,265.50)   (parentheses, dollar sign, comma)
  - Fidelity negatives: -14,372.45    (leading minus, comma, no dollar sign)

Both must resolve to the same Decimal. This module is the only place
string -> Decimal / string -> date conversion happens, so a format quirk
discovered in a future custodian only needs to be handled once.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Optional


def parse_money(raw: Optional[str]) -> Optional[Decimal]:
    """
    Parse a dollar-amount field into a Decimal.

    Handles:
      - None / NaN / empty string           -> None
      - plain values                        "970.89"        -> Decimal("970.89")
      - dollar sign                         "$253.10"       -> Decimal("253.10")
      - thousands separators                "15,267.01"     -> Decimal("15267.01")
      - parentheses-negative (Schwab)        "($1,265.50)"   -> Decimal("-1265.50")
      - leading-minus-negative (Fidelity)    "-14,372.45"    -> Decimal("-14372.45")

    Never uses float() at any point — fractional shares and DRIP amounts
    accumulate float error fast, per the architecture doc's design
    principle.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    text = text.replace("$", "").replace(",", "").strip()
    if text == "":
        return None

    if text.startswith("-"):
        negative = True
        text = text[1:]

    try:
        value = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"Could not parse money field: {raw!r}")

    return -value if negative else value


def parse_quantity(raw: Optional[str]) -> Optional[Decimal]:
    """
    Parse a share-quantity field into a Decimal. Same comma-stripping as
    parse_money but quantities in these exports are never parenthesized
    or dollar-signed, so this is intentionally a thinner wrapper (kept
    separate from parse_money so a future quantity-specific quirk -- e.g.
    a custodian using a trailing 'S' for short positions -- has its own
    home rather than overloading the money parser).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return None
    text = text.replace(",", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        raise ValueError(f"Could not parse quantity field: {raw!r}")


def parse_date(raw: Optional[str], fmt: str = "%m/%d/%Y") -> Optional[date]:
    """Parse an MM/DD/YYYY date field. Blank/None -> None (never inferred
    here — callers decide whether a missing date means quarantine)."""
    if raw is None:
        return None
    text = str(raw).strip()
    if text == "" or text.lower() == "nan":
        return None
    try:
        return datetime.strptime(text, fmt).date()
    except ValueError:
        raise ValueError(f"Could not parse date field: {raw!r} (expected {fmt})")
