"""
Tests for the custodian_positions_df contract (canonical_schema.py +
validation.validate_custodian_positions_df).

There's no real position-snapshot export to test an adapter against yet
(see adapters/positions.py) — these tests exercise the CONTRACT itself:
a well-formed hand-built DataFrame should pass, and each way a row can
be broken should be individually caught. This is what "build the
stub/contract now, even without real data" means in practice: the
contract is verifiably enforceable today, so a future concrete adapter
has an immediate, already-tested pass/fail bar to build against.
"""

from decimal import Decimal

import pandas as pd
import pytest

from pillar1.canonical_schema import CUSTODIAN_POSITIONS_COLUMNS
from pillar1.validation import validate_custodian_positions_df


def _valid_row(**overrides) -> dict:
    row = {
        "account_id": "7042-1193",
        "as_of_date": pd.Timestamp("2025-12-31").date(),
        "security_id": "AAPL",
        "quantity": Decimal("100"),
        "market_value": Decimal("25000.00"),
        "source_custodian": "schwab",
        "source_file": "positions_snapshot.csv",
        "source_row": 2,
    }
    row.update(overrides)
    return row


def _df(*rows) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=CUSTODIAN_POSITIONS_COLUMNS)


def test_well_formed_snapshot_passes_with_zero_violations():
    df = _df(_valid_row(), _valid_row(security_id="MSFT", quantity=Decimal("50")))
    assert validate_custodian_positions_df(df) == []


def test_wrong_columns_caught():
    df = pd.DataFrame([{"account_id": "1", "quantity": Decimal("1")}])
    violations = validate_custodian_positions_df(df)
    assert len(violations) == 1
    assert "Column mismatch" in violations[0]


def test_null_account_id_caught():
    df = _df(_valid_row(account_id=None))
    violations = validate_custodian_positions_df(df)
    assert any("account_id" in v and "null" in v for v in violations)


def test_null_as_of_date_caught():
    df = _df(_valid_row(as_of_date=None))
    violations = validate_custodian_positions_df(df)
    assert any("as_of_date" in v and "null" in v for v in violations)


def test_null_security_id_caught():
    """Unlike transactions_df (where FEE/INTEREST/JOURNAL_CASH rows can
    legitimately have no security), a position snapshot row always
    represents a holding — security_id is unconditionally required."""
    df = _df(_valid_row(security_id=None))
    violations = validate_custodian_positions_df(df)
    assert any("security_id" in v and "null" in v for v in violations)


def test_float_quantity_rejected_must_be_decimal():
    df = _df(_valid_row(quantity=100.0))  # float, not Decimal
    violations = validate_custodian_positions_df(df)
    assert any("quantity" in v and "Decimal" in v for v in violations)


def test_float_market_value_rejected_must_be_decimal():
    df = _df(_valid_row(market_value=25000.0))
    violations = validate_custodian_positions_df(df)
    assert any("market_value" in v and "Decimal" in v for v in violations)


def test_null_market_value_caught_always_required():
    df = _df(_valid_row(market_value=None))
    violations = validate_custodian_positions_df(df)
    assert any("market_value" in v and "null" in v for v in violations)


def test_multiple_violations_all_reported_not_just_first():
    df = _df(_valid_row(account_id=None, security_id=None))
    violations = validate_custodian_positions_df(df)
    assert len(violations) >= 2
