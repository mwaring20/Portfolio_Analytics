"""
Contract validation for transactions_df.

Returns a list of human-readable violation strings (empty list == passes).
Deliberately does not raise, so a caller (adapter, test, or a future
Phase-15 audit pass) can decide what to do with violations rather than
having control flow dictated by this module.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List

import numpy as np
import pandas as pd

from .canonical_schema import (
    ALWAYS_REQUIRED_COLUMNS,
    BENCHMARK_PRICES_COLUMNS,
    BENCHMARK_PRICES_DECIMAL_COLUMNS,
    BENCHMARK_PRICES_REQUIRED_COLUMNS,
    CASH_FLOWS_COLUMNS,
    CASH_FLOWS_DECIMAL_COLUMNS,
    CASH_FLOWS_REQUIRED_COLUMNS,
    CUSTODIAN_POSITIONS_COLUMNS,
    CUSTODIAN_POSITIONS_DECIMAL_COLUMNS,
    CUSTODIAN_POSITIONS_REQUIRED_COLUMNS,
    DAILY_POSITIONS_COLUMNS,
    DAILY_POSITIONS_DECIMAL_COLUMNS,
    DAILY_POSITIONS_REQUIRED_COLUMNS,
    DECIMAL_COLUMNS,
    FLOW_TYPES,
    PRICES_COLUMNS,
    PRICES_DECIMAL_COLUMNS,
    PRICES_REQUIRED_COLUMNS,
    RETURNS_COLUMNS,
    RETURNS_DECIMAL_COLUMNS,
    RETURNS_REQUIRED_COLUMNS,
    TRANSACTIONS_COLUMNS,
    TXN_TYPES,
    TYPES_WITHOUT_AMOUNT_OK,
    TYPES_WITHOUT_SECURITY_OK,
)


def validate_transactions_df(df: pd.DataFrame) -> List[str]:
    violations: List[str] = []

    # 1. Column contract: exact set, exact order.
    if list(df.columns) != TRANSACTIONS_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {TRANSACTIONS_COLUMNS}, got {list(df.columns)}"
        )
        # Column-shape violations make row-level checks unreliable; bail out.
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (source_file={row.get('source_file')!r}, source_row={row.get('source_row')!r})"

        # 2. Always-required columns must be non-null.
        for col in ALWAYS_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        # 3. txn_type must be in the canonical vocabulary.
        if row["txn_type"] not in TXN_TYPES:
            violations.append(f"{loc}: txn_type {row['txn_type']!r} not in canonical TXN_TYPES")

        # 4. Decimal columns must be Decimal or None — never float, never str.
        for col in DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal or None"
                    )

        # 5. security_id nullability depends on txn_type.
        if pd.isna(row["security_id"]) and row["txn_type"] not in TYPES_WITHOUT_SECURITY_OK:
            violations.append(
                f"{loc}: security_id is null but txn_type {row['txn_type']!r} requires one"
            )

        # 6. net_amount nullability depends on txn_type.
        if pd.isna(row["net_amount"]) and row["txn_type"] not in TYPES_WITHOUT_AMOUNT_OK:
            violations.append(
                f"{loc}: net_amount is null but txn_type {row['txn_type']!r} requires one"
            )

        # 7. needs_review must be bool-valued. Note: pandas upcasts an
        # all-True/False object column to a native bool dtype, so scalar
        # access via .iterrows() yields numpy.bool_ rather than Python's
        # bool — that's an acceptable representation of "is a boolean",
        # so we check via np.bool_ instead of isinstance(..., bool).
        if not isinstance(row["needs_review"], (bool, np.bool_)):
            violations.append(f"{loc}: needs_review is {type(row['needs_review']).__name__}, expected bool")

        # 8. If needs_review, there must be a reason (audit trail requires
        #    every flag to be explicable, not just a bare boolean).
        if row["needs_review"] and (pd.isna(row["review_reason"]) or str(row["review_reason"]).strip() == ""):
            violations.append(f"{loc}: needs_review=True but review_reason is empty")

    return violations


def validate_custodian_positions_df(df: pd.DataFrame) -> List[str]:
    """
    Contract validator for custodian_positions_df (see canonical_schema.py
    for why this exists as a stub ahead of any concrete adapter). Same
    non-raising, violation-list pattern as validate_transactions_df, so a
    future positions adapter can reuse this immediately.
    """
    violations: List[str] = []

    if list(df.columns) != CUSTODIAN_POSITIONS_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {CUSTODIAN_POSITIONS_COLUMNS}, got {list(df.columns)}"
        )
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (source_file={row.get('source_file')!r}, source_row={row.get('source_row')!r})"

        for col in CUSTODIAN_POSITIONS_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        for col in CUSTODIAN_POSITIONS_DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal"
                    )

    return violations


def validate_prices_df(df: pd.DataFrame) -> List[str]:
    """
    Contract validator for prices_df (Phase 2).
    """
    violations: List[str] = []

    if list(df.columns) != PRICES_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {PRICES_COLUMNS}, got {list(df.columns)}"
        )
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (security_id={row.get('security_id')!r}, date={row.get('date')!r})"

        for col in PRICES_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        for col in PRICES_DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal"
                    )

    return violations


def validate_daily_positions_df(df: pd.DataFrame) -> List[str]:
    """
    Contract validator for daily_positions_df (Phase 2).
    """
    violations: List[str] = []

    if list(df.columns) != DAILY_POSITIONS_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {DAILY_POSITIONS_COLUMNS}, got {list(df.columns)}"
        )
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (account_id={row.get('account_id')!r}, security_id={row.get('security_id')!r}, date={row.get('date')!r})"

        for col in DAILY_POSITIONS_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        for col in DAILY_POSITIONS_DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal"
                    )

    return violations


def validate_cash_flows_df(df: pd.DataFrame) -> List[str]:
    """
    Contract validator for cash_flows_df (Phase 3).
    """
    violations: List[str] = []

    if list(df.columns) != CASH_FLOWS_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {CASH_FLOWS_COLUMNS}, got {list(df.columns)}"
        )
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (account_id={row.get('account_id')!r}, date={row.get('date')!r})"

        for col in CASH_FLOWS_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        for col in CASH_FLOWS_DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal"
                    )

        # flow_type must be in canonical FLOW_TYPES
        if row["flow_type"] not in FLOW_TYPES:
            violations.append(
                f"{loc}: flow_type {row['flow_type']!r} not in canonical FLOW_TYPES"
            )

    return violations


def validate_benchmark_prices_df(df: pd.DataFrame) -> List[str]:
    """
    Contract validator for benchmark_prices_df (Phase 5).
    """
    violations: List[str] = []

    if list(df.columns) != BENCHMARK_PRICES_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {BENCHMARK_PRICES_COLUMNS}, got {list(df.columns)}"
        )
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (benchmark_id={row.get('benchmark_id')!r}, date={row.get('date')!r})"

        for col in BENCHMARK_PRICES_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        for col in BENCHMARK_PRICES_DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal"
                    )

    return violations


def validate_returns_df(df: pd.DataFrame) -> List[str]:
    """
    Contract validator for returns_df (Phase 4/5).
    """
    violations: List[str] = []

    if list(df.columns) != RETURNS_COLUMNS:
        violations.append(
            f"Column mismatch. Expected {RETURNS_COLUMNS}, got {list(df.columns)}"
        )
        return violations

    for idx, row in df.iterrows():
        loc = f"row {idx} (entity_id={row.get('entity_id')!r}, date={row.get('date')!r})"

        for col in RETURNS_REQUIRED_COLUMNS:
            if pd.isna(row[col]):
                violations.append(f"{loc}: required column '{col}' is null")

        for col in RETURNS_DECIMAL_COLUMNS:
            val = row[col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if not isinstance(val, Decimal):
                    violations.append(
                        f"{loc}: column '{col}' is {type(val).__name__}, expected Decimal"
                    )

        # entity_type must be one of the allowed values
        if row["entity_type"] not in ["account", "composite", "benchmark"]:
            violations.append(
                f"{loc}: entity_type {row['entity_type']!r} not in allowed values (account, composite, benchmark)"
            )

    return violations
