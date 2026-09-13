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
    DECIMAL_COLUMNS,
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
