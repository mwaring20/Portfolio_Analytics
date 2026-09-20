"""
Phase 0 reconciliation report.

The whole point of the quarantine / needs_review / structural-drop split
is that a human can sanity-check an adapter run before the output is
trusted for anything downstream. This module turns the three DataFrames
an adapter produces into a single readable summary for exactly that
purpose — meant to be run once per real client file, not just in tests.

Not itself a CalculationRun (that's Phase 3+ machinery once the returns
engine exists) — this is a pre-ingestion sanity check, closer in spirit
to Phase 0's own "done looks like: zero schema violations" bar than to
the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .validation import validate_transactions_df


@dataclass
class ReconciliationReport:
    source_label: str
    total_input_rows: Optional[int]
    transactions_count: int
    quarantine_count: int
    structural_dropped_count: int
    txn_type_breakdown: "pd.Series"
    needs_review_breakdown: List[dict]
    quarantine_detail: List[dict]
    structural_dropped_detail: List[dict]
    schema_violations: List[str]
    total_source: Optional[str] = None  # e.g. "auto-detected from file preamble" or "user-provided"

    @property
    def accounted_for(self) -> Optional[int]:
        """Rows the adapter can explain: transactions + quarantine. This
        deliberately EXCLUDES structural_dropped_count, because structural
        drops (boilerplate) aren't transaction rows to begin with and
        shouldn't be compared against a 'N records exported' claim that's
        counting actual transactions."""
        return self.transactions_count + self.quarantine_count

    @property
    def reconciles(self) -> Optional[bool]:
        if self.total_input_rows is None:
            return None
        return self.accounted_for == self.total_input_rows

    def as_text(self) -> str:
        lines = []
        lines.append(f"Reconciliation report — {self.source_label}")
        lines.append("=" * (len(lines[0])))

        if self.total_input_rows is not None:
            status = "OK" if self.reconciles else "MISMATCH — INVESTIGATE"
            source_note = f" ({self.total_source})" if self.total_source else ""
            lines.append(
                f"\nRow accounting: {self.transactions_count} transactions + "
                f"{self.quarantine_count} quarantined = {self.accounted_for} "
                f"(expected {self.total_input_rows}{source_note}) -> {status}"
            )
            if not self.reconciles:
                lines.append(
                    "  ** Rows may have been silently lost. Do not trust this "
                    "output until the gap is explained. **"
                )
        else:
            lines.append(
                f"\nRow accounting: {self.transactions_count} transactions + "
                f"{self.quarantine_count} quarantined (no declared total to "
                f"reconcile against for this source)"
            )
        lines.append(f"Structural rows dropped (boilerplate, non-data): {self.structural_dropped_count}")

        lines.append(f"\nSchema validation: {len(self.schema_violations)} violation(s)")
        if self.schema_violations:
            for v in self.schema_violations[:10]:
                lines.append(f"  - {v}")
            if len(self.schema_violations) > 10:
                lines.append(f"  ... and {len(self.schema_violations) - 10} more")

        lines.append("\nTransaction type breakdown:")
        for txn_type, count in self.txn_type_breakdown.items():
            lines.append(f"  {txn_type:<18} {count}")

        review_rows = [r for r in self.needs_review_breakdown if r["count"] > 0]
        lines.append(f"\nRows flagged needs_review=True: {sum(r['count'] for r in review_rows)}")
        for r in review_rows:
            lines.append(f"  {r['txn_type']:<18} {r['count']:<4} — {r['reason']}")

        if self.quarantine_detail:
            lines.append(f"\nQuarantined rows ({len(self.quarantine_detail)}):")
            for q in self.quarantine_detail:
                lines.append(
                    f"  {q['source_file']} row {q['source_row']}: {q['quarantine_reason']}"
                )

        if self.structural_dropped_detail:
            lines.append(f"\nStructural drops ({len(self.structural_dropped_detail)}):")
            for d in self.structural_dropped_detail[:5]:
                preview = d["raw_line"][:60]
                lines.append(f"  {d['source_file']} row {d['source_row']}: {preview!r}...")
            if len(self.structural_dropped_detail) > 5:
                lines.append(f"  ... and {len(self.structural_dropped_detail) - 5} more")

        return "\n".join(lines)


def build_reconciliation_report(
    source_label: str,
    transactions_df: pd.DataFrame,
    quarantine_df: pd.DataFrame,
    structural_dropped_df: Optional[pd.DataFrame] = None,
    total_input_rows: Optional[int] = None,
    total_source: Optional[str] = None,
) -> ReconciliationReport:
    """
    Build a ReconciliationReport from an adapter's output.

    total_input_rows: pass the custodian's own declared record count when
    available (e.g. Schwab's "100 records exported" preamble line) so the
    report can flag a genuine mismatch rather than just describing counts
    in isolation. Leave as None when the source doesn't declare one (as
    with Fidelity's export, which states no total).
    """
    structural_dropped_df = (
        structural_dropped_df if structural_dropped_df is not None else pd.DataFrame()
    )

    txn_type_breakdown = (
        transactions_df["txn_type"].value_counts()
        if not transactions_df.empty
        else pd.Series(dtype=int)
    )

    needs_review_breakdown = []
    if not transactions_df.empty:
        review_df = transactions_df[transactions_df["needs_review"] == True]  # noqa: E712
        for txn_type, group in review_df.groupby("txn_type"):
            reasons = group["review_reason"].dropna().unique()
            reason_text = reasons[0] if len(reasons) == 1 else f"{len(reasons)} distinct reasons"
            needs_review_breakdown.append(
                {"txn_type": txn_type, "count": len(group), "reason": reason_text}
            )

    schema_violations = validate_transactions_df(transactions_df) if not transactions_df.empty else []

    return ReconciliationReport(
        source_label=source_label,
        total_input_rows=total_input_rows,
        transactions_count=len(transactions_df),
        quarantine_count=len(quarantine_df),
        structural_dropped_count=len(structural_dropped_df),
        txn_type_breakdown=txn_type_breakdown,
        needs_review_breakdown=needs_review_breakdown,
        quarantine_detail=quarantine_df.to_dict("records") if not quarantine_df.empty else [],
        structural_dropped_detail=(
            structural_dropped_df.to_dict("records") if not structural_dropped_df.empty else []
        ),
        schema_violations=schema_violations,
        total_source=total_source,
    )
