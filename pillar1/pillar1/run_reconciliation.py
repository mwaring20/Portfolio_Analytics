"""
Command-line runner for the Phase 0 reconciliation report.

Usage:
    python -m pillar1.run_reconciliation schwab path/to/export.csv [--total 100]
    python -m pillar1.run_reconciliation fidelity path/to/account1.csv path/to/account2.csv ...
"""

from __future__ import annotations

import argparse
import sys

from .adapters.fidelity import load_fidelity_household
from .adapters.schwab import load_schwab_export, parse_declared_record_count
from .reconciliation import build_reconciliation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 0 reconciliation report.")
    parser.add_argument("custodian", choices=["schwab", "fidelity"])
    parser.add_argument("paths", nargs="+", help="Path(s) to the raw export file(s)")
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help=(
            "Custodian-declared total record count to reconcile against. "
            "For Schwab, auto-detected from the file's own preamble line "
            "if omitted — pass this only to override that."
        ),
    )
    args = parser.parse_args()

    if args.custodian == "schwab":
        if len(args.paths) != 1:
            print("schwab takes exactly one file", file=sys.stderr)
            sys.exit(1)
        txns, quarantine = load_schwab_export(args.paths[0])
        total = args.total
        total_source = "user-provided" if args.total is not None else None
        if total is None:
            total = parse_declared_record_count(args.paths[0])
            if total is not None:
                total_source = "auto-detected from file's declared record count"
        report = build_reconciliation_report(
            source_label=args.paths[0],
            transactions_df=txns,
            quarantine_df=quarantine,
            total_input_rows=total,
            total_source=total_source,
        )
    else:
        txns, quarantine, dropped = load_fidelity_household(args.paths)
        report = build_reconciliation_report(
            source_label=", ".join(args.paths),
            transactions_df=txns,
            quarantine_df=quarantine,
            structural_dropped_df=dropped,
            total_input_rows=args.total,
        )

    print(report.as_text())
    if report.reconciles is False or report.schema_violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
