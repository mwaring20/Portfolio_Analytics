"""
Phase 0 -> Phase 1 wiring.

Each ingest_* function runs a custodian adapter and immediately resolves
every distinct security it saw against a SecurityMasterRegistry, in one
call — this is the "adapters resolve security_id against SecurityMaster
as part of their output" wiring.

Deliberately implemented as a thin orchestration layer ON TOP OF the
existing adapters (adapters/schwab.py, adapters/fidelity.py) rather than
folding resolution logic into the adapters themselves: Phase 0's
transactions_df contract stays exactly as tested in Phase 0 (unmutated,
independently reusable by Phase 2+), and Phase 1's resolution logic
stays reusable against ANY transactions_df — not just ones freshly
produced by an adapter. A future Phase 2 that re-reads persisted
transactions_df from a prior run gets the same resolution path for free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from .adapters.fidelity import load_fidelity_household
from .adapters.schwab import load_schwab_export
from .providers.base import SecurityLookupProvider
from .security_resolution import SecurityMasterRegistry, resolve_transactions_securities


class IngestionResult:
    def __init__(
        self,
        transactions_df: pd.DataFrame,
        quarantine_df: pd.DataFrame,
        structural_dropped_df: pd.DataFrame,
        resolved_securities: dict,
        unresolved_securities: list,
    ):
        self.transactions_df = transactions_df
        self.quarantine_df = quarantine_df
        self.structural_dropped_df = structural_dropped_df
        self.resolved_securities = resolved_securities
        self.unresolved_securities = unresolved_securities

    def security_master_summary(self) -> str:
        lines = [
            f"Securities resolved: {len(self.resolved_securities)}",
            f"Securities unresolved (manual review needed): {len(self.unresolved_securities)}",
        ]
        flagged = [
            (sec_id, e) for sec_id, e in self.resolved_securities.items() if e.needs_review
        ]
        if flagged:
            lines.append(f"Resolved but flagged needs_review: {len(flagged)}")
            for sec_id, e in flagged:
                lines.append(f"  {sec_id} ({e.security.ticker}): {e.review_reason}")
        if self.unresolved_securities:
            lines.append("Unresolved:")
            for u in self.unresolved_securities:
                lines.append(f"  {u.raw_identifier}: {u.reason}")
        return "\n".join(lines)


def ingest_schwab(path: Union[str, Path], registry: SecurityMasterRegistry) -> IngestionResult:
    transactions_df, quarantine_df = load_schwab_export(path)
    resolved, unresolved = resolve_transactions_securities(transactions_df, registry)
    return IngestionResult(
        transactions_df=transactions_df,
        quarantine_df=quarantine_df,
        structural_dropped_df=pd.DataFrame(),
        resolved_securities=resolved,
        unresolved_securities=unresolved,
    )


def ingest_fidelity_household(
    paths: Iterable[Union[str, Path]], registry: SecurityMasterRegistry
) -> IngestionResult:
    transactions_df, quarantine_df, dropped_df = load_fidelity_household(paths)
    resolved, unresolved = resolve_transactions_securities(transactions_df, registry)
    return IngestionResult(
        transactions_df=transactions_df,
        quarantine_df=quarantine_df,
        structural_dropped_df=dropped_df,
        resolved_securities=resolved,
        unresolved_securities=unresolved,
    )
