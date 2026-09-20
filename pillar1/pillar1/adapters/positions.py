"""
Custodian position-snapshot adapter contract (Phase 0 stub).

Position snapshots are a DIFFERENT export type from the transaction
histories adapters/schwab.py and adapters/fidelity.py handle. Per the
architecture doc, custodian_positions_df is "Raw snapshot —
reconciliation only, never used directly in return calcs" — sourced
from a custodian's dedicated position/holdings export, not derived from
transaction history. Rolling transactions forward into daily positions
is Phase 2's job (the valuation engine); this file is about the
RAW SNAPSHOT used only to reconcile that derived result.

No real Schwab/Fidelity position-snapshot export has been seen yet —
both quirks_manifest.md and the phantom fixtures only cover transaction
history. Rather than fabricate a plausible-looking snapshot format and
risk building against invented quirks (exactly the mistake Phase 0's
real fixtures were built to avoid for transactions), this module defines
only the CONTRACT a concrete adapter must satisfy, per the architecture
doc's own principle: reference/config schemas are settled up front so
they don't change later, even when the concrete implementation waits on
real data.

When a real snapshot export is available:
  1. Capture its quirks in a manifest, same TDD pattern as Phase 0's
     custodian adapters (see quirks_manifest.md for the template).
  2. Implement a class satisfying CustodianPositionsAdapter below, or a
     plain `load_<custodian>_positions(path) -> pd.DataFrame` function
     shaped like the existing load_schwab_export /
     load_fidelity_export functions.
  3. Validate its output with
     validation.validate_custodian_positions_df() before trusting it —
     see test_positions_contract.py for how the contract itself is
     already exercised against a hand-built DataFrame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Union

import pandas as pd


class CustodianPositionsAdapter(Protocol):
    def load(self, path: Union[str, Path]) -> pd.DataFrame:
        """Return a DataFrame shaped per
        canonical_schema.CUSTODIAN_POSITIONS_COLUMNS, validated with
        validation.validate_custodian_positions_df() before being
        returned to callers."""
        ...
