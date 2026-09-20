"""
Manually-verified security seed table.

See config/security_seed.yaml for the accuracy caveat and the entries
themselves. Loaded once and consulted by SecurityMasterRegistry BEFORE
any provider call — a seed-table hit costs zero API quota and carries
no needs_review flag, because it represents a human-confirmed
classification rather than an automated guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import yaml

from .security_master import AssetClass, SecurityMaster, SecurityType

DEFAULT_SEED_PATH = Path(__file__).parent / "config" / "security_seed.yaml"


def load_seed_table(path: Optional[Union[str, Path]] = None) -> Dict[str, SecurityMaster]:
    """Returns a dict keyed by ticker. Raises clearly if an entry's
    asset_class/security_type doesn't match the canonical enums, rather
    than silently accepting a typo in the YAML."""
    path = Path(path) if path else DEFAULT_SEED_PATH
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    table: Dict[str, SecurityMaster] = {}
    for ticker, spec in raw.items():
        cusip = spec.get("cusip")
        table[ticker] = SecurityMaster(
            security_id=cusip or ticker,
            cusip=cusip,
            ticker=ticker,
            name=spec["name"],
            security_type=SecurityType(spec["security_type"]),
            asset_class=AssetClass(spec["asset_class"]),
            sub_asset_class=spec.get("sub_asset_class"),
            sector=spec.get("sector"),
            geography=spec.get("geography"),
            expense_ratio=spec.get("expense_ratio"),
        )
    return table
