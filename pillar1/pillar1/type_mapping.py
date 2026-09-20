"""
Config-driven transaction-type mapping (Phase 0 hardening).

The architecture doc is explicit: "Put transaction-type mapping ... in a
config file, not code." The original adapters had this as a hardcoded
Python dict — this module replaces that with YAML-backed mapping tables,
loaded at runtime, so adding a newly-observed custodian transaction type
is a config edit, not a code change and redeploy.

Two mapping shapes, because the two custodians' raw type fields have
different shapes:
  - ExactTypeMap: exact string match (Schwab's "Type" column is a closed
    vocabulary of short codes).
  - PrefixTypeMap: prefix match, order-sensitive (Fidelity's "Action"
    column is free text like "YOU BOUGHT VOO").

Both fall through to UNKNOWN + needs_review=True for anything not
listed, so an adapter never needs an explicit "else" branch and a
new/unmapped custodian type is never silently misclassified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class TypeMapping:
    txn_type: str
    needs_review: bool = False
    review_reason: Optional[str] = None


class ExactTypeMap:
    """Exact-match mapping, e.g. Schwab's Type column."""

    def __init__(self, mapping: Dict[str, TypeMapping]):
        self._mapping = mapping

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ExactTypeMap":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        mapping = {
            raw_type: TypeMapping(
                txn_type=spec["txn_type"],
                needs_review=spec.get("needs_review", False),
                review_reason=spec.get("review_reason"),
            )
            for raw_type, spec in raw.items()
        }
        return cls(mapping)

    def resolve(self, raw_type: str) -> Tuple[str, bool, Optional[str]]:
        m = self._mapping.get(raw_type)
        if m is None:
            return "UNKNOWN", True, f"Unrecognized transaction type: {raw_type!r}"
        return m.txn_type, m.needs_review, m.review_reason


class PrefixTypeMap:
    """Prefix-match mapping, e.g. Fidelity's free-text Action column.
    Order matters — entries are checked top to bottom, first match wins."""

    def __init__(self, entries: List[Tuple[str, TypeMapping]]):
        self._entries = entries

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PrefixTypeMap":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or []
        entries = [
            (
                spec["prefix"],
                TypeMapping(
                    txn_type=spec["txn_type"],
                    needs_review=spec.get("needs_review", False),
                    review_reason=spec.get("review_reason"),
                ),
            )
            for spec in raw
        ]
        return cls(entries)

    def resolve(self, raw_action: str) -> Tuple[str, bool, Optional[str]]:
        for prefix, m in self._entries:
            if raw_action.startswith(prefix):
                return m.txn_type, m.needs_review, m.review_reason
        return "UNKNOWN", True, f"Unrecognized action: {raw_action!r}"


def load_schwab_type_map(path: Optional[Union[str, Path]] = None) -> ExactTypeMap:
    return ExactTypeMap.from_yaml(path or DEFAULT_CONFIG_DIR / "schwab_type_map.yaml")


def load_fidelity_action_map(path: Optional[Union[str, Path]] = None) -> PrefixTypeMap:
    return PrefixTypeMap.from_yaml(path or DEFAULT_CONFIG_DIR / "fidelity_action_map.yaml")
