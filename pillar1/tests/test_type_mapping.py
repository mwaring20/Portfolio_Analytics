from pathlib import Path

from pillar1.type_mapping import (
    ExactTypeMap,
    PrefixTypeMap,
    load_fidelity_action_map,
    load_schwab_type_map,
)


def test_schwab_default_config_loads_and_resolves_known_type():
    type_map = load_schwab_type_map()
    txn_type, needs_review, reason = type_map.resolve("Buy")
    assert txn_type == "BUY"
    assert needs_review is False
    assert reason is None


def test_schwab_default_config_flags_stock_split_with_reason():
    type_map = load_schwab_type_map()
    txn_type, needs_review, reason = type_map.resolve("Stock Split")
    assert txn_type == "STOCK_SPLIT"
    assert needs_review is True
    assert reason and "split" in reason.lower()


def test_schwab_unmapped_type_falls_back_to_unknown():
    type_map = load_schwab_type_map()
    txn_type, needs_review, reason = type_map.resolve("Some New Type Nobody Has Seen")
    assert txn_type == "UNKNOWN"
    assert needs_review is True
    assert "Unrecognized" in reason


def test_fidelity_default_config_prefix_matches():
    action_map = load_fidelity_action_map()
    txn_type, needs_review, reason = action_map.resolve("YOU BOUGHT VOO")
    assert txn_type == "BUY"
    assert needs_review is False


def test_fidelity_unmapped_action_falls_back_to_unknown():
    action_map = load_fidelity_action_map()
    txn_type, needs_review, reason = action_map.resolve("SOME BRAND NEW ACTION TYPE")
    assert txn_type == "UNKNOWN"
    assert needs_review is True


def test_prefix_order_sensitivity(tmp_path):
    """A more specific prefix listed first must win over a more general
    one listed after it — proves order is respected, not just presence."""
    config = tmp_path / "prefix_order.yaml"
    config.write_text(
        """
- prefix: "YOU BOUGHT SPECIAL"
  txn_type: SPECIAL_BUY
  needs_review: true
  review_reason: "special case"
- prefix: "YOU BOUGHT"
  txn_type: BUY
  needs_review: false
"""
    )
    action_map = PrefixTypeMap.from_yaml(config)
    assert action_map.resolve("YOU BOUGHT SPECIAL WIDGET")[0] == "SPECIAL_BUY"
    assert action_map.resolve("YOU BOUGHT VOO")[0] == "BUY"


def test_custom_config_path_overrides_default(tmp_path):
    config = tmp_path / "custom_schwab.yaml"
    config.write_text(
        """
Buy:
  txn_type: CUSTOM_BUY
  needs_review: false
"""
    )
    type_map = ExactTypeMap.from_yaml(config)
    txn_type, needs_review, reason = type_map.resolve("Buy")
    assert txn_type == "CUSTOM_BUY"

    # And unmapped types still fall through cleanly even in a minimal config.
    txn_type, needs_review, reason = type_map.resolve("Sell")
    assert txn_type == "UNKNOWN"
    assert needs_review is True


def test_config_files_cover_every_type_seen_in_fixtures():
    """Every raw type/action string actually observed in the phantom
    fixtures must have an explicit config entry — this test fails loudly
    if a config edit accidentally drops a mapping, rather than letting
    it silently fall through to UNKNOWN."""
    schwab_map = load_schwab_type_map()
    for raw_type in ["Buy", "Sell", "Qualified Dividend", "Reinvest Shares", "Service Fee", "Bank Interest", "Stock Split", "Journal"]:
        txn_type, _, _ = schwab_map.resolve(raw_type)
        assert txn_type != "UNKNOWN", f"Schwab type {raw_type!r} unexpectedly unmapped"

    fidelity_map = load_fidelity_action_map()
    for raw_action in [
        "YOU BOUGHT VOO",
        "YOU SOLD VOO",
        "REINVESTMENT BND",
        "DIRECT DEBIT ADVISORY FEE",
        "DIVIDEND RECEIVED BND",
        "INTEREST EARNED CORE ACCOUNT",
        "TRANSFERRED FROM ACCT",
    ]:
        txn_type, _, _ = fidelity_map.resolve(raw_action)
        assert txn_type != "UNKNOWN", f"Fidelity action {raw_action!r} unexpectedly unmapped"
