# Pillar 1 — Portfolio Analytics: Phase 0 & Phase 1 (hardened)

## Setup

```
pip install -r requirements.txt
```

For live Phase 1 security lookups (not needed to run the test suite,
which is fully mocked), sign up for a free Financial Modeling Prep API
key at https://financialmodelingprep.com and set it:

```
export FMP_API_KEY=your_key_here
```

## Running tests

```
python -m pytest tests/ -v
```

All 84 tests run offline — no network access or API key required.

## Phase 0 — Custodian Ingestion

```python
from pillar1.adapters.schwab import load_schwab_export
from pillar1.adapters.fidelity import load_fidelity_household

transactions_df, quarantine_df = load_schwab_export("path/to/export.csv")
transactions_df, quarantine_df, dropped_df = load_fidelity_household(
    ["acct1.csv", "acct2.csv"]
)
```

Transaction-type mappings live in `pillar1/config/schwab_type_map.yaml`
and `pillar1/config/fidelity_action_map.yaml` — adding a newly-observed
transaction type is a config edit, not a code change. Pass a custom
`type_map=` / `action_map=` to override the default file.

### Reconciliation report

```
python -m pillar1.run_reconciliation schwab path/to/export.csv
python -m pillar1.run_reconciliation fidelity acct1.csv acct2.csv
```

For Schwab, the expected record count is auto-detected from the file's
own "N records exported" preamble line — `--total` is only needed to
override that. Exits non-zero if row counts don't reconcile or the
schema contract is violated.

## Phase 1 — Security Master

```python
from pillar1.ingest import ingest_schwab
from pillar1.security_resolution import SecurityMasterRegistry
from pillar1.providers.fmp import FinancialModelingPrepProvider
from pillar1.providers.caching import CachingProvider

provider = CachingProvider(
    FinancialModelingPrepProvider(),
    cache_path="security_cache.json",
)
registry = SecurityMasterRegistry(provider)  # seed table loaded automatically
result = ingest_schwab("path/to/export.csv", registry)

print(result.security_master_summary())
```

**Resolution order, cheapest/most-trusted first:**
1. **Seed table** (`pillar1/config/security_seed.yaml`) — manually
   verified securities. Zero API calls, `needs_review=False`. Pass
   `seed_table={}` to `SecurityMasterRegistry(...)` to disable.
2. **On-disk cache** (if using `CachingProvider`) — previously resolved
   securities, persisted across runs. Only positive results are cached;
   a miss is always retried.
3. **Live provider lookup** (FMP) — retried with backoff on transient
   network errors and HTTP 429; a persistent failure for one security is
   recorded in `result.unresolved_securities` rather than crashing the
   run. A bad/missing API key (`ProviderAuthError`) is NOT caught here —
   it propagates immediately, since it's a systemic problem affecting
   every lookup, not a per-security one.

**Known gap, by design:** FMP's basic profile endpoint doesn't return a
true asset-class field for ETFs or mutual funds. An unseeded ETF gets a
name-based heuristic asset class and `needs_review=True`. A mutual fund
gets `AssetClass.PENDING_REVIEW` — an explicit "we don't know yet"
sentinel, not a guessed real value — and is always flagged. Add
genuinely-verified securities to `security_seed.yaml` as you encounter
them to shrink this gap over time; see the accuracy note at the top of
that file before trusting its current entries as-is.

## Position snapshots (contract only — no concrete adapter yet)

`canonical_schema.CUSTODIAN_POSITIONS_COLUMNS` and
`validation.validate_custodian_positions_df()` define the
`custodian_positions_df` contract per the architecture doc, and
`adapters/positions.py` defines the `CustodianPositionsAdapter`
interface a future concrete adapter must satisfy. No real Schwab/Fidelity
position-snapshot export has been seen yet — only transaction-history
exports — so there's no concrete adapter, just the contract, tested
against hand-built DataFrames in `tests/test_positions_contract.py`.
When a real snapshot file is available, see `adapters/positions.py`'s
docstring for the next steps (same manifest-driven TDD pattern as
Phase 0's transaction adapters).

## Package layout

```
pillar1/
  canonical_schema.py       transactions_df + custodian_positions_df contracts
  parsing.py                  Decimal-only money/quantity/date parsing
  type_mapping.py             config-driven transaction-type mapping loader
  validation.py                 contract validators (transactions + positions)
  reconciliation.py             human-readable reconciliation report
  run_reconciliation.py         CLI for the reconciliation report
  security_master.py            Pydantic SecurityMaster model + AssetClass (incl. PENDING_REVIEW)
  security_resolution.py        CUSIP-first/ticker-fallback resolution + classification
  security_seed.py              manually-verified seed table loader
  ingest.py                      Phase 0 -> Phase 1 orchestration
  config/
    schwab_type_map.yaml
    fidelity_action_map.yaml
    security_seed.yaml
  adapters/
    schwab.py
    fidelity.py
    positions.py                 contract/stub only, no concrete implementation yet
  providers/
    base.py                       SecurityLookupProvider interface + exception types
    fmp.py                         Financial Modeling Prep implementation (retry/backoff)
    caching.py                     on-disk JSON cache wrapper
tests/
  fixtures/                       Phantom Schwab/Fidelity export files
  fakes.py                         In-memory fake provider for tests
  test_*.py
```
