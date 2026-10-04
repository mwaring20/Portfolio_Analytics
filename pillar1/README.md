# Pillar 1 — Portfolio Analytics

A comprehensive portfolio analytics system with 15 phases covering data ingestion, valuation, performance measurement, risk analysis, and compliance recordkeeping.

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

## Phase 2 — Valuation Engine

```python
from pillar1.valuation import ValuationEngine

engine = ValuationEngine()
daily_positions_df = engine.compute_daily_positions(
    transactions_df,
    prices_df,
)
```

Rolls transactions forward and marks to market using price data. Handles missing prices with forward-filling and flags data gaps.

## Phase 3 — Cash Flow Extraction

```python
from pillar1.cash_flows import CashFlowExtractor

extractor = CashFlowExtractor()
cash_flows_df = extractor.extract_cash_flows(
    transactions_df,
    include_fees=False,
    household_level=True,
)
```

Classifies cash flows into DEPOSIT, WITHDRAWAL, DIVIDEND, INTEREST, FEE, TRANSFER_IN, TRANSFER_OUT.

## Phase 4 — Returns Engine

```python
from pillar1.returns import ReturnsEngine

engine = ReturnsEngine()
twr = engine.compute_twr(daily_positions_df)
mwr = engine.compute_mwr(daily_positions_df, cash_flows_df)
ytd_return = engine.compute_period_return(daily_positions_df, "YTD")
```

Computes Time-Weighted Returns (TWR) via chain-linking and Money-Weighted Returns (MWR/XIRR) via Newton-Raphson solver.

## Phase 5 — Benchmark Engine

```python
from pillar1.benchmark import BenchmarkEngine

engine = BenchmarkEngine()
benchmark_prices_df = engine.construct_benchmark(
    benchmark_definition,
    component_prices_df,
)
```

Constructs single and composite benchmark price series with rebalancing logic and total return calculations.

## Phase 6 — Period Return Aggregation

Integrated into Phase 4 — supports MTD, QTD, YTD, 1Y, 3Y, 5Y, since-inception returns.

## Phase 7 — Portfolio Construction Analysis

```python
from pillar1.portfolio_analysis import PortfolioAnalyzer

analyzer = PortfolioAnalyzer()
allocation = analyzer.compute_allocation_breakdown(
    daily_positions_df,
    security_master,
    group_by="asset_class",
)
concentration_flags = analyzer.check_concentration_risk(
    daily_positions_df,
    security_master,
)
fee_drag = analyzer.compute_fee_drag(daily_positions_df, security_master)
```

Allocation breakdown, concentration risk flags, and fee drag analysis.

## Phase 8 — Attribution

```python
from pillar1.attribution import AttributionEngine

engine = AttributionEngine()
result = engine.compute_attribution(
    portfolio_returns_df,
    benchmark_returns_df,
    portfolio_weights_df,
    benchmark_weights_df,
    category="sector",
)
```

Brinson-Fachler decomposition (allocation, selection, interaction effects).

## Phase 9 — Risk Metrics Core

```python
from pillar1.risk_metrics import RiskMetricsEngine

engine = RiskMetricsEngine()
metrics = engine.compute_all_metrics(
    returns_df,
    benchmark_returns_df,
    risk_free_rate=Decimal("0.02"),
)
```

Volatility, Sharpe ratio, Sortino ratio, max drawdown, beta, alpha, tracking error, information ratio.

## Phase 10 — Correlation Matrix

```python
from pillar1.correlation import CorrelationEngine

engine = CorrelationEngine()
correlation_matrix = engine.compute_correlation_matrix(
    daily_positions_df,
    prices_df,
    lookback_window=LookbackWindow.ONE_YEAR,
)
```

Correlation across holdings with configurable lookback windows (1M, 3M, 1Y, 3Y).

## Phase 11 — VaR / CVaR

```python
from pillar1.risk_contribution import RiskContributionEngine

engine = RiskContributionEngine()
decomposition = engine.compute_risk_contribution(
    daily_positions_df,
    correlation_matrix,
    monthly_returns_df,
)
```

- **Historical VaR**: Empirical percentile of 10 years of monthly returns (120 data points)
- **Parametric VaR**: Analytic using correlation matrix and weights
- **CVaR**: Expected Shortfall from lowest 5% of returns
- **Marginal VaR**: Change in portfolio VaR from position size change
- **Component VaR**: Position's contribution to portfolio VaR

## Phase 12 — Scenario & Stress Testing

```python
from pillar1.stress_testing import StressTestEngine

engine = StressTestEngine()
results = engine.run_all_standard_scenarios(
    daily_positions_df,
    security_master,
    as_of_date=date(2024, 1, 31),
)
```

Predefined scenarios: 2008 Crisis, Rate Hike, Tech Crash, COVID-19, Inflation Spike. Custom scenarios supported.

## Phase 13 — Fama-French Factor Exposure

```python
from pillar1.fama_french import FamaFrenchEngine

engine = FamaFrenchEngine()
result = engine.run_regression(
    returns_df,
    factor_returns_df,  # Mkt-RF, SMB, HML from Ken French Data Library
    entity_id="ACC001",
    entity_type="account",
    lookback_months=36,
)
```

Multiple linear regression against factor returns with R², t-statistics, and p-values.

## Phase 14 — Monte Carlo Simulation

```python
from pillar1.monte_carlo import MonteCarloEngine

engine = MonteCarloEngine()
result = engine.run_simulation(
    initial_value=Decimal("100000"),
    expected_returns={"Equity": Decimal("0.08"), "Fixed Income": Decimal("0.04")},
    volatilities={"Equity": Decimal("0.15"), "Fixed Income": Decimal("0.05")},
    correlation_matrix=correlation_matrix,
    weights={"Equity": Decimal("0.6"), "Fixed Income": Decimal("0.4")},
    num_years=30,
    num_paths=1000,
)
```

Probability cone of outcomes using Cholesky decomposition for correlated returns. Retirement readiness simulation with cash flow modeling (contributions/withdrawals).

## Phase 15 — Compliance Recordkeeping

```python
from pillar1.compliance import log_analytics_run, get_audit_logger

event = log_analytics_run(
    analytics_type=EventType.VALUATION,
    description="Portfolio valuation run",
    inputs={"account_id": "ACC001"},
    outputs={"total_value": 100000},
)

logger = get_audit_logger()
export = logger.export_audit_trail()
```

Append-only audit trail with SHA-256 checksums for integrity verification. Evidence snapshots for data retention.

## Package layout

```
pillar1/
  canonical_schema.py       DataFrame contracts (transactions, prices, positions, returns, etc.)
  validation.py             Contract validators for all DataFrame types
  parsing.py                Decimal-only money/quantity/date parsing
  type_mapping.py           Config-driven transaction-type mapping loader

  # Phases 0-1
  security_master.py        Pydantic SecurityMaster model + AssetClass
  security_resolution.py    CUSIP-first/ticker-fallback resolution
  security_seed.py          Seed table loader
  ingest.py                 Phase 0 -> Phase 1 orchestration

  # Phase 2
  valuation.py              ValuationEngine for daily positions
  position_reconciliation.py Reconciliation against custodian snapshots

  # Phase 3-4
  cash_flows.py             CashFlowExtractor
  returns.py                ReturnsEngine (TWR, MWR, period returns)

  # Phase 5
  benchmark.py              BenchmarkEngine (single/composite)

  # Phase 7
  portfolio_analysis.py      PortfolioAnalyzer (allocation, concentration, fees)

  # Phase 8
  attribution.py            AttributionEngine (Brinson-Fachler)

  # Phase 9
  risk_metrics.py           RiskMetricsEngine (volatility, Sharpe, etc.)

  # Phase 10
  correlation.py            CorrelationEngine (holdings correlation)

  # Phase 11
  risk_contribution.py      RiskContributionEngine (VaR, CVaR, marginal/component VaR)

  # Phase 12
  stress_testing.py         StressTestEngine (scenario analysis)

  # Phase 13
  fama_french.py            FamaFrenchEngine (factor regression)

  # Phase 14
  monte_carlo.py           MonteCarloEngine (simulation, retirement readiness)

  # Phase 15
  compliance.py             AuditLogger (audit trail, evidence retention)

  adapters/
    schwab.py               Schwab transaction adapter
    fidelity.py             Fidelity household adapter
    positions.py            Custodian positions adapter (contract)

  providers/
    base.py                 SecurityLookupProvider interface
    prices.py               PriceProvider implementations
    caching.py              Caching wrapper

  config/
    schwab_type_map.yaml
    fidelity_action_map.yaml
    security_seed.yaml

tests/
  fixtures/                 Test data files
  fakes.py                 In-memory fake providers
  test_*.py                 Comprehensive test suite
```
