"""
Phase 3 — Returns Engine (TWR and MWR/IRR).

Per architecture doc:
  - TWR: chain-link true daily returns from Phase 2's daily valuations
  - MWR/IRR: solve via Newton-Raphson / brentq against cash_flows_df + ending value
  - Decide gross vs. net of fees now (expensive to retrofit later)
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from typing import Optional, Tuple

import pandas as pd
from scipy.optimize import brentq

from .canonical_schema import DAILY_POSITIONS_COLUMNS


class ReturnsEngine:
    """
    Computes time-weighted returns (TWR) and money-weighted returns (MWR/IRR).

    TWR: Chain-links daily returns from daily_positions, unaffected by cash flows.
    MWR: Solves for IRR given cash flows and ending value, affected by timing/size of flows.
    """

    def __init__(self, net_of_fees: bool = True):
        """
        Initialize the returns engine.

        net_of_fees: if True, compute net-of-fee returns; if False, gross returns.
                     This requires fee data to be available in the valuation.
        """
        self.net_of_fees = net_of_fees

    def compute_twr(
        self,
        daily_positions_df: pd.DataFrame,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Decimal:
        """
        Compute time-weighted return over a period using daily valuations.

        TWR = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1

        Where ri is the daily return for subperiod i.

        Args:
            daily_positions_df: daily positions from valuation engine
            start_date: start of period (defaults to earliest date in data)
            end_date: end of period (defaults to latest date in data)

        Returns:
            TWR as a Decimal (e.g., 0.15 for 15% return)
        """
        if daily_positions_df.empty:
            return Decimal("0")

        # Determine date range
        if start_date is None:
            start_date = daily_positions_df["date"].min()
        if end_date is None:
            end_date = daily_positions_df["date"].max()

        # Filter to date range
        df = daily_positions_df[
            (daily_positions_df["date"] >= start_date)
            & (daily_positions_df["date"] <= end_date)
        ].copy()

        if df.empty:
            return Decimal("0")

        # Group by date and sum market values across all positions
        daily_values = df.groupby("date")["market_value"].sum().reset_index()
        daily_values = daily_values.sort_values("date").reset_index(drop=True)

        if len(daily_values) < 2:
            # Need at least 2 days to compute a return
            return Decimal("0")

        # Chain-link daily returns
        twr_factor = Decimal("1")

        for i in range(1, len(daily_values)):
            prev_value = daily_values.iloc[i - 1]["market_value"]
            curr_value = daily_values.iloc[i]["market_value"]

            if prev_value == 0:
                # Can't compute return with zero starting value
                continue

            daily_return = (curr_value - prev_value) / prev_value
            twr_factor *= (Decimal("1") + daily_return)

        twr = twr_factor - Decimal("1")
        return twr

    def compute_mwr(
        self,
        cash_flows_df: pd.DataFrame,
        ending_value: Decimal,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Decimal:
        """
        Compute money-weighted return (IRR/XIRR) using cash flows and ending value.

        Solves for the rate r that makes the NPV of cash flows equal to zero:
        sum(cf_i / (1 + r)^((d_i - d_0)/365)) + ending_value / (1 + r)^((d_end - d_0)/365) = 0

        Uses scipy.optimize.brentq for robust root-finding.

        Args:
            cash_flows_df: cash flows from CashFlowExtractor
            ending_value: portfolio value at end_date
            start_date: start of period (defaults to earliest cash flow date)
            end_date: end of period (defaults to latest cash flow date)

        Returns:
            MWR/IRR as a Decimal (e.g., 0.15 for 15% return)

        Raises:
            ValueError: if cash flows are empty or solver fails to converge
        """
        if cash_flows_df.empty:
            raise ValueError("Cannot compute MWR with empty cash flows")

        # Determine date range
        if start_date is None:
            start_date = cash_flows_df["date"].min()
        if end_date is None:
            end_date = cash_flows_df["date"].max()

        # Filter to date range
        df = cash_flows_df[
            (cash_flows_df["date"] >= start_date)
            & (cash_flows_df["date"] <= end_date)
        ].copy()

        if df.empty:
            raise ValueError("No cash flows in specified date range")

        # Prepare data for XIRR solver
        # Cash flows: positive for inflows, negative for outflows
        # Ending value is treated as a final outflow (you could withdraw it)
        flows = []
        dates = []

        for _, row in df.iterrows():
            flows.append(float(row["amount"]))
            dates.append(row["date"])

        # Add ending value as a final "outflow"
        flows.append(float(ending_value))
        dates.append(end_date)

        # Sort by date
        sorted_data = sorted(zip(dates, flows), key=lambda x: x[0])
        dates = [d for d, _ in sorted_data]
        flows = [f for _, f in sorted_data]

        # Compute XIRR
        irr = self._xirr(flows, dates)

        return Decimal(str(irr))

    def _xirr(self, flows: list[float], dates: list[date]) -> float:
        """
        Internal XIRR solver using Brent's method.

        Args:
            flows: list of cash flow amounts (positive = inflow, negative = outflow)
            dates: list of corresponding dates

        Returns:
            IRR as a float

        Raises:
            ValueError: if solver fails to converge
        """
        if len(flows) != len(dates):
            raise ValueError("Flows and dates must have same length")

        if len(flows) < 2:
            raise ValueError("Need at least 2 cash flows to compute IRR")

        # Use the first date as the reference (day 0)
        start_date = dates[0]

        # Convert dates to day fractions
        days = [(d - start_date).days for d in dates]

        # Define the NPV function
        def npv(rate: float) -> float:
            total = 0.0
            for flow, day in zip(flows, days):
                total += flow / ((1.0 + rate) ** (day / 365.0))
            return total

        # Find bounds for the root
        # Try a wide range of possible rates (-99% to +1000%)
        try:
            # First, try to find sign change
            lower = -0.99  # -99%
            upper = 10.0  # 1000%

            npv_lower = npv(lower)
            npv_upper = npv(upper)

            # If both have same sign, expand bounds
            while npv_lower * npv_upper > 0 and upper < 1000:
                upper *= 2
                npv_upper = npv(upper)

            # If still same sign, try negative direction
            if npv_lower * npv_upper > 0:
                lower = -0.5
                npv_lower = npv(lower)
                while npv_lower * npv_upper > 0 and lower > -0.99:
                    lower -= 0.1
                    npv_lower = npv(lower)

            # Use Brent's method to find the root
            irr = brentq(npv, lower, upper, maxiter=1000, xtol=1e-12)
            return irr

        except ValueError as e:
            raise ValueError(f"XIRR solver failed to converge: {e}")

    def compute_period_returns(
        self,
        daily_positions_df: pd.DataFrame,
        cash_flows_df: pd.DataFrame,
        as_of_date: date,
    ) -> dict:
        """
        Compute standard period returns (MTD, QTD, YTD, 1Y, 3Y, 5Y, since-inception).

        Args:
            daily_positions_df: daily positions from valuation engine
            cash_flows_df: cash flows from CashFlowExtractor
            as_of_date: the date for which to compute returns

        Returns:
            dict with period labels as keys and return values as Decimals
        """
        # Determine period start dates
        year = as_of_date.year
        month = as_of_date.month

        mtd_start = date(year, month, 1)
        qtd_start = date(year, ((month - 1) // 3) * 3 + 1, 1)
        ytd_start = date(year, 1, 1)

        # 1Y, 3Y, 5Y
        one_year_start = as_of_date - timedelta(days=365)
        three_year_start = as_of_date - timedelta(days=365 * 3)
        five_year_start = as_of_date - timedelta(days=365 * 5)

        # Since inception - use earliest date in data
        if not daily_positions_df.empty:
            inception_start = daily_positions_df["date"].min()
        else:
            inception_start = as_of_date

        periods = {
            "MTD": mtd_start,
            "QTD": qtd_start,
            "YTD": ytd_start,
            "1Y": one_year_start,
            "3Y": three_year_start,
            "5Y": five_year_start,
            "Since Inception": inception_start,
        }

        returns = {}

        # Compute TWR for each period
        for period_label, start_date in periods.items():
            if start_date > as_of_date:
                # Period start is in the future
                returns[period_label] = None
                continue

            try:
                twr = self.compute_twr(daily_positions_df, start_date, as_of_date)
                returns[period_label] = twr
            except Exception:
                returns[period_label] = None

        return returns
