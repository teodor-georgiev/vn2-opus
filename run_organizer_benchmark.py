"""Run the ORGANIZER's exact benchmark code (Seasonal MA + 4wk coverage)
across the full 6-round sacred simulation. Compare to the 4,334 reference.

Translates the user's snippet into a policy function applied each round.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simulation import InventorySimulator  # noqa: E402


def organizer_policy(sim, round_idx, sales_hist):
    """The exact organizer benchmark logic, applied at each round.

    Per round:
      1. Mask shortages: sales[~in_stock] = NaN
      2. Compute multiplicative weekly seasonal factors (normalized to mean=1)
      3. Un-seasonalize past sales
      4. 13-week MA of un-seasonalized = base_forecast (per SKU)
      5. Forecast next 9 future weeks; re-seasonalize
      6. order_up_to = sum of first 4 forecast weeks
      7. order = max(0, order_up_to - (End_Inv + T1 + T2))

    Note: the original snippet uses initial `state` for inventory; here we use
    the CURRENT simulator state at this round, which is what a real participant
    would use round-by-round.
    """
    sales = sales_hist.copy()
    in_stock = sim.in_stock

    # 1. Mask out-of-stock weeks as NaN — verbatim from user code.
    # We have to be careful: in_stock columns may not match sales columns 1:1
    # (in_stock has historical weeks only; sales_hist may include simulated weeks).
    matching = sales.columns[sales.columns.isin(in_stock.columns)]
    aligned = in_stock.reindex(columns=matching)
    mask = ~aligned.astype(bool)
    sales.loc[:, matching] = sales[matching].mask(mask)

    # 2. Multiplicative weekly seasonality
    season = sales.mean().rename("Demand").to_frame()
    season["Week Number"] = season.index.isocalendar().week.values
    season = season.groupby("Week Number").mean()
    season = season / season.mean()

    # 3. Un-seasonalize
    sales_weeks = sales.columns.isocalendar().week
    season_factors = season.loc[sales_weeks.values, "Demand"].values.reshape(-1)
    sales_no_season = sales / season_factors

    # 4. 13-week MA of un-seasonalized sales
    base_forecast = sales_no_season.iloc[:, -13:].mean(axis=1)

    # 5. Project 9 future weekly periods (verbatim: periods=10, inclusive='neither' -> 9 weeks)
    f_periods = pd.date_range(start=sales.columns[-1], periods=10,
                               inclusive="neither", freq="W-MON")
    forecast = pd.DataFrame(
        data=base_forecast.values.reshape(-1, 1).repeat(len(f_periods), axis=1),
        columns=f_periods,
        index=sales.index,
    )
    f_season = season.loc[f_periods.isocalendar().week.values, "Demand"].values.reshape(-1)
    forecast = forecast * f_season

    # 6. order_up_to = sum of first 4 forecast weeks
    order_up_to = forecast.iloc[:, :4].sum(axis=1)

    # 7. net inventory = current sim state (NOT initial state)
    net_inv = sim.end_inventory + sim.in_transit_w1 + sim.in_transit_w2

    order = (order_up_to - net_inv).clip(lower=0).round(0).astype(int)
    return order


def main():
    print("=== ORGANIZER benchmark on SACRED ===\n")
    sim = InventorySimulator()
    res = sim.run_simulation(organizer_policy)

    print(f"competition_cost   = {res['competition_cost']:>10,.2f} EUR")
    print(f"  holding          = {res['competition_holding']:>10,.2f}")
    print(f"  shortage         = {res['competition_shortage']:>10,.2f}")
    print(f"setup_cost (wk1-2) = {res['setup_cost']:>10,.2f}")
    print(f"total_cost (8 wk)  = {res['total_cost']:>10,.2f}")
    print()
    print("Reference: official organizer baseline = 4,334.00 EUR")
    print(f"Match within 1 EUR? {abs(res['competition_cost'] - 4334) < 1.0}")
    print(f"Match within 50 EUR? {abs(res['competition_cost'] - 4334) < 50.0}")
    print()
    print("Comparison vs other policies on sacred:")
    print(f"  Organizer (this run)      : {res['competition_cost']:,.2f}")
    print(f"  Our champion (CostAware)  : 3,349.60")
    print(f"  Bartosz #1 leaderboard    : 3,763.00")
    print()
    print("Weekly breakdown:")
    print(res["weekly_log"].to_string(index=False))


if __name__ == "__main__":
    main()
