"""Compute the theoretical minimum sacred competition_cost via an ORACLE policy
that has perfect knowledge of future demand and orders exactly Just-In-Time.

For each SKU independently, given:
  - end_inv_2 (after setup weeks 1-2, fixed by initial state)
  - perfect knowledge of demand[3..8]
  - decision at round r places an order arriving start of week r+2
  - LT=2, 6 rounds covering arrivals weeks 3..8

Optimal greedy:
  for week r in 3..8:
    if end_inv_{r-1} >= d_r:  order_{r-2} = 0   (use leftover)
    else:                     order_{r-2} = d_r - end_inv_{r-1}
    end_inv_r = max(0, end_inv_{r-1} + order - d_r)   # always 0 unless leftover

This minimizes holding (no over-stock) and shortage (no under-stock).
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

from simulation import (  # noqa: E402
    HOLDING_COST, LEAD_TIME, SHORTAGE_COST, TOTAL_WEEKS,
    InventorySimulator, load_all_actual_sales, load_initial_data,
)


def oracle_orders(end_inv_after_setup: pd.Series,
                  actual_demand_w3to8: pd.DataFrame) -> tuple[pd.Series, list[pd.Series]]:
    """For each SKU, compute optimal orders given perfect demand foresight.

    end_inv_after_setup : per-SKU inventory at end of week 2 (after the setup phase
        runs with whatever T1/T2 was already in transit)
    actual_demand_w3to8 : DataFrame [n_sku x 6] with demand for weeks 3..8

    Returns:
        end_inv_per_week_w3to8 : DataFrame [n_sku x 6] (after each week)
        orders : list of 6 pd.Series (one per round) of integer order quantities
    """
    n_sku = len(end_inv_after_setup)
    end_inv = end_inv_after_setup.copy().astype(float).values  # entering week 3
    end_inv_w = np.zeros((6, n_sku), dtype=float)              # ending each week 3..8
    orders_arr = np.zeros((6, n_sku), dtype=float)             # round 1..6 (1-indexed)

    demand = actual_demand_w3to8.values  # [n_sku, 6]

    # `end_inv` here = end_inv at end of previous week.
    # We treat round r (1-indexed) as placing the order arriving week r+2.
    # That order quantity is what's needed to meet week r+2 demand on top of
    # whatever leftover we'll have at end of week r+1.
    # But we have to decide it AT TIME of round r — with perfect foresight that's fine.
    #
    # Greedy week-by-week is equivalent and simpler:
    #   for each week w in 3..8: if leftover >= demand_w, no order needed; else order the gap.
    # BUT: the order arriving in week 3 was placed at round 1 (end of week 1).
    # Since we know demand perfectly, we can plan all 6 orders up-front.
    for w in range(6):  # w = 0..5 corresponds to weeks 3..8
        d_w = demand[:, w]
        # Need to meet d_w with: end_inv (end of previous week) + order arriving this week
        order_w = np.maximum(0.0, d_w - end_inv)
        # The order placed `LEAD_TIME` weeks ago must arrive now. In our framework,
        # round r-2 places the order for week r (1-indexed). Map week index → round:
        # week 3 -> round 1 (index 0), week 8 -> round 6 (index 5).
        round_idx = w  # because w=0 is week 3, w=5 is week 8
        orders_arr[round_idx, :] = order_w
        # Update end_inv for next iteration:
        end_inv = end_inv + order_w - d_w
        end_inv = np.maximum(end_inv, 0.0)
        end_inv_w[w, :] = end_inv

    end_inv_df = pd.DataFrame(end_inv_w.T,
                              index=end_inv_after_setup.index,
                              columns=[f"w{w+3}" for w in range(6)])
    orders = [pd.Series(orders_arr[r], index=end_inv_after_setup.index).round().astype(int)
              for r in range(6)]
    return end_inv_df, orders


def oracle_policy(sim, round_idx, sales_hist):
    """Wrap the oracle as a policy_fn that the simulator can run.

    On round 0 (first call), pre-compute all 6 orders using sim.actual_sales and
    the inventory state at that time. Return order for the current round.
    """
    if not hasattr(sim, "_oracle_orders"):
        # Project end_inv_2 by simulating weeks 1, 2 in our head (the simulator
        # has already done week 0, so we use sim.end_inventory directly).
        # Wait — by this round_idx=0, simulate_week(0) has run, so sim.end_inventory
        # is end-of-week-1 (1-indexed). But we want end_inv after setup = end of week 2
        # (1-indexed) = simulator week 1 (0-indexed).
        # So we need to look one week further: simulate_week(1) hasn't run yet at this point.
        # For simplicity we'll project it using the actual demand for week 1 (0-indexed) = week 2 (1-indexed).
        actual_cols = sorted(sim.actual_sales.columns)
        demand_w2 = sim.actual_sales[actual_cols[1]]  # 0-indexed week 1 = 1-indexed week 2
        # State at start of week 2 = sim.end_inventory (after week 1 simulated) + sim.in_transit_w1 (T1).
        start_w2 = sim.end_inventory + sim.in_transit_w1
        end_inv_w2 = (start_w2 - demand_w2.clip(upper=start_w2)).clip(lower=0)

        demand_w3to8 = sim.actual_sales[actual_cols[2:8]]  # weeks 3..8 1-indexed
        _, orders = oracle_orders(end_inv_w2, demand_w3to8)
        sim._oracle_orders = orders
    return sim._oracle_orders[round_idx]


def main():
    print("=== ORACLE bound ===")
    sim = InventorySimulator()
    results = sim.run_simulation(oracle_policy)

    print(f"Oracle competition_cost (lower bound): {results['competition_cost']:,.2f} EUR")
    print(f"  oracle holding   = {results['competition_holding']:,.2f}")
    print(f"  oracle shortage  = {results['competition_shortage']:,.2f}")
    print(f"setup_cost (fixed) = {results['setup_cost']:,.2f}")
    print(f"oracle total       = {results['total_cost']:,.2f}")
    print()
    print("Weekly breakdown:")
    print(results["weekly_log"].to_string(index=False))
    print()
    print("=== Comparison ===")
    print(f"ORACLE   : {results['competition_cost']:,.2f}")
    print(f"OUR best : 3,349.60")
    print(f"BENCHMARK: 4,334.00")
    if results['competition_cost'] > 0:
        print(f"\nGap-to-oracle: {3349.60 - results['competition_cost']:,.2f} EUR  "
              f"(we are {3349.60 / results['competition_cost']:.2f}x the floor)")
    else:
        print("\nOracle hits zero (perfect cost): we're {3349.60} EUR over the floor.")


if __name__ == "__main__":
    main()
