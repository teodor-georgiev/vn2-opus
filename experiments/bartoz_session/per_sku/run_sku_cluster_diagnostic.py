"""Diagnose: bin SKUs by demand pattern; see how cost distributes per bin.

Bins by mean weekly demand (quartiles, 4 bins). Simple, deterministic, fast.
Uses CostAware (alpha=0.65, mult=1.0) on cv0 to compute per-SKU cost contribution.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmark.cv_harness import FOLDS_8, build_extended_sales  # noqa: E402
from policies import CostAwarePolicy  # noqa: E402
from simulation import HOLDING_COST, LEAD_TIME, SHORTAGE_COST, TOTAL_WEEKS  # noqa: E402
import simulation  # noqa: E402

print("[diag] loading data...", flush=True)
full_sales, full_in_stock, master = build_extended_sales()
hist = full_sales.iloc[:, :157]

mean_d = hist.mean(axis=1)
std_d = hist.std(axis=1)
cv_d = (std_d / mean_d.clip(lower=0.1)).clip(upper=10)
frac_zero = (hist == 0).mean(axis=1)

# 4 bins by demand mean (quartiles).
mean_bin = pd.qcut(mean_d, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

# Also categorize by intermittency.
zero_bin = pd.cut(frac_zero, [-0.01, 0.3, 0.6, 1.01],
                  labels=["smooth", "intermittent", "lumpy"])

feat = pd.DataFrame({
    "mean": mean_d.round(2),
    "cv": cv_d.round(2),
    "frac_zero": frac_zero.round(2),
    "mean_bin": mean_bin,
    "zero_bin": zero_bin,
}, index=hist.index)

print("\n=== Bin sizes ===")
print(f"By mean quartile:\n{feat['mean_bin'].value_counts()}")
print(f"\nBy intermittency:\n{feat['zero_bin'].value_counts()}")

# Run CostAware on cv0 with per-SKU cost tracking.
print("\n[diag] running CostAware on cv0 with per-SKU cost tracking...", flush=True)
ws = FOLDS_8["cv0"]
sales_cols = sorted(full_sales.columns)
hist_cols = sales_cols[:ws]
demand_cols = sales_cols[ws:ws + TOTAL_WEEKS]
sales_hist = full_sales[hist_cols].copy()
in_stock_hist = full_in_stock[hist_cols].copy()
actual_sales = full_sales[demand_cols].copy()
idx = full_sales.index
init_state = pd.DataFrame({
    "End Inventory": pd.Series(0.0, index=idx),
    "In Transit W+1": pd.Series(0.0, index=idx),
    "In Transit W+2": pd.Series(0.0, index=idx),
})
sim = simulation.InventorySimulator(
    sales_hist=sales_hist, in_stock=in_stock_hist,
    initial_state=init_state, master=master, actual_sales=actual_sales,
)

policy = CostAwarePolicy(
    alpha=0.65, multiplier=1.0, backtest_window=26, safety_floor=0.5, rmse_horizons=1,
    ensemble_cfg={"coverage_weeks": 3, "w_ma": 0.25,
                  "censoring_strategy": "mean_impute", "random_state": 42},
)

per_sku_holding = pd.Series(0.0, index=idx)
per_sku_shortage = pd.Series(0.0, index=idx)

actual_cols = sorted(sim.actual_sales.columns)
sim.reset()
for round_idx in range(TOTAL_WEEKS):
    actual_demand = sim.actual_sales[actual_cols[round_idx]]
    start_inventory = sim.end_inventory + sim.in_transit_w1
    sales_done = np.minimum(start_inventory, actual_demand)
    missed = actual_demand - sales_done
    end_inv = start_inventory - sales_done
    if round_idx >= LEAD_TIME:
        per_sku_holding += end_inv * HOLDING_COST
        per_sku_shortage += missed * SHORTAGE_COST
    sim.end_inventory = end_inv
    sim.in_transit_w1 = sim.in_transit_w2.copy()
    sim.in_transit_w2 = pd.Series(0, index=sim.end_inventory.index)
    if round_idx < 6:
        sales_hist_now = sim.get_sales_history(round_idx + 1)
        order = policy(sim, round_idx, sales_hist_now)
        sim.place_order(order)
    print(f"  round {round_idx} done", flush=True)

per_sku_total = per_sku_holding + per_sku_shortage
df = pd.DataFrame({
    "holding": per_sku_holding,
    "shortage": per_sku_shortage,
    "total": per_sku_total,
}).join(feat[["mean", "cv", "frac_zero", "mean_bin", "zero_bin"]])

print(f"\n[diag] CV0 sanity: per-SKU sum = {per_sku_total.sum():.0f}  (expected ~6,115)")

print("\n=== Cost by demand-mean quartile ===")
g1 = df.groupby("mean_bin", observed=True).agg(
    n=("total", "size"),
    h_sum=("holding", "sum"),
    s_sum=("shortage", "sum"),
    total_sum=("total", "sum"),
    h_per_sku=("holding", "mean"),
    s_per_sku=("shortage", "mean"),
).round(1)
g1["pct_of_total"] = (100 * g1["total_sum"] / g1["total_sum"].sum()).round(1)
print(g1.to_string())

print("\n=== Cost by intermittency bin ===")
g2 = df.groupby("zero_bin", observed=True).agg(
    n=("total", "size"),
    h_sum=("holding", "sum"),
    s_sum=("shortage", "sum"),
    total_sum=("total", "sum"),
    h_per_sku=("holding", "mean"),
    s_per_sku=("shortage", "mean"),
).round(1)
g2["pct_of_total"] = (100 * g2["total_sum"] / g2["total_sum"].sum()).round(1)
print(g2.to_string())

# Pareto: how concentrated is the cost?
sorted_total = df["total"].sort_values(ascending=False)
print(f"\n=== Pareto distribution of cost ===")
for k in [10, 50, 100, 200]:
    top_k_share = sorted_total.head(k).sum() / sorted_total.sum() * 100
    print(f"Top {k:3d} SKUs ({100*k/len(sorted_total):.1f}% of products) account for {top_k_share:.1f}% of cost")

# Save the bin assignments for downstream policy use.
df[["mean_bin", "zero_bin"]].to_csv("sku_bins.csv")
print("\nSaved sku_bins.csv")
