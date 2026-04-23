"""Experiment A: per-round cost matrix for CostAware (alpha=0.65, mult=1.0)
across the 8 CV folds + VAL. Are the same rounds consistently expensive?
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

from benchmark.cv_harness import FOLDS_8, VAL_START, build_extended_sales  # noqa: E402
from policies import CostAwarePolicy  # noqa: E402
from simulation import LEAD_TIME, TOTAL_WEEKS, run_window  # noqa: E402

ALPHA = 0.65
MULT = 1.0

windows = [(name, ws) for name, ws in FOLDS_8.items()]
windows.append(("val", VAL_START))

full_sales, full_in_stock, master = build_extended_sales()

# 8 simulation weeks; competition window = weeks 3..8 = sim indices 2..7.
n_weeks = TOTAL_WEEKS  # 8
comp_idx = list(range(LEAD_TIME, n_weeks))  # [2,3,4,5,6,7]
comp_labels = [f"w{i+1}" for i in comp_idx]  # ['w3'..'w8']

per_fold_costs = {}
for name, ws in windows:
    policy = CostAwarePolicy(
        alpha=ALPHA, multiplier=MULT, backtest_window=26, safety_floor=0.5,
        rmse_horizons=1,
        ensemble_cfg={
            "coverage_weeks": 3, "w_ma": 0.25,
            "censoring_strategy": "mean_impute", "random_state": 42,
        },
    )
    res = run_window(
        full_sales=full_sales, window_start=ws,
        full_in_stock=full_in_stock, master=master,
        n_weeks=n_weeks, policy_fn=policy,
    )
    log = res["weekly_log"]
    week_costs = log.set_index("week")["total_cost"]
    per_fold_costs[name] = week_costs

df = pd.DataFrame(per_fold_costs).T  # rows: fold, cols: week 0..7
df.columns = [f"w{i+1}" for i in df.columns]
print("=== Per-round total_cost (€) across CV folds + VAL ===")
print(df.round(0).to_string())

print("\n=== Competition window only (weeks 3..8) ===")
comp = df[comp_labels]
print(comp.round(0).to_string())
print(f"\nrow sums (= each fold's competition_cost): {comp.sum(axis=1).round(0).to_dict()}")
print(f"\ncol means (avg cost per round across folds):")
print(comp.mean(axis=0).round(1).to_string())
print(f"\ncol std (variability per round across folds):")
print(comp.std(axis=0).round(1).to_string())

# Per-round rank within each fold (1 = cheapest, 6 = most expensive)
ranks = comp.rank(axis=1, ascending=True)
print(f"\n=== Per-round rank within each fold (6 = most expensive) ===")
print(ranks.astype(int).to_string())
print(f"\nMean rank per round (high = consistently expensive):")
print(ranks.mean(axis=0).round(2).to_string())

# Coefficient of variation per round (low CV = structurally consistent)
cv_per_round = (comp.std(axis=0) / comp.mean(axis=0)).round(3)
print(f"\nCV (std/mean) per round (low = structural; high = fold-dependent):")
print(cv_per_round.to_string())
