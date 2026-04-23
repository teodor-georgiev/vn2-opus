"""Test OrderEnsemble (averaging multiple policies' orders) on 8-fold CV+VAL.

Three variants:
  A: 0.5 CostAware(α=0.65,untuned) + 0.5 CostAware(α=0.70,tuned,m=1.05)
  B: 0.33 each of the 3 policies (untuned, tuned, legacy ensemble cov=2 ×1.05)
  C: 0.5 CostAware(α=0.55) + 0.5 CostAware(α=0.75)  — over+under stocker average
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import (  # noqa: E402
    FOLDS_8, build_extended_sales, format_summary, log_experiment,
    score_policy_on_folds,
)
from policies import CostAwarePolicy, EnsemblePolicy, OrderEnsemble  # noqa: E402


def load_tuned_cfg() -> dict:
    with open("best_winner.json") as f:
        bw = json.load(f)
    cfg = bw["config"]
    return {
        "coverage_weeks": max(2, int(cfg.get("coverage_weeks", 2))),
        "w_ma": float(cfg.get("w_ma", 0.25)),
        "w_lgb_share": cfg.get("w_lgb_share", None),
        "censoring_strategy": cfg.get("censoring_strategy", "mean_impute"),
        "per_series_scaling": bool(cfg.get("per_series_scaling", False)),
        "demand_cluster_k": cfg.get("demand_cluster_k", None),
        "extended_features": bool(cfg.get("extended_features", False)),
        "direct_forecast": bool(cfg.get("direct_forecast", False)),
        "recency_decay": cfg.get("recency_decay", None),
        "categorical_features": bool(cfg.get("categorical_features", False)),
        "intermittency_features": bool(cfg.get("intermittency_features", False)),
        "w_stats": float(cfg.get("w_stats", 0.0)),
        "random_state": 42,
    }


tuned_base_cfg = load_tuned_cfg()
default_base_cfg = {
    "coverage_weeks": 3, "w_ma": 0.25,
    "censoring_strategy": "mean_impute", "random_state": 42,
}


def cap(alpha, mult, base_cfg):
    return CostAwarePolicy(
        alpha=alpha, multiplier=mult,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        ensemble_cfg=dict(base_cfg),
    )


def variant_A():
    return OrderEnsemble(
        [cap(0.65, 1.0, default_base_cfg), cap(0.70, 1.05, tuned_base_cfg)],
        weights=[0.5, 0.5],
    )


def variant_B():
    legacy_kwargs = {
        "coverage_weeks": 2, "w_ma": 0.2875, "w_lgb_share": 0.6971,
        "multiplier": 1.0731, "safety_units": 0.5962,
        "censoring_strategy": "mean_impute",
        "per_series_scaling": True, "demand_cluster_k": 8,
        "random_state": 42,
    }
    return OrderEnsemble(
        [
            cap(0.65, 1.0, default_base_cfg),
            cap(0.70, 1.05, tuned_base_cfg),
            EnsemblePolicy(**legacy_kwargs),
        ],
        weights=[0.34, 0.33, 0.33],
    )


def variant_C():
    return OrderEnsemble(
        [cap(0.55, 1.0, default_base_cfg), cap(0.75, 1.0, default_base_cfg)],
        weights=[0.5, 0.5],
    )


full_sales, full_in_stock, master = build_extended_sales()
results_summary = []

for name, factory in [("oe_A_avg2", variant_A),
                      ("oe_B_avg3", variant_B),
                      ("oe_C_lohi", variant_C)]:
    print(f"\n>>> {name}")
    t0 = time.time()
    results = score_policy_on_folds(
        policy_factory=factory,
        full_sales=full_sales, full_in_stock=full_in_stock, master=master,
        folds=FOLDS_8, include_val=True, coverage_weeks=3,
        alpha=0.65, n_workers=1,
    )
    wall = time.time() - t0
    print(format_summary(results, name))
    print(f"[{name}] wall-time: {wall:.1f}s")
    log_experiment(name, results, extra={"ensemble": "order_avg",
                                         "wall_seconds": round(wall, 1)})
    results_summary.append({
        "name": name,
        "cv": results["mean"]["competition_cost"],
        "val": (results.get("val") or {}).get("competition_cost"),
    })

print("\n=== SUMMARY ===")
print(f"{'name':>12} {'cv':>10} {'val':>10}")
for r in sorted(results_summary, key=lambda r: r["cv"]):
    print(f"{r['name']:>12} {r['cv']:>10,.2f} {r['val']:>10,.2f}")
