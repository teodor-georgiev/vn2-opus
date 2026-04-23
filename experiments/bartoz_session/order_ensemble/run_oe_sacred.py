"""ONE-SHOT sacred: OrderEnsemble variant A.

A = 0.5 * CostAware(α=0.65, mult=1.0, untuned base)
  + 0.5 * CostAware(α=0.70, mult=1.05, tuned base from best_winner.json)

CV = 4,276 (vs baseline 4,339, −63), VAL = 2,585 (vs baseline 2,593, −8).
Both CV and VAL improve over the prior sacred winner (3,349.60).
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

from benchmark.cv_harness import log_experiment  # noqa: E402
from policies import CostAwarePolicy, OrderEnsemble  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


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


def main():
    p1 = CostAwarePolicy(
        alpha=0.65, multiplier=1.0,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        ensemble_cfg={"coverage_weeks": 3, "w_ma": 0.25,
                      "censoring_strategy": "mean_impute", "random_state": 42},
    )
    p2 = CostAwarePolicy(
        alpha=0.70, multiplier=1.05,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        ensemble_cfg=load_tuned_cfg(),
    )
    policy = OrderEnsemble([p1, p2], weights=[0.5, 0.5])
    name = "oe_A_avg2_SACRED"
    print(f"=== {name} ===")
    print("Selection: 8-fold CV + VAL both improve over prior sacred 3,350.")
    print("CV 4,276 vs baseline 4,339 (-63), VAL 2,585 vs 2,593 (-8).")
    print("Reference: prior best sacred = 3,349.60.\n")

    t0 = time.time()
    sim = InventorySimulator()
    results = sim.run_simulation(policy)
    wall = time.time() - t0

    print(f"\n--- SACRED result ({name}) ---")
    print(f"competition_cost  =  {results['competition_cost']:,.2f} EUR")
    print(f"  holding         =  {results['competition_holding']:,.2f}")
    print(f"  shortage        =  {results['competition_shortage']:,.2f}")
    print(f"setup_cost (wk1-2)=  {results['setup_cost']:,.2f}")
    print(f"total_cost        =  {results['total_cost']:,.2f}")
    print(f"wall-time         =  {wall:.1f}s\n")
    print("Weekly breakdown:")
    print(results["weekly_log"].to_string(index=False))

    fake = {
        "per_fold": None,
        "mean": {
            "competition_cost": results["competition_cost"],
            "comp_holding": results["competition_holding"],
            "comp_shortage": results["competition_shortage"],
        },
        "val": None,
    }
    log_experiment(name, fake, extra={
        "ensemble": "order_avg",
        "components": "cap2_a0.65_m1.0_untuned + cap2_a0.70_m1.05_tuned",
        "weights": "0.5,0.5",
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
    })


if __name__ == "__main__":
    main()
