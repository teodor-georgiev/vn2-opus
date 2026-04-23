"""ONE-SHOT sacred: CostAware (tuned Ensemble base) at alpha=0.70, mult=1.05.

Selection rationale (under 8-fold CV rule):
  - alpha=0.70 wins CV plateau (CV 4,259 at mult=1.05)
  - mult=1.05 chosen as best CV-VAL trade-off (VAL 2,629 vs strict-CV winner
    mult=1.20 with VAL 2,745). Strict CV gain past 1.05 = 40 EUR but VAL cost = 116 EUR.
  - Tuned base = best_winner.json config (the 3,581 ensemble's forecaster).

Reference: prior best sacred = 3,350 (untuned base, alpha=0.65, mult=1.0).
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
from policies import CostAwarePolicy  # noqa: E402
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
    ALPHA = 0.70
    MULT = 1.05
    base_cfg = load_tuned_cfg()
    policy = CostAwarePolicy(
        alpha=ALPHA, multiplier=MULT,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        ensemble_cfg=base_cfg,
    )
    name = f"cap2_TUNED_a{ALPHA}_m{MULT}_SACRED"
    print(f"=== {name} ===")
    print(f"Selection: 8-fold CV winner alpha=0.70, mult=1.05 (CV 4,259, VAL 2,629).")
    print(f"Reference: prior best sacred = 3,350.\n")

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
        "alpha": ALPHA, "mult": MULT, "tuned_base": True,
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
    })


if __name__ == "__main__":
    main()
