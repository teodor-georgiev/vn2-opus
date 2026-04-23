"""ONE-SHOT sacred run for CostAwarePolicy alpha=0.65, mult=1.0.

Selection: CV winner on full 4-fold CV at 3,356 EUR (−250 vs our prior Ensemble
CV best 3,604). VAL tiebreak at 2,593 EUR. Selected BEFORE this sacred run.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import log_experiment  # noqa: E402
from policies import CostAwarePolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    ALPHA = 0.65
    MULT = 1.0
    policy = CostAwarePolicy(
        alpha=ALPHA,
        multiplier=MULT,
        backtest_window=26,
        safety_floor=0.5,
        rmse_horizons=1,
        ensemble_cfg={
            "coverage_weeks": 3,
            "w_ma": 0.25,
            "censoring_strategy": "mean_impute",
            "random_state": 42,
        },
    )
    name = f"cap2_a{ALPHA}_m{MULT}_SACRED"
    print(f"=== {name} ===")
    print(f"Selection rationale: CV winner at alpha={ALPHA} (3,356); VAL tiebreak (2,593).")
    print(f"Reference: our prior best Ensemble sacred = 3,581.\n")

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
        "alpha": ALPHA, "mult": MULT,
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
    })


if __name__ == "__main__":
    main()
