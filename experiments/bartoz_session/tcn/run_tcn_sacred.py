"""ONE-SHOT sacred run for TCN cov=2 x 1.10 (CV-selected, VAL tiebreak).

Per CLAUDE.md selection rule: touch sacred ONCE per policy variant.
Selected on CV:
    CV mean = 3,674.45  (mult=1.10)  vs  3,671.40 (mult=1.15) — tie
    VAL tiebreak = 2,850.80 (mult=1.10) vs 2,911.80 (mult=1.15) — 1.10 wins
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import log_experiment  # noqa: E402
from policies import TCNPolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402
from tcn_forecaster import TCNConfig  # noqa: E402


def main():
    cfg = TCNConfig(patience=8, fine_tune_epochs=10)
    policy = TCNPolicy(
        coverage_weeks=2,
        multiplier=1.10,
        safety_units=0.0,
        cfg=cfg,
        verbose=False,
    )
    name = "tcn_cov2_m1.10_SACRED"
    print(f"=== {name} ===")
    print("Selection rationale: CV-winner with VAL tiebreak at mult=1.10.")
    print("Reference:  LGB+CB cov=2 x 1.05 scored 3,786 EUR on sacred.\n")

    t0 = time.time()
    sim = InventorySimulator()
    results = sim.run_simulation(policy)
    wall = time.time() - t0

    print(f"\n--- SACRED result ({name}) ---")
    print(f"competition_cost  =  {results['competition_cost']:,.2f} EUR")
    print(f"  holding         =  {results['competition_holding']:,.2f}")
    print(f"  shortage        =  {results['competition_shortage']:,.2f}")
    print(f"setup_cost (wk1-2)=  {results['setup_cost']:,.2f} (const across all policies)")
    print(f"total_cost        =  {results['total_cost']:,.2f}")
    print(f"wall-time         =  {wall:.1f}s\n")
    print("Weekly breakdown:")
    print(results["weekly_log"].to_string(index=False))

    # Log in the same format as CV runs (single-row, no per-fold breakdown).
    fake_results = {
        "per_fold": None,
        "mean": {
            "competition_cost": results["competition_cost"],
            "comp_holding": results["competition_holding"],
            "comp_shortage": results["competition_shortage"],
        },
        "val": None,
    }
    log_experiment(name, fake_results, extra={
        "cov": 2, "mult": 1.10, "safety": 0.0,
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
    })


if __name__ == "__main__":
    main()
