"""ONE-SHOT sacred: DiverseCostAwarePolicy (5 base models + MA blend, alpha=0.65).

Selection:
  8-fold CV: 4,347 (vs baseline 4,339, +8 worse — tied within noise)
  VAL: 2,570 (vs baseline 2,593, −23 better)
  Distinct policy variant; protocol-allowed one-shot.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from benchmark.cv_harness import log_experiment  # noqa: E402
from policies import DiverseCostAwarePolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    sim = InventorySimulator()
    policy = DiverseCostAwarePolicy(
        alpha=0.65, multiplier=1.0, w_ma=0.25,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        censoring_strategy="mean_impute", random_state=42, master=sim.master,
    )
    name = "diverse_5model_a0.65_m1.0_SACRED"
    print(f"=== {name} ===")
    print("CV 4,347  VAL 2,570  (baseline CostAware: CV 4,339 VAL 2,593 sacred 3,350)")
    print()

    t0 = time.time()
    results = sim.run_simulation(policy)
    wall = time.time() - t0

    print(f"\n--- SACRED result ---")
    print(f"competition_cost  =  {results['competition_cost']:,.2f} EUR")
    print(f"  holding         =  {results['competition_holding']:,.2f}")
    print(f"  shortage        =  {results['competition_shortage']:,.2f}")
    print(f"setup_cost        =  {results['setup_cost']:,.2f}")
    print(f"total_cost        =  {results['total_cost']:,.2f}")
    print(f"wall-time         =  {wall:.1f}s\n")
    print("Weekly breakdown:")
    print(results["weekly_log"].to_string(index=False))

    prev = 3349.60
    delta = results["competition_cost"] - prev
    better = delta < 0
    print(f"\nvs prior champion 3,349.60: {delta:+,.2f} EUR  ({'better' if better else 'worse'})")

    fake = {"per_fold": None,
            "mean": {"competition_cost": results["competition_cost"],
                      "comp_holding": results["competition_holding"],
                      "comp_shortage": results["competition_shortage"]},
            "val": None}
    log_experiment(name, fake, extra={
        "alpha": 0.65, "mult": 1.0, "w_ma": 0.25,
        "approach": "diverse_5model_eqwt",
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
        "delta_vs_prior_champion": round(delta, 2),
    })


if __name__ == "__main__":
    main()
