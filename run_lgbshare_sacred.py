"""ONE-SHOT sacred: lgb_share=0.90 (strict CV winner from lgb_share sweep).

Reuses DiverseLGBShareCostAwarePolicy from run_lgb_share_sweep.py.
CV 4,216 (−16 vs current champion 4,232), VAL 2,627 (+15).
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
from run_lgb_share_sweep import DiverseLGBShareCostAwarePolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


def main():
    sim = InventorySimulator()
    policy = DiverseLGBShareCostAwarePolicy(
        lgb_share=0.90,
        alpha=0.5813428492464154,
        mult_low=1.1731901672544072,
        mult_high=1.1918987771673257,
        high_demand_quantile=0.75,
        w_ma=0.33800656001887974,
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        censoring_strategy="mean_impute", random_state=42, master=sim.master,
        n_variants=5,
    )
    name = "diverse5_lgbshare_0.90_SACRED"
    print(f"=== {name} ===")
    print(f"CV 4,216  VAL 2,627  (current champion CV 4,232 VAL 2,611 sacred 3,251.60)")

    t0 = time.time()
    results = sim.run_simulation(policy)
    wall = time.time() - t0

    print(f"\n--- SACRED result ---")
    print(f"  competition_cost  =  {results['competition_cost']:,.2f} EUR")
    print(f"  holding           =  {results['competition_holding']:,.2f}")
    print(f"  shortage          =  {results['competition_shortage']:,.2f}")
    print(f"  total_cost        =  {results['total_cost']:,.2f}")
    print(f"  wall-time         =  {wall:.1f}s")

    prev = 3251.60
    delta = results["competition_cost"] - prev
    tag = "NEW CHAMPION!" if delta < 0 else "worse / tie"
    print(f"  vs champion 3,251.60: {delta:+,.2f} EUR  ({tag})")

    fake = {"per_fold": None,
            "mean": {"competition_cost": results["competition_cost"],
                      "comp_holding": results["competition_holding"],
                      "comp_shortage": results["competition_shortage"]},
            "val": None}
    log_experiment(name, fake, extra={
        "lgb_share": 0.90,
        "alpha": 0.5813428492464154,
        "mult_low": 1.1731901672544072,
        "mult_high": 1.1918987771673257,
        "w_ma": 0.33800656001887974,
        "approach": "diverse_5model_lgb_share_0.90",
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
        "delta_vs_prior_champion": round(delta, 2),
    })


if __name__ == "__main__":
    main()
