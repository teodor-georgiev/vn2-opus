"""ONE-SHOT sacred: two feature-sweep winners.

V1 = only_categorical (CV winner: 4,191 vs champion 4,216)
V2 = all_three       (VAL winner: 2,533 vs champion 2,627)
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


CHAMP = dict(
    lgb_share=0.90,
    alpha=0.5813428492464154,
    mult_low=1.1731901672544072,
    mult_high=1.1918987771673257,
    high_demand_quantile=0.75,
    w_ma=0.33800656001887974,
    backtest_window=26, safety_floor=0.5, rmse_horizons=1,
    censoring_strategy="mean_impute", random_state=42,
    n_variants=5,
)

VARIANTS = [
    {"name": "only_categorical",  "hier": False, "cat": True,  "intm": False},
    {"name": "all_three",         "hier": True,  "cat": True,  "intm": True},
]


def run(v: dict) -> float:
    sim = InventorySimulator()
    p = DiverseLGBShareCostAwarePolicy(
        **CHAMP, master=sim.master,
        hierarchical_features=v["hier"],
        categorical_features=v["cat"],
        intermittency_features=v["intm"],
    )
    name = f"feat_{v['name']}_SACRED"
    print(f"\n=== {name} ===  hier={v['hier']} cat={v['cat']} intm={v['intm']}")
    t0 = time.time()
    results = sim.run_simulation(p)
    wall = time.time() - t0

    print(f"  competition_cost  = {results['competition_cost']:,.2f} EUR")
    print(f"  holding           = {results['competition_holding']:,.2f}")
    print(f"  shortage          = {results['competition_shortage']:,.2f}")
    print(f"  wall-time         = {wall:.1f}s")

    delta = results["competition_cost"] - 3227.80
    tag = "NEW CHAMPION!" if delta < 0 else "worse / tie"
    print(f"  vs champion 3,227.80: {delta:+,.2f}  ({tag})")

    fake = {"per_fold": None,
            "mean": {"competition_cost": results["competition_cost"],
                      "comp_holding": results["competition_holding"],
                      "comp_shortage": results["competition_shortage"]},
            "val": None}
    log_experiment(name, fake, extra={
        **{k: CHAMP[k] for k in CHAMP if isinstance(CHAMP[k], (int, float, str, bool))},
        "hier_features": v["hier"], "cat_features": v["cat"], "intm_features": v["intm"],
        "sacred_total_cost": results["total_cost"],
        "sacred_setup_cost": results["setup_cost"],
        "wall_seconds": round(wall, 1),
        "delta_vs_prior_champion": round(delta, 2),
    })
    return results["competition_cost"]


def main():
    costs = []
    for v in VARIANTS:
        c = run(v)
        costs.append((v["name"], c))
    print("\n=== SUMMARY ===")
    for name, c in sorted(costs, key=lambda x: x[1]):
        print(f"  {name:20s} {c:>10,.2f}")
    print(f"  prior champion {'':10s} {3227.80:>10,.2f}")


if __name__ == "__main__":
    main()
