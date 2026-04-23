"""ONE-SHOT sacred: two bin-aware Optuna winners from run_diverse_bin_optuna.py.

V1 = strict CV winner (trial 13): alpha=0.559, ml=1.184, mh=1.296, w_ma=0.348
V2 = CV-VAL gated (trial 14):     alpha=0.581, ml=1.173, mh=1.192, w_ma=0.338

Current champion: alpha=0.600, mult=1.138, w_ma=0.329 -> sacred 3,255.20
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


VARIANTS = [
    {"name": "bin_strict_cv",  "alpha": 0.559, "mult_low": 1.184, "mult_high": 1.296, "w_ma": 0.348},
    {"name": "bin_cv_val_bal", "alpha": 0.581, "mult_low": 1.173, "mult_high": 1.192, "w_ma": 0.338},
]


def run(variant: dict) -> float:
    sim = InventorySimulator()
    policy = DiverseCostAwarePolicy(
        alpha=variant["alpha"],
        mult_low=variant["mult_low"], mult_high=variant["mult_high"],
        high_demand_quantile=0.75, w_ma=variant["w_ma"],
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        censoring_strategy="mean_impute", random_state=42, master=sim.master,
        n_variants=5,
    )
    name = f"diverse5_{variant['name']}_SACRED"
    print(f"\n=== {name} ===")
    print(f"  params: a={variant['alpha']:.3f} ml={variant['mult_low']:.3f} "
          f"mh={variant['mult_high']:.3f} w_ma={variant['w_ma']:.3f}")

    t0 = time.time()
    results = sim.run_simulation(policy)
    wall = time.time() - t0

    print(f"--- SACRED result ---")
    print(f"  competition_cost  =  {results['competition_cost']:,.2f} EUR")
    print(f"  holding           =  {results['competition_holding']:,.2f}")
    print(f"  shortage          =  {results['competition_shortage']:,.2f}")
    print(f"  total_cost        =  {results['total_cost']:,.2f}")
    print(f"  wall-time         =  {wall:.1f}s")

    delta = results["competition_cost"] - 3255.20
    tag = "NEW CHAMPION!" if delta < 0 else "worse"
    print(f"  vs champion 3,255.20: {delta:+,.2f}  ({tag})")

    fake = {"per_fold": None,
            "mean": {"competition_cost": results["competition_cost"],
                      "comp_holding": results["competition_holding"],
                      "comp_shortage": results["competition_shortage"]},
            "val": None}
    log_experiment(name, fake, extra={
        "alpha": variant["alpha"], "mult_low": variant["mult_low"],
        "mult_high": variant["mult_high"], "w_ma": variant["w_ma"],
        "approach": "diverse_5model_eqwt_bin_optuna",
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
    print(f"  prior champion {'':10s} {3255.20:>10,.2f}")


if __name__ == "__main__":
    main()
