"""Seed-variance audit: run champion + both bin-aware variants across 3 seeds each.

Measures sacred cost noise floor from CatBoost/LGB random-seed variation.
Output: mean/std/min/max per variant across 3 seeds, sorted by mean.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402

from policies import DiverseCostAwarePolicy  # noqa: E402
from simulation import InventorySimulator  # noqa: E402


VARIANTS = [
    {"name": "champion_optuna",
     "kwargs": {"alpha": 0.599816047538945, "multiplier": 1.1376785766024788,
                 "w_ma": 0.3293972738151323}},
    {"name": "bin_strict_cv",
     "kwargs": {"alpha": 0.559, "mult_low": 1.184, "mult_high": 1.296,
                 "high_demand_quantile": 0.75, "w_ma": 0.348, "multiplier": 1.0}},
    {"name": "bin_cv_val_bal",
     "kwargs": {"alpha": 0.581, "mult_low": 1.173, "mult_high": 1.192,
                 "high_demand_quantile": 0.75, "w_ma": 0.338, "multiplier": 1.0}},
]
SEEDS = [42, 43, 44]


def run(name: str, kwargs: dict, seed: int) -> float:
    sim = InventorySimulator()
    policy = DiverseCostAwarePolicy(
        backtest_window=26, safety_floor=0.5, rmse_horizons=1,
        censoring_strategy="mean_impute", random_state=seed, master=sim.master,
        n_variants=5, **kwargs,
    )
    t0 = time.time()
    results = sim.run_simulation(policy)
    wall = time.time() - t0
    print(f"  {name:20s} seed={seed}  sacred={results['competition_cost']:>9,.2f}  "
          f"h={results['competition_holding']:>7,.0f}  s={results['competition_shortage']:>7,.0f}  "
          f"({wall:.0f}s)", flush=True)
    return float(results["competition_cost"])


def main():
    results = {v["name"]: [] for v in VARIANTS}
    for v in VARIANTS:
        print(f"\n>>> {v['name']}")
        for s in SEEDS:
            c = run(v["name"], v["kwargs"], s)
            results[v["name"]].append((s, c))

    print("\n=== SEED NOISE AUDIT ===")
    rows = []
    for v in VARIANTS:
        costs = [c for _, c in results[v["name"]]]
        m = mean(costs)
        sd = stdev(costs) if len(costs) > 1 else 0.0
        rows.append({"name": v["name"], "mean": m, "std": sd,
                      "min": min(costs), "max": max(costs), "range": max(costs)-min(costs)})

    print(f"{'variant':<20s} {'mean':>10s} {'std':>7s} {'min':>10s} {'max':>10s} {'range':>7s}")
    for r in sorted(rows, key=lambda x: x["mean"]):
        print(f"{r['name']:<20s} {r['mean']:>10,.2f} {r['std']:>7,.1f} "
              f"{r['min']:>10,.2f} {r['max']:>10,.2f} {r['range']:>7,.1f}")

    # Compute "noise floor" = max std across variants
    noise = max(r["std"] for r in rows)
    print(f"\nNoise floor (max std across variants): ~{noise:.0f} EUR")

    # Check separation
    champ = next(r for r in rows if r["name"] == "champion_optuna")
    for r in rows:
        if r["name"] == "champion_optuna": continue
        gap = champ["mean"] - r["mean"]
        cohen_d = gap / max(noise, 1.0)
        flag = "REAL" if abs(cohen_d) > 2 else "within noise"
        print(f"  {r['name']:<20s} mean vs champion: {gap:+,.2f} EUR  (cohen d={cohen_d:+.2f}) -> {flag}")


if __name__ == "__main__":
    main()
